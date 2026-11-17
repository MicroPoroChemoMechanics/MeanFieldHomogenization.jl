# =============================================================================
#  mem_spheroids.jl — the multipole expansion method for spheroids in
#  conduction (kushch2013, chapter 6; kushch2015).
#
#  Every particle is a spheroid of revolution axis e₃, possibly layered, with
#  its own confocal coordinates (ξ, η, φ) of focal distance c (real for a
#  prolate spheroid, −i·focal for an oblate one). About it the temperature in
#  the matrix is
#
#       T = Σ_{n,m} a_n^m f_n^m + Σ_{n,m} b_n^m F_n^m ,      b^m = 𝐙^m a^m ,
#
#  with f, F the spheroidal harmonics of `LayeredSpheroids/harmonic_response.jl`
#  and 𝐙^m its response at the order m. The field F_t^s that sphere p sends
#  is carried to the regular harmonics f_k^l of particle q by
#
#       F_t^s(x − X_p) = Σ_{k,l} η_tksl(X_q − X_p) f_k^l(x − X_q) ,
#
#  and the coefficients η are obtained in one of two ways.
#
#  FAR. From F_t^s = (−1)^s Σ_{k′} K⁽³⁾_{tk′}(c_p) Y_{k′}^s (kushch2013,
#  Eq. (6.24), valid for r > |c_p|), the translation of the spherical
#  harmonics, and y_n^l = Σ_k K⁽²⁾_{nk}(c_q) f_k^l (Eq. (6.22), exact):
#
#    η_tksl = (−1)^{s+k+l} π (k+½) (c_p/2)^{t+1} (c_q/2)^k
#             Σ_r Y_{t+k+2r}^{s−l}(R) Σ_{j=0}^r (c_p/2)^{2(r−j)} (c_q/2)^{2j}
#                 / [(r−j)! Γ(t+r−j+3/2) j! Γ(k+j+3/2)] .
#
#  The series in r converges as ((|c_p| + |c_q|)/|R|)^{2r}, and diverges when
#  |R| < |c_p| + |c_q| (kushch2013, section 6.3.3), even for spheroids that do
#  not touch: two oblate spheroids stacked along their axis are a case.
#
#  NEAR. The coefficient of f_k^l is read on the surface ξ = ξ₀ of particle q,
#  where f_k^l = P̄_k^l(ξ₀) χ_k^l and the χ are orthogonal:
#
#    η_tksl = (2k+1)/(4π P_k^{|l|}(ξ₀)) ∫∫ F_t^s(x − X_p) conj(χ_k^l) dη dφ
#
#  (kushch2013, Eq. (6.58)), integrated by Gauss–Legendre in η and the
#  trapezoidal rule in φ. It holds whenever the spheroids do not overlap.
#
#  The far form is used beyond |R| = 5(|c_p| + |c_q|)/2, the near one below.
#  In a periodic assembly the images of particle p (its own included for
#  q = p) within that distance are integrated one by one, and the others enter
#  the far form through their lattice sums, those images being left out of
#  the sums (`_lattice_sums(…; exclude)`). The background adds the uniform
#  gradient of conduction, −4π d/(3V) per spherical dipole d.
#
#  EFFECTIVE CONDUCTIVITY. F_1^m = (−1)^m (c²/3) Y_1^m + …, so the spherical
#  dipoles of a spheroid are (−1)^m c² b_1^m/3, and the effective
#  conductivity follows as for spheres.
# =============================================================================

# The near form below |R| = κ (|c_p| + |c_q|): the far series then converges at
# least as κ⁻²ʳ, and `_MEM_FAR_TERMS` terms leave it below the rounding.
const _MEM_NEAR = 2.5
const _MEM_FAR_TERMS = 20

# ─── The spheroids of the assembly ───────────────────────────────────────────

# A particle as the layered spheroid of axis e₃ whose responses are asked for.
function _mem_spheroid(geom::LayeredSpheroids.LayeredSpheroid, asm, name::Symbol, prop::Symbol)
    _mem_axis_e3(collect(geom.axis), name)
    return geom
end

function _mem_spheroid(geom::Elasticity.Ellipsoid{3}, asm, name::Symbol, prop::Symbol)
    A = Array(TensND.canonical_components(MFH_Core.shape_tensor(geom)))
    # a spheroid of axis e₃: diagonal shape tensor, equal transverse semi-axes
    off = TensND.primal.((A[1, 2], A[1, 3], A[2, 3]))
    (all(iszero, off) && isapprox(TensND.primal(A[1, 1]), TensND.primal(A[2, 2]); rtol = 1.0e-12)) ||
        _mem_axis_e3(nothing, name)
    return _mem_spheroid_carrying(A[3, 3], A[1, 1], particle_property(asm, name, prop))
end

# An isotropic spheroid is its one-layer layered spheroid; an anisotropic one,
# that of unit conductivity carrying its conductivity (`mem_anisotropic.jl`).
_mem_spheroid_carrying(a₃, a₁, P::TensND.TensISO) = LayeredSpheroids.LayeredSpheroid((a₃,), (a₁,), (P,))
_mem_spheroid_carrying(a₃, a₁, P) = throw(
    ArgumentError(
        "MultipoleExpansion: a spheroid of anisotropic stiffness is not taken; its moduli are " *
            "isotropic (TensISO)"
    )
)
_mem_spheroid_carrying(a₃, a₁, P::TensND.AbstractTens{2, 3}) = _MemAnisotropicSpheroid(
    LayeredSpheroids.LayeredSpheroid((a₃,), (a₁,), (TensND.TensISO{3}(one(a₁)),)),
    Array(TensND.canonical_components(P))
)

function _mem_axis_e3(axis, name::Symbol)
    axis !== nothing && iszero(TensND.primal(axis[1])) && iszero(TensND.primal(axis[2])) && return nothing
    throw(
        ArgumentError(
            "MultipoleExpansion: particle :$(name) is not a spheroid of revolution axis e₃; " *
                "the spheroids of an assembly are aligned with e₃"
        )
    )
end

"""
    ShapedMultipoleSolution

Solution of the multipole expansion of an assembly of spheroids or ellipsoids
in conduction: for each particle, in the order of `names`, the scales of its
unknowns (the moduli of its regular harmonics on its surface), its response
``\\mathbf{Z}`` over its harmonics, the matrix that turns the coefficients it
sends into its dipole (`dipoles`, ``3 \\times`` the number of harmonics), and
the scaled coefficients it receives under the three unit remote gradients
(`received`, one column per gradient), these in the frame whose axes are the
axes `axes` of the assembly.
"""
struct ShapedMultipoleSolution{T, C}
    names::Vector{Symbol}
    scales::Vector{Vector{T}}
    responses::Vector{Matrix{C}}
    dipoles::Vector{Matrix{C}}
    received::Matrix{C}
    nmax::Int
    axes::NTuple{3, Int}
end

_mem_spheres_only(sol::ShapedMultipoleSolution, what::String) = throw(
    ArgumentError("MultipoleExpansion: $(what) are available for spheres; the spheroids give the effective property")
)

# What the expansion needs of a particle of any shape, beyond its geometry:
#   scales    the moduli of its regular harmonics on its surface, scaling the unknowns;
#   Z         its response over its harmonics, b = Z a;
#   Π         the projection of the values of a regular field on its surface
#             (`surface`, 3 × points) on its regular harmonics;
#   dipole    the dipole d = dipole · b of the field it sends;
#   gradient  the coefficients a = gradient · g of the uniform gradient g·x.
# The two matrices carry everything the remote field, the background of a
# lattice and the effective conductivity need, whatever the harmonics.

# How a particle reads the field of a close neighbor on its surface: by its
# values, or, for a crack (`mem_cracks.jl`), by its normal derivative on one lip.
struct _MemValues end
struct _MemNormalDerivative end

# Everything the expansion needs of one spheroid.
struct _MemSpheroid{C, R, D <: Union{_MemValues, _MemNormalDerivative}}
    c::C                      # focal distance, complex
    ξ₀::C                     # confocal coordinate of the outer surface
    scales::Vector{R}
    Z::Matrix{C}
    Π::Matrix{C}
    surface::Matrix{R}
    dipole::Matrix{C}
    gradient::Matrix{C}
    reading::D
end
_MemSpheroid{C, R}(c, ξ₀, scales, Z, Π, surface, dipole, gradient) where {C, R} =
    _MemSpheroid{C, R, _MemValues}(c, ξ₀, scales, Z, Π, surface, dipole, gradient, _MemValues())

# The radius of the sphere that holds the singularities of the decaying harmonics.
_mem_focal(d::_MemSpheroid) = abs(d.c)
_mem_far_degree(::_MemSpheroid, nmax::Int) = 2 * (nmax + _MEM_FAR_TERMS)

# The same in another element type, to bring the particles of an assembly to one.
_mem_convert(d::_MemSpheroid{<:Any, <:Any, D}, ::Type{C}) where {C, D} = _MemSpheroid{C, real(C), D}(
    d.c, d.ξ₀, d.scales, d.Z, d.Π, d.surface, d.dipole, d.gradient, d.reading
)

# The spheroidal dipole map: Ŷ₁ᵐ = (−1)^m (c²/3) Y₁ᵐ + …, and the spherical
# dipoles b₁ᵐ give d = (b₁¹ − b₁⁻¹, i (b₁¹ + b₁⁻¹), b₁⁰) (spheres page). The
# gradient map: g·x = g₃ y₁⁰ + (g₁ − i g₂) y₁¹ − (g₁ + i g₂) y₁⁻¹ with y₁ᵐ = c ŷ₁ᵐ.
function _mem_spheroid_maps(c::C, nb::Int) where {C}
    D = zeros(C, 3, nb)
    s(m) = (isodd(m) ? -1 : 1) * c^2 / 3
    D[1, _mem_index(1, 1)], D[1, _mem_index(1, -1)] = s(1), -s(-1)
    D[2, _mem_index(1, 1)], D[2, _mem_index(1, -1)] = im * s(1), im * s(-1)
    D[3, _mem_index(1, 0)] = s(0)
    G = zeros(C, nb, 3)
    G[_mem_index(1, 1), :] .= (c, -im * c, 0)
    G[_mem_index(1, -1), :] .= (-c, -im * c, 0)
    G[_mem_index(1, 0), 3] = c
    return D, G
end

function _mem_spheroid_data(s::LayeredSpheroids.LayeredSpheroid, K₀, nmax::Int)
    c = complex(s.c)
    ξ₀ = complex(s.q[end])
    nb = _mem_block(nmax)
    Zm = [harmonic_response(s, K₀, m, nmax) for m in 0:nmax]
    C = promote_type(typeof(c), eltype(eltype(Zm)))
    R = real(C)
    Z = zeros(C, nb, nb)
    scales = zeros(R, nb)
    for m in 0:nmax
        P = LayeredSpheroids.legendre_order_q(ξ₀, m, nmax)[1]
        n₀ = max(m, 1)
        for n in n₀:nmax, μ in (m == 0 ? (0,) : (m, -m))
            scales[_mem_index(n, μ)] = abs(LayeredSpheroids._factorial_ratio(R, n, m) * P[n + 1])
            for k in n₀:nmax
                Z[_mem_index(n, μ), _mem_index(k, μ)] = Zm[m + 1][n - n₀ + 1, k - n₀ + 1]
            end
        end
    end
    Π, surface, _ = _mem_surface_projection(c, ξ₀, nmax)
    return _MemSpheroid{C, R}(c, ξ₀, scales, Z, Π, surface, _mem_spheroid_maps(C(c), nb)...)
end

# Quadrature of the outer surface ξ = ξ₀: the points, their weights in (η, φ),
# and the matrix Π that turns the values there of a regular field into its
# coefficients a_k^l,
#     a_k^l = (2k+1)/(4π P_k^{|l|}(ξ₀)) Σ w conj(χ_k^l) u .
function _mem_surface_projection(c::C, ξ₀::C, nmax::Int) where {C}
    R = real(C)
    nη = 2 * nmax + 24
    nφ = 2 * nη
    xg, wg = LayeredSpheroids.QuadGK.gauss(nη)
    nb = _mem_block(nmax)
    Π = zeros(C, nb, nη * nφ)
    surface = zeros(R, 3, nη * nφ)
    weights = zeros(R, nη * nφ)
    ρ₀, z₀ = real(c * sqrt(ξ₀^2 - 1)), real(c * ξ₀)      # transverse and axial semi-axes
    Pξ = [LayeredSpheroids.legendre_order_q(ξ₀, m, nmax)[1] for m in 0:nmax]
    for (j, (η, w)) in enumerate(zip(xg, wg))
        Pη = [LayeredSpheroids.legendre_order_p(η, m, nmax)[1] for m in 0:nmax]
        η̄ = sqrt(1 - η^2)
        for i in 1:nφ
            φ = 2 * R(π) * (i - 1) / nφ
            col = (j - 1) * nφ + i
            surface[:, col] .= (ρ₀ * η̄ * cos(φ), ρ₀ * η̄ * sin(φ), z₀ * η)
            weights[col] = w * (2 * R(π) / nφ)
            for k in 1:nmax, l in (-k):k
                μ = abs(l)
                χ = _mem_surface_harmonic(Pη[μ + 1][k + 1], l, φ)
                Π[_mem_index(k, l), col] = (2k + 1) / (4 * R(π) * Pξ[μ + 1][k + 1]) * w * (2 * R(π) / nφ) * conj(χ)
            end
        end
    end
    return Π, surface, weights
end

# χ_k^l from P_k^{|l|} (Ferrers): (−1)^l P e^{ilφ} for l ≥ 0, P e^{ilφ} for l < 0.
_mem_surface_harmonic(P, l::Int, φ) = (l ≥ 0 && isodd(l) ? -P : P) * cis(l * φ)

# Decaying spheroidal harmonics Ŷ_t^s (1 ≤ t ≤ nmax) of the spheroid at x,
# from the table of their renormalized counterparts, F_t^s = (−1)^s κ_t F̃_t^s.
function _mem_irregular!(F::AbstractVector, x, sph::_MemSpheroid, nmax::Int)
    F̃ = _mem_spheroidal_table(x, sph.c, nmax)
    for t in 1:nmax, s in (-t):t
        F[_mem_index(t, s)] = (isodd(s) ? -1 : 1) * _mem_spheroidal_scale(sph.c, t) * F̃[t, s]
    end
    return F
end

# The decaying spheroidal harmonics divided by (−1)^s κ_t, κ_t = c^{t+1}/(2t+1)!!,
# F̃_t^s = (−1)^s F_t^s / κ_t, as the table of the spherical Y_t^s they tend to
# far from the spheroid, for 0 ≤ t ≤ deg: the vector harmonics and the
# solutions of the Lamé equation built on the Y_t^s then carry over to the
# spheroid by replacing Y_t^s with F̃_t^s, which keeps the symmetry
# F̃_t^{−s} = (−1)^s conj(F̃_t^s) of the table. With the exterior harmonics of
# the spheroidal chart of TensND of focal distance f = |c|, whose radial factor
# 𝖰_t^s is real,
#
#     F̃_t^s = (2t+1)!! (t−s)!/(t+s)! P_t^s(p) 𝖰_t^s(ζ) e^{isφ} / f^{t+1} :
#
# on the oblate side the phases of Q_t^s(iτ) = i^{−(t+1)} 𝒬_t^s(τ) and of
# c^{t+1} = (−i f)^{t+1} cancel.
_mem_spheroidal_table(x, c, deg::Int) = _mem_spheroidal_table_on(_mem_chart(c), x, deg)
_mem_spheroidal_table_on(chart, x, deg::Int) = _mem_spheroidal_table(chart, TensND.cartesian_to_coords(chart, x), deg)
# the same from the coordinates (φ, p, ζ) of the point on the chart
function _mem_spheroidal_table(chart::TensND.AbstractChart{3}, ξ, deg::Int)
    f = chart.c
    R = float(promote_type(typeof(f), map(typeof, Tuple(ξ))...))
    G = [TensND.exterior_harmonics(chart, deg, s, ξ) for s in 0:deg]
    return TensND.SolidHarmonicTable(
        (t, s) -> LayeredSpheroids._double_factorial(R, 2t + 1) * LayeredSpheroids._factorial_ratio(R, t, s) *
            G[s + 1][t + 1] * cis(s * ξ[1]) / f^(t + 1), deg
    )
end

# The spheroidal chart of TensND of a focal distance c: real, a prolate one;
# −i·f, an oblate one.
_mem_chart(c) = iszero(imag(c)) ? TensND.ProlateSpheroidalChart(real(c)) : TensND.OblateSpheroidalChart(abs(imag(c)))

_mem_spheroidal_scale(c, t::Int) = c^(t + 1) / LayeredSpheroids._double_factorial(real(typeof(c)), 2t + 1)

# Coefficients η of the regular expansion about particle q of the decaying
# harmonics of particle p, centered at −R from q: by projection on the surface.
function _mem_eta_near(q, p, R, nmax::Int)
    # the relative position may carry a derivative (a center of the assembly)
    C = promote_type(eltype(q.Π), eltype(R))
    npts = size(q.surface, 2)
    U = zeros(C, npts, _mem_block(nmax))
    for j in 1:npts
        U[j, :] .= _mem_read(_mem_reading(q), C, R .+ view(q.surface, :, j), p, nmax)
    end
    return q.Π * U
end

# What particle q reads at x of the decaying harmonics of p: their values, or
# their derivative along e₃ (a crack, `mem_cracks.jl`).
_mem_reading(q) = _MemValues()
_mem_reading(q::_MemSpheroid) = q.reading
_mem_read(::_MemValues, ::Type{C}, x, p, nmax::Int) where {C} = _mem_irregular!(zeros(C, _mem_block(nmax)), x, p, nmax)
function _mem_read(::_MemNormalDerivative, ::Type{C}, x, p, nmax::Int) where {C}
    along(h) = _mem_irregular!(zeros(promote_type(C, typeof(h)), _mem_block(nmax)), x .+ (0, 0, h), p, nmax)
    return ForwardDiff.derivative(along, zero(real(C)))
end

# The same by the far form, from the table `Y` of the decaying spherical
# harmonics at R (or their sums over the images).
function _mem_eta_far(q::_MemSpheroid, p::_MemSpheroid, Y, nmax::Int)
    C = eltype(q.Z)
    R = real(C)
    rmax = _MEM_FAR_TERMS
    hp, hq = p.c / 2, q.c / 2
    # G[n + 1, r + 1] = 1/(r! Γ(n + r + 3/2))
    G = [exp(-logfactorial(r) - loggamma(n + r + 1.5)) for n in 0:nmax, r in 0:rmax]
    η = zeros(C, _mem_block(nmax), _mem_block(nmax))
    for t in 1:nmax, k in 1:nmax
        Mtk = [sum(hp^(2 * (r - j)) * G[t + 1, r - j + 1] * hq^(2j) * G[k + 1, j + 1] for j in 0:r) for r in 0:rmax]
        pre = R(π) * (k + R(1) / 2) * hp^(t + 1) * hq^k
        for s in (-t):t, l in (-k):k
            acc = sum(Y[t + k + 2r, s - l] * Mtk[r + 1] for r in 0:rmax)
            η[_mem_index(k, l), _mem_index(t, s)] = (isodd(s + k + l) ? -pre : pre) * acc
        end
    end
    return η
end

# All the coefficients η of the field that particle p (and its images) sends
# to particle q, centered at −R from q; `nothing` when there is none.
function _mem_eta_total(::MixedBC, q, p, R, self::Bool, nmax::Int)
    self && return nothing
    norm(TensND.primal.(R)) < _MEM_NEAR * (_mem_focal(q) + _mem_focal(p)) && return _mem_eta_near(q, p, R, nmax)
    return _mem_eta_far(q, p, irregular_solid_harmonics(R, _mem_far_degree(q, nmax)), nmax)
end

function _mem_eta_total(b::PeriodicBox, q, p, R, self::Bool, nmax::Int)
    L = box_periods(b)
    rn = _MEM_NEAR * (_mem_focal(q) + _mem_focal(p))
    near = _mem_near_images(R, L, rn, self)
    η = _mem_eta_far(q, p, _mem_far_sums(R, L, near, rn, _mem_far_degree(q, nmax)), nmax)
    for n in near
        η .+= _mem_eta_near(q, p, R .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]), nmax)
    end
    return η
end

# The images of a particle at R from another, other than the particle itself,
# closer than rn.
function _mem_near_images(R, L, rn, self::Bool)
    reach = ntuple(i -> ceil(Int, (rn + abs(TensND.primal(R[i]))) / TensND.primal(L[i])), 3)
    near = NTuple{3, Int}[]
    for n₁ in (-reach[1]):reach[1], n₂ in (-reach[2]):reach[2], n₃ in (-reach[3]):reach[3]
        self && n₁ == n₂ == n₃ == 0 && continue
        x = TensND.primal.(R .+ (n₁ * L[1], n₂ * L[2], n₃ * L[3]))
        norm(x) < rn && push!(near, (n₁, n₂, n₃))
    end
    return near
end

# Sums of the decaying spherical harmonics over the images not in `near`, those
# farther than `rn`. Taking the near images out of the whole lattice sum
# cancels terms that grow as (rn/r_min)^N, so the sums are formed in two ways:
#   - up to the degree `_MEM_EWALD_DEGREE`, where the sums of the lowest
#     degrees converge only conditionally, by Ewald's splitting with the near
#     images excluded and the parameter lowered so that they lie within half a
#     splitting length (accurate to 1e-9 on the lattices of the tests);
#   - above it, where they converge absolutely, directly over the images up
#     to `_MEM_DIRECT_REACH` times rn or the shortest period, whichever is
#     larger (rn vanishes with the focal distances), the rest being below
#     1e-8 of the sum.
const _MEM_EWALD_DEGREE = 8
const _MEM_DIRECT_REACH = 8

# With `Val(true)` the biharmonic sums of `lattice_solid_biharmonics` come along, the
# image at x adding 𝒟_n^m r = −r² Y_n^m/(2n−1) to them above the Ewald degree.
_mem_far_sums(R, L, near, rn, deg::Int) = first(_mem_far_sums(R, L, near, rn, deg, Val(false)))
function _mem_far_sums(R, L, near, rn, deg::Int, bih::Val)
    V = prod(TensND.primal.(L))
    α = sqrt(π) / cbrt(V)
    if !isempty(near)
        rmin = minimum(norm(TensND.primal.(R .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]))) for n in near)
        α = min(α, 1 / (2 * rmin))
    end
    low, lowρ = _mem_lattice_sums(R, L, min(deg, _MEM_EWALD_DEGREE), bih, α, near)
    values = zeros(eltype(low.values), TensND.solid_harmonic_index(deg, deg))
    values[eachindex(low.values)] .= low.values
    ρvalues = _mem_extended(lowρ, values)
    r₂ = _MEM_DIRECT_REACH * max(rn, minimum(TensND.primal.(L)))
    reach = ntuple(i -> ceil(Int, (r₂ + abs(TensND.primal(R[i]))) / TensND.primal(L[i])), 3)
    first_high = TensND.solid_harmonic_index(_MEM_EWALD_DEGREE + 1, 0)
    for n₁ in (-reach[1]):reach[1], n₂ in (-reach[2]):reach[2], n₃ in (-reach[3]):reach[3]
        x = R .+ (n₁ * L[1], n₂ * L[2], n₃ * L[3])
        r = norm(TensND.primal.(x))
        # the near images, and the particle itself
        (rn ≤ r ≤ r₂ && r > 0) || continue
        Y = irregular_solid_harmonics(x, deg)
        values[first_high:end] .+= view(Y.values, first_high:length(values))
        _mem_add_biharmonic!(ρvalues, Y, sum(abs2, x), _MEM_EWALD_DEGREE + 1)
    end
    return TensND.SolidHarmonicTable(deg, values), _mem_table(deg, ρvalues)
end

# The lattice sums of TensND with the near images left out, and the biharmonic
# ones with `Val(true)`.
_mem_lattice_sums(R, L, deg::Int, ::Val{false}, α, near) = (lattice_solid_harmonics(R, L, deg; α, exclude = near), nothing)
_mem_lattice_sums(R, L, deg::Int, ::Val{true}, α, near) = lattice_solid_biharmonics(R, L, deg; α, exclude = near)

# The biharmonic sums widened to the degree of `values`, or nothing.
_mem_extended(::Nothing, values) = nothing
function _mem_extended(low::TensND.SolidHarmonicTable, values)
    ρ = zero(values)
    ρ[eachindex(low.values)] .= low.values
    return ρ
end
_mem_table(deg::Int, ::Nothing) = nothing
_mem_table(deg::Int, ρ::Vector) = TensND.SolidHarmonicTable(deg, ρ)

# 𝒟_n^m r = −r² Y_n^m/(2n−1) of one image, from the degree n₀ up.
_mem_add_biharmonic!(::Nothing, Y, r², n₀::Int) = nothing
function _mem_add_biharmonic!(ρ::Vector, Y, r², n₀::Int)
    for n in n₀:Y.nmax, m in 0:n
        ρ[TensND.solid_harmonic_index(n, m)] -= r² * Y[n, m] / (2n - 1)
    end
    return ρ
end

# The uniform gradient −4π d/(3V) a dipole d of the lattice sends to every
# particle, d = dipole_p · b_p and a_q = gradient_q · g.
_mem_shaped_background!(M, rows, cols, ::MixedBC, q, p, W) = M
function _mem_shaped_background!(M, rows, cols, b::PeriodicBox, q, p, W)
    g = 4π / (3 * prod(box_periods(b)))
    @views M[rows, cols] .+= g .* q.scales .* (q.gradient * (p.dipole * W))
    return M
end

# The linear system of any shaped particles: their received scaled coefficients
# under the three unit remote gradients.
function _mem_shaped_solve(
        asm::ParticleAssembly, names, raw, nmax::Int;
        centers = [particle_center(asm, nm) for nm in names], boundary = asm.boundary, axes = (1, 2, 3)
    )
    # one element type for the particles, their responses and their centers,
    # any of which may carry a derivative
    TX = mapreduce(c -> mapreduce(typeof, promote_type, c), promote_type, centers)
    C = promote_type(mapreduce(d -> eltype(d.Z), promote_type, raw), Complex{TX})
    data = [_mem_convert(d, C) for d in raw]
    X = [SVector{3, TX}(c) for c in centers]
    N, nb = length(data), _mem_block(nmax)
    _check_system_size(N * nb, C, N, nmax)
    M = zeros(C, N * nb, N * nb)
    B = zeros(eltype(M), N * nb, 3)
    for q in 1:N
        rows = ((q - 1) * nb + 1):(q * nb)
        for i in rows
            M[i, i] = one(eltype(M))
        end
        B[rows, :] .= data[q].scales .* data[q].gradient
        for p in 1:N
            η = _mem_eta_total(boundary, data[q], data[p], X[q] - X[p], p == q, nmax)
            η === nothing && continue
            cols = ((p - 1) * nb + 1):(p * nb)
            W = data[p].Z ./ transpose(data[p].scales)          # b_p from the scaled â_p
            @views M[rows, cols] .-= data[q].scales .* (η * W)
            _mem_shaped_background!(M, rows, cols, boundary, data[q], data[p], W)
        end
    end
    â = M \ B
    return ShapedMultipoleSolution(
        copy(names), [d.scales for d in data], [d.Z for d in data], [d.dipole for d in data], â, nmax, axes
    )
end

_mem_spheroid_solve(asm::ParticleAssembly, K₀, names, spheroids, nmax::Int) =
    _mem_shaped_solve(asm, names, [_mem_spheroid_data(s, K₀, nmax) for s in spheroids], nmax)

function _mem_contribution(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, sol::ShapedMultipoleSolution)
    k₀ = LayeredSpheres._iso_scalar(_mem_isotropic(K₀))
    nb = _mem_block(sol.nmax)
    # the dipoles under the three unit gradients, one column each
    D = real.(
        sum(
            sol.dipoles[q] * ((sol.responses[q] ./ transpose(sol.scales[q])) * view(sol.received, ((q - 1) * nb + 1):(q * nb), :))
                for q in eachindex(sol.names)
        )
    )
    # back to the frame of the assembly
    Dg = zeros(eltype(D), 3, 3)
    Dg[collect(sol.axes), collect(sol.axes)] .= D
    return TensND.Tens((-4π * k₀ / assembly_volume(asm)) .* (Dg .+ transpose(Dg)) ./ 2)
end
