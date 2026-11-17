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
    P = particle_property(asm, name, prop)
    P isa TensND.TensISO || throw(
        ArgumentError(
            "MultipoleExpansion: particle :$(name) has a $(nameof(typeof(P))) property " *
                ":$(prop); the spheroids of the expansion are isotropic (TensISO)"
        )
    )
    return LayeredSpheroids.LayeredSpheroid((A[3, 3],), (A[1, 1],), (P,))
end

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
    SpheroidalMultipoleSolution

Solution of the multipole expansion of an assembly of spheroids in
conduction: for each particle, in the order of `names`, its focal distance
`c` (real for a prolate spheroid, ``-\\mathrm{i}`` times the focal distance for
an oblate one), the scales ``|\\bar{P}_n^m(\\xi_0)|`` of its unknowns, its
response ``\\mathbf{Z}`` (block-diagonal over the orders) and the scaled
coefficients it receives under the three unit remote gradients (`received`,
one column per gradient).
"""
struct SpheroidalMultipoleSolution{T, C}
    names::Vector{Symbol}
    c::Vector{C}
    scales::Vector{Vector{T}}
    responses::Vector{Matrix{C}}
    received::Matrix{C}
    nmax::Int
end

_mem_spheres_only(sol::SpheroidalMultipoleSolution, what::String) = throw(
    ArgumentError("MultipoleExpansion: $(what) are available for spheres; the spheroids give the effective property")
)

# Everything the expansion needs of one spheroid.
struct _MemSpheroid{C, R}
    c::C                      # focal distance, complex
    ξ₀::C                     # confocal coordinate of the outer surface
    scales::Vector{R}         # |P̄_n^m(ξ₀)| of the unknowns
    Z::Matrix{C}              # response, over the unknowns
    Π::Matrix{C}              # projection of surface values on the a_k^l
    surface::Matrix{R}        # quadrature points of the surface, 3 × npoints
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
    Π, surface = _mem_surface_projection(c, ξ₀, nmax)
    return _MemSpheroid{C, R}(c, ξ₀, scales, Z, Π, surface)
end

# Quadrature of the outer surface ξ = ξ₀: the points, and the matrix Π that
# turns the values there of a regular field into its coefficients a_k^l,
#     a_k^l = (2k+1)/(4π P_k^{|l|}(ξ₀)) Σ w conj(χ_k^l) u .
function _mem_surface_projection(c::C, ξ₀::C, nmax::Int) where {C}
    R = real(C)
    nη = 2 * nmax + 24
    nφ = 2 * nη
    xg, wg = LayeredSpheroids.QuadGK.gauss(nη)
    nb = _mem_block(nmax)
    Π = zeros(C, nb, nη * nφ)
    surface = zeros(R, 3, nη * nφ)
    ρ₀, z₀ = real(c * sqrt(ξ₀^2 - 1)), real(c * ξ₀)      # transverse and axial semi-axes
    Pξ = [LayeredSpheroids.legendre_order_q(ξ₀, m, nmax)[1] for m in 0:nmax]
    for (j, (η, w)) in enumerate(zip(xg, wg))
        Pη = [LayeredSpheroids.legendre_order_p(η, m, nmax)[1] for m in 0:nmax]
        η̄ = sqrt(1 - η^2)
        for i in 1:nφ
            φ = 2 * R(π) * (i - 1) / nφ
            col = (j - 1) * nφ + i
            surface[:, col] .= (ρ₀ * η̄ * cos(φ), ρ₀ * η̄ * sin(φ), z₀ * η)
            for k in 1:nmax, l in (-k):k
                μ = abs(l)
                χ = _mem_surface_harmonic(Pη[μ + 1][k + 1], l, φ)
                Π[_mem_index(k, l), col] = (2k + 1) / (4 * R(π) * Pξ[μ + 1][k + 1]) * w * (2 * R(π) / nφ) * conj(χ)
            end
        end
    end
    return Π, surface
end

# χ_k^l from P_k^{|l|} (Ferrers): (−1)^l P e^{ilφ} for l ≥ 0, P e^{ilφ} for l < 0.
_mem_surface_harmonic(P, l::Int, φ) = (l ≥ 0 && isodd(l) ? -P : P) * cis(l * φ)

# Decaying spheroidal harmonics F_t^s (1 ≤ t ≤ nmax) of focal distance c at x.
function _mem_spheroidal_irregular!(F::AbstractVector, x, c, nmax::Int)
    ξ, η, φ = _mem_spheroidal_coordinates(x, c)
    Q = LayeredSpheroids._legendre_q_orders(ξ, nmax, nmax)
    for m in 0:nmax
        Pη = LayeredSpheroids.legendre_order_p(η, m, nmax)[1]
        for t in max(m, 1):nmax
            radial = LayeredSpheroids._factorial_ratio(real(typeof(ξ)), t, m) * Q[t + 1, m + 1]
            F[_mem_index(t, m)] = radial * _mem_surface_harmonic(Pη[t + 1], m, φ)
            m > 0 && (F[_mem_index(t, -m)] = radial * _mem_surface_harmonic(Pη[t + 1], -m, φ))
        end
    end
    return F
end

# Confocal coordinates (ξ, η, φ) of x for the focal distance c: c real, a
# prolate spheroid, x₃ = c ξ η; c = −i f, an oblate one, ξ = i s with
# x₃ = f s η and ρ = f √(1+s²) η̄.
function _mem_spheroidal_coordinates(x, c)
    ρ = hypot(x[1], x[2])
    φ = atan(x[2], x[1])
    if iszero(imag(c))
        f = real(c)
        r₁, r₂ = hypot(ρ, x[3] - f), hypot(ρ, x[3] + f)
        return complex((r₁ + r₂) / (2f)), (r₂ - r₁) / (2f), φ
    end
    f = abs(c)
    a = (ρ^2 + x[3]^2) / f^2 - 1
    s = sqrt((a + sqrt(a^2 + 4 * x[3]^2 / f^2)) / 2)
    return complex(zero(s), s), x[3] / (f * s), φ
end

# Coefficients η of the regular expansion about particle q of the decaying
# harmonics of particle p, centered at −R from q: by projection on the surface.
function _mem_eta_near(q::_MemSpheroid{C}, p::_MemSpheroid{C}, R, nmax::Int) where {C}
    npts = size(q.surface, 2)
    U = zeros(C, npts, _mem_block(nmax))
    F = zeros(C, _mem_block(nmax))
    for j in 1:npts
        _mem_spheroidal_irregular!(F, R .+ view(q.surface, :, j), p.c, nmax)
        U[j, :] .= F
    end
    return q.Π * U
end

# The same by the far form, from the table `Y` of the decaying spherical
# harmonics at R (or their sums over the images).
function _mem_eta_far(q::_MemSpheroid{C}, p::_MemSpheroid{C}, Y, nmax::Int) where {C}
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
function _mem_eta_total(::MixedBC, q::_MemSpheroid, p::_MemSpheroid, R, self::Bool, nmax::Int)
    self && return nothing
    norm(TensND.primal.(R)) < _MEM_NEAR * (abs(q.c) + abs(p.c)) && return _mem_eta_near(q, p, R, nmax)
    return _mem_eta_far(q, p, irregular_harmonics(R, 2 * (nmax + _MEM_FAR_TERMS)), nmax)
end

function _mem_eta_total(b::PeriodicBox, q::_MemSpheroid, p::_MemSpheroid, R, self::Bool, nmax::Int)
    L = box_periods(b)
    rn = _MEM_NEAR * (abs(q.c) + abs(p.c))
    reach = ntuple(i -> ceil(Int, (rn + abs(TensND.primal(R[i]))) / TensND.primal(L[i])), 3)
    near = NTuple{3, Int}[]
    for n₁ in (-reach[1]):reach[1], n₂ in (-reach[2]):reach[2], n₃ in (-reach[3]):reach[3]
        self && n₁ == n₂ == n₃ == 0 && continue
        x = TensND.primal.(R .+ (n₁ * L[1], n₂ * L[2], n₃ * L[3]))
        norm(x) < rn && push!(near, (n₁, n₂, n₃))
    end
    η = _mem_eta_far(q, p, _mem_far_sums(R, L, near, rn, nmax), nmax)
    for n in near
        η .+= _mem_eta_near(q, p, R .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]), nmax)
    end
    return η
end

# Sums of the decaying spherical harmonics over the images not in `near`, those
# farther than `rn`. Taking the near images out of the whole lattice sum
# cancels terms that grow as (rn/r_min)^N, so the sums are formed in two ways:
#   - up to the degree `_MEM_EWALD_DEGREE`, where the sums of the lowest
#     degrees converge only conditionally, by Ewald's splitting with the near
#     images excluded and the parameter lowered so that they lie within half a
#     splitting length (accurate to 1e-9 on the lattices of the tests);
#   - above it, where they converge absolutely, directly over the images up
#     to `_MEM_DIRECT_REACH` rn, the rest being below 1e-8 of the sum.
const _MEM_EWALD_DEGREE = 8
const _MEM_DIRECT_REACH = 8

function _mem_far_sums(R, L, near, rn, nmax::Int)
    deg = 2 * (nmax + _MEM_FAR_TERMS)
    V = prod(TensND.primal.(L))
    α = sqrt(π) / cbrt(V)
    if !isempty(near)
        rmin = minimum(norm(TensND.primal.(R .+ (n[1] * L[1], n[2] * L[2], n[3] * L[3]))) for n in near)
        α = min(α, 1 / (2 * rmin))
    end
    low = first(Harmonics._lattice_sums(R, L, min(deg, _MEM_EWALD_DEGREE), Val(false), α; exclude = near))
    values = zeros(eltype(low.values), Harmonics._packed(deg, deg))
    values[eachindex(low.values)] .= low.values
    r₂ = _MEM_DIRECT_REACH * rn
    reach = ntuple(i -> ceil(Int, (r₂ + abs(TensND.primal(R[i]))) / TensND.primal(L[i])), 3)
    first_high = Harmonics._packed(_MEM_EWALD_DEGREE + 1, 0)
    for n₁ in (-reach[1]):reach[1], n₂ in (-reach[2]):reach[2], n₃ in (-reach[3]):reach[3]
        x = R .+ (n₁ * L[1], n₂ * L[2], n₃ * L[3])
        r = norm(TensND.primal.(x))
        rn ≤ r ≤ r₂ || continue
        values[first_high:end] .+= view(irregular_harmonics(x, deg).values, first_high:length(values))
    end
    return Harmonics.HarmonicTable(deg, values)
end

# The uniform gradient −4π d/(3V) a spherical dipole d of the lattice sends to
# every particle: with d the dipoles (−1)^m c_p² b_1^m/3 of p and y_1^m = c_q f_1^m,
# a_1^m of q gains −(4π/3V) (1−m)!(1+m)! c_q (−1)^m c_p² b_1^m/3.
_mem_spheroid_background!(M, rows, cols, ::MixedBC, q, p, W) = M
function _mem_spheroid_background!(M, rows, cols, b::PeriodicBox, q::_MemSpheroid, p::_MemSpheroid, W)
    g = 4π / (3 * prod(box_periods(b)))
    for m in -1:1
        i = _mem_index(1, m)
        w = g * factorial(1 - m) * factorial(1 + m) * q.c * (isodd(m) ? -1 : 1) * p.c^2 / 3 * q.scales[i]
        @views M[rows[i], cols] .+= w .* W[i, :]
    end
    return M
end

function _mem_spheroid_solve(asm::ParticleAssembly, K₀, names, spheroids, nmax::Int)
    raw = [_mem_spheroid_data(s, K₀, nmax) for s in spheroids]
    C = mapreduce(d -> eltype(d.Z), promote_type, raw)
    data = [_MemSpheroid{C, real(C)}(d.c, d.ξ₀, d.scales, d.Z, d.Π, d.surface) for d in raw]
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    X = [SVector{3, TX}(c) for c in centers]
    N, nb = length(data), _mem_block(nmax)
    _check_system_size(N * nb, C, N, nmax)
    M = zeros(promote_type(C, TX), N * nb, N * nb)
    B = zeros(eltype(M), N * nb, 3)
    for q in 1:N
        rows = ((q - 1) * nb + 1):(q * nb)
        for i in rows
            M[i, i] = one(eltype(M))
        end
        _mem_spheroid_remote!(B, rows, data[q])
        for p in 1:N
            η = _mem_eta_total(asm.boundary, data[q], data[p], X[q] - X[p], p == q, nmax)
            η === nothing && continue
            cols = ((p - 1) * nb + 1):(p * nb)
            W = data[p].Z ./ transpose(data[p].scales)          # b_p from the scaled â_p
            @views M[rows, cols] .-= data[q].scales .* (η * W)
            _mem_spheroid_background!(M, rows, cols, asm.boundary, data[q], data[p], W)
        end
    end
    â = M \ B
    return SpheroidalMultipoleSolution(
        copy(names), [d.c for d in data], [d.scales for d in data], [d.Z for d in data], â, nmax
    )
end

# H·x = H₃ y₁⁰ + (H₁ − iH₂) y₁¹ − (H₁ + iH₂) y₁⁻¹ with y₁ᵐ = c f₁ᵐ, scaled, for
# the three unit gradients.
function _mem_spheroid_remote!(B, rows, q::_MemSpheroid)
    s(m) = q.c * q.scales[_mem_index(1, m)]
    B[rows[_mem_index(1, 1)], 1] = s(1)
    B[rows[_mem_index(1, -1)], 1] = -s(-1)
    B[rows[_mem_index(1, 1)], 2] = -im * s(1)
    B[rows[_mem_index(1, -1)], 2] = -im * s(-1)
    B[rows[_mem_index(1, 0)], 3] = s(0)
    return B
end

function _mem_contribution(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, sol::SpheroidalMultipoleSolution)
    k₀ = LayeredSpheres._iso_scalar(_mem_isotropic(K₀))
    nb = _mem_block(sol.nmax)
    D = sum(eachindex(sol.names)) do q
        â = view(sol.received, ((q - 1) * nb + 1):(q * nb), :)
        Wq = sol.responses[q] ./ transpose(sol.scales[q])
        # spherical dipoles (−1)^m c² b₁ᵐ/3 under the three unit gradients
        d(m) = (isodd(m) ? -1 : 1) * sol.c[q]^2 / 3 .* vec(transpose(Wq[_mem_index(1, m), :]) * â)
        d₀, d₁ = d(0), 2 .* d(1)
        real.(permutedims(hcat(d₁, im .* d₁, d₀)))
    end
    D = real.(D)
    return TensND.Tens((-4π * k₀ / assembly_volume(asm)) .* (D .+ transpose(D)) ./ 2)
end
