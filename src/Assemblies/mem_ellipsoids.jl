# =============================================================================
#  mem_ellipsoids.jl — the multipole expansion method for ellipsoids in
#  conduction (kushch2017, kushch2020).
#
#  Every particle is an ellipsoid of semi-axes a₁ > a₂ > a₃, all oriented
#  alike, with a perfect interface. About it the temperature in the matrix is
#
#       T = Σ a_n^m 𝔼_n^m + Σ b_n^m 𝔽_n^m ,
#
#  with 𝔼, 𝔽 the interior and exterior ellipsoidal harmonics of
#  `Harmonics/ellipsoidal.jl` in the coordinates of the particle. On its surface
#  ρ = a₁ they are E_n^m(a₁) S_n^m and F_n^m(a₁) S_n^m, and the continuity of the
#  temperature and of the normal flux, harmonic by harmonic, gives the response
#
#       b_n^m = (k̃ − 1) / [F′/E′ − k̃ F/E] a_n^m ,   k̃ = k₁/k₀ ,
#
#  the functions being taken at a₁ (the same form as for spheroids).
#
#  Between two particles the near translation of `mem_spheroids.jl` holds as
#  it is: the received field is projected on the surface of the receiver,
#  where the S_n^m are orthogonal for the weight dΩ of `ellipsoidal.jl`. On a
#  periodic cell, the images beyond the threshold go through the spherical
#  harmonics: 𝔽 = Σ μ Y on a sphere of radius 2a₁, which holds the focal ellipse
#  (kushch2020, Eq. (38)), the spherical translation with the lattice sums, and
#  y = Σ ν 𝔼 by projection on the surface of the receiver (Eq. (41)).
#
#  The ellipsoids of an assembly share their orientation; the problem is
#  solved in the frame of their axes, ordered by decreasing semi-axis, and the
#  effective conductivity is turned back to the frame of the assembly.
# =============================================================================

# Degrees of the spherical harmonics beyond `nmax` in the far translation of
# ellipsoids: the series behaves as ((h₂ + h₂′)/|R|)^{2r} (kushch2020, § 4.2),
# below the rounding at the threshold `_MEM_NEAR` for this many degrees.
const _MEM_ELLIPSOID_EXTRA = 30

struct _MemEllipsoid{C, R}
    g::Harmonics.EllipsoidalGeometry
    lame::Vector{Harmonics.LameFunction}   # by unknown index
    scales::Vector{R}
    Z::Matrix{C}
    Π::Matrix{C}
    surface::Matrix{R}
    dipole::Matrix{C}
    gradient::Matrix{C}
    nodes::Tuple{Vector{Float64}, Vector{Float64}}   # quadrature of the Lamé functions of the second kind
    μ::Matrix{C}        # 𝔽 → spherical Y, rows by spherical index (empty without far pairs)
    ν::Matrix{C}        # spherical y → 𝔼 (empty without far pairs)
end

_mem_focal(d::_MemEllipsoid) = d.g.h2
_mem_convert(d::_MemEllipsoid, ::Type{C}) where {C} = _MemEllipsoid{C, real(C)}(
    d.g, d.lame, d.scales, d.Z, d.Π, d.surface, d.dipole, d.gradient, d.nodes, d.μ, d.ν
)

# The semi-axes of an aligned ellipsoid and the permutation of the canonical
# axes that orders them, a₁ along the first axis of the frame.
function _mem_ellipsoid_axes(geom::Elasticity.Ellipsoid{3}, name::Symbol)
    # the Lamé functions are built on eigenvalues and on the roots of a cubic,
    # compared with one another: the semi-axes are plain numbers
    eltype(geom.semi_axes) <: AbstractFloat || throw(
        ArgumentError(
            "MultipoleExpansion: the semi-axes of the ellipsoid :$(name) are " *
                "$(eltype(geom.semi_axes)); an ellipsoid of three different semi-axes takes plain " *
                "numbers (the positions and the conductivities may carry derivatives)"
        )
    )
    A = TensND.primal.(Array(TensND.canonical_components(MFH_Core.shape_tensor(geom))))
    iszero(A[1, 2]) && iszero(A[1, 3]) && iszero(A[2, 3]) || throw(
        ArgumentError(
            "MultipoleExpansion: the ellipsoid :$(name) is not aligned with the axes of the " *
                "assembly; the ellipsoids of an assembly are aligned with e₁, e₂, e₃"
        )
    )
    d = (A[1, 1], A[2, 2], A[3, 3])
    order = sortperm(collect(d); rev = true)
    return d[order], Tuple(order)
end

# The interface of a particle: perfect for a plain ellipsoid.
_mem_interface(::Elasticity.Ellipsoid) = LayeredSpheres.PerfectInterface{Float64}()
_mem_interface(ie::InterfacedEllipsoid) = ie.interface
_mem_ellipsoid_axes(ie::InterfacedEllipsoid, name::Symbol) = _mem_ellipsoid_axes(ie.ellipsoid, name)

function _mem_ellipsoid_data(semi, k̃, nmax::Int, far::Bool, interface = LayeredSpheres.PerfectInterface{Float64}(), k₁ = k̃)
    g = Harmonics.EllipsoidalGeometry(semi...)
    nb = _mem_block(nmax)
    a₁ = g.a[1]
    # the interface parameter may carry a derivative of its own
    T = promote_type(typeof(k̃), typeof(k₁), eltype(interface))
    C = Complex{T}
    lame = Vector{Harmonics.LameFunction}(undef, nb)
    E₀ = zeros(nb)
    Z = zeros(C, nb, nb)
    dE, F₀, dF = zeros(nb), zeros(nb), zeros(nb)
    nodes = LayeredSpheroids.QuadGK.gauss(nmax + 24, 0.0, 1.0)
    for n in 1:nmax, (j, E) in enumerate(Harmonics.lame_functions(g, n))
        i = _mem_index(n, j - n - 1)
        lame[i] = E
        E₀[i], dE[i] = Harmonics.lame_value(E, a₁, sqrt(a₁^2 - g.h3^2), sqrt(a₁^2 - g.h2^2), 1, 1)
        F₀[i], dF[i] = Harmonics.lame_second_kind(g, E, a₁, nodes)
    end
    scales = abs.(E₀)
    Π, surface, S, w, hρ = _mem_ellipsoid_projection(g, lame, E₀, nmax)
    _mem_ellipsoid_response!(Z, interface, k̃, k₁, E₀, dE, F₀, dF, S, w, hρ)
    D = zeros(C, 3, nb)
    G = zeros(C, nb, 3)
    for (j, factor) in enumerate((g.h2 * g.h3, g.h1 * g.h3, g.h1 * g.h2))
        # 𝔼₁ = factor · x_j and 𝔽₁ ≈ factor · x_j/r³ far away, by class K, L, M
        i = _mem_index(1, j - 2)
        D[j, i] = factor
        G[i, j] = 1 / factor
    end
    μ, ν = far ? _mem_ellipsoid_far_maps(g, lame, nodes, Π, surface, nmax) : (zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0))
    return _MemEllipsoid{C, real(C)}(g, lame, scales, Z, C.(Π), surface, D, G, nodes, μ, ν)
end

# Quadrature of the surface ρ = a₁, μ² = h₃² + (h₂² − h₃²) sin²α (α ∈ [0, π],
# Gauss–Legendre on an even number of points, none at α = π/2), ν = h₃ sin β
# (β ∈ [0, 2π), trapezoidal), weight (μ² − ν²)/(μ √(h₂² − ν²)) dα dβ, and the
# projection a_n^m = Σ w u S_n^m / (E_n^m(a₁) γ_n^m), γ = Σ w S².
function _mem_ellipsoid_projection(g, lame, E₀, nmax::Int)
    nα = 2 * nmax + 24
    nβ = 2 * nα
    xα, wα = LayeredSpheroids.QuadGK.gauss(nα, 0.0, π)
    h₁, h₂, h₃ = g.h1, g.h2, g.h3
    ρ = g.a[1]
    nb = length(lame)
    S = zeros(nb, nα * nβ)
    w = zeros(nα * nβ)
    surface = zeros(3, nα * nβ)
    for (i, (α, wa)) in enumerate(zip(xα, wα)), j in 1:nβ
        # half a step off the planes ν = 0 and ν = ±h₃, where the coordinates,
        # smooth in the harmonics, have square roots without derivative
        β = 2π * (j - 0.5) / nβ
        μ = sqrt(h₃^2 + (h₂^2 - h₃^2) * sin(α)^2)
        ν = h₃ * sin(β)
        c = (;
            ρ, μ, ν, ρ3 = sqrt(ρ^2 - h₃^2), ρ2 = sqrt(ρ^2 - h₂^2),
            μ3 = sqrt(h₂^2 - h₃^2) * sin(α), μ2 = sqrt(h₂^2 - h₃^2) * cos(α),
            ν3 = h₃ * cos(β), ν2 = sqrt(h₂^2 - ν^2),
        )
        col = (i - 1) * nβ + j
        w[col] = wa * (2π / nβ) * (μ^2 - ν^2) / (μ * c.ν2)
        surface[:, col] .= (ρ * μ * ν / (h₂ * h₃), c.ρ3 * c.μ3 * c.ν3 / (h₁ * h₃), c.ρ2 * c.μ2 * c.ν2 / (h₁ * h₂))
        for (k, E) in enumerate(lame)
            S[k, col] = Harmonics._lame_at_μ(E, c)[1] * Harmonics._lame_at_ν(E, c)[1]
        end
    end
    γ = S .^ 2 * w
    Π = (S .* transpose(w)) ./ (E₀ .* γ)
    # metric coefficient of ρ on the surface, ∂/∂n = (1/h_ρ) ∂/∂ρ
    hρ = [
        begin
            c = Harmonics.ellipsoidal_coordinates(g, surface[:, col])
            sqrt(ρ^2 - c.μ^2) * sqrt(ρ^2 - c.ν^2) / (sqrt(ρ^2 - h₃^2) * sqrt(ρ^2 - h₂^2))
        end for col in axes(surface, 2)
    ]
    return Π, surface, S, w, hρ
end

# The response b = Z a. Through a perfect interface each harmonic answers
# alone: continuity of T and of the normal flux give
#     b = (k̃ − 1) / (F′/E′ − k̃ F/E) a .
# Through a Kapitza interface [T] = −ρ q_n the flux is still continuous mode by
# mode, d = (b F′ + a E′)/(k̃ E′) inside, while the jump, projected on S_j with
# the weight dΩ, couples the harmonics through ∫ S_j S_k / h_ρ dΩ:
#     γ_j (b_j F_j + a_j E_j − d_j E_j) = ρ k₁ Σ_k (∫ S_j S_k / h_ρ dΩ) E′_k d_k .
function _mem_ellipsoid_response!(Z, ::LayeredSpheres.PerfectInterface, k̃, k₁, E, dE, F, dF, S, w, hρ)
    for i in eachindex(E)
        Z[i, i] = (k̃ - 1) / (dF[i] / dE[i] - k̃ * F[i] / E[i])
    end
    return Z
end
function _mem_ellipsoid_response!(Z, intf::LayeredSpheres.KapitzaInterface, k̃, k₁, E, dE, F, dF, S, w, hρ)
    γ = S .^ 2 * w
    Mρ = (S .* transpose(w ./ hρ)) * transpose(S)
    J = Diagonal(γ .* E) + intf.resistance * k₁ .* Mρ .* transpose(dE)   # acting on d
    P, Q = dF ./ (k̃ .* dE), fill(1 / k̃, length(E))                        # d = P b + Q a
    L = Diagonal(γ .* F) - J * Diagonal(P)
    R = Diagonal(γ .* E) - J * Diagonal(Q)
    Z .= -(L \ R)
    return Z
end
_mem_ellipsoid_response!(Z, intf, args...) = throw(
    ArgumentError(
        "MultipoleExpansion: an ellipsoid takes a perfect or a Kapitza interface in conduction, " *
            "not a $(nameof(typeof(intf)))"
    )
)

# The decaying ellipsoidal harmonics 𝔽 (1 ≤ n ≤ nmax) of the ellipsoid at x.
_mem_irregular!(F::AbstractVector, x, e::_MemEllipsoid, nmax::Int) = _mem_ellipsoidal_irregular!(F, x, e.g, e.lame, e.nodes)
function _mem_ellipsoidal_irregular!(F::AbstractVector, x, g, lame, nodes)
    c = Harmonics.ellipsoidal_coordinates(g, x)
    for (k, E) in enumerate(lame)
        f, _ = Harmonics.lame_second_kind(g, E, c.ρ, nodes)
        F[k] = f * Harmonics._lame_at_μ(E, c)[1] * Harmonics._lame_at_ν(E, c)[1]
    end
    return F
end

# μ: the coefficients of 𝔽 on the decaying spherical harmonics about the
# center, read on a sphere of radius r₀ > h₂, where χ_k^l are orthogonal (2a₁:
# on the sphere of radius a₁, close to the focal ellipse, the quadrature loses
# five digits; from 1.3 a₁ on, the far and near forms agree to 1e-9):
#     𝔽 = Σ μ_kl Y_k^l ,  μ_kl = a₁^{k+1} (2k+1)/(4π (k+|l|)!) ∫ 𝔽 conj(χ_k^l) dΩ .
# ν: the coefficients of the regular spherical harmonics on the 𝔼, by the
# projection on the surface of the ellipsoid (exact, y being polynomials).
function _mem_ellipsoid_far_maps(g, lame, nodes, Π, surface, nmax::Int; r₀ = 2 * g.a[1])
    K = nmax + _MEM_ELLIPSOID_EXTRA
    ns = _mem_block(K)
    nθ = K + 4
    nφ = 2K + 4
    xs, ws = LayeredSpheroids.QuadGK.gauss(nθ)
    # μ = Yᵀ F, the rows of Y holding w (2k+1) r₀^{k+1}/(4π) conj(y_k^l) at each point
    Y = zeros(ComplexF64, nθ * nφ, ns)
    F = zeros(nθ * nφ, length(lame))
    f = zeros(length(lame))
    for (i, (t, wt)) in enumerate(zip(xs, ws)), j in 1:nφ
        φ = 2π * (j - 1) / nφ
        x̂ = [sqrt(1 - t^2) * cos(φ), sqrt(1 - t^2) * sin(φ), t]
        row = (i - 1) * nφ + j
        F[row, :] .= _mem_ellipsoidal_irregular!(f, r₀ .* x̂, g, lame, nodes)
        # χ_k^l = (k + |l|)! y_k^l on the unit sphere, the factorial canceling
        y = regular_harmonics(x̂, K)
        for k in 1:K, l in (-k):k
            Y[row, _mem_index(k, l)] = wt * 2π / nφ * r₀^(k + 1) * (2k + 1) / 4π * conj(y[k, l])
        end
    end
    μ = transpose(Y) * F
    # ν = Π y, the regular harmonics on the points of the surface
    Ys = zeros(ComplexF64, size(surface, 2), ns)
    for col in axes(surface, 2)
        y = regular_harmonics(surface[:, col], K)
        for k in 1:K, l in (-k):k
            Ys[col, _mem_index(k, l)] = y[k, l]
        end
    end
    return μ, Π * Ys
end

# The far translation: η = ν_q · T(R) · μ_p, T[(n, l), (k, s)] = (−1)^{n+l} Y_{k+n}^{s−l}(R).
function _mem_eta_far(q::_MemEllipsoid, p::_MemEllipsoid, Y, nmax::Int)
    K = nmax + _MEM_ELLIPSOID_EXTRA
    ns = _mem_block(K)
    T = zeros(eltype(q.Z), ns, ns)
    for n in 1:K, l in (-n):n, k in 1:K, s in (-k):k
        T[_mem_index(n, l), _mem_index(k, s)] = (isodd(n + l) ? -1 : 1) * Y[k + n, s - l]
    end
    return q.ν * T * p.μ
end

_mem_far_degree(::_MemEllipsoid, nmax::Int) = 2 * (nmax + _MEM_ELLIPSOID_EXTRA)

# Ellipsoids aligned alike, solved in the frame of their axes by decreasing
# semi-axis: the centers and the periods are permuted to it.
function _mem_ellipsoid_solve(asm::ParticleAssembly, K₀, names, geometries, prop::Symbol, nmax::Int)
    k₀ = LayeredSpheres._iso_scalar(K₀)
    axes = [_mem_ellipsoid_axes(g, nm) for (g, nm) in zip(geometries, names)]
    order = last(first(axes))
    all(a -> last(a) == order, axes) || throw(
        ArgumentError(
            "MultipoleExpansion: the ellipsoids of an assembly share their orientation, the " *
                "largest semi-axis along the same axis for all"
        )
    )
    # the spherical maps serve the far pairs alone: those of a periodic cell, or
    # of a finite assembly whose particles are far enough apart
    centers = [collect(particle_center(asm, nm))[collect(order)] for nm in names]
    far = _mem_needs_far(asm.boundary, centers, [maximum(first(a)) for a in axes])
    raw = map(zip(names, axes)) do (nm, (semi, _))
        _mem_ellipsoid_carrying(particle_property(asm, nm, prop), semi, k₀, nmax, far, _mem_interface(particle_geometry(asm, nm)), order)
    end
    return _mem_shaped_solve(asm, names, raw, nmax; centers, boundary = _mem_permuted(asm.boundary, order), axes = order)
end

# The data of an ellipsoid of isotropic conductivity, or of a perfectly bonded
# one of anisotropic conductivity (`mem_anisotropic.jl`), turned to the frame
# of its axes.
function _mem_ellipsoid_carrying(P::TensND.TensISO, semi, k₀, nmax::Int, far::Bool, interface, order)
    k₁ = LayeredSpheres._iso_scalar(P)
    return _mem_ellipsoid_data(semi, k₁ / k₀, nmax, far, interface, k₁)
end
function _mem_ellipsoid_carrying(P::TensND.AbstractTens{2, 3}, semi, k₀, nmax::Int, far::Bool, interface, order)
    interface isa LayeredSpheres.PerfectInterface || throw(
        ArgumentError(
            "MultipoleExpansion: an ellipsoid of anisotropic conductivity is perfectly bonded; " *
                "got a $(nameof(typeof(interface)))"
        )
    )
    K = Array(TensND.canonical_components(P))[collect(order), collect(order)]
    return _mem_anisotropic(_mem_ellipsoid_data(semi, one(k₀), nmax, far), K, k₀, nmax)
end

_mem_permuted(b::MixedBC, order) = b
_mem_permuted(b::PeriodicBox, order) = PeriodicBox(Tuple(box_periods(b)[collect(order)]))

# Whether a pair goes through the far form: always on a periodic cell; in a
# finite assembly, when two centers are farther apart than the threshold for
# the largest focal distances (bounded by the semi-axes).
_mem_needs_far(::PeriodicBox, centers, radii) = true
function _mem_needs_far(::MixedBC, centers, radii)
    r = 2 * _MEM_NEAR * maximum(radii)
    return any(norm(TensND.primal.(centers[i] - centers[j])) ≥ r for i in eachindex(centers), j in eachindex(centers) if i < j)
end
