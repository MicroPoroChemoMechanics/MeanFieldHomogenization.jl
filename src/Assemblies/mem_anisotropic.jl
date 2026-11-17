# =============================================================================
#  mem_anisotropic.jl — anisotropic constituents in conduction by the multipole
#  expansion method (kushch2014, kushch2017).
#
#  MATRIX. With K₀ = k̄ 𝐋², k̄ = (det K₀)^{1/3} and 𝐋 symmetric of determinant
#  one, the change of variables x = 𝐋 v turns div(K₀ grad T) = 0 into
#  k̄ Δ_v T = 0: in the variables v the matrix is isotropic of conductivity k̄.
#  A particle of conductivity K becomes one of conductivity 𝐋⁻¹ K 𝐋⁻¹, its
#  surface x·𝐒 x = 1 the surface v·(𝐋 𝐒 𝐋) v = 1, and a center X the center
#  𝐋⁻¹ X. The volumes are kept (det 𝐋 = 1), and an effective conductivity K_v
#  found in the variables v is K = 𝐋 K_v 𝐋 in the variables x, both having the
#  same energy. Written here for a matrix orthotropic in the axes of the
#  assembly, 𝐋 diagonal: the particles keep their axes there and the
#  orthorhombic cells stay orthorhombic.
#
#  PARTICLE. In an isotropic matrix of conductivity k₀, a perfectly bonded
#  particle of anisotropic conductivity K₁ = 𝚲 𝚲ᵀ (Cholesky) carries inside
#  the field T = Σ_j D_j g_j(𝚲⁻¹ x), g_j the regular solid harmonics, which
#  satisfies div(K₁ grad T) = 0 since 𝚲⁻¹ K₁ 𝚲⁻ᵀ = 𝟏. On the surface
#  x·𝐁 x = 1 the outer normal is along 𝐁 x, and continuity of the
#  temperature and of the normal flux read
#
#       Σ a_k f_k + Σ b_k F_k = Σ D_j g_j(𝚲⁻¹x) ,
#       k₀ (𝐁 x)·grad(Σ a_k f_k + Σ b_k F_k) = Σ D_j (𝐁 x)·𝚲 grad g_j(𝚲⁻¹x) ,
#
#  projected on the surface harmonics of the particle with the projection Π
#  of its harmonics (kushch2017, Eqs. (3.7), (3.8)). The regular field is
#  taken as the regular spherical harmonics y_j, whose coefficients on the
#  harmonics f of the particle are Π y_j, so that b = 𝐙 Π y_j for each j
#  gives the response 𝐙. For an isotropic particle the exact response lies
#  in these spaces and is recovered.
# =============================================================================

# ─── The affine map of an orthotropic matrix ─────────────────────────────────

# k̄ and the diagonal 𝐋 of K₀ = k̄ 𝐋²; refused for a matrix not orthotropic in the
# axes of the assembly, and for one that carries a derivative or a symbol (its
# anisotropy sets the shape of every particle in the variables v, and the
# harmonics of those shapes are built in floating point).
function _mem_affine(K₀::TensND.AbstractTens{2, 3})
    K = Array(TensND.canonical_components(K₀))
    isdiag(K) || throw(
        ArgumentError(
            "MultipoleExpansion: the matrix conductivity must be orthotropic in the axes of the " *
                "assembly (a diagonal tensor there); got $(K)"
        )
    )
    eltype(K) <: AbstractFloat || throw(
        ArgumentError(
            "MultipoleExpansion: an anisotropic matrix conductivity takes plain floating-point " *
                "components (got $(eltype(K))): its anisotropy sets the shapes of the particles"
        )
    )
    d = diag(K)
    k̄ = cbrt(prod(d))
    return Diagonal(sqrt.(d ./ k̄)), k̄
end

# The assembly in the variables v, where the matrix is isotropic.
function _mem_mapped_assembly(asm::ParticleAssembly{T}, prop::Symbol, L::Diagonal, k̄) where {T}
    out = ParticleAssembly(asm.matrix_name; boundary = _mem_mapped_boundary(asm.boundary, L), T)
    add_matrix!(out, Dict(prop => TensND.TensISO{3}(k̄)))
    for nm in particle_names(asm)
        shape = _mem_mapped_shape(particle_geometry(asm, nm), L, nm)
        K = Array(TensND.canonical_components(particle_property(asm, nm, prop)))
        add_particle!(out, nm, Tuple(L \ collect(particle_center(asm, nm))), shape, Dict(prop => TensND.Tens(L \ K / L)))
    end
    return out
end

_mem_mapped_boundary(b::MixedBC, L::Diagonal) = MixedBC(_mem_mapped_shape(b.shape, L, :boundary))
_mem_mapped_boundary(b::PeriodicBox, L::Diagonal) = PeriodicBox(Tuple(box_periods(b) ./ diag(L)))

# An ellipsoid with its axes along those of the assembly stays one, its
# semi-axes divided by the diagonal of 𝐋.
function _mem_mapped_shape(geom::Elasticity.Ellipsoid{3}, L::Diagonal, name::Symbol)
    A = Array(TensND.canonical_components(MFH_Core.shape_tensor(geom)))
    isdiag(A) || throw(
        ArgumentError(
            "MultipoleExpansion: in an anisotropic matrix the ellipsoid :$(name) must have its " *
                "axes along those of the assembly"
        )
    )
    return Elasticity.Ellipsoid((diag(A) ./ diag(L))...)
end
_mem_mapped_shape(geom, L::Diagonal, name::Symbol) = throw(
    ArgumentError(
        "MultipoleExpansion: in an anisotropic matrix the particle :$(name) must be a plain " *
            "ellipsoid, perfectly bonded; got a $(nameof(typeof(geom)))"
    )
)

# ─── The response of an anisotropic particle ─────────────────────────────────

# Tag of the derivatives of the decaying harmonics along the surface normal.
struct _MemSurfaceTag end

# The data `d` of the particle with its response replaced by that of the
# perfectly bonded conductivity K₁ (canonical matrix in the frame of `d`), the
# matrix being of conductivity k₀.
function _mem_anisotropic(d, K₁::AbstractMatrix, k₀, nmax::Int)
    nb = _mem_block(nmax)
    a = _mem_semi_axes(d)
    Λ = cholesky(Symmetric(K₁)).L
    npts = size(d.surface, 2)
    C = Complex{promote_type(eltype(K₁), typeof(k₀))}
    # values and derivatives along 𝐁 x of the incoming y, outgoing F and inner g
    y, ∂y = zeros(ComplexF64, npts, nb), zeros(ComplexF64, npts, nb)
    F, ∂F = zeros(ComplexF64, npts, nb), zeros(ComplexF64, npts, nb)
    g, ∂g = zeros(C, npts, nb), zeros(C, npts, nb)
    buffer = Vector{Complex{ForwardDiff.Dual{_MemSurfaceTag, Float64, 3}}}(undef, nb)
    for col in 1:npts
        x = SVector{3}(view(d.surface, :, col))
        n = x ./ SVector{3}(a .^ 2)
        u = Λ \ x
        yx, gu = regular_solid_harmonics(x, nmax), regular_solid_harmonics(u, nmax)
        _mem_irregular!(buffer, _mem_seeded(x), d, nmax)
        for k in 1:nmax, l in (-k):k
            i = _mem_index(k, l)
            y[col, i] = yx[k, l]
            ∂y[col, i] = transpose(n) * regular_vector_harmonic(1, k, l, yx)
            g[col, i] = gu[k, l]
            ∂g[col, i] = transpose(n) * (Λ * regular_vector_harmonic(1, k, l, gu))
            F[col, i], ∂F[col, i] = _mem_value_and_slope(buffer[i], n)
        end
    end
    Π = d.Π
    Πy = Π * y
    lhs = [Π * F -(Π * g); k₀ .* (Π * ∂F) -(Π * ∂g)]
    rhs = -[Πy; k₀ .* (Π * ∂y)]
    Z = (lhs \ rhs)[1:nb, :] / Πy
    out = _mem_convert(d, promote_type(eltype(d.Z), eltype(Z)))
    out.Z .= Z
    return out
end

# A point seeded with the three unit directions.
_mem_seeded(x::SVector{3, Float64}) = SVector{3}(
    ntuple(i -> ForwardDiff.Dual{_MemSurfaceTag}(x[i], ntuple(j -> Float64(i == j), Val(3))), Val(3))
)

# The value of a complex dual and its derivative along n.
function _mem_value_and_slope(z::Complex{<:ForwardDiff.Dual}, n)
    re, ip = real(z), imag(z)
    slope = sum(n[j] * complex(ForwardDiff.partials(re, j), ForwardDiff.partials(ip, j)) for j in 1:3)
    return complex(ForwardDiff.value(re), ForwardDiff.value(ip)), slope
end

# The semi-axes of a particle along the axes of its frame.
_mem_semi_axes(d::_MemSpheroid) = (ρ = real(d.c * sqrt(d.ξ₀^2 - 1)); SVector(ρ, ρ, real(d.c * d.ξ₀)))
_mem_semi_axes(d::_MemEllipsoid) = SVector{3}(d.g.a)

# A perfectly bonded spheroid of anisotropic conductivity: its isotropic
# carrier, which gives the harmonics and their projection, and its conductivity.
struct _MemAnisotropicSpheroid{S, M}
    carrier::S
    K::M
end

_mem_spheroid_data(s::_MemAnisotropicSpheroid, K₀, nmax::Int) =
    _mem_anisotropic(_mem_spheroid_data(s.carrier, K₀, nmax), s.K, LayeredSpheres._iso_scalar(K₀), nmax)

# ─── Spheres of anisotropic conductivity ─────────────────────────────────────

# A sphere of radius a in the generic solver of shaped particles: its
# harmonics are the solid ones, y_n^m inside and Y_n^m outside, which the
# spherical translation carries exactly at any distance,
#     Y_t^s(r + R) = Σ_{n,m} (−1)^{n+m} Y_{t+n}^{s−m}(R) y_n^m(r) .
struct _MemSphere{C, R}
    a::R
    scales::Vector{R}
    Z::Matrix{C}
    Π::Matrix{C}
    surface::Matrix{R}
    dipole::Matrix{C}
    gradient::Matrix{C}
end

_mem_focal(::_MemSphere) = 0.0
_mem_far_degree(::_MemSphere, nmax::Int) = 2 * nmax
_mem_semi_axes(d::_MemSphere) = SVector(d.a, d.a, d.a)
_mem_convert(d::_MemSphere, ::Type{C}) where {C} = _MemSphere{C, real(C)}(
    d.a, d.scales, d.Z, d.Π, d.surface, d.dipole, d.gradient
)

function _mem_irregular!(F::AbstractVector, x, ::_MemSphere, nmax::Int)
    Y = irregular_solid_harmonics(x, nmax)
    for n in 1:nmax, m in (-n):n
        F[_mem_index(n, m)] = Y[n, m]
    end
    return F
end

function _mem_eta_far(q::_MemSphere, p::_MemSphere, Y, nmax::Int)
    η = zeros(eltype(q.Z), _mem_block(nmax), _mem_block(nmax))
    for n in 1:nmax, m in (-n):n, t in 1:nmax, s in (-t):t
        η[_mem_index(n, m), _mem_index(t, s)] = (isodd(n + m) ? -1 : 1) * Y[t + n, s - m]
    end
    return η
end

# The data of a sphere of radius a with a zero response: Gauss–Legendre in
# cos θ and the trapezoidal rule in φ, and the projection Π on the y_n^m by
# least squares, exact since their traces are orthogonal; the dipole and
# gradient maps of the spheres page, d = (b₁¹ − b₁⁻¹, i (b₁¹ + b₁⁻¹), b₁⁰) and
# g·x = g₃ y₁⁰ + (g₁ − i g₂) y₁¹ − (g₁ + i g₂) y₁⁻¹.
function _mem_sphere_data(a::Float64, nmax::Int)
    nb = _mem_block(nmax)
    nθ = 2 * nmax + 24
    nφ = 2 * nθ
    xg, wg = LayeredSpheroids.QuadGK.gauss(nθ)
    surface = zeros(3, nθ * nφ)
    w = zeros(nθ * nφ)
    y = zeros(ComplexF64, nθ * nφ, nb)
    for (j, (p, wp)) in enumerate(zip(xg, wg)), i in 1:nφ
        φ = 2π * (i - 1) / nφ
        col = (j - 1) * nφ + i
        surface[:, col] .= a .* (sqrt(1 - p^2) * cos(φ), sqrt(1 - p^2) * sin(φ), p)
        w[col] = wp * 2π / nφ
        tab = regular_solid_harmonics(view(surface, :, col), nmax)
        for n in 1:nmax, m in (-n):n
            y[col, _mem_index(n, m)] = tab[n, m]
        end
    end
    yw = y' * Diagonal(w)
    Π = (yw * y) \ yw
    scales = [a^n / factorial(n + abs(m)) for n in 1:nmax for m in (-n):n]
    D = zeros(ComplexF64, 3, nb)
    D[:, _mem_index(1, 1)] .= (1, im, 0)
    D[:, _mem_index(1, -1)] .= (-1, im, 0)
    D[3, _mem_index(1, 0)] = 1
    G = zeros(ComplexF64, nb, 3)
    G[_mem_index(1, 1), :] .= (1, -im, 0)
    G[_mem_index(1, -1), :] .= (-1, -im, 0)
    G[_mem_index(1, 0), 3] = 1
    return _MemSphere{ComplexF64, Float64}(a, scales, zeros(ComplexF64, nb, nb), Π, surface, D, G)
end

# Whether a particle is a plain ball of anisotropic property; a layered sphere
# carries its own isotropic layers.
_mem_anisotropic_ball(asm, nm::Symbol, ::LayeredSpheres.LayeredSphere, prop::Symbol) = false
_mem_anisotropic_ball(asm, nm::Symbol, geom, prop::Symbol) = !(particle_property(asm, nm, prop) isa TensND.TensISO)

# Spheres among which one at least is anisotropic, all plain balls: the generic
# solver, with the response of each by `_mem_anisotropic` (exact for the
# isotropic ones); a layered sphere among them is refused.
function _mem_anisotropic_spheres(asm::ParticleAssembly, K₀, prop::Symbol, names, geometries, nmax::Int)
    k₀ = LayeredSpheres._iso_scalar(K₀)
    for (nm, geom) in zip(names, geometries)
        geom isa Elasticity.Ellipsoid{3} || throw(
            ArgumentError(
                "MultipoleExpansion: the particle :$(nm), a $(nameof(typeof(geom))), cannot be " *
                    "solved with spheres of anisotropic conductivity, which are plain balls"
            )
        )
    end
    raw = map(zip(names, geometries)) do (nm, geom)
        a = first(geom.semi_axes)
        a isa Float64 || throw(
            ArgumentError(
                "MultipoleExpansion: the radius of the sphere :$(nm) of anisotropic conductivity is " *
                    "a plain floating-point number, got $(typeof(first(geom.semi_axes)))"
            )
        )
        K = Array(TensND.canonical_components(particle_property(asm, nm, prop)))
        _mem_anisotropic(_mem_sphere_data(a, nmax), K, k₀, nmax)
    end
    return _mem_shaped_solve(asm, names, raw, nmax)
end
