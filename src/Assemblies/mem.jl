# =============================================================================
#  mem.jl — the multipole expansion method of Kushch (kushch2013).
#
#  CONDUCTION.  In the matrix, of conductivity k₀, the temperature is the remote
#  field plus one series of decaying solid harmonics per particle,
#
#       T(x) = H·x + Σ_p Σ_{n′≥1} Σ_{|m′|≤n′} b_n′m′^(p) Y_n′^m′(x − X_p) .
#
#  Seen from the center of particle q, the series of every other particle is a
#  regular field: the translation formula of the solid harmonics of TensND
#  writes it on the regular harmonics y_n^m(x − X_q),
#
#       Y_n′^m′(x − X_p) = Σ_{n,m} (−1)^{n+m} Y_{n′+n}^{m′−m}(X_q − X_p) y_n^m(x − X_q) .
#
#  Particle q thus receives, on the harmonic (n, m), the coefficient
#
#       a_nm^(q) = h_nm + Σ_{p≠q} Σ_{n′,m′} (−1)^{n+m} Y_{n′+n}^{m′−m}(X_q − X_p) b_n′m′^(p) ,
#
#  h_nm being the coefficients of H·x (degree 1 only, since a constant sends
#  nothing back), and answers by its own response ζ_n = `harmonic_response`:
#  with the radial functions rⁿ/(n+m)! and (n−m)!/r^{n+1} of the two families,
#
#       (n−m)! (n+m)! b_nm^(q) = ζ_n^(q) a_nm^(q) .
#
#  That is the linear system. It is solved for the coefficients each particle
#  RECEIVES, a_nm^(q), rather than for those it sends,
#  b_nm = ζ_n a_nm / [(n−m)!(n+m)!]: a particle without contrast (ζ = 0) then
#  keeps a defined received field, which is what its mean gradient needs.
#  The unknowns are scaled as in kushch2013, Eq. (2.52),
#  â_nm^(q) = R_q^n a_nm^(q) / N_nm with N_nm = √((n−m)!(n+m)!) and R_q the radius:
#
#       â_nm^(q) − Σ_{p≠q} Σ_{n′,m′} W_{nm,n′m′}^{qp} ζ̂_n′^(p) â_n′m′^(p) = R_q^n h_nm / N_nm ,
#
#       ζ̂_n = ζ_n / R^{2n+1} ,
#       W_{nm,n′m′}^{qp} = (−1)^{n+m} R_q^n R_p^{n′+1} Y_{n′+n}^{m′−m}(X_q − X_p) / (N_nm N_n′m′) ,
#
#  with entries of order one at every degree.
#
#  EFFECTIVE PROPERTY.  The degree 1 received by particle q is a uniform
#  gradient H^(q) = 𝐀^(q)·H, with a₁₀ = H₃^(q) and a₁₁ = H₁^(q) − i H₂^(q), and
#  the degree 1 it sends is the dipole d^(q)·x/r³ with d^(q) = ζ₁^(q) H^(q). The
#  average flux of a volume V holding the particles is −k₀ H + (4π k₀ / V) Σ_q d^(q)
#  (theory page, section 5), hence the contribution per unit volume
#
#       𝚺 = −(4π k₀ / V) Σ_q ζ₁^(q) 𝐀^(q) ,
#
#  which a finite assembly closes by Maxwell's argument, `_maxwell_closure`,
#  with the Hill tensor of its domain. The mean gradient and the mean flux of
#  particle q are those of the sphere alone under the uniform gradient H^(q):
#  the harmonics of higher degree have a zero average over a ball.
# =============================================================================

# Position of the harmonic (n, m), n ≥ 1, among the unknowns of one particle.
@inline _mem_index(n::Integer, m::Integer) = n * n + n + m
@inline _mem_block(nmax::Integer) = nmax * (nmax + 2)

function MFH_Core.validate_cell(asm::ParticleAssembly, scheme::MultipoleExpansion)
    validate_assembly(asm)
    _assembly_dimension(asm) == 3 || throw(
        ArgumentError(
            "MultipoleExpansion: the expansion in solid harmonics is three-dimensional; " *
                "got a $(_assembly_dimension(asm))-D assembly"
        )
    )
    nmax = _mem_nmax(scheme)
    nmax ≥ 1 || throw(ArgumentError("MultipoleExpansion: `nmax` must be at least 1, got $nmax"))
    return asm
end

_mem_nmax(scheme::MultipoleExpansion) = Int(get(scheme.options, :nmax, 3))

"""
    _evaluate(asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...) -> AbstractTens

Homogenization of property `:p` over a particle assembly by the multipole
expansion method [kushch2013](@cite). The derivation is in the header of the
file and in the theory page [The multipole expansion method](@ref th-mem).
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...
    ) where {p}
    return _mem_evaluate(asm, matrix_property(asm, p), p, _mem_nmax(scheme); kw...)
end

_mem_evaluate(asm::ParticleAssembly, P₀, p::Symbol, nmax::Int; kw...) = _mem_direct(asm, P₀, p, nmax; kw...)
_mem_evaluate(asm::ParticleAssembly, K₀::TensND.TensISO{2, 3}, p::Symbol, nmax::Int; kw...) =
    _mem_direct(asm, K₀, p, nmax; kw...)

# An anisotropic matrix in conduction: solved in the variables where it is
# isotropic (`mem_anisotropic.jl`), the effective conductivity mapped back.
function _mem_evaluate(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, p::Symbol, nmax::Int; kw...)
    L, k̄ = _mem_affine(K₀)
    mapped = _mem_mapped_assembly(asm, p, L, k̄)
    Kv = _mem_direct(mapped, matrix_property(mapped, p), p, nmax; kw...)
    return TensND.Tens(L * Array(TensND.canonical_components(Kv)) * L)
end

function _mem_direct(asm::ParticleAssembly, P₀, p::Symbol, nmax::Int; kw...)
    sol = _mem_solve(asm, P₀, p, nmax)
    Σ = _mem_contribution(asm, P₀, sol)
    return P₀ + Σ ⊙ _mem_remote_field(asm.boundary, P₀, Σ; kw...)
end

# Remote field per unit macroscopic one. A finite assembly is closed as in
# Maxwell's scheme, with the Hill tensor of its domain; in a periodic one the
# average gradient is the remote one.
_mem_remote_field(b::MixedBC, P₀, Σ; kw...) =
    Schemes._maxwell_remote_field(P₀, Σ, Elasticity.hill_tensor(b.shape, P₀; kw...))
_mem_remote_field(::PeriodicBox, P₀, Σ; kw...) = one(P₀)

# ─── The spheres of the assembly ─────────────────────────────────────────────

# A particle as the layered sphere whose harmonic response is asked for: a
# ball becomes its one-layer sphere, a layered sphere is taken as it is.
_mem_sphere(geom::LayeredSpheres.LayeredSphere, asm, name::Symbol, prop::Symbol) = geom

function _mem_sphere(geom::Elasticity.Ellipsoid{3}, asm, name::Symbol, prop::Symbol)
    a = geom.semi_axes
    a[1] == a[2] == a[3] || _mem_not_a_sphere(name, geom)
    P = particle_property(asm, name, prop)
    P isa TensND.TensISO || throw(
        ArgumentError(
            "MultipoleExpansion: particle :$(name) has a $(nameof(typeof(P))) property " *
                ":$(prop); the spheres of the expansion are isotropic (TensISO)"
        )
    )
    return LayeredSpheres.LayeredSphere((a[1],), (P,))
end

_mem_not_a_sphere(name::Symbol, geom) = throw(
    ArgumentError(
        "MultipoleExpansion: particle :$(name) is a $(nameof(typeof(geom))) that is not a " *
            "sphere; the spheres of the scheme are balls (an Ellipsoid of equal semi-axes) and " *
            "LayeredSphere"
    )
)

# The matrix property, checked to be isotropic.
function _mem_isotropic(P₀)
    P₀ isa TensND.TensISO || throw(
        ArgumentError(
            "MultipoleExpansion: the matrix property is a $(nameof(typeof(P₀))); the " *
                "expansion in solid harmonics is written for an isotropic matrix (TensISO)"
        )
    )
    return P₀
end

# ─── Conduction ──────────────────────────────────────────────────────────────

"""
    MultipoleSolution

Solution of the multipole expansion of an assembly in conduction: for each
particle, in the order of `names`, the layered sphere it stands for (`spheres`),
its radius (`radii`), its responses ``\\zeta_n/R^{2n+1}`` (`responses`, one row per
particle), and the scaled amplitudes it receives under the three unit remote
gradients (`received`, one column per gradient).
"""
struct MultipoleSolution{T, C}
    names::Vector{Symbol}
    spheres::Vector{Any}
    radii::Vector{T}
    responses::Matrix{T}
    received::Matrix{C}
    nmax::Int
end

# The fields of the particles are written for spheres (the method of the
# spheroids, which refuses, is in `mem_spheroids.jl`).
_mem_spheres_only(sol::MultipoleSolution, what::String) = sol

function _mem_solve(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, prop::Symbol, nmax::Int)
    _mem_isotropic(K₀)
    names = particle_names(asm)
    geometries = [particle_geometry(asm, nm) for nm in names]
    return _mem_kind_solve(Val(_mem_one_kind(geometries)), asm, K₀, prop, names, geometries, nmax)
end

# The one kind of the particles of an assembly.
function _mem_one_kind(geometries)
    kinds = unique(_mem_kind.(geometries))
    length(kinds) == 1 || throw(
        ArgumentError(
            "MultipoleExpansion: the assembly mixes spheres, spheroids and ellipsoids; it takes " *
                "one kind of particle at a time"
        )
    )
    return only(kinds)
end

# Spheres take the spherical harmonics, spheroids the spheroidal ones and
# ellipsoids of three different semi-axes the ellipsoidal ones; anything else
# goes with the spheres, which refuse it with their own message.
_mem_kind(::LayeredSpheres.LayeredSphere) = :sphere
_mem_kind(::LayeredSpheroids.LayeredSpheroid) = :spheroid
function _mem_kind(geom::Elasticity.Ellipsoid{3})
    n = length(unique(geom.semi_axes))
    return n == 1 ? :sphere : n == 2 ? :spheroid : :ellipsoid
end
_mem_kind(::InterfacedEllipsoid) = :ellipsoid
_mem_kind(geom) = :sphere

function _mem_kind_solve(::Val{:sphere}, asm, K₀, prop, names, geometries, nmax)
    any(_mem_anisotropic_ball(asm, nm, g, prop) for (nm, g) in zip(names, geometries)) ||
        return _mem_sphere_solve(asm, K₀, prop, names, nmax)
    return _mem_anisotropic_spheres(asm, K₀, prop, names, geometries, nmax)
end
_mem_kind_solve(::Val{:spheroid}, asm, K₀, prop, names, geometries, nmax) =
    _mem_spheroid_solve(asm, K₀, names, [_mem_spheroid(g, asm, nm, prop) for (g, nm) in zip(geometries, names)], nmax)
_mem_kind_solve(::Val{:ellipsoid}, asm, K₀, prop, names, geometries, nmax) =
    _mem_ellipsoid_solve(asm, K₀, names, geometries, prop, nmax)

function _mem_sphere_solve(asm::ParticleAssembly, K₀, prop::Symbol, names, nmax::Int)
    spheres = Any[_mem_sphere(particle_geometry(asm, nm), asm, nm, prop) for nm in names]
    # Particles differ in their number types (one radius or one center carrying
    # a derivative, the others not): each array is brought to its common type.
    R = _common_eltype([LayeredSpheres.outer_radius(sphere) for sphere in spheres])
    ζ̂ = _common_eltype(
        [harmonic_response(spheres[q], K₀, n) / R[q]^(2n + 1) for q in eachindex(spheres), n in 1:nmax]
    )
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    X = [SVector{3, TX}(c) for c in centers]
    T = float(promote_type(eltype(R), eltype(ζ̂), TX))
    â = _mem_scalar_system(convert(Vector{T}, R), convert(Matrix{T}, ζ̂), X, asm.boundary, nmax)
    return MultipoleSolution(copy(names), spheres, convert(Vector{T}, R), convert(Matrix{T}, ζ̂), â, nmax)
end

# An array of numbers of different types, in the type that holds them all.
_common_eltype(v::AbstractArray) = convert(AbstractArray{mapreduce(typeof, promote_type, v)}, v)

# Function barrier: radii, responses and centers arrive with their concrete
# element types, whatever the particles were. Returns the scaled amplitudes
# received by every particle, one column per unit remote gradient.
function _mem_scalar_system(R::Vector{T}, ζ̂::Matrix{T}, X::Vector, boundary, nmax::Int) where {T}
    N, n = length(R), _mem_block(nmax)
    _check_system_size(N * n, Complex{T}, N, nmax)
    M = zeros(Complex{T}, N * n, N * n)
    B = zeros(Complex{T}, N * n, 3)
    norms = [solid_harmonic_norm(n, m) for n in 1:nmax for m in (-n):n]
    for q in 1:N
        oq = (q - 1) * n
        for i in 1:n
            M[oq + i, oq + i] = one(T)
        end
        _mem_remote_gradient!(B, oq, R[q])
        for p in 1:N
            Y = _mem_translation(boundary, X[q] - X[p], p == q, 2 * nmax)
            Y === nothing && continue
            _mem_couple!(M, oq, (p - 1) * n, Y, R[q], R[p], view(ζ̂, p, :), norms, nmax)
            _mem_background!(M, oq, (p - 1) * n, boundary, ζ̂[p, 1] * R[q] * R[p]^2)
        end
    end
    return M \ B
end

"""
    _mem_received_gradient(sol, q) -> Matrix

Uniform gradient received by particle `q` per unit remote gradient: the tensor
``\\boldsymbol{A}_I`` such that the degree-1 part of the regular field about
its center is ``(\\boldsymbol{A}_I\\cdot\\underline{H})\\cdot\\underline{x}``.
"""
function _mem_received_gradient(sol::MultipoleSolution{T}, q::Int) where {T}
    oq = (q - 1) * _mem_block(sol.nmax)
    # a₁ₘ = â₁ₘ N₁ₘ / R, with a₁₀ = H₃ and a₁₁ = H₁ − i H₂
    a₀ = view(sol.received, oq + _mem_index(1, 0), :) ./ sol.radii[q]
    a₁ = view(sol.received, oq + _mem_index(1, 1), :) .* (sqrt(T(2)) / sol.radii[q])
    return permutedims(hcat(real.(a₁), -imag.(a₁), real.(a₀)))
end

"""
    _mem_contribution(asm, K₀, sol) -> AbstractTens

Sum ``\\boldsymbol{\\Sigma}`` of the contributions of the particles per unit
volume of the assembly, interactions included: the tensor such that the average
flux is ``-(\\boldsymbol{K}_0 + \\boldsymbol{\\Sigma})\\cdot\\underline{H}`` under the
remote gradient ``\\underline{H}``.
"""
function _mem_contribution(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, sol::MultipoleSolution)
    k₀ = LayeredSpheres._iso_scalar(_mem_isotropic(K₀))
    D = sum(
        sol.responses[q, 1] * sol.radii[q]^3 * _mem_received_gradient(sol, q)
            for q in eachindex(sol.names)
    )
    # symmetric by reciprocity; the rounding of the solve is taken off
    return TensND.Tens((-4π * k₀ / assembly_volume(asm)) .* (D .+ transpose(D)) ./ 2)
end

# Right-hand side of the three unit gradients on the degree-1 rows of one
# particle: H·x = H₃ y₁⁰ + (H₁ − iH₂) y₁¹ − (H₁ + iH₂) y₁⁻¹, scaled by
# R / N₁ₘ with N₁₀ = 1 and N₁,±₁ = √2.
function _mem_remote_gradient!(B::Matrix{Complex{T}}, oq::Int, R) where {T}
    h = R / sqrt(T(2))
    B[oq + _mem_index(1, 1), 1] = h
    B[oq + _mem_index(1, -1), 1] = -h
    B[oq + _mem_index(1, 1), 2] = -im * h
    B[oq + _mem_index(1, -1), 2] = -im * h
    B[oq + _mem_index(1, 0), 3] = R
    return B
end

# Block of the field particle p sends to particle q, from the table `Y` of the
# decaying harmonics of the vector joining p to q and the responses `ζ̂p` of p.
function _mem_couple!(M::Matrix, oq::Int, op::Int, Y, Rq, Rp, ζ̂p, norms, nmax::Int)
    for n in 1:nmax, m in (-n):n
        i = _mem_index(n, m)
        row = Rq^n / norms[i]
        isodd(n + m) && (row = -row)
        for n′ in 1:nmax, m′ in (-n′):n′
            j = _mem_index(n′, m′)
            M[oq + i, op + j] -= row * ζ̂p[n′] * Rp^(n′ + 1) * Y[n′ + n, m′ - m] / norms[j]
        end
    end
    return M
end

# Decaying harmonics of the vector from the sender to the receiver; a particle
# of a finite assembly sends nothing to itself.
_mem_translation(::MixedBC, r, self::Bool, nmax::Int) = self ? nothing : irregular_solid_harmonics(r, nmax)

# In a periodic assembly each particle also receives the harmonics of the
# images, its own included, summed over the lattice.
_mem_translation(b::PeriodicBox, r, self::Bool, nmax::Int) =
    lattice_solid_harmonics(r, box_periods(b), nmax)

# Uniform field of the background that makes a periodic dipole periodic
# (`lattice_solid_harmonics` of TensND): the regular part of S has the Laplacian 4π/V,
# so a dipole of the lattice sends to every particle, on top of the degree-2
# harmonics, the field −(4π/3V) × (its moment), the same in the three
# directions. In the scaled unknowns it is one term on the three degree-1
# entries of the block, `w` being ζ̂₁^(p) R_q R_p².
_mem_background!(M, oq::Int, op::Int, ::MixedBC, w) = M
function _mem_background!(M, oq::Int, op::Int, b::PeriodicBox, w)
    g = 4π / (3 * prod(box_periods(b)))
    for m in -1:1
        M[oq + _mem_index(1, m), op + _mem_index(1, m)] += w * g
    end
    return M
end

# The system is dense: refuse a size the machine cannot hold before allocating
# it, naming what to reduce.
function _check_system_size(n::Int, ::Type{C}, N::Int, nmax::Int; available = Sys.free_memory()) where {C}
    bytes = 2 * n^2 * sizeof(C)     # the matrix and its factorization
    bytes ≤ available / 2 || throw(
        ArgumentError(
            "MultipoleExpansion: $N particles at nmax = $nmax give a dense system of " *
                "$n unknowns ($(round(bytes / 2^30; digits = 1)) GiB), more than half the " *
                "free memory; lower `nmax` or the number of particles"
        )
    )
    return nothing
end

# ─── Concentration tensors and coefficients ──────────────────────────────────

function Schemes._concentration_tensors(
        asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...
    ) where {p}
    P₀ = matrix_property(asm, p)
    _mem_conduction_only(P₀, "the concentration tensors")
    sol = _mem_spheres_only(_mem_solve(asm, P₀, p, _mem_nmax(scheme)), "the concentration tensors")
    Σ = _mem_contribution(asm, P₀, sol)
    𝔼 = _mem_remote_field(asm.boundary, P₀, Σ; kw...)
    P_hom = P₀ + Σ ⊙ 𝔼
    names = sol.names
    weights = Dict(n => particle_volume_fraction(asm, n) for n in names)
    strain = Dict{Symbol, TensND.AbstractTens}()
    stress = Dict{Symbol, TensND.AbstractTens}()
    properties = Dict{Symbol, TensND.AbstractTens}()
    for (q, n) in enumerate(names)
        # the sphere alone, under the uniform gradient it receives
        A, B = MFH_Core.concentration_and_flux_average(sol.spheres[q], P₀, P₀)
        received = TensND.Tens(_mem_received_gradient(sol, q)) ⊙ 𝔼
        strain[n] = A ⊙ received
        stress[n] = B ⊙ received
        _mem_declared_property!(properties, asm, n, p)
    end
    return _with_assembly_matrix(asm, P₀, P_hom, names, strain, stress, weights, properties)
end

# What follows from the field received by each sphere is written in conduction;
# in elasticity the effective stiffness alone is available.
_mem_conduction_only(::TensND.AbstractTens{2, 3}, what::AbstractString) = nothing
_mem_conduction_only(::TensND.AbstractTens, what::AbstractString) = throw(
    ArgumentError("MultipoleExpansion: $what are available in conduction only")
)

# The property a homogeneous particle declares; a layered sphere has no single one.
function _mem_declared_property!(properties, asm::ParticleAssembly, name::Symbol, prop::Symbol)
    haskey(asm.particles[name].properties, prop) &&
        (properties[name] = particle_property(asm, name, prop))
    return properties
end

"""
    MultipoleCoefficients

Coefficients of the series of decaying harmonics of every particle of an
assembly, as [`multipole_coefficients`](@ref) returns them. `c[name, n, m]` is
the coefficient ``b_n^m`` of ``Y_n^m(\\underline{x} - \\underline{X})`` for the
particle `name`, as three values: one per unit remote gradient
``\\underline{e}_1``, ``\\underline{e}_2``, ``\\underline{e}_3``. `c.names` lists
the particles and `c.nmax` is the truncation degree.
"""
struct MultipoleCoefficients{C <: Complex}
    names::Vector{Symbol}
    nmax::Int
    values::Vector{Matrix{C}}
end

function Base.getindex(c::MultipoleCoefficients, name::Symbol, n::Integer, m::Integer)
    q = findfirst(==(name), c.names)
    q === nothing && throw(KeyError(name))
    1 ≤ n ≤ c.nmax && abs(m) ≤ n || throw(BoundsError(c, (name, n, m)))
    return c.values[q][_mem_index(n, m), :]
end

Base.show(io::IO, c::MultipoleCoefficients) =
    print(io, "MultipoleCoefficients(", length(c.names), " particles, nmax = ", c.nmax, ")")

"""
    multipole_coefficients(asm, prop = :K; scheme = MultipoleExpansion()) -> MultipoleCoefficients

Coefficients of the multipole expansion of every particle of `asm`
([`MultipoleExpansion`](@ref)): in the matrix, under the remote gradient
``\\underline{H}``, the field is

```math
T(\\underline{x}) = \\underline{H}\\cdot\\underline{x}
  + \\sum_J \\sum_{n \\ge 1} \\sum_{|m| \\le n} b_{J,n}^m\\,Y_n^m(\\underline{x} - \\underline{X}_J) ,
```

with ``Y_n^m`` the decaying solid harmonics of [`irregular_solid_harmonics`](@ref)
(their sums over the images in a periodic assembly, [`lattice_solid_harmonics`](@ref)).
The result is indexed as `c[name, n, m]`.
"""
function multipole_coefficients(
        asm::ParticleAssembly, prop::Symbol = :K;
        scheme::MultipoleExpansion = MultipoleExpansion()
    )
    MFH_Core.validate_cell(asm, scheme)
    _mem_conduction_only(matrix_property(asm, prop), "the coefficients of the series")
    sol = _mem_spheres_only(_mem_solve(asm, matrix_property(asm, prop), prop, _mem_nmax(scheme)), "the coefficients of the series")
    n = _mem_block(sol.nmax)
    # b_nm = ζ̂_n R^(n+1) â_nm / N_nm
    scale(q) = [
        sol.responses[q, n] * sol.radii[q]^(n + 1) / solid_harmonic_norm(n, m)
            for n in 1:sol.nmax for m in (-n):n
    ]
    values = [
        scale(q) .* sol.received[((q - 1) * n + 1):(q * n), :] for q in eachindex(sol.names)
    ]
    return MultipoleCoefficients(sol.names, sol.nmax, values)
end
