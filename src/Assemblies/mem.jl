# =============================================================================
#  mem.jl — the multipole expansion method of Kushch (kushch2013).
#
#  CONDUCTION.  In the matrix, of conductivity k₀, the temperature is the remote
#  field plus one series of decaying solid harmonics per particle,
#
#       T(x) = G·x + Σ_p Σ_{k≥1} Σ_{|l|≤k} A_kl^(p) Y_k^l(x − X_p) .
#
#  Seen from the center of particle q, the series of every other particle is a
#  regular field: the translation formula of `Harmonics/solid_harmonics.jl`
#  writes it on the regular harmonics y_t^s(x − X_q),
#
#       Y_k^l(x − X_p) = Σ_{t,s} (−1)^{t+s} Y_{k+t}^{l−s}(X_q − X_p) y_t^s(x − X_q) .
#
#  Particle q thus receives, on the harmonic (t, s), the amplitude
#
#       c_ts^(q) + Σ_{p≠q} Σ_{k,l} (−1)^{t+s} Y_{k+t}^{l−s}(X_q − X_p) A_kl^(p) ,
#
#  c_ts being the coefficients of G·x (degree 1 only, since a constant sends
#  nothing back), and answers by its own response ρ_t = `harmonic_response`:
#  with the radial functions rᵗ/(t+s)! and (t−s)!/r^{t+1} of the two families,
#
#       (t−s)! (t+s)! A_ts^(q) = ρ_t^(q) [ c_ts^(q) + Σ_{p≠q} … ] .
#
#  That is the linear system. It is solved for the amplitudes each particle
#  RECEIVES, ĉ_ts^(q) = c_ts^(q) + Σ_{p≠q} …, rather than for those it sends,
#  A_ts = ρ_t ĉ_ts / [(t−s)!(t+s)!]: a particle without contrast (ρ = 0) then
#  keeps a defined received field, which is what its mean gradient needs.
#  The unknowns are scaled as in kushch2013, Eq. (2.52),
#  e_ts^(q) = R_q^t ĉ_ts^(q) / N_ts with N_ts = √((t−s)!(t+s)!) and R_q the radius:
#
#       e_ts^(q) − Σ_{p≠q} Σ_{k,l} W_{ts,kl}^{qp} ρ̂_k^(p) e_kl^(p) = R_q^t c_ts^(q) / N_ts ,
#
#       ρ̂_t = ρ_t / R^{2t+1} ,
#       W_{ts,kl}^{qp} = (−1)^{t+s} R_q^t R_p^{k+1} Y_{k+t}^{l−s}(X_q − X_p) / (N_ts N_kl) ,
#
#  with entries of order one at every degree.
#
#  EFFECTIVE PROPERTY.  The degree 1 received by particle q is a uniform
#  gradient G^(q) = 𝐄^(q)·G, with ĉ_10 = G₃^(q) and ĉ_11 = G₁^(q) − i G₂^(q), and
#  the degree 1 it sends is the dipole d^(q)·x/r³ with d^(q) = ρ₁^(q) G^(q). The
#  average flux of a volume V holding the particles is −k₀ G + (4π k₀ / V) Σ_q d^(q)
#  (theory page, section 5), hence the contribution per unit volume
#
#       𝚺 = −(4π k₀ / V) Σ_q ρ₁^(q) 𝐄^(q) ,
#
#  which a finite assembly closes by Maxwell's argument, `_maxwell_closure`,
#  with the Hill tensor of its domain. The mean gradient and the mean flux of
#  particle q are those of the sphere alone under the uniform gradient G^(q):
#  the harmonics of higher degree have a zero average over a ball.
# =============================================================================

# Position of the harmonic (t, s), t ≥ 1, among the unknowns of one particle.
@inline _mem_index(t::Integer, s::Integer) = t * t + t + s
@inline _mem_block(tmax::Integer) = tmax * (tmax + 2)

function MFH_Core.validate_cell(asm::ParticleAssembly, scheme::MultipoleExpansion)
    validate_assembly(asm)
    _assembly_dimension(asm) == 3 || throw(
        ArgumentError(
            "MultipoleExpansion: the expansion in solid harmonics is three-dimensional; " *
                "got a $(_assembly_dimension(asm))-D assembly"
        )
    )
    tmax = _mem_tmax(scheme)
    tmax ≥ 1 || throw(ArgumentError("MultipoleExpansion: `tmax` must be at least 1, got $tmax"))
    return asm
end

_mem_tmax(scheme::MultipoleExpansion) = Int(get(scheme.options, :tmax, 3))

"""
    _evaluate(asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...) -> AbstractTens

Homogenization of property `:p` over a particle assembly by the multipole
expansion method [kushch2013](@cite). The derivation is in the header of the
file and in the theory page [The multipole expansion method](@ref th-mem).
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::MultipoleExpansion, ::Val{p}; kw...
    ) where {p}
    P₀ = matrix_property(asm, p)
    sol = _mem_solve(asm, P₀, p, _mem_tmax(scheme))
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
            "sphere; the scheme takes balls (an Ellipsoid of equal semi-axes) and LayeredSphere"
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
its radius (`radii`), its responses ``b_t/R^{2t+1}`` (`responses`, one row per
particle), and the scaled amplitudes it receives under the three unit remote
gradients (`received`, one column per gradient).
"""
struct MultipoleSolution{T, C}
    names::Vector{Symbol}
    spheres::Vector{Any}
    radii::Vector{T}
    responses::Matrix{T}
    received::Matrix{C}
    tmax::Int
end

function _mem_solve(asm::ParticleAssembly, K₀::TensND.AbstractTens{2, 3}, prop::Symbol, tmax::Int)
    _mem_isotropic(K₀)
    names = particle_names(asm)
    spheres = Any[_mem_sphere(particle_geometry(asm, nm), asm, nm, prop) for nm in names]
    # Particles differ in their number types (one radius or one center carrying
    # a derivative, the others not): each array is brought to its common type.
    R = _common_eltype([LayeredSpheres.outer_radius(s) for s in spheres])
    ρ̂ = _common_eltype(
        [harmonic_response(spheres[q], K₀, t) / R[q]^(2t + 1) for q in eachindex(spheres), t in 1:tmax]
    )
    centers = [particle_center(asm, nm) for nm in names]
    TX = mapreduce(eltype, promote_type, centers)
    X = [SVector{3, TX}(c) for c in centers]
    T = float(promote_type(eltype(R), eltype(ρ̂), TX))
    e = _mem_scalar_system(convert(Vector{T}, R), convert(Matrix{T}, ρ̂), X, asm.boundary, tmax)
    return MultipoleSolution(copy(names), spheres, convert(Vector{T}, R), convert(Matrix{T}, ρ̂), e, tmax)
end

# An array of numbers of different types, in the type that holds them all.
_common_eltype(v::AbstractArray) = convert(AbstractArray{mapreduce(typeof, promote_type, v)}, v)

# Function barrier: radii, responses and centers arrive with their concrete
# element types, whatever the particles were. Returns the scaled amplitudes
# received by every particle, one column per unit remote gradient.
function _mem_scalar_system(R::Vector{T}, ρ̂::Matrix{T}, X::Vector, boundary, tmax::Int) where {T}
    N, n = length(R), _mem_block(tmax)
    _check_system_size(N * n, Complex{T}, N, tmax)
    M = zeros(Complex{T}, N * n, N * n)
    B = zeros(Complex{T}, N * n, 3)
    norms = [harmonic_norm(t, s) for t in 1:tmax for s in (-t):t]
    for q in 1:N
        oq = (q - 1) * n
        for i in 1:n
            M[oq + i, oq + i] = one(T)
        end
        _mem_remote_gradient!(B, oq, R[q])
        for p in 1:N
            Y = _mem_translation(boundary, X[q] - X[p], p == q, 2 * tmax)
            Y === nothing && continue
            _mem_couple!(M, oq, (p - 1) * n, Y, R[q], R[p], view(ρ̂, p, :), norms, tmax)
            _mem_background!(M, oq, (p - 1) * n, boundary, ρ̂[p, 1] * R[q] * R[p]^2)
        end
    end
    return M \ B
end

"""
    _mem_received_gradient(sol, q) -> Matrix

Uniform gradient received by particle `q` per unit remote gradient: the tensor
``\\boldsymbol{E}^{(q)}`` such that the degree-1 part of the regular field about
its center is ``(\\boldsymbol{E}^{(q)}\\cdot\\underline{G})\\cdot\\underline{x}``.
"""
function _mem_received_gradient(sol::MultipoleSolution{T}, q::Int) where {T}
    oq = (q - 1) * _mem_block(sol.tmax)
    # ĉ₁ₛ = e₁ₛ N₁ₛ / R, with ĉ₁₀ = G₃ and ĉ₁₁ = G₁ − i G₂
    c₀ = view(sol.received, oq + _mem_index(1, 0), :) ./ sol.radii[q]
    c₁ = view(sol.received, oq + _mem_index(1, 1), :) .* (sqrt(T(2)) / sol.radii[q])
    return permutedims(hcat(real.(c₁), -imag.(c₁), real.(c₀)))
end

"""
    _mem_contribution(asm, K₀, sol) -> AbstractTens

Sum ``\\boldsymbol{\\Sigma}`` of the contributions of the particles per unit
volume of the assembly, interactions included: the tensor such that the average
flux is ``-(\\boldsymbol{K}_0 + \\boldsymbol{\\Sigma})\\cdot\\underline{G}`` under the
remote gradient ``\\underline{G}``.
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
# particle: G·x = G₃ y₁⁰ + (G₁ − iG₂) y₁¹ − (G₁ + iG₂) y₁⁻¹, scaled by
# R / N₁ₛ with N₁₀ = 1 and N₁,±₁ = √2.
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
# decaying harmonics of the vector joining p to q and the responses `ρ̂p` of p.
function _mem_couple!(M::Matrix, oq::Int, op::Int, Y, Rq, Rp, ρ̂p, norms, tmax::Int)
    for t in 1:tmax, s in (-t):t
        i = _mem_index(t, s)
        row = Rq^t / norms[i]
        isodd(t + s) && (row = -row)
        for k in 1:tmax, l in (-k):k
            j = _mem_index(k, l)
            M[oq + i, op + j] -= row * ρ̂p[k] * Rp^(k + 1) * Y[k + t, l - s] / norms[j]
        end
    end
    return M
end

# Decaying harmonics of the vector from the sender to the receiver; a particle
# of a finite assembly sends nothing to itself.
_mem_translation(::MixedBC, r, self::Bool, tmax::Int) = self ? nothing : irregular_harmonics(r, tmax)

# In a periodic assembly each particle also receives the harmonics of the
# images, its own included, summed over the lattice.
_mem_translation(b::PeriodicBox, r, self::Bool, tmax::Int) =
    lattice_harmonics(r, box_periods(b), tmax)

# Uniform field of the background that makes a periodic dipole periodic
# (`Harmonics/lattice_sums.jl`): the regular part of S has the Laplacian 4π/V,
# so a dipole of the lattice sends to every particle, on top of the degree-2
# harmonics, the field −(4π/3V) × (its moment), the same in the three
# directions. In the scaled unknowns it is one term on the three degree-1
# entries of the block, `w` being ρ̂₁^(p) R_q R_p².
_mem_background!(M, oq::Int, op::Int, ::MixedBC, w) = M
function _mem_background!(M, oq::Int, op::Int, b::PeriodicBox, w)
    g = 4π / (3 * prod(box_periods(b)))
    for s in -1:1
        M[oq + _mem_index(1, s), op + _mem_index(1, s)] += w * g
    end
    return M
end

# The system is dense: refuse a size the machine cannot hold before allocating
# it, naming what to reduce.
function _check_system_size(n::Int, ::Type{C}, N::Int, tmax::Int; available = Sys.free_memory()) where {C}
    bytes = 2 * n^2 * sizeof(C)     # the matrix and its factorization
    bytes ≤ available / 2 || throw(
        ArgumentError(
            "MultipoleExpansion: $N particles at tmax = $tmax give a dense system of " *
                "$n unknowns ($(round(bytes / 2^30; digits = 1)) GiB), more than half the " *
                "free memory; lower `tmax` or the number of particles"
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
    sol = _mem_solve(asm, P₀, p, _mem_tmax(scheme))
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
assembly, as [`multipole_coefficients`](@ref) returns them. `c[name, t, s]` is
the coefficient ``A_{ts}`` of ``Y_t^s(\\underline{x} - \\underline{X})`` for the
particle `name`, as three values: one per unit remote gradient
``\\underline{e}_1``, ``\\underline{e}_2``, ``\\underline{e}_3``. `c.names` lists
the particles and `c.tmax` is the truncation degree.
"""
struct MultipoleCoefficients{C <: Complex}
    names::Vector{Symbol}
    tmax::Int
    values::Vector{Matrix{C}}
end

function Base.getindex(c::MultipoleCoefficients, name::Symbol, t::Integer, s::Integer)
    q = findfirst(==(name), c.names)
    q === nothing && throw(KeyError(name))
    1 ≤ t ≤ c.tmax && abs(s) ≤ t || throw(BoundsError(c, (name, t, s)))
    return c.values[q][_mem_index(t, s), :]
end

Base.show(io::IO, c::MultipoleCoefficients) =
    print(io, "MultipoleCoefficients(", length(c.names), " particles, tmax = ", c.tmax, ")")

"""
    multipole_coefficients(asm, prop = :K; scheme = MultipoleExpansion()) -> MultipoleCoefficients

Coefficients of the multipole expansion of every particle of `asm`
([`MultipoleExpansion`](@ref)): in the matrix, under the remote gradient
``\\underline{G}``, the field is

```math
T(\\underline{x}) = \\underline{G}\\cdot\\underline{x}
  + \\sum_p \\sum_{t \\ge 1} \\sum_{|s| \\le t} A_{ts}^{(p)}\\,Y_t^s(\\underline{x} - \\underline{X}_p) ,
```

with ``Y_t^s`` the decaying solid harmonics of [`irregular_harmonics`](@ref)
(their sums over the images in a periodic assembly, [`lattice_harmonics`](@ref)).
The result is indexed as `c[name, t, s]`.
"""
function multipole_coefficients(
        asm::ParticleAssembly, prop::Symbol = :K;
        scheme::MultipoleExpansion = MultipoleExpansion()
    )
    MFH_Core.validate_cell(asm, scheme)
    _mem_conduction_only(matrix_property(asm, prop), "the coefficients of the series")
    sol = _mem_solve(asm, matrix_property(asm, prop), prop, _mem_tmax(scheme))
    n = _mem_block(sol.tmax)
    # A_ts = ρ̂_t R^(t+1) e_ts / N_ts
    scale(q) = [
        sol.responses[q, t] * sol.radii[q]^(t + 1) / harmonic_norm(t, s)
            for t in 1:sol.tmax for s in (-t):t
    ]
    values = [
        scale(q) .* sol.received[((q - 1) * n + 1):(q * n), :] for q in eachindex(sol.names)
    ]
    return MultipoleCoefficients(sol.names, sol.tmax, values)
end
