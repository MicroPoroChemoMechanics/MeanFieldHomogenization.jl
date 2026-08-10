# =============================================================================
#  assembly.jl — the `ParticleAssembly` cell: a matrix holding individually
#  located inclusions.
#
#  Every other cell in the package describes a microstructure *statistically*:
#  an `RVE` is a bag of phases with volume fractions and no positions, a
#  `Laminate` is a stacking order with no in-plane geometry.  Both are all a
#  one-site scheme can use.
#
#  The two N-body models added here need strictly more: the field induced in
#  one inclusion by another depends on the vector joining their centers, so the
#  cell has to carry positions.  It is a genuinely different kind of cell, not
#  a richer `RVE` — hence a new type, exactly as `Laminate` is a new type
#  rather than an `RVE` with a stacking flag.
#
#  Volume fractions are *derived* here, not stored: `f_a = |Ω_a| / |Ω|`, with
#  |Ω| given by the boundary treatment.  Storing them would let the geometry
#  and the fractions disagree, which is precisely the error an assembly is
#  supposed to make impossible.
# =============================================================================

# ─── Boundary treatments ─────────────────────────────────────────────────────

"""
    AbstractAssemblyBoundary

How a [`ParticleAssembly`](@ref) closes its far field. This is the one place
where the two N-body models of the package genuinely differ: they solve the
same linear system and differ in what surrounds the cell.
"""
abstract type AbstractAssemblyBoundary end

"""
    MixedBC(shape) <: AbstractAssemblyBoundary

Mixed boundary conditions of [Brisard et al. 2014](@cite brisard2014): the
statistical volume element `shape` — which **must be an ellipsoid**, since the
derivation leans on Eshelby's theorem for the domain itself — is embedded in an
infinite medium of the matrix stiffness, subject to a uniform strain at
infinity plus a uniform surface traction chosen so that the loading parameter
coincides with the macroscopic strain.

Its practical virtue is that no periodization is needed: the far field is
closed by a single correction term involving the Hill tensor of the SVE
domain, ``\\mathbb{P}_\\Omega``, instead of by a lattice sum that is only
conditionally convergent.

`MixedBC(radius)` is shorthand for a spherical (3D) SVE; pass an `Ellipsoid`
for anything else.
"""
struct MixedBC{S} <: AbstractAssemblyBoundary
    shape::S
end

MixedBC(radius::Real) = MixedBC(Elasticity.Ellipsoid(radius))

"""
    PeriodicBox(period; cutoff = 3 * period) <: AbstractAssemblyBoundary

Periodic elementary representative volume of
[Molinari & El Mouden 1996](@cite molinari1996): the cubic cell of side
`period` tiles space, each particle carrying a family of periodic images, and
the interaction sums are truncated to the images lying within `cutoff` of the
receiver.

The cutoff is a **sphere**, not a box, and that is not cosmetic: Molinari &
El Mouden's Appendix B proves convergence from the vanishing of the interaction
kernel integrated over the exterior of a sphere. Their own convergence study
finds the estimate stable beyond `cutoff ≈ 2 · period`; the default of three
periods is comfortably inside that plateau.

`cutoff = 0` reduces every cluster to its own receiver, which is the
degenerate case in which the cluster model collapses exactly onto Mori-Tanaka
(their Appendix C) — a useful thing to be able to ask for.
"""
struct PeriodicBox{T} <: AbstractAssemblyBoundary
    period::T
    cutoff::T
end

function PeriodicBox(period::Real; cutoff::Union{Nothing, Real} = nothing)
    period > 0 || throw(ArgumentError("PeriodicBox: the period must be positive"))
    c = cutoff === nothing ? 3 * period : cutoff
    c ≥ 0 || throw(ArgumentError("PeriodicBox: the cutoff must be non-negative"))
    p, cc = promote(period, c)
    return PeriodicBox(p, cc)
end

# ─── Particle ────────────────────────────────────────────────────────────────

"""
    Particle(center, geometry, properties)

One individually located inclusion of a [`ParticleAssembly`](@ref): where it
is, what shape it has, and what it is made of.

Unlike a [`Phase`](@ref MeanFieldHomogenization.Schemes.Phase), a particle is a
single object rather than a population — it carries no volume fraction, since
the fraction follows from its volume and the size of the cell.

A property value may be a [`Homogenized`](@ref) cell + scheme, resolved lazily
at `homogenize` time exactly as in an `RVE`.
"""
mutable struct Particle
    center::AbstractVector
    geometry::MFH_Core.AbstractInclusion
    properties::Dict{Symbol, Any}
end

Particle(center, geometry, properties::AbstractDict) =
    Particle(collect(center), geometry, Dict{Symbol, Any}(properties...))

# ─── The cell ────────────────────────────────────────────────────────────────

"""
    ParticleAssembly{T<:Number, B<:AbstractAssemblyBoundary} <: AbstractHomogenizationCell

A matrix holding individually located inclusions — the cell the N-body schemes
[`EquivalentInclusion`](@ref) and [`ClusterModel`](@ref) act on.

Fields:

- `matrix_name::Symbol` and `matrix_properties::Dict{Symbol,Any}` — the
  reference phase, which is also the reference medium of both models;
- `particle_names::Vector{Symbol}` — particles in insertion order;
- `particles::Dict{Symbol,Particle}` — position, geometry and moduli of each;
- `boundary::B` — [`MixedBC`](@ref) or [`PeriodicBox`](@ref);
- `families::Dict{Symbol,Int}` — family label per particle. Particles sharing a
  label are constrained to carry the *same* polarization or localization
  tensor, which is how a lattice with a small motif is described without
  repeating its equations. Defaults to one family per particle.

Construction is two-step, mirroring `RVE` and `Laminate`:

```julia
asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 3.0))
add_matrix!(asm, Dict(:C => C_m))
add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3), Dict(:C => C_i))
homogenize(asm, ClusterModel(), :C)
```

See also [`add_particle!`](@ref), [`particle_volume_fraction`](@ref),
[`assembly_volume`](@ref), [`validate_assembly`](@ref).
"""
mutable struct ParticleAssembly{T <: Number, B <: AbstractAssemblyBoundary} <:
    MFH_Core.AbstractHomogenizationCell
    matrix_name::Symbol
    matrix_properties::Dict{Symbol, Any}
    particle_names::Vector{Symbol}
    particles::Dict{Symbol, Particle}
    boundary::B
    families::Dict{Symbol, Int}
end

"""
    ParticleAssembly(matrix_name = :matrix; boundary, T = Float64)

Construct an empty assembly. The matrix properties are set next with
[`add_matrix!`](@ref) and particles are added with [`add_particle!`](@ref).

`T` declares the element-type floor, as in `RVE` and `Laminate`: it is a floor
and not a constraint, so a `Dual`-valued coordinate or modulus lives happily in
a plain `ParticleAssembly()`.
"""
function ParticleAssembly(
        matrix_name::Symbol = :matrix;
        boundary::AbstractAssemblyBoundary,
        T::Type{<:Number} = Float64
    )
    return ParticleAssembly{T, typeof(boundary)}(
        matrix_name, Dict{Symbol, Any}(), Symbol[],
        Dict{Symbol, Particle}(), boundary, Dict{Symbol, Int}()
    )
end

"""
    add_matrix!(asm::ParticleAssembly, properties) -> asm

Set the matrix properties of an assembly. The matrix is also the reference
medium of both N-body models — neither uses a reference distinct from the
matrix, following the papers they come from.
"""
function add_matrix!(asm::ParticleAssembly, properties::AbstractDict)
    asm.matrix_properties = Dict{Symbol, Any}(properties...)
    return asm
end

"""
    add_particle!(asm, name, center, geometry, properties; family = nothing) -> asm

Add one located inclusion to the assembly.

`family` groups particles that must share the same unknown. Leaving it unset
gives the particle a family of its own, which is what a random assembly wants;
passing an explicit label is how a periodic motif with symmetric particles is
declared without duplicating equations.
"""
function add_particle!(
        asm::ParticleAssembly, name::Symbol, center, geometry,
        properties::AbstractDict; family::Union{Nothing, Integer} = nothing
    )
    haskey(asm.particles, name) &&
        throw(ArgumentError("add_particle!: particle :$(name) already exists"))
    name === asm.matrix_name &&
        throw(ArgumentError("add_particle!: :$(name) is the matrix name"))
    d = MFH_Core.dimension(geometry)
    length(center) == d || throw(
        ArgumentError(
            "add_particle!: particle :$(name) has a $(length(center))-component " *
                "center but a $(d)-dimensional geometry"
        )
    )
    push!(asm.particle_names, name)
    asm.particles[name] = Particle(center, geometry, properties)
    asm.families[name] = family === nothing ? length(asm.particle_names) : Int(family)
    return asm
end

# ─── Accessors ───────────────────────────────────────────────────────────────

"""
    particle_names(asm) -> Vector{Symbol}

Particle names in insertion order.
"""
particle_names(asm::ParticleAssembly) = asm.particle_names

"""
    particle(asm, name) -> Particle

The stored [`Particle`](@ref) of the given name.
"""
particle(asm::ParticleAssembly, name::Symbol) = asm.particles[name]

"""
    particle_center(asm, name) -> AbstractVector

Center of a particle.
"""
particle_center(asm::ParticleAssembly, name::Symbol) = asm.particles[name].center

"""
    particle_geometry(asm, name) -> AbstractInclusion

Geometry of a particle.
"""
particle_geometry(asm::ParticleAssembly, name::Symbol) = asm.particles[name].geometry

"""
    particle_property(asm, name, key) -> value

Property `key` of particle `name`, resolving a [`Homogenized`](@ref) value if
one is stored (declarative multiscale nesting). Use
[`cell_container_property`](@ref) for the raw stored value.
"""
particle_property(asm::ParticleAssembly, name::Symbol, key::Symbol) =
    MFH_Core.resolve_property(_fetch_property(asm, name, key), key)

"""
    matrix_property(asm, key) -> value

Property `key` of the matrix, resolving a [`Homogenized`](@ref) value if one is
stored. This is also the reference medium of both N-body models.
"""
matrix_property(asm::ParticleAssembly, key::Symbol) =
    MFH_Core.resolve_property(_fetch_property(asm, asm.matrix_name, key), key)

function _fetch_property(asm::ParticleAssembly, name::Symbol, key::Symbol)
    d = name === asm.matrix_name ? asm.matrix_properties : asm.particles[name].properties
    haskey(d, key) || throw(
        KeyError(
            "ParticleAssembly: member :$(name) has no property :$(key) " *
                "(available: $(collect(keys(d))))"
        )
    )
    return d[key]
end

"""
    particle_family(asm, name) -> Int

Family label of a particle. Particles sharing a label share their unknown.
"""
particle_family(asm::ParticleAssembly, name::Symbol) = asm.families[name]

"""
    family_labels(asm) -> Vector{Int}

Distinct family labels, in order of first appearance.
"""
function family_labels(asm::ParticleAssembly)
    seen = Int[]
    for nm in asm.particle_names
        f = asm.families[nm]
        f in seen || push!(seen, f)
    end
    return seen
end

"""
    assembly_volume(asm) -> Real

Measure of the cell: the volume of the SVE for [`MixedBC`](@ref), `Lᵈ` for a
[`PeriodicBox`](@ref). It is what turns particle volumes into volume
fractions.
"""
assembly_volume(asm::ParticleAssembly) = _boundary_volume(asm.boundary, asm)

_boundary_volume(b::MixedBC, ::ParticleAssembly) = Interactions._inclusion_volume(b.shape)

function _boundary_volume(b::PeriodicBox, asm::ParticleAssembly)
    d = _assembly_dimension(asm)
    return b.period^d
end

"""
    particle_volume(asm, name) -> Real

Volume (3D) or area (2D) of one particle.
"""
particle_volume(asm::ParticleAssembly, name::Symbol) =
    Interactions._inclusion_volume(asm.particles[name].geometry)

"""
    particle_volume_fraction(asm, name) -> Real

Volume fraction `f_a = |Ω_a| / |Ω|` of one particle. Derived from the geometry
and the boundary, never stored — so it cannot disagree with the microstructure
it describes.
"""
particle_volume_fraction(asm::ParticleAssembly, name::Symbol) =
    particle_volume(asm, name) / assembly_volume(asm)

"""
    inclusion_volume_fraction(asm) -> Real

Total volume fraction of the particles, `Σ_a f_a`.
"""
inclusion_volume_fraction(asm::ParticleAssembly) =
    sum(particle_volume_fraction(asm, nm) for nm in asm.particle_names; init = 0.0)

"""
    matrix_volume_fraction(asm) -> Real

Volume fraction of the matrix, `1 - Σ_a f_a`.
"""
matrix_volume_fraction(asm::ParticleAssembly) = 1 - inclusion_volume_fraction(asm)

function _assembly_dimension(asm::ParticleAssembly)
    isempty(asm.particle_names) && return 3
    return MFH_Core.dimension(asm.particles[asm.particle_names[1]].geometry)
end

# ─── Validation ──────────────────────────────────────────────────────────────

"""
    validate_assembly(asm) -> asm

Check that an assembly is solvable: a matrix is present, at least one particle
is present, all particles share the same dimension, the total volume fraction
is below one, and no two particles overlap.

The overlap check is not pedantry — the interaction kernel is only defined
for disjoint regions, and an overlap would otherwise surface much later as a
`DomainError` from deep inside a lattice sum. Overlap is tested between the
bounding spheres of the particles, which is exact for spherical particles and
conservative otherwise.
"""
function validate_assembly(asm::ParticleAssembly)
    isempty(asm.matrix_properties) &&
        throw(ArgumentError("ParticleAssembly: no matrix; call `add_matrix!` first"))
    isempty(asm.particle_names) &&
        throw(ArgumentError("ParticleAssembly: no particle; call `add_particle!` first"))
    d = _assembly_dimension(asm)
    for nm in asm.particle_names
        MFH_Core.dimension(asm.particles[nm].geometry) == d || throw(
            ArgumentError(
                "ParticleAssembly: particle :$(nm) has dimension " *
                    "$(MFH_Core.dimension(asm.particles[nm].geometry)) in a $(d)-D assembly"
            )
        )
    end
    f = inclusion_volume_fraction(asm)
    f < 1 || throw(
        ArgumentError(
            "ParticleAssembly: the particles fill the cell (total volume " *
                "fraction $(f) ≥ 1); check the geometries against the boundary size"
        )
    )
    _check_no_overlap(asm)
    return asm
end

function _check_no_overlap(asm::ParticleAssembly)
    names = asm.particle_names
    for i in eachindex(names), j in (i + 1):lastindex(names)
        na, nb = names[i], names[j]
        ra = _bounding_radius(asm.particles[na].geometry)
        rb = _bounding_radius(asm.particles[nb].geometry)
        ca, cb = asm.particles[na].center, asm.particles[nb].center
        δ = sqrt(sum((ca[k] - cb[k])^2 for k in eachindex(ca)))
        δ > ra + rb || throw(
            ArgumentError(
                "ParticleAssembly: particles :$(na) and :$(nb) overlap " *
                    "(center distance $(δ) ≤ $(ra) + $(rb))"
            )
        )
    end
    return nothing
end

_bounding_radius(ell::Elasticity.Ellipsoid) = maximum(ell.semi_axes)
_bounding_radius(incl::MFH_Core.AbstractInclusion) = throw(
    ArgumentError(
        "ParticleAssembly: no bounding radius is defined for a " *
            "$(nameof(typeof(incl))); the overlap check needs one."
    )
)

# ─── The `AbstractHomogenizationCell` contract ───────────────────────────────

MFH_Core.validate_cell(asm::ParticleAssembly) = validate_assembly(asm)

MFH_Core.cell_member_names(asm::ParticleAssembly) =
    Symbol[asm.matrix_name; asm.particle_names...]

MFH_Core.cell_container_property(asm::ParticleAssembly, name::Symbol, key::Symbol) =
    _fetch_property(asm, name, key)

function MFH_Core.cell_set_property(
        asm::ParticleAssembly{T, B}, name::Symbol, key::Symbol, value
    ) where {T, B}
    new = ParticleAssembly{T, B}(
        asm.matrix_name, copy(asm.matrix_properties), copy(asm.particle_names),
        Dict(
            k => Particle(v.center, v.geometry, copy(v.properties))
                for (k, v) in asm.particles
        ),
        asm.boundary, copy(asm.families)
    )
    if name === asm.matrix_name
        new.matrix_properties[key] = value
    else
        new.particles[name].properties[key] = value
    end
    return new
end

function Base.show(io::IO, asm::ParticleAssembly)
    n = length(asm.particle_names)
    print(
        io, "ParticleAssembly(", n, " particle", n == 1 ? "" : "s",
        ", ", nameof(typeof(asm.boundary))
    )
    isempty(asm.particle_names) ||
        print(io, ", f = ", round(inclusion_volume_fraction(asm); digits = 4))
    return print(io, ")")
end
