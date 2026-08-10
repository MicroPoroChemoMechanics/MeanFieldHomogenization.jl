# =============================================================================
#  parameters.jl — parameter lenses for a `ParticleAssembly`.
#
#  The differentiation entry points (`derivative`, `gradient`, `jacobian`,
#  `sensitivity`) are typed on `AbstractHomogenizationCell` and driven by
#  `AbstractParameter` lenses, so an assembly becomes differentiable simply by
#  answering `get_param` / `set_param` on the quantities that describe it.
#
#  `PropertyParameter` already works on any cell through the cell contract, so
#  moduli need nothing here.  What is specific to an assembly is its geometry:
#  where a particle is, and how big it is — the two quantities no other cell of
#  the package has.
#
#  Both lenses rebuild the assembly rather than mutating it, exactly as the
#  `RVE` lenses do: a lens has to be usable inside an AD closure, where the
#  cell it is handed must not be modified in place.
# =============================================================================

"""
    CenterParameter(name::Symbol, component::Int) <: AbstractParameter

Lens onto one Cartesian component of the center of particle `name` — the
microstructural degree of freedom that no other cell of the package has.

```julia
derivative(asm, ClusterModel(), center_param(:p2, 3); indexer = C -> get_array(C)[1,1,1,1])
```
"""
struct CenterParameter <: AbstractParameter
    name::Symbol
    component::Int
end

"""
    center_param(name, component) -> CenterParameter

Convenience constructor for [`CenterParameter`](@ref).
"""
center_param(name::Symbol, component::Integer) = CenterParameter(name, Int(component))

"""
    RadiusParameter(name::Symbol, axis::Int) <: AbstractParameter

Lens onto one semi-axis of the geometry of particle `name`. For a sphere every
axis is the radius, and setting one sets all of them — otherwise the geometry
would silently stop being a sphere and change its shape class mid-derivative.
"""
struct RadiusParameter <: AbstractParameter
    name::Symbol
    axis::Int
end

"""
    radius_param(name, axis = 1) -> RadiusParameter

Convenience constructor for [`RadiusParameter`](@ref).
"""
radius_param(name::Symbol, axis::Integer = 1) = RadiusParameter(name, Int(axis))

# ─── PropertyParameter: moduli, at any member of the assembly ───────────────
#
# The RVE lens is typed on `RVE`, so an assembly needs its own pair. It is the
# one lens that matters for multiscale work: it is how a modulus at an inner
# scale is reached from an outer one through `nested`.

function get_param(asm::ParticleAssembly, p::Schemes.PropertyParameter)
    t = p.phase === asm.matrix_name ? matrix_property(asm, p.property) :
        particle_property(asm, p.phase, p.property)
    return TensND.get_data(t)[Schemes._resolve_selector(t, p.selector)]
end

function set_param(asm::ParticleAssembly, p::Schemes.PropertyParameter, value)
    d = p.phase === asm.matrix_name ? asm.matrix_properties :
        (
            haskey(asm.particles, p.phase) ||
            throw(ArgumentError("no member named :$(p.phase) in the assembly"));
            asm.particles[p.phase].properties
        )
    haskey(d, p.property) || throw(
        ArgumentError(":$(p.phase) does not carry property :$(p.property)")
    )
    old = d[p.property]
    new = Schemes._replace_data_at(old, Schemes._resolve_selector(old, p.selector), value)
    return MFH_Core.cell_set_property(asm, p.phase, p.property, new)
end

# ─── Lenses that do not apply to an assembly ────────────────────────────────
#
# An assembly has no stored amounts (volume fractions are derived from the
# geometry) and no distribution shape. Say so, and name the lens that does the
# job, rather than letting the call fall through to a bare `MethodError`.

for (P, why) in (
        (
            :AmountParameter,
            "an assembly stores no amount: volume fractions are derived from the " *
                "particle geometry and the cell size, so vary a radius " *
                "(`radius_param`) or the boundary instead",
        ),
        (
            :GeometryParameter,
            "use `radius_param(name, axis)` to vary a semi-axis of a particle, or " *
                "`center_param(name, component)` to move it",
        ),
        (
            :DistributionShapeParameter,
            "an assembly carries explicit positions rather than a distribution shape",
        ),
    )
    @eval begin
        get_param(::ParticleAssembly, ::Schemes.$P) = throw(
            ArgumentError("ParticleAssembly: " * $why)
        )
        set_param(::ParticleAssembly, ::Schemes.$P, _) = throw(
            ArgumentError("ParticleAssembly: " * $why)
        )
    end
end

# ─── get / set ───────────────────────────────────────────────────────────────

get_param(asm::ParticleAssembly, p::CenterParameter) =
    particle_center(asm, p.name)[p.component]

get_param(asm::ParticleAssembly, p::RadiusParameter) =
    particle_geometry(asm, p.name).semi_axes[p.axis]

function set_param(asm::ParticleAssembly, p::CenterParameter, value)
    return _rebuild_assembly(asm) do name, part
        name === p.name || return part
        c = collect(part.center)
        c = [i == p.component ? value : c[i] for i in eachindex(c)]
        return Particle(c, part.geometry, part.properties)
    end
end

function set_param(asm::ParticleAssembly, p::RadiusParameter, value)
    return _rebuild_assembly(asm) do name, part
        name === p.name || return part
        return Particle(part.center, _resize_geometry(part.geometry, p.axis, value), part.properties)
    end
end

# A sphere stays a sphere: setting "the" radius sets every semi-axis, so the
# shape trait — and hence the closed-form interaction kernel — is preserved
# along the whole derivative.
#
# The rebuild goes through `Schemes._replace_geom_field`, the same helper the
# `RVE` geometry lens uses. That is not just reuse: it is what promotes the
# whole struct (semi-axes *and* basis) to absorb a `ForwardDiff.Dual`, which a
# hand-rolled `Ellipsoid(duals...; basis = float_basis)` does not do.
function _resize_geometry(ell::Elasticity.Ellipsoid{dim}, axis::Int, value) where {dim}
    allequal(ell.semi_axes) || return Schemes._replace_geom_field(
        ell, Val(:semi_axes), axis, value
    )
    geom = ell
    for k in 1:dim
        geom = Schemes._replace_geom_field(geom, Val(:semi_axes), k, value)
    end
    return geom
end

_resize_geometry(incl, ::Int, _) = throw(
    ArgumentError(
        "RadiusParameter: no semi-axis is defined for a $(nameof(typeof(incl)))."
    )
)

"""
    _rebuild_assembly(f, asm) -> ParticleAssembly

Rebuild an assembly with `f(name, particle)` applied to each particle, without
mutating the original — the immutable-update discipline every lens in the
package follows.

The element-type parameter is widened to whatever the rebuilt particles carry,
so a `Dual`-valued coordinate produces a `Dual`-typed assembly rather than
being narrowed back to `Float64`.
"""
function _rebuild_assembly(f, asm::ParticleAssembly{T, B}) where {T, B}
    parts = Dict{Symbol, Particle}(
        nm => f(nm, asm.particles[nm]) for nm in asm.particle_names
    )
    Tnew = promote_type(T, mapreduce(p -> eltype(p.center), promote_type, values(parts)))
    return ParticleAssembly{Tnew, B}(
        asm.matrix_name, copy(asm.matrix_properties), copy(asm.particle_names),
        parts, asm.boundary, copy(asm.families)
    )
end
