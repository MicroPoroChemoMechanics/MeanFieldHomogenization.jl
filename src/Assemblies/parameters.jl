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
function _resize_geometry(ell::Elasticity.Ellipsoid{dim}, axis::Int, value) where {dim}
    a = ell.semi_axes
    allequal(a) && return Elasticity.Ellipsoid(ntuple(_ -> value, dim)...; basis = ell.basis)
    return Elasticity.Ellipsoid(
        ntuple(i -> i == axis ? value : a[i], dim)...; basis = ell.basis
    )
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
