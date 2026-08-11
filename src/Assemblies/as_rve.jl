# =============================================================================
#  as_rve.jl — the bridge from a `ParticleAssembly` to an `RVE`, and hence to
#  every one-site scheme of the package.
#
#  An assembly is strictly richer than an RVE: it knows WHERE each inclusion
#  is, and an RVE does not.  Forgetting the positions is therefore always
#  well defined, and it is the only direction that is — an RVE has no
#  positions to invent, so there is deliberately no `ParticleAssembly(rve)`.
#
#  That single conversion is what makes `homogenize(asm, MoriTanaka(), :C)`
#  work: the N-body schemes read the positions, every other scheme goes
#  through here and reads only the fractions.  It is what turns "what does
#  Mori-Tanaka say about THIS microstructure?" from a hand-rebuilt parallel
#  cell into one call — the comparison `scripts/91` used to write out by hand.
#
#  ONE PHASE PER PARTICLE, deliberately.  Grouping identical particles would
#  need an equality heuristic on geometries and property dictionaries, and buy
#  nothing: every scheme of the package sums over phases linearly, so N
#  identical phases of fraction f/N are *exactly* one phase of fraction f —
#  in Mori-Tanaka, in the self-consistent fixed point, and along a
#  differential path alike.  Keeping one phase per particle is therefore
#  exact, needs no heuristic, and preserves the particle names, so a
#  per-particle localization stays addressable by the name it had.
# =============================================================================

"""
    RVE(asm::ParticleAssembly; matrix_geometry = nothing, distribution_shape = nothing)

Statistical view of an assembly: **forget the positions, keep everything
else**. Returns an [`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) whose
matrix carries the assembly's matrix properties and which has one phase per
particle, each with that particle's geometry, properties and *derived* volume
fraction ``f_a = |\\Omega_a| / |\\Omega|``.

This is the bridge that makes every one-site scheme available on an assembly.
It is applied automatically by [`homogenize`](@ref) — `homogenize(asm,
MoriTanaka(), :C)` just works — and is exported so that the intermediate cell
can be inspected:

```julia
asm = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.3)
homogenize(asm, ClusterModel(), :C)     # sees the positions
homogenize(asm, MoriTanaka(), :C)       # goes through RVE(asm); ignores them
rve = RVE(asm)                          # …and here it is, to look at
```

`matrix_geometry` is the shape the matrix phase is given. An assembly has no
matrix shape of its own, so it defaults to a **ball** (a disk in 2D) — the
neutral, orientation-free choice, and the one the one-site scripts of the
package use. It is only ever read by the schemes that localize the matrix
like any other phase, the self-consistent family; `MoriTanaka`, `Dilute` and
the bounds never look at it.

`distribution_shape` is forwarded to the `RVE` constructor and is what
[`PonteCastanedaWillis`](@ref MeanFieldHomogenization.Schemes.PonteCastanedaWillis)
needs; an assembly carries no such statistical descriptor, so it must be
supplied here if that scheme is wanted.

!!! note "One phase per particle"
    The conversion does not merge identical particles. It does not need to:
    every scheme sums over phases linearly, so `N` identical phases of
    fraction `f/N` give exactly the same effective property as one phase of
    fraction `f`. Keeping them apart avoids an equality heuristic on
    geometries and preserves the particle names.

!!! warning "There is no way back"
    `RVE(asm)` discards the positions, which is the whole point — but it means
    the result can no longer feed [`ClusterModel`](@ref) or
    [`EquivalentInclusion`](@ref). Those need the assembly itself.

See also [`ParticleAssembly`](@ref), [`particle_volume_fraction`](@ref).
"""
function Schemes.RVE(
        asm::ParticleAssembly;
        matrix_geometry = nothing, distribution_shape = nothing
    )
    validate_assembly(asm)
    names = particle_names(asm)
    fracs = [particle_volume_fraction(asm, nm) for nm in names]
    # The element-type floor must absorb a `ForwardDiff.Dual` fraction, which
    # is what a derivative with respect to a radius or the cell size produces.
    # Read it off the fractions rather than off the assembly's own parameter:
    # centers may be plain while radii are dual, and it is the radii that the
    # fractions depend on.
    T = promote_type(Float64, eltype(fracs))
    rve = Schemes.RVE(asm.matrix_name; T = T, distribution_shape = distribution_shape)
    geom_m = matrix_geometry === nothing ? _matrix_ball(asm) : matrix_geometry
    Schemes.add_matrix!(rve, geom_m, copy(asm.matrix_properties))
    for (nm, f) in zip(names, fracs)
        Schemes.add_phase!(
            rve, nm, particle_geometry(asm, nm),
            copy(particle(asm, nm).properties); fraction = f
        )
    end
    return rve
end

# A ball of the assembly's own dimension.  Its radius is irrelevant — the Hill
# tensor is size-independent — so the unit one is used; only the *shape class*
# matters, and a ball is the orientation-free choice.
function _matrix_ball(asm::ParticleAssembly)
    d = length(particle_center(asm, first(particle_names(asm))))
    return d == 2 ? Elasticity.Ellipsoid(1.0, 1.0) : Elasticity.Ellipsoid(1.0)
end

# ─── Every one-site scheme, on an assembly ───────────────────────────────────
#
# More specific than the generic `_evaluate` fallback that reports an
# unimplemented (cell, scheme) pair, and less specific than the `ClusterModel`
# and `EquivalentInclusion` methods, which therefore keep winning and keep
# reading the positions.  Everything else is a one-site scheme and is answered
# through `RVE(asm)`.
#
# `Laminated` is refused here rather than delegated: it would otherwise reach
# the RVE fallback and report the failure against an `RVE` the caller never
# built, naming the wrong cell.

"""
    _evaluate(asm::ParticleAssembly, scheme, ::Val{p}; kw...) -> AbstractTens

One-site schemes on an assembly, via [`RVE(asm)`](@ref). Only the volume
fractions are used — the positions are what [`ClusterModel`](@ref) and
[`EquivalentInclusion`](@ref) read, and those have their own methods.

`matrix_geometry` and `distribution_shape` are forwarded to the conversion;
everything else goes to the scheme.
"""
function Schemes._evaluate(
        asm::ParticleAssembly, scheme::HomogenizationScheme, ::Val{p};
        matrix_geometry = nothing, distribution_shape = nothing, kw...
    ) where {p}
    rve = Schemes.RVE(
        asm; matrix_geometry = matrix_geometry, distribution_shape = distribution_shape
    )
    return Schemes._evaluate(rve, scheme, Val(p); kw...)
end

# Refused by dispatch rather than by a branch above, so the message names the
# cell the caller actually built.  Delegating would report the failure against
# an `RVE` that never appeared in their code.
Schemes._evaluate(
    ::ParticleAssembly, ::Schemes.Laminated, ::Val{p}; kw...
) where {p} = throw(
    ArgumentError(
        "homogenize: `Laminated` needs an ordered stack of layers, so it does " *
            "not apply to a ParticleAssembly. Use `ClusterModel` or " *
            "`EquivalentInclusion` to resolve the interactions between " *
            "particles, or any one-site scheme to ignore their positions."
    )
)
