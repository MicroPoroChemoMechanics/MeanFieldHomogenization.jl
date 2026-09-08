# =============================================================================
#  common.jl — what every finite-element inclusion shares.
#
#  A finite-element evaluation costs one assembly, one factorization and a
#  handful of solves, while iterative schemes (self-consistent, differential)
#  ask for the same tensors again at every iteration with a *slightly*
#  different reference medium.  Memoizing on that reference is what keeps those
#  schemes usable; one-shot schemes (dilute, Mori-Tanaka, Maxwell, PCW) only
#  ever hit one key.
# =============================================================================

"""
    FECache()

Mutable side-store of a finite-element inclusion: the assembled discretization
(built once, on first use) and the response tensors already computed, keyed on
the reference medium.

`assemblies` counts the factorizations actually performed — used by the tests
to prove the memoization works. The field is deliberately kept out of the
struct's numeric fields so that it never interferes with `ForwardDiff`
reconstruction of a geometry parameter.
"""
mutable struct FECache
    setup::Any
    tensors::Dict{Any, Any}
    assemblies::Int
end

FECache() = FECache(nothing, Dict{Any, Any}(), 0)

"""
    fe_assembly_count(incl) -> Int

Number of finite-element assemblies (equivalently, factorizations) actually
performed for `incl` so far. Every distinct reference medium costs one; a
repeat costs none. Useful to check that the memoization of [`FECache`](@ref)
is doing its job.
"""
fe_assembly_count(incl) = _fe_cache(incl).assemblies

"""
    fe_reset!(incl) -> incl

Drop the cached discretization and every memoized response tensor.
"""
function fe_reset!(incl)
    c = _fe_cache(incl)
    c.setup = nothing
    empty!(c.tensors)
    c.assemblies = 0
    return incl
end

"Cache of a finite-element inclusion; every such type defines one method."
_fe_cache(incl) = throw(
    ArgumentError("$(typeof(incl)) is not a finite-element inclusion (no `FECache`)")
)

# ─── Refusing a solve the machine cannot afford ──────────────────────────────
#
#  A direct Cholesky of a three-dimensional second-order system is the step
#  that exhausts memory, and an out-of-memory kill does not fail politely: it
#  takes the whole session with it, and on a desktop the whole desktop.  So the
#  size is checked before the factorization is attempted, and the refusal names
#  the knob that fixes it.
#
#  Shared by every finite-element inclusion, not only the largest: the crack
#  cell already factorizes systems of order 10⁵.

"""
    fe_available_gb() -> Float64

`MemAvailable` from `/proc/meminfo`, in GiB, or `Inf` where that file is not
readable — every platform other than Linux, and some containers.

`Inf` is the deliberate answer for "unknown": a guard that refuses to run
because it cannot measure is worse than no guard.
"""
function fe_available_gb()
    try
        for line in eachline("/proc/meminfo")
            startswith(line, "MemAvailable:") || continue
            return parse(Float64, split(line)[2]) / 1024 / 1024
        end
    catch
        # Unreadable, absent, or a format this does not know: fall through.
    end
    return Inf
end

"""
    _fe_check_budget(ndofs; max_dofs, min_free_gb, what)

Refuse a solve of `ndofs` degrees of freedom that exceeds either budget.

Two independent limits, because they catch different mistakes. `max_dofs` is a
*deliberate* ceiling on the problem size, set from what the machine has been
shown to handle; `min_free_gb` is the live state, and catches a second heavy job
already running — a size that was affordable an hour ago is not affordable
alongside a documentation build.
"""
function _fe_check_budget(
        ndofs::Integer; max_dofs::Integer, min_free_gb::Real,
        what::AbstractString = "factorization",
    )
    if ndofs > max_dofs
        error(
            "refusing this $what: $(ndofs) degrees of freedom, above " *
                "max_dofs = $(max_dofs).\nCoarsen the mesh, or raise `max_dofs` " *
                "deliberately if the machine really has the memory — a " *
                "three-dimensional second-order Cholesky is what runs out."
        )
    end
    free = fe_available_gb()
    if free < min_free_gb
        error(
            "refusing this $what: only $(round(free; digits = 1)) GiB available, " *
                "below min_free_gb = $(min_free_gb).\nClose the other heavy job " *
                "first — an out-of-memory kill here takes the whole session with it."
        )
    end
    return nothing
end
