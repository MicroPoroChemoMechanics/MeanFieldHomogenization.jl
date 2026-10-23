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

Every access goes through `lock`: tasks sharing one inclusion (threads over a
parameter sweep) build the discretization once and run each reference medium
once, one solve at a time — which a finite-element solve needs anyway, for
memory.
"""
mutable struct FECache
    setup::Any
    const tensors::Dict{Any, Any}
    assemblies::Int
    const lock::ReentrantLock
end

FECache() = FECache(nothing, Dict{Any, Any}(), 0, ReentrantLock())

# The response for reference `key`, computed by `f()` on first request.
_fe_memo(f, c::FECache, key) = lock(() -> get!(f, c.tensors, key), c.lock)

# The discretization, built by `f()` on first request.
_fe_setup!(f, c::FECache) = lock(c.lock) do
    c.setup === nothing && (c.setup = f())
    c.setup
end

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
    lock(c.lock) do
        c.setup = nothing
        empty!(c.tensors)
        c.assemblies = 0
    end
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
    fe_available_gb(meminfo = "/proc/meminfo"; cgroup = "/proc/self/cgroup",
                    cgroup_root = "/sys/fs/cgroup") -> Float64

The memory this process can still use, in GiB: the smaller of `MemAvailable`
from `/proc/meminfo` and the room left under the control group it runs in
(cgroup v2), or `Inf` where neither can be read — every platform other than
Linux, and some containers.

The room under a control group is, over the group and each of its ancestors,
the limit (`memory.max`, or `memory.high` where the kernel starts reclaiming)
minus `memory.current`, plus the page cache it holds (`file` in `memory.stat`),
which the kernel can reclaim. Without it, a process capped well below the free
memory of the host was let through and killed by the kernel.

`Inf` is the deliberate answer for "unknown": a guard that refuses to run
because it cannot measure is worse than no guard.
"""
function fe_available_gb(
        meminfo::AbstractString = "/proc/meminfo";
        cgroup::AbstractString = "/proc/self/cgroup",
        cgroup_root::AbstractString = "/sys/fs/cgroup",
    )
    return min(_meminfo_available_gb(meminfo), _cgroup_room_gb(cgroup, cgroup_root))
end

function _meminfo_available_gb(meminfo::AbstractString)
    try
        for line in eachline(meminfo)
            startswith(line, "MemAvailable:") || continue
            return parse(Float64, split(line)[2]) / 1024 / 1024
        end
    catch err
        # Unreadable, absent, or a format this does not know: fall through.
        # Anything else is a bug and surfaces.
        err isa Union{SystemError, ArgumentError, BoundsError} || rethrow()
    end
    return Inf
end

function _cgroup_room_gb(cgroup::AbstractString, root::AbstractString)
    # The unified (v2) hierarchy is the `0::` line; a v1-only system has none.
    rel = nothing
    try
        for line in eachline(cgroup)
            startswith(line, "0::") && (rel = line[4:end]; break)
        end
    catch err
        err isa SystemError || rethrow()
    end
    rel === nothing && return Inf
    room = Inf
    dir = normpath(joinpath(root, lstrip(rel, '/')))
    top = normpath(root)
    while true
        current = _cgroup_bytes(joinpath(dir, "memory.current"))
        if current !== nothing
            cache = _cgroup_stat_bytes(joinpath(dir, "memory.stat"), "file")
            for limit in ("memory.max", "memory.high")
                lim = _cgroup_bytes(joinpath(dir, limit))
                lim === nothing || (room = min(room, (lim - current + cache) / 2^30))
            end
        end
        rstrip(dir, '/') == rstrip(top, '/') && break
        dir = dirname(rstrip(dir, '/'))
    end
    return room
end

# A byte count from a cgroup file, or `nothing` for "max" (no limit) or a file
# that cannot be read or parsed.
function _cgroup_bytes(path::AbstractString)
    try
        s = strip(read(path, String))
        return s == "max" ? nothing : parse(Int, s)
    catch err
        err isa Union{SystemError, ArgumentError} || rethrow()
        return nothing
    end
end

function _cgroup_stat_bytes(path::AbstractString, key::AbstractString)
    try
        for line in eachline(path)
            k, v = split(line)
            k == key && return parse(Int, v)
        end
    catch err
        err isa Union{SystemError, ArgumentError, BoundsError} || rethrow()
    end
    return 0
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
