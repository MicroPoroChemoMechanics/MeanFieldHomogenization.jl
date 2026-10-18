# Memory the machine can spare, read the way the kernel enforces it. A cgroup
# quota (a per-user slice capped below the physical memory, say) binds before
# the machine runs out, and libuv's `Sys.free_memory` sees only the cgroup a
# process sits in, never its ancestors nor `memory.high`. A calculation is
# refused below a floor before it starts, rather than discovered too big during
# the solve, and the compute process is stopped if it outgrows what was left.
# Nothing here is tuned to one machine: the figures are read where the studio
# runs, and the floor is what the package and one homogenization need.

const GIB = 2^30

"The least room a calculation starts with, in bytes."
const RUN_FLOOR = 3 * GIB ÷ 2

"What the compute process must leave to the rest of the machine, in bytes."
const HOST_MARGIN = GIB ÷ 2

"""
    cgroup_value(dir, file) -> Int

A cgroup (v2) file read as an integer: `typemax(Int)` for `max`, or for a file
that does not exist (no such controller, not Linux).
"""
function cgroup_value(dir::AbstractString, file::AbstractString)
    path = joinpath(dir, file)
    isfile(path) || return typemax(Int)
    return something(tryparse(Int, strip(read(path, String))), typemax(Int))
end

"Page cache charged to a cgroup, in bytes: the kernel reclaims it before a limit bites."
function page_cache(dir::AbstractString)
    path = joinpath(dir, "memory.stat")
    isfile(path) || return 0
    cache = 0
    for line in eachline(path)
        key, value = split(line)
        key in ("active_file", "inactive_file") && (cache += parse(Int, value))
    end
    return cache
end

"""
    cgroup_budget(; membership = "/proc/self/cgroup", root = "/sys/fs/cgroup") -> Union{Nothing, NamedTuple}

The memory budget of the cgroups (v2) this process runs in, from its own up to
the root: `limit`, the tightest `min(memory.high, memory.max)`, and `room`, the
least left under one of them, page cache counted as free. `nothing` when no
limit applies or none can be read (not Linux, cgroup v1).
"""
function cgroup_budget(; membership::AbstractString = "/proc/self/cgroup", root::AbstractString = "/sys/fs/cgroup")
    # the file that names this process's cgroup exists on Linux only
    isfile(membership) || return nothing
    lines = readlines(membership)
    k = findfirst(startswith("0::"), lines)
    k === nothing && return nothing
    dir = rstrip(root * lines[k][4:end], '/')
    limit, room = typemax(Int), typemax(Int)
    while startswith(dir, root * "/")
        cap = min(cgroup_value(dir, "memory.high"), cgroup_value(dir, "memory.max"))
        if cap < typemax(Int)
            limit = min(limit, cap)
            room = min(room, cap - cgroup_value(dir, "memory.current") + page_cache(dir))
        end
        dir = dirname(dir)
    end
    return limit == typemax(Int) ? nothing : (; limit, room)
end

"""
    available_memory() -> Int

The bytes a new process may still take: the lower of the machine's available
memory and of the room the tightest enclosing cgroup leaves.
"""
function available_memory()
    machine = machine_memory()
    budget = cgroup_budget()
    return budget === nothing ? machine : min(machine, budget.room)
end

"""
    machine_memory() -> Int

The memory the system can hand to a new process, in bytes. Linux and Windows
count in it the cache they would drop (`MemAvailable`, the standby list), as
libuv reports it; macOS keeps most of its memory as inactive pages and reports
only the few pages it has not touched as free, so there the inactive and
speculative pages are counted too, read with `host_statistics64`.
"""
function machine_memory end

@static if Sys.isapple()
    # `host_statistics64(…, HOST_VM_INFO64, …)` fills a `vm_statistics64` of
    # <mach/vm_statistics.h>, 38 words of 32 bits: the free, active, inactive
    # and wired page counts first, then nine 64-bit counters, then the
    # purgeable and the speculative counts.
    const HOST_VM_INFO64 = Cint(4)
    const VM_STATISTICS64_WORDS = 38
    function machine_memory()
        stats = zeros(UInt32, VM_STATISTICS64_WORDS)
        count = Ref{UInt32}(VM_STATISTICS64_WORDS)
        host = ccall(:mach_host_self, UInt32, ())
        kr = ccall(:host_statistics64, Cint, (UInt32, Cint, Ptr{UInt32}, Ref{UInt32}), host, HOST_VM_INFO64, stats, count)
        # the send right `mach_host_self` gives is returned (`mach_task_self()` is
        # a macro of C reading the global `mach_task_self_`)
        ccall(:mach_port_deallocate, Cint, (UInt32, UInt32), unsafe_load(cglobal(:mach_task_self_, UInt32)), host)
        # a refusal of the kernel leaves the free pages only, the cautious figure
        kr == 0 || return Int(Sys.free_memory())
        free, inactive, speculative = stats[1], stats[3], stats[24]
        return (Int(free) + Int(inactive) + Int(speculative)) * Int(ccall(:getpagesize, Cint, ()))
    end
else
    machine_memory() = min(Int(Sys.free_physical_memory()), Int(Sys.free_memory()))
end

"""
    process_rss(pid) -> Union{Nothing, Int}

The resident memory of process `pid`, in bytes, read from the system: `/proc`
on Linux, `proc_pidinfo` on macOS, `K32GetProcessMemoryInfo` on Windows.
`nothing` for a process that does not exist (any more), or a system that
offers none of these. Each system compiles its own method only.
"""
function process_rss end

@static if Sys.islinux()
    function process_rss(pid::Integer)
        lines = try
            readlines("/proc/$pid/status")
        catch err
            # no such process: it has ended
            err isa SystemError || rethrow()
            return nothing
        end
        k = findfirst(startswith("VmRSS:"), lines)
        return k === nothing ? nothing : 1024 * parse(Int, split(lines[k])[2])
    end
elseif Sys.isapple()
    # `struct proc_taskinfo` of <sys/proc_info.h>, 96 bytes: the resident size
    # is its second field, after the virtual size, both 64-bit.
    function process_rss(pid::Integer)
        info = zeros(UInt64, 12)
        PROC_PIDTASKINFO = Cint(4)
        n = ccall(
            (:proc_pidinfo, "libproc"), Cint, (Cint, Cint, UInt64, Ptr{UInt64}, Cint),
            pid, PROC_PIDTASKINFO, 0, info, sizeof(info),
        )
        return n == sizeof(info) ? Int(info[2]) : nothing
    end
elseif Sys.iswindows()
    # `PROCESS_MEMORY_COUNTERS` of <psapi.h>: two 32-bit counters, then SIZE_T
    # fields, the working set (the resident memory) being the second of them.
    struct ProcessMemoryCounters
        cb::UInt32
        page_fault_count::UInt32
        peak_working_set_size::Csize_t
        working_set_size::Csize_t
        quota_peak_paged_pool_usage::Csize_t
        quota_paged_pool_usage::Csize_t
        quota_peak_non_paged_pool_usage::Csize_t
        quota_non_paged_pool_usage::Csize_t
        pagefile_usage::Csize_t
        peak_pagefile_usage::Csize_t
    end
    function process_rss(pid::Integer)
        PROCESS_QUERY_LIMITED_INFORMATION = UInt32(0x1000)
        handle = ccall(
            (:OpenProcess, "kernel32"), stdcall, Ptr{Cvoid}, (UInt32, Cint, UInt32),
            PROCESS_QUERY_LIMITED_INFORMATION, false, pid,
        )
        handle == C_NULL && return nothing
        counters = Ref(ProcessMemoryCounters(sizeof(ProcessMemoryCounters), 0, 0, 0, 0, 0, 0, 0, 0, 0))
        ok = ccall(
            (:K32GetProcessMemoryInfo, "kernel32"), stdcall, Cint,
            (Ptr{Cvoid}, Ptr{ProcessMemoryCounters}, UInt32), handle, counters, sizeof(ProcessMemoryCounters),
        )
        ccall((:CloseHandle, "kernel32"), stdcall, Cint, (Ptr{Cvoid},), handle)
        return ok == 0 ? nothing : Int(counters[].working_set_size)
    end
else
    process_rss(::Integer) = nothing
end

"""
    heap_size_hint(available) -> String

The `--heap-size-hint` given to the compute process: six tenths of the memory
available when it starts, and at least 1 GiB. The garbage collector works
harder as the process nears it, which keeps the process within what the
machine had to spare.
"""
heap_size_hint(available::Integer) = string(max(6 * (available ÷ 10), GIB) ÷ 2^20, "M")

"Memory in gigabytes, for messages."
gigabytes(bytes::Integer) = round(bytes / GIB; digits = 1)
