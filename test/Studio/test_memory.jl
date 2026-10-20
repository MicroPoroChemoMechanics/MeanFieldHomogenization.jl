# The memory figures: positive and bounded by the machine; the cgroup readers
# on files written for the test; the heap hint within its bounds.

avail = MS.available_memory()
@test 0 < avail <= Sys.total_physical_memory()
# what the system can hand out, at least what it reports as free (on macOS, the
# inactive pages counted besides; elsewhere libuv's own figure)
@test Sys.free_memory() ÷ 2 <= MS.machine_memory() <= Sys.total_physical_memory()
# read on every system the studio supports
@test MS.process_rss(getpid()) > 0
@test MS.process_rss(typemax(Int32)) === nothing
# another process, as the compute process is to the studio: its memory read too,
# and seen to grow with what it allocates (random bytes, which no system compresses)
child = run(pipeline(`$(Base.julia_cmd()) --startup-file=no -e "x = rand(UInt8, 100_000_000); sleep(60)"`; stdout = devnull, stderr = devnull); wait = false)
try
    held = 0
    for _ in 1:120
        held = something(MS.process_rss(getpid(child)), 0)
        held > 100_000_000 && break
        sleep(0.5)
    end
    @test held > 100_000_000
finally
    kill(child)
end

@test MS.heap_size_hint(0) == "1024M"
@test MS.heap_size_hint(100 * MS.GIB) == string(60 * 1024, "M")     # no upper cap
@test MS.heap_size_hint(3 * MS.GIB) == string(6 * (3 * MS.GIB ÷ 10) ÷ 2^20, "M")
@test MS.gigabytes(3 * MS.GIB ÷ 2) == 1.5

mktempdir() do dir
    write(joinpath(dir, "memory.max"), "max\n")
    write(joinpath(dir, "memory.high"), "1073741824\n")
    write(joinpath(dir, "memory.stat"), "anon 10\nactive_file 100\ninactive_file 23\nshmem 5\n")
    @test MS.cgroup_value(dir, "memory.max") == typemax(Int)
    @test MS.cgroup_value(dir, "memory.high") == MS.GIB
    @test MS.cgroup_value(dir, "absent") == typemax(Int)
    @test MS.page_cache(dir) == 123
    @test MS.page_cache(joinpath(dir, "nowhere")) == 0
end

b = MS.cgroup_budget()
@test b === nothing || (b.limit > 0 && b.room <= b.limit)

# A budget on a cgroup tree written for the test: a slice capped by memory.high,
# a user's slice inside it capped by memory.max, the process's scope with no
# limit of its own. The tightest limit, and the least room counting page cache.
mktempdir() do root
    slice = joinpath(root, "user.slice")
    user = joinpath(slice, "user-1.slice")
    mkpath(joinpath(user, "session.scope"))
    write(joinpath(slice, "memory.high"), "$(8 * MS.GIB)\n")
    write(joinpath(slice, "memory.current"), "$(3 * MS.GIB)\n")
    write(joinpath(slice, "memory.stat"), "active_file $(MS.GIB)\n")
    write(joinpath(user, "memory.max"), "$(6 * MS.GIB)\n")
    write(joinpath(user, "memory.current"), "$(5 * MS.GIB)\n")
    write(joinpath(user, "memory.stat"), "inactive_file $(MS.GIB ÷ 2)\n")
    membership = joinpath(root, "cgroup")
    write(membership, "0::/user.slice/user-1.slice/session.scope\n")
    @test MS.cgroup_budget(; membership, root) == (; limit = 6 * MS.GIB, room = 3 * MS.GIB ÷ 2)
    # no limit on the way up, a cgroup v1 membership, no membership at all
    write(membership, "0::/user.slice/user-1.slice/session.scope/\n")
    @test MS.cgroup_budget(; membership, root = joinpath(root, "elsewhere")) === nothing
    write(membership, "4:memory:/user.slice\n")
    @test MS.cgroup_budget(; membership, root) === nothing
    @test MS.cgroup_budget(; membership = joinpath(root, "absent"), root) === nothing
end

# the compute process is measured in the coverage mode of this process
path = "/some/path"
GC.@preserve path begin
    tracked = pointer(path)
    @test MS.coverage_flags(0, tracked) == String[]
    @test MS.coverage_flags(1, tracked) == ["--code-coverage=user"]
    @test MS.coverage_flags(2, tracked) == ["--code-coverage=all"]
    @test MS.coverage_flags(3, tracked) == ["--code-coverage=@/some/path"]
end
