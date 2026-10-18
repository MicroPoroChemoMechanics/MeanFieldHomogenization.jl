# The compute process: a separate Julia process that loads the package and runs
# the calculations, one at a time. Keeping it apart from the server keeps the
# interface responsive during a long run, lets a run be stopped (an interrupt,
# then a kill if the code does not yield) and keeps the compilation from one
# calculation to the next, which is most of the cost of a first one.

"""
    ComputeProcess(; project, julia, boot)

The handle of the compute process: the environment it runs in (`project`, as
for `julia --project`), the Julia executable, and the file it includes at
start. Its `state` is one of `:stopped`, `:preparing`, `:starting`, `:ready`,
`:busy` and `:failed`, with a `message` saying why; `log` keeps the last lines
it printed.
"""
mutable struct ComputeProcess
    const project::String
    const julia::String
    const boot::String
    const lock::ReentrantLock
    const log::Vector{String}
    worker::Union{Nothing, Malt.Worker}
    state::Symbol
    message::String
    stop_requested::Bool
    started_at::Float64
end

function ComputeProcess(; project::AbstractString, julia::AbstractString, boot::AbstractString)
    return ComputeProcess(project, julia, boot, ReentrantLock(), String[], nothing, :stopped, "", false, 0.0)
end

const LOG_LIMIT = 400

"Append a line to the log, keeping the last `LOG_LIMIT`."
function log!(cp::ComputeProcess, line::AbstractString)
    lock(cp.lock) do
        push!(cp.log, line)
        excess = length(cp.log) - LOG_LIMIT
        excess > 0 && deleteat!(cp.log, 1:excess)
    end
    return nothing
end

function set_state!(cp::ComputeProcess, state::Symbol, message::AbstractString = "")
    lock(cp.lock) do
        cp.state = state
        cp.message = message
    end
    isempty(message) || log!(cp, message)
    return nothing
end

"The state, message, recent log and elapsed run time, for the status route."
function status(cp::ComputeProcess; lines::Integer = 200)
    return lock(cp.lock) do
        n = length(cp.log)
        (;
            state = String(cp.state), message = cp.message,
            log = cp.log[max(1, n - lines + 1):n],
            elapsed = cp.state === :busy ? time() - cp.started_at : 0.0,
        )
    end
end

# A test run with coverage measures the code the compute process includes too,
# in the same mode: none, user code, all code, or the files under one path.
coverage_flags() = coverage_flags(Base.JLOptions().code_coverage, Base.JLOptions().tracked_path)
function coverage_flags(mode::Integer, tracked::Ptr{UInt8})
    mode == 0 && return String[]
    mode == 2 && return ["--code-coverage=all"]
    mode == 3 && return ["--code-coverage=@" * unsafe_string(tracked)]
    return ["--code-coverage=user"]
end

"The message of an error raised in the compute process, without the task wrapping."
function error_text(err)
    err isa TaskFailedException && return error_text(err.task.exception)
    err isa Malt.RemoteException && return err.message
    err isa Malt.TerminatedWorkerException && return "The compute process has stopped."
    return sprint(showerror, err)
end

# Ctrl-C raises an InterruptException in whichever task runs at that moment:
# the waiting loop of a blocking run, or one of the studio's own tasks (the
# boot, a request, a reader of the compute process's output). Those record it
# here instead of ending with it, and the waiting loop raises it again
# (`wait(::StudioHandle)`).
const INTERRUPTED = Threads.Atomic{Bool}(false)
interrupted!() = (INTERRUPTED[] = true; nothing)

"Log what the compute process writes, until it closes; a Ctrl-C landing here is relayed."
function drain!(cp::ComputeProcess, pipe)
    closed = false
    while !closed
        try
            foreach(line -> log!(cp, line), eachline(pipe))
            closed = true
        catch err
            err isa InterruptException || rethrow()
            interrupted!()
        end
    end
    return nothing
end

"""
    start!(cp) -> Nothing

Spawn the compute process and include its boot file. An error (a package that
does not load) propagates; the caller records it as the `:failed` state.
"""
function start!(cp::ComputeProcess)
    set_state!(cp, :starting, "Starting the compute process.")
    flags = vcat(
        ["--project=$(cp.project)", "--threads=1", "--heap-size-hint=$(heap_size_hint(available_memory()))"],
        coverage_flags(),
    )
    w = Malt.Worker(;
        exename = cp.julia, exeflags = flags, env = ["OPENBLAS_NUM_THREADS=1", join(DEFAULT_LOAD_PATH, '=')],
        monitor_stdout = false, monitor_stderr = false,
    )
    lock(cp.lock) do
        cp.worker = w
    end
    errormonitor(@async drain!(cp, w.stdout))
    errormonitor(@async drain!(cp, w.stderr))
    set_state!(cp, :starting, "Loading the packages.")
    Malt.remote_eval_wait(w, :(include($(cp.boot))))
    set_state!(cp, :ready)
    return nothing
end

"""
    call!(cp, expr; floor = RUN_FLOOR, cap = nothing) -> NamedTuple

Evaluate `expr` in the compute process, under the memory watchdog, when no
other call is running. A call is refused when less than `floor` bytes are
available; the process is killed if its resident memory exceeds `cap` (by
default what it holds plus what is available, less a margin for the machine).
Returns `(; ok, value, error, stopped)`: `stopped` is `:memory` or `:user` when
the process had to be killed, which leaves it `:stopped` for the caller to
restart, and `:none` otherwise.
"""
function call!(cp::ComputeProcess, expr::Expr; floor::Integer = RUN_FLOOR, cap::Union{Nothing, Integer} = nothing)
    w = lock(cp.lock) do
        cp.state === :ready || return nothing
        cp.state = :busy
        cp.stop_requested = false
        cp.started_at = time()
        cp.worker
    end
    w === nothing && return (; ok = false, value = nothing, error = unavailable_message(cp), stopped = :none)
    room = available_memory()
    if room < floor
        set_state!(cp, :ready)
        return (; ok = false, value = nothing, error = memory_refusal(room, floor), stopped = :none)
    end
    limit = cap === nothing ? something(process_rss(w.proc_pid), 0) + room - HOST_MARGIN : Int(cap)
    task = Malt.remote_eval(w, expr)
    stopped = watch!(cp, w, task, limit)
    if stopped !== :none
        kill_worker!(cp, w)
        return (; ok = false, value = nothing, error = stop_message(stopped, limit), stopped)
    end
    outcome = try
        (; ok = true, value = fetch(task), error = "", stopped = :none)
    catch err
        # a failed remote call is reported, not lost in the task that ran it
        err isa TaskFailedException || err isa Malt.RemoteException || rethrow()
        (; ok = false, value = nothing, error = error_text(err), stopped = :none)
    end
    user_stop = outcome.ok ? false : lock(() -> cp.stop_requested, cp.lock)
    set_state!(cp, :ready)
    # an interrupt that took leaves the process alive, compiled, and ready again
    user_stop && return (; ok = false, value = nothing, error = "Stopped.", stopped = :none)
    return outcome
end

function unavailable_message(cp::ComputeProcess)
    state, message = lock(() -> (cp.state, cp.message), cp.lock)
    state === :busy && return "A calculation is already running."
    state === :failed && return "The compute process could not start: $message"
    return "The compute process is not ready yet ($(state)). Wait for the status badge to turn green."
end

memory_refusal(room, floor) =
    "Only $(gigabytes(room)) GB of memory are free, and a calculation needs " *
    "$(gigabytes(floor)) GB. Close other programs, then run again."

function stop_message(reason::Symbol, cap::Integer)
    reason === :user && return "Stopped. The compute process is restarting, so the next run compiles again."
    return "The calculation was stopped: it needed more than the $(gigabytes(cap)) GB the machine can spare. " *
        "Make the calculation smaller (fewer points, fewer species), then run again."
end

# Waits for a running call, killing the process if it outgrows `cap`, if the
# machine runs short, or if a requested stop has not taken within 5 s (code
# that does not yield ignores the interrupt). Where the resident memory cannot
# be read, the floor checked before the call is the protection left.
function watch!(cp::ComputeProcess, w::Malt.Worker, task::Task, cap::Integer)
    stop_since = 0.0
    while !istaskdone(task)
        sleep(0.25)
        rss = process_rss(w.proc_pid)
        (rss !== nothing && rss > cap) && return :memory
        available_memory() < HOST_MARGIN ÷ 2 && return :memory
        if lock(() -> cp.stop_requested, cp.lock)
            stop_since == 0.0 && (stop_since = time())
            time() - stop_since > 5 && return :user
        end
    end
    return :none
end

"""
    stop!(cp) -> Bool

Interrupt the running call; the watchdog kills the process if the interrupt
does not take. `false` when nothing is running.
"""
function stop!(cp::ComputeProcess)
    w = lock(cp.lock) do
        cp.state === :busy || return nothing
        cp.stop_requested = true
        cp.worker
    end
    w === nothing && return false
    Malt.interrupt(w)
    return true
end

function kill_worker!(cp::ComputeProcess, w::Malt.Worker)
    Malt.isrunning(w) && kill(w, Base.SIGKILL)
    for _ in 1:50
        Malt.isrunning(w) || break
        sleep(0.1)
    end
    lock(cp.lock) do
        cp.worker === w && (cp.worker = nothing)
    end
    set_state!(cp, :stopped, "The compute process was stopped.")
    return nothing
end

"""
    shutdown!(cp) -> Nothing

Stop the compute process, if any, for good: by a normal exit when it is idle,
so that it finishes what it writes at exit (coverage counts, for one), and by a
kill when it is busy or does not exit within 5 s.
"""
function shutdown!(cp::ComputeProcess)
    w, busy = lock(() -> (cp.worker, cp.state === :busy), cp.lock)
    w === nothing && return nothing
    if !busy && Malt.isrunning(w)
        Malt.remote_do(Base.exit, w)
        for _ in 1:50
            Malt.isrunning(w) || break
            sleep(0.1)
        end
    end
    kill_worker!(cp, w)
    return nothing
end
