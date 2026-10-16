# =============================================================================
#  sidecar.jl — the long-lived Julia companion of MFH Studio.
#
#  Loading MeanFieldHomogenization costs ~10 s, so it is paid once and the process then
#  answers requests on stdin. The protocol is JSON lines: one request object
#  per line in, one response object per line out. Everything the interface
#  needs from Julia goes through here -- the feature catalog, 3-D traces,
#  reading an existing script, and running one.
#
#  Anything printed by user code would corrupt the protocol, so stdout is
#  redirected for the duration of every request and the captured text is
#  returned as data instead.
# =============================================================================

import Pkg

# The sidecar runs with `--project=<MeanFieldHomogenization root>`. On a fresh checkout —
# or after a dependency is added — that environment has not been instantiated,
# and `using JSON` dies with a stack trace whose only advice is to run
# `Pkg.instantiate()`. Printing that advice at somebody is worse than following
# it, so this does.
function _ensure_dependencies()
    try
        @eval using JSON
        return nothing
    catch
        @info "mfhstudio: installing the project's dependencies (first run — this can take a few minutes)"
        flush(stderr)
        Pkg.instantiate()
        @eval using JSON
        return nothing
    end
end

_ensure_dependencies()

const HERE = @__DIR__

# `ok` is flipped to false if MFH fails to load, so the interface can report a
# precise reason rather than appearing to hang.
const STATE = Dict{String, Any}("ready" => false, "error" => nothing)

function _boot()
    try
        include(joinpath(HERE, "introspect.jl"))
        include(joinpath(HERE, "geometry.jl"))
        include(joinpath(HERE, "parse_script.jl"))
        STATE["ready"] = true
    catch e
        STATE["error"] = sprint(showerror, e, catch_backtrace())
    end
    return nothing
end

# ── Request handlers ────────────────────────────────────────────────────────

# The handlers below live in files `include`d at run time by `_boot`, so they
# are newer than this function's world age and must be reached through
# `invokelatest`. Calling them directly raises "method too new to be called
# from this world context".
function handle(op::String, payload)
    op == "ping" && return Dict("pong" => true, "ready" => STATE["ready"])
    op == "catalog" && return Base.invokelatest(catalog)
    op == "traces" && return _traces(payload)
    op == "parse" && return Base.invokelatest(parse_script, String(payload["source"]))
    op == "run" && return _run(payload)
    error("unknown op `$op`")
end

"""
Build a geometry from the interface's description and return its Plotly scene.

The description is deliberately re-evaluated through the real constructors
(`Spheroid`, `EllipticCrack`, `LayeredSphere`, …) rather than drawn from raw
numbers: that way the picture shows what MFH will actually build, including
the degenerate-limit redirections (a zero semi-axis becomes a crack, an
infinite one a cylinder).
"""
function _traces(payload)
    src = String(payload["expr"])
    geom = _eval_geometry(src)
    kw = Dict{Symbol, Any}()
    haskey(payload, "cutaway") && (kw[:cutaway] = Bool(payload["cutaway"]))
    haskey(payload, "guides") && (kw[:guides] = Bool(payload["guides"]))
    # `scene` comes from a run-time `include` — see the note on `handle`.
    return Base.invokelatest(scene, geom; kw...)
end

# Geometry expressions come from the interface's own form state, not from a
# user-typed string; they are evaluated in a bare module holding only the names
# a geometry can legitimately use.
module GeomSandbox
using MeanFieldHomogenization
using TensND
end

function _eval_geometry(src::AbstractString)
    ex = Meta.parse(src)
    return Base.eval(GeomSandbox, ex)
end

"""
Run a generated script and return its output.

Each run gets a fresh anonymous module, so definitions from one run cannot
leak into the next, while MeanFieldHomogenization itself stays loaded in the session.
"""
function _run(payload)
    src = String(payload["source"])
    timeout = Float64(get(payload, "timeout", 300.0))

    result = Dict{String, Any}("ok" => false, "stdout" => "", "error" => nothing)
    # The sidecar is a server: there is never a display to open. Without this,
    # a script that calls `display(p)` prints "GKS: cannot open display" into
    # the captured output, where it reads like a failure of the model.
    get!(ENV, "GKSwstype", "100")

    mod = Module(:MFHStudioRun)
    Base.eval(mod, :(using MeanFieldHomogenization, TensND, LinearAlgebra, Printf))

    # The protocol itself travels on stdout, so the redirect has to be undone
    # no matter how the script ends. The `do` form restores it while unwinding,
    # which is why a timeout *interrupts* the task rather than abandoning it:
    # abandoning would leave stdout pointing at the capture file and every
    # later reply would vanish into it.
    path, io = mktemp()
    task = @task begin
        try
            redirect_stdout(io) do
                redirect_stderr(io) do
                    Base.include_string(mod, src, "mfhstudio_script.jl")
                end
            end
            result["ok"] = true
        catch e
            if e isa InterruptException || (e isa TaskFailedException)
                result["error"] = "interrupted"
            else
                result["error"] = sprint(showerror, e, catch_backtrace())
            end
        end
    end

    schedule(task)
    t0 = time()
    timed_out = false
    while !istaskdone(task)
        if time() - t0 > timeout
            timed_out = true
            try
                schedule(task, InterruptException(); error = true)
            catch
            end
            # give the unwind a moment to restore the streams
            for _ in 1:100
                istaskdone(task) && break
                sleep(0.05)
            end
            break
        end
        sleep(0.05)
    end

    if timed_out
        result["error"] = "timed out after $(round(timeout; digits = 1)) s"
        result["timeout"] = true
        # If the task refused to unwind, the streams may still be redirected;
        # say so plainly so the client restarts rather than trusting silence.
        result["wedged"] = !istaskdone(task)
    end

    try
        flush(io)
        close(io)
        result["stdout"] = read(path, String)
        rm(path; force = true)
    catch
    end

    # Curves the script chose to publish for the interface.
    if isdefined(mod, :MFHSTUDIO_RESULTS)
        try
            result["results"] = Base.eval(mod, :MFHSTUDIO_RESULTS)
        catch
        end
    end
    return result
end

# ── The loop ────────────────────────────────────────────────────────────────

function serve()
    _boot()
    # The first line tells the client whether the session is usable at all.
    println(JSON.json(Dict("event" => "ready", "ok" => STATE["ready"], "error" => STATE["error"])))
    flush(stdout)

    for line in eachline(stdin)
        isempty(strip(line)) && continue
        local req
        try
            req = JSON.parse(line)
        catch e
            println(JSON.json(Dict("id" => nothing, "ok" => false, "error" => "bad JSON: $(sprint(showerror, e))")))
            flush(stdout)
            continue
        end
        id = get(req, "id", nothing)
        op = String(get(req, "op", ""))
        payload = get(req, "payload", Dict{String, Any}())
        resp = try
            Dict("id" => id, "ok" => true, "result" => handle(op, payload))
        catch e
            Dict("id" => id, "ok" => false, "error" => sprint(showerror, e, catch_backtrace()))
        end
        println(JSON.json(resp))
        flush(stdout)
    end
    return nothing
end

# `@__FILE__` must be parenthesized: a macro call swallows everything to its
# right as arguments, so the bare form would parse as `@__FILE__(&& serve())`.
if abspath(PROGRAM_FILE) == (@__FILE__)
    serve()
end
