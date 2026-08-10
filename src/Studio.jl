# =============================================================================
#  Studio.jl — the Julia entry point for MFH Studio.
#
#  MFH Studio (tools/mfhstudio) is a local web interface for building
#  MeanFieldHomogenization scripts. It is implemented in Python (standard library only)
#  and drives a Julia "sidecar" for 3-D traces, script read-back and running
#  models. This file is a thin launcher: it finds the studio in the checkout,
#  resolves a Python interpreter and runs `python3 -m mfhstudio`, forwarding
#  every option. The browser is opened and the sidecar warmed by the Python
#  process itself, exactly as when it is started from a shell.
# =============================================================================

"""
    mfhstudio(; host="127.0.0.1", port=8765, no_browser=false,
              project=nothing, julia=nothing, python=nothing,
              check=false, wait=true)

Start MFH Studio, the graphical builder for MeanFieldHomogenization scripts, and block
until it stops (press Ctrl-C in the REPL — the SIGINT reaches the server in the
same process group and shuts it down cleanly). With `wait = false`, spawn the
server and return its `Process` handle immediately, leaving the REPL free.

The server is the Python app under `tools/mfhstudio` of the MeanFieldHomogenization
checkout, run as `python -m mfhstudio`. Python 3.10+ (standard library only)
must be on `PATH`, or pointed at with the `python` keyword argument or the
`MFHSTUDIO_PYTHON` environment variable. The first interpreter on `PATH` that
can actually import the studio is used, which on Windows means `py -3` is tried
after `python` and `python3` — those two are often a Store stub or an MSYS2
build there.

# Arguments
- `host`/`port` — where to bind; the printed URL is `http://host:port/`.
  Pick a different `port` when the default 8765 is taken.
- `no_browser` — start without asking the browser to open.
- `project` — Julia environment for the sidecar: a directory, or `@name` for a
  shared one. Defaults to the MeanFieldHomogenization checkout (as the Python app does),
  honoring `MFHSTUDIO_PROJECT`.
- `julia` — Julia executable for the sidecar if it is not on `PATH` (the
  `JULIA` environment variable is honored too).
- `check` — verify the Julia side and exit instead of serving.
- `wait` — block until the server stops (default) or return its process handle.

# Examples
```julia
using MeanFieldHomogenization
mfhstudio()                  # http://127.0.0.1:8765, opens a browser
mfhstudio(port = 9000)       # pick a different port
mfhstudio(no_browser = true) # stay in the terminal
p = mfhstudio(wait = false)  # keep the REPL; later `wait(p)` or `kill(p)`
```
"""
function mfhstudio(;
        host::AbstractString = "127.0.0.1",
        port::Integer = 8765,
        no_browser::Bool = false,
        project::Union{AbstractString, Nothing} = nothing,
        julia::Union{AbstractString, Nothing} = nothing,
        python::Union{AbstractString, Nothing} = nothing,
        check::Bool = false,
        wait::Bool = true,
    )
    cmd = _studio_cmd(;
        host, port, no_browser, project, julia, python, check,
    )
    # `run(; wait=false)` sends the child's streams to devnull
    # (`spawn_opts_swallow`), so they are passed through explicitly to keep the
    # studio's console output visible, exactly as when it is started from a
    # shell. All three must be given: the redirections are positional
    # (stdin, stdout, stderr), and naming only two silently leaves the child's
    # *stderr* on devnull — which is where every Python error message would go.
    proc = run(cmd, devnull, stdout, stderr; wait = false)
    # The studio's lifetime is tied to the session that launched it, like a
    # Pluto workspace: if this Julia process goes away while the studio is
    # still up, the server (and its sidecar, which exits on stdin EOF) is
    # taken down with it rather than orphaned. Ctrl-C in the REPL never
    # reaches here: python shuts down first and the hook becomes a no-op.
    atexit() do
        try
            Base.kill(proc)
        catch
        end
    end
    wait && _wait_studio(proc)
    # A studio that is stopped cleanly (Ctrl-C, or `--check` reporting a
    # verdict) exits without complaint; any other non-zero status means it
    # never came up, and returning a dead `Process` as if nothing happened
    # leaves the user to work that out from an exit code.
    if wait && !check && !success(proc)
        error(
            "MFH Studio exited with status $(proc.exitcode) instead of serving. " *
                "Its own message is above; if there is none, run the command by " *
                "hand to see it:\n    $(_shell_form(cmd))",
        )
    end
    return proc
end

"""
The command as one would type it, for an error message.

Quoted the way a shell — or `cmd.exe`, where these paths have spaces far more
often — needs it, so the line can be pasted rather than repaired.
"""
function _shell_form(cmd::Cmd)
    quoted(s) = occursin(' ', s) ? "\"$s\"" : s
    return "cd " * quoted(cmd.dir) * " && " * join(map(quoted, cmd.exec), " ")
end

"""The MFH Studio application directory in this checkout, or a helpful error."""
function _studio_dir()
    root = pkgdir(MeanFieldHomogenization)
    dir = joinpath(root, "tools", "mfhstudio")
    isdir(dir) || error(
        "MFH Studio lives in the MeanFieldHomogenization checkout at `tools/mfhstudio`, " *
            "which is missing here ($dir). It is not part of the released package: " *
            "develop a clone and start the studio from there\n" *
            "    julia -e 'using Pkg; Pkg.develop(path = raw\"<MeanFieldHomogenization.jl>\")'",
    )
    return dir
end

"""
The interpreters to try, in order, when the user names none.

On Windows the `py` launcher comes last but matters: a PATH there routinely
offers a `python3.exe` that is a Microsoft Store stub or an MSYS2 build, while
`py -3` finds the real installation.
"""
_python_candidates() =
    Sys.iswindows() ? (["python"], ["python3"], ["py", "-3"]) : (["python3"], ["python"])

"""The last non-empty line of `s` — for Python, the exception message."""
function _last_line(s::AbstractString)
    for line in Iterators.reverse(split(s, '\n'))
        stripped = strip(line)
        isempty(stripped) || return first(stripped, 200)
    end
    return "no output"
end

"""
Can `argv` actually run the studio?

Two things have to hold and neither is visible from the file system: the
interpreter must be 3.10+, and it must see the `mfhstudio` package. Checking
them costs one short process and turns the failure mode this guards against —
a child that exits with status 1 and nothing else to say — into a message
naming the interpreter that could not do it. `-c` prepends the working
directory to `sys.path` exactly as `-m` does, so importing from the studio
directory here tests the same lookup the real command performs.
"""
function _python_probe(argv::Vector{String})
    code = "import sys, mfhstudio; sys.exit(0 if sys.version_info >= (3, 10) else 3)"
    cmd = Cmd(Cmd([argv; "-c"; code]); dir = _studio_dir())
    err = IOBuffer()
    ok = try
        success(pipeline(cmd; stdout = devnull, stderr = err))
    catch
        false
    end
    return ok, String(take!(err))
end

"""
A Python interpreter able to run the studio, as an argument vector, or a
helpful error. An explicitly named one is taken at its word; a discovered one
is probed first, so the search skips interpreters that would only fail later.
"""
function _find_python(explicit::Union{AbstractString, Nothing})
    explicit !== nothing && return [String(explicit)]
    env = get(ENV, "MFHSTUDIO_PYTHON", nothing)
    env !== nothing && return [env]

    reasons = String[]
    for argv in _python_candidates()
        exe = Sys.which(argv[1])
        exe === nothing && continue
        argv = [exe; argv[2:end]]
        ok, err = _python_probe(argv)
        ok && return argv
        push!(reasons, "  · $(join(argv, " ")) — $(_last_line(err))")
    end

    detail = isempty(reasons) ? "  · nothing named `python`, `python3`" *
        (Sys.iswindows() ? " or `py`" : "") * " is on PATH" : join(reasons, "\n")
    error(
        "MFH Studio needs Python 3.10+ (standard library only), and none of the " *
            "interpreters found here can run it:\n" * detail * "\n\n" *
            "Install Python 3.10 or newer, or point the launcher at one with " *
            "`mfhstudio(python = raw\"...\")` or the `MFHSTUDIO_PYTHON` environment " *
            "variable.",
    )
end

"""The `Cmd` that starts the studio — kept pure so the launcher is testable."""
function _studio_cmd(;
        host::AbstractString = "127.0.0.1",
        port::Integer = 8765,
        no_browser::Bool = false,
        project::Union{AbstractString, Nothing} = nothing,
        julia::Union{AbstractString, Nothing} = nothing,
        python::Union{AbstractString, Nothing} = nothing,
        check::Bool = false,
    )
    args = [_find_python(python); "-m"; "mfhstudio"]
    append!(args, ["--host", String(host)])
    append!(args, ["--port", string(port)])
    no_browser && push!(args, "--no-browser")
    project !== nothing && append!(args, ["--project", String(project)])
    julia !== nothing && append!(args, ["--julia", String(julia)])
    check && push!(args, "--check")
    return Cmd(Cmd(args); dir = _studio_dir())
end

"""
Wait for the studio process, letting Ctrl-C shut it down cleanly.

A terminal sends SIGINT to the whole foreground process group, so the Python
server receives it at the same time as Julia and stops on its own; here we just
wait for that to happen. Only when the signal never reached it (no controlling
terminal) is the process killed outright, and the interrupt rethrown.
"""
function _wait_studio(proc)
    try
        wait(proc)
    catch err
        for _ in 1:100
            Base.process_exited(proc) && return
            sleep(0.05)
        end
        Base.kill(proc)
        try
            wait(proc)
        catch
        end
        rethrow(err)
    end
    return nothing
end
