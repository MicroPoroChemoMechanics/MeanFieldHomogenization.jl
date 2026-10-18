# Starting a studio from the REPL: the options of `mfhstudio`, the server on
# the loopback interface, the browser on the page, the compute process prepared
# in the background so that the page answers at once and shows the progress.

"""
    StudioHandle

A running studio, as `mfhstudio(wait = false)` returns it: `wait(h)` blocks until
it stops, `close(h)` stops it, `studio_url(h)` is the page's address with its
token.
"""
mutable struct StudioHandle{V, S <: Session}
    const server::V
    const session::S
    const url::String
    closed::Bool
end

studio_url(h::StudioHandle) = h.url

# Ctrl-C interrupts a task asleep on a timer, never one waiting on a condition
# (as `wait(server)` does): the server is watched by short sleeps instead, and
# a Ctrl-C another task of the studio received is raised here.
function Base.wait(h::StudioHandle)
    while !h.closed && isopen(h.server)
        Threads.atomic_xchg!(INTERRUPTED, false) && throw(InterruptException())
        sleep(0.2)
    end
    return nothing
end

function Base.close(h::StudioHandle)
    h.closed && return nothing
    h.closed = true
    HTTP.forceclose(h.server)
    shutdown!(h.session.compute)
    return nothing
end

# The token stays out of the REPL's display of the handle.
Base.show(io::IO, h::StudioHandle) =
    print(io, "StudioHandle(", h.session.app.name, " on port ", h.session.port, h.closed ? ", closed)" : ")")

"""
    serve(session, router; tries) -> server

Bind the server on the session's port, or, when that port is taken and
`tries > 1`, on the next free one (the session's port is updated). A port
taken that cannot be replaced is refused with the way out.
"""
function serve(session::Session, router::HTTP.Router; tries::Integer = 1)
    first_port = session.port
    for k in 0:(tries - 1)
        session.port = first_port + k
        server = try
            HTTP.serve!(handler(session, router), session.host, session.port)
        catch err
            err isa HTTP.AddressInUseError || rethrow()
            continue
        end
        # port 0 asks the system for a free one: the address bound says which
        session.port = parse(Int, last(split(HTTP.server_addr(server), ':')))
        k == 0 || println("Port $first_port is taken; the studio uses port $(session.port).")
        return server
    end
    throw(
        ArgumentError(
            tries == 1 ?
                "port $first_port is already in use; start with another one, `port = $(first_port + 1)`." :
                "ports $first_port to $(first_port + tries - 1) are all in use; start with `port = …`."
        )
    )
end

"The host as a browser should reach it."
url_host(host::AbstractString) = host in ("0.0.0.0", "::") ? "localhost" : host == "::1" ? "[::1]" : host

"""
    run_check(app, compute; managed, update) -> Bool

Prepare the compute environment, start the compute process, print what the
app's check returns, and stop: the diagnosis of `check = true`, without a
server.
"""
function run_check(app::StudioApp, compute::ComputeProcess; managed::Bool, update::Bool)
    println("$(app.name) $(app.version): checking the compute environment $(compute.project).")
    println("Free memory: $(gigabytes(available_memory())) GB.")
    prepare_environment(
        compute.project, app.environment; managed, update, julia = compute.julia, log = println,
    )
    start!(compute)
    outcome = call!(compute, app.check)
    shutdown!(compute)
    if outcome.ok
        foreach(println, outcome.value)
        println("The compute environment is ready.")
    else
        println("The check failed: ", outcome.error)
    end
    return outcome.ok
end

"""
    run_studio(app; host, port, no_browser, project, julia, check, wait, update)

Start the studio `app`; see [`mfhstudio`](@ref) for the options. Returns a
[`StudioHandle`](@ref) when `wait = false`, `nothing` after a blocking run, and
whether the check passed with `check = true`.
"""
function run_studio(
        app::StudioApp;
        host::AbstractString, port::Union{Nothing, Integer}, no_browser::Bool,
        project::Union{Nothing, AbstractString}, julia::Union{Nothing, AbstractString},
        check::Bool, wait::Bool, update::Bool,
    )
    from_env = get(ENV, app.project_variable, "")
    managed = project === nothing && isempty(from_env)
    proj = project !== nothing ? project : isempty(from_env) ? app.default_project : from_env
    exe = julia !== nothing ? julia : get(ENV, "JULIA", first(Base.julia_cmd().exec))
    compute = ComputeProcess(; project = proj, julia = exe, boot = app.boot)
    check && return run_check(app, compute; managed, update)

    host in LOOPBACK || @warn "The studio listens on $host: it can be reached from other machines, " *
        "and runs code on request. Its token is the only protection; share it with nobody."
    # A blocking script ends on Ctrl-C at once (the compute process follows it:
    # its connection closes, which ends Malt's loop): in a script, the interrupt reaches whichever task
    # runs next, often one of the server's own, where it would be lost. Set
    # before the address is printed, the first moment a user may press it. At
    # the REPL the interrupt is raised in the waiting loop, or relayed to it.
    wait && !isinteractive() && Base.exit_on_sigint(true)
    token = Random.randstring(Random.RandomDevice(), 32)
    session = Session(app, token, String(host), Int(something(port, app.default_port)), compute, app.new_state(), managed)
    router = HTTP.Router()
    core_routes!(router, session)
    app.routes!(router, session)
    # the default port gives way to the next free one; a port the user chose does not
    server = serve(session, router; tries = port === nothing ? 20 : 1)
    url = "http://$(url_host(host)):$(session.port)/?token=$token"
    handle = StudioHandle(server, session, url, false)
    # The waiting loop compiled before the address is printed, the first moment
    # a user may press Ctrl-C: a Ctrl-C during a compilation can hold the exit.
    wait && precompile(Base.wait, (typeof(handle),))
    foreach(println, banner(app.name, url, session.port))
    flush(stdout)   # a terminal shows it at once; a log file would hold it back
    no_browser || open_browser(url) || println("  Open that address in your browser.")
    errormonitor(@async boot!(session; update))
    # No hook at exit: the system closes the server's socket, and the compute
    # process ends when its connection closes. A hook would compile the whole
    # shutdown on the way out, which held a Ctrl-C for tens of seconds.
    wait || return handle
    try
        Base.wait(handle)
    catch err
        err isa InterruptException || rethrow()
        println("\nStopping $(app.name).")
    finally
        close(handle)
    end
    return nothing
end
