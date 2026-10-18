# The HTTP side of a studio. Every `/api` call carries the session's random
# token and comes from the studio's own page: the server runs code on request,
# so a call from another web page the user happens to have open, or from
# another user of the machine, must never get through. Everything else is a
# static file of the web directory, guarded against `..`.

"""
    StudioApp(; name, version, default_port, project_variable, default_project,
              environment, boot, web_dir, new_state, routes!, warm_up!, check)

What a studio adds to the core: its name and version, the port it listens on
by default, the environment variable naming the compute environment and the
default one, the packages that environment needs, the file the compute
process includes at start, the web directory, a function building its state,
one registering its routes on a router `(router, session) -> nothing`, one run
once the compute process is ready `(session) -> nothing`, and the expression
the compute process evaluates for `check = true`, returning lines to print.
"""
struct StudioApp{N, R, W}
    name::String
    version::VersionNumber
    default_port::Int
    project_variable::String
    default_project::String
    environment::EnvironmentSpec
    boot::String
    web_dir::String
    new_state::N
    routes!::R
    warm_up!::W
    check::Expr
end

function StudioApp(;
        name, version, default_port, project_variable, default_project,
        environment, boot, web_dir, new_state, routes!, warm_up!, check,
    )
    return StudioApp(
        name, version, default_port, project_variable, default_project,
        environment, boot, web_dir, new_state, routes!, warm_up!, check,
    )
end

"""
    Session

One running studio: its app, the token the page must present, where it
listens, its compute process and the state of its domain.
"""
mutable struct Session{S, A <: StudioApp}
    const app::A
    const token::String
    const host::String
    port::Int
    const compute::ComputeProcess
    const state::S
    const managed::Bool
end

const TOKEN_HEADER = "X-Studio-Token"
const LOOPBACK = ("127.0.0.1", "localhost", "::1")

"Compare two strings in a time that does not depend on where they differ."
function secure_equal(a::AbstractString, b::AbstractString)
    ncodeunits(a) == ncodeunits(b) || return false
    return foldl(|, (x ⊻ y for (x, y) in zip(codeunits(a), codeunits(b))); init = 0x00) == 0x00
end

"""
    request_problem(session, req) -> Union{Nothing, String}

Why an `/api` request is refused, or `nothing`: a missing or wrong token, a
`Host` that is not the studio's (a DNS-rebinding page), an `Origin` that is not
the studio's page, or a `POST` that is not JSON (which a form on another page
could send without the browser asking first).
"""
function request_problem(session::Session, req::HTTP.Request)
    secure_equal(HTTP.header(req, TOKEN_HEADER, ""), session.token) || return "missing or wrong token"
    host = HTTP.header(req, "Host", "")
    if session.host in LOOPBACK
        host in ("127.0.0.1:$(session.port)", "localhost:$(session.port)", "[::1]:$(session.port)") ||
            return "unexpected Host header"
    end
    origin = HTTP.header(req, "Origin", "")
    (isempty(origin) || origin == "http://" * host) || return "cross-origin request refused"
    if req.method == "POST"
        startswith(lowercase(HTTP.header(req, "Content-Type", "")), "application/json") ||
            return "a POST must carry application/json"
    end
    return nothing
end

"A JSON response, never cached."
function json_response(status::Integer, payload)
    headers = ["Content-Type" => "application/json; charset=utf-8", "Cache-Control" => "no-store"]
    return HTTP.Response(status, headers, JSON.json(payload))
end

"""
    read_json(req) -> Dict{String, Any}

The JSON object a request carries; an empty object for an empty body.
"""
function read_json(req::HTTP.Request)
    isempty(req.body) && return Dict{String, Any}()
    value = JSON.parse(String(copy(req.body)); dicttype = Dict{String, Any})
    value isa Dict{String, Any} || throw(ArgumentError("the request body is not a JSON object"))
    return value
end

"The query parameters of a request."
query(req::HTTP.Request) = HTTP.queryparams(HTTP.URI(req.target))

const CONTENT_TYPES = Dict(
    ".html" => "text/html; charset=utf-8",
    ".js" => "text/javascript; charset=utf-8",
    ".css" => "text/css; charset=utf-8",
    ".json" => "application/json; charset=utf-8",
    ".svg" => "image/svg+xml",
    ".png" => "image/png",
    ".ico" => "image/x-icon",
    ".woff2" => "font/woff2",
)

# Scripts come from the studio's own files only, which blocks an injected
# inline script; Plotly sets inline styles and exports images as data URLs.
const CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " *
    "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"

"""
    static_file(web_dir, path) -> Union{Nothing, String}

The file of `web_dir` a URL path names (`/` is `index.html`), or `nothing` for
a path that leaves the directory or names no file.
"""
function static_file(web_dir::AbstractString, path::AbstractString)
    rel = path == "/" ? "index.html" : HTTP.unescapeuri(lstrip(path, '/'))
    occursin('\0', rel) && return nothing
    rel = normpath(rel)
    (isabspath(rel) || rel == ".." || startswith(rel, ".." * Base.Filesystem.path_separator) || startswith(rel, "../")) &&
        return nothing
    file = joinpath(web_dir, rel)
    return isfile(file) ? file : nothing
end

function static_response(web_dir::AbstractString, path::AbstractString)
    file = static_file(web_dir, path)
    file === nothing && return HTTP.Response(404, "not found")
    headers = [
        "Content-Type" => get(CONTENT_TYPES, splitext(file)[2], "application/octet-stream"),
        "Cache-Control" => "no-cache",
        "X-Content-Type-Options" => "nosniff",
        "Content-Security-Policy" => CSP,
    ]
    return HTTP.Response(200, headers, read(file))
end

# A failing route answers 500 with its message instead of dropping the
# connection: this is the boundary between the server and the page, and the
# terminal shows the backtrace for a bug report.
function answer(router::HTTP.Router, req::HTTP.Request)
    try
        return router(req)
    catch err
        # a Ctrl-C that landed in this request stops the studio, not the request
        err isa InterruptException && (interrupted!(); return json_response(503, Dict("error" => "Interrupted.")))
        @error "a studio route failed" target = req.target exception = (err, catch_backtrace())
        return json_response(500, Dict("error" => sprint(showerror, err)))
    end
end

"""
    handler(session, router)

The function the HTTP server calls: `/api` requests go through the guard to
the router, other `GET`s to the static files.
"""
function handler(session::Session, router::HTTP.Router)
    return function (req::HTTP.Request)
        path = HTTP.URI(req.target).path
        if startswith(path, "/api/")
            problem = request_problem(session, req)
            problem === nothing || return json_response(403, Dict("error" => problem))
            return answer(router, req)
        end
        req.method == "GET" || return HTTP.Response(405, "method not allowed")
        return static_response(session.app.web_dir, path)
    end
end

"The status of the studio and of its compute process, for the page's badge."
function status_payload(session::Session)
    s = status(session.compute)
    return Dict{String, Any}(
        "app" => session.app.name, "version" => string(session.app.version),
        "state" => s.state, "message" => s.message, "log" => s.log, "elapsed" => s.elapsed,
        "memory_gb" => gigabytes(available_memory()),
    )
end

"""
    boot!(session; update = false) -> Nothing

Prepare the compute environment, start the compute process, then run the app's
warm-up. A failure leaves the process `:failed` with its reason, which the page
shows.
"""
function boot!(session::Session; update::Bool = false)
    cp = session.compute
    try
        set_state!(cp, :preparing, "Checking the compute environment $(cp.project).")
        prepare_environment(
            cp.project, session.app.environment;
            managed = session.managed, update, julia = cp.julia, log = line -> log!(cp, line),
        )
        start!(cp)
        session.app.warm_up!(session)
    catch err
        boot_failed!(cp, err)
    end
    return nothing
end

# The boot runs in a task of its own: its failure is shown on the page rather
# than lost with the task, and a Ctrl-C that landed in it is relayed.
function boot_failed!(cp::ComputeProcess, err)
    err isa InterruptException && interrupted!()
    set_state!(cp, :failed, err isa InterruptException ? "Interrupted." : error_text(err))
    return nothing
end

"Start the compute process again, in the background."
function respawn!(session::Session)
    shutdown!(session.compute)
    errormonitor(@async boot!(session))
    return nothing
end

"Register the routes every studio has: status, stop and restart."
function core_routes!(router::HTTP.Router, session::Session)
    HTTP.register!(router, "GET", "/api/status", _ -> json_response(200, status_payload(session)))
    HTTP.register!(
        router, "POST", "/api/compute/stop",
        _ -> json_response(200, Dict("stopping" => stop!(session.compute))),
    )
    HTTP.register!(
        router, "POST", "/api/compute/restart",
        _ -> (respawn!(session); json_response(200, Dict("restarting" => true))),
    )
    return nothing
end

"""
    compute!(session, expr) -> NamedTuple

[`call!`](@ref) on the session's compute process, restarting the process in the
background when the call had to kill it.
"""
function compute!(session::Session, expr::Expr)
    outcome = call!(session.compute, expr)
    outcome.stopped === :none || respawn!(session)
    return outcome
end
