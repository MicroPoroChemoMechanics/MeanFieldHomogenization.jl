# The routes of MFH Studio: the model and its script, the catalog, the 3-D
# views, a run in the compute process, and the files (open, save, browse).
# The JSON is shaped as the page has always read it.

"""
    StudioState()

The state of a session: the model, the file it is bound to (`nothing` until
saved or opened), the catalog once read, and what the last open understood.
"""
mutable struct StudioState
    model::StudioModel
    path::Union{Nothing, String}
    catalog::Union{Nothing, Dict{String, Any}}
    const lock::ReentrantLock
end

StudioState() = StudioState(default_model(), nothing, nothing, ReentrantLock())

"The catalog, read from the package once per session."
function session_catalog(st::StudioState)
    return lock(st.lock) do
        st.catalog === nothing && (st.catalog = catalog())
        st.catalog
    end
end

"What the page needs after any change: the model, its script, its problems, the bound file."
function state_payload(st::StudioState)
    m, path = lock(() -> (st.model, st.path), st.lock)
    return Dict{String, Any}(
        "model" => to_dict(m), "script" => generate(m), "problems" => validate(m),
        "path" => path, "multiscale" => uses_multiscale(m),
    )
end

bad_request(text::AbstractString) = json_response(400, Dict("error" => text))

"""
    prune_scheme_options!(model, catalog)

Drop the solver options a scheme does not read: switching from
`SelfConsistent` to `MoriTanaka` carried the old options along, and
`MoriTanaka(; verbose = false)` is a MethodError. A scheme the catalog does not
know keeps its options.
"""
function prune_scheme_options!(m::StudioModel, cat::AbstractDict)
    allowed = Dict(s["name"] => Set(o["name"] for o in s["options"] if o["editable"]) for s in cat["schemes"])
    function clean(name, options)
        keep = get(allowed, name, nothing)
        return keep === nothing ? options : JDict(k => v for (k, v) in options if k in keep)
    end
    for entry in m.sweep.schemes
        entry["options"] = clean(field(entry, "name", ""), field(entry, "options", JDict()))
    end
    for c in m.cells, ph in c.phases, pr in ph.properties
        (pr.source == "cell" && pr.scheme !== nothing) && (pr.scheme_options = clean(pr.scheme, pr.scheme_options))
    end
    return m
end

function post_model(st::StudioState, req::HTTP.Request)
    raw = get(read_json(req), "model", nothing)
    raw isa AbstractDict || return bad_request("the request carries no model")
    m = prune_scheme_options!(model_from_dict(raw), session_catalog(st))
    lock(() -> (st.model = m), st.lock)
    return json_response(200, state_payload(st))
end

function post_traces(st::StudioState, req::HTTP.Request)
    body = read_json(req)
    kw = (; (Symbol(k) => Bool(body[k]) for k in ("cutaway", "guides") if haskey(body, k))...)
    m = lock(() -> st.model, st.lock)
    return json_response(200, view_of(m, get(body, "cell", nothing), get(body, "phase", nothing); kw...))
end

function post_run(session::Session, st::StudioState, req::HTTP.Request)
    # the script of the model, or the text the request carries
    script = either(get(read_json(req), "source", nothing), generate(lock(() -> st.model, st.lock)))
    outcome = compute!(session, :(Main.StudioWorker.run_script($script)))
    outcome.ok && return json_response(200, outcome.value)
    return json_response(200, Dict{String, Any}("ok" => false, "stdout" => "", "error" => outcome.error))
end

"""
    open_script(st, path; source, output) -> Dict

Read a script into the model and bind the file. An Echoes script (`.py`) is
translated first, the Julia written to `output` (beside it by default), and
that is what is opened. A file that cannot be read leaves the model and the
bound file as they were, so that a Save cannot write over a script.
"""
function open_script(st::StudioState, path::AbstractString; source = nothing, output = nothing)
    conversion = nothing
    if is_echoes_script(path)
        conv = convert_echoes(path)
        target = something(output, suggest_output(path))
        mkpath(dirname(abspath(target)))
        write(target, conv["julia"])
        conversion = Dict{String, Any}(
            "source" => path, "output" => target, "summary" => conv["summary"],
            "findings" => conv["findings"], "blocking" => conv["blocking"],
        )
        source, path = conv["julia"], target
    end
    if source === nothing
        isfile(path) || throw(ArgumentError("no such file: $path"))
        source = read(path, String)
    end
    m, report = model_from_script(source)
    lock(st.lock) do
        st.model = m
        st.path = bound_path(path)
    end
    out = state_payload(st)
    out["read_report"] = report
    conversion === nothing || (out["conversion"] = conversion)
    return out
end

# A library example is a starting point, not a file to write over (an installed
# package is read-only): it opens unbound, and Save asks for a name.
bound_path(path::AbstractString) = isempty(path) || in_library(path) ? nothing : String(path)
in_library(path::AbstractString) = (d = dirname(abspath(path)); isdir(d) && samefile(d, LIBRARY_DIR))

function post_open(st::StudioState, req::HTTP.Request)
    body = read_json(req)
    return try
        json_response(200, open_script(st, field(body, "path", ""); source = get(body, "source", nothing), output = get(body, "output", nothing)))
    catch err
        err isa ArgumentError || rethrow()
        json_response(422, Dict("error" => err.msg))
    end
end

function post_save(st::StudioState, req::HTTP.Request)
    body = read_json(req)
    path = either(get(body, "path", nothing), lock(() -> st.path, st.lock))
    path === nothing && return bad_request("no path given")
    source = either(get(body, "source", nothing), generate(lock(() -> st.model, st.lock)))
    try
        mkpath(dirname(abspath(path)))
        write(path, source)
    catch err
        # a folder one may not write in, a read-only disk
        err isa Union{SystemError, Base.IOError} || rethrow()
        return json_response(422, Dict("error" => "Cannot write $path ($(sprint(showerror, err))). Choose another place with Save as…"))
    end
    lock(() -> (st.path = path), st.lock)
    return json_response(200, Dict("path" => path, "bytes" => ncodeunits(source)))
end

"The places the file picker offers, those that exist: examples, demo scripts, the package, home, the current folder."
function places()
    root = pkgdir(MeanFieldHomogenization)
    candidates = (
        "examples" => LIBRARY_DIR,
        "scripts" => joinpath(root, "scripts"), "package" => root, "home" => homedir(), "cwd" => pwd(),
    )
    return [Dict("label" => label, "path" => abspath(p)) for (label, p) in candidates if isdir(p)]
end

"""
    browse(path, bound; saving = false) -> Dict

A folder for the file picker: its folders and its Julia and Echoes scripts,
hidden entries aside. The page cannot have a real path from a file input, and a
path is what saving back to the same file needs; the server's file system is
also what a remote session must reach. Without a path, the folder of the bound
file, else the examples to open one, or the session's folder to save, since the
library is no place for one's own files.
"""
function browse(path::AbstractString, bound; saving::Bool = false)
    fallback = bound !== nothing ? dirname(abspath(bound)) : saving ? pwd() : first(p["path"] for p in places())
    start = isempty(path) ? fallback : abspath(expanduser(path))
    isfile(start) && (start = dirname(start))
    isdir(start) || (start = fallback)
    dirs, files = Dict{String, Any}[], Dict{String, Any}[]
    names = try
        readdir(start)
    catch err
        # a folder the user may not read lists as empty, its parent reachable
        err isa Base.IOError || rethrow()
        String[]
    end
    for name in sort(names; by = lowercase)
        startswith(name, '.') && continue
        full = joinpath(start, name)
        if isdir(full)
            push!(dirs, Dict("name" => name, "path" => full))
        elseif isfile(full) && (endswith(name, ".jl") || endswith(name, ".py"))
            # an Echoes script is opened by translating it: the listing says so
            push!(files, Dict("name" => name, "path" => full, "size" => filesize(full), "echoes" => endswith(name, ".py")))
        end
    end
    parent = dirname(start)
    return Dict{String, Any}(
        "path" => start, "parent" => parent == start ? nothing : parent, "dirs" => dirs, "files" => files,
        "places" => places(), "sep" => Base.Filesystem.path_separator,
    )
end

browse_route(st::StudioState, q) =
    browse(get(q, "path", ""), lock(() -> st.path, st.lock); saving = get(q, "for", "") == "save")

"Register the routes of MFH Studio."
function mfh_routes!(router::HTTP.Router, session::Session)
    st = session.state
    HTTP.register!(router, "GET", "/api/state", _ -> json_response(200, state_payload(st)))
    HTTP.register!(router, "GET", "/api/catalog", _ -> json_response(200, session_catalog(st)))
    HTTP.register!(router, "POST", "/api/model", req -> post_model(st, req))
    HTTP.register!(router, "POST", "/api/traces", req -> post_traces(st, req))
    HTTP.register!(router, "POST", "/api/run", req -> post_run(session, st, req))
    HTTP.register!(router, "POST", "/api/open", req -> post_open(st, req))
    HTTP.register!(router, "POST", "/api/save", req -> post_save(st, req))
    HTTP.register!(router, "GET", "/api/browse", req -> json_response(200, browse_route(st, query(req))))
    HTTP.register!(router, "GET", "/api/convert/available", _ -> json_response(200, Dict("available" => find_translator() !== nothing)))
    return nothing
end

"""
    warm_up!(session)

Load the package in the compute process and run the default model once, so
that the compilation of a homogenization is paid before the user's first run.
"""
function warm_up!(session::Session)
    cp = session.compute
    set_state!(cp, :ready, "Compiling a first homogenization, once per session.")
    m = default_model()
    m.sweep.mode = "single"
    warm = call!(cp, :(Main.StudioWorker.run_script($(generate(m)))))
    warm.stopped === :none || return nothing
    set_state!(cp, :ready)
    return nothing
end
