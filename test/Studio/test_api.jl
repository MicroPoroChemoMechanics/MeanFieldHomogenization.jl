# The routes of MFH Studio, through a real HTTP client, with a session whose
# compute process is never started: the model and its script, the catalog, the
# views, the files, and a run refused while no process is ready.

cp0 = MS.ComputeProcess(; project = joinpath(tempdir(), "no-such-environment"), julia = first(Base.julia_cmd().exec), boot = "none.jl")
session = MS.Session(MS.APP, "a"^32, "127.0.0.1", 0, cp0, MS.StudioState(), false)
router = HTTP.Router()
MS.core_routes!(router, session)
MS.mfh_routes!(router, session)
server = MS.serve(session, router)
base = "http://127.0.0.1:$(session.port)"
token = ["X-Studio-Token" => session.token]
json_headers = vcat(token, ["Content-Type" => "application/json"])
api_get(path) = HTTP.get(base * path, token; status_exception = false)
api_post(path, body) = HTTP.post(base * path, json_headers, body isa AbstractString ? body : JSON.json(body); status_exception = false)
api_json(r) = JSON.parse(String(r.body))
st = session.state

# An Echoes script (the porous benchmark the translator's own tests use).
const ECHOES_SCRIPT = """
from echoes import *
ks=72.;mus=32.;Cs=stiff_kmu(ks,mus)
kp=1.e-6;mup=1.e-6;Cp=stiff_kmu(kp,mup)
ver=rve(matrix="SOLID")
ver["SOLID"]=ellipsoid(shape=spheroidal(1.),symmetrize=[ISO],prop={"C":Cs})
ver["PORE"]=ellipsoid(shape=spheroidal(1.),symmetrize=[ISO],prop={"C":Cp})
def Chom(f):
    ver["PORE"].fraction=f ; ver["SOLID"].fraction=1-f
    C=homogenize(prop="C",rve=ver,scheme=MT)
    return C.k
"""

@testset "the model and its script" begin
    s = api_json(api_get("/api/state"))
    @test s["model"]["title"] == "porous_benchmark" && occursin("function build_rve()", s["script"])
    @test isempty(s["problems"]) && s["path"] === nothing && s["multiscale"] == false
    # the options a scheme does not read are dropped, those it reads kept
    m = s["model"]
    m["sweep"]["schemes"] = [
        Dict("name" => "MoriTanaka", "options" => Dict("verbose" => false)),
        Dict("name" => "SelfConsistent", "options" => Dict("abstol" => 1.0e-10, "nsteps" => 3)),
        Dict("name" => "NotAScheme", "options" => Dict("x" => 1)),
    ]
    m["cells"][1]["phases"][2]["properties"][1] = Dict("key" => ":C", "source" => "cell", "cell" => "elsewhere", "scheme" => "Voigt", "scheme_options" => Dict("x" => 1))
    s = api_json(api_post("/api/model", Dict("model" => m)))
    @test [sc["options"] for sc in s["model"]["sweep"]["schemes"]] == [Dict(), Dict("abstol" => 1.0e-10), Dict("x" => 1)]
    @test s["model"]["cells"][1]["phases"][2]["properties"][1]["scheme_options"] == Dict()
    @test occursin("SelfConsistent(; abstol = 1.0e-10)", s["script"])
    # the arguments of an interface keep their order, as the page sent them
    body = """{"model": {"cells": [{"name": "lam", "kind": "laminate", "layers": [{"name": "A", "amount": 1.0,
    "interface": {"kind": "SpringInterface", "args": {"kt": 2.0, "kn": 1.0}}}]}]}}"""
    @test occursin("interface = SpringInterface(2.0, 1.0)", api_json(api_post("/api/model", body))["script"])
    @test api_post("/api/model", Dict("no" => "model")).status == 400
    @test api_post("/api/model", "[1, 2]").status == 500
    lock(() -> (st.model = MS.default_model()), st.lock)
end

@testset "the catalog and the views" begin
    cat = api_json(api_get("/api/catalog"))
    @test cat["introspected"] && any(s -> s["name"] == "MoriTanaka", cat["schemes"])
    @test MS.session_catalog(st) === MS.session_catalog(st)       # read once
    c = only(st.model.cells)
    sc = api_json(api_post("/api/traces", Dict("cell" => c.id, "phase" => "PORE", "cutaway" => false, "guides" => false)))
    @test length(sc["data"]) == 1 && sc["data"][1]["type"] == "surface"
    @test api_json(api_post("/api/traces", Dict("cell" => "gone")))["note"] == "no such cell"
    @test api_json(api_get("/api/convert/available"))["available"] isa Bool
end

@testset "files" begin
    mktempdir() do dir
        # a failed open leaves the model and the bound file alone, so that a
        # Save cannot write over a script
        lock(() -> (st.path = joinpath(dir, "before.jl")), st.lock)
        before = JSON.json(MS.to_dict(st.model))
        r = api_post("/api/open", Dict("path" => joinpath(dir, "missing.jl")))
        @test r.status == 422 && occursin("no such file", api_json(r)["error"])
        @test st.path == joinpath(dir, "before.jl") && JSON.json(MS.to_dict(st.model)) == before
        # save, then open what was saved: it reopens exactly
        target = joinpath(dir, "sub", "saved.jl")
        r = api_json(api_post("/api/save", Dict("path" => target)))
        @test r["path"] == target && r["bytes"] == filesize(target) && st.path == target
        r = api_json(api_post("/api/open", Dict("path" => target)))
        @test r["read_report"]["exact"] && r["path"] == target
        # a script written by hand, and a text sent without a file
        write(joinpath(dir, "hand.jl"), "const k = 1.0\nprintln(k)\n")
        r = api_json(api_post("/api/open", Dict("path" => joinpath(dir, "hand.jl"))))
        @test !r["read_report"]["exact"] && r["read_report"]["recognized"] == 1
        r = api_json(api_post("/api/open", Dict("source" => "x = 1\n")))
        @test r["path"] === nothing && length(r["model"]["opaque"]) == 1
        # save with the text given, to the bound file; nothing to save to
        @test api_json(api_post("/api/save", Dict("path" => target, "source" => "# mine\n")))["bytes"] == 7
        @test read(target, String) == "# mine\n"
        lock(() -> (st.path = nothing), st.lock)
        @test api_post("/api/save", Dict()).status == 400
        # the file picker: folders and scripts, hidden entries left out
        mkpath(joinpath(dir, "folder"))
        write(joinpath(dir, "B.py"), "")
        write(joinpath(dir, ".hidden.jl"), "")
        write(joinpath(dir, "notes.txt"), "")
        listing = api_json(api_get("/api/browse?path=" * HTTP.escapeuri(dir)))
        @test listing["path"] == dir && [d["name"] for d in listing["dirs"]] == ["folder", "sub"]
        @test [f["name"] for f in listing["files"]] == ["B.py", "hand.jl"]
        @test only(f for f in listing["files"] if f["name"] == "B.py")["echoes"]
        @test listing["parent"] == dirname(dir) && listing["sep"] == string(Base.Filesystem.path_separator)
        @test any(p -> p["label"] == "home", listing["places"])
        # a file names its folder; a path that is not there falls back
        @test api_json(api_get("/api/browse?path=" * HTTP.escapeuri(joinpath(dir, "hand.jl"))))["path"] == dir
        @test api_json(api_get("/api/browse?path=" * HTTP.escapeuri(joinpath(dir, "nowhere"))))["path"] == first(MS.places())["path"]
        @test api_json(api_get("/api/browse"))["path"] == first(MS.places())["path"]
        lock(() -> (st.path = joinpath(dir, "hand.jl")), st.lock)
        @test api_json(api_get("/api/browse"))["path"] == dir
        @test MS.browse("/", nothing)["parent"] === nothing
        if Sys.isunix() && !(Base.get(ENV, "USER", "") == "root")
            locked = mkpath(joinpath(dir, "locked"))
            chmod(locked, 0o000)
            @test isempty(MS.browse(locked, nothing)["files"])
            chmod(locked, 0o755)
        end
    end
end

@testset "an Echoes script" begin
    mktempdir() do dir
        py = joinpath(dir, "porous.py")
        write(py, ECHOES_SCRIPT)
        @test MS.is_echoes_script(py) && MS.suggest_output(py) == joinpath(dir, "porous.jl")
        # without a Python that runs the translator, the page is told what to do
        withenv("MFHSTUDIO_PYTHON" => "no-such-python") do
            r = api_post("/api/open", Dict("path" => py))
            @test r.status == 422 && occursin("python3 -m echoes2mfh translate", api_json(r)["error"])
        end
        @test_throws ArgumentError MS.convert_echoes(joinpath(dir, "missing.py"))
        if MS.find_translator() === nothing
            @info "No Python 3.10 with the translator here: the conversion itself is not tested."
        else
            out = joinpath(dir, "out", "porous.jl")
            opened = api_json(api_post("/api/open", Dict("path" => py, "output" => out)))
            @test isfile(out) && opened["path"] == out && opened["conversion"]["output"] == out
            @test occursin("Translated with", opened["conversion"]["summary"])
            @test occursin("function build_ver(f)", read(out, String))
            # Python 2 is refused with its remedy
            old = joinpath(dir, "old.py")
            write(old, "print 'x'\n")
            r = api_post("/api/open", Dict("path" => old))
            @test r.status == 422 && occursin("2to3", api_json(r)["error"])
            # a script the translator cannot read
            if Sys.isunix() && !(Base.get(ENV, "USER", "") == "root")
                hidden = joinpath(dir, "hidden.py")
                write(hidden, ECHOES_SCRIPT)
                chmod(hidden, 0o000)
                @test_throws ArgumentError MS.convert_echoes(hidden)
                chmod(hidden, 0o644)
            end
            conv = opened["conversion"]
            @test conv["summary"] == MS.conversion_summary(length(conv["findings"]), conv["blocking"])
        end
    end
end

@testset "the summary of a translation" begin
    @test MS.conversion_summary(0, 0) == "Translated with no findings."
    @test startswith(MS.conversion_summary(1, 0), "Translated with 1 finding (0 blocking).")
    @test startswith(MS.conversion_summary(3, 2), "Translated with 3 findings (2 blocking).")
end

@testset "a run needs a compute process" begin
    r = api_json(api_post("/api/run", Dict()))
    @test r["ok"] == false && occursin("not ready", r["error"]) && r["stdout"] == ""
end

HTTP.forceclose(server)
