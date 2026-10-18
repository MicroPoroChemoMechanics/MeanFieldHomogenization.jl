# The HTTP side, with a session whose compute process is never started: the
# guard of every `/api` route, the static files, and the routes that need no
# calculation. Requests go through a real HTTP client.

cp0 = MS.ComputeProcess(; project = joinpath(tempdir(), "no-such-environment"), julia = first(Base.julia_cmd().exec), boot = "none.jl")
session = MS.Session(MS.APP, "s"^32, "127.0.0.1", 0, cp0, MS.StudioState(), false)
router = HTTP.Router()
MS.core_routes!(router, session)
MS.mfh_routes!(router, session)
server = MS.serve(session, router)
base = "http://127.0.0.1:$(session.port)"
token = ["X-Studio-Token" => session.token]
json_headers = vcat(token, ["Content-Type" => "application/json"])
get(path; headers = token) = HTTP.get(base * path, headers; status_exception = false)
post(path, body; headers = json_headers) = HTTP.post(base * path, headers, JSON.json(body); status_exception = false)
payload(r) = JSON.parse(String(r.body); dicttype = Dict{String, Any})

@testset "static files" begin
    r = get("/"; headers = Pair{String, String}[])
    @test r.status == 200
    @test occursin("MFH Studio", String(r.body))
    @test occursin("script-src 'self'", HTTP.header(r, "Content-Security-Policy"))
    @test get("/app.js"; headers = Pair{String, String}[]).status == 200
    # the French words, from a folder of the page, as JSON
    r = get("/i18n/fr.json"; headers = Pair{String, String}[])
    @test r.status == 200 && startswith(HTTP.header(r, "Content-Type"), "application/json")
    @test HTTP.header(get("/favicon.svg"; headers = Pair{String, String}[]), "Content-Type") == "image/svg+xml"
    @test get("/nothing.js"; headers = Pair{String, String}[]).status == 404
    @test get("/%2e%2e/%2e%2e/etc/passwd"; headers = Pair{String, String}[]).status == 404
    @test MS.static_file(MS.WEB_DIR, "/../model.jl") === nothing
    @test MS.static_file(MS.WEB_DIR, "/a%00b") === nothing
    @test HTTP.post(base * "/", [], ""; status_exception = false).status == 405
end

@testset "guard" begin
    @test get("/api/status"; headers = Pair{String, String}[]).status == 403
    @test get("/api/status"; headers = ["X-Studio-Token" => "t"^32]).status == 403
    @test get("/api/status"; headers = vcat(token, ["Origin" => "http://evil.example"])).status == 403
    @test get("/api/status"; headers = vcat(token, ["Origin" => "http://127.0.0.1:$(session.port)"])).status == 200
    @test get("/api/status"; headers = vcat(token, ["Host" => "evil.example:$(session.port)"])).status == 403
    @test HTTP.post(base * "/api/compute/stop", vcat(token, ["Content-Type" => "text/plain"]), "{}"; status_exception = false).status == 403
    @test MS.secure_equal("abc", "abc")
    @test !MS.secure_equal("abc", "abd")
    @test !MS.secure_equal("abc", "ab")
end

@testset "status, stop, restart" begin
    s = payload(get("/api/status"))
    @test s["state"] == "stopped"
    @test s["app"] == "MFH Studio"
    @test s["memory_gb"] > 0
    @test payload(post("/api/compute/stop", Dict()))["stopping"] == false
    # restarting boots in the background; this environment lacks the packages
    @test payload(post("/api/compute/restart", Dict()))["restarting"]
    for _ in 1:200
        payload(get("/api/status"))["state"] == "failed" && break
        sleep(0.05)
    end
    s = payload(get("/api/status"))
    @test s["state"] == "failed" && occursin("lacks MeanFieldHomogenization", s["message"])
end

@testset "an app is built by keyword" begin
    app = MS.StudioApp(;
        name = "Test Studio", version = v"0.0.1", default_port = 9999, project_variable = "X",
        default_project = "@x", environment = MS.EnvironmentSpec(Pair{String, String}[]),
        boot = "none.jl", web_dir = MS.WEB_DIR, new_state = () -> nothing,
        routes! = (r, s) -> nothing, warm_up! = s -> nothing, check = :(String[]),
    )
    @test app.name == "Test Studio" && app.default_port == 9999
end

@testset "a failing route answers 500" begin
    r = HTTP.Router()
    HTTP.register!(r, "GET", "/api/boom", _ -> error("boom"))
    resp = @test_logs (:error,) match_mode = :any MS.answer(r, HTTP.Request("GET", "/api/boom"))
    @test resp.status == 500 && occursin("boom", String(resp.body))
    @test_throws ArgumentError MS.read_json(HTTP.Request("POST", "/api/x", [], "[1, 2]"))
    @test MS.read_json(HTTP.Request("POST", "/api/x", [], "")) == Dict{String, Any}()
end

HTTP.forceclose(server)
