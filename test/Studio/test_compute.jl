# A studio started as a user starts it, on the environment
# MFHSTUDIO_WORKER_PROJECT names (MeanFieldHomogenization, TensND, Plots): its
# boot and warm-up; runs through the page's route against published and
# closed-form values; a failing script, a stop, the refusal for lack of memory;
# then the watchdog and `check = true`, each on a process of its own.

project = ENV["MFHSTUDIO_WORKER_PROJECT"]
julia = first(Base.julia_cmd().exec)

h = redirect_stdout(devnull) do
    withenv(() -> mfhstudio(; port = 0, no_browser = true, wait = false, project), "MFHSTUDIO_PROJECT" => nothing)
end
cp = h.session.compute
token = last(split(MS.studio_url(h), "token="))
base = "http://127.0.0.1:$(h.session.port)"
run_get(path) = JSON.parse(String(HTTP.get(base * path, ["X-Studio-Token" => token]).body))
run_post(path, body) = JSON.parse(String(HTTP.post(base * path, ["X-Studio-Token" => token, "Content-Type" => "application/json"], JSON.json(body)).body))

# the boot prepares nothing (the environment is complete), loads the package
# and compiles a first homogenization
for _ in 1:2400
    local s = MS.status(cp)
    (s.state == "ready" && isempty(s.message) || s.state == "failed") && break
    sleep(0.25)
end
@test MS.status(cp).state == "ready"

"The values a run printed for `scheme`, by output label."
function printed(stdout, scheme)
    got = Dict{String, Float64}()
    for line in split(stdout, '\n')
        m = match(Regex("^\\s*" * scheme * " (\\w+)\\s+first = ([-\\d.eE+]+)"), line)
        m === nothing || (got[m[1]] = parse(Float64, m[2]))
    end
    return got
end

@testset "the porous benchmark against Echoes" begin
    m = run_get("/api/state")["model"]
    m["sweep"]["start"], m["sweep"]["stop"], m["sweep"]["length"], m["sweep"]["plot"] = 0.3, 0.3, 2, false
    run_post("/api/model", Dict("model" => m))
    r = run_post("/api/run", Dict())
    @test r["ok"]
    got = printed(r["stdout"], "MoriTanaka")
    # the values Echoes 1.0 computes for 30 % of spherical pores
    @test abs(got["k"] - 33.460582) < 1.0e-5 && abs(got["mu"] - 17.626742) < 1.0e-5
    @test r["results"]["xlabel"] == "φ" && r["results"]["x"] == [0.3, 0.3] && length(r["results"]["MoriTanaka k"]) == 2
end

@testset "the bilayer against Backus" begin
    lam = MS.Cell(; name = "lam", kind = "laminate", layers = MS.default_layers())
    m = MS.StudioModel(; title = "bilayer", cells = [lam], root_cell = lam.id)
    m.sweep = MS.Sweep(;
        enabled = true, mode = "single", cell = lam.id, projection = "none",
        schemes = Any[MS.jdict("name" => n, "options" => MS.JDict()) for n in ("Laminated", "Voigt", "Reuss")],
        outputs = Any[MS.jdict("kind" => "km", "i" => 3, "j" => 3), MS.jdict("kind" => "km", "i" => 6, "j" => 6)],
    )
    run_post("/api/model", Dict("model" => MS.to_dict(m)))
    r = run_post("/api/run", Dict())
    @test r["ok"]
    got = Dict(
        m[1] => (parse(Float64, m[2]), parse(Float64, m[3]))
            for m in (match(r"^\s*(\w+)\s+KM33 = ([-\d.eE+]+)\s+KM66 = ([-\d.eE+]+)", l) for l in split(r["stdout"], '\n')) if m !== nothing
    )
    @test Set(keys(got)) == Set(["Laminated", "Voigt", "Reuss"])
    # C₃₃₃₃ = 1 / ⟨1/(λ + 2μ)⟩ and 2 C₁₂₁₂ = 2 ⟨μ⟩ for the 30/70 stack, read off
    # the printed table (`%.6f`), which is what a user sees
    λ2μ(k, mu) = k + 4mu / 3
    @test abs(got["Laminated"][1] - 1 / (0.3 / λ2μ(2.0, 0.8) + 0.7 / λ2μ(0.5, 0.2))) < 5.0e-7
    @test abs(got["Laminated"][2] - 2 * (0.3 * 0.8 + 0.7 * 0.2)) < 5.0e-7
    # two of the bracketings are equalities: Reuss across the layers, Voigt in their plane
    @test got["Laminated"][1] == got["Reuss"][1] && got["Laminated"][2] == got["Voigt"][2]
    run_post("/api/model", Dict("model" => MS.to_dict(MS.default_model())))
end

@testset "what a run prints and publishes" begin
    r = run_post("/api/run", Dict("source" => "x = 1\nerror(\"boom\")\n"))
    @test !r["ok"] && occursin("boom", r["error"]) && occursin("mfhstudio_script.jl:2", r["error"])
    r = run_post("/api/run", Dict("source" => "println(\"hello\")\n@warn \"careful\"\n"))
    @test r["ok"] && occursin("hello", r["stdout"]) && occursin("careful", r["stdout"]) && r["results"] === nothing && r["error"] === nothing
    # the results as JSON carries them: a non-finite number is null
    r = run_post("/api/run", Dict("source" => "const MFHSTUDIO_RESULTS = Dict(\"x\" => [1.0, NaN], :s => :sym, \"t\" => (1, true), \"z\" => 1 + 2im, \"n\" => nothing)"))
    @test r["results"] == Dict("x" => [1.0, nothing], "s" => "sym", "t" => [1.0, true], "z" => "1 + 2im", "n" => nothing)
    # the preamble of a studio script finds no project above the folder it runs from
    r = run_post("/api/run", Dict("source" => MS.ACTIVATE * "\nprintln(Base.active_project())\n"))
    @test r["ok"] && occursin(normpath(project), r["stdout"])
    @test MS.call!(cp, :(println("from the compute process"); flush(stdout))).ok
    for _ in 1:40
        "from the compute process" in MS.status(cp).log && break
        sleep(0.05)
    end
    @test "from the compute process" in MS.status(cp).log
    r = MS.call!(cp, :(error("internal")))
    @test !r.ok && occursin("internal", r.error)
end

@testset "busy, stop, restart, refusal" begin
    # A stop may kill the process (on Windows, where the interrupt does not
    # reach it), and its coverage with it: it exits normally first, writing what
    # the tests above measured.
    MS.shutdown!(cp)
    MS.start!(cp)
    @test !MS.stop!(cp)
    t = @async run_post("/api/run", Dict("source" => "sleep(60)\n"))
    for _ in 1:200
        MS.status(cp).state == "busy" && break
        sleep(0.05)
    end
    @test run_post("/api/run", Dict("source" => "1 + 1"))["error"] == "A calculation is already running."
    sleep(1)
    @test run_post("/api/compute/stop", Dict())["stopping"]
    r = fetch(t)
    @test !r["ok"]
    if occursin("restarting", r["error"])
        # the interrupt did not take (Windows): the watchdog killed the process,
        # and the route restarted it
        for _ in 1:2400
            MS.status(cp).state == "ready" && break
            sleep(0.25)
        end
    else
        @test r["error"] == "Stopped." && MS.status(cp).state == "ready"
    end
    r = MS.call!(cp, :(1 + 1); floor = typemax(Int))
    @test !r.ok && occursin("GB of memory are free", r.error)
    @test MS.call!(cp, :(1 + 1)).value == 2
    # the page's Restart boots the process again, warm-up included
    @test run_post("/api/compute/restart", Dict())["restarting"]
    for _ in 1:2400
        local s = MS.status(cp)
        (s.state == "ready" && isempty(s.message) || s.state == "failed") && break
        sleep(0.25)
    end
    @test MS.status(cp).state == "ready"
end

close(h)          # a normal exit of the compute process
@test MS.status(cp).state == "stopped"

@testset "watchdog" begin
    cp2 = MS.ComputeProcess(; project, julia, boot = joinpath(MS.WORKER_DIR, "boot.jl"))
    MS.start!(cp2)
    rss = something(MS.process_rss(cp2.worker.proc_pid), 0)
    r = MS.call!(cp2, :(x = ones(UInt8, 600_000_000); sleep(30); sum(x)); cap = rss + 200 * 2^20)
    @test !r.ok && r.stopped === :memory && occursin("Make the calculation smaller", r.error)
    @test MS.status(cp2).state == "stopped"
    @test occursin("not ready", MS.call!(cp2, :(1 + 1)).error)
end

@testset "check = true" begin
    out = Pipe()
    ok = redirect_stdout(out) do
        withenv(() -> mfhstudio(; check = true, project), "MFHSTUDIO_PROJECT" => nothing)
    end
    close(out.in)
    text = read(out, String)
    @test ok && occursin("MeanFieldHomogenization", text) && occursin("The compute environment is ready.", text)
end

@testset "a check that fails" begin
    a = MS.APP
    app = MS.StudioApp(;
        name = a.name, version = a.version, default_port = a.default_port,
        project_variable = a.project_variable, default_project = a.default_project,
        environment = a.environment, boot = a.boot, web_dir = a.web_dir, new_state = a.new_state,
        routes! = a.routes!, warm_up! = a.warm_up!, check = :(error("no check today")),
    )
    cp3 = MS.ComputeProcess(; project, julia, boot = a.boot)
    out = Pipe()
    ok = redirect_stdout(out) do
        MS.run_check(app, cp3; managed = false, update = false)
    end
    close(out.in)
    @test !ok && occursin("no check today", read(out, String))
end
