# Starting a studio: the address and the remote-session hint, the browser
# command, the port fallback, a handle that stops cleanly, and a blocking run
# stopped by an interrupt.

@testset "remote session and browser" begin
    withenv("SSH_CONNECTION" => "1 2 3 4", "VSCODE_IPC_HOOK_CLI" => nothing, "BROWSER" => nothing) do
        @test MS.is_remote_session()
        lines = MS.banner("Studio", "http://127.0.0.1:9000/", 9000)
        @test length(lines) == 2 && occursin("ssh -L 9000:127.0.0.1:9000", lines[2])
        @test MS.browser_command("http://x/") === nothing
        @test !MS.open_browser("http://x/")
    end
    withenv("SSH_CONNECTION" => nothing, "VSCODE_IPC_HOOK_CLI" => nothing, "BROWSER" => "mybrowser --new-tab") do
        @test !MS.is_remote_session()
        @test length(MS.banner("Studio", "http://127.0.0.1:9000/", 9000)) == 1
        @test MS.browser_command("http://x/").exec == ["mybrowser", "--new-tab", "http://x/"]
    end
    withenv("SSH_CONNECTION" => nothing, "VSCODE_IPC_HOOK_CLI" => nothing, "BROWSER" => nothing) do
        cmd = MS.browser_command("http://x/")
        @test last(cmd.exec) == "http://x/"
        @test first(cmd.exec) == (Sys.isapple() ? "open" : Sys.iswindows() ? "rundll32" : "xdg-open")
        # the opener of each system, whichever runs the tests
        @test MS.browser_command("http://x/?a&b"; kernel = :Darwin).exec == ["open", "http://x/?a&b"]
        @test MS.browser_command("http://x/?a&b"; kernel = :NT).exec == ["rundll32", "url.dll,FileProtocolHandler", "http://x/?a&b"]
        @test MS.browser_command("http://x/?a&b"; kernel = :Linux).exec == ["xdg-open", "http://x/?a&b"]
    end
    if Sys.isunix()
        withenv("BROWSER" => "true") do
            @test MS.open_browser("http://x/")
        end
        withenv("BROWSER" => "/no/such/browser") do
            @test !MS.open_browser("http://x/")
        end
    end
    @test MS.url_host("0.0.0.0") == "localhost"
    @test MS.url_host("::1") == "[::1]"
    @test MS.url_host("127.0.0.1") == "127.0.0.1"
end

lacking = mktempdir()     # an environment without the packages: the boot fails at once
newsession(port) = MS.Session(
    MS.APP, "p"^32, "127.0.0.1", port,
    MS.ComputeProcess(; project = lacking, julia = first(Base.julia_cmd().exec), boot = "none.jl"),
    MS.StudioState(), false,
)

# A stand-in for the wrapper of the socket layer, which carries the system error
# in its `err` field.
struct _WrappedBindError <: Exception
    err::Exception
end

@testset "port taken" begin
    blocker = HTTP.serve!(_ -> HTTP.Response(200), "127.0.0.1", 0)
    taken = parse(Int, last(split(HTTP.server_addr(blocker), ':')))
    s = newsession(taken)
    e = try
        MS.serve(s, HTTP.Router())
    catch err
        err
    end
    @test e isa ArgumentError && occursin("`port = $(taken + 1)`", e.msg)
    # A port the system reserves (Windows) is skipped like a busy one; another
    # failure to bind is not hidden.
    # The socket layer wraps the system error in an error with an `err` field.
    wrapped(errnum) = HTTP.ConnectError("127.0.0.1:49682", _WrappedBindError(SystemError("bind", errnum)))
    @test MS.port_unavailable(wrapped(Libc.EACCES))
    @test MS.port_unavailable(wrapped(10013))
    @test MS.port_unavailable(HTTP.ConnectError("127.0.0.1:49682", SystemError("bind", Libc.EACCES)))
    @test !MS.port_unavailable(wrapped(Libc.ENOENT))
    @test !MS.port_unavailable(HTTP.ConnectError("127.0.0.1:49682", ArgumentError("no system error")))
    @test !MS.port_unavailable(ArgumentError("not a bind failure"))
    s = newsession(taken)
    server = redirect_stdout(devnull) do
        MS.serve(s, HTTP.Router(); tries = 5)
    end
    @test s.port > taken
    HTTP.forceclose(server)
    HTTP.forceclose(blocker)
end

@testset "a studio that cannot boot" begin
    h = redirect_stdout(devnull) do
        withenv("MFHSTUDIO_PROJECT" => nothing) do
            mfhstudio(; port = 0, no_browser = true, wait = false, project = lacking)
        end
    end
    @test occursin("?token=", MS.studio_url(h))
    shown = sprint(show, h)
    @test occursin("MFH Studio", shown) && !occursin("token", shown)
    for _ in 1:200
        MS.status(h.session.compute).state == "failed" && break
        sleep(0.05)
    end
    @test MS.status(h.session.compute).state == "failed"
    @test occursin("lacks", MS.status(h.session.compute).message)
    # the page loads, with the token of the address
    port = h.session.port
    token = last(split(MS.studio_url(h), "token="))
    @test HTTP.get("http://127.0.0.1:$port/api/status", ["X-Studio-Token" => token]).status == 200
    close(h)
    close(h)                        # a second close does nothing
    @test h.closed
    @test occursin("closed", sprint(show, h))
end

@testset "a user's environment is checked, not modified" begin
    e = try
        redirect_stdout(devnull) do
            mfhstudio(; check = true, project = lacking)
        end
    catch err
        err
    end
    @test e isa ArgumentError && occursin("lacks MeanFieldHomogenization", e.msg)
    # the environment variable names the environment just as the keyword does
    e = try
        redirect_stdout(devnull) do
            withenv(() -> mfhstudio(; check = true), "MFHSTUDIO_PROJECT" => lacking)
        end
    catch err
        err
    end
    @test e isa ArgumentError && occursin(lacking, e.msg)
end

@testset "a blocking run stops on an interrupt" begin
    result = redirect_stdout(devnull) do
        task = @async mfhstudio(; port = 0, no_browser = true, project = lacking)
        sleep(2)
        schedule(task, InterruptException(); error = true)
        fetch(task)
    end
    @test result === nothing
end

# A Ctrl-C that lands in another task of the studio than the waiting loop: the
# task records it and goes on (a reader) or stops (the boot, a request), and the
# waiting loop raises it.
mutable struct OnceInterrupting <: IO
    lines::Vector{String}
    thrown::Bool
end
Base.eof(io::OnceInterrupting) = (io.thrown || (io.thrown = true; throw(InterruptException())); isempty(io.lines))
Base.readline(io::OnceInterrupting; keep::Bool = false) = popfirst!(io.lines)

@testset "a Ctrl-C another task received" begin
    cp = MS.ComputeProcess(; project = lacking, julia = first(Base.julia_cmd().exec), boot = "")
    MS.drain!(cp, OnceInterrupting(["one", "two"], false))
    @test MS.status(cp).log == ["one", "two"]
    @test Threads.atomic_xchg!(MS.INTERRUPTED, false)
    MS.boot_failed!(cp, InterruptException())
    @test MS.status(cp).state == "failed" && MS.status(cp).message == "Interrupted."
    @test Threads.atomic_xchg!(MS.INTERRUPTED, false)
    router = HTTP.Router()
    HTTP.register!(router, "GET", "/interrupted", _ -> throw(InterruptException()))
    r = MS.answer(router, HTTP.Request("GET", "/interrupted"))
    @test r.status == 503
    @test Threads.atomic_xchg!(MS.INTERRUPTED, false)
    # the waiting loop raises what was recorded, and the studio stops
    h = redirect_stdout(devnull) do
        mfhstudio(; port = 0, no_browser = true, wait = false, project = lacking)
    end
    MS.INTERRUPTED[] = true
    @test_throws InterruptException wait(h)
    @test !MS.INTERRUPTED[]
    close(h)
    wait(h)                         # a closed studio is not waited for
end

# A terminal's Ctrl-C on a studio run as a script ends it at once; its compute
# process follows (test_compute.jl). Windows has no SIGINT to send.
Sys.iswindows() || @testset "Ctrl-C stops a studio run as a script" begin
    code = "using MeanFieldHomogenization; mfhstudio(; port = 0, no_browser = true, project = $(repr(lacking)))"
    cmd = `$(Base.julia_cmd()) $(MS.coverage_flags()) --project=$(Base.active_project()) -e $code`
    log = tempname()
    p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log); wait = false)
    # the address is printed once the server listens
    for _ in 1:1200
        (occursin("token=", read(log, String)) || !process_running(p)) && break
        sleep(0.1)
    end
    @test occursin("token=", read(log, String))
    # as a user would, once the studio has started (here its boot has failed,
    # the environment lacking MeanFieldHomogenization): the process is idle, not compiling
    m = match(r"http://127\.0\.0\.1:(\d+)/\?token=(\w+)", read(log, String))
    for _ in 1:600
        r = HTTP.get("http://127.0.0.1:$(m[1])/api/status", ["X-Studio-Token" => m[2]]; status_exception = false)
        occursin("\"failed\"", String(r.body)) && break
        sleep(0.1)
    end
    sleep(1)
    kill(p, Base.SIGINT)
    t0 = time()
    while process_running(p) && time() - t0 < 60
        sleep(0.2)
    end
    stopped = !process_running(p)
    stopped || kill(p, Base.SIGKILL)
    @test stopped
    rm(log; force = true)
end

@testset "a boot that prepares its environment" begin
    # the whole boot on a managed environment holding a standard library only,
    # and a boot file that needs nothing: no download, no MeanFieldHomogenization
    mktempdir() do dir
        env = joinpath(dir, "env")
        boot = joinpath(dir, "boot.jl")
        write(boot, "const BOOTED = true\n")
        app = MS.StudioApp(;
            name = "Test Studio", version = v"0.0.1", default_port = 9999, project_variable = "X",
            default_project = "@x", environment = MS.EnvironmentSpec(["Printf" => "1"]), boot,
            web_dir = MS.WEB_DIR, new_state = () -> nothing, routes! = (r, s) -> nothing,
            warm_up! = s -> MS.set_state!(s.compute, :ready, "warmed"), check = :(String[]),
        )
        cp = MS.ComputeProcess(; project = env, julia = first(Base.julia_cmd().exec), boot)
        session = MS.Session(app, "b"^32, "127.0.0.1", 0, cp, nothing, true)
        # a load path without the standard libraries, as under `Pkg.test`: the
        # environment and the compute process start from Julia's default one
        withenv(() -> MS.boot!(session), "JULIA_LOAD_PATH" => "@")
        s = MS.status(cp)
        @test s.state == "ready" && s.message == "warmed"
        @test any(occursin("Preparing the compute environment", l) for l in s.log)
        @test MS.call!(cp, :(Main.BOOTED)).value === true
        MS.shutdown!(cp)
        @test MS.status(cp).state == "stopped"
    end
end
