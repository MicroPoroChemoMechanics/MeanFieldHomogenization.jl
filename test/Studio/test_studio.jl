# MFH Studio launcher — tests that never need Python or a running server:
# the launcher is a pure `Cmd` builder, and building it is all that can be
# asserted without spawning the studio.

@testset "MFH Studio launcher" begin
    dir = MeanFieldHomogenization._studio_dir()
    @test isdir(dir)
    @test isfile(joinpath(dir, "mfhstudio", "__main__.py"))

    # Explicit interpreter wins over anything on PATH, and is taken at its word.
    @test MeanFieldHomogenization._find_python("/usr/bin/python3") == ["/usr/bin/python3"]

    # A discovered interpreter is probed before it is used: the studio has to
    # be importable from the studio directory, which is what `-m` will do.
    py = Sys.which("python3")
    if py !== nothing
        ok, _ = MeanFieldHomogenization._python_probe([py])
        @test ok
    end
    # An interpreter that is not there fails the probe rather than the spawn.
    bad, _ = MeanFieldHomogenization._python_probe(["mfhstudio-no-such-python"])
    @test !bad

    # The last line of a Python traceback is the part worth reporting.
    @test MeanFieldHomogenization._last_line(
        "Traceback (most recent call last):\n  File \"x\"\nModuleNotFoundError: nope\n",
    ) == "ModuleNotFoundError: nope"
    @test MeanFieldHomogenization._last_line("   \n\n") == "no output"

    # The default command is `python3 -m mfhstudio --host <host> --port <port>`
    # run from the studio directory, so `-m mfhstudio` resolves.
    cmd = MeanFieldHomogenization._studio_cmd(python = "python3")
    @test cmd.exec == ["python3", "-m", "mfhstudio", "--host", "127.0.0.1", "--port", "8765"]
    @test cmd.dir == dir

    # Every option is forwarded, and only when set.
    cmd = MeanFieldHomogenization._studio_cmd(;
        host = "0.0.0.0", port = 9000, no_browser = true,
        project = "@mfhstudio", julia = "/opt/julia/bin/julia",
        check = true, python = "python3",
    )
    @test cmd.exec == [
        "python3", "-m", "mfhstudio",
        "--host", "0.0.0.0", "--port", "9000",
        "--no-browser", "--project", "@mfhstudio", "--julia", "/opt/julia/bin/julia",
        "--check",
    ]

    # `mfhstudio` is the exported, public name.
    @test mfhstudio == MeanFieldHomogenization.mfhstudio
end
