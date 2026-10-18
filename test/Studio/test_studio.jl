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

    # Named in the environment, an interpreter is taken at its word too.
    withenv("MFHSTUDIO_PYTHON" => "/opt/python/bin/python3") do
        @test MeanFieldHomogenization._find_python(nothing) == ["/opt/python/bin/python3"]
    end
    # Discovered on PATH, it is probed, and every candidate that fails is named
    # in the error with the last line it printed.
    if Sys.isunix()
        mktempdir() do bin
            fake = joinpath(bin, "python3")
            write(fake, "#!/bin/sh\necho 'ImportError: no mfhstudio here' 1>&2\nexit 1\n")
            chmod(fake, 0o755)
            withenv("MFHSTUDIO_PYTHON" => nothing, "PATH" => bin) do
                err = try
                    MeanFieldHomogenization._find_python(nothing)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("python3", err.msg) && occursin("ImportError: no mfhstudio here", err.msg)
            end
            # With nothing on PATH, the error says so.
            withenv("MFHSTUDIO_PYTHON" => nothing, "PATH" => mktempdir()) do
                @test_throws r"nothing named `python`" MeanFieldHomogenization._find_python(nothing)
            end
        end
    end
    # A working interpreter on PATH is found and returned with its full path.
    if py !== nothing
        withenv("MFHSTUDIO_PYTHON" => nothing) do
            argv = MeanFieldHomogenization._find_python(nothing)
            @test isfile(first(argv))
        end
    end

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
