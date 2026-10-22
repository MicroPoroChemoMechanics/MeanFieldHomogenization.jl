# The compute environment: where it lives, what it lacks, the script that
# builds it, and what is done (or refused) with a user's own environment.

spec = MS.EnvironmentSpec(["Alpha" => "0.3", "Beta" => "1"])

@test MS.environment_dir("@chemistrylab-studio") == joinpath(first(DEPOT_PATH), "environments", "chemistrylab-studio")
@test MS.environment_dir("some/dir") == abspath("some/dir")

script = MS.environment_script("@studio-test", spec; update = false)
@test !any(a -> a isa Expr && a.head in (:error, :incomplete), Meta.parseall(script).args)
@test occursin("Pkg.activate(\"studio-test\"; shared = true)", script)
@test occursin("Pkg.PackageSpec(name = \"Alpha\", version = Pkg.Types.semver_spec(\"0.3\"))", script)
@test occursin("Pkg.compat(\"Beta\", \"1\")", script)
@test !occursin("Pkg.update()", script)
@test occursin("Pkg.update()", MS.environment_script("/tmp/x", spec; update = true))
@test occursin("shared = false", MS.environment_script("/tmp/x", spec; update = false))

mktempdir() do dir
    @test MS.missing_packages(dir, spec) == ["Alpha", "Beta"]
    write(joinpath(dir, "Project.toml"), "[deps]\nAlpha = \"00000000-0000-0000-0000-000000000001\"\n")
    @test MS.missing_packages(dir, spec) == ["Beta"]
    lines = String[]
    log = line -> push!(lines, line)
    julia = first(Base.julia_cmd().exec)
    # a user's environment is checked, never modified
    e = try
        MS.prepare_environment(dir, spec; managed = false, update = false, julia, log)
    catch err
        err
    end
    @test e isa ArgumentError && occursin("lacks Beta", e.msg)
    write(joinpath(dir, "Project.toml"), "[deps]\nAlpha = \"00000000-0000-0000-0000-000000000001\"\nBeta = \"00000000-0000-0000-0000-000000000002\"\n")
    @test MS.prepare_environment(dir, spec; managed = false, update = false, julia, log) == dir
    # a managed environment that is complete is left alone
    write(joinpath(dir, "Manifest.toml"), "")
    @test MS.prepare_environment(dir, spec; managed = true, update = false, julia, log) == dir
    @test isempty(lines)
end

# The studio's own package, as the compute process must hold it: at its version
# when installed from a registry, from its folder when developed (a test run
# works on a clone).
own = MS.PackageSource(MeanFieldHomogenization)
@test own.name == "MeanFieldHomogenization" && own.version == pkgversion(MeanFieldHomogenization)
@test own.path === nothing || samefile(own.path, pkgdir(MeanFieldHomogenization))
installed = MS.PackageSource("Alpha", v"1.2.3", nothing)
whole_script(src) = !any(a -> a isa Expr && a.head in (:error, :incomplete), Meta.parseall(src).args)
s1 = MS.environment_script("@x", MS.EnvironmentSpec(installed, ["Beta" => "1"]); update = false)
@test whole_script(s1)
@test occursin("Pkg.add(Pkg.PackageSpec(name = \"Alpha\", version = \"1.2.3\"))", s1) && occursin("Pkg.pin(\"Alpha\")", s1)
@test occursin("d.is_pinned", s1) && occursin("Pkg.free(\"Alpha\")", s1)   # `add` would keep a previous pin
# The bounds an earlier version wrote into the environment are replaced before
# anything is added: left until after, they keep the new version of the studio's
# package out, as ChemistryLab 0.40's `OptimaSolver = "0.8"` kept 0.41.0 out.
@test occursin("for (name, bound) in [\"Beta\" => \"1\"]", s1)
@test findfirst("Pkg.compat(name, bound)", s1).start < findfirst("Pkg.add(Pkg.PackageSpec(name = \"Alpha\"", s1).start

# The bounds of the studio's environment are MeanFieldHomogenization's own
# wherever it declares one: copied by hand, a bound outlives the version it was
# written for.
own_compat = MS.TOML.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "Project.toml"))["compat"]
for (name, bound) in MS.ENVIRONMENT.packages
    haskey(own_compat, name) && @test bound == own_compat[name]
end
@test any(p -> first(p) == "TensND", MS.ENVIRONMENT.packages)
mktempdir() do clone
    developed = MS.PackageSource("Alpha", v"1.2.3", clone)
    s2 = MS.environment_script("@x", MS.EnvironmentSpec(developed, ["Beta" => "1"]); update = false)
    @test whole_script(s2) && occursin("Pkg.develop(path = $(repr(clone)))", s2) && !occursin("Pkg.pin", s2) && !occursin("Pkg.free", s2)
    @test MS.package_names(MS.EnvironmentSpec(developed, ["Beta" => "1"])) == ["Alpha", "Beta"]
    mktempdir() do dir
        manifest(entry) = write(joinpath(dir, "Manifest.toml"), "[[deps.Alpha]]\nuuid = \"00000000-0000-0000-0000-000000000001\"\n$entry\n")
        @test MS.parent_mismatch(dir, installed) === nothing              # no manifest yet
        manifest("version = \"1.2.3\"")
        @test MS.parent_mismatch(dir, installed) === nothing
        @test occursin("does not develop Alpha", MS.parent_mismatch(dir, developed))
        manifest("version = \"1.2.2\"")
        @test occursin("holds Alpha 1.2.2", MS.parent_mismatch(dir, installed))
        manifest("path = $(repr(clone))")
        @test MS.parent_mismatch(dir, developed) === nothing
        @test occursin("develops Alpha from", MS.parent_mismatch(dir, installed))
        write(joinpath(dir, "Manifest.toml"), "")
        @test MS.parent_mismatch(dir, installed) === nothing              # the package not there yet
        # a user's environment holding another version is refused, by name
        write(joinpath(dir, "Project.toml"), "[deps]\nAlpha = \"00000000-0000-0000-0000-000000000001\"\nBeta = \"00000000-0000-0000-0000-000000000002\"\n")
        manifest("version = \"1.2.2\"")
        e = try
            MS.prepare_environment(dir, MS.EnvironmentSpec(installed, ["Beta" => "1"]); managed = false, update = false, julia = "julia", log = l -> nothing)
        catch err
            err
        end
        @test e isa ArgumentError && occursin("does not hold the studio's own package", e.msg) && occursin("1.2.2", e.msg)
    end
end

# a managed environment is created in a process of its own: a standard
# library stands for the packages, so that no download is needed. The session's
# load path leaves out the standard libraries, as under `Pkg.test`: the process
# must start from Julia's default one, or it cannot even load `Pkg`.
mktempdir() do dir
    lines = String[]
    julia = first(Base.julia_cmd().exec)
    spec1 = MS.EnvironmentSpec(["Printf" => "1"])
    @test withenv("JULIA_LOAD_PATH" => "@") do
        MS.prepare_environment(dir, spec1; managed = true, update = false, julia, log = l -> push!(lines, l))
    end == dir
    @test occursin("Preparing the compute environment", first(lines))
    @test isempty(MS.missing_packages(dir, spec1))
    @test occursin("Printf = \"1\"", read(joinpath(dir, "Project.toml"), String))
    empty!(lines)
    MS.prepare_environment(dir, spec1; managed = true, update = true, julia, log = l -> push!(lines, l))
    @test occursin("Updating the compute environment", first(lines))
end

# a managed environment follows the studio's own package: developed from one
# clone, then from another, as after moving the clone (two copies of a package
# of one module stand for it, so that nothing is downloaded)
mktempdir() do root
    clones = map(("a", "b")) do c
        dir = mkpath(joinpath(root, c, "src"))
        write(joinpath(dir, "Gamma.jl"), "module Gamma end\n")
        write(joinpath(root, c, "Project.toml"), "name = \"Gamma\"\nuuid = \"9a7c3b52-6a0e-4b8e-9d55-2f1f3c0d4e61\"\nversion = \"0.1.0\"\n")
        joinpath(root, c)
    end
    env = joinpath(root, "env")
    lines = String[]
    julia = first(Base.julia_cmd().exec)
    specs = [MS.EnvironmentSpec(MS.PackageSource("Gamma", v"0.1.0", c), ["Printf" => "1"]) for c in clones]
    MS.prepare_environment(env, specs[1]; managed = true, update = false, julia, log = l -> push!(lines, l))
    @test MS.parent_mismatch(env, specs[1].parent) === nothing
    @test occursin("does not develop Gamma", MS.parent_mismatch(env, specs[2].parent))
    empty!(lines)
    MS.prepare_environment(env, specs[2]; managed = true, update = false, julia, log = l -> push!(lines, l))
    @test occursin("Updating the compute environment", first(lines)) && occursin("does not develop Gamma", first(lines))
    @test MS.parent_mismatch(env, specs[2].parent) === nothing
end

# the output of a command reaches the log line by line; a failure names its last lines
lines = String[]
julia = first(Base.julia_cmd().exec)
MS.run_logged(`$julia --startup-file=no -e 'println("one"); println("two")'`, l -> push!(lines, l))
@test lines == ["one", "two"]
e = try
    MS.run_logged(`$julia --startup-file=no -e 'println("last words"); exit(3)'`, l -> nothing)
catch err
    err
end
@test e isa ErrorException && occursin("status 3", e.msg) && occursin("last words", e.msg)
