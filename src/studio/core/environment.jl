# The Julia environment the compute process runs in. By default the studio
# manages a shared environment of its own (`@mfh-studio`): created at
# the first launch, with the package the studio ships with at the very version
# running, and the other packages a calculation needs within the `[compat]`
# bounds the studio is tested against. The same environment runs the scripts
# the studio writes, outside it. A project the user names instead is never
# modified: a missing package, or another version of the studio's own, is
# reported with the command that fixes it.

"""
    PackageSource(name, version, path)
    PackageSource(mod)

The package a studio ships with, as the compute process must hold it, so that
the calculations run the code the studio was written for: its `version` when
it is installed from a registry, its folder `path` when it is developed (a
clone), `nothing` otherwise.
"""
struct PackageSource
    name::String
    version::VersionNumber
    path::Union{Nothing, String}
end

function PackageSource(m::Module)
    dir = pkgdir(m)
    installed = any(d -> startswith(dir, joinpath(d, "packages") * Base.Filesystem.path_separator), DEPOT_PATH)
    return PackageSource(String(nameof(m)), pkgversion(m), installed ? nothing : dir)
end

"""
    EnvironmentSpec([parent], packages)

What the compute process needs: the package the studio ships with (`parent`,
a [`PackageSource`](@ref), at the version running), and the others as
`name => compat` pairs, the bound being the range the studio is tested against.
"""
struct EnvironmentSpec
    parent::Union{Nothing, PackageSource}
    packages::Vector{Pair{String, String}}
end
EnvironmentSpec(packages::Vector{Pair{String, String}}) = EnvironmentSpec(nothing, packages)

"The names of the packages an environment must list."
package_names(spec::EnvironmentSpec) = vcat(parent_name(spec.parent), first.(spec.packages))
parent_name(::Nothing) = String[]
parent_name(p::PackageSource) = [p.name]

"""
    DEFAULT_LOAD_PATH

The load path a child Julia of the studio starts with: Julia's default, whatever
the session's. A session run by `Pkg.test`, or with `JULIA_LOAD_PATH` set, would
otherwise pass on a path without the standard libraries, and the child could load
neither `Pkg` nor the libraries Malt's own loop needs.
"""
const DEFAULT_LOAD_PATH = "JULIA_LOAD_PATH" => join(("@", "@v#.#", "@stdlib"), Sys.iswindows() ? ';' : ':')

"""
    environment_dir(project) -> String

The directory of an environment named as for `julia --project`: a path, or
`@name` for a shared environment of the first depot.
"""
function environment_dir(project::AbstractString)
    startswith(project, "@") || return abspath(project)
    return joinpath(first(DEPOT_PATH), "environments", project[2:end])
end

"The packages of `spec` that the environment at `dir` does not list."
function missing_packages(dir::AbstractString, spec::EnvironmentSpec)
    names = package_names(spec)
    file = joinpath(dir, "Project.toml")
    isfile(file) || return names
    deps = get(TOML.parsefile(file), "deps", Dict{String, Any}())
    return filter(p -> !haskey(deps, p), names)
end

"""
    parent_mismatch(dir, parent) -> Union{Nothing, String}

What differs between the studio's own package and the one the environment at
`dir` holds: another version, a registered copy where a developed one runs, or
the reverse; `nothing` when they agree, or when the environment has no
manifest yet.
"""
parent_mismatch(dir::AbstractString, ::Nothing) = nothing
function parent_mismatch(dir::AbstractString, p::PackageSource)
    file = joinpath(dir, "Manifest.toml")
    isfile(file) || return nothing
    entries = get(get(TOML.parsefile(file), "deps", Dict{String, Any}()), p.name, Any[])
    isempty(entries) && return nothing
    entry = only(entries)
    return held_mismatch(p, p.path, get(entry, "path", nothing), entry, dir)
end
# A registered version runs: the environment must hold that version, not a clone.
function held_mismatch(p::PackageSource, ::Nothing, held, entry, dir)
    held === nothing || return "it develops $(p.name) from $held, where $(p.name) $(p.version) runs"
    version = VersionNumber(get(entry, "version", "0"))
    version == p.version || return "it holds $(p.name) $version, where $(p.name) $(p.version) runs"
    return nothing
end
# A clone runs: the environment must develop that very folder.
function held_mismatch(p::PackageSource, path::String, held, entry, dir)
    held !== nothing && samefile(abspath(dir, held), path) && return nothing
    return "it does not develop $(p.name) from $path, the copy that runs"
end

"""
    parent_lines(parent) -> Vector{String}

The lines of the environment script that put the studio's own package in it:
developed from its folder, or added at its version and pinned there. `develop`
replaces whatever the environment held, and `add` a developed copy, but `add`
keeps a pinned version: a previous pin is removed first.
"""
parent_lines(::Nothing) = String[]
parent_lines(p::PackageSource) = parent_lines(p, p.path)
parent_lines(p::PackageSource, path::String) = ["Pkg.develop(path = $(repr(path)))"]
function parent_lines(p::PackageSource, ::Nothing)
    return [
        "any(d -> d.name == $(repr(p.name)) && d.is_pinned, values(Pkg.dependencies())) && Pkg.free($(repr(p.name)))",
        "Pkg.add(Pkg.PackageSpec(name = $(repr(p.name)), version = $(repr(string(p.version)))))",
        "Pkg.pin($(repr(p.name)))",
    ]
end

"""
    environment_script(project, spec; update) -> String

The code a separate Julia process runs to create, complete or update the
environment `project`, then precompile it.
"""
function environment_script(project::AbstractString, spec::EnvironmentSpec; update::Bool)
    shared = startswith(project, "@")
    target = shared ? project[2:end] : environment_dir(project)
    adds = join(("Pkg.PackageSpec(name = $(repr(p)), version = $(repr(c)))" for (p, c) in spec.packages), ", ")
    bounds = join(("Pkg.compat($(repr(p)), $(repr(c)))" for (p, c) in spec.packages), "\n")
    return """
    import Pkg
    Pkg.activate($(repr(target)); shared = $shared)
    $(join(parent_lines(spec.parent), "\n"))
    Pkg.add([$adds])
    $bounds
    $(update ? "Pkg.update()" : "")
    Pkg.precompile()
    """
end

"""
    run_logged(cmd, log) -> Nothing

Run `cmd`, passing each line it prints to `log`; an error naming the last lines
when it fails.
"""
function run_logged(cmd::Cmd, log::F) where {F}
    out = Pipe()
    proc = run(pipeline(cmd; stdout = out, stderr = out); wait = false)
    close(out.in)
    tail = String[]
    for line in eachline(out)
        log(line)
        push!(tail, line)
        length(tail) > 20 && popfirst!(tail)
    end
    wait(proc)
    success(proc) || error("the command exited with status $(proc.exitcode):\n" * join(tail, "\n"))
    return nothing
end

"""
    prepare_environment(project, spec; managed, update, julia, log) -> String

Make sure the environment `project` holds the packages of `spec`, and return
its directory. A `managed` environment (the studio's own) is created or
completed in a separate process, its output passed to `log`; it is left as it
is when complete, unless `update`. An environment the user named is only
checked.
"""
function prepare_environment(
        project::AbstractString, spec::EnvironmentSpec;
        managed::Bool, update::Bool, julia::AbstractString, log::F,
    ) where {F}
    dir = environment_dir(project)
    lacking = missing_packages(dir, spec)
    mismatch = parent_mismatch(dir, spec.parent)
    if !managed
        isempty(lacking) || throw(
            ArgumentError(
                "the environment $project lacks $(join(lacking, ", ")). Add them " *
                    "(`] activate $project`, then `] add $(join(lacking, ' '))`), or start " *
                    "without `project` to let the studio manage an environment of its own."
            )
        )
        mismatch === nothing || throw(
            ArgumentError(
                "the environment $project does not hold the studio's own package: $mismatch. " *
                    "Start without `project` to let the studio manage an environment of its own."
            )
        )
        return dir
    end
    complete = isempty(lacking) && mismatch === nothing && isfile(joinpath(dir, "Manifest.toml"))
    (complete && !update) && return dir
    log(
        update ? "Updating the compute environment $project." :
            mismatch !== nothing ? "Updating the compute environment $project: $mismatch." :
            "Preparing the compute environment $project: this happens once and takes several minutes."
    )
    script = environment_script(project, spec; update)
    cmd = addenv(
        `$julia --startup-file=no -e $script`,
        "JULIA_NUM_PRECOMPILE_TASKS" => "1", "OPENBLAS_NUM_THREADS" => "1", DEFAULT_LOAD_PATH,
    )
    run_logged(cmd, log)
    return dir
end
