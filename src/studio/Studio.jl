"""
    MeanFieldHomogenization.Studio

MFH Studio, a local web interface for mean-field homogenization: build the
scales of a model, its phases and its schemes with forms, see its inclusions in
3-D, run it, plot its results, and get the Julia script that does the same on
its own.

The server runs in the user's session, where MeanFieldHomogenization is
loaded; the scripts run in a separate Julia process, in an environment of its
own (`@mfh-studio`) holding this very version of MeanFieldHomogenization, which
the first launch prepares. The files under `core/` hold what any studio needs
and never refer to homogenization; `worker/` is the code the compute process
includes.
"""
module Studio

using ..MeanFieldHomogenization: MeanFieldHomogenization
using HTTP: HTTP
using JSON: JSON
using Malt: Malt
using Printf: Printf
using Random: Random
using TOML: TOML

const STUDIO_ROOT = @__DIR__
const WORKER_DIR = joinpath(STUDIO_ROOT, "worker")
# The page of the studio, served from the tool's folder until it moves here.
const WEB_DIR = normpath(STUDIO_ROOT, "..", "..", "tools", "mfhstudio", "web")

"The version of the studio: that of MeanFieldHomogenization, which ships it."
const STUDIO_VERSION = pkgversion(MeanFieldHomogenization)

include("core/memory.jl")
include("core/browser.jl")
include("core/environment.jl")
include("core/compute.jl")
include("core/server.jl")
include("core/launcher.jl")

include("pyformat.jl")
include("model.jl")
include("codegen.jl")

"""
The packages the compute process needs: MeanFieldHomogenization itself, at the
version running (the studio and the homogenization it drives are one), and the
others with the bounds the studio is tested against. TensND writes the tensors
of the scripts; Plots draws their figures.
"""
const ENVIRONMENT = EnvironmentSpec(
    PackageSource(MeanFieldHomogenization),
    ["TensND" => "0.6", "Plots" => "1"],
)

"""
    StudioState()

The state of a session.
"""
mutable struct StudioState
    const lock::ReentrantLock
end

StudioState() = StudioState(ReentrantLock())

"Register the routes of MFH Studio."
mfh_routes!(router::HTTP.Router, session::Session) = nothing

"""
    warm_up!(session)

Load the package in the compute process, so that its compilation is paid before
the user's first run.
"""
function warm_up!(session::Session)
    cp = session.compute
    set_state!(cp, :ready, "Loading MeanFieldHomogenization in the compute process.")
    loaded = call!(cp, :(Main.StudioWorker.check_report()))
    loaded.stopped === :none || return nothing
    set_state!(cp, :ready)
    return nothing
end

const APP = StudioApp(;
    name = "MFH Studio",
    version = STUDIO_VERSION,
    default_port = 8765,
    project_variable = "MFHSTUDIO_PROJECT",
    default_project = "@mfh-studio",
    environment = ENVIRONMENT,
    boot = joinpath(WORKER_DIR, "boot.jl"),
    web_dir = WEB_DIR,
    new_state = StudioState,
    routes! = mfh_routes!,
    warm_up! = warm_up!,
    check = :(Main.StudioWorker.check_report()),
)

"""
    studio(; host = "127.0.0.1", port = nothing, no_browser = false,
             project = nothing, julia = nothing, check = false, wait = true,
             update = false)

Start MFH Studio in Julia; `mfhstudio` takes these options. Without `port`, the
studio listens on 8765, or on the next free port when 8765 is taken (ChemistryLab
Studio uses 8766, so both run together). The compute environment is
`@mfh-studio` unless `project` or `MFHSTUDIO_PROJECT` names another, which is
then only checked.
"""
function studio(;
        host::AbstractString = "127.0.0.1",
        port::Union{Nothing, Integer} = nothing,
        no_browser::Bool = false,
        project::Union{Nothing, AbstractString} = nothing,
        julia::Union{Nothing, AbstractString} = nothing,
        check::Bool = false,
        wait::Bool = true,
        update::Bool = false,
    )
    return run_studio(APP; host, port, no_browser, project, julia, check, wait, update)
end

end
