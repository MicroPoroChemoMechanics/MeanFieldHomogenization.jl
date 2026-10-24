"""
    MeanFieldHomogenization.Studio

MFH Studio, a local web interface for mean-field homogenization: build the
scales of a model, its phases and its schemes with forms, see its inclusions in
3-D, run it, plot its results, and get the Julia script that does the same on
its own.

The server runs in the user's session, where MeanFieldHomogenization is
loaded; the scripts run in a separate Julia process, in an environment of its
own (`@mfh-studio`) holding this very version of MeanFieldHomogenization, which
the first launch prepares. Start it with `mfhstudio`. The files under `core/`
hold what any studio needs and never refer to homogenization; `worker/` is the
code the compute process includes; `web/` is the page; `library/` holds the
worked examples.
"""
module Studio

using ..MeanFieldHomogenization: MeanFieldHomogenization
using HTTP: HTTP
using InteractiveUtils: InteractiveUtils
using JSON: JSON
using LinearAlgebra: I, cross, normalize
using Malt: Malt
using Printf: Printf
using Random: Random
using TOML: TOML

export mfhstudio

const STUDIO_ROOT = @__DIR__
const WORKER_DIR = joinpath(STUDIO_ROOT, "worker")
const WEB_DIR = joinpath(STUDIO_ROOT, "web")

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
include("parse.jl")
include("readback.jl")
include("catalog.jl")
include("geometry.jl")
include("convert.jl")
include("library.jl")
include("api.jl")

# The bounds MeanFieldHomogenization declares, read from its own Project.toml.
# Copied by hand, a bound of the studio outlives the version it was written for:
# ChemistryLab Studio bounded OptimaSolver at "0.8" after ChemistryLab 0.41.0
# came to require 0.9.3, and its compute process never started.
const OWN_COMPAT = TOML.parsefile(joinpath(@__DIR__, "..", "..", "Project.toml"))["compat"]

"""
The packages the compute process needs: MeanFieldHomogenization itself, at the
version running (the studio and the homogenization it drives are one), and the
others with MeanFieldHomogenization's own bounds where it declares one. TensND
writes the tensors of the scripts; Plots draws their figures, and is no
dependency of MeanFieldHomogenization: its bound is the studio's.
"""
const ENVIRONMENT = EnvironmentSpec(
    PackageSource(MeanFieldHomogenization),
    ["TensND" => OWN_COMPAT["TensND"], "Plots" => "1"],
)

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
    mfhstudio(; host = "127.0.0.1", port = nothing, no_browser = false,
                project = nothing, julia = nothing, check = false, wait = true,
                update = false)

Start MFH Studio, the graphical builder of MeanFieldHomogenization models, and
block until it stops (press Ctrl-C in the REPL). With `wait = false`, return a
handle at once and keep the REPL free: `close(handle)` stops the studio.

The page builds the scales of a model, its phases and its schemes, shows its
inclusions in 3-D, writes the Julia script of the model as it changes, reads a
script back (keeping verbatim what it does not model), and runs it. The scripts
run in a separate Julia process, in an environment of their own,
`@mfh-studio`, which the first launch prepares (several minutes; the page opens
at once and shows the progress).

# Arguments
- `host`/`port` — where to listen; the address printed is
  `http://host:port/?token=…`, the token being what the page presents with
  every request (the studio runs code on request). Without `port`, the studio
  takes 8765, or the next free port when 8765 is taken (ChemistryLab Studio
  uses 8766, so both run together); a `port` given explicitly is used or
  refused.
- `no_browser` — start without opening the browser (the address is printed).
- `project` — the environment the scripts run in: a folder, or `@name` for a
  shared one. By default the studio's own `@mfh-studio`, which it creates and
  keeps; the `MFHSTUDIO_PROJECT` environment variable is honored. An
  environment named this way is only checked, never modified.
- `julia` — the Julia executable of the compute process (default: the running
  one; the `JULIA` environment variable is honored).
- `check` — prepare and check the compute environment, print what it holds,
  and stop instead of serving.
- `wait` — block until the studio stops (default), or return its handle.
- `update` — update the packages of the studio's environment before starting.

Opening an Echoes script (`.py`) translates it with `tools/echoes2mfh`, which
needs Python 3.10 or later (the `MFHSTUDIO_PYTHON` environment variable names
one); nothing else in the studio does.

On a remote machine (SSH, a VS Code remote window) the studio prints its
address: open it on your own machine, after forwarding the port
(`ssh -L 8765:127.0.0.1:8765 …`; VS Code forwards it by itself).

# Examples
```julia
using MeanFieldHomogenization
mfhstudio()                     # opens the browser on http://127.0.0.1:8765
mfhstudio(port = 9000)          # another port
mfhstudio(no_browser = true)    # stay in the terminal
h = mfhstudio(wait = false)     # keep the REPL; later `close(h)`
mfhstudio(check = true)         # diagnose the compute environment
```
"""
function mfhstudio(;
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
