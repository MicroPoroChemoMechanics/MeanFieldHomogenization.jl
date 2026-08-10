# docs/literate.jl
#
# Runs Literate.jl over the `scripts/` demos that are published as tutorial
# pages. Produces, per script, three artifacts:
#
#   - a Documenter-ready markdown page  -> docs/src/tutorials/generated/
#     (executed by Documenter itself via `@example` when `makedocs` runs —
#     Literate's `markdown(...; documenter=true)` leaves `execute=false`)
#   - a pre-run Jupyter notebook        -> docs/generated_notebooks/
#   - a cleaned standalone .jl script   -> docs/generated_scripts/
#     (markup-only lines stripped, `#jl` directives resolved)
#
# Called from `docs/make.jl` *before* `makedocs`, so the generated markdown
# exists when Documenter's `pages` list references it.
#
# There is no longer a separate "Gallery" section: a page generated from a
# script and a page written by hand are both just tutorials, and the reader has
# no reason to care which is which. What decides whether a script is published
# is the same as before — does it cover a topic no other tutorial does — with
# the classification kept in `Assets/plans/MFH_LITERATE_SCRIPTS.md`. Scripts
# that duplicate an existing tutorial stay plain scripts, and SymPy-heavy ones
# are never published (they are re-executed on every docs build).
#
# `PUBLISHED_SCRIPTS` maps each script to its **page name**. Scripts keep their
# numeric prefixes (they encode a running order), while pages carry thematic
# names, so inserting a tutorial never forces a renumbering. The mapping is what
# Literate's `name` option is for.

using Literate

const SCRIPTS_DIR = joinpath(@__DIR__, "..", "scripts")
const TUTORIAL_MD_DIR = joinpath(@__DIR__, "src", "tutorials", "generated")
const NOTEBOOK_DIR = joinpath(@__DIR__, "generated_notebooks")
const CLEAN_SCRIPT_DIR = joinpath(@__DIR__, "generated_scripts")

const PUBLISHED_SCRIPTS = [
    "02_hill_elasticity.jl" => "hill_tensors",
    "30_average_nlayers.jl" => "layered_sphere",
    "32_spheroid_effective_conductivity.jl" => "layered_spheroid_effective",
    "33_laminate_basics.jl" => "laminate",
    "34_laminate_interfaces.jl" => "laminate_interfaces",
    "36_laminate_multiscale.jl" => "laminate_multiscale",
    "39_laminate_alv.jl" => "laminate_alv",
    "35_spheroid_interfaces.jl" => "layered_spheroid_interfaces",
    "37_spheroid_hc_conductivity.jl" => "layered_spheroid_hc",
    "43_secant_elastoplasticity.jl" => "secant_elastoplasticity",
    "59_alv_sensitivities.jl" => "alv_sensitivities",
    "61_freq_vs_time.jl" => "freq_vs_time",
    "62_alv_schemes.jl" => "alv_schemes",
    "70_symmetrization_showcase.jl" => "symmetrization",
    "80_custom_inclusion_contract.jl" => "custom_inclusion_contract",
    "84_neural_inclusion_ellipsoid.jl" => "neural_inclusion",
    "85_neural_excentered_sphere.jl" => "neural_excentered_sphere",
    "86_crack_distributions.jl" => "crack_distributions",
    "87_ageing_ages_aspect.jl" => "ageing_ages_aspect",
    "91_cluster_cubic_arrays.jl" => "cluster_model",
    "93_eim_disk_assembly_2d.jl" => "eim_assembly",
    "96_nano_spheroids.jl" => "nano_spheroids",
]

# `81_fe_crack_eshelby.jl`, `82_fe_crack_schemes.jl` and
# `83_fe_excentered_sphere.jl` are deliberately *not* published: they mesh and
# factorize (up to a 2·10⁵-dof system, several times over, for the crack), so a
# gallery page would add minutes to every documentation build and pull
# `gmsh_jll` into the docs environment.  They are standalone scripts, run from
# `scripts/fe/`, and their content is covered by the hand-written manual pages
# `manual/fe_inclusions.md` and
# `applications/recycled_aggregate.md`.

function build_tutorial_pages()
    mkpath(TUTORIAL_MD_DIR)
    mkpath(NOTEBOOK_DIR)
    mkpath(CLEAN_SCRIPT_DIR)
    for (script, page) in PUBLISHED_SCRIPTS
        src = joinpath(SCRIPTS_DIR, script)
        Literate.markdown(src, TUTORIAL_MD_DIR; documenter = true, name = page)
        Literate.notebook(src, NOTEBOOK_DIR; name = page)
        Literate.script(src, CLEAN_SCRIPT_DIR; name = page)
    end
    return nothing
end

build_tutorial_pages()
