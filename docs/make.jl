using Documenter
using DocumenterCitations
# Renders the ```mermaid blocks of the home page. The diagram's source is the
# page itself, so the architecture picture is edited by editing text — no
# regenerated asset to keep in sync.
using DocumenterMermaid
using MeanFieldHomogenization

# GR needs a headless display driver on CI runners; without this the figures in
# the Applications pages fail to render.
ENV["GKSwstype"] = "100"

# Composite panels crop their outer labels unless the margins are explicit — a
# big enough canvas is not enough on its own. The Literate scripts each set
# their own left/bottom margins; setting the defaults once here extends the same
# treatment to the hand-written `@example` pages, which never picked up that
# idiom. Must come before `literate.jl`, whose notebook pass runs the scripts.
using Plots
Plots.gr()
Plots.default(;
    left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
    right_margin = 4Plots.mm,
    top_margin = 3Plots.mm,
)

# Generates the Gallery pages (+ companion notebooks/scripts) from the
# curated `scripts/` demos before `makedocs` runs, so the generated markdown
# exists when `pages` below references it. Must run after the GKSwstype
# assignment above — the Literate notebook pass actually executes the
# scripts, including their Plots/GR calls.
include("literate.jl")

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :numeric,
)

DocMeta.setdocmeta!(
    MeanFieldHomogenization,
    :DocTestSetup,
    :(using MeanFieldHomogenization);
    recursive = true,
)

makedocs(;
    clean = false,
    modules = [
        MeanFieldHomogenization,
        MeanFieldHomogenization.Elliptic,
        MeanFieldHomogenization.Core,
        MeanFieldHomogenization.Elasticity,
        MeanFieldHomogenization.Cracks,
        MeanFieldHomogenization.Conductivity,
        MeanFieldHomogenization.LayeredSpheres,
        MeanFieldHomogenization.LayeredSpheroids,
        MeanFieldHomogenization.Interactions,
        MeanFieldHomogenization.Schemes,
        MeanFieldHomogenization.Assemblies,
        MeanFieldHomogenization.Laminates,
        MeanFieldHomogenization.Poromechanics,
        MeanFieldHomogenization.Constitutive,
        MeanFieldHomogenization.Viscoelasticity,
        MeanFieldHomogenization.CustomInclusions,
        MeanFieldHomogenization.FiniteElements,
        MeanFieldHomogenization.NeuralInclusions,
    ],
    remotes = nothing,
    authors = "Jean-François Barthélémy",
    sitename = "MeanFieldHomogenization.jl",
    format = Documenter.HTML(;
        canonical = "https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl",
        repolink = "https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/custom.css", "assets/mathjax-flash.js"],
        prettyurls = (get(ENV, "CI", nothing) == "true"),
        collapselevel = 1,
        mathengine = Documenter.MathJax3(),
        # The interactive Plotly 3D percolation surfaces in the cement-paste
        # diffusion chapter embed their data inline, exceeding the 200 KiB
        # default; raise the ceiling for those pages.
        size_threshold = 3_000_000,
        size_threshold_warn = 1_500_000,
        # The interactive 3D surfaces embed their data as inline HTML; allow it.
        example_size_threshold = 2_000_000,
    ),
    plugins = [bib],
    pages = [
        "Home" => "index.md",
        # Ordered as a reading path, and grouped so that the standard theory
        # comes before what is built on top of it: conventions, then the
        # Eshelby framework and the tools it produces (Hill tensor,
        # localization, the schemes), then the specializations (cracks,
        # layered inclusions, laminates, viscoelasticity), then the N-body
        # models, and finally the appendices — pages that support the rest but
        # are written in its language rather than the other way round.
        "Theory" => [
            "theory/index.md",
            "theory/notation.md",
            "Foundations" => [
                "theory/eshelby_problem.md",
                "theory/hill_tensors.md",
                "theory/localization.md",
                "theory/homogenization.md",
            ],
            "Schemes and specializations" => [
                "theory/differential_scheme.md",
                "theory/cod_tensors.md",
                "theory/thermal_cracks.md",
                "theory/layered_sphere.md",
                "theory/layered_spheroid.md",
                "theory/laminate.md",
                "theory/laplace_carson.md",
                "theory/viscoelasticity.md",
            ],
            "N-body models" => [
                "theory/interaction_tensors.md",
                "theory/cluster_model.md",
                "theory/eim.md",
            ],
            "Appendices" => [
                "theory/corrected_cell.md",
                "theory/elliptic_integrals.md",
            ],
        ],
        # Same principle: the inclusion families first, then the cells and
        # schemes that consume them, then what goes beyond elasticity.
        "Manual" => [
            "manual/installation.md",
            "Inclusions" => [
                "manual/inclusion_gallery.md",
                "manual/ellipsoidal_inclusions.md",
                "manual/cylindrical_inclusions.md",
                "manual/cracks.md",
                "manual/custom_inclusions.md",
                "manual/fe_inclusions.md",
                "manual/neural_inclusions.md",
            ],
            "Cells and schemes" => [
                "manual/schemes.md",
                "manual/particle_assemblies.md",
                "manual/laminates.md",
                "manual/multiscale.md",
            ],
            "Beyond elasticity" => [
                "manual/conductivity.md",
                "manual/viscoelasticity.md",
                "manual/rheological_models.md",
                "manual/laplace_inversion.md",
                "manual/poromechanics.md",
            ],
            "Differentiation" => [
                "manual/sensitivities.md",
            ],
            "Appendices" => [
                "manual/elliptic_examples.md",
            ],
        ],
        # One learning path, grouped by theme rather than by how the page
        # happens to be produced. Pages under `tutorials/generated/` are built
        # from `scripts/` by Literate (see `docs/literate.jl`); that is an
        # implementation detail the reader has no reason to care about, so they
        # sit alongside the hand-written ones.
        "Tutorials" => [
            "tutorials/index.md",
            "Fundamentals" => [
                "tutorials/first_estimate.md",
                "tutorials/bounds_and_schemes.md",
                "tutorials/porous_materials.md",
                "tutorials/porous_benchmark.md",
                "tutorials/transport.md",
                "tutorials/differential_paths.md",
                "tutorials/differential_loading_paths.md",
            ],
            "Inclusions, geometries and orientation" => [
                "tutorials/generated/hill_tensors.md",
                "tutorials/cracks.md",
                "tutorials/generated/crack_distributions.md",
                "tutorials/fe_crack.md",
                "tutorials/generated/layered_sphere.md",
                "tutorials/generated/layered_spheroid_effective.md",
                "tutorials/generated/layered_spheroid_interfaces.md",
                "tutorials/generated/layered_spheroid_hc.md",
                "tutorials/generated/nano_spheroids.md",
                "tutorials/generated/laminate.md",
                "tutorials/generated/laminate_interfaces.md",
                "tutorials/generated/symmetrization.md",
                "tutorials/generated/custom_inclusion_contract.md",
                "tutorials/generated/neural_inclusion.md",
                "tutorials/generated/neural_excentered_sphere.md",
            ],
            # After the inclusion families, since an N-body tutorial assumes
            # the reader knows them. `nano_spheroids` is NOT here: it condenses
            # a single particle's interface into an equivalent stiffness and
            # feeds an ordinary Mori-Tanaka — no N-body content at all.
            "Interacting particle assemblies" => [
                "tutorials/generated/cluster_model.md",
                "tutorials/generated/multiscale_assemblies.md",
                "tutorials/generated/eim_assembly.md",
            ],
            "Beyond elasticity" => [
                "tutorials/viscoelasticity.md",
                "tutorials/generated/rheological_models.md",
                "tutorials/generated/kelvin_maxwell.md",
                "tutorials/generated/laplace_inversion.md",
                "tutorials/generated/freq_vs_time.md",
                "tutorials/generated/alv_schemes.md",
                "tutorials/generated/ageing_ages_aspect.md",
                "tutorials/generated/alv_sensitivities.md",
                "tutorials/generated/laminate_alv.md",
            ],
            "Differentiation and solvers" => [
                "tutorials/sensitivities.md",
                "tutorials/strength_criteria.md",
                "tutorials/nonlinear_solvers.md",
                "tutorials/generated/secant_elastoplasticity.md",
            ],
            "Interoperability and tools" => [
                "tutorials/symbolic_spheres.md",
                "tutorials/symbolic_laminate.md",
                "tutorials/symbolic_viscoelasticity.md",
                "tutorials/generated/laminate_multiscale.md",
            ],
        ],
        "Applications" => [
            "applications/cement_paste.md",
            "applications/hydrating_blended_paste.md",
            "applications/ionic_hydrating_paste.md",
            "applications/cement_paste_diffusion.md",
            "applications/strength.md",
            "applications/itz_concrete.md",
            "applications/recycled_aggregate.md",
            "applications/bituminous.md",
            "applications/ageing_creep.md",
        ],
        # MeanFieldHomogenization *inside* a finite-element code — the exact
        # opposite of `manual/fe_inclusions.md`, which is the FE solver inside
        # MeanFieldHomogenization. Kept as its own top-level section so the two
        # can never be read as a continuation of one another.
        # Sorted the way the section is read: the equations first, then how to
        # build a model with them, then worked models.
        "Finite-element coupling" => [
            "fe_coupling/index.md",
            "Theory" => [
                "fe_coupling/scale_transition.md",
                "fe_coupling/poroelastic_coupling.md",
                "fe_coupling/permeability.md",
            ],
            "Manual" => [
                "fe_coupling/materials.md",
                "fe_coupling/fractured_rock.md",
                "fe_coupling/backends.md",
            ],
            "Examples" => [
                "fe_coupling/thick_cylinder.md",
                "fe_coupling/arma2011.md",
            ],
        ],
        # Getting work into and out of MeanFieldHomogenization. These are companions to
        # the library rather than chapters about it, which is why they sit
        # together at the end of the user-facing material instead of
        # interrupting the manual.
        "Tools and migration" => [
            "tools/from_echoes.md",
            "tools/echoes2mfh.md",
            "tools/mfhstudio.md",
        ],
        "Developer" => [
            "developer/architecture.md",
            "developer/adding_inclusion.md",
            "developer/adding_algorithm.md",
            "developer/adding_scheme.md",
            "developer/testing_conventions.md",
            "developer/validation.md",
            "developer/performance_notes.md",
            "developer/benchmarks.md",
            "developer/roadmap.md",
        ],
        "API" => [
            "api/elliptic.md",
            "api/core.md",
            "api/elasticity.md",
            "api/cracks.md",
            "api/conductivity.md",
            "api/localization.md",
            "api/layered_sphere.md",
            "api/layered_spheroid.md",
            "api/laminate.md",
            "api/interactions.md",
            "api/schemes.md",
            "api/poromechanics.md",
            "api/constitutive.md",
            "api/assemblies.md",
            "api/viscoelasticity.md",
            "api/laplace_carson.md",
            "api/sensitivities.md",
        ],
        "References" => "references.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl.git",
    devbranch = "main",
    push_preview = false,
)
