# The page tree, grouped by what a chapter is *about* rather than by the order the
# pages were written in. `make.jl` includes this file, and both the draft
# pre-flight and the real build read the same list; a partial build
# (`MFH_DOCS_ONLY`, see `partial.jl`) prunes it.
#
# The difference between the chapters is the question each one answers.
#
#   **Getting started** -- how is the package installed, and what is the
#                   smallest calculation that does something useful?
#   **Theory**    -- why is this the right estimate? Readable without running
#                   anything; every equation is the one the code evaluates.
#   **Manual**    -- how is this written? One page per kind of object, syntax
#                   first, on the smallest example that shows it.
#   **Tutorials** -- how is a calculation driven from end to end? Narrative, and
#                   each one goes somewhere.
#   **Applications** -- what does a real material look like, and what do the
#                   modeling choices cost in numbers?
#   **Tools**, **Finite-element coupling**, **Developer** -- companions to the
#                   library: migrating from Echoes, the library inside a
#                   finite-element code, and extending the library itself.
#   **API**       -- the docstrings, generated.
#   **Nomenclature** -- every symbol of the formulas, with its meaning and unit.
#
# A page that answers two of those questions is worth splitting; a page in the
# wrong chapter is worth moving.
#
# `pages` is a plain global rather than a `const`: `partial.jl` replaces it with
# the pruned tree of a partial build.

pages = [
    "Home" => "index.md",
    "Getting Started" => "quickstart.md",
    # Ordered as a reading path, and grouped so that the standard theory
    # comes before what is built on top of it: conventions, then the
    # Eshelby framework and the tools it produces (Hill tensor,
    # localization, the schemes), then the specializations (cracks,
    # layered inclusions, laminates, viscoelasticity), then the N-body
    # models, and finally the appendices — pages that support the rest but
    # are written in its language rather than the other way round.
    # Grouped by what a chapter *is about*, not by how it is derived. The
    # order follows the dependency chain: the Eshelby problem and the
    # tensors it produces, then the schemes built on them, then the three
    # ways the problem is generalized — a richer pattern in place of the
    # ellipsoid, a different physics, a different time dependence — then
    # periodic homogenization, which is a different construction entirely,
    # and finally the N-body models that drop the one-site picture.
    "Theory" => [
        "theory/index.md",
        "theory/notation.md",
        "Foundations — the Eshelby problem" => [
            "theory/eshelby_problem.md",
            "theory/hill_tensors.md",
            "theory/localization.md",
        ],
        "Homogenization schemes" => [
            "theory/homogenization.md",
            "theory/differential_scheme.md",
        ],
        # A layered sphere or a confocal spheroid is not an inclusion with a
        # Hill tensor: it is a *pattern* whose generalized Eshelby problem is
        # solved for its average concentration tensor. Any pattern admitting
        # that treatment belongs here.
        "The generalized Eshelby problem — morphological patterns" => [
            "theory/layered_sphere.md",
            "theory/layered_spheroid.md",
            "theory/layered_spheroid_elasticity.md",
        ],
        # A crack is a degenerate ellipsoid, so it stays close to the
        # foundations rather than joining the composite patterns.
        "Cracks" => [
            "theory/cod_tensors.md",
            "theory/thermal_cracks.md",
        ],
        "Extension to conductivity" => [
            "theory/conductivity.md",
        ],
        # Two distinct extensions: the correspondence principle, which maps a
        # non-ageing problem onto an elastic one, and the ageing case, where
        # no such map exists and the Eshelby problem itself is generalized.
        "Extension to viscoelasticity" => [
            "theory/laplace_carson.md",
            "theory/viscoelasticity.md",
        ],
        # NOT a morphological pattern: the laminate result comes out of
        # periodic homogenization, a construction of its own.
        "Periodic homogenization" => [
            "theory/laminate.md",
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
        "manual/index.md",
        "Inclusions" => [
            "manual/inclusion_gallery.md",
            "manual/ellipsoidal_inclusions.md",
            "manual/cylindrical_inclusions.md",
            "manual/cracks.md",
            "manual/layered_inclusions.md",
            "manual/custom_inclusions.md",
            "manual/fe_inclusions.md",
            "manual/neural_inclusions.md",
        ],
        "Cells and schemes" => [
            "manual/schemes.md",
            "manual/particle_assemblies.md",
            "manual/multiscale.md",
        ],
        # Separated from the schemes for the same reason as in Theory: a
        # laminate is the closed form of a periodic problem, not a cell
        # holding inclusions.
        "Periodic homogenization" => [
            "manual/laminates.md",
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
            "tutorials/generated/layered_sphere_local_fields.md",
            "tutorials/generated/layered_spheroid_effective.md",
            "tutorials/generated/layered_spheroid_interfaces.md",
            "tutorials/generated/layered_spheroid_hc.md",
            # The finite-element counterpart of the three pages above,
            # calibrated against them and then taken past what they cover.
            "tutorials/axi_layered_spheroid.md",
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
        #
        # `cluster_model` and `eim_assembly` used to sit here and are now
        # under Applications: each exists to reproduce one paper's numbers
        # — Molinari & El Mouden's figures and a published table — which is
        # what an application is, where a tutorial teaches the library.
        "Interacting particle assemblies" => [
            "tutorials/generated/multiscale_assemblies.md",
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
    # Grouped by the material or the result, not by the machinery — a
    # reader arrives here with a subject in mind. Pages under
    # `applications/generated/` are built from `scripts/` by Literate,
    # which is an implementation detail; they sit with the others.
    "Applications" => [
        # General concepts first, in the order they introduce one another:
        # interactions between particles, then two morphologies without a
        # closed form, computed by finite elements. The materials follow, each
        # in its own section; within a section a page comes after the pages it
        # builds on. A page may move within its section, not across chapters.
        "General concepts" => [
            # Both reproduce one paper's published numbers, which is why they
            # are applications and not tutorials.
            "applications/generated/cluster_model.md",
            "applications/generated/eim_assembly.md",
            # The Fourier reduction of the first is reused by the second.
            "applications/recycled_aggregate.md",
            "applications/concave_pores.md",
        ],
        # Elasticity, then chemistry, then transport, then time, then failure;
        # the ITZ pages use the paste model of the strength page.
        "Cementitious materials" => [
            "applications/cement_paste.md",
            "applications/hydrating_blended_paste.md",
            "applications/ionic_hydrating_paste.md",
            "applications/cement_paste_diffusion.md",
            "applications/ageing_creep.md",
            "applications/strength.md",
            "applications/itz_concrete.md",
            "applications/itz_elastic_limit.md",
        ],
        # Particles in contact through interfaces: sliding platelets, then
        # frictional contacts between rigid grains, then deformable grains.
        "Geomaterials" => [
            "applications/lamellar_clay.md",
            "applications/granular_friction.md",
            "applications/sandstone_strength.md",
        ],
        "Bituminous materials" => [
            "applications/bituminous.md",
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
        "api/superspheres.md",
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
    "Nomenclature" => "nomenclature.md",
    "References" => "references.md",
]
