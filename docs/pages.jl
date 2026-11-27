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
    # Ordered as a reading path. The problems of a single heterogeneity come
    # first: the Eshelby problem and the tensors it produces, with the
    # interaction of two inclusions; the crack, its flat limit, with the
    # degenerate shapes; the layered patterns, which generalize the Eshelby
    # problem; the inclusions that are not ellipsoids, solved exactly, then
    # approximately for any shape. The homogenization schemes that turn their
    # solutions into effective properties come after, one family per page, from
    # the bounds to the N-body schemes, then the derivatives of an estimate and
    # the second moments they give. Conductivity and viscoelasticity extend the
    # whole chain, and the appendices support it in its own language. A page
    # goes in the chapter of the notion it treats, and is listed in
    # `theory/index.md`: the pre-flight of `make.jl` checks it.
    "Theory" => [
        "theory/index.md",
        "theory/notation.md",
        # What a cell of the plane stands for, and where each problem of the
        # plane is treated: a convention of every chapter, read before them.
        "theory/plane_problems.md",
        "The Eshelby problem" => [
            "theory/eshelby_problem.md",
            "theory/hill_tensors.md",
            "theory/concentration.md",
            "theory/interaction_tensors.md",
        ],
        # A crack is a degenerate ellipsoid, so it stays close to the
        # foundations; the page on degenerate shapes treats its closed forms
        # together with those of the sphere and the needle.
        "Cracks and degenerate shapes" => [
            "theory/cod_tensors.md",
            "theory/thermal_cracks.md",
            "theory/rice_crack_contributions.md",
            "theory/degenerate_shapes.md",
        ],
        # A layered sphere or a confocal spheroid is not an inclusion with a
        # Hill tensor: it is a *pattern* whose generalized Eshelby problem is
        # solved for its average concentration tensor. Any pattern admitting
        # that treatment belongs here.
        "The generalized Eshelby problem — morphological patterns" => [
            # The radial problem and its two strategies first, then the two
            # patterns it serves, each on one page.
            "theory/radial_problems.md",
            "theory/layered_sphere.md",
            "theory/layered_cylinder.md",
            "theory/layered_spheroid.md",
            "theory/layered_spheroid_elasticity.md",
            "theory/layered_ellipsoid.md",
        ],
        # Inclusions that are not ellipsoids and still have an exact solution.
        # They have no Hill tensor either, and are solved for their concentration
        # tensor: the torus by a series in the toroidal coordinates of TensND, the
        # cylindrical cavity whose section is given by a conformal map by finite
        # complex potentials. They are inclusions, whatever physics they are
        # solved in, and sit with the inclusions.
        "Inclusions beyond the ellipsoid" => [
            "theory/toroidal_inclusions.md",
            "theory/conformal_mapping.md",
        ],
        # Shapes with no closed form: the integral equation of one crack or one
        # heterogeneity, solved on a regular grid of Gaussians. The approximation
        # first, on its own, then the problems it is applied to. It closes the
        # problems of a single heterogeneity.
        "Arbitrary shapes by Gaussian approximation" => [
            "theory/gaussian_approximation.md",
            "theory/gaussian_cracks.md",
            "theory/gaussian_volume.md",
        ],
        # One page per family of schemes, each with its figure, its assumption
        # and its equations, in the order of the overview's table. The laminate
        # is solved exactly by periodic homogenization rather than estimated,
        # and the cluster model and the equivalent inclusion method need the
        # positions; they are schemes all the same, and sit with the others.
        "Homogenization schemes" => [
            "theory/homogenization.md",
            "theory/scheme_bounds.md",
            "theory/scheme_dilute.md",
            "theory/scheme_mori_tanaka.md",
            "theory/scheme_maxwell.md",
            "theory/scheme_pcw.md",
            "theory/scheme_self_consistent.md",
            "theory/differential_scheme.md",
            "theory/laminate.md",
            "theory/cluster_model.md",
            "theory/eim.md",
            "theory/multipole_expansion.md",
            "theory/multipole_expansion_spheroids.md",
            "theory/multipole_expansion_anisotropic.md",
            "theory/multipole_expansion_cracks.md",
            # Not a scheme: what every scheme's concentration tensors give
            # beyond the effective stiffness.
            "theory/levin.md",
        ],
        # What the derivatives of an estimate are and what they give: how the
        # package differentiates every estimate exactly, then the second moments
        # of the local fields, which are derivatives of the effective energy with
        # respect to the moduli of a phase (Kreher's theorem).
        "Derivatives and second moments" => [
            "theory/automatic_differentiation.md",
            "theory/second_moments.md",
        ],
        "Extension to conductivity" => [
            "theory/conductivity.md",
            # The permeability from the Stokes flow in the pores: the same
            # schemes, fed with size-dependent Brinkman inclusions.
            "theory/brinkman.md",
        ],
        # Two distinct extensions: the correspondence principle, which maps a
        # non-aging problem onto an elastic one, and the aging case, where
        # no such map exists and the Eshelby problem itself is generalized.
        "Extension to viscoelasticity" => [
            "theory/laplace_carson.md",
            "theory/viscoelasticity.md",
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
            # The routes compared (closed form, layered pattern, custom, grid,
            # mesh, network) before the page of each.
            "manual/choosing_an_inclusion.md",
            "manual/ellipsoidal_inclusions.md",
            "manual/cylindrical_inclusions.md",
            "manual/cracks.md",
            "manual/planar_cracks.md",
            "manual/layered_inclusions.md",
            "manual/custom_inclusions.md",
            "manual/brinkman_inclusions.md",
            "manual/toroidal_inclusions.md",
            "manual/conformal_holes.md",
            # Three treatments of a shape with no closed form, all through the
            # custom-inclusion interface: a grid of Gaussians, a mesh, a
            # trained network.
            "Shapes with no closed form" => [
                "manual/gaussian_inclusions.md",
                "manual/fe_inclusions.md",
                "manual/neural_inclusions.md",
            ],
        ],
        "Cells and schemes" => [
            "manual/schemes.md",
            # Two-dimensional media through the same schemes.
            "manual/plane_problems.md",
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
            "manual/rheological_models.md",
            "manual/nonaging_viscoelasticity.md",
            "manual/aging_viscoelasticity.md",
            "manual/poromechanics.md",
        ],
        "Differentiation" => [
            "manual/sensitivities.md",
        ],
        "Performance" => [
            "manual/parallelism.md",
        ],
        "Appendices" => [
            "manual/elliptic_examples.md",
        ],
    ],
    # The same chapters as Theory, in the same order, so that a tutorial is
    # found where its theory is: first steps, then the problems of a single
    # heterogeneity (the Eshelby problem, cracks, morphological patterns), then
    # the schemes that turn them into effective properties, then the extensions
    # to conductivity and viscoelasticity. Differentiation and solvers
    # follow the theory chapter on derivatives and second moments; symbolic
    # computation has no theory chapter of its own. Pages
    # under `tutorials/generated/` are built from `scripts/` by Literate (see
    # `docs/literate.jl`), an implementation detail; they sit with the others.
    "Tutorials" => [
        "tutorials/index.md",
        "First steps" => [
            "tutorials/first_estimate.md",
            "tutorials/bounds_and_schemes.md",
        ],
        "The Eshelby problem" => [
            "tutorials/generated/hill_tensors.md",
            "tutorials/generated/hill_reduction.md",
            "tutorials/generated/symmetrization.md",
            "tutorials/generated/custom_inclusion_contract.md",
            "tutorials/generated/neural_inclusion.md",
            "tutorials/generated/neural_excentered_sphere.md",
        ],
        "Cracks and degenerate shapes" => [
            "tutorials/cracks.md",
            "tutorials/generated/crack_routes.md",
            "tutorials/generated/crack_distributions.md",
            "tutorials/generated/rice_cracks.md",
            "tutorials/fe_crack.md",
        ],
        "The generalized Eshelby problem — morphological patterns" => [
            "tutorials/generated/layered_sphere.md",
            "tutorials/generated/layered_sphere_local_fields.md",
            "tutorials/generated/layered_sphere_symbolic.md",
            "tutorials/generated/layered_sphere_first_order_symbolic.md",
            "tutorials/generated/layered_sphere_radial_symbolic.md",
            "tutorials/generated/layered_cylinder.md",
            "tutorials/generated/layered_cylinder_symbolic.md",
            "tutorials/generated/layered_cylinder_first_order_symbolic.md",
            "tutorials/generated/layered_cylinder_radial_symbolic.md",
            "tutorials/generated/graded_layers_interfaces.md",
            "tutorials/generated/layered_spheroid_effective.md",
            "tutorials/generated/layered_spheroid_interfaces.md",
            "tutorials/generated/layered_spheroid_hc.md",
            # The finite-element counterpart of the three pages above,
            # calibrated against them and then taken past what they cover.
            "tutorials/axi_layered_spheroid.md",
            "tutorials/generated/nano_spheroids.md",
        ],
        # As in Theory: the laminate is solved exactly rather than estimated,
        # and the N-body assemblies need the positions; they are schemes all the
        # same. Levin's theorem closes the group, as it closes the theory one.
        "Homogenization schemes" => [
            "tutorials/porous_materials.md",
            "tutorials/differential_paths.md",
            "tutorials/generated/laminate.md",
            "tutorials/generated/laminate_interfaces.md",
            "tutorials/generated/laminate_multiscale.md",
            "tutorials/generated/multiscale_assemblies.md",
            "tutorials/levin.md",
        ],
        "Extension to conductivity" => [
            "tutorials/transport.md",
        ],
        "Extension to viscoelasticity" => [
            "tutorials/viscoelasticity.md",
            "tutorials/generated/rheological_models.md",
            "tutorials/generated/kelvin_maxwell.md",
            "tutorials/generated/laplace_inversion.md",
            "tutorials/generated/freq_vs_time.md",
            "tutorials/generated/volterra_inversion.md",
            "tutorials/generated/alv_schemes.md",
            "tutorials/generated/aging_ages_aspect.md",
            "tutorials/generated/alv_sensitivities.md",
            "tutorials/generated/laminate_alv.md",
        ],
        "Differentiation and solvers" => [
            "tutorials/sensitivities.md",
            "tutorials/generated/derivative_routes.md",
            "tutorials/strength_criteria.md",
            "tutorials/nonlinear_solvers.md",
            "tutorials/generated/secant_elastoplasticity.md",
        ],
        "Symbolic computation" => [
            "tutorials/symbolic_spheres.md",
            "tutorials/symbolic_laminate.md",
            "tutorials/symbolic_viscoelasticity.md",
        ],
    ],
    # Grouped by the material or the result, not by the machinery — a
    # reader arrives here with a subject in mind. Pages under
    # `applications/generated/` are built from `scripts/` by Literate,
    # which is an implementation detail; they sit with the others.
    "Applications" => [
        # The sections that hold whatever the material first, then the
        # materials, one section each; within a section a page comes after the
        # pages it builds on.
        "applications/index.md",
        # The four sections that hold whatever the material come first, each
        # from the simplest object to the most involved: one inclusion of any
        # shape, the interactions between several, the cracks, the coated
        # inclusions. Then transport, then the materials.
        "Inclusions of any shape" => [
            # The Hill tensor of an ellipsoid in closed form, then the shapes
            # without one: by finite elements, by the four routes, by a series
            # (the torus), by a conformal map (prismatic pores).
            "applications/generated/hill_transformation.md",
            "applications/concave_pores.md",
            "applications/generated/pore_routes.md",
            "applications/generated/insulating_torus.md",
            "applications/generated/conformal_holes.md",
        ],
        "Interactions between particles" => [
            "applications/generated/cluster_model.md",
            "applications/generated/eim_assembly.md",
            "applications/generated/multipole_spheres.md",
            "applications/generated/multipole_spheroids.md",
        ],
        "Cracks" => [
            "applications/generated/crack_contacts.md",
            "applications/generated/narrow_cracks.md",
        ],
        # A shell of uniform properties around the particle, then a graded
        # one, then the published closures set beside the exact solution, then
        # the aging case.
        "Coated inclusions and interphases" => [
            "applications/recycled_aggregate.md",
            "applications/generated/graded_interphase.md",
            "applications/generated/coated_inclusion_estimates.md",
            "applications/generated/aging_coated_spheres.md",
        ],
        "Transport in pores" => [
            "applications/generated/stokes_darcy_permeability.md",
        ],
        # Elasticity, then chemistry, then transport, then time, then failure;
        # the ITZ pages use the paste model of the strength page.
        "Cementitious materials" => [
            "applications/cement_paste.md",
            "applications/hydrating_blended_paste.md",
            "applications/ionic_hydrating_paste.md",
            "applications/cement_paste_diffusion.md",
            "applications/aging_creep.md",
            "applications/aging_basic_creep.md",
            "applications/strength.md",
            "applications/itz_concrete.md",
            "applications/generated/itz_elastic_limit.md",
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
    # Material laws at one material point: written, integrated, driven along a
    # path, and plugged into any caller of the Gauss-point contract. Before the
    # finite-element coupling, whose element loops call these laws.
    "Material laws" => [
        "material_laws/index.md",
        "material_laws/theory.md",
        "Manual" => [
            "material_laws/writing.md",
            "material_laws/laws.md",
            "material_laws/driving.md",
            "material_laws/plugging.md",
        ],
        "Laws" => [
            "material_laws/generated/ml_plasticity.md",
            "material_laws/generated/ml_damage.md",
            "material_laws/generated/ml_biot.md",
        ],
        "Homogenized laws" => [
            "material_laws/generated/ml_secant.md",
            "material_laws/generated/ml_uniform_fields.md",
            "material_laws/generated/ml_polycrystals.md",
            "material_laws/generated/ml_incremental_tangent.md",
            "material_laws/generated/ml_affine.md",
            "material_laws/generated/ml_affine_evp.md",
            "material_laws/generated/ml_quasi_elastic.md",
            "material_laws/generated/ml_crack_damage.md",
            "material_laws/generated/ml_closing_cracks.md",
            "material_laws/generated/ml_strength.md",
            "material_laws/generated/ml_frictional_strength.md",
        ],
    ],
    # MeanFieldHomogenization *inside* a finite-element code — the exact
    # opposite of `manual/fe_inclusions.md`, which is the FE solver inside
    # MeanFieldHomogenization. Kept as its own top-level section so the two
    # can never be read as a continuation of one another.
    # Sorted the way the section is read: the equations first, then how to
    # build a model with them, then worked models.
    "Finite-element coupling" => [
        "fe_coupling/index.md",
        "Getting started" => [
            "fe_coupling/quickstart.md",
            "fe_coupling/step_by_step.md",
        ],
        "Theory" => [
            "fe_coupling/scale_transition.md",
            "fe_coupling/poroelastic_coupling.md",
            "fe_coupling/permeability.md",
        ],
        "Manual" => [
            "fe_coupling/materials.md",
            "fe_coupling/fractured_rock.md",
            "fe_coupling/backends.md",
            "fe_coupling/fenicsx.md",
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
        "api/gaussian_approximation.md",
        "api/conductivity.md",
        "api/concentration.md",
        "api/layered_sphere.md",
        "api/layered_cylinder.md",
        "api/layered_spheroid.md",
        "api/layered_ellipsoid.md",
        "api/superspheres.md",
        "api/laminate.md",
        "api/interactions.md",
        "api/schemes.md",
        "api/poromechanics.md",
        "api/constitutive.md",
        "api/material_laws.md",
        "api/assemblies.md",
        "api/viscoelasticity.md",
        "api/laplace_carson.md",
        "api/sensitivities.md",
    ],
    "Nomenclature" => "nomenclature.md",
    "References" => "references.md",
]
