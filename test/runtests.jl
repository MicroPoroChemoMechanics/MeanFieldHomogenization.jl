using Test
using Aqua
using MeanFieldHomogenization
using Random

# Load DECUHR + Integrals so the `MeanFieldHomogenizationDECUHRExt` extension activates:
# several tests cross-validate the `method = :decuhr` path against the residue
# and nested-QuadGK backends. (DECUHR is a weak dependency of MeanFieldHomogenization.)
import DECUHR, Integrals

# Load NonlinearSolve so the `MeanFieldHomogenizationNonlinearSolveExt` extension
# activates: `test_self_consistent_nls.jl` exercises SC / ASC through
# SciML algorithms (NewtonRaphson, TrustRegion) and `AutoNonlinear`.
# (NonlinearSolve is a weak dependency of MeanFieldHomogenization.)
import NonlinearSolve

# Load Ferrite / FerriteGmsh / Gmsh so the `MeanFieldHomogenizationFerriteExt` extension
# activates: `test_ferrite_crack.jl` and `test_axi_excentered_sphere.jl`
# exercise the finite-element inclusions.  They are weak dependencies, so a
# missing stack skips those tests rather than failing the suite.
const HAS_FERRITE = try
    @eval import Ferrite, FerriteGmsh, Gmsh
    true
catch
    false
end

# The second finite-element backend.  Deliberately **not** a dependency of
# `test/Project.toml`: `GridapGmsh` pulls `GridapDistributed`,
# `PartitionedArrays` (hence `MPI`) and `Metis`, which is a lot of machinery to
# install on every CI run for a cross-check.  Add the two packages to the test
# environment locally to exercise `test_axi_gridap.jl`.
const HAS_GRIDAP = try
    @eval import Gridap, GridapGmsh
    true
catch
    false
end

# Both symbolic backends, for `test_symbolic_rheology.jl`.  They are in
# `test/Project.toml`, so this is a guard against a broken SymPyPythonCall install
# (a failed CondaPkg resolution, no network on the first load) rather than against
# a missing dependency — a `Sym` needs a working Python, which is the one thing in
# this suite that can fail for reasons outside Julia.
const HAS_SYMBOLIC_BACKENDS = try
    @eval import SymPyPythonCall, Symbolics
    true
catch
    false
end

# Load Lux / Optimisers / Zygote so the `MeanFieldHomogenizationLuxExt` extension activates.
# Only *training* a neural surrogate needs them: loading a committed model,
# running it through every scheme and differentiating it need nothing beyond the
# package, so almost all of `test_neural_inclusion.jl` runs unconditionally and
# only the short end-to-end fit is guarded by this flag.
const NN_HAS_LUX = try
    @eval import Lux, Optimisers, Zygote
    true
catch
    false
end

# Several test files draw random operators (`test_ti_alv.jl`, `test_ortho_alv.jl`,
# `test_volterra_inverse.jl`, …).  Seed once here so a CI failure is always
# reproducible locally instead of depending on the draw.
Random.seed!(20260723)

"""
    aqua_persistent_tasks(m::Module) -> Bool

`Aqua.test_persistent_tasks` in a child process with default bounds checking.

The check precompiles a synthetic package that depends on `m` and waits for a
sentinel written from that package's module body. Julia writes no
precompilation cache when `--check-bounds` is forced, so the body never runs,
the sentinel never appears, and the check fails for a reason that says nothing
about `m`. `Pkg.test()` forces the flag by default.

Running it in a child with `--check-bounds=auto` keeps both halves: the test
suite proper still runs with bounds checking on, and the check still runs.
"""
function aqua_persistent_tasks(m::Module)
    code = """
    using Aqua, $(nameof(m)), Test
    @testset "persistent_tasks" begin
        Aqua.test_persistent_tasks($(nameof(m)))
    end
    """
    cmd = `$(first(Base.julia_cmd())) --check-bounds=auto --startup-file=no --project=$(Base.active_project()) -e $code`
    return success(run(ignorestatus(cmd)))
end

# The suite runs in parts. `Pkg.test(; test_args = ["schemes"])` runs the groups
# of one part, several parts may be named, and with no argument every group runs,
# as it always has. CI runs the parts as parallel jobs: the whole suite under
# coverage instrumentation came within minutes of the job timeout and crossed it
# on slower runners. Each group belongs to exactly one part, which the last
# testset checks.
const TEST_PARTS = Dict(
    "core" => (
        "Aqua", "Elliptic", "Core", "Elasticity", "Cracks", "Conductivity",
        "Interactions", "Poromechanics", "Constitutive", "MaterialPoints", "Regression", "Studio",
        "Documentation",
    ),
    "schemes" => ("Schemes", "Assemblies", "CustomInclusions"),
    "patterns" => (
        "LayeredSpheres", "Laminates", "LayeredSpheroids", "Superspheres",
        "FiniteElementInclusions",
    ),
    "visco-neural" => ("Viscoelasticity", "NeuralInclusions"),
)
const TEST_GROUPS_SEEN = Set{String}()
function rungroup(name::String)
    push!(TEST_GROUPS_SEEN, name)
    isempty(ARGS) && return true
    for part in ARGS
        haskey(TEST_PARTS, part) || error("unknown test part \"$(part)\"; the parts are $(join(keys(TEST_PARTS), ", "))")
    end
    return any(part -> name in TEST_PARTS[part], ARGS)
end

# `verbose` prints the time of every group, passing or not, which is what the
# split of the suite into parts is balanced on.
@testset "MeanFieldHomogenization" verbose = true begin
    # Ambiguities, unbound type parameters, undefined exports, dependency
    # hygiene, type piracy and tasks left running at load. Nothing is exempted:
    # every default check passes on the package as it stands, across all
    # eighteen sub-modules and nine extensions.
    rungroup("Aqua") && @testset "Aqua" begin
        # `persistent_tasks` is run apart, see `aqua_persistent_tasks` above.
        Aqua.test_all(MeanFieldHomogenization; persistent_tasks = false)
        @test aqua_persistent_tasks(MeanFieldHomogenization)
    end

    rungroup("Elliptic") && @testset "Elliptic" begin
        include("Elliptic/test_elliptic.jl")
    end

    rungroup("Core") && @testset "Core" begin
        include("Core/test_traits.jl")
        include("Core/test_rotational_average.jl")
        include("Core/test_newton.jl")
        include("Core/test_newton_cylinder.jl")
        include("Core/test_green_dipole.jl")
        include("Core/test_green_aniso.jl")
        include("Core/test_cluster_residues.jl")
        include("Core/test_counters.jl")
        include("Core/test_hot_paths.jl")
        include("Core/test_order_aliases.jl")
        include("Core/test_random_bases.jl")
    end

    rungroup("Elasticity") && @testset "Elasticity" begin
        include("Elasticity/test_hill.jl")
        include("Elasticity/test_hill_2d.jl")
        include("Elasticity/test_hill_cylinder.jl")
        include("Elasticity/test_hill_random_bases.jl")
        include("Elasticity/test_hill_symbolic.jl")
        include("Elasticity/test_shape_tensor.jl")
        include("Elasticity/test_eshelby.jl")
        include("Elasticity/test_second_hill.jl")
        include("Elasticity/test_concentration.jl")
        include("Elasticity/test_contribution.jl")
        include("Elasticity/test_hill_nestedquadgk_oblate.jl")
        include("Elasticity/test_degenerate_shapes.jl")
        include("Elasticity/test_hill_ti_coaxial.jl")
        include("Elasticity/test_hill_residue_TI_aligned.jl")
        include("Elasticity/test_param_conversions.jl")
        include("Elasticity/test_cubic.jl")
        include("Elasticity/test_surface_stiffness.jl")
    end

    rungroup("Cracks") && @testset "Cracks" begin
        include("Cracks/test_cod.jl")
        include("Cracks/test_crack_near_circle.jl")
        include("Cracks/test_cod_ti_aligned.jl")
        include("Cracks/test_cod_symbolic.jl")
        include("Cracks/test_crack_orientation.jl")
        include("Cracks/test_conductive.jl")
        include("Cracks/test_residue_accuracy.jl")
        include("Cracks/test_H_oracle.jl")
        include("Cracks/test_thermal.jl")
        include("Cracks/test_interface_stiffness.jl")
    end

    rungroup("Conductivity") && @testset "Conductivity" begin
        include("Conductivity/test_hill_order2.jl")
        include("Conductivity/test_hill_cylinder.jl")
        include("Conductivity/test_eshelby.jl")
        include("Conductivity/test_concentration.jl")
    end

    # `Interactions` sits between the one-inclusion kernels it builds on and
    # the N-body schemes that consume it: the pair tensors must be trusted
    # before either scheme is exercised.
    rungroup("Interactions") && @testset "Interactions" begin
        include("Interactions/test_pair_tensors.jl")
    end

    # `Assemblies` carries the positional cell and the two N-body schemes.  It
    # runs after `Interactions` (whose pair tensors it consumes) and needs the
    # `Schemes` types, so it sits between them and the rest.
    rungroup("Assemblies") && @testset "Assemblies" begin
        include("Assemblies/test_assembly.jl")
        include("Assemblies/test_cluster_model.jl")
        include("Assemblies/test_eim.jl")
        include("Assemblies/test_as_rve.jl")
        include("Assemblies/test_multiscale.jl")
    end

    rungroup("Schemes") && @testset "Schemes" begin
        include("Schemes/test_rve.jl")
        include("Schemes/test_matrix_role.jl")
        include("Schemes/test_dispatch.jl")
        include("Schemes/test_voigt_reuss.jl")
        include("Schemes/test_one_shot.jl")
        include("Schemes/test_maxwell_pcw.jl")
        include("Schemes/test_self_consistent.jl")
        include("Schemes/test_sc_crack_orientation.jl")
        include("Schemes/test_self_consistent_nls.jl")
        include("Schemes/test_differential.jl")
        include("Schemes/test_complex_moduli.jl")
        include("Schemes/test_dual_compat.jl")
        include("Schemes/test_parameters.jl")
        include("Schemes/test_sensitivities.jl")
        include("Schemes/test_symmetrize.jl")
        include("Schemes/test_orientation.jl")
        include("Schemes/test_concentration_bundles.jl")
        include("Schemes/test_concentrations.jl")
        include("Schemes/test_crack_families.jl")
    end

    # The user-defined-inclusion contract spans Core (the abstractions) and
    # Schemes (every consumer), so it runs after both — the scheme kernels are
    # already compiled by then, which keeps this testset cheap.
    rungroup("CustomInclusions") && @testset "CustomInclusions" begin
        include("CustomInclusions/test_custom_inclusion.jl")
    end

    # Neural surrogates are another client of that same contract, so they run
    # straight after it. The committed models under
    # `src/NeuralInclusions/models/` are loaded, not retrained, which is what
    # keeps this testset deterministic and independent of the Lux stack.
    rungroup("NeuralInclusions") && @testset "NeuralInclusions" begin
        include("NeuralInclusions/test_neural_inclusion.jl")
        # The cubic concentration class and the surrogate that stands in for a
        # supershape cell. The teacher is synthetic -- a closed-form cubic
        # tensor -- so no mesh and no solve appear anywhere; the fit itself is
        # guarded by `NN_HAS_LUX` like the others.
        include("NeuralInclusions/test_cubic_surrogate.jl")
        # The axisymmetric counterpart: the transverse-isotropic transport
        # class, the reference medium a cavity has to state rather than have
        # guessed, the two scaling knobs, and the two shipped models. No fit
        # here at all -- the machinery around one is what was silent when wrong.
        include("NeuralInclusions/test_axi_surrogate.jl")
        # The anchored output specification: exactness on the face where its
        # baseline is the answer, and the encode/decode round trip that keeps
        # training and prediction in one space. No fit here.
        include("NeuralInclusions/test_anchored_spec.jl")
    end

    # Superspherical and superspheroidal geometry. Pure closed forms, no mesh
    # and no solver, so this runs unconditionally and in seconds — and it is
    # what everything built on those shapes assumes.
    rungroup("Superspheres") && @testset "Superspheres" begin
        include("Superspheres/test_shapes.jl")
        include("Superspheres/test_surface_mesh.jl")
    end

    # Finite-element inclusions: skipped when the Ferrite stack is unavailable
    # (it is a weak dependency), and slow when it is — the crack cases mesh a
    # ball and factorize a ~10⁵-dof system, while the axisymmetric ones are
    # two-dimensional and cost a fraction of that.
    if HAS_FERRITE && rungroup("FiniteElementInclusions")
        @testset "FiniteElementInclusions" begin
            # The octant's parity algebra and shape guard: closed form, no
            # mesh and no solve, so it runs first and costs nothing.
            include("FiniteElements/test_cell_octant_algebra.jl")
            include("FiniteElements/test_ferrite_crack.jl")
            include("FiniteElements/test_axi_excentered_sphere.jl")
            # The axisymmetric cavity: two-dimensional, so seconds.
            include("FiniteElements/test_axi_supershape_pore.jl")
            # The N-layer spheroid, against the two exact families that cross
            # the space of nested spheroids: confocal (`LayeredSpheroid`, both
            # the prolate and the oblate branch) and concentric spheres of free
            # radii (`LayeredSphere`).
            include("FiniteElements/test_axi_layered_spheroid.jl")
            # Geometry and meshing of the three-dimensional cell around a
            # supershape -- no solve, and kept at level 2, so it costs seconds.
            # It needs only gmsh, but rides the same guard.
            include("FiniteElements/test_cell_mesh.jl")
            # And the corrected solve on it, against the exact spherical pore
            # in both physics. Level 2, so about twenty-five seconds.
            include("FiniteElements/test_cell_solve.jl")
            # And the same solve on one eighth of the cell.
            include("FiniteElements/test_cell_octant.jl")
            # And the inclusion type built on it, through the schemes in both
            # physics. About thirty seconds.
            include("FiniteElements/test_supershape_pore.jl")
            if HAS_GRIDAP
                include("FiniteElements/test_gridap_backend.jl")
            else
                @info "Gridap / GridapGmsh unavailable — skipping the " *
                    "cross-backend finite-element tests."
            end
        end
    elseif rungroup("FiniteElementInclusions")
        @info "Ferrite / FerriteGmsh / Gmsh unavailable — skipping the " *
            "finite-element inclusion tests."
    end

    rungroup("LayeredSpheres") && @testset "LayeredSpheres" begin
        include("LayeredSpheres/test_bulk.jl")
        include("LayeredSpheres/test_interfaces.jl")
        include("LayeredSpheres/test_interface_jumps.jl")
        include("LayeredSpheres/test_incompressible.jl")
        include("LayeredSpheres/test_conductivity.jl")
        include("LayeredSpheres/test_christensen.jl")
        include("LayeredSpheres/test_local_fields.jl")
        include("LayeredSpheres/test_generic.jl")
        include("LayeredSpheres/test_interface_sensitivity.jl")
        include("LayeredSpheres/test_scheme_integration.jl")
    end

    # The periodic multilayer cell. Placed after the layered morphologies
    # whose interface models it reuses, and before `Viscoelasticity`, which
    # carries its aging-viscoelastic twin.
    rungroup("Laminates") && @testset "Laminates" begin
        include("Laminates/test_km_blocks.jl")
        include("Laminates/test_laminate_cell.jl")
        include("Laminates/test_laminate_oracles.jl")
        include("Laminates/test_laminate_interfaces.jl")
        include("Laminates/test_laminate_nesting.jl")
        include("Laminates/test_laminate_dual_compat.jl")
        include("Laminates/test_laminate_symbolic.jl")
    end

    rungroup("LayeredSpheroids") && @testset "LayeredSpheroids" begin
        include("LayeredSpheroids/test_legendre.jl")
        include("LayeredSpheroids/test_coupling.jl")
        include("LayeredSpheroids/test_conductivity.jl")
        include("LayeredSpheroids/test_scheme_integration.jl")
        include("LayeredSpheroids/test_legendre_stability.jl")
        include("LayeredSpheroids/test_local_fields.jl")
        include("LayeredSpheroids/test_ad.jl")
        include("LayeredSpheroids/test_pn_symbolic.jl")
        include("LayeredSpheroids/test_pn_modes.jl")
        include("LayeredSpheroids/test_elasticity.jl")
        include("LayeredSpheroids/test_elastic_cases.jl")
    end

    # `Poromechanics` post-processes a homogenized stiffness, so it only needs
    # `Schemes` to be exercised — same position as in the load order.
    rungroup("Poromechanics") && @testset "Poromechanics" begin
        include("Poromechanics/test_biot.jl")
    end

    # `Constitutive` is the Gauss-point law built on a cell + scheme, so it runs
    # after both, and after `Poromechanics` whose parameters the poroelastic
    # materials consume.
    rungroup("Constitutive") && @testset "Constitutive" begin
        include("Constitutive/test_material_api.jl")
        include("Constitutive/test_cracked.jl")
        include("Constitutive/test_poroelastic.jl")
        # The same laws as the Gauss-point law of a Ferrite computation.
        HAS_FERRITE && include("Constitutive/test_ferrite_backend.jl")
    end

    # Nonlinear laws written as materials of the same contract, so they run
    # after it.
    rungroup("MaterialPoints") && @testset "MaterialPoints" begin
        include("MaterialPoints/test_laws.jl")
        include("MaterialPoints/test_engine.jl")
        include("MaterialPoints/test_driver.jl")
        include("MaterialPoints/test_crystal.jl")
        include("MaterialPoints/test_homogenized.jl")
        include("MaterialPoints/test_crack_stiffness.jl")
        include("MaterialPoints/test_crack_damage.jl")
        include("MaterialPoints/test_closing_cracks.jl")
        include("MaterialPoints/test_strength.jl")
        include("MaterialPoints/test_frictional_strength.jl")
        # The coupling to a finite-element code, through the Ferrite adapter.
        HAS_FERRITE && include("MaterialPoints/test_fe_coupling.jl")
    end

    rungroup("Viscoelasticity") && @testset "Viscoelasticity" begin
        # The Laplace-Carson half (non-aging) first: it depends on nothing
        # from the ALV pipeline, whereas `test_rheology_iso.jl` closes the loop
        # by checking that the two routes agree.
        include("Viscoelasticity/test_laplace_inversion.jl")
        include("Viscoelasticity/test_laplace_inversion_ad.jl")
        include("Viscoelasticity/test_laplace_tensor.jl")
        include("Viscoelasticity/test_prony.jl")
        include("Viscoelasticity/test_rheology.jl")
        include("Viscoelasticity/test_rheology_iso.jl")
        HAS_SYMBOLIC_BACKENDS &&
            include("Viscoelasticity/test_symbolic_rheology.jl")
        include("Viscoelasticity/test_symmetrize_alv.jl")
        include("Viscoelasticity/test_laminate_alv.jl")
        include("Viscoelasticity/test_visco_law.jl")
        include("Viscoelasticity/test_trapezoidal.jl")
        include("Viscoelasticity/test_volterra_inverse.jl")
        include("Viscoelasticity/test_hill_alv_iso.jl")
        include("Viscoelasticity/test_schemes_alv.jl")
        include("Viscoelasticity/test_sc_alv.jl")
        include("Viscoelasticity/test_sc_alv_newton.jl")
        include("Viscoelasticity/test_layered_alv.jl")
        include("Viscoelasticity/test_layered_alv_order2.jl")
        include("Viscoelasticity/test_ti_alv.jl")
        include("Viscoelasticity/test_ortho_alv.jl")
        include("Viscoelasticity/test_ortho_dispatch_alv.jl")
        include("Viscoelasticity/test_alv_kernel_types.jl")
        include("Viscoelasticity/test_sensitivities_alv.jl")
        include("Viscoelasticity/test_order2_alv.jl")
        include("Viscoelasticity/test_extra_schemes_alv.jl")
        include("Viscoelasticity/test_crack_schemes_alv.jl")
        include("Viscoelasticity/test_alv_symmetry_paths.jl")
        include("Viscoelasticity/test_glassy_limit_alv.jl")
    end

    rungroup("Regression") && @testset "Regression" begin
        include("regression/test_hill_cases.jl")
        include("regression/test_crack_cases.jl")
        include("regression/test_anisotropic.jl")
    end

    rungroup("Studio") && @testset "Studio" begin
        include("Studio/runtests.jl")
    end

    rungroup("Documentation") && @testset "Documentation" begin
        include("Documentation/test_docs_nomenclature.jl")
    end

    # Every group above belongs to exactly one part, and every name a part lists
    # is a group above: a group added without a part would run in no CI job.
    @testset "Test parts" begin
        listed = collect(Iterators.flatten(values(TEST_PARTS)))
        @test allunique(listed)
        @test Set(listed) == TEST_GROUPS_SEEN
    end
end
