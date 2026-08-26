using Test
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
# `test/Project.toml`, so this is a guard against a broken PyCall/SymPy install
# rather than against a missing dependency — a `Sym` needs a working Python,
# which is the one thing in this suite that can fail for reasons outside Julia.
const HAS_SYMBOLIC_BACKENDS = try
    @eval import SymPy, Symbolics
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

@testset "MeanFieldHomogenization" begin
    @testset "Elliptic" begin
        include("Elliptic/test_elliptic.jl")
    end

    @testset "Core" begin
        include("Core/test_traits.jl")
        include("Core/test_rotational_average.jl")
        include("Core/test_newton.jl")
        include("Core/test_newton_cylinder.jl")
        include("Core/test_green_dipole.jl")
        include("Core/test_green_aniso.jl")
    end

    @testset "Elasticity" begin
        include("Elasticity/test_hill.jl")
        include("Elasticity/test_hill_2d.jl")
        include("Elasticity/test_hill_cylinder.jl")
        include("Elasticity/test_shape_tensor.jl")
        include("Elasticity/test_eshelby.jl")
        include("Elasticity/test_localization.jl")
        include("Elasticity/test_contribution.jl")
        include("Elasticity/test_hill_nestedquadgk_oblate.jl")
        include("Elasticity/test_hill_ti_coaxial.jl")
        include("Elasticity/test_param_conversions.jl")
        include("Elasticity/test_surface_stiffness.jl")
    end

    @testset "Cracks" begin
        include("Cracks/test_cod.jl")
        include("Cracks/test_cod_ti_aligned.jl")
        include("Cracks/test_cod_symbolic.jl")
        include("Cracks/test_crack_orientation.jl")
        include("Cracks/test_conductive.jl")
        include("Cracks/test_residue_accuracy.jl")
        include("Cracks/test_H_oracle.jl")
        include("Cracks/test_thermal.jl")
        include("Cracks/test_interface_stiffness.jl")
    end

    @testset "Conductivity" begin
        include("Conductivity/test_hill_order2.jl")
        include("Conductivity/test_hill_cylinder.jl")
        include("Conductivity/test_eshelby.jl")
        include("Conductivity/test_localization.jl")
    end

    # `Interactions` sits between the one-inclusion kernels it builds on and
    # the N-body schemes that consume it: the pair tensors must be trusted
    # before either scheme is exercised.
    @testset "Interactions" begin
        include("Interactions/test_pair_tensors.jl")
    end

    # `Assemblies` carries the positional cell and the two N-body schemes.  It
    # runs after `Interactions` (whose pair tensors it consumes) and needs the
    # `Schemes` types, so it sits between them and the rest.
    @testset "Assemblies" begin
        include("Assemblies/test_assembly.jl")
        include("Assemblies/test_cluster_model.jl")
        include("Assemblies/test_eim.jl")
        include("Assemblies/test_as_rve.jl")
        include("Assemblies/test_multiscale.jl")
    end

    @testset "Schemes" begin
        include("Schemes/test_rve.jl")
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
        include("Schemes/test_loc_bundles.jl")
        include("Schemes/test_crack_families.jl")
    end

    # The user-defined-inclusion contract spans Core (the abstractions) and
    # Schemes (every consumer), so it runs after both — the scheme kernels are
    # already compiled by then, which keeps this testset cheap.
    @testset "CustomInclusions" begin
        include("CustomInclusions/test_custom_inclusion.jl")
    end

    # Neural surrogates are another client of that same contract, so they run
    # straight after it. The committed models under
    # `src/NeuralInclusions/models/` are loaded, not retrained, which is what
    # keeps this testset deterministic and independent of the Lux stack.
    @testset "NeuralInclusions" begin
        include("NeuralInclusions/test_neural_inclusion.jl")
    end

    # Finite-element inclusions: skipped when the Ferrite stack is unavailable
    # (it is a weak dependency), and slow when it is — the crack cases mesh a
    # ball and factorize a ~10⁵-dof system, while the axisymmetric ones are
    # two-dimensional and cost a fraction of that.
    if HAS_FERRITE
        @testset "FiniteElementInclusions" begin
            include("FiniteElements/test_ferrite_crack.jl")
            include("FiniteElements/test_axi_excentered_sphere.jl")
            if HAS_GRIDAP
                include("FiniteElements/test_gridap_backend.jl")
            else
                @info "Gridap / GridapGmsh unavailable — skipping the " *
                    "cross-backend finite-element tests."
            end
        end
    else
        @info "Ferrite / FerriteGmsh / Gmsh unavailable — skipping the " *
            "finite-element inclusion tests."
    end

    @testset "LayeredSpheres" begin
        include("LayeredSpheres/test_bulk.jl")
        include("LayeredSpheres/test_interfaces.jl")
        include("LayeredSpheres/test_incompressible.jl")
        include("LayeredSpheres/test_conductivity.jl")
        include("LayeredSpheres/test_christensen.jl")
        include("LayeredSpheres/test_generic.jl")
        include("LayeredSpheres/test_scheme_integration.jl")
    end

    # The periodic multilayer cell. Placed after the layered morphologies
    # whose interface models it reuses, and before `Viscoelasticity`, which
    # carries its ageing-viscoelastic twin.
    @testset "Laminates" begin
        include("Laminates/test_km_blocks.jl")
        include("Laminates/test_laminate_cell.jl")
        include("Laminates/test_laminate_oracles.jl")
        include("Laminates/test_laminate_interfaces.jl")
        include("Laminates/test_laminate_nesting.jl")
        include("Laminates/test_laminate_dual_compat.jl")
        include("Laminates/test_laminate_symbolic.jl")
    end

    @testset "LayeredSpheroids" begin
        include("LayeredSpheroids/test_legendre.jl")
        include("LayeredSpheroids/test_coupling.jl")
        include("LayeredSpheroids/test_conductivity.jl")
        include("LayeredSpheroids/test_scheme_integration.jl")
        include("LayeredSpheroids/test_local_fields.jl")
    end

    # `Poromechanics` post-processes a homogenized stiffness, so it only needs
    # `Schemes` to be exercised — same position as in the load order.
    @testset "Poromechanics" begin
        include("Poromechanics/test_biot.jl")
    end

    # `Constitutive` is the Gauss-point law built on a cell + scheme, so it runs
    # after both, and after `Poromechanics` whose parameters the poroelastic
    # materials consume.
    @testset "Constitutive" begin
        include("Constitutive/test_material_api.jl")
        include("Constitutive/test_cracked.jl")
        include("Constitutive/test_poroelastic.jl")
    end

    @testset "Viscoelasticity" begin
        # The Laplace-Carson half (non-ageing) first: it depends on nothing
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
        include("Viscoelasticity/test_ti_alv.jl")
        include("Viscoelasticity/test_ortho_alv.jl")
        include("Viscoelasticity/test_ortho_dispatch_alv.jl")
        include("Viscoelasticity/test_alv_kernel_types.jl")
        include("Viscoelasticity/test_sensitivities_alv.jl")
        include("Viscoelasticity/test_order2_alv.jl")
        include("Viscoelasticity/test_extra_schemes_alv.jl")
        include("Viscoelasticity/test_crack_schemes_alv.jl")
        include("Viscoelasticity/test_glassy_limit_alv.jl")
    end

    @testset "Regression" begin
        include("regression/test_hill_cases.jl")
        include("regression/test_crack_cases.jl")
        include("regression/test_anisotropic.jl")
    end

    @testset "Studio" begin
        include("Studio/test_studio.jl")
    end
end
