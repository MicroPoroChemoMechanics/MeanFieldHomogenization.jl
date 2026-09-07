# The cavity of superspherical shape, as a phase of an RVE.
#
# What is tested here is the *contract*, not the solve -- that was step 7's job.
# The question is whether a shape with no closed-form Eshelby solution reaches
# every scheme, in both physics, with the right tensors and without the phase
# property leaking into the answer.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Ferrite, FerriteGmsh, Gmsh

const FE = MeanFieldHomogenization.FiniteElements

const _POPTS = FECellMeshOptions(;
    level = 2, outer_level = 3, radius_ratio = 3.0, relax = 20
)
const _C0 = iso_stiffness(0.8333, 0.3846)          # E = 1, ν = 0.3
const _K0 = TensND.TensISO{3}(1.0)

@testset "FESupershapePore" begin

    @testset "level 0 of the contract" begin
        pore = FESupershapePore(Superspheroid(1.0, 2.0, 0.7); opts = _POPTS)
        @test dimension(pore) == 3
        @test shape_trait(pore) isa SupershapePoreShape
        @test inclusion_basis(pore) isa TensND.AbstractBasis
        # `false` is the statement that inv(C₁) is meaningless, which for a
        # cavity it is -- and it is what keeps the phase property out of the
        # contribution tensors.
        @test !MeanFieldHomogenization.Core.is_homogeneous_inclusion(pore)
        D = get_array(change_tens_canon(shape_tensor(pore)))
        @test diag(D) ≈ [1.0, 1.0, 2.0]
        @test occursin("Superspheroid", sprint(show, pore))
        @test eltype(pore.shape) === Float64
    end

    @testset "a cavity's stress side is exactly zero" begin
        pore = FESupershapePore(Supersphere(1.0, 1.0); opts = _POPTS)
        Z4 = MeanFieldHomogenization.Core.stress_strain_loc(pore, _C0, _C0)
        Z2 = MeanFieldHomogenization.Core.flux_gradient_loc(pore, _K0, _K0)
        # Identically zero, not zero to the discretization error: no solve is
        # even run for it.
        @test iszero(KM(Z4))
        @test iszero(get_array(change_tens_canon(Z2)))
        @test fe_assembly_count(pore) == 0
    end

    pore = FESupershapePore(Supersphere(1.0, 1.0); opts = _POPTS)

    @testset "the spherical pore through the contract" begin
        A = fe_cell_localization(pore, _C0)
        @test cubic_residual(A) < 2.0e-3
        @test fe_assembly_count(pore) == 1
        # Memoized on the reference: asking again is free.
        fe_cell_localization(pore, _C0)
        @test fe_assembly_count(pore) == 1

        # The two contribution tensors satisfy an exact identity that nothing
        # here was asked to enforce: for a cavity `H = A:S₀` and `N = -C₀:A`,
        # hence `H:C₀ = -S₀:N`. It holds to machine precision because both come
        # from the same `A` through the package's own generic formulas.
        N = stiffness_contribution(pore, _C0, _C0)
        H = compliance_contribution(pore, _C0, _C0)
        @test norm(KM(H ⊡ _C0 + inv(_C0) ⊡ N)) < 1.0e-12 * norm(KM(H ⊡ _C0))

        # And the phase property really is ignored: a wildly different `C₁`
        # changes nothing.
        Nother = stiffness_contribution(pore, iso_stiffness(99.0, 42.0), _C0)
        @test norm(KM(Nother) - KM(N)) < 1.0e-12 * norm(KM(N))
        @test fe_assembly_count(pore) == 1
    end

    @testset "conduction on the same object and the same mesh" begin
        A = fe_cell_localization(pore, _K0)
        d = diag(get_array(change_tens_canon(A)))
        @test all(x -> abs(x - 1.5) < 5.0e-3, d)        # exact value is 3/2
        # One assembly per physics, and the mesh was built once.
        @test fe_assembly_count(pore) == 2
        R = resistivity_contribution(pore, _K0, _K0)
        @test tr(get_array(change_tens_canon(R))) / 3 ≈ sum(d) / 3 rtol = 1.0e-10
    end

    @testset "it reaches the schemes, in both physics" begin
        rve = RVE()
        add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => _C0, :K => _K0); fraction = :rest)
        add_phase!(rve, :p, pore, Dict(:C => _C0, :K => _K0); fraction = 0.05)

        Cd = homogenize(rve, Dilute(), :C)
        N = stiffness_contribution(pore, _C0, _C0)
        # `Dilute` *is* `C₀ + f N` by definition, so this is a wiring check with
        # no tolerance to argue about.
        @test norm(KM(Cd) - KM(_C0 + 0.05 * N)) < 1.0e-12 * norm(KM(Cd))

        Cmt = homogenize(rve, MoriTanaka(), :C)
        k, μ = k_mu(best_fit_iso(Cmt))
        k0, μ0 = k_mu(best_fit_iso(_C0))
        @test 0 < k < k0                     # pores soften, and not to nothing
        @test 0 < μ < μ0
        # Mori-Tanaka is stiffer than dilute at this fraction, as it must be for
        # a porous material.
        @test k_mu(best_fit_iso(Cmt))[1] > k_mu(best_fit_iso(Cd))[1]

        Kmt = homogenize(rve, MoriTanaka(), :K)
        keff = get_array(change_tens_canon(Kmt))[1, 1]
        @test 0 < keff < 1
        @test isapprox(keff, get_array(change_tens_canon(Kmt))[2, 2]; rtol = 5.0e-3)
    end

    @testset "the mesh report makes the geometry error visible" begin
        r = fe_cell_mesh_report(pore)
        @test r.ncells > 100
        @test r.area_outer ≈ r.area_outer_exact rtol = 1.0e-3
        # The three volumes are the point: the flat triangulation the surface
        # was built from is off by percent, the curved boundary the solve
        # actually sees by a tenth of that. Snapping the mid-edge nodes is what
        # separates them, and `mesh_volume` alone cannot see it.
        eflat = abs(r.volume_flat - r.volume_exact) / r.volume_exact
        ecurv = abs(r.volume_curved - r.volume_exact) / r.volume_exact
        @test eflat > 0.05
        @test ecurv < 0.01
        @test eflat > 10 * ecurv
        @test r.volume_error ≈ (r.volume_curved - r.volume_exact) / r.volume_exact
    end

    @testset "what it refuses, and why" begin
        # An anisotropic reference: the corrected condition uses the closed-form
        # Kelvin dipole, which exists for an isotropic matrix only.
        Caniso = TensND.Tens(
            TensND.inv_KM(Matrix(KM(_C0)) + Diagonal([0.0, 0.0, 0.4, 0, 0, 0]))
        )
        err = try
            fe_cell_localization(pore, Caniso)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("isotropic", err.msg)
        @test occursin("IsoSymmetrize", err.msg)

        # Sensitivity through a finite-element solve is refused rather than
        # answered with a silent zero.
        @test_throws ErrorException derivative(
            let
                rve = RVE()
                add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => _C0); fraction = :rest)
                add_phase!(rve, :p, pore, Dict(:C => _C0); fraction = 0.05)
                rve
            end,
            Dilute(), geometry(:p, :a)
        )
    end

    @testset "fe_reset! drops everything" begin
        p2 = FESupershapePore(Supersphere(1.0, 1.0); opts = _POPTS)
        fe_cell_localization(p2, _C0)
        @test fe_assembly_count(p2) == 1
        fe_reset!(p2)
        @test fe_assembly_count(p2) == 0
        fe_cell_localization(p2, _C0)
        @test fe_assembly_count(p2) == 1
    end
end
