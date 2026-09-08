# The superspheroidal cavity by axisymmetric Fourier elements.
#
# Two oracles, and the second is the strong one. At `p = 1` the body is an
# exact spheroid, whose cavity localization the package already knows in closed
# form through `hill_tensor` — so `[𝕀 − ℙ:ℂ₀]⁻¹` is the answer, prolate and
# oblate, in both physics. And the response must be transversely isotropic,
# which `A₁₂₁₂ = (A₁₁₁₁ − A₁₁₂₂)/2` checks for free and to machine precision:
# it exercises the three Fourier modes, the azimuthal projections, the boundary
# line integral and the Kelvin reassembly all at once.
#
# The mesh is two-dimensional, so `nradial = 24` costs a fraction of a second
# and the whole file runs in well under a minute.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Ferrite, FerriteGmsh, Gmsh

const _FEA = MeanFieldHomogenization.FiniteElements
const _COA = MeanFieldHomogenization.Core

# Canonical components, never the tensor's own basis: `Spheroid` sorts its
# semi-axes and permutes its basis, so comparing `get_array` of two tensors
# that disagree on their frame is the classic way to invent a bug.
_kmc(t) = _COA.mandel66_minor(_COA._C_array(t))
_mc(t) = TensND.components_canon(t)

const _AXP_ν = 0.3
const _AXP_C0 = iso_stiffness(
    1.0 / (3 * (1 - 2 * _AXP_ν)), 1.0 / (2 * (1 + _AXP_ν))
)
const _AXP_K0 = TensISO{3}(1.0)

"Exact cavity localization of a spheroid of aspect ratio `c`, canonical Kelvin."
_axp_exact4(c) = inv(Matrix(1.0I, 6, 6) - _kmc(hill_tensor(Spheroid(c), _AXP_C0)) * _kmc(_AXP_C0))
_axp_exact2(c) = inv(Matrix(1.0I, 3, 3) - _mc(hill_tensor(Spheroid(c), _AXP_K0)) * _mc(_AXP_K0))

_axp_pore(sh; nradial = 24, radius_ratio = 8.0, kw...) = FEAxiSupershapePore(
    sh; opts = FEAxiMeshOptions(; nradial, radius_ratio), kw...
)

@testset "axisymmetric superspheroidal pore" begin

    @testset "the meridian profile is exact by construction" begin
        # The parametrization satisfies the level set identically, so every
        # sampled point is on the surface — checked rather than assumed, and
        # the two corners are pinned so the pole stays on the axis and the
        # equator in the equatorial plane.
        for sh in (
                Superspheroid(1.0, 1.0, 1.0), Superspheroid(1.0, 2.0, 0.8),
                Superspheroid(1.5, 0.6, 0.35),
            )
            pts = _FEA._superspheroid_meridian(sh, 41)
            @test length(pts) == 41
            @test all(abs(level_set(sh, (ρ, 0.0, z))) < 1.0e-12 for (ρ, z) in pts)
            @test pts[1] == (sh.a, 0.0)                 # the equator, exactly
            @test pts[end] == (0.0, sh.c)               # the pole, exactly
            # Monotone in both coordinates: a profile that doubles back would
            # make a spline overshoot and a mesh invert.
            @test issorted([ρ for (ρ, _) in pts]; rev = true)
            @test issorted([z for (_, z) in pts])
        end
        @test_throws ArgumentError _FEA._superspheroid_meridian(
            Superspheroid(1.0, 1.0, 1.0), 2
        )
    end

    @testset "the mesh resolves the profile" begin
        pore = _axp_pore(Superspheroid(1.0, 1.0, 1.0); nradial = 16, radius_ratio = 4.0)
        r = fe_axi_pore_mesh_report(pore)
        @test r.ncells > 500
        @test r.volume_cavity_exact ≈ 4π / 3
        # The measured revolution volume of the matrix against the closed form.
        @test abs(r.volume_error) < 0.01
    end

    @testset "against the exact spheroidal cavity, prolate and oblate" begin
        # `radius_ratio = 8` because what is left after the correction is
        # truncation, not discretization — see the sweep below.
        for (c, tol4, tol2) in ((0.5, 2.0e-3, 1.0e-3), (1.0, 5.0e-4, 1.0e-4), (2.0, 5.0e-4, 1.0e-4))
            pore = _axp_pore(Superspheroid(1.0, c, 1.0))
            A = _kmc(strain_strain_loc(pore, _AXP_C0, _AXP_C0))
            a = _mc(gradient_gradient_loc(pore, _AXP_K0, _AXP_K0))
            Aex, aex = _axp_exact4(c), _axp_exact2(c)
            @test norm(A - Aex) / norm(Aex) < tol4
            @test norm(a - aex) / norm(aex) < tol2
        end
    end

    @testset "the response is transversely isotropic to machine precision" begin
        # A structural identity of the class, so it holds however coarse the
        # mesh: the three modes are decoded onto the Kelvin basis and
        # reassembled, and nothing else can make `A₁₂₁₂` agree with
        # `(A₁₁₁₁ − A₁₁₂₂)/2` to round-off.
        for c in (0.5, 1.0, 2.0), p in (0.6, 1.0)
            pore = _axp_pore(Superspheroid(1.0, c, p); nradial = 14, radius_ratio = 4.0)
            A = _kmc(strain_strain_loc(pore, _AXP_C0, _AXP_C0))
            @test A[6, 6] / 2 ≈ (A[1, 1] - A[1, 2]) / 2 atol = 1.0e-12
            @test A[1, 1] ≈ A[2, 2] atol = 1.0e-12
            @test A[4, 4] ≈ A[5, 5] atol = 1.0e-12
            @test maximum(abs, A[1:3, 4:6]) < 1.0e-12
            a = _mc(gradient_gradient_loc(pore, _AXP_K0, _AXP_K0))
            @test a[1, 1] ≈ a[2, 2] atol = 1.0e-12
            @test maximum(abs, a - Diagonal(diag(a))) < 1.0e-12
        end
    end

    @testset "the correction is what makes the cell small" begin
        # The only test that proves the dipole sign. A wrong one leaves twice
        # the truncation bias; a right one leaves `O((a/R)⁵)` where the
        # uncorrected answer carries `O((a/R)³)`.
        Aex = _axp_exact4(1.0)
        errs = map((3.0, 6.0)) do rr
            b = fe_axi_pore_breakdown(
                _axp_pore(Superspheroid(1.0, 1.0, 1.0); nradial = 20, radius_ratio = rr),
                _AXP_C0,
            )
            (
                cor = norm(_kmc(b.A) - Aex) / norm(Aex),
                unc = norm(_kmc(b.A_uncorrected) - Aex) / norm(Aex),
            )
        end
        # Corrected beats uncorrected at both radii, and by more at the larger.
        @test errs[1].cor < errs[1].unc / 5
        @test errs[2].cor < errs[2].unc / 20
        # `(a/R)³` for the uncorrected: doubling `R` divides it by about eight.
        @test errs[1].unc / errs[2].unc > 4
        # And faster than that for the corrected.
        @test errs[1].cor / errs[2].cor > errs[1].unc / errs[2].unc
    end

    @testset "conduction is exact on a sphere, and that is not luck" begin
        # A spherical cavity's exterior perturbation *is* a pure dipole, so the
        # corrected condition has no higher multipole left to truncate. This is
        # the sharpest single number in the file.
        for nr in (16, 32)
            pore = _axp_pore(Superspheroid(1.0, 1.0, 1.0); nradial = nr, radius_ratio = 8.0)
            a = _mc(gradient_gradient_loc(pore, _AXP_K0, _AXP_K0))
            @test a[1, 1] ≈ 1.5 rtol = 2.0e-5
            @test a[3, 3] ≈ 1.5 rtol = 2.0e-5
        end
    end

    @testset "a cavity carries no stress and no flux" begin
        pore = _axp_pore(Superspheroid(1.0, 1.5, 0.7); nradial = 14, radius_ratio = 4.0)
        @test maximum(abs, _kmc(stress_strain_loc(pore, _AXP_C0, _AXP_C0))) == 0.0
        @test maximum(abs, _mc(flux_gradient_loc(pore, _AXP_K0, _AXP_K0))) == 0.0
        @test !MeanFieldHomogenization.Core.is_homogeneous_inclusion(pore)
    end

    @testset "it reaches the schemes" begin
        pore = _axp_pore(Superspheroid(1.0, 2.0, 0.8); nradial = 14, radius_ratio = 4.0)
        rve = RVE()
        add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => _AXP_C0, :K => _AXP_K0); fraction = :rest)
        add_phase!(rve, :pores, pore, Dict(:C => _AXP_C0, :K => _AXP_K0); fraction = 0.04)
        C = homogenize(rve, MoriTanaka(), :C)
        K = homogenize(rve, MoriTanaka(), :K)
        # Porosity softens, in both physics, and keeps the symmetry class.
        @test k_mu(_COA.isotropify(C))[1] < k_mu(_AXP_C0)[1]
        km = _mc(K)
        @test km[1, 1] < 1.0 && km[3, 3] < 1.0
        @test km[1, 1] ≈ km[2, 2] atol = 1.0e-12
    end

    @testset "memoization, and the cache is per reference" begin
        pore = _axp_pore(Superspheroid(1.0, 1.0, 1.0); nradial = 12, radius_ratio = 4.0)
        @test fe_assembly_count(pore) == 0
        gradient_gradient_loc(pore, _AXP_K0, _AXP_K0)
        @test fe_assembly_count(pore) == 1
        gradient_gradient_loc(pore, _AXP_K0, _AXP_K0)
        @test fe_assembly_count(pore) == 1
        gradient_gradient_loc(pore, TensISO{3}(4.0), TensISO{3}(4.0))
        @test fe_assembly_count(pore) == 2
        fe_reset!(pore)
        @test fe_assembly_count(pore) == 0
    end

    @testset "guard rails" begin
        sh = Superspheroid(1.0, 1.0, 1.0)
        @test_throws ArgumentError FEAxiSupershapePore(sh; guard = :nope)
        @test_throws ArgumentError FEAxiMeshOptions(; nprofile = 2)
        # An anisotropic reference is out of contract: the corrected condition
        # uses the closed-form isotropic dipole field.
        pore = _axp_pore(sh; nradial = 12, radius_ratio = 4.0)
        Cti = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0, 0, 1))
        @test_throws ArgumentError strain_strain_loc(pore, Cti, Cti)
        # And sensitivity is refused while no surrogate answers.
        @test_throws ErrorException MeanFieldHomogenization.Schemes._replace_geom_field(
            pore, Val(:p), nothing, 0.9
        )
    end
end
