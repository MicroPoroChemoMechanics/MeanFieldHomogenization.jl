using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# ── Material parameters for cross-validation ──────────────────────────────────
# Isotropic elastic: E=210 GPa, ν=0.3
const E_ref = 210.0e3
const ν_ref = 0.3
const λ_ref = E_ref * ν_ref / ((1 + ν_ref) * (1 - 2ν_ref))
const μ_ref = E_ref / (2 * (1 + ν_ref))
const k_ref = λ_ref + 2μ_ref / 3

const C_iso3 = TensISO{3}(3k_ref, 2μ_ref)
const K_iso3 = TensISO{3}(5.0)

@testset "Hill tensor regression cases" begin

    @testset "Auxiliary tensors — sphere 3D" begin
        ell = Ellipsoid(1.0)

        IA = tens_IA(ell)
        UA = tens_UA(ell)
        VA = tens_VA(ell)

        @test IA[1, 1] ≈ 1 / 3  atol = 1.0e-12
        @test IA[2, 2] ≈ 1 / 3  atol = 1.0e-12
        @test IA[3, 3] ≈ 1 / 3  atol = 1.0e-12
        @test IA[1, 2] ≈ 0.0  atol = 1.0e-12
        sum_IA = IA[1, 1] + IA[2, 2] + IA[3, 3]
        @test sum_IA ≈ 1.0  atol = 1.0e-12

        @test VA[1, 1, 1, 1] ≈ 1 / 3  atol = 1.0e-12
        @test VA[1, 2, 1, 2] ≈ 1 / 6  atol = 1.0e-12
        @test VA[1, 1, 2, 2] ≈ 0.0  atol = 1.0e-12

        @test UA[1, 1, 1, 1] ≈ 1 / 5  atol = 1.0e-12
        @test UA[1, 1, 2, 2] ≈ 1 / 15  atol = 1.0e-10
        @test UA[1, 2, 1, 2] ≈ 1 / 15  atol = 1.0e-10
    end

    @testset "Auxiliary tensors — prolate spheroid 3D" begin
        ell = Ellipsoid(2.0, 1.0, 1.0)
        IA = tens_IA(ell)
        @test IA[2, 2] ≈ IA[3, 3]  atol = 1.0e-10
        @test IA[1, 1] + IA[2, 2] + IA[3, 3] ≈ 1.0  atol = 1.0e-10
        @test IA[1, 1] < IA[2, 2]
    end

    @testset "Auxiliary tensors — oblate spheroid 3D" begin
        ell = Ellipsoid(2.0, 2.0, 1.0)
        IA = tens_IA(ell)
        @test IA[1, 1] ≈ IA[2, 2]  atol = 1.0e-10
        @test IA[1, 1] + IA[2, 2] + IA[3, 3] ≈ 1.0  atol = 1.0e-10
    end

    @testset "Auxiliary tensors — circle 2D" begin
        ell = Ellipsoid(1.0; dim = 2)
        IA = tens_IA(ell)
        @test IA[1, 1] ≈ 0.5  atol = 1.0e-12
        @test IA[2, 2] ≈ 0.5  atol = 1.0e-12
        @test IA[1, 1] + IA[2, 2] ≈ 1.0  atol = 1.0e-12
    end

    @testset "Hill 3D iso — sphere" begin
        ell = Ellipsoid(1.0)
        P = hill_tensor(ell, C_iso3)

        P1111_expect = (1 / 5) / (λ_ref + 2μ_ref) + (1 / 3 - 1 / 5) / μ_ref
        @test P[1, 1, 1, 1] ≈ P1111_expect  atol = 1.0e-10

        @test P[1, 1, 2, 2] ≈ P[2, 2, 1, 1]  atol = 1.0e-12
        @test P[1, 2, 1, 2] ≈ P[2, 1, 1, 2]  atol = 1.0e-12
        @test P[1, 1, 1, 1] ≈ P[2, 2, 2, 2]  atol = 1.0e-10
        @test P[1, 1, 1, 1] ≈ P[3, 3, 3, 3]  atol = 1.0e-10
    end

    @testset "Hill 3D iso vs DECUHR — prolate spheroid" begin
        # Note: the :residues path on an iso matrix is numerically unstable
        # (pre-existing degeneracy in the root-finder) — cross-check against
        # DECUHR which is ForwardDiff- and iso-safe.
        ell = Ellipsoid(3.0, 1.0, 1.0)
        C_full = Tens(Array(C_iso3))

        P_iso = hill_tensor(ell, C_iso3)
        P_dcr = hill_tensor(ell, C_full; method = :decuhr, abstol = 1.0e-8, reltol = 1.0e-6)

        @test all(isapprox.(components_canon(P_dcr), components_canon(P_iso); atol = 5.0e-5))
    end

    @testset "Hill 2D iso — circle" begin
        ell = Ellipsoid(1.0; dim = 2)
        C_iso2 = TensISO{2}(3k_ref, 2μ_ref)
        P = hill_tensor(ell, C_iso2)

        @test P[1, 1, 1, 1] ≈ P[2, 2, 2, 2]  atol = 1.0e-10
        @test P[1, 1, 2, 2] ≈ P[2, 2, 1, 1]  atol = 1.0e-12
    end

    @testset "Hill order-2 (conductivity) — sphere 3D iso" begin
        ell = Ellipsoid(1.0)
        P = hill_tensor(ell, K_iso3)

        k_cond = K_iso3[1, 1]
        @test P[1, 1] ≈ 1 / (3k_cond)  atol = 1.0e-12
        @test P[2, 2] ≈ 1 / (3k_cond)  atol = 1.0e-12
        @test P[3, 3] ≈ 1 / (3k_cond)  atol = 1.0e-12
        @test P[1, 2] ≈ 0.0           atol = 1.0e-12
    end

    @testset "Hill order-2 (conductivity) — prolate spheroid 3D iso" begin
        ell = Ellipsoid(2.0, 1.0, 1.0)
        P = hill_tensor(ell, K_iso3)

        k_cond = K_iso3[1, 1]
        IA = tens_IA(ell)
        @test P[1, 1] ≈ IA[1, 1] / k_cond  atol = 1.0e-10
        @test P[2, 2] ≈ IA[2, 2] / k_cond  atol = 1.0e-10
        @test P[3, 3] ≈ IA[3, 3] / k_cond  atol = 1.0e-10
        @test P[2, 2] ≈ P[3, 3]  atol = 1.0e-10
    end

    @testset "Hill order-2 (conductivity) — circle 2D" begin
        ell = Ellipsoid(1.0; dim = 2)
        K_iso2 = TensISO{2}(3.0)
        P = hill_tensor(ell, K_iso2)

        @test P[1, 1] ≈ P[2, 2]  atol = 1.0e-10
        @test P[1, 2] ≈ 0.0     atol = 1.0e-12
    end

end
