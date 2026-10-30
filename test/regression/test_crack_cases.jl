using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

@testset "Crack regression cases" begin
    # ── Crack geometry ────────────────────────────────────────────────────
    @testset "Crack geometry" begin
        c_penny = PennyCrack(1.0)
        @test c_penny isa EllipticCrack{Float64, MeanFieldHomogenization.Cracks.Penny}
        @test aspect_ratio(c_penny) == 1.0
        @test semi_major(c_penny) == 1.0
        @test semi_minor(c_penny) == 1.0
        @test MeanFieldHomogenization.Cracks.crack_chi(c_penny) ≈ 2 / 3

        c_ell = EllipticCrack(2.0, 0.5)
        @test c_ell isa EllipticCrack{Float64, MeanFieldHomogenization.Cracks.EllipticShape}
        @test semi_major(c_ell) == 2.0
        @test semi_minor(c_ell) == 0.5
        @test aspect_ratio(c_ell) == 0.25

        c_sorted = EllipticCrack(0.5, 2.0)
        @test semi_major(c_sorted) == 2.0
        @test semi_minor(c_sorted) == 0.5

        r = RibbonCrack(0.3)
        @test r isa RibbonCrack{Float64}
        @test semi_minor(r) == 0.3
        @test aspect_ratio(r) == 0.0
        @test MeanFieldHomogenization.Cracks.crack_chi(r) ≈ π / 4

        c_rot = EllipticCrack(1.0, 0.5; euler_angles = (π / 4, π / 6, 0.0))
        @test c_rot isa EllipticCrack
        @test !isa(crack_basis(c_rot), CanonicalBasis)
    end

    # ── Analytical ISO (selected, keeping what is load-bearing) ───────────
    @testset "Analytical COD — isotropic matrix" begin
        E, ν = 210.0, 0.3
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        pc = PennyCrack(1.0)
        B_penny = cod_tensor(pc, C₀)
        # penny formulas (code-consistent)
        Bnn_expect = 16 * (1 - ν^2) / (3 * π * E)
        Bll_expect = 32 * (1 - ν^2) / (3 * π * E * (2 - ν))
        @test B_penny[3, 3] ≈ Bnn_expect rtol = 1.0e-10
        @test B_penny[1, 1] ≈ Bll_expect rtol = 1.0e-10
        @test B_penny[2, 2] ≈ Bll_expect rtol = 1.0e-10

        # Ribbon analytical values match closed-form
        r = RibbonCrack(1.0)
        B_r = cod_tensor(r, C₀)
        χ = π * (1 - ν^2) / E
        @test B_r[2, 2] ≈ χ                         rtol = 1.0e-10
        @test B_r[3, 3] ≈ χ                         rtol = 1.0e-10
        @test B_r[1, 1] ≈ χ / (1 - ν)               rtol = 1.0e-10
    end

    @testset "H ↔ B conversion, every component" begin
        # `compliance_from_cod` is k n ⊗ˢ B ⊗ˢ n and `cod_from_compliance` its
        # exact inverse. Up to 0.17.2 the first divided the shear components by
        # a further 2 to 4, and only the B[3,3] slot survived a round trip.
        ℬ = CanonicalBasis{3, Float64}()
        B = Tens([1.2 0.1 0.3; 0.1 0.9 -0.2; 0.3 -0.2 1.5], ℬ)
        n = [0.0, 0.0, 1.0]
        for (crack, k) in ((PennyCrack(1.0), 3 / 4), (RibbonCrack(1.0), 2 / π))
            H = compliance_from_cod(B, crack, ℬ)
            # The symmetrized product, written out from its definition.
            H_ref = [
                k * (
                    n[i] * B[j, p] * n[q] + n[j] * B[i, p] * n[q] +
                        n[i] * B[j, q] * n[p] + n[j] * B[i, q] * n[p]
                ) / 4
                    for i in 1:3, j in 1:3, p in 1:3, q in 1:3
            ]
            @test canonical_components(H) ≈ H_ref rtol = 1.0e-14
            @test cod_from_compliance(H, crack, ℬ) ≈ B rtol = 1.0e-14
        end
        # Without a crack, the elliptic factor.
        @test canonical_components(compliance_from_cod(B, ℬ)) ≈
            canonical_components(compliance_from_cod(B, PennyCrack(1.0), ℬ)) rtol = 1.0e-14
        @test canonical_components(cod_from_compliance(compliance_from_cod(B, ℬ), ℬ)) ≈
            canonical_components(B) rtol = 1.0e-14
    end

    @testset "Compliance contribution — H factorization (ellipse vs ribbon)" begin
        E, ν = 1.0, 0.2
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        # Elliptic (penny): H = (3/4) n̂ ⊗ˢ B ⊗ˢ n̂
        pc = PennyCrack(1.0)
        B_p = cod_tensor(pc, C₀)
        H_p = inverse_property_contribution(pc, C₀)
        n̂ = tens_basis(crack_basis(pc), 3)
        expected_H_p = (3 / 4) * (n̂ ⊗ˢ B_p ⊗ˢ n̂)
        @test H_p ≈ expected_H_p rtol = 1.0e-12
        # Analytical penny value: H[3,3,3,3] = (3/4) × 16(1-ν²)/(3πE) = 4(1-ν²)/(πE)
        @test H_p[3, 3, 3, 3] ≈ 4 * (1 - ν^2) / (π * E) rtol = 1.0e-10

        # Ribbon: H = (2/π) n̂ ⊗ˢ B ⊗ˢ n̂
        r = RibbonCrack(1.0)
        B_r = cod_tensor(r, C₀)
        H_r = inverse_property_contribution(r, C₀)
        n̂r = tens_basis(crack_basis(r), 3)
        expected_H_r = (2 / π) * (n̂r ⊗ˢ B_r ⊗ˢ n̂r)
        @test H_r ≈ expected_H_r rtol = 1.0e-12
        # Analytical ribbon: H[3,3,3,3] = (2/π) × π(1-ν²)/E = 2(1-ν²)/E
        @test H_r[3, 3, 3, 3] ≈ 2 * (1 - ν^2) / E rtol = 1.0e-10
    end

    @testset "delta_inverse_property — Budiansky density helpers" begin
        E, ν = 1.0, 0.2
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)
        ε = 0.1

        pc = PennyCrack(1.0)
        H_p = inverse_property_contribution(pc, C₀)
        ΔS_p = delta_inverse_property(pc, H_p, ε)
        # Elliptic: ΔS = (4π/3) ε H
        @test ΔS_p ≈ (4π / 3) * ε * H_p rtol = 1.0e-12

        r = RibbonCrack(1.0)
        H_r = inverse_property_contribution(r, C₀)
        ΔS_r = delta_inverse_property(r, H_r, ε)
        # Ribbon: ΔS = π ε H
        @test ΔS_r ≈ π * ε * H_r rtol = 1.0e-12
    end

    @testset "property_contribution — crack API symmetry" begin
        E, ν = 1.0, 0.25
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)
        K₀ = TensISO{3}(2.0)

        for crack in (PennyCrack(1.0), EllipticCrack(1.0, 0.3), RibbonCrack(0.5))
            # Elasticity: N = -C₀ : H : C₀
            H = inverse_property_contribution(crack, C₀)
            N = property_contribution(crack, C₀)
            expected_N = -(C₀ ⊡ H ⊡ C₀)
            @test N ≈ expected_N rtol = 1.0e-12

            # Conductivity: N_K = -K₀ · R · K₀
            R = inverse_property_contribution(crack, K₀)
            N_K = property_contribution(crack, K₀)
            expected_N_K = -(K₀ ⋅ R ⋅ K₀)
            @test N_K ≈ expected_N_K rtol = 1.0e-12

            # Dilute coherence on cracks: delta_property gives
            # (4π/3) ε N (elliptic) or π ε N (ribbon); at first order in ε,
            # matches -C₀ : delta_inverse_property : C₀.
            ε_small = 1.0e-3
            ΔC = delta_property(crack, N, ε_small)
            ΔS = delta_inverse_property(crack, H, ε_small)
            ΔC_from_ΔS = -(C₀ ⊡ ΔS ⊡ C₀)
            @test ΔC ≈ ΔC_from_ΔS rtol = 1.0e-10
        end
    end

    @testset "2D/3D consistency — B²ᵈ = (3π/8) lim_{η→0} B³ᵈ" begin
        E, ν = 1.0, 0.25
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        r = RibbonCrack(1.0)
        B_r = cod_tensor(r, C₀)
        # Take a very flat elliptic crack and check the limit relation
        # on the normal-normal component.
        ec = EllipticCrack(1.0, 1.0e-4)
        B_e = cod_tensor(ec, C₀)
        @test (3π / 8) * B_e[3, 3] ≈ B_r[3, 3] rtol = 1.0e-2
    end

    # The B → H factor is the one place where MeanFieldHomogenization's convention
    # deliberately differs from the Echoes / Kachanov one, so it gets its own
    # test rather than relying on the internal-consistency checks above (which
    # hold for *any* value of the factor, since it appears on both sides).
    #
    # MeanFieldHomogenization normalizes the flat limit by the in-plane half-width b,
    # uniformly for 3D and 2D:  H = lim (c/b) Q⁻¹ = (cS/V) n ⊗ˢ B ⊗ˢ n,
    # giving 3/4 for an ellipse and 2/π for a ribbon — both INDEPENDENT of η.
    # Echoes normalizes the elliptic case by the major semi-axis a instead, so
    # H_echoes = η · H_MFH (they coincide only for a penny, η = 1). Measured
    # against Echoes 2026-07-25 at η = 0.7, 0.5, 0.3, 0.1: ratio = η to 4
    # decimals. See docs/src/theory/cod_tensors.md § Conventions.
    @testset "B → H factor is η-independent (MFH uniform-b convention)" begin
        E, ν = 1.0, 0.25
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        for η in (1.0, 0.7, 0.5, 0.3, 0.1)
            ec = EllipticCrack(1.0, η)
            B = cod_tensor(ec, C₀)
            H = inverse_property_contribution(ec, C₀)
            # H_3333 = (3/4) B_33 for every aspect ratio — this is what pins
            # the convention. A 3η/4 factor would fail for every η ≠ 1.
            @test H[3, 3, 3, 3] ≈ 0.75 * B[3, 3] rtol = 1.0e-12
        end

        # Ribbon: same statement with the 2/π factor.
        r = RibbonCrack(1.0)
        B_r = cod_tensor(r, C₀)
        H_r = inverse_property_contribution(r, C₀)
        @test H_r[3, 3, 3, 3] ≈ (2 / π) * B_r[3, 3] rtol = 1.0e-12

        # Round trip through the documented bridge, at η ≠ 1.
        ec = EllipticCrack(1.0, 0.4)
        B = cod_tensor(ec, C₀)
        H = compliance_from_cod(B, ec)
        @test H ≈ inverse_property_contribution(ec, C₀) rtol = 1.0e-12
        @test cod_from_compliance(H, ec) ≈ B rtol = 1.0e-12
    end

    @testset "SIF — ribbon analytical" begin
        E, ν = 210.0, 0.3
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        b = 0.5
        r = RibbonCrack(b)
        e1, e2, e3 = tens_basis(CanonicalBasis{3, Float64}())
        Σ = e3 ⊗ˢ e3

        𝐊, modes = sif(r, C₀, Σ)
        Kᴵ, Kᴵᴵ, Kᴵᴵᴵ = modes

        @test norm(𝐊) ≈ √(π * b) rtol = 1.0e-10
        @test Kᴵ ≈ √(π * b) rtol = 1.0e-10
        @test abs(Kᴵᴵ) < 1.0e-12
        @test abs(Kᴵᴵᴵ) < 1.0e-12
    end

    @testset "DIF — ellipse isotropic penny under uniaxial tension" begin
        E, ν = 1.0, 0.3
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)

        pc = PennyCrack(1.0)
        e1, e2, e3 = tens_basis(CanonicalBasis{3, Float64}())
        Σ = e3 ⊗ˢ e3

        d = dif(pc, C₀, Σ)
        # Expected value follows the code formula: d[3] = B[3,3] = 16(1-ν²)/(3πE)
        @test d[3] ≈ 16 * (1 - ν^2) / (3 * π * E) rtol = 1.0e-10
        @test abs(d[1]) < 1.0e-12
        @test abs(d[2]) < 1.0e-12
    end

end
