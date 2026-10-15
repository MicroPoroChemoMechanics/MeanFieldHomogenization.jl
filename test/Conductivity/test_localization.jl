using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy   # top level on purpose: `@syms` below is expanded before any
# `using` nested in a @testset body would have run.

# =============================================================================
#  Localization and contribution tensors for conductivity (2nd order).
# =============================================================================

@testset "gradient_gradient_loc — iso sphere, iso matrix" begin
    # Analytical for iso sphere: A_∇∇ = 3k₀/(2k₀+k₁) 𝟙
    ell = Ellipsoid(1.0)
    for (k₀, k₁) in ((1.0, 5.0), (2.5, 0.3), (10.0, 10.0))
        K₀ = TensISO{3}(k₀)
        K₁ = TensISO{3}(k₁)
        A = gradient_gradient_loc(ell, K₁, K₀)
        expected = 3 * k₀ / (2 * k₀ + k₁)
        @test A[1, 1] ≈ expected rtol = 1.0e-12
        @test A[2, 2] ≈ expected rtol = 1.0e-12
        @test A[3, 3] ≈ expected rtol = 1.0e-12
        @test abs(A[1, 2]) < 1.0e-14
    end

    # K₁ = K₀ → A = 𝟙
    K₀ = TensISO{3}(2.0)
    A0 = gradient_gradient_loc(ell, K₀, K₀)
    @test A0[1, 1] ≈ 1.0 rtol = 1.0e-12
end

@testset "4 conductivity localization tensors consistent" begin
    ell = Ellipsoid(2.0, 1.5, 1.0)
    K₀ = TensISO{3}(2.0)
    K₁ = TensISO{3}(5.0)

    A_∇∇ = gradient_gradient_loc(ell, K₁, K₀)
    A_q∇ = flux_gradient_loc(ell, K₁, K₀)
    A_∇q = gradient_flux_loc(ell, K₁, K₀)
    A_qq = flux_flux_loc(ell, K₁, K₀)

    # A_q∇ = K₁ · A_∇∇
    lhs = A_q∇
    rhs = K₁ ⋅ A_∇∇
    @test lhs ≈ rhs rtol = 1.0e-12
    # A_∇q = A_∇∇ · R₀
    rhs2 = A_∇∇ ⋅ inv(K₀)
    @test A_∇q ≈ rhs2 rtol = 1.0e-12
end

@testset "Fully-anisotropic conductivity — sanity" begin
    # For a general aniso ellipsoid in an aniso matrix, A_∇∇ is NOT
    # symmetric (localization is a linear map between tensors, not a
    # symmetric 2-tensor itself). We just verify it is finite and
    # reduces to the identity when K₁ = K₀.
    K_arr = [3.0 0.5 0.3; 0.5 2.0 0.2; 0.3 0.2 1.5]
    K₀ = TensND.Tens(K_arr)
    K₁ = 2.0 * K₀

    ell = Ellipsoid(2.0, 1.5, 1.0)
    A = gradient_gradient_loc(ell, K₁, K₀)
    @test all(isfinite, A)

    # K₁ = K₀ → A = 𝟙
    A0 = gradient_gradient_loc(ell, K₀, K₀)
    @test A0 ≈ TensISO{3}(1.0) atol = 1.0e-10
end

@testset "conductivity_contribution and resistivity_contribution" begin
    K₀ = TensISO{3}(2.0)
    K₁ = TensISO{3}(5.0)
    ell = Ellipsoid(1.0)

    N_K = conductivity_contribution(ell, K₁, K₀)
    H_R = resistivity_contribution(ell, K₁, K₀)
    # For iso sphere: N_K = (K₁-K₀) · A_∇∇ = 3k₀(k₁-k₀)/(2k₀+k₁) · 𝟙
    expected_N = 3 * 2.0 * (5.0 - 2.0) / (2 * 2.0 + 5.0)
    @test N_K[1, 1] ≈ expected_N rtol = 1.0e-12

    # Dilute coherence: K_eff ≈ K₀ + f N_K ; R_eff ≈ R₀ + f H_R ≈ inv(K_eff) to first order
    f = 0.01
    ΔK = delta_conductivity(N_K, f)
    ΔR = delta_resistivity(H_R, f)
    K_eff = K₀ + ΔK
    R_eff = inv(K₀) + ΔR
    R_from_K = inv(K_eff)
    @test norm(R_eff - R_from_K) / norm(R_from_K) < 5 * f^2
end

# An insulating sphere has the finite resistivity contribution 3/(2k₀) 𝟏,
# read as −𝐊₀⁻¹ ⋅ 𝐍 ⋅ 𝐊₀⁻¹ without inverting K₁ = 0 (NaN before 0.16.3).
@testset "resistivity_contribution — insulating sphere" begin
    for k₀ in (2.0, 7.5)
        K₀ = TensISO{3}(k₀)
        R = @inferred resistivity_contribution(Ellipsoid(1.0), zero(K₀), K₀)
        @test R ≈ TensISO{3}(3 / (2k₀)) rtol = 1.0e-13
    end
    d = ForwardDiff.derivative(
        k -> resistivity_contribution(Ellipsoid(1.0), zero(TensISO{3}(k)), TensISO{3}(k))[1, 1], 2.0
    )
    @test d ≈ -3 / (2 * 2.0^2) rtol = 1.0e-12
end

@testset "Conductivity — SymPy genericity" begin
    @syms ka kb
    K₀ = TensISO{3}(ka)
    K₁ = TensISO{3}(kb)
    A = gradient_gradient_loc(Ellipsoid(1.0), K₁, K₀)
    # Expected symbolic: A[1,1] = 3ka/(2ka+kb)
    simplified = simplify(A[1, 1] - 3ka / (2ka + kb))
    @test simplified == 0
end
