using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  Incompressibility (ν → 1/2) robustness of the  bulk
#  recurrence in the `(u, σ)` state vector formulation.
# =============================================================================

@testset "Quasi-incompressible core (κ_1 ≫ μ, ν_1 → 0.5)" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)

    # Sweep ν_1 toward 0.5 (κ_1/μ_1 diverges).
    μ₁ = 80.0
    νs = [0.3, 0.45, 0.49, 0.499, 0.4999, 0.49999]
    αs = Float64[]
    for ν₁ in νs
        κ₁ = 2 * μ₁ * (1 + ν₁) / (3 * (1 - 2 * ν₁))
        C₁ = TensISO{3}(3κ₁, 2μ₁)
        s = LayeredSphere((1.0,), (C₁,))
        α = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s, κ₀, μ₀)[1]
        push!(αs, α)
    end
    # Convergence toward the incompressible limit α = 4μ₀ / (3κ∞ + 4μ₀)  → 0.
    @test all(isfinite, αs)
    @test all(αs .> 0)
    @test αs[end] < αs[1]    # monotone decrease
end

@testset "Exactly incompressible core (κ_1 = 1e30)" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C_inc = TensISO{3}(3.0e30, 2 * 80.0)

    # Single layer — α should be very small but finite.
    s1 = LayeredSphere((1.0,), (C_inc,))
    α = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s1, κ₀, μ₀)[1]
    @test isfinite(α)
    @test α < 1.0e-25   # ≈ 4μ₀/(3 · 1e30) at leading order

    # Core-shell with incompressible core: layer 1 α → 0, layer 2 finite.
    C_shell = TensISO{3}(3κ₀ * 1.5, 2μ₀ * 1.5)
    s2 = LayeredSphere((0.5, 1.0), (C_inc, C_shell))
    α2 = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s2, κ₀, μ₀)
    @test all(isfinite, α2)
    @test α2[1] < 1.0e-25
end

@testset "Matrix is compressible — single-layer exact Eshelby" begin
    # With κ_1 very large but κ_0 finite, α_1 should match
    # (3κ_0 + 4μ_0) / (3κ_1 + 4μ_0) analytically.
    κ₀, μ₀ = 50.0, 30.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    for κ₁ in [100.0, 1.0e4, 1.0e8, 1.0e16]
        C₁ = TensISO{3}(3κ₁, 2μ₀)
        s = LayeredSphere((1.0,), (C₁,))
        α_comp = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s, κ₀, μ₀)[1]
        α_expected = (3κ₀ + 4μ₀) / (3κ₁ + 4μ₀)
        @test α_comp ≈ α_expected rtol = 1.0e-10
    end
end
