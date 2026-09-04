using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy   # top level on purpose: `@syms` below is expanded before any
# `using` nested in a @testset body would have run.

# =============================================================================
#  Type-genericity tests for LayeredSphere: ForwardDiff.Dual + Sym.
# =============================================================================

@testset "LayeredSphere — ForwardDiff through bulk localization" begin
    # Derivative of α_1 w.r.t. κ_1 for single-layer composite.
    κ₀, μ₀ = 100.0, 70.0
    function α1_of_κ1(κ₁)
        C₁ = TensISO{3}(3 * κ₁, 2 * 140.0)
        s = LayeredSphere((1.0,), (C₁,))
        return MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, κ₀, μ₀)[1]
    end
    κ₁ = 200.0
    d_auto = ForwardDiff.derivative(α1_of_κ1, κ₁)
    # Analytical: α₁ = (3κ₀ + 4μ₀)/(3κ₁ + 4μ₀) → dα/dκ₁ = -3(3κ₀+4μ₀)/(3κ₁+4μ₀)²
    d_analytic = -3 * (3κ₀ + 4μ₀) / (3 * κ₁ + 4μ₀)^2
    @test d_auto ≈ d_analytic rtol = 1.0e-10
end

@testset "LayeredSphere — ForwardDiff through conductivity localization" begin
    k₀ = 2.0
    function α1_of_k1(k₁)
        s = LayeredSphere((1.0,), (TensISO{3}(k₁),))
        return MeanFieldHomogenization.LayeredSpheres._cond_localization(s, k₀)[1]
    end
    k₁ = 5.0
    d_auto = ForwardDiff.derivative(α1_of_k1, k₁)
    # α₁ = 3k₀/(2k₀ + k₁) → dα/dk₁ = -3k₀/(2k₀+k₁)²
    d_analytic = -3 * k₀ / (2 * k₀ + k₁)^2
    @test d_auto ≈ d_analytic rtol = 1.0e-12
end

@testset "LayeredSphere — Symbolic (SymPy) bulk single-layer" begin
    @syms kk0 mm0 kk1 mm1
    C₀ = TensISO{3}(3 * kk0, 2 * mm0)
    C₁ = TensISO{3}(3 * kk1, 2 * mm1)
    s = LayeredSphere((Sym(1),), (C₁,))
    α_sym = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, kk0, mm0)[1]
    # Expected: (3k0 + 4m0)/(3k1 + 4m0)
    expected = (3 * kk0 + 4 * mm0) / (3 * kk1 + 4 * mm0)
    @test simplify(α_sym - expected) == 0
end

@testset "LayeredSphere — Symbolic conductivity single-layer" begin
    @syms ka kb
    s = LayeredSphere((Sym(1),), (TensISO{3}(kb),))
    α_sym = MeanFieldHomogenization.LayeredSpheres._cond_localization(s, ka)[1]
    expected = 3 * ka / (2 * ka + kb)
    @test simplify(α_sym - expected) == 0
end
