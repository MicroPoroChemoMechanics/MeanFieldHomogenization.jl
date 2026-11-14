using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Symbolics

# =============================================================================
#  test_symbolics_num.jl — the layered sphere on `Symbolics.Num`. `Num <: Real`
#  without being ordered, so every comparison on a modulus or a radius has to
#  be routed around: the recurrences must build closed forms, and the entry
#  points that do need a comparison (locating a radius) must refuse a symbol.
#  References: closed forms, and the numeric path at substituted values.
# =============================================================================

@testset "LayeredSphere — Symbolics.Num" begin
    Symbolics.@variables k₀ μ₀ k₁ μ₁ k₂ μ₂ a R kₙ kₜ κˢ μˢ ρ κ c₀ c₁ c₂
    num(x, vals) = Float64(Symbolics.value(Symbolics.substitute(x, vals; fold = Val(true))))
    zero_expr(x) = isequal(Symbolics.simplify(x; expand = true), 0)
    C(k, μ) = TensISO{3}(3k, 2μ)
    vals = Dict(
        k₀ => 2.0, μ₀ => 1.5, k₁ => 10.0, μ₁ => 4.0, k₂ => 3.0, μ₂ => 1.0, a => 0.6, R => 1.0,
        kₙ => 10.0, kₜ => 4.0, κˢ => 1.3, μˢ => 0.7, ρ => 0.2, κ => 0.3, c₀ => 2.0, c₁ => 5.0, c₂ => 1.0,
    )
    # one layer: Eshelby's sphere, in closed form
    s₁ = LayeredSphere((R,), (C(k₁, μ₁),))
    α, β = TensND.get_data(gradient_gradient_concentration(s₁, C(k₀, μ₀), C(k₀, μ₀)))
    @test zero_expr(α - (3k₀ + 4μ₀) / (3k₁ + 4μ₀))
    β_ref = 5μ₀ * (3k₀ + 4μ₀) / (μ₀ * (9k₀ + 8μ₀) + 6μ₁ * (k₀ + 2μ₀))
    @test num(β, vals) ≈ num(β_ref, vals) rtol = 1.0e-12
    # a grain bonded by springs: its bulk modulus becomes k₁/(1 + 3k₁/(kₙR))
    grain = LayeredSphere((R,), (C(k₁, μ₁),); interfaces = (SpringInterface(kₙ, kₜ),))
    αg = TensND.get_data(gradient_gradient_concentration(grain, C(k₀, μ₀), C(k₀, μ₀)))[1]
    @test zero_expr(αg - (3k₀ + 4μ₀) / (3k₁ / (1 + 3k₁ / (kₙ * R)) + 4μ₀))
    # two layers with a spring and a membrane: every tensor against the numbers
    s₂ = LayeredSphere((a, R), (C(k₁, μ₁), C(k₂, μ₂)); interfaces = (SpringInterface(kₙ, kₜ), MembraneInterface(κˢ, μˢ)))
    n₂ = LayeredSphere(
        (0.6, 1.0), (C(10.0, 4.0), C(3.0, 1.0));
        interfaces = (SpringInterface(10.0, 4.0), MembraneInterface(1.3, 0.7))
    )
    C₀, C₀n = C(k₀, μ₀), C(2.0, 1.5)
    for g in (gradient_gradient_concentration, flux_gradient_concentration, MeanFieldHomogenization.property_contribution)
        @test [num(x, vals) for x in TensND.get_data(g(s₂, C₀, C₀))] ≈ collect(TensND.get_data(g(n₂, C₀n, C₀n))) rtol = 1.0e-10
    end
    for k in 1:2
        @test [num(x, vals) for x in TensND.get_data(gradient_gradient_concentration(s₂, C₀; layer = k, external = true))] ≈
            collect(TensND.get_data(gradient_gradient_concentration(n₂, C₀n; layer = k, external = true))) rtol = 1.0e-10
    end
    # conduction: Kapitza grain k₁/(1 + ρk₁/R), conducting surface k₁ + 2κ/R
    K(x) = TensISO{3}(x)
    αK = TensND.get_data(gradient_gradient_concentration(LayeredSphere((R,), (K(c₁),); interfaces = (KapitzaInterface(ρ),)), K(c₀), K(c₀)))[1]
    @test zero_expr(αK - 3c₀ / (2c₀ + c₁ / (1 + ρ * c₁ / R)))
    NS = TensND.get_data(MeanFieldHomogenization.property_contribution(LayeredSphere((R,), (K(c₁),); interfaces = (SurfaceConductiveInterface(κ),)), K(c₀)))[1]
    @test zero_expr(NS - (c₁ + 2κ / R - c₀) * 3c₀ / (2c₀ + c₁ + 2κ / R))
    sK = LayeredSphere((a, R), (K(c₁), K(c₂)); interfaces = (KapitzaInterface(ρ), SurfaceConductiveInterface(κ)))
    nK = LayeredSphere((0.6, 1.0), (K(5.0), K(1.0)); interfaces = (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3)))
    @test num(TensND.get_data(MeanFieldHomogenization.property_contribution(sK, K(c₀)))[1], vals) ≈
        TensND.get_data(MeanFieldHomogenization.property_contribution(nK, K(2.0)))[1] rtol = 1.0e-10
    # pointwise fields: the region is named, a symbolic radius is refused
    f = LayeredSphereFields(s₂, C₀)
    @test_throws ArgumentError local_gradient_gradient_concentration(f, a / 2, 0.4, 0.9)
    @test_throws ArgumentError get_layer(s₂, a / 2)
    A_loc = local_gradient_gradient_concentration(f, a / 2, 0.4, 0.9; layer = 1)
    A_num = local_gradient_gradient_concentration(LayeredSphereFields(n₂, C₀n), 0.3, 0.4, 0.9)
    @test [num(x, vals) for x in TensND.walpole_coeffs(A_loc)] ≈ collect(TensND.walpole_coeffs(A_num)) rtol = 1.0e-10
    g = LayeredSphereTransportFields(sK, K(c₀))
    T_loc = local_temperature(g, 3a / 2, 0.4, 0.9, TensND.Vec{3}((0.3, -0.2, 0.9)); layer = 2)
    @test num(T_loc, vals) ≈ local_temperature(LayeredSphereTransportFields(nK, K(2.0)), 0.9, 0.4, 0.9, TensND.Vec{3}((0.3, -0.2, 0.9))) rtol = 1.0e-10
end
