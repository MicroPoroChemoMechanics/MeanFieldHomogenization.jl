# =============================================================================
#  test_layered_ellipsoid_symbolic.jl — the confocal layered ellipsoid with
#  symbolic conductivities: the relations between the mean gradients have no
#  comparison, and are solved on `Symbolics.Num`. Reference: the numeric path.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
import Symbolics

@testset "Layered ellipsoid on symbols" begin
    Symbolics.@variables k₁ k₂ k₀
    numeric = layered_ellipsoid_from_fractions((0.5, 0.8, 1.3), (0.4, 0.6), (TensISO{3}(50.0), TensISO{3}(0.2)))
    symbolic = LayeredEllipsoid(numeric.semi_axes, (TensISO{3}(k₁), TensISO{3}(k₂)))
    at = Dict(k₁ => 50.0, k₂ => 0.2, k₀ => 1.0)
    value(x) = Symbolics.value(Symbolics.substitute(x, at; fold = Val(true)))
    for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
        S = canonical_components(f(symbolic, TensISO{3}(k₀), TensISO{3}(k₀)))
        N = canonical_components(f(numeric, TensISO{3}(1.0), TensISO{3}(1.0)))
        @test [value(S[i, i]) for i in 1:3] ≈ [N[i, i] for i in 1:3] rtol = 1.0e-12
    end
    # Symbolic semi-axes are taken as given, and the family from fractions is refused.
    Symbolics.@variables a
    @test LayeredEllipsoid(((a, a, a),), (TensISO{3}(k₁),)) isa LayeredEllipsoid{Symbolics.Num}
    @test_throws "symbols do not" layered_ellipsoid_from_fractions((a, 0.8, 1.3), (0.4, 0.6), (TensISO{3}(1.0), TensISO{3}(2.0)))
end
