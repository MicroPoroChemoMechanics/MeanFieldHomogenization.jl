# =============================================================================
#  test_conformal_holes_symbolic.jl — a conformal hole with symbolic
#  coefficients: the system of the potentials is solved on `Symbolics.Num`, and
#  the condition on the coefficients, a comparison, is left to the caller.
#  References: the closed forms of the ellipse and the numeric path.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
import Symbolics

@testset "Conformal holes on symbols" begin
    Symbolics.@variables m p
    value(x, v) = Symbolics.value(Symbolics.substitute(x, v))
    hole = ConformalHole((m,))
    @test hole isa ConformalHole{Symbolics.Num}
    # conduction of the ellipse: 2/(1 + m) and 2/(1 - m) in its plane
    Ak = canonical_components(gradient_gradient_concentration(hole, TensISO{3}(0.0), TensISO{3}(1.0)))
    @test isequal(Symbolics.simplify(Ak[2, 2] - 2 / (1 + m)), 0)
    @test isequal(Symbolics.simplify(Ak[3, 3] - 2 / (1 - m)), 0)
    # elasticity, against the numeric path at a value
    C₀ = iso_stiffness(1.7, 0.6)
    A = canonical_components(gradient_gradient_concentration(hole, 0 * C₀, C₀))
    reference = canonical_components(gradient_gradient_concentration(ConformalHole((0.3,)), 0 * C₀, C₀))
    @test [value(x, Dict(m => 0.3)) for x in A] ≈ reference rtol = 1.0e-10
    # two coefficients, one of them symbolic
    two = ConformalHole(1 => 0.1, 3 => p)
    A = canonical_components(gradient_gradient_concentration(two, 0 * C₀, C₀))
    reference = canonical_components(gradient_gradient_concentration(ConformalHole(1 => 0.1, 3 => 0.05), 0 * C₀, C₀))
    @test [value(x, Dict(p => 0.05)) for x in A] ≈ reference rtol = 1.0e-9
    # no condition is checked on symbols
    @test ConformalHole(2 => 3 * m) isa ConformalHole
end
