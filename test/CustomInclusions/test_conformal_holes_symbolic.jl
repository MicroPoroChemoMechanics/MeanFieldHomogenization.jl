# =============================================================================
#  test_conformal_holes_symbolic.jl — a conformal hole with symbolic
#  coefficients: the system of the potentials is solved on `Symbolics.Num`, and
#  the condition on the coefficients, a comparison, is left to the caller.
#  References: the closed forms of the ellipse and the numeric path. The
#  properties of the plane of a plane problem, on symbols too.
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
    # a transversely isotropic matrix about the axis, with symbolic components
    Symbolics.@variables c₁ c₂ c₃ c₄ c₅
    values = Dict(c₁ => 5.2, c₂ => 1.9, c₃ => 1.4, c₄ => 9.0, c₅ => 1.1)
    Cs = ti_from_constants(c₁, c₂, c₃, c₄, c₅, (1, 0, 0))
    Cn = ti_from_constants(5.2, 1.9, 1.4, 9.0, 1.1, (1.0, 0.0, 0.0))
    square = ConformalHole(polygon_map(4, 2)...)
    A = canonical_components(gradient_gradient_concentration(square, 0 * Cs, Cs))
    reference = canonical_components(gradient_gradient_concentration(square, 0 * Cn, Cn))
    @test [value(x, values) for x in A] ≈ reference rtol = 1.0e-10
    # the circle under a hydrostatic strain e of the plane: the remote stress is 2k e,
    # and each strain of the hole (κ + 1)/(4μ) × 2k e = (1 + k/μ) e
    circle = canonical_components(gradient_gradient_concentration(ConformalHole(()), 0 * Cs, Cs))
    k, μ = (c₁ + c₂) / 2, (c₁ - c₂) / 2
    @test isequal(Symbolics.simplify(circle[2, 2, 2, 2] + circle[2, 2, 3, 3] - (1 + k / μ); expand = true), 0)
end

@testset "Conformal holes of the plane on symbols" begin
    Symbolics.@variables E ν m
    value(x, v) = Symbolics.value(Symbolics.substitute(x, v))
    # a circular hole in a plate, in plane stress, under a tension: 3/E along it, -1/E
    # across, whatever E and ν (the expressions are evaluated: expanding them is costly)
    c = plane_stress_stiffness(iso_stiffness_young_poisson(E, ν))
    H = canonical_components(gradient_gradient_concentration(ConformalHole(()), 0 * c, c) ⊡ inv(c))
    for at in (Dict(E => 70.0, ν => 0.25), Dict(E => 3.0, ν => 0.4))
        @test value(H[1, 1, 1, 1] * E, at) ≈ 3
        @test value(H[2, 2, 1, 1] * E, at) ≈ -1
    end
    # an elliptic hole with a symbolic coefficient, against the numeric path
    cn = plane_stress_stiffness(iso_stiffness_young_poisson(70.0, 0.25))
    A = canonical_components(gradient_gradient_concentration(ConformalHole((m,)), 0 * cn, cn))
    @test [value(x, Dict(m => 0.3)) for x in A] ≈ canonical_components(gradient_gradient_concentration(ConformalHole((0.3,)), 0 * cn, cn)) rtol = 1.0e-10
    # a symbolic reference of the plane is a TensISO, or is refused
    @test_throws ArgumentError gradient_gradient_concentration(ConformalHole(()), 0 * c, Tens(canonical_components(c)))
end
