# =============================================================================
#  test_elliptic_hole_anisotropic_symbolic.jl — the elliptic hole in an
#  anisotropic matrix of the plane, on `Symbolics.Num`.
#
#  The orthotropic class has its characteristic pair in closed form, so the
#  concentration and the stress along the contour are expressions of the
#  moduli and of the coefficient of the map. Reference: the numeric path, at
#  the values substituted. A general symbolic tensor has no closed-form pair
#  and is refused.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
import Symbolics

const EHAS = MeanFieldHomogenization.ConformalHoles

@testset "Elliptic hole in an anisotropic plane on symbols" begin
    Symbolics.@variables c₁₁ c₂₂ c₁₂ c₆₆ m k μ
    # `fold`: the square roots of the characteristic pair are evaluated, not kept.
    value(x, v) = Symbolics.value(Symbolics.substitute(x, v; fold = Val(true)))
    at = Dict(c₁₁ => 40.0, c₂₂ => 6.0, c₁₂ => 2.5, c₆₆ => 3.0, m => 0.3, k => 5.0, μ => 2.0)
    Σ = Tens([0.7 0.3; 0.3 -0.4])
    θ = 0.7
    hole, numeric_hole = ConformalHole((m,)), ConformalHole((0.3,))
    @test hole isa ConformalHole{Symbolics.Num}

    # an orthotropic matrix in its axes, then in a frame turned by a number
    for (frame, numeric_frame) in ((CanonicalBasis{2, Symbolics.Num}(), CanonicalBasis{2, Float64}()), (Basis(0.37), Basis(0.37)))
        Cs = TensOrtho(c₁₁, c₂₂, c₁₂, c₆₆, frame)
        Cn = TensOrtho(40.0, 6.0, 2.5, 3.0, numeric_frame)
        A = canonical_components(gradient_gradient_concentration(hole, 0 * Cs, Cs))
        reference = canonical_components(gradient_gradient_concentration(numeric_hole, 0 * Cn, Cn))
        @test [value(x, at) for x in A] ≈ reference rtol = 1.0e-10
        @test value(contour_stress(hole, Σ, θ, Cs), at) ≈ contour_stress(numeric_hole, Σ, θ, Cn) rtol = 1.0e-10
    end

    # a symbolic tensor without structure: its quartic is not solved
    Cg = Tens(canonical_components(TensOrtho(c₁₁, c₂₂, c₁₂, c₆₆, CanonicalBasis{2, Symbolics.Num}())))
    @test_throws ArgumentError gradient_gradient_concentration(numeric_hole, 0 * Cg, Cg)
    @test_throws ArgumentError contour_stress(numeric_hole, Σ, θ, Cg)

    # an isotropic matrix: the route of the isotropic class, and the solution of
    # the anisotropic one taken at its double root, both against the numbers
    Ci, Cin = iso_stiffness(Val(2), k, μ), iso_stiffness(Val(2), 5.0, 2.0)
    reference = canonical_components(gradient_gradient_concentration(numeric_hole, 0 * Cin, Cin))
    A = canonical_components(gradient_gradient_concentration(hole, 0 * Ci, Ci))
    @test [value(x, at) for x in A] ≈ reference rtol = 1.0e-10
    M = EHAS._anisotropic_ellipse_concentration(complex(m), Ci)
    @test [value(x, at) for x in M] ≈ mandel(gradient_gradient_concentration(numeric_hole, 0 * Cin, Cin), CanonicalBasis{2, Float64}()) rtol = 1.0e-10
    s, p = EHAS._characteristic_pair(Ci)
    @test isequal(s, complex(Symbolics.Num(0), Symbolics.Num(2))) && isequal(p, complex(Symbolics.Num(-1)))
    @test value(contour_stress(hole, Σ, θ, Ci), at) ≈ contour_stress(numeric_hole, Σ, θ) rtol = 1.0e-10
end
