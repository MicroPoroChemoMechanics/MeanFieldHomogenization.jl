# =============================================================================
#  test_elliptic_hole_conduction.jl — the insulating elliptic hole in an
#  anisotropic conductivity of the plane (`ConformalHole` of one term).
#
#  References, independent of the closed form the code evaluates:
#   1. the gradient concentration of the elliptic inclusion of the package
#      (`Ellipsoid` of the plane, through its Hill tensor), on every component
#      and on its derivatives, and the estimates of a cell built with it;
#   2. exact properties: the contribution K₀·A of an insulating hole is
#      symmetric, the rotation of the conductivity and the hole together turns
#      the concentration, an isotropic conductivity gives the same tensor
#      whatever its type, and nothing flows through the hole.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using Random

ehc_A(shape, K) = canonical_components(gradient_gradient_concentration(shape, 0 * K, K))
# The elliptic inclusion of the package with the shape of the map ζ + m/ζ.
ehc_ellipse(m) = Ellipsoid(1 + abs(m), 1 - abs(m); angle = angle(m) / 2)
ehc_general(K₁₂ = 0.7) = Tens([3.0 K₁₂; K₁₂ 1.2])
ehc_ortho(K₁ = 3.0, β = 0.4) = TensOrtho{2}(K₁, 0.5, Basis(β))
ehc_rotation(β) = [cos(β) -sin(β); sin(β) cos(β)]

# A cell of the plane: a matrix of conductivity K₀ and a fraction f of holes.
function ehc_cell(K₀, hole, f)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0; dim = 2), Dict(:K => K₀); fraction = :rest)
    add_phase!(rve, :holes, hole, Dict(:K => 0 * K₀); fraction = f)
    return rve
end

@testset "Insulating elliptic hole in an anisotropic conductivity" begin
    rng = Xoshiro(20261129)
    shapes = (0.4, 0.3 * cis(0.9), 0.0)

    @testset "against the Hill tensor of the ellipse" begin
        for K in (ehc_general(), ehc_ortho()), m in shapes
            @test ehc_A(ConformalHole((m,)), K) ≈ ehc_A(ehc_ellipse(m), K) rtol = 1.0e-14
        end
        # orthotropic conductivities in random frames
        for β in 2π .* rand(rng, 3), m in shapes
            K = ehc_ortho(3.0, β)
            @test ehc_A(ConformalHole((m,)), K) ≈ ehc_A(ehc_ellipse(m), K) rtol = 1.0e-12
        end
        # the concentration is not symmetric, the contribution K₀·A is
        K = ehc_general()
        A = ehc_A(ConformalHole((0.3 * cis(0.9),)), K)
        @test norm(A - transpose(A)) > 1.0e-3 * norm(A)
        KA = Array(canonical_components(K)) * Array(A)
        @test KA ≈ transpose(KA) rtol = 1.0e-13
        # nothing flows through the hole
        @test iszero(canonical_components(flux_gradient_concentration(ConformalHole((0.4,)), 0 * K, K)))
    end

    @testset "an isotropic conductivity of any type" begin
        κ = TensISO{2}(1.3)
        general = Tens(Array(canonical_components(κ)))
        for m in shapes
            @test ehc_A(ConformalHole((m,)), general) ≈ ehc_A(ConformalHole((m,)), κ) rtol = 1.0e-14
        end
        # a map with several terms takes it as the isotropic tensor it is
        two = ConformalHole((0.2, 0.1))
        @test ehc_A(two, general) ≈ ehc_A(two, κ) rtol = 1.0e-12
    end

    @testset "rotation" begin
        # the conductivity and the hole turned together by β: the hole of map
        # ζ + m e^{2iβ}/ζ, and the concentration turned
        m = 0.3 * cis(0.9)
        for K in (ehc_general(), ehc_ortho()), β in 2π .* rand(rng, 2)
            R = ehc_rotation(β)
            turned = Tens(R * canonical_components(K) * R')
            @test ehc_A(ConformalHole((m * cis(2β),)), turned) ≈ R * ehc_A(ConformalHole((m,)), K) * R' rtol = 1.0e-12
        end
    end

    @testset "derivatives" begin
        m = 0.3 * cis(0.9)
        # with respect to the coupling K₁₂ and to a principal conductivity
        coupling(shape) = K₁₂ -> ehc_A(shape, ehc_general(K₁₂))[1, 2]
        @test ForwardDiff.derivative(coupling(ConformalHole((m,))), 0.7) ≈ ForwardDiff.derivative(coupling(ehc_ellipse(m)), 0.7) rtol = 1.0e-8
        principal(shape) = K₁ -> ehc_A(shape, ehc_ortho(K₁))[2, 1]
        @test ForwardDiff.derivative(principal(ConformalHole((m,))), 3.0) ≈ ForwardDiff.derivative(principal(ehc_ellipse(0.3 * cis(0.9))), 3.0) rtol = 1.0e-8
        # with respect to the shape
        shape(a) = ehc_A(ConformalHole((a,)), ehc_general())[1, 1]
        hill(a) = ehc_A(Ellipsoid(1 + a, 1 - a; angle = 0.0), ehc_general())[1, 1]
        @test ForwardDiff.derivative(shape, 0.4) ≈ ForwardDiff.derivative(hill, 0.4) rtol = 1.0e-8
    end

    @testset "a cell of the plane" begin
        # aligned holes in an anisotropic matrix: the same estimates with the
        # map and with the ellipse of the package
        f = 0.05
        for K₀ in (ehc_general(), ehc_ortho()), m in (0.4, 0.3 * cis(0.9)), scheme in (Dilute(), MoriTanaka())
            with_map = homogenize(ehc_cell(K₀, ConformalHole((m,)), f), scheme, :K)
            with_ellipse = homogenize(ehc_cell(K₀, ehc_ellipse(m), f), scheme, :K)
            @test canonical_components(with_map) ≈ canonical_components(with_ellipse) rtol = 1.0e-12
        end
    end

    @testset "inference and refusals" begin
        hole = ConformalHole((0.3 * cis(0.9),))
        for K in (ehc_general(), ehc_ortho())
            # a tensor of the plane whose storage, symmetric or not, follows its values
            inferred = only(Base.return_types(gradient_gradient_concentration, typeof.((hole, 0 * K, K))))
            @test inferred <: TensND.TensCanonical{2, 2, Float64}
        end
        # a conductivity that is not symmetric
        K = Tens([3.0 0.7; 0.2 1.2])
        @test_throws ArgumentError gradient_gradient_concentration(hole, 0 * K, K)
        # a map with two terms in an anisotropic conductivity
        K = ehc_general()
        @test_throws ArgumentError gradient_gradient_concentration(ConformalHole((0.2, 0.1)), 0 * K, K)
    end
end
