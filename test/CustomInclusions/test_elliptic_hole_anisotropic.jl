# =============================================================================
#  test_elliptic_hole_anisotropic.jl — the elliptic hole in an anisotropic
#  matrix of the plane (Lekhnitskii's complex potentials).
#
#  References, independent of the closed forms the code evaluates:
#   1. the characteristic pair (s, p): the factorization of the quartic of the
#      compliances, the closed forms of the orthotropic axes, and the values
#      printed by Lekhnitskii (1981, Eq. (31.17)) for a birch plywood,
#      data/literature/lekhnitskii1981.json;
#   2. the Hill tensor of the elliptic inclusion of the package (another
#      algorithm, through the Green operator of the anisotropic plane), on
#      every component, its derivatives included;
#   3. exact properties: the free surface of the hole, the rotation of the
#      matrix, the hole and the load together;
#   4. the stresses of Lekhnitskii (1981, Eqs. (31.13)-(31.16)) at the ends of
#      the axes, and his Tables 8, 9 and 10 for a hole of axis ratio 2;
#   5. at isotropy, Muskhelishvili's hoop stress of the elliptic hole under a
#      uniaxial tension (1977, section 82a, example 1).
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON
using Random

const EHA = MeanFieldHomogenization.ConformalHoles

# Four stiffnesses of the plane: without symmetry, orthotropic and square off
# their axes, isotropic.
eha_general() = from_mandel([30.0 4.0 3.0; 4.0 9.0 -2.0; 3.0 -2.0 7.0])
eha_ortho(β = 0.37) = TensOrtho(40.0, 6.0, 2.5, 3.0, Basis(β))
eha_square(β = 1.1) = cubic_from_constants(20.0, 12.0, 9.0, Basis(β))
eha_iso() = iso_stiffness(Val(2), 5.0, 2.0)
eha_stiffnesses() = (eha_general(), eha_ortho(), eha_square(), eha_iso())
# The same tensor as a general one: the numeric route of the characteristic pair.
eha_copy(C) = Tens(Array(canonical_components(C)))

eha_A(shape, C) = canonical_components(gradient_gradient_concentration(shape, 0 * C, C))
# The elliptic inclusion of the package with the shape of the map ζ + m/ζ.
eha_ellipse(m) = Ellipsoid(1 + abs(m), 1 - abs(m); angle = angle(m) / 2)

eha_rotation(β) = [cos(β) -sin(β); sin(β) cos(β)]
# Components of a tensor of order 4 turned by R, by sums over the indices.
eha_turn(A, R) = [
    sum(R[i, a] * R[j, b] * R[k, c] * R[l, d] * A[a, b, c, d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
]

# A uniaxial tension of unit intensity at the angle φ from e₁.
eha_uniaxial(φ) = (d = [cos(φ), sin(φ)]; Tens(d * d'))

eha_literature() =
    JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "lekhnitskii1981.json"))
eha_quantity(name) = eha_literature()["quantities"][name]["value"]

# Plane-stress stiffness of an orthotropic sheet of engineering constants
# (E₁, E₂, G, ν₁) whose stiff axis makes the angle ψ with e₁: the compliance
# written in the material frame, read in the canonical one, inverted.
function eha_sheet(E₁, E₂, G, ν₁, ψ)
    S = [1 / E₁ -ν₁ / E₁ 0; -ν₁ / E₁ 1 / E₂ 0; 0 0 1 / (2G)]
    return inv(express(from_mandel(S, Basis(ψ)), CanonicalBasis{2, Float64}()))
end
eha_plywood(ψ) = eha_sheet(eha_quantity("E1"), eha_quantity("E2"), eha_quantity("G"), eha_quantity("nu1"), ψ)

# Lekhnitskii's parameters of Eq. (31.1), the first axis along the larger modulus.
eha_k(E₁, E₂) = sqrt(E₁ / E₂)
eha_n(E₁, E₂, G, ν₁) = sqrt(2 * eha_k(E₁, E₂) + E₁ / G - 2ν₁)

# The characteristic quartic of the compliances of C at μ, factored with the
# pair (s, p) and as it is written.
function eha_quartic(C, μ)
    a₁₁, a₂₂, a₁₂, a₆₆, a₁₆, a₂₆ = EHA._canonical_compliances(C)
    s, p = EHA._characteristic_pair(C)
    factored = a₁₁ * (μ^2 - s * μ + p) * (μ^2 - conj(s) * μ + conj(p))
    return factored, a₁₁ * μ^4 - 2a₁₆ * μ^3 + (2a₁₂ + a₆₆) * μ^2 - 2a₂₆ * μ + a₂₂
end

# Muskhelishvili's hoop stress on the hole of map ζ + m/ζ, m real, under a unit
# uniaxial tension at the angle α from e₁, at the point of parameter θ.
eha_muskhelishvili(m, α, θ) =
    (1 - m^2 + 2m * cos(2α) - 2cos(2(θ - α))) / (1 - 2m * cos(2θ) + m^2)

@testset "Elliptic hole in an anisotropic plane" begin
    rng = Xoshiro(20261127)

    @testset "characteristic pair" begin
        # the closed forms of the orthotropic and the square classes, against
        # the numeric roots of their general copies, in several frames
        for β in (0.0, 2π * rand(rng), 2π * rand(rng), 2π * rand(rng))
            for C in (eha_ortho(β), eha_square(β))
                s, p = EHA._characteristic_pair(C)
                s₀, p₀ = EHA._characteristic_pair(eha_copy(C))
                @test s ≈ s₀ rtol = 1.0e-12
                @test p ≈ p₀ rtol = 1.0e-12
            end
        end
        # the compliances read as strains: ε = 𝕊 : σ under the three unit stresses
        C = eha_general()
        a₁₁, a₂₂, a₁₂, a₆₆, a₁₆, a₂₆ = EHA._canonical_compliances(C)
        strain(S) = canonical_components(inv(C) ⊡ Tens(S))
        ε = strain([1.0 0.0; 0.0 0.0])
        @test [ε[1, 1], ε[2, 2], 2ε[1, 2]] ≈ [a₁₁, a₁₂, a₁₆] rtol = 1.0e-12
        ε = strain([0.0 0.0; 0.0 1.0])
        @test [ε[1, 1], ε[2, 2], 2ε[1, 2]] ≈ [a₁₂, a₂₂, a₂₆] rtol = 1.0e-12
        ε = strain([0.0 1.0; 1.0 0.0])
        @test [ε[1, 1], ε[2, 2], 2ε[1, 2]] ≈ [a₁₆, a₂₆, a₆₆] rtol = 1.0e-12
        # the quartic is a₁₁ (μ² − sμ + p)(μ² − s̄μ + p̄), and the two roots of
        # the first factor lie in the upper half-plane
        for C in eha_stiffnesses()
            for μ in randn(rng, ComplexF64, 4)
                factored, quartic = eha_quartic(C, μ)
                @test factored ≈ quartic rtol = 1.0e-11
            end
            s, p = EHA._characteristic_pair(C)
            δ = sqrt(s^2 - 4p)
            @test imag((s + δ) / 2) > 0 && imag((s - δ) / 2) > 0
        end
        # isotropy: the double root i, exactly
        @test EHA._characteristic_pair(eha_iso()) === (complex(0.0, 2.0), complex(-1.0))
        # the plywood of Lekhnitskii, Eq. (31.17): s = i n and p = −k in the
        # axes of the sheet
        E₁, E₂, G, ν₁ = eha_quantity("E1"), eha_quantity("E2"), eha_quantity("G"), eha_quantity("nu1")
        k, n = eha_k(E₁, E₂), eha_n(E₁, E₂, G, ν₁)
        s, p = EHA._characteristic_pair(eha_plywood(0.0))
        @test s ≈ im * n rtol = 1.0e-12
        @test p ≈ -k rtol = 1.0e-12
        @test n ≈ eha_quantity("n") rtol = 1.0e-3
        @test k ≈ eha_quantity("k") rtol = 1.0e-3
        # the two roots, printed with three significant digits
        δ = sqrt(s^2 - 4p)
        μ₁, μ₂ = (s + δ) / 2, (s - δ) / 2
        @test abs(real(μ₁)) + abs(real(μ₂)) < 1.0e-12
        @test imag(μ₁) ≈ eha_quantity("mu1_imag") atol = 2.0e-3
        @test imag(μ₂) ≈ eha_quantity("mu2_imag") atol = 2.0e-3
        # refused: a stiffness without the major symmetry, and one that is not
        # positive definite (μ⁴ + 2μ² − 1 has two real roots)
        A = Array(canonical_components(eha_general()))
        A[1, 1, 2, 2] += 1.5
        @test_throws ArgumentError EHA._characteristic_pair(Tens(A))
        @test_throws ArgumentError gradient_gradient_concentration(ConformalHole((0.3,)), 0 * Tens(A), Tens(A))
        @test_throws DomainError EHA._characteristic_pair(from_mandel([1.0 0 0; 0 -1.0 0; 0 0 1.0]))
    end

    @testset "the ellipse against its Hill tensor" begin
        for C in eha_stiffnesses(), m in (0.4, 0.3 * cis(0.9), 0.0)
            @test eha_A(ConformalHole((m,)), C) ≈ eha_A(eha_ellipse(m), C) rtol = 1.0e-10
        end
        C = eha_ortho()
        # a map with trailing zeros is still the ellipse
        @test eha_A(ConformalHole((0.3, 0.0)), C) ≈ eha_A(ConformalHole((0.3,)), C) rtol = 1.0e-14
        # an isotropic stiffness typed as a general tensor: the double root
        # found numerically, against the path of the isotropic matrix
        for m in (0.4, 0.3 * cis(0.9))
            @test eha_A(ConformalHole((m,)), eha_copy(eha_iso())) ≈ eha_A(ConformalHole((m,)), eha_iso()) rtol = 1.0e-10
        end
        # a map with two terms is solved in an isotropic matrix only
        @test_throws ArgumentError gradient_gradient_concentration(ConformalHole((0.2, 0.1)), 0 * C, C)
        # nothing passes through the hole
        @test iszero(canonical_components(flux_gradient_concentration(ConformalHole((0.3,)), 0 * C, C)))
        # the matrix and the hole turned together by β: the hole of map
        # ζ + m e^{2iβ}/ζ, and the concentration turned
        for C in (eha_general(), eha_ortho()), β in (2π * rand(rng), 2π * rand(rng))
            R = eha_rotation(β)
            m = 0.3 * cis(0.9)
            turned = Tens(eha_turn(canonical_components(C), R))
            @test eha_A(ConformalHole((m * cis(2β),)), turned) ≈ eha_turn(eha_A(ConformalHole((m,)), C), R) rtol = 1.0e-10
        end
    end

    @testset "stress along the contour, anisotropic" begin
        # the contour is free: with the tangent t = dω(e^{iθ})/dθ, the traction
        # σ·n, n ∝ (t₂, −t₁), vanishes
        C = eha_general()
        s, p = EHA._characteristic_pair(C)
        for m in (complex(0.4), 0.3 * cis(0.9)), Σ in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, -0.7, 0.5))
            for θ in 2π .* rand(rng, 5)
                (σ₁₁, σ₂₂, σ₁₂), (t₁′, t₂′) = EHA._anisotropic_ellipse_stress(m, s, p, Σ, θ)
                t = im * (cis(θ) - m * cis(-θ))
                t₁, t₂ = real(t), imag(t)
                @test abs(t₁′ * t₂ - t₂′ * t₁) < 1.0e-13             # the tangent returned
                @test abs(σ₁₁ * t₂ - σ₁₂ * t₁) < 1.0e-11
                @test abs(σ₁₂ * t₂ - σ₂₂ * t₁) < 1.0e-11
            end
        end

        # Lekhnitskii (31.13)-(31.16): the sheet pulled across the major axis,
        # its stiff axis along (ψ = 0) or across (ψ = π/2) that axis
        data = eha_literature()
        E₁, E₂, G, ν₁ = (data["quantities"][q]["value"] for q in ("E1", "E2", "G", "nu1"))
        k, n = eha_k(E₁, E₂), eha_n(E₁, E₂, G, ν₁)
        c = data["quantities"]["axis_ratio"]["value"]
        hole = ConformalHole(((c - 1) / (c + 1),))              # a/b = c, a along e₁
        pull = eha_uniaxial(π / 2)
        @test contour_stress(hole, pull, 0.0, eha_plywood(0.0)) ≈ 1 + c * n / k rtol = 1.0e-12
        @test contour_stress(hole, pull, π / 2, eha_plywood(0.0)) ≈ -k rtol = 1.0e-12
        @test contour_stress(hole, pull, 0.0, eha_plywood(π / 2)) ≈ 1 + c * n rtol = 1.0e-12
        @test contour_stress(hole, pull, π / 2, eha_plywood(π / 2)) ≈ -1 / k rtol = 1.0e-12

        # Tables 8 and 9: the largest tension and compression on the edge, for
        # the pull at φ and the stiff axis at ψ from the major axis. The
        # columns ψ = 0, 30°, 150°, 180° are compared; on the cell φ = 90°,
        # ψ = 0 (or 180°) the table prints 7.20 where Eq. (31.13) of the same
        # book gives 1 + c n/k = 7.298, which is the value checked there.
        θs = range(0, 2π; length = 7201)
        tension, compression = data["tables"]["max_tension_plywood"], data["tables"]["max_compression_plywood"]
        for ψ in (0, 30, 150, 180)
            C = eha_plywood(deg2rad(ψ))
            column = "psi_$ψ"
            for (row, row′) in zip(tension, compression)
                φ = row["phi_deg"]
                σ = contour_stress(hole, eha_uniaxial(deg2rad(φ)), θs, C)
                if φ == 90 && ψ in (0, 180)
                    @test maximum(σ) ≈ 1 + c * n / k rtol = 1.0e-8
                else
                    @test maximum(σ) ≈ row[column] atol = 0.015
                end
                ψ in (0, 180) && @test -minimum(σ) ≈ row′[column] atol = 0.011
            end
        end

        # at isotropy, typed as a general tensor: Muskhelishvili's hoop stress
        Ciso = eha_copy(eha_iso())
        for m in (1 / 3, -0.4), α in (0.0, 0.6, π / 2), θ in (0.0, 0.4, 1.3, 2.9)
            @test contour_stress(ConformalHole((m,)), eha_uniaxial(α), θ, Ciso) ≈ eha_muskhelishvili(m, α, θ) atol = 1.0e-9
        end
        # Table 10 on the pulls along an axis and at 30°, where the two printed
        # decimals are those of the closed form; along an axis the largest
        # compression is the pull itself.
        for row in data["tables"]["max_tension_isotropic"]
            φ = row["phi_deg"]
            φ in (0, 30, 90) || continue
            σ = contour_stress(hole, eha_uniaxial(deg2rad(φ)), θs, Ciso)
            @test maximum(σ) ≈ row["sigma_max_over_p"] atol = 0.011
            φ in (0, 90) && @test minimum(σ) ≈ -1 atol = 1.0e-4
        end

        # everything turned by β: the matrix, the load and the hole
        C = eha_general()
        S = [0.7 0.3; 0.3 -0.4]
        m = 0.3 * cis(0.9)
        for β in (2π * rand(rng), 2π * rand(rng))
            R = eha_rotation(β)
            turned = Tens(eha_turn(canonical_components(C), R))
            for θ in (0.0, 1.0, 2.5, 4.4)
                @test contour_stress(ConformalHole((m * cis(2β),)), Tens(R * S * R'), θ + β, turned) ≈
                    contour_stress(ConformalHole((m,)), Tens(S), θ, C) rtol = 1.0e-10
            end
        end

        # a collection of angles gives the values one by one
        θs = [0.0, 0.7, 2.1, 5.0]
        h = ConformalHole((m,))
        expected = [contour_stress(h, Tens(S), θ, C) for θ in θs]
        @test contour_stress(h, Tens(S), θs, C) == expected
        @test contour_stress(h, Tens(S), range(0, 2π; length = 5), C) ==
            [contour_stress(h, Tens(S), θ, C) for θ in range(0, 2π; length = 5)]
        # an isotropic stiffness of its own type takes the isotropic route,
        # which holds a map with several terms
        two = ConformalHole((0.2, 0.1))
        @test contour_stress(two, Tens(S), 0.7, eha_iso()) == contour_stress(two, Tens(S), 0.7)
        @test_throws ArgumentError contour_stress(two, Tens(S), 0.7, eha_ortho())
    end

    @testset "derivatives" begin
        # with respect to a modulus of the matrix: the numeric roots of a
        # general tensor carry the dual, against the Hill tensor
        hole(x) = eha_A(ConformalHole((0.4,)), eha_copy(TensOrtho(x, 6.0, 2.5, 3.0, Basis(0.37))))[1, 1, 2, 2]
        hill(x) = eha_A(Ellipsoid(1.4, 0.6; angle = 0.0), TensOrtho(x, 6.0, 2.5, 3.0, Basis(0.37)))[1, 1, 2, 2]
        @test ForwardDiff.derivative(hole, 40.0) ≈ ForwardDiff.derivative(hill, 40.0) rtol = 1.0e-7
        # with respect to the shape
        C = eha_general()
        shape_hole(m) = eha_A(ConformalHole((m,)), C)[1, 1, 1, 1]
        shape_hill(m) = eha_A(Ellipsoid(1 + m, 1 - m; angle = 0.0), C)[1, 1, 1, 1]
        @test ForwardDiff.derivative(shape_hole, 0.3) ≈ ForwardDiff.derivative(shape_hill, 0.3) rtol = 1.0e-7
        @test ForwardDiff.derivative(m -> ForwardDiff.derivative(shape_hole, m), 0.3) ≈
            ForwardDiff.derivative(m -> ForwardDiff.derivative(shape_hill, m), 0.3) rtol = 1.0e-5
        # nested, with respect to the matrix
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(hole, x), 40.0) ≈
            ForwardDiff.derivative(x -> ForwardDiff.derivative(hill, x), 40.0) rtol = 1.0e-5
        # Eq. (31.13) differentiated in E₁: d(1 + c n/k)/dE₁, with
        # dk/dE₁ = k/(2E₁) and dn/dE₁ = (2 dk/dE₁ + 1/G)/(2n)
        E₁, E₂, G, ν₁ = eha_quantity("E1"), eha_quantity("E2"), eha_quantity("G"), eha_quantity("nu1")
        c = eha_quantity("axis_ratio")
        tip(E) = contour_stress(ConformalHole(((c - 1) / (c + 1),)), eha_uniaxial(π / 2), 0.0, eha_sheet(E, E₂, G, ν₁, 0.0))
        k, n = eha_k(E₁, E₂), eha_n(E₁, E₂, G, ν₁)
        dk = k / (2E₁)
        dn = (2dk + 1 / G) / (2n)
        @test ForwardDiff.derivative(tip, E₁) ≈ c * (dn * k - n * dk) / k^2 rtol = 1.0e-10
    end

    @testset "inference" begin
        hole = ConformalHole((0.3 * cis(0.9),))
        Σ = Tens([0.7 0.3; 0.3 -0.4])
        for C in (eha_general(), eha_ortho())
            @inferred gradient_gradient_concentration(hole, 0 * C, C)
            @inferred contour_stress(hole, Σ, 0.3, C)
        end
        @inferred contour_stress(hole, Σ, 0.3)
    end
end
