# =============================================================================
#  test_hole_matrix_stress.jl — the stress in the matrix around a free hole of
#  the plane (`matrix_stress`), isotropic matrix of any map and anisotropic
#  matrix around the ellipse.
#
#  References, independent of the evaluations the code makes:
#   1. Kirsch's field around the circular hole under a uniaxial tension, in
#      polar coordinates, and its derivative in the distance;
#   2. the potentials of the elliptic hole in an isotropic matrix in closed
#      form (Muskhelishvili 1977, section 82), written here in the variable ζ
#      and differentiated by hand;
#   3. the stress along the contour, `contour_stress`, which follows from
#      another formula: the hoop stress t·σ·t on the contour, and the free
#      surface σ·n = 0 there;
#   4. exact properties of any stress field of the matrix: equilibrium and,
#      in an isotropic matrix, the harmonic trace (by ForwardDiff), the remote
#      stress at infinity with a perturbation decaying as 1/r², linearity in
#      the load, and the rotation of the matrix, the hole, the load and the
#      point together;
#   5. Lekhnitskii (1981, Eq. (31.13)): the stress at the tip of the elliptic
#      hole in a birch plywood, data/literature/lekhnitskii1981.json.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using StaticArrays: SVector
using JSON
using Random

const HMS = MeanFieldHomogenization.ConformalHoles

# (σ₁₁, σ₂₂, σ₁₂) of a tensor of order 2 of the plane, in the canonical axes.
hms_components(σ) = (c = canonical_components(σ); SVector(c[1, 1], c[2, 2], c[1, 2]))
hms_tuple(Σ) = Tuple(hms_components(Σ))

# The map z = ω(ζ) = ζ + Σ mₖ ζ⁻ᵏ and its derivative.
hms_map(m, ζ) = ζ + sum(m[k] * ζ^(-k) for k in eachindex(m))
hms_map′(m, ζ) = 1 - sum(k * m[k] * ζ^(-k - 1) for k in eachindex(m))

# The point ω(e^{iθ}) of the contour, the unit tangent t ∝ dω/dθ = i e^{iθ} ω′
# and the normal n = (t₂, −t₁).
function hms_contour_point(m, θ)
    σ = cis(θ)
    z = hms_map(m, σ)
    w = im * σ * hms_map′(m, σ)
    t = w / abs(w)
    return [real(z), imag(z)], [real(t), imag(t)], [imag(t), -real(t)]
end

# Points of the matrix: images of circles |ζ| = ρ > 1, from close to the
# contour to far from it, on twelve directions.
hms_grid(m) = vec([(z = hms_map(m, ρ * cis(φ)); [real(z), imag(z)]) for ρ in (1.02, 1.3, 2.0, 4.0), φ in (0:11) .* (π / 6)])

# The two equilibrium equations ∂₁σ₁₁ + ∂₂σ₁₂ and ∂₁σ₁₂ + ∂₂σ₂₂ at x, and the
# largest derivative, which sets the scale of the residue.
function hms_equilibrium(f, x)
    J = ForwardDiff.jacobian(f, SVector{2}(x[1], x[2]))
    return (J[1, 1] + J[3, 2], J[3, 1] + J[2, 2]), maximum(abs, J)
end

hms_rotation(β) = [cos(β) -sin(β); sin(β) cos(β)]
# Components of a tensor of order 4 turned by R, by sums over the indices.
hms_turn(A, R) = [
    sum(R[i, a] * R[j, b] * R[k, c] * R[l, d] * A[a, b, c, d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
]
# The hole turned by β: e^{iβ} ω(e^{−iβ}ζ) has the coefficients mₖ e^{i(k+1)β}.
hms_turned_hole(m, β) = ConformalHole([m[k] * cis((k + 1) * β) for k in eachindex(m)])

# Kirsch: the circular hole of unit radius under a unit tension along e₂, the
# angle θ counted from e₁; the components are those of the polar frame.
function hms_kirsch(r, θ)
    a², a⁴ = 1 / r^2, 1 / r^4
    c, s = cos(2θ), sin(2θ)
    σrr = (1 - a²) / 2 - (1 - 4a² + 3a⁴) * c / 2
    σθθ = (1 + a²) / 2 + (1 + 3a⁴) * c / 2
    σrθ = (1 + 2a² - 3a⁴) * s / 2
    er, eθ = [cos(θ), sin(θ)], [-sin(θ), cos(θ)]
    return σrr * er * er' + σθθ * eθ * eθ' + σrθ * (er * eθ' + eθ * er')
end

# The elliptic hole of map ζ + m/ζ in an isotropic matrix, with the potentials
#   Φ(ζ) = Γζ + a₁/ζ ,  a₁ = −conj(Γ′) − Γm ,
#   Ψ(ζ) = Γ′ζ − Γ/ζ − (Bζ² − a₁)/(ζ(ζ² − m)) ,  B = Γ(1 + |m|²) − conj(m) a₁ ,
# and σ₁₁ + σ₂₂ = 4 Re(Φ′/ω′), σ₂₂ − σ₁₁ + 2iσ₁₂ = 2[z̄ (Φ″ω′ − Φ′ω″)/ω′³ + Ψ′/ω′].
function hms_ellipse_stress(m, (Σ₁₁, Σ₂₂, Σ₁₂), x)
    z = complex(x[1], x[2])
    Γ = (Σ₁₁ + Σ₂₂) / 4
    Γ′ = (Σ₂₂ - Σ₁₁) / 2 + im * Σ₁₂
    a₁ = -conj(Γ′) - Γ * m
    B = Γ * (1 + abs2(m)) - conj(m) * a₁
    # the root of larger modulus of ζ² − zζ + m = 0
    ζ = argmax(abs, ((z + sqrt(z^2 - 4m)) / 2, (z - sqrt(z^2 - 4m)) / 2))
    Φ′, Φ″ = Γ - a₁ / ζ^2, 2a₁ / ζ^3
    ω′, ω″ = 1 - m / ζ^2, 2m / ζ^3
    N, N′ = B * ζ^2 - a₁, 2B * ζ
    D, D′ = ζ^3 - m * ζ, 3ζ^2 - m
    Ψ′ = Γ′ + Γ / ζ^2 - (N′ * D - N * D′) / D^2
    trace = 4 * real(Φ′ / ω′)
    deviator = 2 * (conj(z) * (Φ″ * ω′ - Φ′ * ω″) / ω′^3 + Ψ′ / ω′)
    σ₁₁, σ₂₂, σ₁₂ = (trace - real(deviator)) / 2, (trace + real(deviator)) / 2, imag(deviator) / 2
    return [σ₁₁ σ₁₂; σ₁₂ σ₂₂]
end

# Four stiffnesses of the plane: without symmetry, orthotropic off its axes,
# isotropic, and the isotropic one typed as a general tensor.
hms_general() = from_mandel([30.0 4.0 3.0; 4.0 9.0 -2.0; 3.0 -2.0 7.0])
hms_ortho(c₁₁ = 40.0) = TensOrtho(c₁₁, 6.0, 2.5, 3.0, Basis(0.37))
hms_iso() = iso_stiffness(Val(2), 5.0, 2.0)
hms_copy(C) = Tens(Array(canonical_components(C)))

# The arguments of the two internal evaluations of the anisotropic field, built
# as the functor of `field.jl` builds them.
function hms_pair_arguments(m, C, Σ, x)
    ξ, η = HMS._contour_vectors(complex(m))
    c₁, c₂ = HMS._boundary_constants(Σ, ξ, η)
    s, p = characteristic_pair(C)
    δ = sqrt(s^2 - 4p)
    μ = ((s + δ) / 2, (s - δ) / 2)
    R = map(μₖ -> HMS._exterior_root(x[1] + μₖ * x[2], (conj(ξ) + μₖ * conj(η)) * (ξ + μₖ * η) / 4), μ)
    return ξ, η, c₁, c₂, s, p, μ, R
end
# The stress of the three sums Sₙ = Σₖ μₖⁿ Bₖ Tₖ.
function hms_stress_of_sums(Σ, (S₀, S₁, S₂))
    σ₁₁, σ₂₂, σ₁₂ = Σ[1] + 2real(S₂), Σ[2] + 2real(S₀), Σ[3] - 2real(S₁)
    return [σ₁₁ σ₁₂; σ₁₂ σ₂₂]
end

hms_literature() =
    JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "lekhnitskii1981.json"))
hms_quantity(name) = hms_literature()["quantities"][name]["value"]
# Plane-stress stiffness of the sheet of engineering constants (E₁, E₂, G, ν₁),
# its stiff axis along e₁.
function hms_plywood()
    E₁, E₂, G, ν₁ = hms_quantity("E1"), hms_quantity("E2"), hms_quantity("G"), hms_quantity("nu1")
    return inv(from_mandel([1 / E₁ -ν₁ / E₁ 0; -ν₁ / E₁ 1 / E₂ 0; 0 0 1 / (2G)]))
end

@testset "Stress in the matrix around a hole of the plane" begin
    rng = Xoshiro(20261128)
    Σ = Tens([0.7 0.3; 0.3 -0.4])
    pull = Tens([0.0 0.0; 0.0 1.0])

    @testset "isotropic matrix: the circle" begin
        circle = ConformalHole(())
        # on the axis across the pull
        for r in (1.0, 1.5, 4.0)
            σ = hms_components(matrix_stress(circle, pull, [r, 0.0]))
            @test σ[2] ≈ 1 + 1 / (2r^2) + 3 / (2r^4) rtol = 1.0e-15
            @test σ[1] ≈ 3 / (2r^2) - 3 / (2r^4) atol = 1.0e-15
            @test abs(σ[3]) ≤ 1.0e-15
        end
        # the whole field, at oblique points
        for r in (1.0, 1.5, 4.0), θ in (0.3, 1.1, 2.0, 4.0)
            σ = matrix_stress(circle, pull, r .* [cos(θ), sin(θ)])
            @test canonical_components(σ) ≈ hms_kirsch(r, θ) atol = 1.0e-14
        end
        # a tuple of coordinates, and a collection of points
        @test canonical_components(matrix_stress(circle, pull, (1.5, 0.0))) == canonical_components(matrix_stress(circle, pull, [1.5, 0.0]))
        points = [[1.5, 0.2], [0.0, 2.0], [-3.0, 1.0]]
        @test canonical_components.(matrix_stress(circle, pull, points)) == [canonical_components(matrix_stress(circle, pull, x)) for x in points]
        # an isotropic stiffness of its own type takes the same path
        @test canonical_components(matrix_stress(circle, pull, [1.5, 0.2], hms_iso())) == canonical_components(matrix_stress(circle, pull, [1.5, 0.2]))
    end

    @testset "isotropic matrix: maps of several terms" begin
        holes = (ConformalHole((0.4,)), ConformalHole(1 => 0.1 + 0.15im, 2 => -0.2im), ConformalHole(polygon_map(4, 8)...))
        @test length(holes[3].coefficients) == 31
        for hole in holes
            m = hole.coefficients
            # the contour: the hoop stress, and no traction
            for θ in (0.0, 0.4, 1.3, 2.2, 3.7, 5.1, (2π .* rand(rng, 3))...)
                xc, t, n = hms_contour_point(m, θ)
                S = canonical_components(matrix_stress(hole, Σ, xc))
                @test t' * S * t ≈ contour_stress(hole, Σ, θ) rtol = 1.0e-13 atol = 1.0e-13
                @test norm(S * n) ≤ 1.0e-12
            end
            # far away: the remote stress, approached as 1/r²
            @test norm(canonical_components(matrix_stress(hole, Σ, [300.0, 200.0])) - canonical_components(Σ)) < 1.0e-4
            gap(x) = norm(canonical_components(matrix_stress(hole, Σ, x)) - canonical_components(Σ))
            @test gap([50.0, 10.0]) / gap([100.0, 20.0]) ≈ 4 atol = 0.01
            # equilibrium, and the harmonic trace of an isotropic matrix
            f(x) = hms_components(matrix_stress(hole, Σ, x))
            (e₁, e₂), _ = hms_equilibrium(f, [1.3, 0.9])
            @test abs(e₁) ≤ 1.0e-13 && abs(e₂) ≤ 1.0e-13
            H = ForwardDiff.hessian(x -> tr(canonical_components(matrix_stress(hole, Σ, x))), SVector(1.3, 0.9))
            @test abs(tr(H)) ≤ 1.0e-10
            # linear in the load, the shear entering as the component σ₁₂
            x = [1.3, 0.9]
            unit = map(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) do load
                [HMS._isotropic_field(hole, load)(x[1], x[2])...]
            end
            @test [HMS._isotropic_field(hole, hms_tuple(Σ))(x[1], x[2])...] ≈ 0.7 * unit[1] - 0.4 * unit[2] + 0.3 * unit[3] rtol = 1.0e-13
            @test [HMS._isotropic_field(hole, hms_tuple(Σ))(x[1], x[2])...] ≈ hms_components(matrix_stress(hole, Σ, x)) rtol = 1.0e-15
            # the hole, the load and the point turned together by β
            for β in 2π .* rand(rng, 2)
                R = hms_rotation(β)
                turned = canonical_components(matrix_stress(hms_turned_hole(m, β), Tens(R * canonical_components(Σ) * R'), R * x))
                @test turned ≈ R * canonical_components(matrix_stress(hole, Σ, x)) * R' rtol = 1.0e-12
            end
        end
        # the preimage of a point: ω(ζ) = z, outside the unit circle
        square = holes[3]
        for x in hms_grid(square.coefficients)
            z = complex(x[1], x[2])
            ζ = HMS._exterior_preimage(square.coefficients, z)
            @test hms_map(square.coefficients, ζ) ≈ z rtol = 1.0e-13
            @test abs(ζ) > 1
        end
        # the product of two polynomials: its values are the products of theirs
        a, b = randn(rng, ComplexF64, 4), randn(rng, ComplexF64, 7)
        for t in randn(rng, ComplexF64, 3)
            @test evalpoly(t, HMS._polynomial_product(a, b)) ≈ evalpoly(t, a) * evalpoly(t, b) rtol = 1.0e-13
        end
    end

    @testset "the ellipse against its closed-form potentials" begin
        for m in (0.4, 0.3 * cis(0.9), -0.25), load in (Σ, pull)
            hole = ConformalHole((m,))
            for x in hms_grid([m])
                σ = matrix_stress(hole, load, x)
                @test canonical_components(σ) ≈ hms_ellipse_stress(m, hms_tuple(load), x) rtol = 1.0e-12 atol = 1.0e-12
            end
            # the preimage, against the root of the quadratic
            z = complex(1.6, 0.4)
            @test HMS._exterior_preimage(hole.coefficients, z) ≈ argmax(abs, ((z + sqrt(z^2 - 4m)) / 2, (z - sqrt(z^2 - 4m)) / 2)) rtol = 1.0e-14
        end
        # the contour vectors: x = (Re(ξ σ̄), Re(η σ̄)) is the point ω(σ)
        for m in (complex(0.4), 0.3 * cis(0.9)), θ in (0.0, 1.2, 4.0)
            ξ, η = HMS._contour_vectors(m)
            σ = cis(θ)
            @test [real(ξ * conj(σ)), real(η * conj(σ))] ≈ [real(σ + m / σ), imag(σ + m / σ)] rtol = 1.0e-15
        end
    end

    @testset "anisotropic matrix" begin
        for C in (hms_general(), hms_ortho()), m in (0.4, 0.3 * cis(0.9))
            hole = ConformalHole((m,))
            # the contour: the hoop stress of `contour_stress`, and no traction
            for θ in (0.0, 0.4, 1.3, 2.2, 3.7, 5.1, (2π .* rand(rng, 3))...)
                xc, t, normal = hms_contour_point([m], θ)
                S = canonical_components(matrix_stress(hole, Σ, xc, C))
                @test t' * S * t ≈ contour_stress(hole, Σ, θ, C) rtol = 1.0e-13 atol = 1.0e-13
                @test norm(S * normal) ≤ 1.0e-13
            end
            # equilibrium, at a point and on a grid around the hole, whose
            # points take one evaluation or the other
            f(x) = hms_components(matrix_stress(hole, Σ, x, C))
            (e₁, e₂), _ = hms_equilibrium(f, [1.6, 0.9])
            @test abs(e₁) ≤ 1.0e-13 && abs(e₂) ≤ 1.0e-13
            for x in hms_grid([m])
                (e₁, e₂), scale = hms_equilibrium(f, x)
                @test abs(e₁) ≤ 1.0e-11 * max(1, scale) && abs(e₂) ≤ 1.0e-11 * max(1, scale)
            end
            # far away: the remote stress, approached as 1/r²
            gap(x) = norm(canonical_components(matrix_stress(hole, Σ, x, C)) - canonical_components(Σ))
            @test gap([300.0, 200.0]) < 1.0e-4
            @test gap([50.0, 10.0]) / gap([100.0, 20.0]) ≈ 4 atol = 0.01
        end
        # the matrix, the hole, the load and the point turned together by β
        for C in (hms_general(), hms_ortho()), β in 2π .* rand(rng, 2)
            R = hms_rotation(β)
            m = 0.3 * cis(0.9)
            turned_C = Tens(hms_turn(canonical_components(C), R))
            for x in ([1.6, 0.4], [0.2, 1.9], [-3.0, 2.0])
                turned = canonical_components(matrix_stress(ConformalHole((m * cis(2β),)), Tens(R * canonical_components(Σ) * R'), R * x, turned_C))
                @test turned ≈ R * canonical_components(matrix_stress(ConformalHole((m,)), Σ, x, C)) * R' rtol = 1.0e-12
            end
        end
        # an isotropic stiffness typed as a general tensor: the field of the
        # isotropic matrix; and a stiffness 1e-9 away from it
        Ci = hms_iso()
        near = Tens(Array(canonical_components(Ci)) .+ 1.0e-9 .* Array(canonical_components(hms_general())))
        for m in (0.4, 0.3 * cis(0.9)), x in ([1.6, 0.4], [0.2, 1.9], [-3.0, 2.0])
            hole = ConformalHole((m,))
            isotropic = canonical_components(matrix_stress(hole, Σ, x))
            @test canonical_components(matrix_stress(hole, Σ, x, hms_copy(Ci))) ≈ isotropic rtol = 1.0e-14
            @test canonical_components(matrix_stress(hole, Σ, x, near)) ≈ isotropic rtol = 1.0e-8
        end
        # Lekhnitskii (31.13): the sheet pulled across the major axis, its
        # stiff axis along that axis; at the tip the free surface leaves σ₂₂
        E₁, E₂, G, ν₁ = hms_quantity("E1"), hms_quantity("E2"), hms_quantity("G"), hms_quantity("nu1")
        k = sqrt(E₁ / E₂)
        n = sqrt(2k + E₁ / G - 2ν₁)
        c = hms_quantity("axis_ratio")
        m = (c - 1) / (c + 1)                                   # a/b = c, a along e₁
        σ = hms_components(matrix_stress(ConformalHole((m,)), pull, [1 + m, 0.0], hms_plywood()))
        @test σ[2] ≈ 1 + c * n / k rtol = 1.0e-12
        @test abs(σ[1]) ≤ 1.0e-12 && abs(σ[3]) ≤ 1.0e-12
    end

    @testset "the two evaluations" begin
        load = hms_tuple(Σ)
        points = ([1.6, 0.4], [0.2, 1.9], [-3.0, 2.0])
        relative(a, b) = norm(collect(a) .- collect(b)) / norm(collect(b))
        for C in (hms_general(), hms_ortho()), m in (0.4, 0.3 * cis(0.9)), x in points
            ξ, η, c₁, c₂, s, p, μ, R = hms_pair_arguments(m, C, load, x)
            roots = HMS._field_by_roots(x[1], x[2], ξ, η, c₁, c₂, μ, R)
            pairs = HMS._field_by_pairs(x[1], x[2], ξ, η, c₁, c₂, s, p, R...)
            @test relative(roots, pairs) ≤ 1.0e-14
            @test hms_stress_of_sums(load, pairs) ≈ canonical_components(matrix_stress(ConformalHole((m,)), Σ, x, C)) rtol = 1.0e-13
        end
        # at isotropy, typed as a general tensor, the roots nearly coincide:
        # the pairs keep the isotropic field, the roots lose digits
        for m in (0.4, 0.3 * cis(0.9)), x in points
            ξ, η, c₁, c₂, s, p, μ, R = hms_pair_arguments(m, hms_copy(hms_iso()), load, x)
            roots = HMS._field_by_roots(x[1], x[2], ξ, η, c₁, c₂, μ, R)
            pairs = HMS._field_by_pairs(x[1], x[2], ξ, η, c₁, c₂, s, p, R...)
            @test hms_stress_of_sums(load, pairs) ≈ canonical_components(matrix_stress(ConformalHole((m,)), Σ, x)) rtol = 1.0e-14
            @test relative(roots, pairs) > 1.0e-12
        end
        # the root R of R² = z² − 4αβ on the side of the outside: ζ = (z + R)/(2α)
        # is the root of larger modulus of αζ² − zζ + β
        for z in randn(rng, ComplexF64, 4), αβ in randn(rng, ComplexF64, 2)
            R = HMS._exterior_root(z, αβ)
            @test R^2 ≈ z^2 - 4αβ rtol = 1.0e-14
            @test abs(z + R) ≥ abs(z - R)
        end
    end

    @testset "derivatives" begin
        # Kirsch, across the pull: dσ₂₂/dr = −1/r³ − 6/r⁵
        circle = ConformalHole(())
        @test ForwardDiff.derivative(r -> hms_components(matrix_stress(circle, pull, [r, zero(r)]))[2], 2.0) ≈ -1 / 8 - 6 / 32 rtol = 1.0e-13
        # with respect to a modulus of the matrix: its orthotropic class and the
        # general tensor of the same components, which find (s, p) differently
        hole = ConformalHole((0.3 * cis(0.9),))
        x = [1.6, 0.4]
        ortho(c₁₁) = hms_components(matrix_stress(hole, Σ, x, hms_ortho(c₁₁)))[2]
        general(c₁₁) = hms_components(matrix_stress(hole, Σ, x, hms_copy(hms_ortho(c₁₁))))[2]
        @test ForwardDiff.derivative(ortho, 40.0) ≈ ForwardDiff.derivative(general, 40.0) rtol = 1.0e-8
        # on the contour, against the derivative of `contour_stress`
        θ = 0.7
        xc, t, _ = hms_contour_point(hole.coefficients, θ)
        hoop(C) = (S = canonical_components(matrix_stress(hole, Σ, xc, C)); t' * S * t)
        @test ForwardDiff.derivative(c₁₁ -> hoop(hms_ortho(c₁₁)), 40.0) ≈
            ForwardDiff.derivative(c₁₁ -> contour_stress(hole, Σ, θ, hms_ortho(c₁₁)), 40.0) rtol = 1.0e-10
        # through isotropy, typed as a general tensor and perturbed on C₁₁₁₁:
        # on the contour against `contour_stress`, which has no division by
        # μ₁ − μ₂; inside the matrix, a derivative that is finite and continuous
        Cᵢ = Array(canonical_components(hms_iso()))
        E = [(i == j == k == l == 1) * 1.0 for i in 1:2, j in 1:2, k in 1:2, l in 1:2]
        perturbed(ε) = Tens(Cᵢ .+ ε .* E)
        @test ForwardDiff.derivative(ε -> hoop(perturbed(ε)), 0.0) ≈
            ForwardDiff.derivative(ε -> contour_stress(hole, Σ, θ, perturbed(ε)), 0.0) rtol = 1.0e-8
        inside(ε) = hms_components(matrix_stress(hole, Σ, x, perturbed(ε)))[2]
        d₀ = ForwardDiff.derivative(inside, 0.0)
        @test isfinite(d₀)
        @test d₀ ≈ ForwardDiff.derivative(inside, 1.0e-6) rtol = 1.0e-4
    end

    @testset "inference and refusals" begin
        x = [1.6, 0.4]
        for hole in (ConformalHole((0.3 * cis(0.9),)), ConformalHole(1 => 0.1 + 0.15im, 2 => -0.2im))
            @inferred matrix_stress(hole, Σ, x)
        end
        ellipse = ConformalHole((0.3 * cis(0.9),))
        for C in (hms_general(), hms_ortho())
            @inferred matrix_stress(ellipse, Σ, x, C)
        end
        # a point inside the hole
        @test_throws DomainError matrix_stress(ConformalHole((0.4,)), Σ, [0.3, 0.1])
        @test_throws DomainError matrix_stress(ConformalHole(polygon_map(4, 8)...), Σ, [0.0, 0.0])
        @test_throws DomainError matrix_stress(ConformalHole((0.4,)), Σ, [0.3, 0.1], hms_general())
        # a map with two terms in an anisotropic matrix, and a stiffness without
        # the major symmetry
        @test_throws ArgumentError matrix_stress(ConformalHole((0.2, 0.1)), Σ, x, hms_ortho())
        A = Array(canonical_components(hms_general()))
        A[1, 1, 2, 2] += 1.5
        @test_throws ArgumentError matrix_stress(ellipse, Σ, x, Tens(A))
    end
end
