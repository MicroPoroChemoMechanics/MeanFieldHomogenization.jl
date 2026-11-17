using Test
using MeanFieldHomogenization
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Lamé functions and ellipsoidal harmonics.
#
#  References: the closed forms of the low degrees (Dassios 2012, Eqs. (3.37),
#  (3.38), (3.95), (3.96)), harmonicity checked by automatic differentiation,
#  the exact normalization constants of degree 1 and those of Kushch (2020),
#  Table 1.
# =============================================================================

const _ELH = MeanFieldHomogenization.Harmonics

@testset "Lamé functions — classes and closed forms" begin
    g = _ELH.EllipsoidalGeometry(1.3, 0.9, 0.5)
    for n in 0:6
        E = _ELH.lame_functions(g, n)
        @test length(E) == 2n + 1
        @test all(e -> e.coefficients[1] == 1, E)
    end
    # degree 2, class K: x² + Λ − a₁², Λ the roots of Σ 1/(Λ − aᵢ²) = 0
    a² = g.a .^ 2
    s₁, s₂ = sum(a²), a²[1] * a²[2] + a²[2] * a²[3] + a²[3] * a²[1]
    Λ = sort([(s₁ + sqrt(s₁^2 - 3s₂)) / 3, (s₁ - sqrt(s₁^2 - 3s₂)) / 3])
    @test sort([E.coefficients[2] for E in _ELH.lame_functions(g, 2)[1:2]]) ≈ Λ .- a²[1] rtol = 1.0e-13
    # class N: p = 1 at degree 2 and 4 at degree 3 (Dassios, § 3.4)
    @test only(E.p for E in _ELH.lame_functions(g, 2) if E.a == E.b == 1) ≈ 1 rtol = 1.0e-13
    @test only(E.p for E in _ELH.lame_functions(g, 3) if E.a == E.b == 1) ≈ 4 rtol = 1.0e-13
    @test_throws ArgumentError _ELH.lame_functions(g, -1)
    @test_throws ArgumentError _ELH.EllipsoidalGeometry(1.0, 1.0, 0.5)
end

@testset "ellipsoidal harmonics — harmonic inside and outside" begin
    g = _ELH.EllipsoidalGeometry(1.3, 0.9, 0.5)
    nodes = MeanFieldHomogenization.LayeredSpheroids.QuadGK.gauss(30, 0.0, 1.0)
    interior(E, x) = (c = _ELH.ellipsoidal_coordinates(g, x); _ELH._lame_at_ρ(E, c)[1] * _ELH._lame_at_μ(E, c)[1] * _ELH._lame_at_ν(E, c)[1])
    exterior(E, x) = (c = _ELH.ellipsoidal_coordinates(g, x); _ELH.lame_second_kind(g, E, c.ρ, nodes)[1] * _ELH._lame_at_μ(E, c)[1] * _ELH._lame_at_ν(E, c)[1])
    x, y = [0.7, -0.4, 0.3], [1.9, 0.8, -1.2]
    for n in 0:3, E in _ELH.lame_functions(g, n)
        @test abs(tr(ForwardDiff.hessian(z -> interior(E, z), x))) ≤ 1.0e-11
        @test abs(tr(ForwardDiff.hessian(z -> exterior(E, z), y))) ≤ 1.0e-12
    end
    # degree 1: the three coordinates, x₁ = ρμν/(h₂h₃), and their decaying counterparts
    K1, L1, M1 = _ELH.lame_functions(g, 1)
    @test interior(K1, x) ≈ g.h2 * g.h3 * x[1] rtol = 1.0e-12
    @test interior(L1, x) ≈ g.h1 * g.h3 * x[2] rtol = 1.0e-12
    @test interior(M1, x) ≈ g.h1 * g.h2 * x[3] rtol = 1.0e-12
    far = 300 .* [0.3, 0.5, -0.8]
    @test exterior(K1, far) ≈ g.h2 * g.h3 * far[1] / norm(far)^3 rtol = 1.0e-4
    # the derivative of the second kind against the integral that defines it
    E = _ELH.lame_functions(g, 2)[1]
    F(ρ) = _ELH.lame_second_kind(g, E, ρ, nodes)[1]
    @test ForwardDiff.derivative(F, 1.7) ≈ _ELH.lame_second_kind(g, E, 1.7, nodes)[2] rtol = 1.0e-10
end

@testset "ellipsoidal harmonics — normalization, Kushch (2020), Table 1" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2020.json"))
    a = Tuple(lit["quantities"]["semi_axes"]["value"])
    g = _ELH.EllipsoidalGeometry(a...)
    A = MeanFieldHomogenization.Assemblies
    lame = vcat([_ELH.lame_functions(g, n) for n in 1:10]...)
    _, _, S, w, _ = A._mem_ellipsoid_projection(g, lame, ones(length(lame)), 10)
    γ = S .^ 2 * w
    # degree 1 in closed form: (4π/3) H_m², H_m = h₁h₂h₃/h_m
    @test γ[1:3] ≈ (4π / 3) .* [(g.h2 * g.h3)^2, (g.h1 * g.h3)^2, (g.h1 * g.h2)^2] rtol = 1.0e-12
    for row in lit["tables"]["normalization"]
        n = row["n"]
        candidates = γ[[i for (i, E) in enumerate(lame) if E.n == n]]
        @test minimum(abs.(candidates .- row["gamma"])) ≤ 1.0e-6 * row["gamma"]
    end
end
