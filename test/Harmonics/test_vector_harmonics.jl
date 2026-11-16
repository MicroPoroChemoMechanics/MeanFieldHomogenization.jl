using Test
using MeanFieldHomogenization
using ForwardDiff
using LinearAlgebra

# =============================================================================
#  Vector solid harmonics, solutions of the Lamé equation and their translation.
#
#  References independent of the Cartesian formulas under test: the differential
#  relations that define the families (gradient, divergence, curl), the Lamé
#  equation itself, both through ForwardDiff, and the direct evaluation of a
#  translated solution.
# =============================================================================

# Real and imaginary parts differentiated apart: the fields are complex.
_vh_jac(f, x) = ForwardDiff.jacobian(v -> real.(f(v)), x) + im * ForwardDiff.jacobian(v -> imag.(f(v)), x)
_vh_div(f, x) = tr(_vh_jac(f, x))
function _vh_curl(f, x)
    J = _vh_jac(f, x)
    return [J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2]]
end
# Lamé operator 2(1−ν)/(1−2ν) ∇(∇·u) − ∇×∇×u
function _vh_lame(f, x, ν)
    graddiv = ForwardDiff.gradient(v -> real(_vh_div(f, v)), x) + im * ForwardDiff.gradient(v -> imag(_vh_div(f, v)), x)
    return 2 * (1 - ν) / (1 - 2ν) * graddiv - _vh_curl(v -> _vh_curl(f, v), x)
end

const _VH_X = [0.3, -0.7, 0.45]

@testset "vector harmonics — the relations that define the families" begin
    x = _VH_X
    Yt(v, n) = irregular_harmonics(v, n)
    yt(v, n) = regular_harmonics(v, n)
    grad(f) = ForwardDiff.gradient(v -> real(f(v)), x) + im * ForwardDiff.gradient(v -> imag(f(v)), x)
    for (t, s) in ((1, 0), (2, 1), (3, -2), (4, 4))
        Y1(v) = vector_irregular_harmonic(1, t, s, Yt(v, t + 1))
        Y2(v) = vector_irregular_harmonic(2, t, s, Yt(v, t + 1))
        Y3(v) = vector_irregular_harmonic(3, t, s, Yt(v, t + 1))
        y1(v) = vector_regular_harmonic(1, t, s, yt(v, t + 1))
        y2(v) = vector_regular_harmonic(2, t, s, yt(v, t + 1))
        y3(v) = vector_regular_harmonic(3, t, s, yt(v, t + 1))
        @test Y1(x) ≈ grad(v -> Yt(v, t)[t, s]) rtol = 1.0e-11
        @test y1(x) ≈ grad(v -> yt(v, t)[t, s]) rtol = 1.0e-11
        @test Y2(x) ≈ cross(Y1(x), x) / t rtol = 1.0e-12          # a rotation field
        @test abs(_vh_div(Y2, x)) < 1.0e-9 && abs(_vh_div(y2, x)) < 1.0e-11
        @test _vh_div(Y3, x) ≈ -Yt(x, t)[t, s] rtol = 1.0e-10
        @test _vh_div(y3, x) ≈ -yt(x, t)[t, s] rtol = 1.0e-10
        @test _vh_curl(Y2, x) ≈ -Y1(x) rtol = 1.0e-10
        @test _vh_curl(Y3, x) ≈ Y2(x) rtol = 1.0e-10
        @test _vh_curl(y3, x) ≈ y2(x) rtol = 1.0e-10
        # x·Y⁽³⁾ = r² Y_t^s/(2t−1) and x·y⁽³⁾ = −r² y_t^s/(2t+3)
        @test sum(x .* Y3(x)) ≈ sum(abs2, x) * Yt(x, t)[t, s] / (2t - 1) rtol = 1.0e-12
        @test sum(x .* y3(x)) ≈ -sum(abs2, x) * yt(x, t)[t, s] / (2t + 3) rtol = 1.0e-12
    end
    @test_throws ArgumentError vector_regular_harmonic(4, 1, 0, yt(x, 2))
    @test_throws ArgumentError vector_irregular_harmonic(0, 1, 0, Yt(x, 2))
end

@testset "Lamé solutions — the equation and the low degrees" begin
    x, ν = _VH_X, 0.3
    for (t, s) in ((1, 1), (2, 0), (3, -1)), i in 1:3
        @test norm(_vh_lame(v -> lame_irregular(i, t, s, v, ν), x, ν)) < 1.0e-7 * norm(lame_irregular(i, t, s, x, ν)) / norm(x)^2
        @test norm(_vh_lame(v -> lame_regular(i, t, s, v, ν), x, ν)) < 1.0e-9
    end
    # the dilatation, the translations, a rotation and a uniform deviatoric strain
    @test lame_regular(3, 0, 0, x, ν) ≈ (4ν - 2) / 3 * x
    @test lame_regular(1, 1, 0, x, ν) ≈ [0, 0, 1]
    @test lame_regular(1, 1, 1, x, ν) ≈ [1, im, 0] / 2
    @test lame_regular(2, 1, 0, x, ν) ≈ [x[2], -x[1], 0] / 2
    @test lame_regular(1, 2, 0, x, ν) ≈ [-x[1] / 2, -x[2] / 2, x[3]]
    # a center of dilatation, and the point force (the only solutions with a net force)
    @test lame_irregular(1, 0, 0, x, ν) ≈ -x / norm(x)^3
    # divergence and curl of the third family (kushch2013, Eq. (4.35))
    for (t, s) in ((2, 1), (3, 0))
        @test _vh_div(v -> lame_irregular(3, t, s, v, ν), x) ≈ 2 * (2ν - 1) * irregular_harmonics(x, t)[t, s] rtol = 1.0e-9
        @test _vh_curl(v -> lame_regular(3, t, s, v, ν), x) ≈ 4 * (1 - ν) * lame_regular(2, t, s, x, ν) rtol = 1.0e-9
    end
    @test @inferred(lame_regular(3, 2, 1, x, ν)) isa AbstractVector{ComplexF64}
    @test @inferred(lame_irregular(2, 2, 1, x, ν)) isa AbstractVector{ComplexF64}
end

@testset "Lamé solutions — translation to another center" begin
    R, r, ν = [1.1, -0.6, 0.9], [0.12, 0.2, -0.1], 0.3
    kmax = 26
    Y = irregular_harmonics(R, kmax + 4)
    for (i, t, s) in ((1, 0, 0), (1, 2, 1), (2, 1, 0), (2, 3, 2), (3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 2), (3, 3, -1))
        series = zeros(ComplexF64, 3)
        for k in 0:kmax, l in (-k):k, j in 1:3
            a, b = lame_translation(i, j, t, k, s, l, Y, R)
            series += (-1)^(k + l) * (a + 4 * (1 - ν) * b) * lame_regular(j, k, l, r, ν)
        end
        direct = lame_irregular(i, t, s, R + r, ν)
        @test series ≈ direct rtol = 1.0e-12
    end
    # the families above the one translated do not appear
    @test lame_translation(1, 2, 2, 3, 0, 1, Y, R) == (0, 0)
    @test lame_translation(2, 3, 2, 3, 0, 1, Y, R) == (0, 0)
    # derivative with respect to the position of the second center
    f(h) = real(sum(lame_translation(3, 1, 2, 2, 1, 0, irregular_harmonics(R + h * [0, 0, 1], 6), R + h * [0, 0, 1])))
    g(h) = real(sum(lame_translation(3, 1, 2, 2, 1, 0, irregular_harmonics(R + h * [0, 0, 1], 6), R + h * [0, 0, 1])))
    @test ForwardDiff.derivative(f, 0.0) ≈ (g(1.0e-5) - g(-1.0e-5)) / 2.0e-5 rtol = 1.0e-7
end
