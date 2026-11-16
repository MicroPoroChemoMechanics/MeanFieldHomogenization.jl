using Test
using MeanFieldHomogenization
using ForwardDiff
using LinearAlgebra

# =============================================================================
#  Solid harmonics in the normalization of kushch2013, Eq. (2.9).
# =============================================================================

@testset "solid harmonics — closed forms of the low degrees" begin
    x = [0.3, -0.7, 0.45]
    ζ, z, r = complex(x[1], x[2]), x[3], norm(x)
    y = regular_harmonics(x, 3)
    Y = irregular_harmonics(x, 3)
    @test y[0, 0] ≈ 1
    @test y[1, 0] ≈ z
    @test y[1, 1] ≈ ζ / 2
    @test y[1, -1] ≈ -conj(ζ) / 2
    @test y[2, 0] ≈ (3z^2 - r^2) / 4
    @test y[2, 1] ≈ z * ζ / 2
    @test y[2, 2] ≈ ζ^2 / 8
    @test y[2, -2] ≈ conj(ζ)^2 / 8
    @test Y[0, 0] ≈ 1 / r
    @test Y[1, 0] ≈ z / r^3
    @test Y[1, 1] ≈ ζ / r^3
    @test Y[1, -1] ≈ -conj(ζ) / r^3
    @test Y[2, 0] ≈ (3z^2 - r^2) / r^5
    @test Y[2, 1] ≈ 3z * ζ / r^5
    # an order beyond the degree reads as zero, a degree beyond the table is refused
    @test y[2, 3] == 0 && Y[1, -2] == 0
    @test_throws BoundsError y[4, 0]
    @test_throws BoundsError y[-1, 0]
    @test regular_harmonics(zeros(3), 2)[0, 0] == 1
end

@testset "solid harmonics — the two families at every degree" begin
    # Y_t^s = (t−s)! (t+s)! y_t^s / r^(2t+1): the two recurrences are independent
    x = [-0.4, 0.25, 0.8]
    r = norm(x)
    nmax = 12
    y = regular_harmonics(x, nmax)
    Y = irregular_harmonics(x, nmax)
    for t in 0:nmax, s in (-t):t
        @test Y[t, s] ≈ harmonic_norm(t, s)^2 * y[t, s] / r^(2t + 1) rtol = 1.0e-12
    end
    @test harmonic_norm(2, 1) ≈ sqrt(6)
    @test harmonic_norm(3, -3) ≈ sqrt(720)
end

@testset "solid harmonics — derivatives (Hobson) and harmonicity" begin
    x = [0.3, -0.7, 0.45]
    nmax = 5
    y = regular_harmonics(x, nmax + 1)
    Y = irregular_harmonics(x, nmax + 1)
    dy = ForwardDiff.derivative(h -> real.(regular_harmonics(x .+ h .* [0, 0, 1], nmax).values), 0.0)
    dY = ForwardDiff.derivative(h -> imag.(irregular_harmonics(x .+ h .* [0, 0, 1], nmax).values), 0.0)
    k = 0
    for t in 0:nmax, s in 0:t
        k += 1
        # ∂₃ y_t^s = y_{t−1}^s and ∂₃ Y_t^s = −Y_{t+1}^s
        @test dy[k] ≈ (t > 0 ? real(y[t - 1, s]) : 0.0) atol = 1.0e-13
        @test dY[k] ≈ -imag(Y[t + 1, s]) rtol = 1.0e-12 atol = 1.0e-13
    end
    lap(f) = tr(ForwardDiff.hessian(f, x))
    @test abs(lap(v -> real(regular_harmonics(v, 4)[4, 2]))) < 1.0e-12
    @test abs(lap(v -> imag(irregular_harmonics(v, 4)[4, 3]))) < 1.0e-9
end

@testset "solid harmonics — translation of a decaying harmonic" begin
    # Y_k^l(r + R) = Σ_{t,s} (−1)^{t+s} Y_{k+t}^{l−s}(R) y_t^s(r), ‖r‖ < ‖R‖
    R = [1.1, -0.6, 0.9]
    r = [0.15, 0.2, -0.1]
    nmax = 30
    YR = irregular_harmonics(R, nmax + 3)
    yr = regular_harmonics(r, nmax)
    direct = irregular_harmonics(R .+ r, 3)
    for (k, l) in ((1, 0), (2, -1), (3, 3))
        series = sum((-1)^(t + s) * YR[k + t, l - s] * yr[t, s] for t in 0:nmax for s in (-t):t)
        @test series ≈ direct[k, l] rtol = 1.0e-12
    end
end

@testset "solid harmonics — types and refusals" begin
    @test @inferred(regular_harmonics([1.0, 2.0, 3.0], 3)) isa HarmonicTable{ComplexF64}
    @test @inferred(irregular_harmonics([1, 2, 3], 3)) isa HarmonicTable{ComplexF64}
    @test regular_harmonics(Float32[1, 2, 3], 2) isa HarmonicTable{ComplexF32}
    @test_throws ArgumentError regular_harmonics([1.0, 2.0], 2)
    @test_throws ArgumentError regular_harmonics([1.0, 2.0, 3.0], -1)
    @test_throws DomainError irregular_harmonics(zeros(3), 2)
end
