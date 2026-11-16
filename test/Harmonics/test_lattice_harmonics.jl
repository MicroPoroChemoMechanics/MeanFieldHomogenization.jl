using Test
using MeanFieldHomogenization
using ForwardDiff

# =============================================================================
#  Harmonics of a periodic array of sources (Ewald sums).
#
#  References: the constant of the periodic potential of a simple cubic
#  lattice, −2.837297479480620/L; the lattice sums tabulated by Kushch (2013),
#  Table A.1, for the unit cube; direct sums where they converge absolutely.
# =============================================================================

@testset "lattice harmonics — simple cubic values" begin
    h = lattice_harmonics(zeros(3), (1.0, 1.0, 1.0), 8)
    @test real(h[0, 0]) ≈ -2.83729747948062 rtol = 1.0e-13
    # cubic symmetry leaves no harmonic of degree 2, and none of odd degree
    @test abs(h[2, 0]) < 1.0e-13 && abs(h[2, 2]) < 1.0e-13 && abs(h[3, 2]) < 1.0e-12
    for (t, s, ref) in (
            (4, 0, 3.10822668269944), (4, 4, 1.85752072772946),
            (6, 0, 0.573329289434499), (6, 4, -1.07260088543324), (8, 0, 3.25929309334983),
        )
        @test real(h[t, s]) / harmonic_norm(t, s) ≈ ref rtol = 1.0e-12
        @test abs(imag(h[t, s])) < 1.0e-9
    end
    # the sums scale as L^-(t+1)
    h2 = lattice_harmonics(zeros(3), (2.0, 2.0, 2.0), 4)
    @test h2[4, 0] ≈ h[4, 0] / 2^5 rtol = 1.0e-12
    @test h2[0, 0] ≈ h[0, 0] / 2 rtol = 1.0e-12
end

@testset "lattice harmonics — direct sum on an orthogonal lattice" begin
    x = [0.2, -0.1, 0.35]
    L = (1.0, 1.3, 0.8)
    h = lattice_harmonics(x, L, 8)
    direct = sum(
        irregular_harmonics(x .+ [i * L[1], j * L[2], k * L[3]], 8)[8, 3]
            for i in -9:9, j in -9:9, k in -9:9
    )
    @test h[8, 3] ≈ direct rtol = 1.0e-6
    # periodicity
    hp = lattice_harmonics(x .+ [L[1], -2 * L[2], L[3]], L, 4)
    for t in 0:4, s in 0:t
        @test hp[t, s] ≈ h[t, s] rtol = 1.0e-10 atol = 1.0e-11
    end
end

@testset "lattice harmonics — a lattice point and its neighborhood" begin
    # Near a lattice point the periodic harmonic is the harmonic of that point
    # plus the regular field of the others, given by the on-site sums and the
    # translation formula. On the degree-1 harmonics the uniform background
    # adds the field −(4π/3V) x of `lattice_sums.jl`.
    L = (1.0, 1.2, 0.9)
    V = prod(L)
    x = [0.04, -0.03, 0.05]
    kmax = 10
    on = lattice_harmonics(zeros(3), L, kmax + 4)
    off = lattice_harmonics(x, L, 4)
    Y = irregular_harmonics(x, 4)
    y = regular_harmonics(x, kmax)
    regular(t, s) = sum((-1)^(k + l) * on[t + k, s - l] * y[k, l] for k in 0:kmax for l in (-k):k)
    for (t, s) in ((2, 0), (2, 1), (3, 2), (4, 0))
        @test off[t, s] ≈ Y[t, s] + regular(t, s) rtol = 1.0e-9
    end
    background = (-4π / (3V)) .* (y[1, 0], 2 * y[1, 1])
    @test off[1, 0] ≈ Y[1, 0] + regular(1, 0) + background[1] rtol = 1.0e-9
    @test off[1, 1] ≈ Y[1, 1] + regular(1, 1) + background[2] rtol = 1.0e-9
end

@testset "lattice harmonics — derivatives and refusals" begin
    x = [0.2, -0.1, 0.35]
    L = (1.0, 1.3, 0.8)
    h = lattice_harmonics(x, L, 4)
    d = ForwardDiff.derivative(u -> real.(lattice_harmonics(x .+ u .* [0, 0, 1], L, 3).values), 0.0)
    k = 0
    for t in 0:3, s in 0:t
        k += 1
        # ∂₃ Y*_t^s = −Y*_{t+1}^s, but for the dipole along e₃: its derivative
        # is a second derivative of S, whose trace 4π/V the harmonics of
        # degree 2 do not hold
        trace = (t, s) == (1, 0) ? 4π / (3 * prod(L)) : 0.0
        @test d[k] ≈ -real(h[t + 1, s]) - trace rtol = 1.0e-9 atol = 1.0e-10
    end
    # with respect to a period, against the scaling of the cubic sums
    c = real(lattice_harmonics(zeros(3), (1.0, 1.0, 1.0), 4)[4, 0])
    @test ForwardDiff.derivative(a -> real(lattice_harmonics(zeros(3), (a, a, a), 4)[4, 0]), 1.5) ≈
        -5 * c / 1.5^6 rtol = 1.0e-10
    @test @inferred(lattice_harmonics(x, L, 2)) isa HarmonicTable{ComplexF64}
    @test_throws ArgumentError lattice_harmonics([0.1, 0.2], L, 2)
    @test_throws ArgumentError lattice_harmonics(x, (1.0, 1.0), 2)
end
