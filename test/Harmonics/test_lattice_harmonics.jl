using Test
using MeanFieldHomogenization
using ForwardDiff
using JSON

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
    # Kushch (2013), Table A.1; the degree 2, conditionally convergent, is a
    # matter of convention and is treated apart
    file = joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json")
    for row in JSON.parsefile(file)["tables"]["lattice_sums_sc"]
        t, s, ref = row["t"], row["s"], row["Y_over_N"]
        t > 2 || continue
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

@testset "lattice biharmonics — Ewald parameter, direct sums, derivatives" begin
    x = [0.2, -0.1, 0.35]
    L = (1.0, 1.3, 0.8)
    # the sums do not depend on the splitting parameter, degree 0 included
    for p in (x, zeros(3))
        S₁, B₁ = MeanFieldHomogenization.Harmonics._lattice_sums(p, L, 6, Val(true), 2.0)
        S₂, B₂ = MeanFieldHomogenization.Harmonics._lattice_sums(p, L, 6, Val(true), 3.5)
        @test maximum(abs, B₁.values .- B₂.values) < 1.0e-10 * maximum(abs, B₁.values)
        @test maximum(abs, S₁.values .- S₂.values) < 1.0e-10 * maximum(abs, S₁.values)
    end
    # B*_n^m = Σ −r² Y_n^m/(2n−1) over the images, absolutely convergent at n = 7
    S, B = lattice_biharmonics(x, L, 8)
    @test S.values == lattice_harmonics(x, L, 8).values
    direct = zero(ComplexF64)
    K = 12
    for a in (-K):K, b in (-K):K, c in (-K):K
        xn = x .+ (a * L[1], b * L[2], c * L[3])
        direct -= sum(abs2, xn) * irregular_harmonics(xn, 7)[7, 3] / 13
    end
    @test B[7, 3] ≈ direct rtol = 1.0e-4
    # ∂₃ 𝒟_n^m = −𝒟_{n+1}^m + β Δ 𝒟_{n−1}^m with Δρ = 2S:
    # ∂₃ B*_n^m = −B*_{n+1}^m − 2(n² − m²) Y*_{n−1}^m / [(2n+1)(2n−1)]
    d = ForwardDiff.derivative(u -> lattice_biharmonics(x .+ u .* [0, 0, 1], L, 3)[2].values, 0.0)
    S, B = lattice_biharmonics(x, L, 4)
    k = 0
    for n in 0:3, m in 0:n
        k += 1
        n == 0 && continue
        @test d[k] ≈ -B[n + 1, m] - 2 * (n^2 - m^2) * S[n - 1, m] / ((2n + 1) * (2n - 1)) rtol = 1.0e-9
    end
    @test @inferred(lattice_biharmonics(x, L, 2)) isa Tuple{HarmonicTable{ComplexF64}, HarmonicTable{ComplexF64}}
    @test_throws ArgumentError lattice_biharmonics(x, L, -1)
end

@testset "lattice sums with excluded images" begin
    x = [0.2, -0.1, 0.35]
    L = (1.0, 1.3, 0.8)
    out = [(0, 0, 1), (-1, 0, 0), (1, 1, -1)]
    S, B = MeanFieldHomogenization.Harmonics._lattice_sums(x, L, 6, Val(true), 1.2; exclude = out)
    Sf, Bf = lattice_biharmonics(x, L, 6)
    Sd = zero(S.values)
    Bd = zero(B.values)
    for n in out
        xn = x .+ n .* L
        Y = irregular_harmonics(xn, 6)
        r² = sum(abs2, xn)
        k = 0
        for N in 0:6, M in 0:N
            k += 1
            Sd[k] += Y[N, M]
            Bd[k] += N == 0 ? sqrt(r²) : -r² * Y[N, M] / (2N - 1)
        end
    end
    @test S.values ≈ Sf.values .- Sd rtol = 1.0e-10
    @test B.values ≈ Bf.values .- Bd rtol = 1.0e-10
end
