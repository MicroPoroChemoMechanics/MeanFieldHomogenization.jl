using Test
using MeanFieldHomogenization
using LinearAlgebra
using TensND

# =============================================================================
#  test_volterra_inverse.jl — block-triangular forward-substitution.
# =============================================================================

@testset "volterra_inverse — scalar (block_size = 1)" begin
    # Random lower-triangular matrix.
    n = 6
    A = LowerTriangular(rand(n, n)) + 0.5 * I
    A_full = Matrix(A)
    A_inv = volterra_inverse(A_full; block_size = 1)
    @test size(A_inv) == (n, n)
    # Lower-triangular structure is preserved.
    for i in 1:n, j in 1:n
        if j > i
            @test A_inv[i, j] == 0.0
        end
    end
    @test norm(A_full * A_inv - I) ≤ 1.0e-12
    @test norm(A_inv * A_full - I) ≤ 1.0e-12
end

@testset "volterra_inverse — block (block_size = 6)" begin
    # Maxwell iso law produces a 6n×6n lower-block-triangular matrix.
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    times = collect(0.0:0.2:2.0)
    M = trapezoidal_matrix(law, times)
    n = length(times)
    @test size(M) == (6n, 6n)
    M_inv = volterra_inverse(M; block_size = 6)
    @test size(M_inv) == (6n, 6n)
    @test norm(M * M_inv - I) ≤ 1.0e-12
    @test norm(M_inv * M - I) ≤ 1.0e-12
end

@testset "volterra_inverse — round-trip relaxation ↔ creep" begin
    law = maxwell_iso(20.0, 10.0, 0.5, 0.3)
    times = collect(0.0:0.1:1.0)
    R = trapezoidal_matrix(law, times)
    J = volterra_inverse(R; block_size = 6)
    R_back = volterra_inverse(J; block_size = 6)
    @test norm(R - R_back) ≤ 1.0e-10
end

@testset "volterra_inverse — elastic limit (Heaviside)" begin
    # For a Heaviside law with stiffness C, the inverse is the compliance.
    C = TensISO{3}(30.0, 8.0)   # k = 10, μ = 4
    law = heaviside_law(C)
    times = collect(0.0:0.5:2.0)
    R = trapezoidal_matrix(law, times)
    J = volterra_inverse(R; block_size = 6)

    # Each diagonal block of J must equal inv(M_block) where M_block is the
    # Mandel form of C.  Off-diagonal blocks must be zero (no history coupling).
    inv_block = inv(R[1:6, 1:6])
    for i in eachindex(times), j in eachindex(times)
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        if i == j
            @test J[rows, cols] ≈ inv_block atol = 1.0e-12
        else
            @test all(iszero, J[rows, cols])
        end
    end
end

@testset "volterra_inverse — singular detection" begin
    # A matrix with a zero diagonal block raises SingularException.
    M = zeros(2, 2)
    M[2, 1] = 1.0
    @test_throws SingularException volterra_inverse(M; block_size = 1)
end

@testset "volterra_inverse — argument validation" begin
    @test_throws ArgumentError volterra_inverse(zeros(5, 6); block_size = 1)
    @test_throws ArgumentError volterra_inverse(zeros(7, 7); block_size = 6)   # 7 not divisible by 6
    @test_throws ArgumentError volterra_inverse(zeros(2, 2); block_size = 0)
end

@testset "volterra_product — basic check" begin
    A = LowerTriangular(rand(4, 4)) + 0.1 * I
    B = LowerTriangular(rand(4, 4)) + 0.1 * I
    @test volterra_product(Matrix(A), Matrix(B)) == Matrix(A) * Matrix(B)
end

@testset "volterra_left_divide — the BLAS path of a long scalar history" begin
    # From 64 time steps a scalar Volterra division goes to a triangular solve,
    # converting a mixed pair of element types first; it must solve S X = M.
    n = 80
    S = LowerTriangular(rand(n, n) + n * I)
    M = Float32.(tril(rand(n, n)))
    X = volterra_left_divide(Matrix(S), M; block_size = 1)
    @test eltype(X) == Float64
    @test S * X ≈ M rtol = 1.0e-10
end

@testset "volterra_inverse — long grids, blocks: halving the instants" begin
    # Past `_HALVING_BASE` instants the blocks are inverted by halving the grid:
    # against LAPACK's LU inverse of the same matrix, with the blocks above the
    # diagonal exactly zero, and through ForwardDiff.
    import ForwardDiff
    lower(B, n, s) = [
        i ÷ B ≥ j ÷ B ? (sin(1.3i + 0.7j + s) / (1 + abs(i ÷ B - j ÷ B))) + 3.0 * (i == j) : 0.0
            for i in 0:(B * n - 1), j in 0:(B * n - 1)
    ]
    for B in (2, 3, 6)
        n = 40
        M = lower(B, n, 0.0)
        X = volterra_inverse(M; block_size = B)
        @test X ≈ inv(M) rtol = 1.0e-12
        @test all(iszero, (X[i, j] for i in 1:(B * n), j in 1:(B * n) if (j - 1) ÷ B > (i - 1) ÷ B))
        out = similar(M)
        @test MeanFieldHomogenization.Viscoelasticity.volterra_inverse!(out, M; block_size = B) ≈ X
        d = ForwardDiff.derivative(s -> volterra_inverse(lower(B, n, 0.0) .+ s .* lower(B, n, 1.0); block_size = B), 0.0)
        @test d ≈ -X * lower(B, n, 1.0) * X rtol = 1.0e-10
    end
    # In place, scalar blocks on a short grid: the substitution.
    M = lower(1, 10, 0.0)
    @test MeanFieldHomogenization.Viscoelasticity.volterra_inverse!(similar(M), M; block_size = 1) ≈ inv(M) rtol = 1.0e-13
end
