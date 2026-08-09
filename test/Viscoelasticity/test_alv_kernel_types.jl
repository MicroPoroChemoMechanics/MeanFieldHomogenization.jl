using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_alv_kernel_types.jl — `ALVKernelISO / ALVKernelTI / ALVKernelOrtho`
#  AbstractMatrix interface, algebra closure, cross-symmetry promotion,
#  round-trip with the dense (6n × 6n) form.
# =============================================================================

# Helper: random lower-triangular n×n Volterra-like matrix (positive diag).
function _alvk_rand_lt(n)
    M = randn(n, n)
    @inbounds for i in 1:n, j in (i + 1):n
        M[i, j] = 0.0
    end
    @inbounds for i in 1:n
        M[i, i] = 1.0 + abs(randn())
    end
    return M
end

@testset "ALVKernelISO — AbstractMatrix interface" begin
    n = 4
    α = _alvk_rand_lt(n); β = _alvk_rand_lt(n)
    K = ALVKernelISO(α, β)

    @test K isa AbstractMatrix{Float64}
    @test size(K) == (6n, 6n)
    @test size(K, 1) == 6n
    @test eltype(K) === Float64
    @test istril(K)

    # Dense round-trip
    M = Matrix(K)
    @test size(M) == (6n, 6n)
    K_back = ALVKernelISO(M)
    @test isapprox(K_back.α, α; atol = 1.0e-14)
    @test isapprox(K_back.β, β; atol = 1.0e-14)

    # `getindex` lazy view matches dense materialization
    for i in 1:6n, j in 1:6n
        @test K[i, j] == M[i, j]
    end
end

@testset "ALVKernelISO — algebra closure" begin
    n = 5
    α₁ = _alvk_rand_lt(n); β₁ = _alvk_rand_lt(n)
    α₂ = _alvk_rand_lt(n); β₂ = _alvk_rand_lt(n)
    K₁ = ALVKernelISO(α₁, β₁)
    K₂ = ALVKernelISO(α₂, β₂)

    K_sum = K₁ + K₂
    @test K_sum isa ALVKernelISO
    @test isapprox(Matrix(K_sum), Matrix(K₁) + Matrix(K₂); atol = 1.0e-12)

    K_diff = K₁ - K₂
    @test K_diff isa ALVKernelISO
    @test isapprox(Matrix(K_diff), Matrix(K₁) - Matrix(K₂); atol = 1.0e-12)

    K_prod = K₁ * K₂
    @test K_prod isa ALVKernelISO
    @test isapprox(Matrix(K_prod), Matrix(K₁) * Matrix(K₂); atol = 1.0e-12)

    K_scaled = 2.5 * K₁
    @test K_scaled isa ALVKernelISO
    @test isapprox(Matrix(K_scaled), 2.5 * Matrix(K₁); atol = 1.0e-12)

    K_inv = volterra_inverse(K₁)
    @test K_inv isa ALVKernelISO
    M_id = Matrix(K₁) * Matrix(K_inv)
    H_id = zeros(6n, 6n)
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        H_id[rows, rows] = Matrix{Float64}(I, 6, 6)
    end
    @test isapprox(M_id, H_id; atol = 1.0e-10)

    K_div = volterra_left_divide(K₁, K₂)
    @test K_div isa ALVKernelISO
    @test isapprox(Matrix(K_div), Matrix(K_inv) * Matrix(K₂); atol = 1.0e-9)
end

@testset "ALVKernelTI — round-trip and algebra" begin
    n = 4
    ℓ = ntuple(_ -> _alvk_rand_lt(n), 6)
    K = ALVKernelTI(ℓ)

    @test K isa AbstractMatrix{Float64}
    @test size(K) == (6n, 6n)
    @test K.axis == (0.0, 0.0, 1.0)

    M = Matrix(K)
    K_back = ALVKernelTI(M)
    for k in 1:6
        @test isapprox(K_back.ℓ[k], ℓ[k]; atol = 1.0e-14)
    end

    for i in 1:6n, j in 1:6n
        @test K[i, j] ≈ M[i, j] atol = 1.0e-14
    end

    # Algebra
    ℓ₂ = ntuple(_ -> _alvk_rand_lt(n), 6)
    K₂ = ALVKernelTI(ℓ₂)
    K_prod = K * K₂
    @test K_prod isa ALVKernelTI
    @test isapprox(Matrix(K_prod), M * Matrix(K₂); atol = 1.0e-12)
    K_inv = volterra_inverse(K)
    @test K_inv isa ALVKernelTI
end

@testset "ALVKernelOrtho — round-trip and algebra" begin
    n = 4
    o = ntuple(_ -> _alvk_rand_lt(n), 12)
    # Make the normal 3×3 block diagonally dominant so the inverse is clean.
    for i in 1:n
        o[1][i, i] += 5.0
        o[5][i, i] += 5.0
        o[9][i, i] += 5.0
    end
    K = ALVKernelOrtho(o)

    @test K isa AbstractMatrix{Float64}
    @test size(K) == (6n, 6n)

    M = Matrix(K)
    K_back = ALVKernelOrtho(M)
    for k in 1:12
        @test isapprox(K_back.o[k], o[k]; atol = 1.0e-14)
    end

    for i in 1:6n, j in 1:6n
        @test K[i, j] ≈ M[i, j] atol = 1.0e-14
    end

    K₂ = K
    K_prod = K * K₂
    @test K_prod isa ALVKernelOrtho
    @test isapprox(Matrix(K_prod), M * Matrix(K₂); atol = 1.0e-12)
    K_inv = volterra_inverse(K)
    @test K_inv isa ALVKernelOrtho
end

@testset "ALV kernel ladder — iso ⊂ TI ⊂ ortho" begin
    n = 4
    α = _alvk_rand_lt(n); β = _alvk_rand_lt(n)
    K_iso = ALVKernelISO(α, β)
    K_TI = ALVKernelTI(K_iso)
    K_O = ALVKernelOrtho(K_iso)

    @test K_TI isa ALVKernelTI
    @test K_O isa ALVKernelOrtho

    # Conversions preserve the materialized matrix.
    @test isapprox(Matrix(K_iso), Matrix(K_TI); atol = 1.0e-12)
    @test isapprox(Matrix(K_iso), Matrix(K_O); atol = 1.0e-12)

    # TI → ortho promotion
    K_O2 = ALVKernelOrtho(K_TI)
    @test isapprox(Matrix(K_O2), Matrix(K_TI); atol = 1.0e-12)

    # Mixed arithmetic auto-promotes
    K_sum_iso_TI = K_iso + K_TI
    @test K_sum_iso_TI isa ALVKernelTI
    @test isapprox(Matrix(K_sum_iso_TI), Matrix(K_iso) + Matrix(K_TI); atol = 1.0e-12)

    K_prod_iso_O = K_iso * K_O
    @test K_prod_iso_O isa ALVKernelOrtho
    @test isapprox(Matrix(K_prod_iso_O), Matrix(K_iso) * Matrix(K_O); atol = 1.0e-12)

    K_div_TI_O = volterra_left_divide(K_TI, K_O)
    @test K_div_TI_O isa ALVKernelOrtho
end

@testset "ALV kernel — Maxwell iso ViscoLaw round-trip" begin
    # Build through `trapezoidal_matrix` and verify ALVKernelISO captures
    # the iso symmetry exactly for a Maxwell iso law.
    times = collect(range(0.0, 1.5; length = 5))

    function R_iso(t, tp)
        α = 3 * (1.0 + 2.0 * exp(-(t - tp) / 0.5))
        β = 2 * (0.8 + 1.2 * exp(-(t - tp) / 0.7))
        return TensISO{3}(α, β)
    end
    law = ViscoLaw(R_iso, :relaxation)
    M = trapezoidal_matrix(law, times)

    K = ALVKernelISO(M)
    @test isapprox(Matrix(K), M; atol = 1.0e-12)

    # Volterra inverse on K matches the dense block-LU.
    K_inv = volterra_inverse(K)
    M_inv = volterra_inverse(M; block_size = 6)
    @test isapprox(Matrix(K_inv), M_inv; atol = 1.0e-9)
end
