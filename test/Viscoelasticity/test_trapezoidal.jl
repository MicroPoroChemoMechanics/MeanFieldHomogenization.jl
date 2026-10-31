using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_trapezoidal.jl — discretization of Stieltjes integrals
# =============================================================================

@testset "trapezoidal_matrix — scalar Heaviside (elastic limit)" begin
    # H(t-t') · C → diagonal-only matrix with C on every block.
    law = heaviside_law(2.5)
    times = collect(0.0:0.5:2.0)
    M = trapezoidal_matrix(law, times)
    @test size(M) == (5, 5)
    @test all(diag(M) .≈ 2.5)
    @test all(abs.(M - Diagonal(M)) .≤ 1.0e-14)
end

@testset "trapezoidal_matrix — scalar Maxwell relaxation" begin
    τ = 0.5
    law = ViscoLaw((t, tp) -> t < tp ? 0.0 : exp(-(t - tp) / τ), :relaxation)
    times = collect(0.0:0.25:1.0)
    M = trapezoidal_matrix(law, times)
    @test size(M) == (5, 5)

    @test istril(M)                     # lower-triangular structure

    # Verify a couple of entries against the trapezoidal formula.
    # M[1, 1] = f(t_0, t_0) = 1.0
    @test M[1, 1] ≈ 1.0
    # M[2, 2] = 0.5 * (f(t_1, t_0) + f(t_1, t_1)) = 0.5 * (exp(-0.5) + 1.0)
    @test M[2, 2] ≈ 0.5 * (exp(-0.25 / τ) + 1.0)
    # M[2, 1] = 0.5 * (f(t_1, t_0) - f(t_1, t_1)) = 0.5 * (exp(-0.5) - 1.0)
    @test M[2, 1] ≈ 0.5 * (exp(-0.25 / τ) - 1.0)
end

@testset "trapezoidal_matrix — scalar Maxwell convergence to analytical creep" begin
    # For a Maxwell scalar relaxation R(t-t') = exp(-(t-t')/τ), the convolution
    # `R * 1` (response to a unit step) is `R(t-t_0)` itself.  The discrete
    # `M * (1, 1, …, 1)` should converge to `R(t_i - t_0)` as the grid refines.
    τ = 0.5
    law = ViscoLaw((t, tp) -> t < tp ? 0.0 : exp(-(t - tp) / τ), :relaxation)
    for n in (50, 200)
        times = collect(range(0.0, 2.0; length = n))
        M = trapezoidal_matrix(law, times)
        unit_step = ones(n)
        y = M * unit_step
        # Compare to R(t_i - t_0) at each i (== exp(-t_i/τ)) with rtol that
        # tightens with grid resolution.
        rtol = n == 50 ? 1.0e-2 : 5.0e-4
        for i in 2:n
            @test y[i] ≈ exp(-times[i] / τ) rtol = rtol
        end
    end
end

@testset "trapezoidal_matrix — 4-tensor Heaviside (elastic limit)" begin
    C = TensISO{3}(30.0, 8.0)   # k = 10, μ = 4
    law = heaviside_law(C)
    times = collect(0.0:0.25:1.0)
    M = trapezoidal_matrix(law, times)
    @test size(M) == (30, 30)   # 6n × 6n with n = 5

    # Each diagonal 6×6 block must equal Mandel(C); off-diagonal must be 0.
    voigt = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
    sq2 = sqrt(2.0)
    expected_block = zeros(Float64, 6, 6)
    arr_C = TensND.get_array(C)
    for I in 1:6, J in 1:6
        i, j = voigt[I]
        k, l = voigt[J]
        scale = (I ≥ 4 ? sq2 : 1.0) * (J ≥ 4 ? sq2 : 1.0)
        expected_block[I, J] = arr_C[i, j, k, l] * scale
    end

    for i in 1:5, j in 1:5
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        if i == j
            @test M[rows, cols] ≈ expected_block atol = 1.0e-12
        else
            @test all(iszero, M[rows, cols])
        end
    end
end

@testset "trapezoidal_matrix — Maxwell iso 4-tensor block layout" begin
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    times = collect(0.0:0.25:1.0)
    M = trapezoidal_matrix(law, times)
    @test size(M) == (30, 30)

    # Block (1, 1) at t = 0 should be the elastic stiffness in Mandel.
    block_11 = M[1:6, 1:6]
    # Mandel diagonal: K + 4μ/3 = 10 + 16/3 = 46/3
    @test block_11[1, 1] ≈ 10.0 + 4 * 4.0 / 3
    # Off-diagonal (1,2): K - 2μ/3 = 10 - 8/3 = 22/3
    @test block_11[1, 2] ≈ 10.0 - 2 * 4.0 / 3
    # Mandel shear (4,4): 2μ = 8
    @test block_11[4, 4] ≈ 2 * 4.0
end

@testset "trapezoidal_matrix — empty / single-step grid" begin
    law = heaviside_law(1.0)
    @test_throws ArgumentError trapezoidal_matrix(law, Float64[])
    M = trapezoidal_matrix(law, [0.0])
    @test M == fill(1.0, 1, 1)
end

@testset "trapezoidal_matrix and homogenize_alv — a law of neither order is refused" begin
    # A kernel valued in 2×2 matrices is neither a conductivity (3×3) nor a stiffness
    # (6×6 Kelvin–Mandel): both entry points say so instead of guessing.
    law = ViscoLaw((t, tp) -> zeros(2, 2))
    @test_throws "only 3×3 (order-2) or 6×6" trapezoidal_matrix(law, [0.0, 1.0])
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => law); fraction = :rest)
    @test_throws "cannot infer ALV order" homogenize_alv(r, Voigt(), :C; times = [0.0, 1.0])
end
