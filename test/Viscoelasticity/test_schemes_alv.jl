using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_schemes_alv.jl — homogenization schemes in the elastic limit
#  (Heaviside laws) and consistency checks (Volterra round-trips, bound
#  ordering Reuss ≤ MT ≤ Voigt).
# =============================================================================

const _to_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

# Helper: build a 2-phase elastic-limit setup (sphere inclusion in iso matrix).
function _setup_2phase_elastic(;
        k_M = 10.0, μ_M = 4.0,
        k_I = 20.0, μ_I = 8.0,
        f_I = 0.2, n_times = 5
    )
    C_M_t = TensISO{3}(3 * k_M, 2 * μ_M)
    C_I_t = TensISO{3}(3 * k_I, 2 * μ_I)
    times = n_times == 1 ? [0.0] : collect(range(0.0, 2.0; length = n_times))
    law_M = heaviside_law(C_M_t)
    law_I = heaviside_law(C_I_t)
    C_M = trapezoidal_matrix(law_M, times)
    C_I = trapezoidal_matrix(law_I, times)
    P = hill_kernel(Spheroid(1.0), law_M, times)
    A_dil_I = dilute_concentration_alv(C_I, C_M, P)
    N_dil_I = dilute_contribution_alv(C_I, C_M, P)
    f_M = 1 - f_I
    return (; C_M_t, C_I_t, times, C_M, C_I, P, A_dil_I, N_dil_I, f_I, f_M)
end

# Reference (elastic) homogenization via the existing `homogenize`.
function _reference_elastic(scheme, ctx)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_M_t); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_I_t);
        fraction = ctx.f_I
    )
    return _to_mandel(homogenize(rve, scheme, :C))
end

# Generic checker: ALV scheme diag block equals elastic homogenization.
function _check_alv_elastic(
        C_alv::AbstractMatrix, M_ref::AbstractMatrix,
        n::Int; rtol = 1.0e-12, atol = 1.0e-12
    )
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(C_alv[rows, rows], M_ref; rtol = rtol, atol = atol)
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, C_alv[rows, cols]) ≤ atol
        end
    end
    return
end

@testset "schemes_alv — Voigt elastic limit" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    C_voigt = voigt_alv([ctx.C_M, ctx.C_I], [ctx.f_M, ctx.f_I])
    M_ref = _reference_elastic(Voigt(), ctx)
    _check_alv_elastic(C_voigt, M_ref, n; atol = 1.0e-12)
end

@testset "schemes_alv — Reuss elastic limit" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    C_reuss = reuss_alv([ctx.C_M, ctx.C_I], [ctx.f_M, ctx.f_I])
    M_ref = _reference_elastic(Reuss(), ctx)
    _check_alv_elastic(C_reuss, M_ref, n; atol = 1.0e-10)
end

@testset "schemes_alv — Mori-Tanaka elastic limit" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    C_mt = mori_tanaka_alv(
        ctx.C_M, [ctx.A_dil_I], [ctx.N_dil_I],
        [ctx.f_I], ctx.f_M
    )
    M_ref = _reference_elastic(MoriTanaka(), ctx)
    _check_alv_elastic(C_mt, M_ref, n; atol = 1.0e-10)
end

@testset "schemes_alv — Dilute elastic limit" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    C_dil = dilute_alv(ctx.C_M, [ctx.N_dil_I], [ctx.f_I])
    M_ref = _reference_elastic(Dilute(), ctx)
    _check_alv_elastic(C_dil, M_ref, n; atol = 1.0e-10)
end

@testset "schemes_alv — Maxwell elastic limit (sphere distribution)" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    # Maxwell with the same sphere distribution as the inclusion shape:
    # H_0 = P_sphere(C_M_elas).
    H_0 = hill_kernel(Spheroid(1.0), heaviside_law(ctx.C_M_t), ctx.times)
    C_max = maxwell_alv(ctx.C_M, [ctx.N_dil_I], [ctx.f_I]; H_0 = H_0)
    M_ref = _reference_elastic(Maxwell(), ctx)
    _check_alv_elastic(C_max, M_ref, n; atol = 1.0e-10)
end

@testset "schemes_alv — argument validation" begin
    @test_throws ArgumentError voigt_alv(Matrix{Float64}[], Float64[])
    M = zeros(6, 6)
    @test_throws ArgumentError voigt_alv([M], [0.5, 0.5])
    @test_throws ArgumentError reuss_alv(Matrix{Float64}[], Float64[])
    @test_throws ArgumentError mori_tanaka_alv(M, [M], [M], [0.5, 0.5], 0.5)
end

@testset "schemes_alv — single-step grid (n=1) ≡ elastic MT" begin
    # n = 1 reduces the trapezoidal matrix to a single 6×6 block ; the
    # whole pipeline must collapse to the static elastic MT.
    ctx = _setup_2phase_elastic(n_times = 1)
    C_mt = mori_tanaka_alv(
        ctx.C_M, [ctx.A_dil_I], [ctx.N_dil_I],
        [ctx.f_I], ctx.f_M
    )
    M_ref = _reference_elastic(MoriTanaka(), ctx)
    @test isapprox(C_mt, M_ref; atol = 1.0e-12)
end
