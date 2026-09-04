using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_sc_alv.jl — Self-Consistent ALV scheme.
# =============================================================================

const _to_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

@testset "self_consistent_alv — elastic limit (Heaviside)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 8.0))); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(60.0, 16.0)));
        fraction = 0.2
    )

    times = collect(0.0:0.5:2.0)
    n = length(times)
    C_alv = self_consistent_alv(
        rve, :C; times = times, abstol = 1.0e-12,
        maxiters = 500
    )

    # Reference: elastic SC.
    rve_e = RVE()
    add_phase!(rve_e, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => TensISO{3}(30.0, 8.0)); fraction = :rest)
    add_phase!(
        rve_e, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => TensISO{3}(60.0, 16.0)); fraction = 0.2
    )
    C_e_M = _to_mandel(homogenize(rve_e, SelfConsistent(), :C))

    # Every diagonal block must match the elastic SC ; off-diag = 0.
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(C_alv[rows, rows], C_e_M; atol = 1.0e-10)
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, C_alv[rows, cols]) ≤ 1.0e-12
        end
    end
end

@testset "self_consistent_alv — homogenize_alv dispatch (SelfConsistent)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(60.0, 16.0)));
        fraction = 0.2
    )
    times = collect(0.0:0.25:1.0)
    n = length(times)

    # Direct vs dispatcher.
    C_direct = self_consistent_alv(rve, :C; times = times, abstol = 1.0e-9)
    C_dispatch = homogenize_alv(
        rve, SelfConsistent(), :C; times = times,
        abstol = 1.0e-9
    )
    @test C_direct ≈ C_dispatch atol = 1.0e-12
    @test size(C_direct) == (6n, 6n)

    # Block lower-triangular structure preserved by SC iteration.
    for i in 1:n, j in 1:n
        if j > i
            rows = (6 * (i - 1) + 1):(6 * i)
            cols = (6 * (j - 1) + 1):(6 * j)
            # SC iterates can introduce small residuals above the
            # diagonal — tolerate machine-precision noise.
            @test maximum(abs, C_direct[rows, cols]) ≤ 1.0e-8
        end
    end
end

@testset "self_consistent_alv — single-phase trivial fixed point" begin
    # If only the matrix exists (no inclusions), SC should return the
    # matrix kernel itself at the first iteration.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); fraction = :rest)
    times = collect(0.0:0.5:2.0)
    C_alv = self_consistent_alv(
        rve, :C; times = times, abstol = 1.0e-12,
        maxiters = 50
    )
    C_M = trapezoidal_matrix(maxwell_iso(10.0, 4.0, 1.0, 0.5), times)
    @test isapprox(C_alv, C_M; atol = 1.0e-10)
end
