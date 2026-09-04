using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_sc_alv_newton.jl — row-by-row Newton-Raphson for the ALV SC scheme
#  (`src/Viscoelasticity/schemes_alv_sc_newton.jl`).
#
#  NOTE: `self_consistent_alv_newton` is neither exported nor called from the
#  rest of the package — it is reachable only through its qualified path. It
#  is tested here against the two references available: the elastic SC in the
#  Heaviside limit, and the Anderson-Picard ALV SC in viscoelasticity.
# =============================================================================

const _sc_newton = MeanFieldHomogenization.Viscoelasticity.self_consistent_alv_newton
const _to_mandel66 = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

@testset "sc_alv_newton — elastic limit (Heaviside)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 8.0))); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(60.0, 16.0)));
        fraction = 0.2
    )

    times = collect(0.0:0.5:1.5)
    n = length(times)
    C_newton = _sc_newton(rve, :C; times = times, abstol = 1.0e-12)

    @test size(C_newton) == (6n, 6n)

    # Reference: the elastic SC on the same moduli.
    rve_e = RVE()
    add_phase!(rve_e, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => TensISO{3}(30.0, 8.0)); fraction = :rest)
    add_phase!(
        rve_e, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => TensISO{3}(60.0, 16.0)); fraction = 0.2
    )
    C_e = _to_mandel66(homogenize(rve_e, SelfConsistent(), :C))

    # Under a Heaviside law the response is constant: the diagonal blocks are
    # the elastic SC and the off-diagonal ones vanish.  The row-by-row Newton
    # and the elastic SC are two distinct solvers: they converge to the same
    # root up to their own tolerances (~1e-10 relative), hence an `rtol`
    # rather than a tight `atol`.
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(C_newton[rows, rows], C_e; rtol = 1.0e-7, atol = 1.0e-10)
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, C_newton[rows, cols]) ≤ 1.0e-10
        end
    end
end

@testset "sc_alv_newton — agreement with the Anderson-Picard ALV SC" begin
    # Both solvers look for the same root `C_eff = step(C_eff)`; on a
    # low-contrast configuration they must land on the same
    # point fixe.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(20.0, 8.0)));
        fraction = 0.15
    )

    times = collect(0.0:0.25:1.0)
    C_newton = _sc_newton(rve, :C; times = times, abstol = 1.0e-12)
    C_picard = self_consistent_alv(
        rve, :C; times = times, abstol = 1.0e-12,
        maxiters = 2000
    )

    @test isapprox(C_newton, C_picard; rtol = 1.0e-6, atol = 1.0e-9)
end

@testset "sc_alv_newton — causality (block lower triangular)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(12.0, 5.0, 1.0, 0.4)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(30.0, 12.0)));
        fraction = 0.1
    )

    times = collect(0.0:0.25:0.75)
    n = length(times)
    C = _sc_newton(rve, :C; times = times, abstol = 1.0e-12)

    # The ALV SC equation is causal: row i depends only on rows ≤ i, so the
    # strictly upper block must vanish.
    for i in 1:n, j in (i + 1):n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        @test maximum(abs, C[rows, cols]) ≤ 1.0e-10
    end
end

@testset "sc_alv_newton — cracked phase (CrackDensity)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); fraction = :rest)
    add_phase!(
        rve, :F, Ellipsoid(1.0, 1.0, 0.0),
        Dict(:C => heaviside_law(TensISO{3}(1.0e-9, 1.0e-9)));
        density = 0.05
    )

    times = collect(0.0:0.25:0.75)
    n = length(times)
    C = _sc_newton(rve, :C; times = times, abstol = 1.0e-10)

    @test size(C) == (6n, 6n)
    @test all(isfinite, C)

    # Cracks soften the composite: at t = 0 the effective stiffness must be
    # strictly below that of the matrix alone.
    C_M = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(
        phase_property(rve, :M, :C), times, 6
    )
    @test C[1, 1] < C_M[1, 1]
end

@testset "sc_alv_newton — input errors" begin
    # An elastic matrix property (not a ViscoLaw).
    rve_e = RVE()
    add_phase!(rve_e, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => TensISO{3}(30.0, 8.0)); fraction = :rest)
    @test_throws ArgumentError _sc_newton(rve_e, :C; times = [0.0, 1.0])

    # An elastic phase property while the matrix is viscous.
    rve_p = RVE()
    add_phase!(rve_p, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 8.0))); fraction = :rest)
    add_phase!(
        rve_p, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => TensISO{3}(60.0, 16.0)); fraction = 0.1
    )
    @test_throws ArgumentError _sc_newton(rve_p, :C; times = [0.0, 1.0])

    # A CrackDensity on a geometry that is not a crack.
    rve_c = RVE()
    add_phase!(rve_c, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 8.0))); fraction = :rest)
    add_phase!(
        rve_c, :F, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(TensISO{3}(1.0e-9, 1.0e-9)));
        density = 0.05
    )
    @test_throws ArgumentError _sc_newton(rve_c, :C; times = [0.0, 1.0])
end
