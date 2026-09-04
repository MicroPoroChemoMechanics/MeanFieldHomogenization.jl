# =============================================================================
#  test_dispatch.jl — scheme-type hierarchy + homogenize entry point.
#
#  At this stage no concrete scheme implements `_evaluate` yet, so the
#  fallback must throw an explicit "not yet implemented" error. The Symbol
#  shortcut + alias table are validated here too.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND

@testset "Scheme types — hierarchy and constructors" begin
    @test Voigt() isa HomogenizationScheme
    @test Reuss() isa HomogenizationScheme
    @test Dilute() isa HomogenizationScheme
    @test DiluteDual() isa HomogenizationScheme
    @test MoriTanaka() isa HomogenizationScheme
    @test Maxwell() isa HomogenizationScheme
    @test PonteCastanedaWillis() isa HomogenizationScheme

    sc = SelfConsistent()
    asc = AsymmetricSelfConsistent()
    @test sc isa HomogenizationScheme
    @test asc isa HomogenizationScheme
    @test sc.algorithm isa AndersonDefault
    @test asc.algorithm isa AndersonDefault

    # Custom solver/options
    sc_newton = SelfConsistent(; algorithm = NewtonDefault(), abstol = 1.0e-12, maxiters = 200)
    @test sc_newton.algorithm isa NewtonDefault
    @test sc_newton.options.abstol == 1.0e-12
    @test sc_newton.options.maxiters == 200

    # Differential + trajectories
    @test Proportional() isa DifferentialTrajectory
    @test Sequential([:I1, :I2]) isa DifferentialTrajectory
    @test CustomPath(Dict(:I1 => collect(0.0:0.1:1.0))) isa DifferentialTrajectory
    diff_default = DifferentialScheme()
    @test diff_default.trajectory isa Proportional
    @test diff_default.options.nsteps == 100
end

@testset "homogenize — Symbol shortcut + alias table" begin
    # Every concrete scheme appears in SCHEME_ALIAS at least once
    expected = [
        Voigt, Reuss, Dilute, DiluteDual, MoriTanaka, Maxwell,
        PonteCastanedaWillis, SelfConsistent, AsymmetricSelfConsistent,
        DifferentialScheme,
    ]
    for T in expected
        @test T in values(MeanFieldHomogenization.Schemes.SCHEME_ALIAS)
    end

    # Canonical lowercase aliases (consistency with :auto / :residues / :decuhr)
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:voigt] === Voigt
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:mori_tanaka] === MoriTanaka
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:mt] === MoriTanaka
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:self_consistent] === SelfConsistent
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:sc] === SelfConsistent
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:differential] === DifferentialScheme
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:diff] === DifferentialScheme
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:pcw] === PonteCastanedaWillis

    # CamelCase / ECHOES-compatible aliases still accepted for backwards compat
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:MT] === MoriTanaka
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:SC] === SelfConsistent
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:DIFF] === DifferentialScheme
    @test MeanFieldHomogenization.Schemes.SCHEME_ALIAS[:PCW] === PonteCastanedaWillis

    # Unknown alias raises a clear error
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    @test_throws ArgumentError homogenize(rve, :bogus)
end

@testset "homogenize — unknown Symbol fallback" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    @test_throws ArgumentError homogenize(rve, :not_a_scheme)
    @test_throws ArgumentError homogenize(rve, :foobar)
end
