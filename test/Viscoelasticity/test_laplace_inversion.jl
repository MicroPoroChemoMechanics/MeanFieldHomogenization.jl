using Test
using MeanFieldHomogenization
using SpecialFunctions

# =============================================================================
#  test_laplace_inversion.jl — the four numerical inverse-Laplace algorithms
#  against transform pairs whose inverse is known in closed form.
#
#  The tolerances are not decoration: they encode the accuracy budget stated in
#  the docstrings, so a regression in any algorithm shows up as a failure
#  rather than as a quietly worse number.
# =============================================================================

const METHODS = (
    ("GaverStehfest", GaverStehfest(16)),
    ("FixedTalbot", FixedTalbot(24)),
    ("TalbotTrefethen", TalbotTrefethen(24)),
    ("DeHoog", DeHoog()),
)

relerr(got, want) = abs(got - want) / abs(want)

@testset "inverse_laplace — exponential 1/(p+a) ↔ exp(-a t)" begin
    a = 2.0
    F(p) = 1 / (p + a)
    # The criterion is deliberately *absolute*, measured against `f(0) = 1`:
    # that is what every one of these algorithms actually controls, and at
    # `t = 3` the function is already 2.5e-3 of its own scale, so a relative
    # criterion would be testing the tail rather than the method.
    for (name, m) in METHODS, t in (0.05, 0.5, 1.0, 3.0)
        # Measured maxima over this grid: GS 8e-6, FixedTalbot 9e-13,
        # TalbotTrefethen 7e-15, DeHoog 6e-10.
        atol = m isa GaverStehfest ? 5.0e-5 : 1.0e-8
        @test abs(inverse_laplace(F, t, m) - exp(-a * t)) < atol
    end
    # Where the function has not yet decayed, the relative accuracy is the
    # budget the docstrings quote.
    for (name, m) in METHODS
        tol = m isa GaverStehfest ? 1.0e-6 : 1.0e-9
        @test relerr(inverse_laplace(F, 0.1, m), exp(-0.2)) < tol
    end
end

@testset "inverse_laplace — polynomial 1/p^(n+1) ↔ t^n/n!" begin
    for (name, m) in METHODS, n in (0, 1, 2)
        F(p) = 1 / p^(n + 1)
        for t in (0.5, 3.0)
            tol = m isa GaverStehfest ? 1.0e-5 : (m isa DeHoog ? 1.0e-6 : 1.0e-9)
            @test relerr(inverse_laplace(F, t, m), t^n / factorial(n)) < tol
        end
    end
end

@testset "inverse_laplace — branch cut 1/√p ↔ 1/√(πt)" begin
    # The Talbot contours are Hankel-type: they wrap *around* the cut on the
    # negative real axis rather than crossing it, so a branch point at the
    # origin is their home ground rather than a hazard.  This test exists to
    # pin that down — an earlier design note had it the other way round.
    F(p) = 1 / sqrt(p)
    for (name, m) in METHODS, t in (0.5, 2.0, 50.0)
        tol = m isa GaverStehfest ? 1.0e-5 : 1.0e-8
        @test relerr(inverse_laplace(F, t, m), 1 / sqrt(π * t)) < tol
    end
end

@testset "inverse_laplace — erfc pair exp(-a√p)/p ↔ erfc(a/(2√t))" begin
    a = 0.7
    F(p) = exp(-a * sqrt(p)) / p
    for t in (0.5, 4.0)
        @test relerr(inverse_laplace(F, t, FixedTalbot(32)), erfc(a / (2 * sqrt(t)))) < 1.0e-8
        @test relerr(inverse_laplace(F, t, DeHoog()), erfc(a / (2 * sqrt(t)))) < 1.0e-6
    end
end

@testset "inverse_laplace — oscillatory 1/(p²+ω²) ↔ sin(ωt)/ω" begin
    ω = 3.0
    F(p) = 1 / (p^2 + ω^2)
    for t in (1.0, 2.0)
        want = sin(ω * t) / ω
        # FixedTalbot and DeHoog handle oscillation; the other two do not, and
        # that separation is exactly what the docstrings claim.
        @test relerr(inverse_laplace(F, t, FixedTalbot(24)), want) < 1.0e-9
        @test relerr(inverse_laplace(F, t, DeHoog()), want) < 1.0e-6
        @test relerr(inverse_laplace(F, t, GaverStehfest(16)), want) > 1.0e-3
        @test relerr(inverse_laplace(F, 2.0, TalbotTrefethen(24)), sin(2ω) / ω) > 1.0e-3
    end
end

@testset "inverse_carson — the Prony pair" begin
    E_inf, E1, τ = 2.0, 3.0, 1.5
    Rstar(p) = E_inf + E1 * p * τ / (1 + p * τ)
    Rt(t) = E_inf + E1 * exp(-t / τ)
    for (name, m) in METHODS, t in (0.1, 1.0, 5.0)
        tol = m isa GaverStehfest ? 1.0e-5 : 1.0e-8
        @test relerr(inverse_carson(Rstar, t, m), Rt(t)) < tol
    end
end

@testset "inverse_carson_rate — the derivative identity" begin
    E_inf, E1, τ = 2.0, 3.0, 1.5
    Rstar(p) = E_inf + E1 * p * τ / (1 + p * τ)
    dR(t) = -E1 / τ * exp(-t / τ)
    for (name, m) in METHODS, t in (0.3, 2.0)
        got = inverse_carson_rate(Rstar, t, m; f_glassy = E_inf + E1)
        tol = m isa GaverStehfest ? 1.0e-4 : 1.0e-8
        @test relerr(got, dR(t)) < tol
    end
end

@testset "inverse_laplace — grid form matches the pointwise one" begin
    F(p) = 1 / (p + 1)
    ts = [0.2, 0.5, 1.0, 2.0]
    for (name, m) in METHODS
        grid = inverse_laplace(F, ts, m)
        @test length(grid) == length(ts)
        if m isa DeHoog
            # The grid form blocks the times by scale and shares one node set
            # per block, so it is not bit-identical to the per-point call — but
            # it must be just as accurate.
            @test all(relerr(grid[i], exp(-ts[i])) < 1.0e-7 for i in eachindex(ts))
        else
            @test all(grid[i] == inverse_laplace(F, ts[i], m) for i in eachindex(ts))
        end
    end
end

@testset "DeHoog — blocked grid keeps accuracy over many decades" begin
    # A single shared node set fails badly at small t/T; the default
    # `T = nothing` path splits the grid into scale blocks so that every point
    # sits in the reliable window.  Seven decades is the case that motivated it.
    F(p) = 1 / (p + 1)
    ts = exp10.(range(-3, 3; length = 40))
    grid = inverse_laplace(F, ts, DeHoog())
    # Absolute, for the reason spelled out in the tail testset below: past
    # `t ≈ 30` the exact value is under 1e-13 and only the absolute error means
    # anything.  Measured maximum over the whole seven decades: 1e-9.
    @test all(abs(grid[i] - exp(-ts[i])) < 1.0e-8 for i in eachindex(ts))
    # And it is genuinely uniform — the small-t end is as good as the middle.
    @test abs(grid[1] - exp(-ts[1])) < 1.0e-8

    # One node set for all seven decades is what the blocking avoids: it warns,
    # and at the small-t end it loses everything.
    shared = @test_logs (:warn,) match_mode = :any inverse_laplace(
        F, ts, DeHoog(; T = 2 * maximum(ts))
    )
    @test relerr(shared[1], exp(-ts[1])) > 1.0e-2
end

@testset "inverse_laplace — argument validation" begin
    F(p) = 1 / (p + 1)
    @test_throws DomainError inverse_laplace(F, 0.0)
    @test_throws DomainError inverse_laplace(F, -1.0)
    @test_throws DomainError inverse_laplace(F, [1.0, 0.0])
    @test_throws ArgumentError GaverStehfest(15)      # must be even
    @test_throws ArgumentError GaverStehfest(2)       # too few
    @test_throws ArgumentError GaverStehfest(42)      # too many
    @test_throws ArgumentError FixedTalbot(2)
    @test_throws ArgumentError DeHoog(; tol = 2.0)
    @test_throws ArgumentError DeHoog(; T = -1.0)
end

@testset "GaverStehfest — the weights really are exact rationals" begin
    # At N = 24 the alternating sum needs more than Float64 can carry, so the
    # method is useless there in double precision.  If the weights were baked
    # in as Float64 the BigFloat run would be no better; it is, by seven orders
    # of magnitude, which is the proof that they are converted on demand from
    # exact rationals.
    F64 = inverse_laplace(p -> 1 / (p + 2), 1.0, GaverStehfest(24))
    err64 = relerr(F64, exp(-2.0))
    errbig = setprecision(256) do
        fbig = inverse_laplace(p -> 1 / (p + big(2)), big(1.0), GaverStehfest(24))
        Float64(abs(fbig - exp(big(-2.0))) / exp(big(-2.0)))
    end
    @test err64 > 1.0e-4
    @test errbig < 1.0e-8
    @test errbig < err64 / 1.0e5
end

@testset "the tail is controlled in absolute, not relative, terms" begin
    # Documented behavior, not a defect: once the function has decayed far
    # below its own scale, the relative error is unbounded while the absolute
    # one still meets the method's promise.
    F(p) = 3 / (p + 1)
    t = 40.0
    got = inverse_laplace(F, t, FixedTalbot(24))
    @test abs(got - 3 * exp(-t)) < 1.0e-11          # absolute: excellent
    @test relerr(got, 3 * exp(-t)) > 1.0            # relative: meaningless
end
