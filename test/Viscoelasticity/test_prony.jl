using Test
using MeanFieldHomogenization
using ForwardDiff

# =============================================================================
#  test_prony.jl — the exact Kelvin ↔ Maxwell conversion.
#
#  Three independent oracles, deliberately chosen so that no single mistake
#  could satisfy all of them:
#
#    * the reciprocal identity `J*(p) R*(p) = 1`, which is what the conversion
#      *means*, checked at machine precision across a `p` sweep;
#    * the analytic Zener relations printed by the ECHOES reference
#      `Kelvin2Maxwell.py`, which pin the residue formulas against hand algebra;
#    * the closed-form Burgers relaxation function, a `cosh`/`sinh` expression
#      derived independently in Maple, which exercises the *fluid* path —
#      degenerate-branch detection, the unbounded outermost bracket, and both
#      residue signs — in one shot.
# =============================================================================

@testset "interlacing — the structure the algorithm rests on" begin
    τ = [0.5, 3.0, 17.0, 120.0]
    m = PronyRelaxation(4.0, [1.0, 2.0, 3.0, 4.0], τ)
    k = maxwell_to_kelvin(m)
    # A solid: one root in every gap, plus one beyond the last pole.
    @test length(k) == length(m)
    @test all(τ[i] < k.tau[i] < τ[i + 1] for i in 1:(length(τ) - 1))
    @test k.tau[end] > τ[end]
    # There is deliberately no root below the first pole in this direction.
    @test k.tau[1] > τ[1]

    # The other direction is *not* symmetric: it does have one below τ₁.
    r = kelvin_to_maxwell(k)
    @test r.tau[1] < k.tau[1]
end

@testset "reciprocity J*(p) R*(p) = 1" begin
    m = PronyRelaxation(8.0, [3.0, 17.0], [12.0, 23.0])   # the ECHOES numbers
    k = maxwell_to_kelvin(m)
    for p in (1.0e-6, 1.0e-3, 0.037, 1.0, 37.0, 1.0e4)
        @test isapprox(carson_creep(k, p) * carson_relaxation(m, p), 1.0; atol = 1.0e-12)
    end
    for ω in (1.0e-3, 1.0, 1.0e3)                          # and off the real axis
        @test isapprox(
            carson_creep(k, im * ω) * carson_relaxation(m, im * ω), 1.0 + 0im;
            atol = 1.0e-12
        )
    end
end

@testset "round trip at 1, 2, 3 and 12 branches" begin
    cases = (
        PronyRelaxation(5.0, [2.0], [1.0]),
        PronyRelaxation(8.0, [3.0, 17.0], [12.0, 23.0]),
        PronyRelaxation(1.0, [1.0, 2.0, 4.0], [0.1, 1.0, 10.0]),
        # Twelve branches over six decades — the regime where expanding the
        # transform into a polynomial, as the SymPy reference does, loses its
        # conditioning entirely.
        PronyRelaxation(2.0, fill(1.0, 12), exp10.(range(-3, 3; length = 12))),
    )
    for m in cases
        back = kelvin_to_maxwell(maxwell_to_kelvin(m))
        @test back.E_inf ≈ m.E_inf rtol = 1.0e-12
        @test all(isapprox.(back.E, m.E; rtol = 1.0e-12))
        @test all(isapprox.(back.tau, m.tau; rtol = 1.0e-12))
    end
end

@testset "Zener — the analytic identities of Kelvin2Maxwell.py" begin
    # `Kelvin2Maxwell.py` lines 89-101: with E₀ the equilibrium spring and
    # (E₁, τ₁) the single Maxwell branch,
    #     J₀ = 1/(E₀+E₁),  J₁ = 1/E₀ - J₀,  η₁ = η (E₀+E₁)²/E₁²,
    # and the retardation time must equal both η₁J₁ and τ₁(E₀+E₁)/E₀.
    E0, E1, τ1 = 8.0, 3.0, 12.0
    k = maxwell_to_kelvin(zener_maxwell(E0, E1, τ1))
    η = E1 * τ1
    J0 = 1 / (E0 + E1)
    J1 = 1 / E0 - J0
    η1 = η * ((E0 + E1) / E1)^2

    @test k.J_0 ≈ J0 rtol = 1.0e-14
    @test k.J[1] ≈ J1 rtol = 1.0e-12
    @test k.tau[1] ≈ η1 * J1 rtol = 1.0e-12
    @test k.tau[1] ≈ τ1 * (E0 + E1) / E0 rtol = 1.0e-12
    @test k.tau[1] ≈ 16.5 rtol = 1.0e-12
    @test iszero(k.phi)

    # And the two named constructors describe the same material.
    z_k = zener_kelvin(1 / k.J_0, 1 / k.J[1], k.tau[1])
    @test carson_creep(z_k, 0.37) ≈ carson_creep(k, 0.37) rtol = 1.0e-12
end

@testset "Burgers — the fluid path against the closed-form relaxation" begin
    ks, ηs, kp, ηp = 1.0, 3.0, 2.0, 6.0
    b = burgers(ks, ηs, kp, ηp)
    @test is_fluid(b)
    r = kelvin_to_maxwell(b)

    # A fluid Kelvin chain of one branch converts to a *two*-branch Maxwell
    # chain with no equilibrium spring — the degree count of the transform.
    @test length(r) == 2
    @test isapprox(r.E_inf, 0.0; atol = 1.0e-12)
    @test is_fluid(r)
    @test all(>(0), r.E)

    # The Maple closed form, from
    # tests/python/creep/fluage_echoes_ijss2013_jsanahuja_relaxBurgers.py
    function Rb(t)
        d = sqrt(
            ηp^2 * ks^2 - 2ηp * ηs * ks * kp + 2ηp * ηs * ks^2 +
                ηs^2 * kp^2 + 2ηs^2 * kp * ks + ηs^2 * ks^2
        )
        return exp(-0.5t * kp / ηp - 0.5t * ks / ηp - 0.5t * ks / ηs) * (
            ks * cosh(0.5t * d / (ηp * ηs)) +
                sinh(0.5t * d / (ηp * ηs)) * (-ηp * ks + ηs * kp - ks * ηs) * ks / d
        )
    end
    for t in (0.0, 0.1, 1.0, 5.0, 20.0, 50.0)
        @test isapprox(relaxation(r, t), Rb(t); atol = 1.0e-12)
    end

    # The creep side is untouched by the conversion.
    for t in (0.5, 5.0)
        @test creep(b, t) ≈ 1 / ks + t / ηs + (1 - exp(-t * kp / ηp)) / kp rtol = 1.0e-14
    end
end

@testset "fluid Maxwell chain converts back to a fluid Kelvin chain" begin
    # The mirror image: E_∞ = 0 means the outermost root has receded to
    # infinity, which is the series dashpot of the Kelvin form.
    m = PronyRelaxation(0.0, [2.0, 5.0], [1.0, 10.0])
    @test is_fluid(m)
    k = maxwell_to_kelvin(m)
    @test length(k) == length(m) - 1
    @test k.phi ≈ 1 / sum(m.E .* m.tau) rtol = 1.0e-14      # φ = 1/R*'(0)
    for p in (1.0e-4, 1.0, 1.0e4)
        @test isapprox(carson_creep(k, p) * carson_relaxation(m, p), 1.0; atol = 1.0e-12)
    end
end

@testset "the free consistency identity, and the sign structure" begin
    k = PronyCreep(0.1, [0.02, 0.05, 0.01], [0.3, 4.0, 60.0])
    r = kelvin_to_maxwell(k)
    # Both sides are R*(0).
    @test r.E_inf ≈ 1 / (k.J_0 + sum(k.J)) rtol = 1.0e-10
    @test 1 / k.J_0 - sum(r.E) ≈ 1 / (k.J_0 + sum(k.J)) rtol = 1.0e-10
    # Positivity is structural, not incidental.
    @test all(>(0), r.E)
    @test r.E_inf > 0
    @test all(>(0), maxwell_to_kelvin(r).J)
end

@testset "degenerate spectra: merging and the ill-conditioning warning" begin
    # Two branches at the same time are one branch, not an error.
    m = PronyRelaxation(1.0, [2.0, 3.0], [5.0, 5.0])
    @test length(m) == 1
    @test m.E[1] ≈ 5.0
    # Nearly equal times still convert exactly, but warn about conditioning.
    near = PronyRelaxation(1.0, [2.0, 3.0], [5.0, 5.0 * (1 + 1.0e-6)])
    @test length(near) == 2
    k = @test_logs (:warn,) match_mode = :any maxwell_to_kelvin(near)
    @test isapprox(carson_creep(k, 1.0) * carson_relaxation(near, 1.0), 1.0; atol = 1.0e-9)
end

@testset "constructor validation" begin
    @test_throws ArgumentError PronyRelaxation(1.0, [1.0, 2.0], [1.0])
    @test_throws ArgumentError PronyRelaxation(1.0, [1.0], [-1.0])
    @test_throws ArgumentError PronyRelaxation(1.0, [1.0], [Inf])
    @test_throws ArgumentError PronyCreep(-1.0, [1.0], [1.0])
end

@testset "ForwardDiff through the conversion" begin
    # The roots are found by bisection on values and then lifted by a single
    # Newton step in the Dual type — the implicit function theorem for a simple
    # root.  Compared here against central differences.
    mk(θ) = PronyRelaxation(θ[1], [θ[2], θ[3]], [θ[4], θ[5]])
    θ0 = [8.0, 3.0, 17.0, 12.0, 23.0]
    fd(f, θ, i; h = 1.0e-6) = begin
        θp = copy(θ)
        θm = copy(θ)
        θp[i] += h * abs(θ[i])
        θm[i] -= h * abs(θ[i])
        (f(θp) - f(θm)) / (2h * abs(θ[i]))
    end
    for f in (
            θ -> maxwell_to_kelvin(mk(θ)).tau[1],
            θ -> maxwell_to_kelvin(mk(θ)).tau[2],
            θ -> maxwell_to_kelvin(mk(θ)).J[1],
            θ -> maxwell_to_kelvin(mk(θ)).J_0,
            θ -> creep(mk(θ), 3.0),
        )
        g = ForwardDiff.gradient(f, θ0)
        gfd = [fd(f, θ0, i) for i in 1:5]
        @test maximum(abs.(g .- gfd) ./ max.(abs.(gfd), 1.0e-12)) < 1.0e-6
    end

    # …including through the fluid branch of the other direction.
    mb(x) = PronyCreep(1 / x[1], [1 / x[3]], [x[4] / x[3]], 1 / x[2])
    bp = [1.0, 3.0, 2.0, 6.0]
    for f in (
            x -> relaxation(kelvin_to_maxwell(mb(x)), 1.0),
            x -> kelvin_to_maxwell(mb(x)).tau[1],
        )
        g = ForwardDiff.gradient(f, bp)
        gfd = [fd(f, bp, i) for i in 1:4]
        @test maximum(abs.(g .- gfd) ./ max.(abs.(gfd), 1.0e-12)) < 1.0e-6
    end

    # Nested duals: catches any hard-coded working type in the root lift.
    g2 = ForwardDiff.derivative(
        y -> ForwardDiff.derivative(
            x -> maxwell_to_kelvin(PronyRelaxation(8.0, [3.0, 17.0], [x, 23.0])).tau[1], y
        ), 12.0
    )
    @test isfinite(g2)
    @test g2 isa Float64
end

@testset "prony_fit — collocation onto an arbitrary transform" begin
    # Fit a Prony chain to a transform that is *not* one: a fractional Zener.
    fz = FractionalZener(2.0, 10.0, 1.0, 0.6)
    τs = exp10.(range(-2, 2; length = 12))
    fit = prony_fit_relaxation(p -> carson_relaxation(fz, p), τs)
    @test fit isa PronyRelaxation
    @test all(≥(0), fit.E)                       # non-negative by default
    for ω in (0.1, 1.0, 10.0)
        @test isapprox(
            real(complex_modulus(fit, ω)), real(complex_modulus(fz, ω)); rtol = 5.0e-2
        )
    end
    # The point of fitting: the result has a closed-form time function.
    @test relaxation(fit, 1.0) ≈ fit.E_inf + sum(fit.E .* exp.(-1.0 ./ fit.tau))

    # And the creep-side twin.
    fitJ = prony_fit_creep(p -> carson_creep(fz, p), τs)
    @test fitJ isa PronyCreep
    @test all(≥(0), fitJ.J)
    @test isapprox(creep(fitJ, 1.0), creep(fz, 1.0); rtol = 5.0e-2)
end

@testset "prony_fit — unconstrained fit is differentiable" begin
    τs = [0.1, 1.0, 10.0]
    f(E1) = prony_fit_relaxation(
        p -> carson_relaxation(zener_maxwell(2.0, E1, 1.0), p), τs; nonneg = false
    ).E_inf
    @test isfinite(ForwardDiff.derivative(f, 3.0))
end
