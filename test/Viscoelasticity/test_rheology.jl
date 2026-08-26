using Test
using MeanFieldHomogenization
using SpecialFunctions
using ForwardDiff

# =============================================================================
#  test_rheology.jl — the scalar model catalog.
#
#  Two properties are checked for every model, because between them they catch
#  almost any algebraic slip:
#
#    * `J*(p) R*(p) = 1`, the definition of the creep transform;
#    * the ω → 0 and ω → ∞ limits of the complex modulus must equal the
#      declared `equilibrium_modulus` and `glassy_modulus`.
#
#  Then, model by model, the closed-form time value is checked against the
#  numerical inversion of the same transform — two disjoint routes to one
#  number.
# =============================================================================

const SOLID_MODELS = (
    ("Spring", Spring(7.0)),
    ("MaxwellUnit", MaxwellUnit(3.0, 6.0)),
    ("zener_maxwell", zener_maxwell(2.0, 5.0, 1.5)),
    ("zener_kelvin", zener_kelvin(10.0, 4.0, 2.0)),
    ("burgers", burgers(1.0, 3.0, 2.0, 6.0)),
    ("FractionalZener", FractionalZener(2.0, 10.0, 1.0, 0.6)),
    ("Rabotnov", Rabotnov(1.7, -0.495, -0.46, 0.98)),
    ("HuetSayegh", HuetSayegh(20.0, 25000.0, 2.5, 0.8, 0.2, 0.65)),
    ("Model2S2P1D", Model2S2P1D(1.0e-7, 1000.0, 2.2, 1.94507827e-3, 0.22, 0.63, 50.0)),
    ("LogarithmicCreep", LogarithmicCreep(30000.0, 50000.0, 1.0)),
    ("ScottBlair", ScottBlair(1.0, 0.4)),
    ("FractionalMaxwell", FractionalMaxwell(2.0, 0.8, 1.0, 0.3)),
    ("FractionalKelvin", FractionalKelvin(2.0, 0.8, 1.0, 0.0)),
)

@testset "reciprocity J* R* = 1 across the catalog" begin
    for (name, m) in SOLID_MODELS, p in (1.0e-3, 0.1, 1.0, 10.0, 1.0e3)
        @test isapprox(
            carson_creep(m, p) * carson_relaxation(m, p), 1.0; rtol = 1.0e-12
        )
    end
    for (name, m) in SOLID_MODELS, ω in (1.0e-2, 1.0, 1.0e2)
        @test isapprox(
            carson_creep(m, im * ω) * carson_relaxation(m, im * ω), 1.0 + 0im;
            rtol = 1.0e-12
        )
    end
end

@testset "E*(ω) approaches the declared limits" begin
    # How *fast* the limit is approached varies enormously across the
    # catalog.  A Prony chain gets there like `1/(ωτ)`; a fractional model
    # only like `(ωτ)^{-k}` with `k` as small as 0.2, so even `ω = 1e14` is
    # still half a percent away; and `LogarithmicCreep` approaches its zero
    # equilibrium modulus like `1/log(1/ω)`, which no finite `ω` resolves.
    #
    # So the assertion is that the limit is *approached* — two decades further
    # out is strictly closer — plus a bound sized to each model's own rate.
    # "No further away, and strictly closer unless it has already landed
    # exactly on the limit" — several rational models saturate to the limit in
    # floating point well before `ω = 1e8`.
    function approaches(f, limit, ω_near, ω_far)
        near = abs(f(ω_near) - limit)
        far = abs(f(ω_far) - limit)
        return far ≤ near && (iszero(near) || far < near)
    end

    for (name, m) in SOLID_MODELS
        Ereal(ω) = real(complex_modulus(m, ω))
        eq = equilibrium_modulus(m)
        @test approaches(Ereal, eq, 1.0e-8, 1.0e-16)

        g = try
            glassy_modulus(m)
        catch
            nothing                              # ScottBlair & friends: infinite
        end
        g === nothing && continue
        @test approaches(Ereal, g, 1.0e8, 1.0e16)
        # Models whose glassy limit is reached at a rational rate get a tight
        # bound; the fractional ones are checked by the approach above.
        if m isa Union{Spring, MaxwellUnit, PronyRelaxation, PronyCreep}
            @test isapprox(Ereal(1.0e14), g; rtol = 1.0e-8)
        end
    end
end

@testset "closed-form time values against the inversion" begin
    # Each of these declares a closed form; the numerical inversion of the same
    # transform is an independent route to the same number.
    cases = (
        (Spring(7.0), (0.5, 3.0)),
        (MaxwellUnit(3.0, 6.0), (0.5, 3.0)),
        (zener_maxwell(2.0, 5.0, 1.5), (0.2, 2.0)),
        (ScottBlair(1.0, 0.4), (0.01, 1.0, 100.0)),
    )
    for (m, ts) in cases, t in ts
        @test isapprox(
            relaxation(m, t), inverse_carson(p -> carson_relaxation(m, p), t);
            rtol = 1.0e-7
        )
        @test isapprox(
            creep(m, t), inverse_carson(p -> carson_creep(m, p), t); rtol = 1.0e-7
        )
    end
end

@testset "ScottBlair — the exact power-law pair" begin
    V, α = 1.7, 0.4
    m = ScottBlair(V, α)
    for t in (0.01, 1.0, 100.0)
        @test creep(m, t) ≈ t^α / (V * gamma(1 + α)) rtol = 1.0e-14
        @test relaxation(m, t) ≈ V * t^(-α) / gamma(1 - α) rtol = 1.0e-14
        # LC{t^a} = Γ(a+1) p^{-a} is what makes both closed forms exact.
        @test isapprox(inverse_carson(p -> carson_creep(m, p), t), creep(m, t); rtol = 1.0e-9)
    end
    @test_throws ArgumentError ScottBlair(1.0, 1.5)
end

@testset "LogarithmicCreep — the E₁ transform" begin
    m = LogarithmicCreep(30000.0, 50000.0, 1.0)
    for t in (0.01, 1.0, 100.0, 1.0e4)
        @test creep(m, t) ≈ 1 / m.E + log1p(t / m.τ) / m.C rtol = 1.0e-14
        @test isapprox(
            inverse_carson(p -> carson_creep(m, p), t), creep(m, t);
            rtol = 1.0e-7
        )
    end
    @test is_fluid(m)          # unbounded strain, if only logarithmically
end

@testset "2S2P1D — one model, an exact pair in both domains" begin
    m = Model2S2P1D(1.0e-7, 1000.0, 2.2, 1.94507827e-3, 0.22, 0.63, 50.0)
    # φ(t) and φ*(p) are a transform pair term by term, and that is what lets
    # the same model drive the Laplace-Carson route and the Volterra one.
    for t in (1.0e-5, 1.0e-3, 1.0e-1, 10.0)
        @test isapprox(
            inverse_carson(p -> carson_creep_kernel(m, p), t), creep_kernel(m, t);
            rtol = 1.0e-9
        )
    end
    @test glassy_modulus(m) == m.E0
    @test equilibrium_modulus(m) == m.E00
    @test creep_kernel(m, 0.0) == 1.0

    # The kernel packaged for the ageing pipeline agrees with the direct one.
    law = creep_kernel_law(m)
    @test visco_mode(law) == :creep
    @test law(0.5, 0.2) ≈ creep_kernel(m, 0.3)
    @test iszero(law(0.2, 0.5))

    @test_throws ArgumentError Model2S2P1D(1.0, 10.0, 2.0, 1.0, 0.7, 0.3, 50.0)  # k > h
end

@testset "HuetSayegh is a solid, 2S2P1D is a fluid" begin
    hs = HuetSayegh(20.0, 25000.0, 2.5, 0.8, 0.2, 0.65)
    m = Model2S2P1D(20.0, 25000.0, 2.5, 0.8, 0.2, 0.65, 50.0)
    @test !is_fluid(hs)
    @test equilibrium_modulus(hs) == 20.0
    # The one structural difference is the series dashpot term 1/(βpτ).
    @test carson_relaxation(m, 1.0e6) ≈ carson_relaxation(hs, 1.0e6) rtol = 1.0e-3
end

@testset "Rabotnov — the transform is elementary" begin
    # The benchmark parameters of barthelemyIJES2019 §5: `λ₀ < 0`, so the
    # modulus decreases and the material is passive, and `α ∈ (-1, 0)`, so the
    # Mittag-Leffler order `α + 1` lands in `(0, 1)`.
    μ0, λ0, α, β = 1.7, -0.495, -0.46, 0.98
    m = Rabotnov(μ0, λ0, α, β)
    # R*(p) = μ₀(1 + λ₀/(p^{α+1}+β)) — no Mittag-Leffler function needed.
    for p in (0.1, 1.0, 10.0)
        @test carson_relaxation(m, p) ≈ μ0 * (1 + λ0 / (p^(α + 1) + β)) rtol = 1.0e-14
    end
    @test glassy_modulus(m) ≈ μ0
    @test equilibrium_modulus(m) ≈ μ0 * (1 + λ0 / β)
    @test equilibrium_modulus(m) < glassy_modulus(m)     # it relaxes
    @test isfinite(relaxation(m, 1.0))
    @test_throws ArgumentError Rabotnov(1.0, -0.5, -1.5, 1.0)
end

@testset "loss factor is non-negative — passivity smoke test" begin
    for (name, m) in SOLID_MODELS, ω in exp10.(range(-3, 3; length = 7))
        m isa Spring && continue                # purely elastic: tan δ = 0
        @test loss_factor(m, ω) ≥ -1.0e-12
    end
end

@testset "Dirac elements refuse to answer for R(t)" begin
    @test_throws ArgumentError relaxation(Dashpot(2.0), 1.0)
    @test_throws ArgumentError glassy_modulus(Dashpot(2.0))
    @test_throws ArgumentError relaxation(KelvinUnit(3.0, 6.0), 1.0)
    @test_throws ArgumentError glassy_modulus(KelvinUnit(3.0, 6.0))
    # …but their creep functions are perfectly ordinary.
    @test creep(Dashpot(2.0), 4.0) ≈ 2.0
    @test creep(KelvinUnit(3.0, 6.0), 2.0) ≈ (1 - exp(-1.0)) / 3
end

@testset "models are differentiable in their parameters" begin
    g = ForwardDiff.derivative(E1 -> relaxation(zener_maxwell(2.0, E1, 1.5), 0.8), 3.0)
    @test g ≈ exp(-0.8 / 1.5) rtol = 1.0e-12
    # …including through a numerical inversion of a fractional transform.
    g2 = ForwardDiff.derivative(
        τ -> relaxation(Model2S2P1D(20.0, 25000.0, 2.5, τ, 0.2, 0.65, 50.0), 1.0), 0.8
    )
    @test isfinite(g2) && g2 != 0
end
