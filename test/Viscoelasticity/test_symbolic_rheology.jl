using Test
using MeanFieldHomogenization
using TensND
using Symbolics
using SymPy

# =============================================================================
#  test_symbolic_rheology.jl — symbolic parameters, and symbolic inversion.
#
#  Two independent capabilities:
#
#    * the model constructors validate their arguments only when those are
#      hard numeric, so a symbolic `τ` no longer trips `0 < τ`.  `Num <: Real`
#      is famously *not* the predicate for that, which is why the guard is
#      `Elliptic.is_hard_numeric` and not a type bound;
#    * with SymPy loaded, `inverse_laplace` / `inverse_carson` at a symbolic
#      time return the closed form instead of a number.
# =============================================================================

@testset "Symbolics: models take symbolic parameters" begin
    Symbolics.@variables E∞ E₁ τ₁ p t kk

    m = zener_maxwell(E∞, E₁, τ₁)
    @test m isa PronyRelaxation

    # The transform is an exact expression, and the reciprocity identity holds
    # symbolically — which is the strongest statement available here.
    @test isequal(
        Symbolics.simplify(carson_creep(m, p) * carson_relaxation(m, p) - 1), 0
    )
    # The closed-form time function too.
    @test isequal(
        Symbolics.simplify(relaxation(m, t) - (E∞ + E₁ * exp(-t / τ₁))), 0
    )

    # Fractional and bituminous models, whose constructors carry the tightest
    # validations, build symbolically as well.
    @test HuetSayegh(E∞, E₁, 2.5, τ₁, 0.2, 0.65) isa HuetSayegh
    @test Model2S2P1D(E∞, E₁, 2.2, τ₁, 0.22, 0.63, 50.0) isa Model2S2P1D
    @test ScottBlair(E₁, 0.4) isa ScottBlair
    @test FractionalZener(E∞, E₁, τ₁, 0.6) isa FractionalZener

    # …and the lift to a fourth-order tensor stays symbolic.
    C = carson_relaxation(iso_rheology(Spring(kk), m), p)
    @test C isa TensISO
    @test isequal(Symbolics.simplify(get_data(C)[1] - 3kk), 0)
end

@testset "Symbolics: a numeric model still accepts a symbolic p" begin
    Symbolics.@variables p
    m = Model2S2P1D(1.0e-7, 1000.0, 2.2, 1.945e-3, 0.22, 0.63, 50.0)
    @test carson_relaxation(m, p) isa Num
end

@testset "the conversion refuses a symbolic spectrum, and says why" begin
    Symbolics.@variables E∞ E₁ τ₁ p
    m = zener_maxwell(E∞, E₁, τ₁)
    err = try
        maxwell_to_kelvin(m)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bisection", err.msg)
    @test occursin("symbolic", err.msg)

    # But the derived transform falls back on `J* = 1/R*`, which *is* symbolic,
    # rather than propagating the refusal.
    @test isequal(
        Symbolics.simplify(carson_creep(m, p) - 1 / carson_relaxation(m, p)), 0
    )
end

@testset "SymPy: symbolic Laplace-Carson inversion" begin
    SymPy.@syms ts::positive τ::positive Einf::positive E1::positive

    # The plain transform.  `inverse_laplace` is qualified here because
    # `Symbolics` exports a function of the same name (a five-argument
    # transform for solving ODEs), so the two bindings are ambiguous once both
    # packages are `using`-ed — see the note on `inverse_laplace`.
    ilt = MeanFieldHomogenization.inverse_laplace
    @test SymPy.simplify(ilt(p -> E1 / (p + 1 / τ), ts) - E1 * exp(-ts / τ)) == 0

    # The standard solid: both functions of time, recovered from R*(p) alone
    # and agreeing with the closed forms the model already carries.
    z = zener_maxwell(Einf, E1, τ)
    R_t = inverse_carson(p -> carson_relaxation(z, p), ts)
    @test SymPy.simplify(R_t - relaxation(z, ts)) == 0

    # Burgers, a fluid: the creep function carries a term linear in t.
    SymPy.@syms ks::positive ηs::positive kp::positive ηp::positive
    b = burgers(ks, ηs, kp, ηp)
    J_t = inverse_carson(p -> carson_creep(b, p), ts)
    @test SymPy.simplify(J_t - creep(b, ts)) == 0
end

@testset "SymPy: how far the symbolic route reaches" begin
    SymPy.@syms ts::positive V::positive α::positive τ::positive
    SymPy.@syms Einf::positive E0::positive δ::positive

    # A rational exponent inverts exactly, and matches the model's own form —
    # `5/(2Γ(2/5)) = 1/Γ(7/5)` is the same number written two ways.
    sb = ScottBlair(V, Sym(2) // 5)
    @test SymPy.simplify(
        inverse_carson(q -> carson_creep(sb, q), ts) - creep(sb, ts)
    ) == 0

    # The Cole-Cole model at α = 1/2 has a genuine closed form, `E_{1/2}(-x) =
    # e^{x²}erfc(x)`, and SymPy finds it.
    fz = FractionalZener(Einf, E0, τ, Sym(1) // 2)
    r = inverse_carson(q -> carson_relaxation(fz, q), ts)
    @test occursin("erfc", string(r))

    # A *symbolic* exponent is where it stops, and it stops badly: SymPy
    # returns `nan`, so the extension warns rather than let it propagate.
    bad = @test_logs (:warn,) match_mode = :any inverse_carson(
        q -> carson_creep(ScottBlair(V, α), q), ts
    )
    @test occursin("nan", lowercase(string(bad)))

    # A sum of several fractional powers is left unevaluated — honest, and
    # also warned about.
    hs = HuetSayegh(Einf, E0, δ, τ, Sym(1) // 5, Sym(2) // 3)
    un = @test_logs (:warn,) match_mode = :any inverse_carson(
        q -> carson_relaxation(hs, q), ts
    )
    @test occursin("InverseLaplaceTransform", string(un))
end

@testset "symbolic and numerical inversion agree" begin
    SymPy.@syms ts::positive τ::positive Einf::positive E1::positive
    z = zener_maxwell(Einf, E1, τ)
    R_t = inverse_carson(p -> carson_relaxation(z, p), ts)

    zn = zener_maxwell(2.0, 3.0, 1.5)
    for tv in (0.2, 1.0, 5.0)
        sym = Float64(R_t.subs(Dict(Einf => 2, E1 => 3, τ => Sym(3) // 2, ts => Sym(tv))))
        num = inverse_carson(p -> carson_relaxation(zn, p), tv)
        @test isapprox(sym, num; rtol = 1.0e-10)
    end
end
