using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Symbolics
using JSON

# =============================================================================
#  Jumps on a grid (an instant given twice) and operators, laws and histories
#  read as functions of time.
#
#  References: closed forms of a held load and of a phase switched on or off at
#  the doubled instant; an exact identity of the trapezoidal rule — the operator
#  applied to a history equals the Stieltjes integral of the interpolated
#  kernel against the interpolated history, computed by the midpoint rule
#  (exact for a product of two linear functions) plus the jumps; kernels that
#  are linear or bilinear, which the interpolation reproduces exactly; analytic
#  derivatives for ForwardDiff; Echoes on the same grids (`echoes_doubled_instants.json`,
#  written by `scripts/bench_echoes/alv_harness/make_doubled_instants_reference.py`).
# =============================================================================

const JE_TIMES = [0.0, 0.3, 1.0, 1.0, 2.0, 5.0, 9.0]          # 1.0 twice: a jump
const JE_A = 1.0
je_f(t, tp) = t < tp ? 0.0 : exp(-(t - tp)) * (1 + tp)
je_scalar() = ViscoLaw(je_f, :relaxation)
je_iso() = ViscoLaw((t, tp) -> t < tp ? zero(TensISO{3}(1.0, 1.0)) : TensISO{3}(3 * (2 + exp(tp - t)), 2 * (1 + exp((tp - t) / 2)) * (1 + 0.1tp)), :relaxation)
je_step(a) = t -> t ≥ a ? 1.0 : 0.0
je_unwrap(R) = MeanFieldHomogenization.Viscoelasticity._without_law(R)

# The Stieltjes integral ∫ R(tᵢ, τ) dx(τ) of the interpolated kernel against the interpolated
# history: the midpoint rule on each interval (exact: both are linear there) and R(tᵢ, t⁺) Δx
# at a jump (the first instant, a doubled one).
function je_stieltjes(R, h, times, i, comp)
    y = R(times[i], times[1]) * h(times[1])[comp]
    for k in 2:i
        Δx = h.values[comp, k] - h.values[comp, k - 1]
        τ = times[k] == times[k - 1] ? times[k] : (times[k - 1] + times[k]) / 2
        y += R(times[i], τ) * Δx
    end
    return y
end

@testset "doubled instants: grids and the weights" begin
    @test_throws "must increase" TimeGrid([0.0, 2.0, 1.0])
    @test_throws "must increase" trapezoidal_matrix(je_scalar(), [0.0, 2.0, 1.0])
    # An instant given three times: two jumps at once, which add up.
    t3 = [0.0, 1.0, 1.0, 1.0, 4.0]
    y3 = vec((VolterraOperator(je_scalar(), t3) * TimeHistory([0.0, 0.0, 1.0, 2.0, 2.0], t3)).values)
    @test y3 ≈ [0.0, 0.0, je_f(1.0, 1.0), 2 * je_f(1.0, 1.0), 2 * je_f(4.0, 1.0)] rtol = 1.0e-14
    g = TimeGrid(JE_TIMES)
    @test length(g) == 7
    # Without a doubled instant, the classical trapezoidal weights to the last bit.
    t = [0.0, 0.3, 1.0, 2.0, 5.0]
    M = trapezoidal_matrix(je_scalar(), t)
    @test M[1, 1] == je_f(t[1], t[1])
    @test M[4, 4] == (je_f(t[4], t[3]) + je_f(t[4], t[4])) / 2
    @test M[4, 1] == (je_f(t[4], t[1]) - je_f(t[4], t[2])) / 2
    @test M[4, 2] == (je_f(t[4], t[1]) - je_f(t[4], t[3])) / 2
    # The dense and the block constructions agree on a doubled grid.
    @test Matrix(VolterraOperator(je_iso(), JE_TIMES)) ≈ trapezoidal_matrix(je_iso(), JE_TIMES) rtol = 1.0e-14
end

@testset "doubled instants: a load held from t′ is a jump" begin
    R = VolterraOperator(je_scalar(), JE_TIMES)
    x = TimeHistory(je_step(JE_A), JE_TIMES)
    # The function is read just before 1 on the first node: 0 there, 1 on the second.
    @test vec(x.values) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # The response is the kernel itself, f(t, 1), from 1⁺ on, and nothing before.
    y = vec((R * x).values)
    @test y[1:3] == zeros(3)
    @test y[4:end] ≈ [je_f(t, JE_A) for t in JE_TIMES[4:end]] rtol = 1.0e-14
    # The same load on a grid without the doubled instant rises over [0.3, 1]: at 1, the mean
    # of the kernel over that ramp.
    t1 = unique(JE_TIMES)
    @test vec((VolterraOperator(je_scalar(), t1) * TimeHistory(je_step(JE_A), t1)).values)[3] ≈ (je_f(JE_A, 0.3) + je_f(JE_A, JE_A)) / 2
    # A doubled first instant: a jump from zero to 0.5 at 0⁻, then from 0.5 to 1 at 0⁺; the
    # response at 0⁻ is half the kernel, and the full one afterwards.
    t0 = [0.0, 0.0, 1.0, 3.0]
    y0 = vec((VolterraOperator(je_scalar(), t0) * TimeHistory([0.5, 1.0, 1.0, 1.0], t0)).values)
    @test y0 ≈ [0.5 * je_f(0.0, 0.0), je_f(0.0, 0.0), je_f(1.0, 0.0), je_f(3.0, 0.0)] rtol = 1.0e-14
end

@testset "doubled instants: precipitation and dissolution" begin
    law = je_scalar()
    P = VolterraOperator(precipitated(law, JE_A), JE_TIMES)
    # A strain jump at the instant of precipitation loads the phase (H(0) = 1) …
    @test vec((P * TimeHistory(je_step(JE_A), JE_TIMES)).values) ≈ [t < JE_A || i == 3 ? 0.0 : je_f(t, JE_A) for (i, t) in enumerate(JE_TIMES)] atol = 1.0e-15
    # … a strain applied before it does not.
    ramp = TimeHistory(t -> min(t, JE_A), JE_TIMES)
    @test vec((P * ramp).values) ≈ zeros(7) atol = 1.0e-15
    # Switched on the operator, node by node, as on the law.
    R = VolterraOperator(law, JE_TIMES)
    @test Matrix(precipitated(R, JE_A)) ≈ Matrix(P) atol = 1.0e-15
    # Dissolved at 1: carries stress at 1⁻, none at 1⁺.
    D = dissolved(R, JE_A)
    held = vec((D * TimeHistory(t -> 1.0, JE_TIMES)).values)
    @test held ≈ [i ≤ 3 ? je_f(t, 0.0) : 0.0 for (i, t) in enumerate(JE_TIMES)] rtol = 1.0e-14
    @test Matrix(D) ≈ Matrix(VolterraOperator(dissolved(law, JE_A), JE_TIMES)) atol = 1.0e-15
end

@testset "operators as functions: the kernel the matrix integrates" begin
    R = VolterraOperator(je_iso(), JE_TIMES)
    # Built from a law: the law, and zero for t′ > t.
    @test R(4.0, 0.5) == je_iso()(4.0, 0.5)
    @test iszero(R(1.0, 2.0))
    @test R isa VolterraOperator && !isnothing(R.law)
    # Computed from others: its kernel on the grid, exact at the instants …
    Rf = je_unwrap(R)
    for (i, t) in enumerate(JE_TIMES), (j, tp) in enumerate(JE_TIMES)
        j ≤ i || continue
        @test norm(Rf(t, tp) - je_iso()(t, tp)) ≤ 1.0e-13 * norm(je_iso()(t, tp))
    end
    @test iszero(Rf(1.0, 2.0))
    # … and between them the one the trapezoidal rule integrates: the response of the creep
    # operator J = R⁻¹ to a history with a jump is the Stieltjes integral of its interpolated
    # kernel against the interpolated history.
    J = inv(VolterraOperator(je_scalar(), JE_TIMES))
    h = TimeHistory([0.2, 0.5, 0.5, 1.5, 1.0, 1.2, 0.7], JE_TIMES)
    y = vec((J * h).values)
    @test all(isapprox(je_stieltjes(J, h, JE_TIMES, i, 1), y[i]; rtol = 1.0e-12) for i in eachindex(JE_TIMES))
    # A kernel linear in (t, t′) is interpolated exactly, a bilinear one below the diagonal.
    lin(t, tp) = 2.0 + 0.5t - 0.3tp
    bil(t, tp) = lin(t, tp) + 0.1t * tp
    L = je_unwrap(VolterraOperator(ViscoLaw((t, tp) -> t < tp ? 0.0 : lin(t, tp)), JE_TIMES))
    B = je_unwrap(VolterraOperator(ViscoLaw((t, tp) -> t < tp ? 0.0 : bil(t, tp)), JE_TIMES))
    for (t, tp) in ((0.2, 0.1), (1.5, 1.2), (3.0, 0.5), (7.0, 2.5), (9.0, 9.0), (1.0, 0.65))
        @test L(t, tp) ≈ lin(t, tp) rtol = 1.0e-14
    end
    @test B(3.0, 0.5) ≈ bil(3.0, 0.5) rtol = 1.0e-14
    @test B(7.0, 1.5) ≈ bil(7.0, 1.5) rtol = 1.0e-14
    # The doubled instant is read on its right: a kernel that jumps there.
    S = je_unwrap(VolterraOperator(precipitated(je_scalar(), JE_A), JE_TIMES))
    @test S(5.0, JE_A) ≈ je_f(5.0, JE_A) rtol = 1.0e-13
    @test S(5.0, prevfloat(JE_A)) ≈ 0.0 atol = 1.0e-12
    # Many evaluations: the law of the samples once, the same values.
    law = operator_law(Rf)
    @test law(3.0, 0.5) ≈ Rf(3.0, 0.5) rtol = 1.0e-14
    @test operator_law(J)(3.0, 0.5) ≈ J(3.0, 0.5) rtol = 1.0e-14
    # A phase that precipitates after the grid is zero all along, and so is its kernel.
    Z = VolterraOperator(precipitated(je_scalar(), 100.0), JE_TIMES)
    @test iszero(Matrix(Z)) && iszero(je_unwrap(Z)(5.0, 1.0)) && iszero(Z(5.0, 1.0))
    # Off the grid, nothing is extrapolated.
    @test_throws DomainError Rf(10.0, 1.0)
    @test_throws DomainError Rf(2.0, -1.0)
end

@testset "histories as functions" begin
    ε(t) = TensND.Tens(Symmetric([1.0e-3t 0.0 0.0; 0.0 -3.0e-4t 0.0; 0.0 0.0 -3.0e-4t]))
    h = TimeHistory(ε, JE_TIMES)
    @test h(0.65) ≈ mandel(canonical_components(ε(0.65)); symmetrize = true)
    # Given by its values: linear between the instants, right-continuous at the jump.
    x = TimeHistory([0.0, 1.0, 2.0, 5.0, 6.0, 6.0, 0.0], JE_TIMES)
    @test x(0.15) ≈ [0.5]
    @test x(JE_A) == [5.0] && x(prevfloat(JE_A)) ≈ [2.0]
    @test x(9.0) == [0.0] && x(7.0) ≈ [3.0]
    @test_throws DomainError x(9.5)
    @test_throws "cannot do" x(first(Symbolics.@variables s))
end

@testset "jumps and evaluation: derivatives and symbols" begin
    # The derivative in t′ of a linear kernel read between the instants.
    L = je_unwrap(VolterraOperator(ViscoLaw((t, tp) -> t < tp ? 0.0 : 2.0 + 0.5t - 0.3tp), JE_TIMES))
    @test ForwardDiff.derivative(tp -> L(6.0, tp), 3.0) ≈ -0.3 rtol = 1.0e-13
    @test ForwardDiff.derivative(t -> L(t, 3.0), 6.0) ≈ 0.5 rtol = 1.0e-13
    # Through a parameter of the law: the interpolated derivative of a kernel linear in p.
    dk(p) = je_unwrap(VolterraOperator(ViscoLaw((t, tp) -> t < tp ? zero(p) : p * (1 + t) - p^2 * tp), JE_TIMES))(4.0, 1.5)
    @test ForwardDiff.derivative(dk, 2.0) ≈ (1 + 4.0) - 2 * 2.0 * 1.5 rtol = 1.0e-13
    # A history read through ForwardDiff: the slope of its interpolation.
    x = TimeHistory([0.0, 1.0, 2.0, 5.0, 6.0, 6.0, 0.0], JE_TIMES)
    @test ForwardDiff.derivative(t -> x(t)[1], 7.0) ≈ -6 / 4
    # The instant of a jump as a dual: the held load answers f(t, a), whose derivative in a is
    # ∂f/∂t′.
    function held(a)
        times = [zero(a), 0.3 + zero(a), a, a, 2.0 + zero(a), 5.0 + zero(a)]
        R = VolterraOperator(je_scalar(), times)
        return (R * TimeHistory(je_step(a), times)).values[1, 5]
    end
    @test ForwardDiff.derivative(held, 1.0) ≈ exp(-1.0) * (1 + 1.0) + exp(-1.0) rtol = 1.0e-13
    # A law-backed operator takes symbols; one read on the grid refuses them.
    Symbolics.@variables t s
    R = VolterraOperator(ViscoLaw((t, tp) -> exp(-(t - tp)) * (1 + tp)), JE_TIMES)   # no comparison
    v = Symbolics.substitute(R(t, s), Dict(t => 4.0, s => 1.5); fold = Val(true))
    @test Symbolics.value(v) ≈ je_f(4.0, 1.5)
    @test_throws "cannot do" je_unwrap(R)(t, 1.0)
    # Inference of the evaluations, for numbers and duals.
    Rf = je_unwrap(VolterraOperator(je_iso(), JE_TIMES))
    @inferred Rf(3.0, 0.5)
    @inferred VolterraOperator(je_iso(), JE_TIMES)(3.0, 0.5)
    @inferred Rf(ForwardDiff.Dual(3.0, 1.0), 0.5)
    @inferred x(0.5)
    @inferred operator_law(Rf)(3.0, 0.5)
end

@testset "doubled instants against Echoes" begin
    ref = JSON.parsefile(joinpath(@__DIR__, "echoes_doubled_instants.json"))["cases"]
    rows(x) = permutedims(reduce(hcat, [Float64.(r) for r in x]))
    rel(a, b) = norm(a - b) / norm(b)
    k, μ, η, γ = 10.0, 6.0, 3.0, 1.0
    aging = ViscoLaw((t, tp) -> t < tp ? zero(TensISO{3}(1.0, 1.0)) : TensISO{3}(3k * (1 + 0.5exp(-tp)) * exp(-(t - tp) / η), 2μ * exp(-(t - tp) / γ)), :relaxation)
    Ci = TensISO{3}(3 * 30.0, 2 * 20.0)
    # Echoes reads both nodes of a doubled instant at the instant itself: a switch at 1 is
    # read there as MFH reads a switch placed just before 1.
    step = precipitated(heaviside_law(Ci), prevfloat(1.0))
    for (key, r) in ref
        times = Float64.(r["times"])
        phase = endswith(key, "aging") ? aging : step
        α, β = iso_params_from_blocks(trapezoidal_matrix(phase, times))
        @test rel(α, rows(r["alpha"])) < 1.0e-14 && rel(β, rows(r["beta"])) < 1.0e-14
        rve = RVE()
        add_phase!(rve, :MAT, Ellipsoid(1.0), Dict(:C => aging); fraction = :rest)
        add_phase!(rve, :INC, Ellipsoid(1.0), Dict(:C => endswith(key, "aging") ? Ci : step); fraction = 0.3)
        αh, βh = iso_params_from_blocks(Matrix(homogenize_alv(rve, MoriTanaka(), :C; times)))
        @test rel(αh, rows(r["alpha_MT"])) < 1.0e-14 && rel(βh, rows(r["beta_MT"])) < 1.0e-14
    end
end
