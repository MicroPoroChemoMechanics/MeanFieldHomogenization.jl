using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Symbolics

# =============================================================================
#  A law given by its kernel on a grid (`SampledViscoLaw`, `operator_law`), and
#  the stress-free precipitation and dissolution of a phase.
#
#  References: the kernel the operator was built from (round trip), the
#  linearity of the Voigt average (two scales against one), and the closed
#  forms of an elastic phase switched on or off — a row of a trapezoidal
#  operator sums to f(t_i, t_1), the response to a unit strain held from t_1.
# =============================================================================

const SL_TIMES = [0.0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0]
const SL_ROW = 6 * (length(SL_TIMES) - 1) + 1     # (1, 1) Mandel entry of the last block row
sl_mandel(x::AbstractTens) = mandel(canonical_components(x); symmetrize = true)
sl_mandel(x::AbstractMatrix) = x
sl_maxwell() = maxwell_iso(10.0, 6.0, 0.8, 2.5)
sl_rel(a, b) = norm(a - b) / norm(b)

@testset "operator_law recovers the kernel of a trapezoidal operator" begin
    # Order 4, blocks: the samples are the kernel the operator was built from.
    law = sl_maxwell()
    R = trapezoidal_matrix(law, SL_TIMES)
    S = operator_law(R, SL_TIMES)
    @test S isa SampledViscoLaw
    for (i, t) in enumerate(SL_TIMES), (j, tp) in enumerate(SL_TIMES)
        j ≤ i || continue
        @test sl_rel(visco_eval(S, t, tp), sl_mandel(visco_eval(law, t, tp))) < 1.0e-13
    end
    @test sl_rel(trapezoidal_matrix(S, SL_TIMES), R) < 1.0e-14
    @test S(SL_TIMES[4], SL_TIMES[2]) === visco_eval(S, SL_TIMES[4], SL_TIMES[2])
    @test visco_mode(S) === :relaxation
    # Scalar kernel, block size 1.
    f = ViscoLaw((t, tp) -> t < tp ? 0.0 : exp(-(t - tp)) * (1 + tp), :relaxation)
    Rs = trapezoidal_matrix(f, SL_TIMES)
    Ss = operator_law(Rs, SL_TIMES; block_size = 1)
    @test all(isapprox(visco_eval(Ss, t, tp), visco_eval(f, t, tp); atol = 1.0e-14) for (i, t) in enumerate(SL_TIMES) for tp in SL_TIMES[1:i])
    @test trapezoidal_matrix(Ss, SL_TIMES) ≈ Rs atol = 1.0e-14
    # Order 2, 3×3 blocks: a conductivity that relaxes.
    K = ViscoLaw((t, tp) -> t < tp ? zero(TensISO{3}(1.0)) : TensISO{3}(2.0 + exp(tp - t)), :relaxation)
    RK = trapezoidal_matrix(K, SL_TIMES)
    SK = operator_law(RK, SL_TIMES; block_size = 3)
    @test size(visco_eval(SK, SL_TIMES[3], SL_TIMES[2])) == (3, 3)
    @test visco_eval(SK, SL_TIMES[5], SL_TIMES[2]) ≈ (2.0 + exp(SL_TIMES[2] - SL_TIMES[5])) * I(3)
    @test sl_rel(trapezoidal_matrix(SK, SL_TIMES), RK) < 1.0e-14
end

@testset "a composite as a phase of the next scale" begin
    E_I = heaviside_law(TensISO{3}(30.0, 20.0))
    E_O = heaviside_law(TensISO{3}(60.0, 45.0))
    f, g = 0.3, 0.6
    inner = RVE()
    add_phase!(inner, :M, Ellipsoid(1.0), Dict(:C => sl_maxwell()); fraction = :rest)
    add_phase!(inner, :I, Ellipsoid(1.0), Dict(:C => E_I); fraction = f)
    R_in = homogenize_alv(inner, Voigt(), :C; times = SL_TIMES)
    outer = RVE()
    add_phase!(outer, :A, Ellipsoid(1.0), Dict(:C => operator_law(R_in, SL_TIMES)); fraction = g)
    add_phase!(outer, :O, Ellipsoid(1.0), Dict(:C => E_O); fraction = :rest)
    two = homogenize_alv(outer, Voigt(), :C; times = SL_TIMES)
    # The Voigt average is linear: the two scales are one, with the products of fractions.
    one_scale = RVE()
    add_phase!(one_scale, :M, Ellipsoid(1.0), Dict(:C => sl_maxwell()); fraction = g * (1 - f))
    add_phase!(one_scale, :I, Ellipsoid(1.0), Dict(:C => E_I); fraction = g * f)
    add_phase!(one_scale, :O, Ellipsoid(1.0), Dict(:C => E_O); fraction = :rest)
    @test sl_rel(two, homogenize_alv(one_scale, Voigt(), :C; times = SL_TIMES)) < 1.0e-13
    # A phase given by the operator of its own law is that law, in a Mori-Tanaka cell too.
    mt(law) = (
        r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => law); fraction = :rest);
        add_phase!(r, :I, Spheroid(4.0), Dict(:C => E_I); fraction = 0.25, symmetrize = :iso);
        homogenize_alv(r, MoriTanaka(), :C; times = SL_TIMES)
    )
    @test sl_rel(mt(operator_law(trapezoidal_matrix(sl_maxwell(), SL_TIMES), SL_TIMES)), mt(sl_maxwell())) < 1.0e-12
end

@testset "stress-free precipitation and dissolution" begin
    C = TensISO{3}(30.0, 20.0)
    law = heaviside_law(C)
    t_s = SL_TIMES[4]
    P, D = precipitated(law, t_s), dissolved(law, t_s)
    @test iszero(visco_eval(P, SL_TIMES[5], SL_TIMES[3]))      # loaded before precipitating
    @test visco_eval(P, SL_TIMES[5], SL_TIMES[4]) == C
    @test visco_eval(D, SL_TIMES[3], SL_TIMES[2]) == C          # before dissolving
    @test iszero(visco_eval(D, SL_TIMES[5], SL_TIMES[2]))
    # A unit strain held from t_1: a row of the operator sums to f(t_i, t_1).
    held(M) = [sum(M[(6(i - 1) + 1):(6i), 1:6:end]; dims = 2)[1] for i in eachindex(SL_TIMES)]
    C11 = sl_mandel(C)[1, 1]
    @test held(trapezoidal_matrix(P, SL_TIMES)) ≈ zeros(length(SL_TIMES)) atol = 1.0e-14
    @test held(trapezoidal_matrix(D, SL_TIMES)) ≈ [t < t_s ? C11 : 0.0 for t in SL_TIMES]
    # On a sampled law as well, which is how a scale below precipitates.
    S = operator_law(trapezoidal_matrix(law, SL_TIMES), SL_TIMES)
    @test sl_rel(trapezoidal_matrix(precipitated(S, t_s), SL_TIMES), trapezoidal_matrix(P, SL_TIMES)) < 1.0e-14
end

@testset "misuse is refused with a message" begin
    S = operator_law(trapezoidal_matrix(sl_maxwell(), SL_TIMES), SL_TIMES)
    @test_throws "not an instant of its grid" visco_eval(S, 0.2, 0.0)
    @test_throws DimensionMismatch operator_law(zeros(12, 12), SL_TIMES)
    @test_throws DimensionMismatch SampledViscoLaw(SL_TIMES, zeros(2, 2))
    @test_throws "mode must be one of" SampledViscoLaw(SL_TIMES, zeros(7, 7); mode = :other)
    creep = ViscoLaw((t, tp) -> 1.0, :creep)
    @test_throws "relaxation kernel" precipitated(creep, 1.0)
    @test_throws "relaxation kernel" dissolved(creep, 1.0)
end

@testset "derivatives and symbols through a nested scale" begin
    # d/dg of a two-scale Voigt estimate, against the one-scale estimate it equals.
    E_O = heaviside_law(TensISO{3}(60.0, 45.0))
    function two_scales(k)
        inner = RVE()
        add_phase!(inner, :M, Ellipsoid(1.0), Dict(:C => maxwell_iso(k, 6.0, 0.8, 2.5)); fraction = :rest)
        add_phase!(inner, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(30.0, 20.0))); fraction = 0.3)
        R_in = homogenize_alv(inner, Voigt(), :C; times = SL_TIMES)
        outer = RVE()
        add_phase!(outer, :A, Ellipsoid(1.0), Dict(:C => operator_law(R_in, SL_TIMES)); fraction = 0.6)
        add_phase!(outer, :O, Ellipsoid(1.0), Dict(:C => E_O); fraction = :rest)
        return homogenize_alv(outer, Voigt(), :C; times = SL_TIMES)[SL_ROW, 1]
    end
    # The Voigt estimate is linear in the phases: its derivative is 0.6 × 0.7 times that of
    # the Maxwell operator alone, differentiated without any nesting.
    d_ref = ForwardDiff.derivative(k -> trapezoidal_matrix(maxwell_iso(k, 6.0, 0.8, 2.5), SL_TIMES)[SL_ROW, 1], 10.0)
    @test ForwardDiff.derivative(two_scales, 10.0) ≈ 0.6 * 0.7 * d_ref rtol = 1.0e-12
    @test !iszero(d_ref)
    # A symbolic kernel: sums and halvings only, so the kernel comes back exactly.
    Symbolics.@variables a
    f = ViscoLaw((t, tp) -> t < tp ? zero(a) : a * (1 + t) / (1 + tp), :relaxation)
    S = operator_law(trapezoidal_matrix(f, SL_TIMES), SL_TIMES; block_size = 1)
    v = Symbolics.value(Symbolics.substitute(visco_eval(S, SL_TIMES[6], SL_TIMES[3]), Dict(a => 2.0)))
    @test v ≈ 2.0 * (1 + SL_TIMES[6]) / (1 + SL_TIMES[3]) rtol = 1.0e-14
end
