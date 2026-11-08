# =============================================================================
#  test_brinkman_symbolic.jl — Brinkman inclusions with symbolic numbers.
#
#  The concentration tensors have no value-dependent branch apart from exact
#  degenerate cases, so they accept SymPy and Symbolics numbers: the dilute and
#  Mori–Tanaka estimates of a bed of impervious spheres come out in closed form
#  (Bignonnet 2022, eqs. 55–56), and every symbolic tensor evaluates to the
#  numerical one. The self-consistent and differential schemes refuse symbols
#  (iterative solves), as for every inclusion.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
import SymPyPythonCall
import Symbolics

const BRS = MeanFieldHomogenization.Brinkman
_brs_iso(x) = TensISO{3}(x)
_brs_val(t) = first(TensND.get_data(t))
_brs_num(x) = Float64(SymPyPythonCall.N(x))

@testset "Brinkman — symbolic dilute and Mori–Tanaka estimates" begin
    R, f = SymPyPythonCall.symbols("R f", positive = true)
    z = _brs_iso(SymPyPythonCall.Sym(0))
    rve = RVE(T = SymPyPythonCall.Sym)
    add_phase!(rve, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => z); fraction = :rest)
    add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => z); fraction = f)
    rdil = _brs_val(homogenize(rve, Dilute(:p), :r))
    rmt = _brs_val(homogenize(rve, MoriTanaka(:p), :r))
    @test iszero(SymPyPythonCall.simplify(rdil - 9f / (2R^2)))
    @test iszero(SymPyPythonCall.simplify(rmt - 9f / (2R^2 * (1 - f))))
    @test_throws ArgumentError homogenize(rve, SelfConsistent(init = _brs_iso(R^-2)), :r)
end

@testset "Brinkman — symbolic concentration tensors evaluate to the numerical ones" begin
    Sym = SymPyPythonCall.Sym
    R, r0, r1 = SymPyPythonCall.symbols("R r0 r1", positive = true)
    vals = (R => Sym(13) / 10, r0 => Sym(7) / 10, r1 => Sym(3) / 2)
    num(x) = _brs_num(SymPyPythonCall.subs(x, vals...))
    Rn, r0n, r1n = 1.3, 0.7, 1.5
    shapes = (BrinkmanSphere, BrinkmanCylinder, BrinkmanLayer)
    for S in shapes, m in (BrinkmanPorous(), BrinkmanFluid(), BrinkmanSolid())
        S === BrinkmanLayer && m isa BrinkmanSolid && continue   # refused: blocks the normal flow
        isym, inum = S(R, m), S(Rn, m)
        for gen in (gradient_gradient_concentration, flux_gradient_concentration)
            Ts = gen(isym, _brs_iso(r1), _brs_iso(r0))
            Tn = gen(inum, _brs_iso(r1n), _brs_iso(r0n))
            @test num.(canonical_components(Ts)) ≈ canonical_components(Tn) rtol = 1.0e-12
        end
    end
    # Permeability form and the coated sphere.
    k0, k1 = SymPyPythonCall.symbols("k0 k1", positive = true)
    valsk = (R => Sym(13) / 10, k0 => Sym(7) / 10, k1 => Sym(3) / 2)
    numk(x) = _brs_num(SymPyPythonCall.subs(x, valsk...))
    P = PermeabilityForm()
    for (isym, inum) in (
            (BrinkmanSphere(R; form = P), BrinkmanSphere(1.3; form = P)),
            (BrinkmanCylinder(R; form = P), BrinkmanCylinder(1.3; form = P)),
            (BrinkmanCylinder(R, BrinkmanSolid(); form = P), BrinkmanCylinder(1.3, BrinkmanSolid(); form = P)),
            (BrinkmanLayer(R, BrinkmanFluid(); form = P), BrinkmanLayer(1.3, BrinkmanFluid(); form = P)),
            (BrinkmanCoatedSphere(R, 2R), BrinkmanCoatedSphere(1.3, 2.6)),
            (BrinkmanCoatedSphere(R, 2R; form = P), BrinkmanCoatedSphere(1.3, 2.6; form = P)),
        )
        Ts = gradient_gradient_concentration(isym, _brs_iso(k1), _brs_iso(k0))
        Tn = gradient_gradient_concentration(inum, _brs_iso(1.5), _brs_iso(0.7))
        @test numk.(canonical_components(Ts)) ≈ canonical_components(Tn) rtol = 1.0e-12
    end
    # The Stokes sphere is a rational function of α₀ (Bignonnet, A.3a).
    a = SymPyPythonCall.symbols("a", positive = true)
    A = _brs_val(gradient_gradient_concentration(BrinkmanSphere(a, BrinkmanFluid()), _brs_iso(Sym(0)), _brs_iso(Sym(1))))
    @test iszero(SymPyPythonCall.simplify(A - 3 * (15 + 15a + 6a^2 + a^3) / (45 + 45a + 6a^2 + a^3)))
end

@testset "Brinkman — Symbolics.Num on the sphere and the layer" begin
    Symbolics.@variables R r0 r1
    sub = Dict(R => 1.3, r0 => 0.7, r1 => 1.5)
    ev(x) = Symbolics.value(Symbolics.substitute(x, sub; fold = Val(true)))
    for (isym, inum) in (
                (BrinkmanSphere(R), BrinkmanSphere(1.3)),
                (BrinkmanSphere(R, BrinkmanSolid()), BrinkmanSphere(1.3, BrinkmanSolid())),
                (BrinkmanLayer(R), BrinkmanLayer(1.3)),
            ), gen in (gradient_gradient_concentration, flux_gradient_concentration)
        Ts = gen(isym, _brs_iso(r1), _brs_iso(r0))
        Tn = gen(inum, _brs_iso(1.5), _brs_iso(0.7))
        @test ev.(canonical_components(Ts)) ≈ canonical_components(Tn) rtol = 1.0e-12
    end
    # The symbolic route of the special functions is the closed form itself.
    x = SymPyPythonCall.symbols("x", positive = true)
    @test string(BRS._k0k1(x)) == "besselk(0, x)/besselk(1, x)"
    @test string(BRS._i1i0(x)) == "besseli(1, x)/besseli(0, x)"
end
