using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff

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
const _to_mandel66 = (t -> TensND.mandel(TensND.canonical_components(t); symmetrize = true))

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
        density = 0.05, symmetrize = :iso
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

    # Without an isotropic orientation average, a crack family makes the estimate
    # transversely isotropic: refused by name, by Newton as by Picard, instead of
    # being projected silently on the isotropic class.
    aligned = RVE()
    add_phase!(aligned, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); fraction = :rest)
    add_phase!(aligned, :F, PennyCrack(1.0), Dict(:C => maxwell_iso(10.0, 4.0, 1.0, 0.5)); density = 0.05)
    @test_throws "transversely isotropic crack contribution of phase :F" _sc_newton(aligned, :C; times)
    @test_throws "self_consistent_alv: the transversely isotropic crack" homogenize_alv(aligned, SelfConsistent(), :C; times)
end

@testset "sc_alv_newton — input errors" begin
    # An elastic tensor is an ALV property: its kernel is C H(t - t′), as with
    # `heaviside_law(C)`, for the matrix and for a phase.
    elastic(Cm, Ci) = (
        r = RVE(); add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => Cm); fraction = :rest);
        add_phase!(r, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => Ci); fraction = 0.1); r
    )
    Cm, Ci = TensISO{3}(30.0, 8.0), TensISO{3}(60.0, 16.0)
    @test _sc_newton(elastic(Cm, Ci), :C; times = [0.0, 1.0]) ≈
        _sc_newton(elastic(heaviside_law(Cm), heaviside_law(Ci)), :C; times = [0.0, 1.0]) rtol = 1.0e-12
    @test _sc_newton(elastic(heaviside_law(Cm), Ci), :C; times = [0.0, 1.0]) ≈
        _sc_newton(elastic(heaviside_law(Cm), heaviside_law(Ci)), :C; times = [0.0, 1.0]) rtol = 1.0e-12

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

@testset "sc_alv_newton — cracks: the fixed point of the Picard iteration" begin
    # Newton solves the same self-consistent equation as the Picard iteration of
    # `homogenize_alv`, an independent code path: with cracks, in which the crack
    # compliance enters the concentration composed with the running estimate,
    # both reach the same estimate (they differed by 24 % at ε = 0.1 when the
    # Newton step added the compliance itself).
    times = collect(range(0.0, 3.0; length = 6))
    m = maxwell_iso(10.0, 6.0, 0.8, 2.5)
    for (ε, f) in ((0.1, 0.0), (0.3, 0.3))
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => m); fraction = :rest)
        f > 0 && add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(0.03, 0.02))); fraction = f)
        add_phase!(r, :F, PennyCrack(1.0), Dict(:C => m); density = ε, symmetrize = :iso)
        picard = Matrix(homogenize_alv(r, SelfConsistent(reltol = 1.0e-12, maxiters = 2000), :C; times))
        @test _sc_newton(r, :C; times) ≈ picard rtol = 1.0e-8
    end
end

@testset "sc_alv_newton — beyond the percolation of a stiff phase, the physical root" begin
    # Beyond 40 % of stiff spheres the self-consistent equation has several roots.
    # Started from the Picard estimate, Newton refines the physical one: under constant
    # kernels, the elastic self-consistent estimate of `homogenize`, an independent
    # solver, at every instant (its Newton solver: its Picard iteration crawls there and
    # stops at 4e-6 within its default 100 iterations). Started from the matrix, the
    # march reaches a root of negative stiffness, which is refused.
    mandel(C) = TensND.mandel(TensND.canonical_components(C); symmetrize = true)
    Cm, Ci = TensISO{3}(3 * 10.0, 2 * 6.0), TensISO{3}(3.0e4, 1.4e4)
    times = [0.0, 1.0]
    for ε in (0.0, 0.3)
        el, al = RVE(), RVE()
        add_phase!(el, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest)
        add_phase!(el, :I, Ellipsoid(1.0), Dict(:C => Ci); fraction = 0.6)
        add_phase!(al, :M, Ellipsoid(1.0), Dict(:C => heaviside_law(Cm)); fraction = :rest)
        add_phase!(al, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(Ci)); fraction = 0.6)
        if ε > 0
            add_phase!(el, :F, PennyCrack(1.0), Dict(:C => Cm); density = ε, symmetrize = :iso)
            add_phase!(al, :F, PennyCrack(1.0), Dict(:C => heaviside_law(Cm)); density = ε, symmetrize = :iso)
        end
        E = mandel(homogenize(el, SelfConsistent(algorithm = NewtonDefault(), reltol = 1.0e-12), :C))
        N = _sc_newton(al, :C; times)
        @test N[1:6, 1:6] ≈ E rtol = 1.0e-8
        @test N[7:12, 7:12] ≈ E rtol = 1.0e-8
    end
    al = RVE()
    add_phase!(al, :M, Ellipsoid(1.0), Dict(:C => heaviside_law(Cm)); fraction = :rest)
    add_phase!(al, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(Ci)); fraction = 0.6)
    @test_throws "negative instantaneous modulus" _sc_newton(al, :C; times, init = :matrix)
    @test_throws "init must be :picard or :matrix" _sc_newton(al, :C; times, init = :elastic)
end

@testset "sc_alv_newton — SelfConsistent(algorithm = NewtonDefault()) in homogenize_alv" begin
    # The algorithm of `SelfConsistent` selects the solver of the ALV equation, at
    # order 4 and at order 2: Newton reaches the estimate of the Picard iteration,
    # with layered spheres too, whose recurrences enter the same step.
    times = [0.0, 0.5, 1.0, 2.0]
    newton = SelfConsistent(algorithm = NewtonDefault(), reltol = 1.0e-12)
    picard = SelfConsistent(reltol = 1.0e-13, maxiters = 3000)
    mat, inc = maxwell_iso(10.0, 6.0, 0.8, 2.5), maxwell_iso(40.0, 25.0, 0.5, 4.0)
    grain = LayeredSphere((0.6, 1.0), (iso_stiffness(50.0, 20.0), iso_stiffness(3.0, 1.5)))
    for f in (0.3, 0.7), layered in (false, true)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => mat); fraction = :rest)
        add_phase!(r, :I, layered ? grain : Ellipsoid(1.0), Dict(:C => inc); fraction = f)
        P = Matrix(homogenize_alv(r, picard, :C; times))
        @test Matrix(homogenize_alv(r, newton, :C; times)) ≈ P rtol = 1.0e-10
    end
    K0 = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(2.0 * exp(-(t - tp) / 0.9)) : TensISO{3}(0.0))
    K1 = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(5.0 * (1 + 0.5tp) * exp(-(t - tp) / 1.4)) : TensISO{3}(0.0))
    coated = LayeredSphere((0.7, 1.0), (TensISO{3}(20.0), TensISO{3}(1.0)))
    for f in (0.3, 0.8), layered in (false, true)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
        add_phase!(r, :I, layered ? coated : Ellipsoid(1.0), Dict(:K => K1); fraction = f)
        P = Matrix(homogenize_alv(r, picard, :K; times))
        @test Matrix(homogenize_alv(r, newton, :K; times)) ≈ P rtol = 1.0e-10
    end
    # The Newton keywords: its iteration cap per row, its tolerances; the Picard ones
    # are refused by name.
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => mat); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => inc); fraction = 0.3)
    @test_logs (:warn, r"row 1 not converged in 1 iterations") match_mode = :any homogenize_alv(
        r, SelfConsistent(algorithm = NewtonDefault(), maxiters = 1, reltol = 0.0, abstol = 0.0), :C; times
    )
    @test_throws "damping" homogenize_alv(r, SelfConsistent(algorithm = NewtonDefault(), damping = 0.5), :C; times)
    # Another algorithm, or a seed, is refused rather than ignored.
    @test_throws "SelfConsistent(algorithm = AutoNonlinear()) is not available" homogenize_alv(r, SelfConsistent(algorithm = AutoNonlinear()), :C; times)
    @test_throws "SelfConsistent(init = :voigt) is not supported" homogenize_alv(r, SelfConsistent(init = :voigt), :C; times)
    @test_throws "AsymmetricSelfConsistent(algorithm = NewtonDefault()) is not available" homogenize_alv(r, AsymmetricSelfConsistent(:M; algorithm = NewtonDefault()), :C; times)
    @test_throws "AsymmetricSelfConsistent(init = :voigt) is not supported" homogenize_alv(r, AsymmetricSelfConsistent(:M; init = :voigt), :C; times)
end

@testset "sc_alv_newton — order 2 beyond percolation: the Bruggeman root" begin
    # Spheres of constant conductivities: the self-consistent estimate is the root of
    # Bruggeman's Σ f (kᵢ − k)/(kᵢ + 2k) = 0 at every instant, a closed form (the
    # positive root of a quadratic for two phases). At 60 % of a phase 1000 times as
    # conductive, the Newton march started from the matrix reaches another root, with
    # a negative conductivity, which is refused.
    k₁, k₂, f = 1.0, 1000.0, 0.6
    b = (3f - 1) * k₂ + (2 - 3f) * k₁
    bruggeman = (b + sqrt(b^2 + 8k₁ * k₂)) / 4
    times = [0.0, 1.0, 3.0]
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{3}(k₁))); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{3}(k₂))); fraction = f)
    λ = iso_order2_params_from_blocks(_sc_newton(r, :K; times, reltol = 1.0e-13))
    @test all(λ[i, i] ≈ bruggeman for i in eachindex(times))
    @test all(abs(λ[i, j]) ≤ 1.0e-10 * bruggeman for i in eachindex(times) for j in 1:(i - 1))
    @test_throws "negative instantaneous modulus" _sc_newton(r, :K; times, init = :matrix)
end

@testset "sc_alv_newton — ForwardDiff through the march, both orders" begin
    # Scaling every modulus by k scales the self-consistent estimate by k: its
    # derivative in k is the estimate at k = 1, whatever k. The derivative is carried
    # by the start, the Newton iterations and their last full step.
    times = collect(range(0.0, 2.0; length = 4))
    function cell(k, prop)
        law(a, b) = prop === :C ? ViscoLaw((t, tp) -> k * TensISO{3}(a * (2 + exp(-(t - tp))), b)) :
            ViscoLaw((t, tp) -> TensISO{3}(k * a * (2 + exp(-(t - tp)))))
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(prop => law(3.0, 2.0)); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0), Dict(prop => law(30.0, 20.0)); fraction = 0.45)
        return r
    end
    for prop in (:C, :K)
        sc(k) = _sc_newton(cell(k, prop), prop; times, reltol = 1.0e-12)
        @test ForwardDiff.derivative(sc, 1.7) ≈ sc(1.0) rtol = 1.0e-9
    end
end
