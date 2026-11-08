# =============================================================================
#  test_brinkman.jl — Brinkman inclusions (Stokes-to-Darcy permeability).
#
#  References are independent of the code under test:
#   1. the closed-form estimates of Bignonnet (IJES 2022) for granular, fibrous
#      and tubular media (dilute, Mori–Tanaka, self-consistent, differential),
#      solved here by bisection on their scalar equations;
#   2. Darcy's limit R → ∞ against the conduction inclusions of the package;
#   3. the identity k·B = A·k₀ between the two formulations;
#   4. dimensional analysis (r ∝ R⁻²) and the implicit-function theorem for AD;
#   5. the refusals of the problems that have no solution.
#  The symbolic route is tested in `test_brinkman_symbolic.jl`.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SpecialFunctions: besselk, besseli

const BR = MeanFieldHomogenization.Brinkman

_br_iso(x) = TensISO{3}(x)
_br_val(t) = first(TensND.get_data(t))
_br_ρ(x) = besselk(0, x) / besselk(1, x)

# Root of a scalar equation with a sign change on [a, b], by bisection: slow and
# independent of anything the package does.
function _br_bisect(f, a, b; n = 200)
    fa = f(a)
    for _ in 1:n
        m = (a + b) / 2
        fm = f(m)
        (fm > 0) == (fa > 0) ? (a = m; fa = fm) : (b = m)
    end
    return (a + b) / 2
end

# Impervious grains of radius R in a pore space, resistance form.
function _br_granular(φ, R, pore; matrix = :p)
    rve = RVE()
    add_phase!(rve, :p, pore, Dict(:r => _br_iso(0.0)); fraction = :rest, symmetrize = :iso)
    add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => _br_iso(0.0)); fraction = 1 - φ)
    return rve
end

@testset "Brinkman — granular medium, closed-form schemes" begin
    R = 1.3
    for φ in (0.9, 0.6, 0.4)
        rve = _br_granular(φ, R, BrinkmanSphere(R, BrinkmanFluid()))
        rdil = 9 * (1 - φ) / (2R^2)
        @test _br_val(homogenize(rve, Dilute(:p), :r)) ≈ rdil rtol = 1.0e-13
        @test _br_val(homogenize(rve, MoriTanaka(:p), :r)) ≈ rdil / φ rtol = 1.0e-13
        # Self-consistent, spherical pores of the grain radius (eq. 61, α = R√r)
        eq61(α) = (1 - φ) / φ - 2α^2 * (15 + 15α + 6α^2 + α^3) / ((3 + 3α + α^2) * (45 + 45α + 6α^2 + α^3))
        αsc = _br_bisect(eq61, 1.0e-6, 1.0e3)
        rsc = homogenize(rve, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(rdil / φ), reltol = 1.0e-12), :r)
        @test _br_val(rsc) ≈ αsc^2 / R^2 rtol = 1.0e-8
        # Differential (Wilkinson's iterated dilute scheme, eq. 58)
        eq58(α) = -1.5 * log(φ) - log(1 + α + α^2 / 3) + 2sqrt(3) * (atan((2α + 3) / sqrt(3)) - π / 3)
        αdif = _br_bisect(eq58, 0.0, 1.0e3)
        rdif = homogenize(rve, DifferentialScheme(:p; reltol = 1.0e-11, abstol = 1.0e-13), :r)
        @test _br_val(rdif) ≈ αdif^2 / R^2 rtol = 1.0e-8
    end
end

@testset "Brinkman — self-consistent with cylindrical pores and fibers" begin
    Rs = 1.0
    for φ in (0.9, 0.5)
        # Spherical grains, cylindrical pores of equal specific surface (eq. 62)
        rve = _br_granular(φ, Rs, BrinkmanCylinder(2Rs / 3, BrinkmanFluid(); euler_angles = (0.4, 1.1, 0.0)))
        r = _br_val(homogenize(rve, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(1.0), reltol = 1.0e-12), :r))
        function eq62(rsc)
            αs, αp = Rs * sqrt(rsc), 2Rs / 3 * sqrt(rsc)
            g = 8 + αp^2 + 4αp * _br_ρ(αp)
            return (1 - φ) * 3 * (3 + 3αs + αs^2) / (2Rs^2) - φ * rsc * (g / 24 + 4g / (3 * (g + 8)))
        end
        @test r ≈ _br_bisect(eq62, 1.0e-6, 1.0e6) rtol = 1.0e-8
        # Fibers and cylindrical pores of the same radius (eq. 67b)
        fib = RVE()
        add_phase!(fib, :p, BrinkmanCylinder(Rs, BrinkmanFluid()), Dict(:r => _br_iso(0.0)); fraction = φ, symmetrize = :iso)
        add_phase!(fib, :s, BrinkmanCylinder(Rs, BrinkmanSolid()), Dict(:r => _br_iso(0.0)); fraction = 1 - φ, symmetrize = :iso)
        rf = _br_val(homogenize(fib, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(1.0), reltol = 1.0e-12), :r))
        function eq67b(rsc)
            α = Rs * sqrt(rsc)
            g = 8 + α^2 + 4α * _br_ρ(α)
            return (1 - φ) * 5α * (2 + α * _br_ρ(α)) / (3 * _br_ρ(α)) - φ * α^2 / 3 * (g / 8 + 4g / (g + 8))
        end
        @test rf ≈ _br_bisect(eq67b, 1.0e-6, 1.0e6) rtol = 1.0e-8
    end
end

@testset "Brinkman — coated sphere, self-consistent and cell identity" begin
    for φ in (0.8, 0.5)
        η = cbrt(1 - φ)
        c = BrinkmanCoatedSphere(1.0, 1 / η)
        rve = RVE()
        add_phase!(rve, :c, c, Dict(:r => _br_iso(0.0)); fraction = 1.0)
        r = homogenize(rve, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(1.0), reltol = 1.0e-12), :r)
        # With one pattern at fraction 1, SC is r_sc = ⟨r·A⟩ ⋅ A⁻¹ in its own reference (eq. 63).
        A = gradient_gradient_concentration(c, _br_iso(0.0), r)
        rA = flux_gradient_concentration(c, _br_iso(0.0), r)
        @test _br_val(r) ≈ _br_val(rA) / _br_val(A) rtol = 1.0e-8
    end
    # Without reference resistance the coated sphere tends to its own core: at
    # η → 1 the shell vanishes and the drag is Brinkman's (A.3b).
    a = 0.7
    _, R²rA = BR._coated_r(a, 1 - 1.0e-9)
    @test R²rA ≈ BR._sphere_solid_r(a) rtol = 1.0e-7
end

@testset "Brinkman — spanning pores in an impervious matrix (permeability form)" begin
    R = 1.0
    P = PermeabilityForm()
    for φ in (0.1, 0.4)
        for (pore, kmt) in (
                (BrinkmanCylinder(R, BrinkmanFluid(); form = P), R^2 / 24 * φ / (1 - φ)),
                (BrinkmanLayer(R, BrinkmanFluid(); form = P), 2R^2 / 9 * φ / (1 - φ)),
            )
            rve = RVE()
            add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid(); form = P), Dict(:k => _br_iso(0.0)); fraction = :rest)
            add_phase!(rve, :p, pore, Dict(:k => _br_iso(0.0)); fraction = φ, symmetrize = :iso)
            @test _br_val(homogenize(rve, MoriTanaka(:s), :k)) ≈ kmt rtol = 1.0e-12
        end
    end
    # Differential scheme from the solid, planar pores (eq. 70b), integrated
    # here by a fixed-step Runge-Kutta from k = 0.
    function rk4(f, y, a, b; n = 20000)
        h = (b - a) / n
        x = a
        for _ in 1:n
            k1 = f(x, y)
            k2 = f(x + h / 2, y + h / 2 * k1)
            k3 = f(x + h / 2, y + h / 2 * k2)
            k4 = f(x + h, y + h * k3)
            y += h / 6 * (k1 + 2k2 + 2k3 + k4)
            x += h
        end
        return y
    end
    f70b(φ, k) = (k + 2 * (k + R * sqrt(k) + R^2 / 3)) / (3 * (1 - φ))
    rve = RVE()
    add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid(); form = P), Dict(:k => _br_iso(0.0)); fraction = :rest)
    add_phase!(rve, :p, BrinkmanLayer(R, BrinkmanFluid(); form = P), Dict(:k => _br_iso(0.0)); fraction = 0.4, symmetrize = :iso)
    kd = homogenize(rve, DifferentialScheme(:s; reltol = 1.0e-10, abstol = 1.0e-12), :k)
    @test _br_val(kd) ≈ rk4(f70b, 0.0, 0.0, 0.4) rtol = 1.0e-7
end

@testset "Brinkman — resistance and permeability forms describe the same flow" begin
    # k·B = A·k₀ and B = ⟨r·A⟩ k₀ (Bignonnet, eq. 20), with r₀ = 1/k₀ and r₁ = 1/k₁.
    k₀, k₁ = 0.5, 0.2
    Rr, Kr = _br_iso(1 / k₀), _br_iso(1 / k₁)
    R0, K0 = _br_iso(k₀), _br_iso(k₁)
    ang = (0.3, 0.7, 0.0)
    pairs = [
        (m, f) -> BrinkmanSphere(1.3, m; form = f),
        (m, f) -> BrinkmanCylinder(1.3, m; form = f, euler_angles = ang),
    ]
    for make in pairs, m in (BrinkmanPorous(), BrinkmanFluid(), BrinkmanSolid())
        ir, ik = make(m, ResistanceForm()), make(m, PermeabilityForm())
        A = gradient_gradient_concentration(ir, Kr, Rr)
        rA = flux_gradient_concentration(ir, Kr, Rr)
        B = gradient_gradient_concentration(ik, K0, R0)
        kB = flux_gradient_concentration(ik, K0, R0)
        @test canonical_components(kB) ≈ canonical_components(A) * k₀ atol = 1.0e-14
        @test canonical_components(B) ≈ canonical_components(rA) * k₀ atol = 1.0e-14
    end
    for m in (BrinkmanPorous(), BrinkmanFluid())
        ir = BrinkmanLayer(0.8, m; euler_angles = ang)
        ik = BrinkmanLayer(0.8, m; form = PermeabilityForm(), euler_angles = ang)
        @test canonical_components(flux_gradient_concentration(ik, K0, R0)) ≈
            canonical_components(gradient_gradient_concentration(ir, Kr, Rr)) * k₀ atol = 1.0e-14
        @test canonical_components(gradient_gradient_concentration(ik, K0, R0)) ≈
            canonical_components(flux_gradient_concentration(ir, Kr, Rr)) * k₀ atol = 1.0e-14
    end
    c, ck = BrinkmanCoatedSphere(1.0, 1.4), BrinkmanCoatedSphere(1.0, 1.4; form = PermeabilityForm())
    @test _br_val(flux_gradient_concentration(ck, K0, R0)) ≈ _br_val(gradient_gradient_concentration(c, Kr, Rr)) * k₀ rtol = 1.0e-14
    @test _br_val(gradient_gradient_concentration(ck, K0, R0)) ≈ _br_val(flux_gradient_concentration(c, Kr, Rr)) * k₀ rtol = 1.0e-14
end

@testset "Brinkman — porous closed forms against extended precision" begin
    # The published closed forms (sphere A.2, layer A.11), evaluated in 256-bit
    # arithmetic where their cancellation as α₁ → 0 is harmless.
    function sphere_big(a, b)
        T = tanh(b)
        β = 3 * (b^2 - a^2) * (1 + a)
        return 3 * ((a^3 * b^2 - β) * T + (a^2 * b^2 + β) * b) /
            ((a^3 * b^2 + 2a * b^4 - β) * T + (a^2 * b^2 + 2b^4 + β) * b)
    end
    layer_big(a, b) = a * ((b^2 * (1 + a) - a^2) * tanh(b) + a^2 * b) / (b^3 * (b * tanh(b) + a))
    setprecision(BigFloat, 256) do
        for a in (0.0, 0.3, 2.0, 40.0), b in (1.0e-6, 1.0e-3, 0.2, 0.999, 1.001, 3.0, 50.0)
            a == 0 && continue
            ab, bb = big(a), big(b)
            @test BR._sphere_porous_r(a, b) ≈ sphere_big(ab, bb) rtol = 4.0e-15
            @test BR._sphere_porous_k(1 / a, b) ≈ sphere_big(ab, bb) rtol = 4.0e-15
            @test BR._layer_porous_r(a, b) ≈ layer_big(ab, bb) rtol = 4.0e-15
            @test BR._layer_porous_k(1 / a, b) * a^2 ≈ layer_big(ab, bb) rtol = 4.0e-15
        end
    end
    # The cylinder (A.8) has no BigFloat Bessel functions: its closed form is
    # compared in Float64 where it is well conditioned, and its small-α₁ limit
    # is the Stokes cylinder.
    ρ(x) = besselk(0, x) / besselk(1, x)
    q(x) = besseli(1, x) / besseli(0, x)
    for a in (0.3, 2.0, 15.0), b in (0.7, 3.0, 20.0)
        c = (2 * (b^2 - a^2) + a * b^2 * ρ(a)) * q(b) + b * a^2
        An = a * c / (b^3 * (b * ρ(a) * q(b) + a))
        At = 2a * c / ((2a * (b^2 - a^2) + b^2 * (b^2 + a^2) * ρ(a)) * q(b) + a * b * (a^2 + b^2))
        @test all(BR._cylinder_porous_r(a, b) .≈ (An, At))
        @test all(BR._cylinder_porous_k(1 / a, b) .* a^2 .≈ (An, At))
    end
    for a in (0.3, 2.0)
        @test all(isapprox.(BR._cylinder_porous_r(a, 1.0e-7), BR._cylinder_fluid_r(a); rtol = 1.0e-13))
    end
end

@testset "Brinkman — limits" begin
    k₀, k₁ = 0.5, 0.2
    # Darcy (R → ∞): the permeability form is the conduction problem with k as
    # conductivity and the force as gradient.
    P = PermeabilityForm()
    B = gradient_gradient_concentration(BrinkmanSphere(1.0e7; form = P), _br_iso(k₁), _br_iso(k₀))
    @test canonical_components(B) ≈ canonical_components(gradient_gradient_concentration(Ellipsoid(1.0), _br_iso(k₁), _br_iso(k₀))) rtol = 1.0e-6
    n = (0.0, 0.0, 1.0)
    Bc = gradient_gradient_concentration(BrinkmanCylinder(1.0e8; form = P), _br_iso(k₁), _br_iso(k₀))
    @test canonical_components(Bc) ≈ canonical_components(TensND.TensTI{2}(2k₀ / (k₀ + k₁), 1.0, n)) rtol = 1.0e-6
    Bl = gradient_gradient_concentration(BrinkmanLayer(1.0e8; form = P), _br_iso(k₁), _br_iso(k₀))
    @test canonical_components(Bl) ≈ canonical_components(TensND.TensTI{2}(1.0, k₀ / k₁, n)) rtol = 1.0e-6
    # A porous inclusion tends to the fluid one as r₁ → 0 and to the solid one as
    # r₁ → ∞; exactly r₁ = 0 (resp. k₁ = 0) is the fluid (resp. solid) closed form.
    for make in (
            m -> BrinkmanSphere(1.1, m), m -> BrinkmanCylinder(1.1, m), m -> BrinkmanLayer(1.1, m),
        )
        r0 = _br_iso(0.7)
        Afl = canonical_components(gradient_gradient_concentration(make(BrinkmanFluid()), r0, r0))
        @test canonical_components(gradient_gradient_concentration(make(BrinkmanPorous()), _br_iso(1.0e-10), r0)) ≈ Afl rtol = 1.0e-7
        @test canonical_components(gradient_gradient_concentration(make(BrinkmanPorous()), _br_iso(0.0), r0)) ≈ Afl rtol = 1.0e-14
        @test canonical_components(flux_gradient_concentration(make(BrinkmanPorous()), _br_iso(0.0), r0)) == zeros(3, 3)
    end
    for S in (BrinkmanSphere, BrinkmanCylinder)
        r0 = _br_iso(0.7)
        rAs = canonical_components(flux_gradient_concentration(S(1.1, BrinkmanSolid()), r0, r0))
        @test canonical_components(flux_gradient_concentration(S(1.1, BrinkmanPorous()), _br_iso(1.0e10), r0)) ≈ rAs rtol = 1.0e-4
        kf = PermeabilityForm()
        k0 = _br_iso(1 / 0.7)
        @test canonical_components(gradient_gradient_concentration(S(1.1, BrinkmanPorous(); form = kf), _br_iso(0.0), k0)) ≈
            canonical_components(gradient_gradient_concentration(S(1.1, BrinkmanSolid(); form = kf), k0, k0)) rtol = 1.0e-14
    end
    # Impervious matrix (k₀ = 0): tubes and slits carry Poiseuille's flow.
    z = _br_iso(0.0)
    @test _br_val(flux_gradient_concentration(BrinkmanSphere(2.0, BrinkmanFluid(); form = P), z, z)) == 0
    @test canonical_components(flux_gradient_concentration(BrinkmanCylinder(2.0, BrinkmanFluid(); form = P), z, z)) ≈
        canonical_components(TensND.TensTI{2}(0.0, 4.0 / 8, n)) atol = 1.0e-15
    @test canonical_components(flux_gradient_concentration(BrinkmanLayer(2.0, BrinkmanFluid(); form = P), z, z)) ≈
        canonical_components(TensND.TensTI{2}(4.0 / 3, 0.0, n)) atol = 1.0e-15
    # A porous tube in an impervious matrix: Brinkman flow with no slip at the wall.
    kp = 0.3
    b = 2.0 / sqrt(kp)
    kBc = flux_gradient_concentration(BrinkmanCylinder(2.0; form = P), _br_iso(kp), z)
    @test canonical_components(kBc)[3, 3] ≈ kp * (1 - 2besseli(1, b) / (b * besseli(0, b))) rtol = 1.0e-13
    kBl = flux_gradient_concentration(BrinkmanLayer(2.0; form = P), _br_iso(kp), z)
    @test canonical_components(kBl)[1, 1] ≈ kp * (1 - tanh(b) / b) rtol = 1.0e-13
    # Bessel ratios on both sides of the switch to their asymptotic series, and at ∞.
    x = BR._BESSEL_ASYMPTOTIC
    @test BR._k0k1(prevfloat(x)) ≈ BR._k0k1(nextfloat(x)) rtol = 4.0e-16
    @test BR._i1i0(prevfloat(x)) ≈ BR._i1i0(nextfloat(x)) rtol = 4.0e-16
    @test BR._k0k1(Inf) == 1 && BR._i1i0(Inf) == 1
    # A porous tube of vanishing permeability reaches the series for I₁/I₀ and
    # carries k₁ times the force on an impervious one, 1 + 2K₁(1)/K₀(1) at α₀ = 1.
    kt = flux_gradient_concentration(BrinkmanCylinder(1.0; form = P), _br_iso(1.0e-20), _br_iso(1.0))
    @test canonical_components(kt)[3, 3] ≈ 1.0e-20 * (1 + 2besselk(1, 1.0) / besselk(0, 1.0)) rtol = 1.0e-9
end

@testset "Brinkman — Maxwell and bounds" begin
    # Maxwell in the permeability form with a spherical distribution: the Darcy
    # Hill tensor 1/(3k₀) of eq. (54).
    k₀, R, φ = 1.0, 3.0, 0.3
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :m, Ellipsoid(1.0), Dict(:k => _br_iso(k₀)); fraction = :rest)
    pore = BrinkmanSphere(R, BrinkmanFluid(); form = PermeabilityForm())
    add_phase!(rve, :p, pore, Dict(:k => _br_iso(0.0)); fraction = φ)
    N = φ * (_br_val(flux_gradient_concentration(pore, _br_iso(0.0), _br_iso(k₀))) - k₀ * _br_val(gradient_gradient_concentration(pore, _br_iso(0.0), _br_iso(k₀))))
    @test _br_val(homogenize(rve, Maxwell(), :k)) ≈ k₀ + N / (1 - N / (3k₀)) rtol = 1.0e-13
    # Bounds: a solid has r = ∞ (k = 0), a fluid r = 0 (k = ∞), the coated
    # sphere both, in either formulation.
    Rf, Kf = ResistanceForm(), PermeabilityForm()
    for (geom, prop, voigt, reuss) in (
            (BrinkmanSphere(1.0, BrinkmanSolid()), :r, Inf, 0.0),
            (BrinkmanSphere(1.0, BrinkmanFluid()), :r, 0.0, Inf),
            (BrinkmanSphere(1.0, BrinkmanSolid(); form = Kf), :k, 0.0, Inf),
            (BrinkmanSphere(1.0, BrinkmanFluid(); form = Kf), :k, Inf, 0.0),
            (BrinkmanCoatedSphere(1.0, 2.0), :r, Inf, Inf),
        )
        ref = _br_iso(1.0)
        @test _br_val(MeanFieldHomogenization.Schemes._layer_voigt(geom, ref)) == voigt
        @test _br_val(MeanFieldHomogenization.Schemes._layer_reuss(geom, ref)) == reuss
    end
    # Hence a self-consistent start from Voigt is refused with a pointer to `init`.
    rve = _br_granular(0.6, 1.0, BrinkmanSphere(1.0, BrinkmanFluid()))
    @test _br_val(homogenize(rve, Voigt(), :r)) == Inf
    @test_throws ArgumentError homogenize(rve, SelfConsistent(init = :voigt), :r)
end

@testset "Brinkman — automatic differentiation" begin
    φ = 0.6
    function rsc(R; φ = φ)
        rve = RVE()
        add_phase!(rve, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => _br_iso(0.0)); fraction = :rest)
        add_phase!(rve, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => _br_iso(0.0)); fraction = 1 - φ)
        return _br_val(homogenize(rve, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(9 * (1 - φ) / (2φ * R^2)), reltol = 1.0e-12), :r))
    end
    # r scales as R⁻² at fixed porosity.
    @test ForwardDiff.derivative(R -> log(rsc(R)), 1.3) * 1.3 ≈ -2 rtol = 1.0e-10
    # d r_sc / dφ against the implicit-function theorem on eq. (61).
    F(α, φ) = (1 - φ) / φ - 2α^2 * (15 + 15α + 6α^2 + α^3) / ((3 + 3α + α^2) * (45 + 45α + 6α^2 + α^3))
    α = _br_bisect(a -> F(a, φ), 1.0e-6, 1.0e3)
    dαdφ = -ForwardDiff.derivative(p -> F(α, p), φ) / ForwardDiff.derivative(a -> F(a, φ), α)
    @test ForwardDiff.derivative(p -> rsc(1.0; φ = p), φ) ≈ 2α * dαdφ rtol = 1.0e-7
    # The same through `derivative` on the geometric field.
    rve = _br_granular(φ, 1.3, BrinkmanSphere(1.3, BrinkmanFluid()))
    d = derivative(
        rve, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(5.0), reltol = 1.0e-12), geometry(:s, :radius); output = :r,
        indexer = r -> canonical_components(r)[1, 1]
    )
    @test d ≈ ForwardDiff.derivative(
        R -> begin
            g = RVE()
            add_phase!(g, :p, BrinkmanSphere(1.3, BrinkmanFluid()), Dict(:r => _br_iso(0.0)); fraction = :rest)
            add_phase!(g, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => _br_iso(0.0)); fraction = 1 - φ)
            _br_val(homogenize(g, SelfConsistent(algorithm = NewtonDefault(), init = _br_iso(5.0), reltol = 1.0e-12), :r))
        end, 1.3
    ) rtol = 1.0e-8
    # Nested duals: second derivative of Brinkman's drag with respect to r₀,
    # d²/dr₀² [3(3 + 3R√r₀ + R²r₀)/(2R²)] = -9/(8R r₀^{3/2}).
    R, r₀ = 1.2, 0.8
    drag(x) = _br_val(flux_gradient_concentration(BrinkmanSphere(R, BrinkmanSolid()), _br_iso(x), _br_iso(x)))
    @test ForwardDiff.derivative(x -> ForwardDiff.derivative(drag, x), r₀) ≈ -9 / (8R * r₀^1.5) rtol = 1.0e-13
    # Duals in the cylinder (Bessel ratios) and inference.
    cyl = BrinkmanCylinder(1.0)
    f(x) = canonical_components(gradient_gradient_concentration(cyl, _br_iso(2x), _br_iso(x)))[1, 1]
    d1 = ForwardDiff.derivative(f, 0.9)
    @test isfinite(d1) && d1 != 0
    @inferred gradient_gradient_concentration(cyl, _br_iso(1.8), _br_iso(0.9))
    @inferred flux_gradient_concentration(BrinkmanSphere(1.0, BrinkmanSolid()), _br_iso(1.0), _br_iso(0.9))
    dual = ForwardDiff.Dual(0.9, 1.0)
    @inferred gradient_gradient_concentration(cyl, _br_iso(2dual), _br_iso(dual))
end

@testset "Brinkman — refusals and accessors" begin
    z, one_ = _br_iso(0.0), _br_iso(1.0)
    # Stokes' paradox: unbounded inclusions in a clear fluid.
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanCylinder(1.0, BrinkmanSolid()), one_, z)
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanLayer(1.0, BrinkmanFluid()), one_, z)
    @test_throws ArgumentError BrinkmanLayer(1.0, BrinkmanSolid())
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanLayer(1.0; form = PermeabilityForm()), z, one_)
    @test_throws ArgumentError BrinkmanSphere(-1.0)
    @test_throws ArgumentError BrinkmanCylinder(Inf)
    @test_throws ArgumentError BrinkmanCoatedSphere(1.0, 1.0)
    # Anisotropic reference or phase property, fourth-order tensors.
    ti = TensND.TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanSphere(1.0), one_, ti)
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanSphere(1.0), ti, one_)
    C = TensISO{3}(3.0, 2.0)
    @test_throws ArgumentError gradient_gradient_concentration(BrinkmanSphere(1.0), C, C)
    @test_throws ArgumentError flux_gradient_concentration(BrinkmanSphere(1.0), C, C)
    # Accessors and the contract.
    b = TensND.euler_basis((0.2, 0.5, 0.1))
    cyl = BrinkmanCylinder(1.0; basis = b)
    @test MeanFieldHomogenization.inclusion_basis(cyl) === b
    @test MeanFieldHomogenization.inclusion_basis(BrinkmanSphere(1.0)) isa TensND.CanonicalBasis
    @test MeanFieldHomogenization.dimension(cyl) == 3
    @test !MeanFieldHomogenization.is_homogeneous_inclusion(BrinkmanCoatedSphere(1.0, 2.0))
    @test MeanFieldHomogenization.is_homogeneous_inclusion(BrinkmanSphere(1.0))
    for i in (
            BrinkmanSphere(1.0, BrinkmanSolid()), cyl, BrinkmanLayer(1.0, BrinkmanFluid()),
            BrinkmanCoatedSphere(1.0, 2.0),
        )
        @test check_inclusion_interface(i; physics = :conduction, verbose = false)
    end
    # The axis of a tilted cylinder carries its axial component.
    n = Tuple(TensND.vecbasis(b)[:, 3])
    A = gradient_gradient_concentration(cyl, _br_iso(3.0), _br_iso(1.0))
    a = canonical_components(A)
    An, At = BR._cylinder_porous_r(1.0, sqrt(3.0))
    @test transpose(collect(n)) * a * collect(n) ≈ An rtol = 1.0e-13
    @test tr(a) ≈ An + 2At rtol = 1.0e-13
end
