# =============================================================================
#  test_self_consistent_nls.jl — SelfConsistent / AsymmetricSelfConsistent
#  through the `MeanFieldHomogenizationNonlinearSolveExt` weak extension.
#
#  Coverage:
#   1. SC via `NewtonRaphson()` / `TrustRegion()` agrees with the built-in
#      `AndersonDefault` / `NewtonDefault` solvers.
#   2. ASC (both stiffness and compliance branches) likewise.
#   3. Conductivity (`property = :K`) via a SciML algorithm.
#   4. `AutoNonlinear()` matches the explicit `TrustRegion()` result when
#      the extension is active (the ext-loaded branch of the resolver);
#      the extension-absent fallback path is unit-tested directly by
#      calling `_solve_sc(NewtonDefault(), …)`, since the extension
#      cannot be "unloaded" mid-process.
#   5. ForwardDiff sensitivity through a SciML algorithm (inclusion
#      modulus, volume fraction, matrix modulus) matches the built-in
#      Newton solver, whose derivatives come from converging the partials
#      themselves — a different algorithm from the implicit-function-theorem
#      lift of the extension. Nested duals (a second derivative) included.
#   6. Regression: differentiating w.r.t. the MATRIX'S OWN modulus on a
#      non-spherical, `IsoSymmetrize`d RVE (the strength-criterion
#      recipe of `docs/src/tutorials/strength_criteria.md`) through a SciML algorithm.
#      This used to crash: the primal/IFT solve rebuilt the iterating
#      tensor at a plain `Float64` type while the RVE's other captured
#      state stayed `Dual`, which is unsupported for a mixed-type
#      `TensTI` dcontract (hit only by non-spherical / symmetrized
#      geometries, never by a sphere) — see `_solve_sc`'s `embed`
#      helper in the extension.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using NonlinearSolve

const ATOL_NLS = 1.0e-6
const RTOL_NLS = 1.0e-6

_cmp(a, b; atol = ATOL_NLS, rtol = RTOL_NLS) =
    isapprox(get_array(a), get_array(b); atol = atol, rtol = rtol)

@testset "SelfConsistent — NonlinearSolve algorithms agree with built-ins" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_anderson = homogenize(rve, SelfConsistent())
    C_newton = homogenize(rve, SelfConsistent(; algorithm = NewtonDefault()))
    C_nr = homogenize(rve, SelfConsistent(; algorithm = NewtonRaphson()))
    C_tr = homogenize(rve, SelfConsistent(; algorithm = TrustRegion()))

    @test _cmp(C_nr, C_anderson)
    @test _cmp(C_tr, C_anderson)
    @test _cmp(C_nr, C_newton)
end

@testset "AsymmetricSelfConsistent — NonlinearSolve, stiffness branch" begin
    # Matrix-soft / inclusion-stiff selects the stiffness-form iteration.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    A_anderson = homogenize(rve, AsymmetricSelfConsistent())
    A_tr = homogenize(rve, AsymmetricSelfConsistent(; algorithm = TrustRegion()))
    @test _cmp(A_tr, A_anderson)
end

@testset "AsymmetricSelfConsistent — NonlinearSolve, compliance branch" begin
    # Matrix-stiff / inclusion-soft (porous-like) selects the
    # compliance-form iteration.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(90.0, 30.0)); fraction = :rest)
    add_phase!(
        rve, :V, Ellipsoid(1.0), Dict(:C => TensISO{3}(0.01, 0.005));
        fraction = 0.2
    )
    A_anderson = homogenize(rve, AsymmetricSelfConsistent())
    A_tr = homogenize(rve, AsymmetricSelfConsistent(; algorithm = TrustRegion()))
    @test _cmp(A_tr, A_anderson)
end

@testset "SelfConsistent — NonlinearSolve, conductivity" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(5.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(20.0)); fraction = 0.25)
    K_anderson = homogenize(rve, SelfConsistent(), :K)
    K_tr = homogenize(rve, SelfConsistent(; algorithm = TrustRegion()), :K)
    @test _cmp(K_tr, K_anderson)
end

@testset "SelfConsistent — AutoNonlinear resolves to the SciML backend" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    C_tr = homogenize(rve, SelfConsistent(; algorithm = TrustRegion()))
    C_auto = homogenize(rve, SelfConsistent(; algorithm = AutoNonlinear()))
    @test _cmp(C_auto, C_tr; atol = 1.0e-10, rtol = 1.0e-10)

    # Extension-absent fallback: the resolver dispatches on the extension, so
    # its other branch is reached directly with `nothing` (the extension is
    # process-global and cannot be unloaded once `using NonlinearSolve` has run).
    step = C -> MeanFieldHomogenization.Schemes._sc_step(rve, C, :C)
    C_m = phase_property(rve, :M, :C)
    C_fallback = MeanFieldHomogenization.Schemes._solve_sc_auto(nothing, step, C_m)
    C_newton = homogenize(rve, SelfConsistent(; algorithm = NewtonDefault()))
    @test _cmp(C_fallback, C_newton)
end

@testset "SelfConsistent — ForwardDiff through NonlinearSolve (IFT lift)" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )
    idxC = C -> canonical_components(C)[1, 1, 1, 1]
    tight = (; abstol = 1.0e-13, reltol = 1.0e-13)
    newton = SelfConsistent(; algorithm = NewtonDefault(), tight...)
    params = (property(:I, :C, :bulk), amount(:I), property(:M, :C, :shear))
    refs = [derivative(rve, newton, q; indexer = idxC) for q in params]

    for algo in (NewtonRaphson(), TrustRegion())
        sc = SelfConsistent(; algorithm = algo, tight...)
        for (q, ref) in zip(params, refs)
            @test isapprox(derivative(rve, sc, q; indexer = idxC), ref; rtol = 1.0e-8)
        end
    end

    # Second derivative in the inclusion's bulk modulus: nested duals, whose
    # cross terms the primal Jacobian of the lift does not give on its own.
    function c1111(k_I, scheme)
        r = RVE(; T = typeof(k_I))
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(k_I, 20.0)); fraction = 0.3)
        return idxC(homogenize(r, scheme, :C))
    end
    d2(scheme) = ForwardDiff.derivative(k -> ForwardDiff.derivative(k′ -> c1111(k′, scheme), k), 20.0)
    d2_ref = d2(newton)
    @test !iszero(d2_ref)
    for algo in (NewtonRaphson(), TrustRegion())
        @test isapprox(d2(SelfConsistent(; algorithm = algo, tight...)), d2_ref; rtol = 1.0e-7)
    end
end

@testset "SelfConsistent — NonlinearSolve, non-spherical IsoSymmetrize regression" begin
    # Strength-criterion recipe (docs/src/tutorials/strength_criteria.md and nonlinear_solvers.md):
    # oblate spheroid solid + pore, IsoSymmetrize'd, differentiated w.r.t.
    # the SOLID's own shear modulus — so a seed taken from that phase
    # is ITSELF Dual-typed from the very first SC iterate (unlike the
    # inclusion-modulus / fraction cases above, where only the residual
    # promotes to Dual while `x0` stays real).
    k_s, TINY, ω_aspect, φ_value = 1.0e6, 1.0e-12, 0.1, 0.15

    function C_hom_iso_2vec(μs::T, scheme) where {T}
        r = RVE(; T = T)
        add_phase!(r, :SOLID, Spheroid(ω_aspect), Dict(:C => iso_stiffness(convert(T, k_s), μs)); fraction = :rest, symmetrize = IsoSymmetrize())
        add_phase!(
            r, :PORE, Spheroid(ω_aspect),
            Dict(:C => iso_stiffness(convert(T, TINY), convert(T, TINY)));
            fraction = convert(T, φ_value), symmetrize = IsoSymmetrize()
        )
        C = homogenize(r, scheme, :C)
        return [k_mu(best_fit_iso(C))...]
    end

    μs0 = 1.0
    for algo in (NewtonRaphson(), TrustRegion())
        sc = SelfConsistent(; algorithm = algo)
        # Must not throw (this used to crash with a `TensTI` mixed-type
        # `MethodError` before the `embed` fix).
        d = ForwardDiff.derivative(μ -> C_hom_iso_2vec(μ, sc), μs0)
        d_picard = ForwardDiff.derivative(μ -> C_hom_iso_2vec(μ, SelfConsistent()), μs0)
        @test isapprox(d, d_picard; rtol = 1.0e-3)
    end
end
