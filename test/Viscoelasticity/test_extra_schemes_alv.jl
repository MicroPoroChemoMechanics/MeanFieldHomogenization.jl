using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_extra_schemes_alv.jl — PCW / ASC / DIFF ALV vs elastic limit and
#  vs the existing ALV schemes (consistency on identity / pure-matrix
#  edge cases).
#
#  The differential ALV part also covers: the compliance formulation
#  (elastic limit and genuinely ageing), `LayeredSphere` and crack phases,
#  the isotropy guard on the running effective medium, and the order-2
#  (conduction / diffusion) driver.
# =============================================================================

const _to_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

function _setup_2phase_elastic(;
        k_M = 10.0, μ_M = 4.0,
        k_I = 20.0, μ_I = 8.0,
        f_I = 0.2, n_times = 4
    )
    C_M_t = TensISO{3}(3 * k_M, 2 * μ_M)
    C_I_t = TensISO{3}(3 * k_I, 2 * μ_I)
    times = collect(range(0.0, 1.0; length = n_times))
    return (; C_M_t, C_I_t, times, f_I, f_M = 1 - f_I)
end

function _build_alv(ctx)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(ctx.C_M_t)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(ctx.C_I_t)); fraction = ctx.f_I
    )
    return rve
end

function _build_el(ctx)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_M_t); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_I_t);
        fraction = ctx.f_I
    )
    return rve
end

function _check_alv_elastic(
        C_alv::AbstractMatrix, M_ref::AbstractMatrix,
        n::Int; rtol = 1.0e-12, atol = 1.0e-12
    )
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test isapprox(C_alv[rows, rows], M_ref; rtol = rtol, atol = atol)
        for j in 1:(i - 1)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test maximum(abs, C_alv[rows, cols]) ≤ atol
        end
    end
    return
end

@testset "asymmetric_self_consistent_alv — elastic limit (sphere)" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    M_ref = _to_mandel(homogenize(_build_el(ctx), AsymmetricSelfConsistent(), :C))
    C_alv = homogenize_alv(
        _build_alv(ctx), AsymmetricSelfConsistent(), :C;
        times = ctx.times
    )
    _check_alv_elastic(C_alv, M_ref, n; atol = 1.0e-9, rtol = 1.0e-9)
end

@testset "pcw_alv — elastic limit (sphere distribution)" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    M_ref = _to_mandel(homogenize(_build_el(ctx), PonteCastanedaWillis(), :C))
    C_alv = homogenize_alv(
        _build_alv(ctx), PonteCastanedaWillis(), :C;
        times = ctx.times
    )
    _check_alv_elastic(C_alv, M_ref, n; atol = 1.0e-12)
end

@testset "differential_alv — elastic limit (sphere)" begin
    ctx = _setup_2phase_elastic()
    n = length(ctx.times)
    # Tighten ODE tolerances so the elastic limit holds at 1e-9 across
    # both the elastic and ALV pipelines (Tsit5's default 1e-6 reltol
    # is too loose for the strict atol = 1e-12 used previously).
    sch = DifferentialScheme(; nsteps = 50, abstol = 1.0e-12, reltol = 1.0e-10)
    M_ref = _to_mandel(homogenize(_build_el(ctx), sch, :C))
    C_alv = homogenize_alv(_build_alv(ctx), sch, :C; times = ctx.times)
    _check_alv_elastic(C_alv, M_ref, n; atol = 1.0e-9, rtol = 1.0e-9)
end

@testset "PCW vs Maxwell — equivalence in single-shape case" begin
    # PCW with a declared unit-sphere distribution ≡ Maxwell with the same
    # spherical distribution shape.
    ctx = _setup_2phase_elastic()
    rve = _build_alv(ctx)
    R_pcw = homogenize_alv(rve, PonteCastanedaWillis(), :C; times = ctx.times)
    R_max = homogenize_alv(rve, Maxwell(), :C; times = ctx.times)
    @test isapprox(R_pcw, R_max; atol = 1.0e-12)
end

@testset "ASC vs SC — same fixed point in elastic limit" begin
    ctx = _setup_2phase_elastic()
    rve = _build_alv(ctx)
    R_sc = homogenize_alv(rve, SelfConsistent(), :C; times = ctx.times)
    R_asc = homogenize_alv(
        rve, AsymmetricSelfConsistent(), :C;
        times = ctx.times
    )
    @test isapprox(R_sc, R_asc; atol = 1.0e-9, rtol = 1.0e-9)
end

@testset "Differential — independent of nsteps in elastic limit" begin
    ctx = _setup_2phase_elastic()
    rve = _build_alv(ctx)
    R20 = homogenize_alv(rve, DifferentialScheme(; nsteps = 20), :C; times = ctx.times)
    R100 = homogenize_alv(rve, DifferentialScheme(; nsteps = 100), :C; times = ctx.times)
    # Higher nsteps should converge — finite difference in nsteps is small.
    @test isapprox(R20, R100; atol = 1.0e-3, rtol = 1.0e-3)
end

# =============================================================================
#  Differential ALV : compliance formulation, layered spheres, cracks,
#  order-2 (conduction), and the isotropy guard.
# =============================================================================

@testset "differential_alv — stiffness ≡ compliance formulation" begin
    ctx = _setup_2phase_elastic()
    rve = _build_alv(ctx)
    tol = (abstol = 1.0e-12, reltol = 1.0e-10)
    R_s = homogenize_alv(rve, DifferentialScheme(; tol...), :C; times = ctx.times)
    R_c = homogenize_alv(
        rve, DifferentialScheme(; formulation = :compliance, tol...), :C;
        times = ctx.times
    )
    @test isapprox(R_s, R_c; rtol = 1.0e-8, atol = 1.0e-8)

    # Genuinely ageing matrix — the equivalence is not an artifact of the
    # elastic limit.
    law_M = maxwell_iso(20.0, 8.0, 2.0, 1.5)
    t = collect(range(0.0, 2.0; length = 5))
    aged = RVE()
    add_phase!(aged, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        aged, :I, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(ctx.C_I_t)); fraction = 0.25
    )
    A_s = homogenize_alv(aged, DifferentialScheme(; tol...), :C; times = t)
    A_c = homogenize_alv(
        aged, DifferentialScheme(; formulation = :compliance, tol...), :C; times = t
    )
    @test isapprox(A_s, A_c; rtol = 1.0e-8, atol = 1.0e-8)
end

@testset "differential_alv — LayeredSphere phase (elastic limit)" begin
    sphere = LayeredSphere(
        (0.8, 1.0),
        (TensISO{3}(3 * 80.0, 2 * 35.0), TensISO{3}(3 * 5.0, 2 * 2.0))
    )
    C_M_t = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_I_t = TensISO{3}(3 * 50.0, 2 * 20.0)
    times = collect(range(0.0, 1.0; length = 4))
    n = length(times)
    sch = DifferentialScheme(; abstol = 1.0e-12, reltol = 1.0e-10)

    rve_alv = RVE()
    add_phase!(rve_alv, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M_t)); fraction = :rest)
    add_phase!(rve_alv, :I, sphere, Dict(:C => heaviside_law(C_I_t)); fraction = 0.3)

    rve_el = RVE()
    add_phase!(rve_el, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_t); fraction = :rest)
    add_phase!(rve_el, :I, sphere, Dict(:C => C_I_t); fraction = 0.3)

    C_alv = homogenize_alv(rve_alv, sch, :C; times = times)
    _check_alv_elastic(C_alv, _to_mandel(homogenize(rve_el, sch, :C)), n; atol = 1.0e-8, rtol = 1.0e-8)
end

@testset "differential_alv — crack phase (elastic limit)" begin
    C_M_t = TensISO{3}(3 * 20.0, 2 * 8.0)
    times = collect(range(0.0, 1.0; length = 4))
    n = length(times)

    rve_alv = RVE()
    add_phase!(rve_alv, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M_t)); fraction = :rest)
    add_phase!(
        rve_alv, :CR, PennyCrack(1.0), Dict(:C => heaviside_law(C_M_t));
        density = 0.1, symmetrize = :iso
    )

    rve_el = RVE()
    add_phase!(rve_el, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_t); fraction = :rest)
    add_phase!(rve_el, :CR, PennyCrack(1.0), Dict(:C => C_M_t); density = 0.1, symmetrize = :iso)

    C_alv = homogenize_alv(rve_alv, DifferentialScheme(), :C; times = times)
    _check_alv_elastic(
        C_alv, _to_mandel(homogenize(rve_el, DifferentialScheme(), :C)), n;
        atol = 1.0e-6, rtol = 1.0e-6
    )
end

# The ALV Hill kernel is built for an isotropic reference; the differential
# scheme evaluates it against its RUNNING medium, so anything that takes that
# medium out of the iso class must be refused, not silently mis-evaluated.
@testset "differential_alv — isotropy guard" begin
    C_M_t = TensISO{3}(3 * 20.0, 2 * 8.0)
    C_I_t = TensISO{3}(3 * 50.0, 2 * 20.0)
    times = collect(range(0.0, 1.0; length = 4))

    # Aligned (non-spherical) inclusion.
    aligned = RVE()
    add_phase!(aligned, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M_t)); fraction = :rest)
    add_phase!(aligned, :I, Spheroid(0.2), Dict(:C => heaviside_law(C_I_t)); fraction = 0.2)
    @test_throws ArgumentError homogenize_alv(aligned, DifferentialScheme(), :C; times = times)

    # …accepted with an isotropic orientation average.
    randomized = RVE()
    add_phase!(randomized, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M_t)); fraction = :rest)
    add_phase!(
        randomized, :I, Spheroid(0.2), Dict(:C => heaviside_law(C_I_t));
        fraction = 0.2, symmetrize = :iso
    )
    @test size(homogenize_alv(randomized, DifferentialScheme(), :C; times = times)) ==
        (6 * length(times), 6 * length(times))

    # A crack without isotropic orientation average leaks its TI contribution.
    cracked = RVE()
    add_phase!(cracked, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_M_t)); fraction = :rest)
    add_phase!(cracked, :CR, PennyCrack(1.0), Dict(:C => heaviside_law(C_M_t)); density = 0.1)
    @test_throws ArgumentError homogenize_alv(cracked, DifferentialScheme(), :C; times = times)

    # A non-isotropic ALV matrix is refused outright.
    n̂ = (0.0, 0.0, 1.0)
    C_TI = TensND.TensTI{4, Float64, 5}((20.0, 30.0, 10.0, 8.0, 9.0), n̂)
    ti_matrix = RVE()
    add_phase!(ti_matrix, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_TI)); fraction = :rest)
    add_phase!(
        ti_matrix, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => heaviside_law(C_I_t));
        fraction = 0.2
    )
    @test_throws ArgumentError homogenize_alv(ti_matrix, DifferentialScheme(), :C; times = times)
end

@testset "differential_alv_order2 — elastic limit and dual form" begin
    K_M_t = TensISO{3}(1.0)
    K_I_t = TensISO{3}(10.0)
    times = collect(range(0.0, 1.0; length = 4))
    n = length(times)
    sch(form) = DifferentialScheme(;
        formulation = form, abstol = 1.0e-12, reltol = 1.0e-10
    )

    rve_alv = RVE()
    add_phase!(rve_alv, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => heaviside_law(K_M_t)); fraction = :rest)
    add_phase!(
        rve_alv, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => heaviside_law(K_I_t));
        fraction = 0.3
    )

    rve_el = RVE()
    add_phase!(rve_el, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M_t); fraction = :rest)
    add_phase!(rve_el, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_I_t); fraction = 0.3)
    K_ref = Array(homogenize(rve_el, sch(:stiffness), :K))

    K_alv = homogenize_alv(rve_alv, sch(:stiffness), :K; times = times)
    @test size(K_alv) == (3n, 3n)
    for i in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        @test isapprox(K_alv[rows, rows], K_ref; atol = 1.0e-9)
        for j in 1:(i - 1)
            cols = (3 * (j - 1) + 1):(3 * j)
            @test maximum(abs, K_alv[rows, cols]) ≤ 1.0e-9
        end
    end

    K_dual = homogenize_alv(rve_alv, sch(:compliance), :K; times = times)
    @test isapprox(K_alv, K_dual; rtol = 1.0e-8, atol = 1.0e-8)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Regression: `Maxwell` must read the RVE's distribution shape on the ALV path
#  as it does on the elastic one.  The dispatcher used to build its Hill kernel
#  on a hard-coded `Spheroid(1.0)`, so a non-spherical `distribution_shape` was
#  silently ignored and the ALV answer diverged from the elastic one.  Every
#  pre-existing test used the spherical default, which is why it went unseen.
# ─────────────────────────────────────────────────────────────────────────────
@testset "Maxwell/PCW ALV honour distribution_shape" begin
    C_M = TensISO{3}(3 * 5.0, 2 * 2.0)
    C_I = TensISO{3}(3 * 30.0, 2 * 12.0)
    times = collect(range(0.0, 1.0; length = 5))
    n = length(times)

    function build(dist; alv)
        rve = RVE(; distribution_shape = UniformDistribution(dist))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => alv ? heaviside_law(C_M) : C_M); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0, 1.0, 0.1),
            Dict(:C => alv ? heaviside_law(C_I) : C_I);
            fraction = 0.3, symmetrize = :iso
        )
        return rve
    end

    # Unit strain step: the effective modulus is the row sum of the ALV blocks.
    step_response(R, comp) =
        sum(R[6 * (n - 1) + comp, 6 * (j - 1) + comp] for j in 1:n)

    for dist in (Ellipsoid(1.0), Ellipsoid(1.0, 1.0, 0.1))
        for sch in (Maxwell(), PonteCastanedaWillis())
            C_el = TensND.components_canon(homogenize(build(dist; alv = false), sch, :C))
            R_alv = homogenize_alv(build(dist; alv = true), sch, :C; times = times)
            @test isapprox(step_response(R_alv, 1), C_el[1, 1, 1, 1]; rtol = 1.0e-10)
            @test isapprox(step_response(R_alv, 3), C_el[3, 3, 3, 3]; rtol = 1.0e-10)
        end
    end

    # An oblate envelope must give a different answer from a spherical one —
    # otherwise the field is being ignored again.
    R_sph = homogenize_alv(build(Ellipsoid(1.0); alv = true), Maxwell(), :C; times = times)
    R_obl = homogenize_alv(
        build(Ellipsoid(1.0, 1.0, 0.1); alv = true), Maxwell(), :C; times = times
    )
    @test !isapprox(step_response(R_sph, 1), step_response(R_obl, 1); rtol = 1.0e-3)
end
