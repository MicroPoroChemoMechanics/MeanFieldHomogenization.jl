# =============================================================================
#  cases_alv.jl — ageing linear viscoelasticity.
#
#  Everything here is O(n²) or worse in the number of time points, so the
#  cases are the ones that expose the Volterra / trapezoidal buffer
#  allocations and the per-block TI-symmetrize operator rebuild.
#
#  The RVE builder is a verbatim copy of `_build_setup` in `bench_alv.jl`
#  (same laws, same fractions, same shapes) so the two scripts describe the
#  same physics.
# =============================================================================

"""ALV setup — verbatim from `scripts/bench/bench_alv.jl:50-76`."""
function alv_setup(n::Int; transversely_isotropic::Bool = false)
    times = collect(range(0.0, 5.0; length = n))

    function R_iso(t, tp)
        α = 3 * (1.0 + 4.0 * exp(-(t - tp) / 1.0))
        β = 2 * (0.5 + 1.5 * exp(-(t - tp) / 0.5))
        return TensISO{3}(α, β)
    end
    law_M = ViscoLaw(R_iso, :relaxation)

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)

    C_I_t = TensISO{3}(60.0, 20.0)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 0.5),
        Dict(:C => heaviside_law(C_I_t)); fraction = 0.2
    )

    if transversely_isotropic
        ℓ_I = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0.0, 0.0, 1.0))
        add_phase!(
            rve, :J, Spheroid(1.5),
            Dict(:C => heaviside_law(ℓ_I)); fraction = 0.05
        )
    end
    return (; rve, times, law_M)
end

for n in (50, 100)
    bcase(
        "alv/trapezoidal.n$n";
        group = :alv, tags = [:alv, :trapezoidal],
        setup = () -> alv_setup(n),
        body = ctx -> trapezoidal_matrix(ctx.law_M, ctx.times),
    )

    bcase(
        "alv/volterra_inverse.n$n";
        group = :alv, tags = [:alv, :volterra],
        setup = () -> begin
            ctx = alv_setup(n)
            trapezoidal_matrix(ctx.law_M, ctx.times)
        end,
        body = M -> volterra_inverse(M; block_size = 6),
    )

    bcase(
        "alv/mt.n$n";
        group = :alv, tags = [:alv, :mt],
        setup = () -> alv_setup(n),
        body = ctx -> homogenize_alv(ctx.rve, MoriTanaka(), :C; times = ctx.times),
    )
end

# Transversely isotropic variant — drives `_ti_project_blocks`, i.e. the
# per-block rebuild of a loop-invariant 6×6 projection operator.
bcase(
    "alv/mt.ti.n50";
    group = :alv, tags = [:alv, :mt, :symmetrize],
    setup = () -> alv_setup(50; transversely_isotropic = true),
    body = ctx -> homogenize_alv(ctx.rve, MoriTanaka(), :C; times = ctx.times),
)

# ── Controls ────────────────────────────────────────────────────────────────

bcase(
    "control/alv.voigt.n50";
    group = :control, tags = [:alv], control = true,
    setup = () -> alv_setup(50),
    body = ctx -> homogenize_alv(ctx.rve, Voigt(), :C; times = ctx.times),
)
