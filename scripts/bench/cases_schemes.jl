# =============================================================================
#  cases_schemes.jl — homogenization schemes and ForwardDiff sensitivities.
#
#  Mori-Tanaka / SelfConsistent / AsymmetricSelfConsistent(stiffness) are the
#  schemes touched by the de-duplication tier.  Dilute, DiluteDual, Maxwell,
#  PCW, Differential, Voigt and Reuss call only ONE of the two duplicated
#  helpers, so they are exact controls for that tier.
# =============================================================================

# ── Mori-Tanaka ─────────────────────────────────────────────────────────────

bcase(
    "schemes/mt.iso2.sphere";
    group = :schemes, tags = [:mt, :dedup],
    setup = () -> rve_iso2(),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

bcase(
    "schemes/mt.porous.oblate.isosym";
    group = :schemes, tags = [:mt, :dedup, :symmetrize],
    setup = () -> rve_porous_oblate_isosym(),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

# Flagship de-duplication case: 20 phases × 2 Hill solves → 20.
bcase(
    "schemes/mt.theta_binned_ti.n20";
    group = :schemes, tags = [:mt, :dedup, :symmetrize],
    setup = () -> rve_theta_binned(20),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

# Largest absolute win: anisotropic matrix ⇒ each Hill solve is a full
# adaptive quadrature.
bcase(
    "schemes/mt.aniso_matrix";
    group = :schemes, tags = [:mt, :dedup, :quadrature],
    setup = () -> rve_aniso_matrix(),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

bcase(
    "schemes/mt.crack.penny";
    group = :schemes, tags = [:mt, :dedup, :cod],
    setup = () -> rve_crack(),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

bcase(
    "schemes/mt.crack.penny.tri";
    group = :schemes, tags = [:mt, :dedup, :cod],
    setup = () -> rve_crack(; C_m = C_tri()),
    body = rve -> homogenize(rve, MoriTanaka(), :C),
)

bcase(
    "schemes/mt.conductivity.iso2";
    group = :schemes, tags = [:mt, :dedup, :conductivity],
    setup = () -> begin
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(20.0)); fraction = 0.25)
        rve
    end,
    body = rve -> homogenize(rve, MoriTanaka(), :K),
)

# ── Self-consistent ─────────────────────────────────────────────────────────

bcase(
    "schemes/sc.porous.sphere.phi30";
    group = :schemes, tags = [:sc, :dedup],
    setup = () -> rve_porous_sc(; φ = 0.30, shape = 1.0),
    body = rve -> homogenize(rve, SelfConsistent(; SC_OPTS...), :C),
)

bcase(
    "schemes/sc.porous.oblate.phi15";
    group = :schemes, tags = [:sc, :dedup],
    setup = () -> rve_porous_sc(; φ = 0.15, shape = 0.1),
    body = rve -> homogenize(rve, SelfConsistent(; SC_OPTS...), :C),
)

# Third duplication instance (`_phase_stress_strain_average` slow branch).
bcase(
    "schemes/sc.multi_axis_ti";
    group = :schemes, tags = [:sc, :dedup, :symmetrize],
    setup = () -> rve_multi_axis_ti(1.0),
    body = rve -> homogenize(rve, SelfConsistent(), :C),
)

bcase(
    "schemes/sc.newton";
    group = :schemes, tags = [:sc, :newton],
    setup = () -> rve_porous_sc(; φ = 0.30, shape = 1.0),
    body = rve -> homogenize(
        rve, SelfConsistent(; algorithm = NewtonDefault(), SC_OPTS...), :C
    ),
)

bcase(
    "schemes/asc.stiffness";
    group = :schemes, tags = [:asc, :dedup],
    setup = () -> rve_iso2(),
    body = rve -> homogenize(rve, AsymmetricSelfConsistent(; SC_OPTS...), :C),
)

# ── ForwardDiff sensitivities ───────────────────────────────────────────────

bcase(
    "sens/mt.dC_df";
    group = :schemes, tags = [:forwarddiff, :mt],
    setup = () -> rve_iso2(),
    body = rve -> derivative(
        rve, MoriTanaka(), amount(:I);
        indexer = C -> TensND.get_array(C)[1, 1, 1, 1]
    ),
    checksum = r -> Float64[r],
)

bcase(
    "sens/sc.dC_dCi";
    group = :schemes, tags = [:forwarddiff, :sc],
    setup = () -> rve_porous_sc(; φ = 0.2, shape = 1.0),
    body = rve -> derivative(
        rve, SelfConsistent(; SC_OPTS...), property(:M, :C, :bulk);
        indexer = C -> TensND.get_array(C)[1, 1, 1, 1]
    ),
    checksum = r -> Float64[r],
)

# Dual through NestedQuadGK inside a multi-axis SC — test_symmetrize.jl:236.
bcase(
    "sens/sc.multi_axis_ti.dual";
    group = :schemes, tags = [:forwarddiff, :sc, :symmetrize],
    setup = () -> nothing,
    body = _ -> FD.derivative(
        x -> TensND.get_array(homogenize(rve_multi_axis_ti(x), SelfConsistent(), :C))[3, 3, 3, 3],
        1.0
    ),
    checksum = r -> Float64[r],
)

# ── Controls: schemes untouched by the de-duplication tier ──────────────────

for (nm, sch) in (
        ("voigt", Voigt()), ("reuss", Reuss()),
        ("dilute", Dilute()), ("dilute_dual", DiluteDual()),
        ("maxwell", Maxwell()), ("differential", DifferentialScheme()),
    )
    bcase(
        "control/$nm.iso2";
        group = :control, tags = [:scheme], control = true,
        setup = () -> rve_iso2(),
        body = rve -> homogenize(rve, sch, :C),
    )
end
