# =============================================================================
#  cases_common.jl: the cases common to MFH, echoes and tfel.
#
#  The functional comparison of October 2026 timed the features the three
#  libraries share. Unlike the other files of the suite, the bodies below build
#  their inputs (shape, RVE) inside the timed region: the comparison measured a
#  call as a user makes it, and object construction turned out to dominate the
#  cheap cases, so it is exactly what an optimization of these paths must move.
# =============================================================================

const COMMON_C0 = iso_stiffness(30.0, 10.0)
const COMMON_C1 = iso_stiffness(150.0, 80.0)

# Orthotropic stiffness of the comparison (Mandel); equal shear entries, so the
# shear ordering convention of the three libraries does not matter.
const COMMON_KM_ORTHO = [
    200.0 30.0 20.0 0.0 0.0 0.0;
    30.0 50.0 25.0 0.0 0.0 0.0;
    20.0 25.0 80.0 0.0 0.0 0.0;
    0.0 0.0 0.0 80.0 0.0 0.0;
    0.0 0.0 0.0 0.0 80.0 0.0;
    0.0 0.0 0.0 0.0 0.0 80.0
]
common_C_ortho() = TensND.from_mandel(COMMON_KM_ORTHO, CB3)

# ── Shapes and Hill tensors in an isotropic matrix ─────────────────────────

bcase(
    "common/ctor.spheroid5";
    group = :common, tags = [:common, :ctor],
    setup = () -> 5.0,
    body = ω -> Spheroid(ω),
    checksum = ell -> Float64[ell.semi_axes..., vec(Matrix(TensND.vecbasis(ell.basis, :cov)))...],
)

bcase(
    "common/hill.sphere.iso";
    group = :common, tags = [:common, :hill],
    setup = () -> COMMON_C0,
    body = C -> hill_tensor(Ellipsoid(1.0), C),
)

bcase(
    "common/hill.spheroid5.iso";
    group = :common, tags = [:common, :hill],
    setup = () -> COMMON_C0,
    body = C -> hill_tensor(Spheroid(5.0), C),
)

bcase(
    "common/hill.ellipsoid.iso";
    group = :common, tags = [:common, :hill],
    setup = () -> COMMON_C0,
    body = C -> hill_tensor(Ellipsoid(1.0, 0.5, 0.2), C),
)

# ── Hill tensor in an orthotropic matrix ───────────────────────────────────

bcase(
    "common/hill.ellipsoid.ortho";
    group = :common, tags = [:common, :hill, :quadrature],
    setup = () -> common_C_ortho(),
    body = C -> hill_tensor(Ellipsoid(1.0, 0.5, 0.2), C),
)

# ── Schemes, RVE rebuilt at every call ─────────────────────────────────────

function common_rve(spheres::Bool)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => COMMON_C0); fraction = :rest)
    if spheres
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => COMMON_C1); fraction = 0.3)
    else
        add_phase!(
            rve, :I, Spheroid(5.0), Dict(:C => COMMON_C1);
            fraction = 0.3, symmetrize = IsoSymmetrize()
        )
    end
    return rve
end

bcase(
    "common/rve.build";
    group = :common, tags = [:common, :rve],
    setup = () -> true,
    body = common_rve,
    checksum = _ -> Float64[],
)

for (tag, spheres) in (("spheres", true), ("spheroids5", false))
    bcase(
        "common/mt.$tag";
        group = :common, tags = [:common, :schemes],
        setup = () -> spheres,
        body = s -> homogenize(common_rve(s), MoriTanaka(), :C),
    )
    bcase(
        "common/asc.$tag";
        group = :common, tags = [:common, :schemes, :sc],
        setup = () -> spheres,
        body = s -> homogenize(
            common_rve(s), AsymmetricSelfConsistent(; reltol = 1.0e-10, abstol = 0.0), :C
        ),
    )
end
