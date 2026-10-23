# =============================================================================
#  cases_kernels.jl — Hill / COD kernels and the per-node primitives.
#
#  These are the cases moved by the quadrature-kernel and StaticArrays tiers.
#  The `control/*` cases go through analytical branches that none of the
#  planned changes touch; they calibrate the noise floor.
# =============================================================================

# ── Per-node primitive (100 ns tier — only BenchmarkTools can resolve it) ────

bcase(
    "kernels/sym3_inv_acoustic";
    group = :kernels, tags = [:pernode, :quadrature],
    setup = () -> (TensND.components_canon(C_tri()), [0.3, 0.5, 0.8]),
    body = ctx -> MFH.Elasticity._sym3_inv_acoustic(ctx[1], ctx[2]),
)

# ── Hill tensor, anisotropic matrix, three back-ends ────────────────────────

for (tag, m) in (("nqgk", :nestedquadgk), ("residues", :residues))
    bcase(
        "kernels/hill.$tag.tri.321";
        group = :kernels, tags = [:hill, :quadrature, Symbol(m)],
        setup = () -> (Ellipsoid(3.0, 2.0, 1.0), C_tri()),
        body = ctx -> hill_tensor(ctx[1], ctx[2]; method = m),
    )
end

bcase(
    "kernels/hill.decuhr.tri.321";
    group = :kernels, tags = [:hill, :quadrature, :decuhr],
    setup = () -> (Ellipsoid(3.0, 2.0, 1.0), C_tri()),
    body = ctx -> hill_tensor(ctx[1], ctx[2]; method = :decuhr, reltol = 1.0e-10),
    skip_if = () -> Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationDECUHRExt) === nothing,
)

# Oblate spheroid: exercises the `α = max(1, -log10(ω))` change of variable.
bcase(
    "kernels/hill.nqgk.oblate.tri";
    group = :kernels, tags = [:hill, :quadrature],
    setup = () -> (Spheroid(0.05), C_tri()),
    body = ctx -> hill_tensor(ctx[1], ctx[2]; method = :nestedquadgk),
)

bcase(
    "kernels/hill.cubic.residues";
    group = :kernels, tags = [:hill, :residues],
    setup = () -> (Ellipsoid(3.0, 2.0, 1.0), C_cubic()),
    body = ctx -> hill_tensor(ctx[1], ctx[2]; method = :residues),
)

# Conductivity (2nd-order) Hill with a fully anisotropic K₀.
bcase(
    "kernels/hill2.aniso";
    group = :kernels, tags = [:hill, :conductivity],
    setup = () -> (Ellipsoid(3.0, 2.0, 1.0), K_aniso()),
    body = ctx -> hill_tensor(ctx[1], ctx[2]),
)

# ForwardDiff through the `:auto → NestedQuadGK` Dual fallback.
bcase(
    "kernels/hill.dual.nqgk.tri";
    group = :kernels, tags = [:hill, :forwarddiff],
    setup = () -> (Ellipsoid(3.0, 2.0, 1.0), C_tri()),
    body = ctx -> FD.derivative(
        x -> TensND.get_array(hill_tensor(ctx[1], x * ctx[2]))[1, 1, 1, 1], 1.0
    ),
    checksum = r -> Float64[r],
)

# ── COD / crack kernels ─────────────────────────────────────────────────────

bcase(
    "kernels/cod.residues.penny.tri";
    group = :kernels, tags = [:cod, :residues],
    setup = () -> (PennyCrack(1.0), C_tri()),
    body = ctx -> cod_tensor(ctx[1], ctx[2]; method = :residues),
)

bcase(
    "kernels/cod.nqgk.ellipse03.tri";
    group = :kernels, tags = [:cod, :quadrature],
    setup = () -> (EllipticCrack(1.0, 0.3), C_tri()),
    body = ctx -> cod_tensor(ctx[1], ctx[2]; method = :nestedquadgk),
)

bcase(
    "kernels/H.compliance.penny.tri";
    group = :kernels, tags = [:cod, :residues],
    setup = () -> (PennyCrack(1.0), C_tri()),
    body = ctx -> inverse_property_contribution(ctx[1], ctx[2]; method = :residues),
)

# ── Controls: analytical branches, must not move ────────────────────────────

bcase(
    "control/hill.iso.sphere";
    group = :control, tags = [:hill], control = true,
    setup = () -> (Ellipsoid(1.0), TensISO{3}(216.0, 64.0)),
    body = ctx -> hill_tensor(ctx[1], ctx[2]),
)

bcase(
    "control/hill.iso.oblate";
    group = :control, tags = [:hill], control = true,
    setup = () -> (Spheroid(0.2), TensISO{3}(216.0, 64.0)),
    body = ctx -> hill_tensor(ctx[1], ctx[2]),
)

bcase(
    "control/hill.ti_coaxial";
    group = :control, tags = [:hill], control = true,
    setup = () -> (Spheroid(0.2), C_ti()),
    body = ctx -> hill_tensor(ctx[1], ctx[2]),
)

bcase(
    "control/elliptic.ellKE";
    group = :control, tags = [:elliptic], control = true,
    setup = () -> 0.7,
    body = m -> (MFH.Elliptic.ell_K(m), MFH.Elliptic.ell_E(m)),
)
