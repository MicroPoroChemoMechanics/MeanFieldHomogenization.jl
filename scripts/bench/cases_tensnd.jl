# =============================================================================
#  cases_tensnd.jl — TensND primitives.
#
#  Every MeanFieldHomogenization case is downstream of these, so a TensND-side change
#  should show up here first and in the scheme cases second — that pair is
#  the cross-check that a measured gain is real.
#
#  `TensOrtho` gets `getindex` / `Array` / `⊡` cases because it is the only
#  structured type still routed through the dense `get_array` path.
# =============================================================================

_iso4() = TensISO{3}(3 * 20.0, 2 * 12.0)
_ti4() = C_ti()
_ortho4() = TensND.TensOrtho(
    220.0, 80.0, 75.0, 195.0, 90.0, 210.0, 60.0, 65.0, 55.0,
    TensND.CanonicalBasis{3, Float64}()
)
_gen4() = C_tri()

# ── getindex / get_array on each structured type ────────────────────────────

for (nm, f) in (("iso", _iso4), ("ti", _ti4), ("ortho", _ortho4))
    bcase(
        "tensnd/getindex.$nm";
        group = :tensnd, tags = [:getindex],
        setup = f,
        body = t -> t[1, 1, 1, 1] + t[1, 2, 1, 2] + t[3, 3, 3, 3],
        checksum = r -> Float64[r],
    )
    bcase(
        "tensnd/get_array.$nm";
        group = :tensnd, tags = [:getindex],
        setup = f,
        body = t -> TensND.get_array(t),
    )
    # Generic AbstractArray traversal — the O(81²) blow-up when `getindex`
    # materializes the whole array on every scalar access.
    bcase(
        "tensnd/collect.$nm";
        group = :tensnd, tags = [:getindex],
        setup = f,
        body = t -> sum(abs, Array(t)),
        checksum = r -> Float64[r],
    )
end

# ── dcontract for each structured pair ──────────────────────────────────────

for (nm, fa, fb) in (
        ("iso_iso", _iso4, _iso4),
        ("ti_ti", _ti4, _ti4),
        ("ortho_ortho", _ortho4, _ortho4),
        ("iso_ortho", _iso4, _ortho4),
        ("gen_gen", _gen4, _gen4),
    )
    bcase(
        "tensnd/dcontract.$nm";
        group = :tensnd, tags = [:dcontract],
        setup = () -> (fa(), fb()),
        body = ctx -> ctx[1] ⊡ ctx[2],
    )
end

# ── inv ─────────────────────────────────────────────────────────────────────

for (nm, f) in (("iso", _iso4), ("ti", _ti4), ("ortho", _ortho4), ("gen", _gen4))
    bcase(
        "tensnd/inv.$nm";
        group = :tensnd, tags = [:inv],
        setup = f,
        body = t -> inv(t),
    )
end

# ── Rotation-group averages (the ALV TI-symmetrize hot spot) ────────────────

bcase(
    "tensnd/isotropify.gen";
    group = :tensnd, tags = [:average],
    setup = _gen4,
    body = t -> MFHC.isotropify(t),
)

bcase(
    "tensnd/transverse_isotropify.gen";
    group = :tensnd, tags = [:average],
    setup = _gen4,
    body = t -> MFHC.transverse_isotropify(t, (0.0, 0.0, 1.0)),
)

bcase(
    "tensnd/ti_average_mandel66";
    group = :tensnd, tags = [:average, :alv],
    setup = () -> TensND.mandel(_gen4()),
    body = M -> MFHC.ti_average_mandel66(M, (0.0, 0.0, 1.0)),
)

# ── KM round-trip ───────────────────────────────────────────────────────────

bcase(
    "tensnd/KM.gen";
    group = :tensnd, tags = [:km],
    setup = _gen4,
    body = t -> TensND.mandel(t),
)

bcase(
    "tensnd/inv_KM.gen";
    group = :tensnd, tags = [:km],
    setup = () -> KM_TRI,
    body = M -> TensND.from_mandel(M, CB3),
)
