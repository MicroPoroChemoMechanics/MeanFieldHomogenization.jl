# =============================================================================
#  fixtures.jl — shared benchmark fixtures.
#
#  Every fixture is copied **verbatim** from an already-validated setup in
#  `test/` or `scripts/`.  Deliberately not simplified: a simplified extract
#  can behave differently from the real path (e.g. by never taking a branch
#  the real case takes), which would make the whole comparison worthless.
# =============================================================================

using LinearAlgebra

# ── Stiffnesses ─────────────────────────────────────────────────────────────

# Triclinic KM — test/regression/test_anisotropic.jl:26-33 (itself reused
# from test_residue_accuracy.jl).
const KM_TRI = [
    210.0 80.0 75.0 5.0 4.0 3.0;
    80.0 195.0 90.0 -2.0 3.0 -1.0;
    75.0 90.0 220.0 1.0 -2.0 2.0;
    5.0 -2.0 1.0 60.0 2.5 1.5;
    4.0 3.0 -2.0 2.5 65.0 -1.0;
    3.0 -1.0 2.0 1.5 -1.0 55.0
]

# Cubic (Fe-like) — test/regression/test_anisotropic.jl:36-43
const KM_CUBIC = [
    237.0 141.0 141.0 0.0 0.0 0.0;
    141.0 237.0 141.0 0.0 0.0 0.0;
    141.0 141.0 237.0 0.0 0.0 0.0;
    0.0 0.0 0.0 232.0 0.0 0.0;
    0.0 0.0 0.0 0.0 232.0 0.0;
    0.0 0.0 0.0 0.0 0.0 232.0
]

const CB3 = CanonicalBasis{3, Float64}()

C_tri() = TensND.inv_KM(KM_TRI, CB3)
C_cubic() = TensND.inv_KM(KM_CUBIC, CB3)

# Fully-anisotropic K₀ — test/regression/test_anisotropic.jl:46
K_aniso() = TensND.Tens([3.0 0.5 0.3; 0.5 2.0 0.2; 0.3 0.2 1.5])

# Isotropic moduli — scripts/28_porous_schemes.jl / test_symmetrize.jl:180-181
C_matrix_iso() = TensISO{3}(3 * 20.0, 2 * 12.0)
C_incl_iso() = TensISO{3}(3 * 80.0, 2 * 50.0)
C_solid_iso() = TensISO{3}(3 * 72.0, 2 * 32.0)
C_pore_iso() = TensISO{3}(3.0e-6, 2.0e-6)

# TI stiffness — scripts/bench/bench_alv.jl:80
C_ti() = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0.0, 0.0, 1.0))

# ── RVE builders ────────────────────────────────────────────────────────────

"""Two-phase isotropic RVE with spherical inclusions."""
function rve_iso2(; f = 0.3)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_matrix_iso()); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_incl_iso()); fraction = f)
    return rve
end

"""Anisotropic (triclinic) matrix + triaxial ellipsoid — forces the
anisotropic Hill branch inside a scheme."""
function rve_aniso_matrix(; f = 0.2)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_tri()); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(3.0, 2.0, 1.0), Dict(:C => C_incl_iso()); fraction = f)
    return rve
end

"""Porous oblate spheroid, both phases `symmetrize = :iso`
— test/Schemes/test_symmetrize.jl:114-131."""
function rve_porous_oblate_isosym(; f = 0.2)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_solid_iso()); fraction = :rest)
    add_phase!(
        rve, :P, Spheroid(0.2), Dict(:C => C_pore_iso());
        fraction = f, symmetrize = :iso
    )
    return rve
end

"""θ-binned TI(ez) family — test/Schemes/test_symmetrize.jl:173-206.
The flagship de-duplication case: `nθ` phases, each needing one Hill solve
that is currently computed twice."""
function rve_theta_binned(nθ::Int)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_matrix_iso()); fraction = :rest)
    ez = (0.0, 0.0, 1.0)
    edges = range(0, π / 2; length = nθ + 1)
    for k in 1:nθ
        θm, θp = edges[k], edges[k + 1]
        θ = (θm + θp) / 2
        w = cos(θm) - cos(θp)
        add_phase!(
            rve, Symbol(:B, k),
            Spheroid(5.0; euler_angles = (θ, 0.0, 0.0)), Dict(:C => C_incl_iso());
            fraction = 0.15 * w, symmetrize = TISymmetrize(ez)
        )
    end
    return rve
end

"""Three non-coaxial TI phases inside SC — test/Schemes/test_symmetrize.jl:208-240.
Exercises the third duplication instance (`_phase_stress_strain_average`)."""
function rve_multi_axis_ti(x = 1.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 20.0 * x, 2 * 12.0 * x)); fraction = :rest)
    for (k, θ) in enumerate((0.0, π / 4, π / 2))
        add_phase!(
            rve, Symbol(:I, k),
            Spheroid(5.0; euler_angles = (θ, 0.0, 0.0)), Dict(:C => C_incl_iso());
            fraction = 0.05,
            symmetrize = TISymmetrize((sin(θ), 0.0, cos(θ)))
        )
    end
    return rve
end

"""Penny-crack RVE — test/Schemes/test_one_shot.jl:71-86 style."""
function rve_crack(; C_m = C_solid_iso(), density = 0.1)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :Cr, PennyCrack(1.0), Dict(:C => C_pore_iso()); density = density)
    return rve
end

"""Porous RVE for the self-consistent percolation cases — scripts/28."""
function rve_porous_sc(; φ = 0.3, shape = 1.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_solid_iso()); fraction = :rest)
    add_phase!(rve, :P, Spheroid(shape), Dict(:C => C_pore_iso()); fraction = φ)
    return rve
end

const SC_OPTS = (; abstol = 1.0e-10, reltol = 1.0e-10, maxiters = 300, select_best = true)
