using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  REGRESSION — the COD of a crack that is NOT aligned with the canonical axes.
#
#  Two bugs used to make `cod_tensor` wrong for a tilted crack in an
#  anisotropic reference medium, and both were invisible to a test suite built
#  on aligned cracks, because the aligned case coincides:
#
#  1. `TensND.change_tens(::TensTI, ℬ)` relabeled the stored (canonical)
#     components instead of rotating them, so the same physical medium gave a
#     different answer depending on whether it was stored as a `TensTI` or as a
#     generic `Tens`. Fixed in TensND.
#
#  2. The anisotropic COD kernels mixed two frames: the stiffness was rotated
#     into the crack basis while the crack's frame vectors were passed in
#     *global* coordinates. Fixed by `Cracks._crack_local_frame`, which returns
#     both in one frame.
#
#  The tests below pin the behavior three ways: an exact internal symmetry
#  (rotation invariance), storage independence, and an external cross-check
#  against ECHOES.
# =============================================================================

# TI medium used throughout, and its explicit components — the same numbers are
# fed to ECHOES to produce the reference values below.
#   C1111 = 17.5, C1122 = 12.5, C1133 = 2.82842712474619, C3333 = 20.0, C2323 = 4.0
const TI_PARAMS = (20.0, 30.0, 4.0, 5.0, 8.0)
_ti_medium(axis = (0.0, 0.0, 1.0)) = TensTI{4}(TI_PARAMS..., axis)

# Frame-independent invariants of a COD/compliance tensor. `B[i,j]` returns the
# components in the tensor's OWN basis, so a component-wise comparison across
# orientations is meaningless; the Frobenius norm and the full contraction are
# not.
_fro(t) = sqrt(sum(abs2, get_array(t)))
_contract(H) = (A = get_array(H); sum(A[i, j, i, j] for i in 1:3, j in 1:3))

# ── ECHOES reference ─────────────────────────────────────────────────────────
#
# Produced with the C++/Python ECHOES reference implementation:
#
#     from echoes import *
#     ref = stiff_TI(17.5, 12.5, 2.82842712474619, 20.0, 4.0, 0.0, 0.0)
#     H   = array(crack_compliance(ellipsoidal(1.0, 1.0, 0.0, θ, 0.0, 0.0), ref)).real
#     (linalg.norm(H), trace(H))          # Kelvin-Mandel 6x6 -> Frobenius, trace
#
# The Kelvin-Mandel map is an isometry, so `linalg.norm(H)` is the Frobenius
# norm of the 4th-order tensor and `trace(H)` is `H_ijij`. ECHOES runs with its
# default cubature tolerances (1e-4), which is what sets the 1e-5 rtol below.
const ECHOES_H = (
    (0.0, 0.1219187521, 0.2088525191),
    (π / 8, 0.1282112127, 0.2180423659),
    (π / 4, 0.1412296428, 0.2364110777),
    (3π / 8, 0.1520134297, 0.2519588658),
    (π / 2, 0.1557562607, 0.2577411702),
)

@testset "PREREQUISITE — TensND rotates structured tensors" begin
    # If this one fails, nothing below is meaningful and the cause is NOT in
    # MeanFieldHomogenization: the loaded TensND still has the `change_tens`
    # bug, where a `TensTI`/`TensOrtho` handed a rotated basis kept its
    # canonical components instead of transforming them. The crack kernels then
    # receive an unrotated stiffness and every tilted-crack result collapses
    # onto the aligned one.
    #
    # Watch out for the environment: `test/Manifest.toml` may resolve TensND
    # from the registry rather than from a local checkout, in which case a local
    # TensND fix is simply not loaded.
    C = _ti_medium()
    ℬrot = TensND.RotatedBasis(0.4, 0.7, 0.3)
    @test !isapprox(
        get_array(TensND.change_tens(C, ℬrot)), get_array(C); atol = 1.0e-8
    )
    # …and it must agree with the same tensor in generic storage, for which the
    # rotation has always been correct.
    @test get_array(TensND.change_tens(C, ℬrot)) ≈
        get_array(TensND.change_tens(Tens(get_array(C)), ℬrot)) atol = 1.0e-12
end

@testset "tilted crack — cross-check against ECHOES" begin
    C = _ti_medium()
    for (θ, fro_ref, contract_ref) in ECHOES_H
        crack = PennyCrack(1.0; euler_angles = (θ, 0.0))
        H = compliance_contribution(crack, C)      # `:auto`
        @test _fro(H) ≈ fro_ref rtol = 1.0e-5
        @test _contract(H) ≈ contract_ref rtol = 1.0e-5
    end
end

@testset "tilted crack — rotation invariance" begin
    # Tilting the crack by θ in a medium whose axis is e₃ is the same physical
    # problem as leaving the crack on e₃ and tilting the medium by -θ. This is
    # exact, and it is what the frame mixing broke.
    for θ in (0.0, π / 8, π / 4, 3π / 8, π / 2)
        H_crack = compliance_contribution(
            PennyCrack(1.0; euler_angles = (θ, 0.0)), _ti_medium()
        )
        H_medium = compliance_contribution(
            PennyCrack(1.0), _ti_medium((sin(-θ), 0.0, cos(-θ)))
        )
        @test _fro(H_crack) ≈ _fro(H_medium) rtol = 1.0e-8
        @test _contract(H_crack) ≈ _contract(H_medium) rtol = 1.0e-8
    end
end

@testset "rotating the whole problem changes nothing" begin
    # Medium axis and crack normal tilted TOGETHER: still an aligned
    # configuration, so every invariant must equal the canonical aligned one.
    # This is the case that `change_tens` used to corrupt, and it exercises the
    # closed-form TI branch (`_ti_aligned` → `Analytical`).
    H0 = compliance_contribution(PennyCrack(1.0), _ti_medium())
    for θ in (π / 8, π / 4, 3π / 8, π / 2)
        n = (sin(θ), 0.0, cos(θ))
        H = compliance_contribution(PennyCrack(1.0; euler_angles = (θ, 0.0)), _ti_medium(n))
        @test _fro(H) ≈ _fro(H0) rtol = 1.0e-8
        @test _contract(H) ≈ _contract(H0) rtol = 1.0e-8
    end
end

@testset "the answer does not depend on how the medium is stored" begin
    # `TensTI{4}` and a generic `Tens` holding the very same components are the
    # same tensor and must give the same COD, for every backend.
    C_ti = _ti_medium()
    C_generic = Tens(get_array(C_ti))
    @test get_array(C_generic) ≈ get_array(C_ti)

    for θ in (0.0, π / 6, π / 4, π / 3)
        crack = PennyCrack(1.0; euler_angles = (θ, 0.0))
        for method in (:auto, :nestedquadgk)
            H_ti = compliance_contribution(crack, C_ti; method = method)
            H_gen = compliance_contribution(crack, C_generic; method = method)
            @test _fro(H_ti) ≈ _fro(H_gen) rtol = 1.0e-9
            @test _contract(H_ti) ≈ _contract(H_gen) rtol = 1.0e-9
        end
    end
end

@testset "azimuth is honored too" begin
    # A TI medium is invariant about its axis, so the azimuth φ of the crack
    # normal cannot change any invariant of ℍ. A frame mix-up would break this.
    C = _ti_medium()
    θ = π / 4
    ref = compliance_contribution(PennyCrack(1.0; euler_angles = (θ, 0.0)), C)
    for φ in (π / 5, π / 2, π, 3π / 2)
        H = compliance_contribution(PennyCrack(1.0; euler_angles = (θ, φ)), C)
        @test _fro(H) ≈ _fro(ref) rtol = 1.0e-8
        @test _contract(H) ≈ _contract(ref) rtol = 1.0e-8
    end
end

@testset "ribbon crack — rotation invariance" begin
    # `_cod_ribbon_numerical` shares the frame convention, so it gets the same
    # guarantee.
    #
    # Pinned to `:nestedquadgk` rather than `:auto` on purpose: the DECUHR
    # cubature fails outright (`retcode = Failure`) on a *tilted ribbon*, and it
    # does so on the pristine tree too — a pre-existing robustness limitation of
    # that backend, independent of the frame convention tested here. `:auto`
    # selects DECUHR for this configuration when the extension is loaded, which
    # would make this test's outcome depend on whether `DECUHR` happens to be
    # imported.
    for θ in (0.0, π / 6, π / 4)
        H_crack = compliance_contribution(
            RibbonCrack(1.0; euler_angles = (θ, 0.0)), _ti_medium();
            method = :nestedquadgk
        )
        H_medium = compliance_contribution(
            RibbonCrack(1.0), _ti_medium((sin(-θ), 0.0, cos(-θ)));
            method = :nestedquadgk
        )
        @test _fro(H_crack) ≈ _fro(H_medium) rtol = 1.0e-7
    end

    # The frame fix also reconciled the two backends on a tilted ribbon: before
    # it they disagreed by ~57% at θ = π/4.
    for θ in (π / 6, π / 4)
        crack = RibbonCrack(1.0; euler_angles = (θ, 0.0))
        r = _fro(compliance_contribution(crack, _ti_medium(); method = :residues))
        q = _fro(compliance_contribution(crack, _ti_medium(); method = :nestedquadgk))
        @test r ≈ q rtol = 1.0e-8
    end
end

@testset "the residue backend degenerates as n̂ ⟂ TI axis" begin
    # KNOWN LIMITATION, not a frame bug. The `Residue` algorithm reduces the
    # crack-plane integral to a sum over the roots of a sextic; those roots
    # coalesce as the crack normal becomes perpendicular to the symmetry axis,
    # and the reduction loses accuracy and then breaks down.
    #
    # This is benign in practice because `:auto` never selects `Residue` for a
    # non-aligned crack (it hands over to a cubature) — which the first block
    # below pins — but an explicit `method = :residues` does hit it.
    C = _ti_medium()

    # `:auto` stays on a cubature and stays correct, including at θ = π/2.
    for (θ, fro_ref, _) in ECHOES_H
        crack = PennyCrack(1.0; euler_angles = (θ, 0.0))
        algo = MeanFieldHomogenization.Core._resolve_algo(Val(:auto), crack, C)
        θ == 0.0 || @test !(algo isa MeanFieldHomogenization.Core.Residue)
        @test _fro(compliance_contribution(crack, C; method = :auto)) ≈ fro_ref rtol = 1.0e-5
    end

    # Away from the degeneracy the residue reduction agrees with the cubature to
    # near machine precision — that is the regime it is meant for.
    for θ in (0.0, π / 8, π / 4, 0.4π)
        crack = PennyCrack(1.0; euler_angles = (θ, 0.0))
        r = _fro(compliance_contribution(crack, C; method = :residues))
        q = _fro(compliance_contribution(crack, C; method = :nestedquadgk))
        @test r ≈ q rtol = 1.0e-8
    end

    # …and it is documented as unreliable in the immediate neighborhood of
    # π/2. Guard the *documented* behavior so a future improvement shows up as
    # a failing test rather than going unnoticed.
    crack90 = PennyCrack(1.0; euler_angles = (π / 2, 0.0))
    r90 = _fro(compliance_contribution(crack90, C; method = :residues))
    q90 = _fro(compliance_contribution(crack90, C; method = :nestedquadgk))
    @test !isapprox(r90, q90; rtol = 1.0e-3)
end
