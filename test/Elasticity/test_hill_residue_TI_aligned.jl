# PR-2 validation: residue algorithm robustness for TI / ORTHO matrices
# aligned with the ellipsoid principal axes (mirror of the C++ test
# tests/python/echoes_tests/hill_residues_TI_aligned.py).
#
# Before this PR the residue path silently dropped the contribution of
# multiple roots (PolynomialRoots Newton polish skipped them), producing
# wrong Hill tensor values. The new path (a) detects multiplicities via the
# scale-relative polynomial-derivative criterion in
# `_gather_almost_multiple_roots`, (b) uses the multiplicity-2 / -3 residue
# formulas ported from polynoms.h, and (c) when multiplicity exceeds the
# implemented formulas (or PolynomialRoots fails to converge), silently
# falls back to the DECUHR backend.
using Test
using MeanFieldHomogenization
using TensND

const ATOL = 5.0e-7
const basis = TensND.CanonicalBasis{3, Float64}()


@testset "Hill residue — TI matrix aligned with ellipsoid axis" begin
    # Cortical-bone-like TI stiffness with axis along e3.
    C11, C33, C12, C13, C44 = 18.0, 28.0, 9.98, 10.1, 6.23
    C66 = (C11 - C12) / 2
    KM = [
        C11 C12 C13 0.0   0.0   0.0;
        C12 C11 C13 0.0   0.0   0.0;
        C13 C13 C33 0.0   0.0   0.0;
        0.0 0.0 0.0 2C44  0.0   0.0;
        0.0 0.0 0.0 0.0   2C44  0.0;
        0.0 0.0 0.0 0.0   0.0   2C66
    ]
    C_TI = TensND.inv_KM(KM, basis)

    ell = Ellipsoid(2.0, 2.0, 1.0)        # spheroid, axis along e3

    P_res = hill_tensor(ell, C_TI; method = :residues)
    P_dec = hill_tensor(ell, C_TI; method = :decuhr)

    diff = maximum(abs, components_canon(P_res - P_dec))
    @test diff < ATOL
    @test all(isfinite, P_res)
end


@testset "Hill residue — ORTHO matrix aligned with ellipsoid axes" begin
    # Three distinct simple roots — no multiplicity expected.
    KM = [
        110.0 35.0 30.0 0.0  0.0  0.0;
        35.0  95.0 28.0 0.0  0.0  0.0;
        30.0  28.0 130.0 0.0 0.0  0.0;
        0.0   0.0  0.0   160.0 0.0 0.0;
        0.0   0.0  0.0   0.0   150.0 0.0;
        0.0   0.0  0.0   0.0   0.0   120.0
    ]
    C_O = TensND.inv_KM(KM, basis)
    ell = Ellipsoid(3.0, 1.5, 1.0)

    P_res = hill_tensor(ell, C_O; method = :residues)
    P_dec = hill_tensor(ell, C_O; method = :decuhr)

    diff = maximum(abs, components_canon(P_res - P_dec))
    @test diff < ATOL
end


@testset "Hill residue — TI near-aligned (5deg tilt, no multiplicity)" begin
    # Ellipsoid axis tilted 5° away from matrix symmetry axis: should not
    # produce exact multiplicity, residue path stays accurate.
    C11, C33, C12, C13, C44 = 18.0, 28.0, 9.98, 10.1, 6.23
    C66 = (C11 - C12) / 2
    KM = [
        C11 C12 C13 0.0  0.0  0.0;
        C12 C11 C13 0.0  0.0  0.0;
        C13 C13 C33 0.0  0.0  0.0;
        0.0 0.0 0.0 2C44 0.0  0.0;
        0.0 0.0 0.0 0.0  2C44 0.0;
        0.0 0.0 0.0 0.0  0.0  2C66
    ]
    C_TI = TensND.inv_KM(KM, basis)

    ell = Ellipsoid(2.0, 2.0, 1.0; euler_angles = (deg2rad(5.0), 0.0, 0.0))

    P_res = hill_tensor(ell, C_TI; method = :residues)
    P_dec = hill_tensor(ell, C_TI; method = :decuhr)

    diff = maximum(abs, components_canon(P_res - P_dec))
    @test diff < ATOL
end


@testset "Hill residue — generic anisotropic regression" begin
    # Generic triclinic stiffness — case that worked before PR-2, must
    # still work to guard against regression.
    KM = [
        210.0  80.0  75.0   5.0  4.0   3.0;
        80.0 195.0  90.0  -2.0  3.0  -1.0;
        75.0  90.0 220.0   1.0 -2.0   2.0;
        5.0  -2.0   1.0 120.0  5.0   3.0;
        4.0   3.0  -2.0   5.0 130.0 -2.0;
        3.0  -1.0   2.0   3.0 -2.0 110.0
    ]
    C = TensND.inv_KM(KM, basis)
    ell = Ellipsoid(10.6, 1.2, 0.5)

    P_res = hill_tensor(ell, C; method = :residues)
    P_dec = hill_tensor(ell, C; method = :decuhr)

    diff = maximum(abs, components_canon(P_res - P_dec))
    @test diff < 1.0e-6
end
