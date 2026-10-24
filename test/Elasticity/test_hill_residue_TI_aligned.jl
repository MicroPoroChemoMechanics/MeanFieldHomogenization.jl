# The residue algorithm on references whose acoustic polynomial has multiple
# roots, or nearly: transversely isotropic or orthotropic matrices aligned with
# the ellipsoid, isotropic ones entered as general tensors, and matrices close
# to either. The residues of coinciding or nearly coinciding poles are summed by
# groups (`Core._cluster_weights`); each case is held to a cubature or to a
# closed form.
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
    C_TI = TensND.from_mandel(KM, basis)

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
    C_O = TensND.from_mandel(KM, basis)
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
    C_TI = TensND.from_mandel(KM, basis)

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
    C = TensND.from_mandel(KM, basis)
    ell = Ellipsoid(10.6, 1.2, 0.5)

    P_res = hill_tensor(ell, C; method = :residues)
    P_dec = hill_tensor(ell, C; method = :decuhr)

    diff = maximum(abs, components_canon(P_res - P_dec))
    @test diff < 1.0e-6
end

# A tetragonal stiffness about e₃, in the Kelvin-Mandel form `TensND.from_mandel`
# reads (shear terms doubled): transversely isotropic when C66 = (C11 - C12)/2,
# cubic when C33 = C11, C13 = C12 and C44 = C66.
function _ti_km(C11, C12, C13, C33, C44, C66)
    KM = [
        C11 C12 C13 0.0 0.0 0.0;
        C12 C11 C13 0.0 0.0 0.0;
        C13 C13 C33 0.0 0.0 0.0;
        0.0 0.0 0.0 2C44 0.0 0.0;
        0.0 0.0 0.0 0.0 2C44 0.0;
        0.0 0.0 0.0 0.0 0.0 2C66
    ]
    return TensND.from_mandel(KM, basis)
end

@testset "Hill residue — roots of multiplicity 2 and 3, and what is refused" begin
    sphere = Ellipsoid(1.0)
    # An isotropic matrix, entered as a general tensor so that the residues are
    # used: around a sphere `Q = Q₆ (z² + 1)³`, a triple root at i, which the
    # closed form of multiplicity 3 handles. The reference is Eshelby's tensor.
    k, μ = 30.0, 18.0
    λ = k - 2μ / 3
    C_iso = _ti_km(λ + 2μ, λ, λ, λ + 2μ, μ, μ)
    P_eshelby = hill_tensor(sphere, TensISO{3}(3k, 2μ))
    @test components_canon(hill_tensor(sphere, C_iso; method = :residues)) ≈
        components_canon(P_eshelby) rtol = 1.0e-12

    # Transversely isotropic about the sphere's axis. Along ζ = (cos φ, sin φ, z)
    # the wave polarized in the plane of isotropy vanishes at z² = -C66/C44, the
    # two coupled ones at the roots w = z² of
    #     C44 C33 w² + (C11 C33 + C44² - (C13 + C44)²) w + C11 C44 = 0.
    # (a) C44 = C66 puts the first at i, and (C13 + C44)² = (C11 - C44)(C33 - C44)
    #     one of the others too: a double root at i.
    C11, C33, C44 = 60.0, 45.0, 20.0
    C_a = _ti_km(C11, C11 - 2C44, sqrt((C11 - C44) * (C33 - C44)) - C44, C33, C44, C44)
    # (b) C66 ≠ C44 moves the first to z² = -C66/C44, and a C13 that makes the
    #     quadratic vanish there too puts a double root away from i.
    C44b, C66b = 15.0, 20.0
    w = -C66b / C44b
    B = -(C44b * C33 * w^2 + C11 * C44b) / w     # the middle coefficient it needs
    C_b = _ti_km(C11, C11 - 2C66b, sqrt(C11 * C33 + C44b^2 - B) - C44b, C33, C44b, C66b)
    for C in (C_a, C_b)
        @test components_canon(hill_tensor(sphere, C; method = :residues)) ≈
            components_canon(hill_tensor(sphere, C; method = :nestedquadgk)) rtol = 1.0e-9
    end

    # Around a spheroid the triple root of an isotropic matrix leaves i, and
    # around a triaxial ellipsoid it moves with the angle: Eshelby's tensor again.
    for ell in (Ellipsoid(1.0, 1.0, 0.5), Ellipsoid(2.0, 1.0, 0.5))
        @test components_canon(hill_tensor(ell, C_iso; method = :residues)) ≈
            components_canon(hill_tensor(ell, TensISO{3}(3k, 2μ))) rtol = 1.0e-12
    end
    # A reference without shear stiffness has no Green operator: refused, rather
    # than integrated on a polynomial cut down to a lower degree.
    @test_throws ArgumentError hill_tensor(sphere, _ti_km(30.0, 30.0, 30.0, 30.0, 0.0, 0.0); method = :residues)
    # Nor has one whose acoustic tensor changes sign between two directions, here
    # through (C13 + C44)² > (C11 - C44)(C33 - C44): a real root of the polynomial.
    @test_throws ArgumentError hill_tensor(sphere, _ti_km(10.0, 8.0, 30.0, 10.0, 1.0, 1.0); method = :residues)
end

@testset "Hill residue — roots that nearly coincide" begin
    # Close but distinct roots, whose residues are each large and nearly cancel:
    # summed by groups, they keep the accuracy of separated ones. Taken one at a
    # time or moved together, they lost up to 1e-2 here.
    tight = (; abstol = 1.0e-13, reltol = 1.0e-12)
    cases = [
        # Two roots near i, where the inner integrand has its double pole.
        (Ellipsoid(1.0), _ti_km(60.0, 60.0 - 40.0 * (1 + 1.0e-3), 15.0, 45.0, 20.0, 20.0 * (1 + 1.0e-3))),
        (Ellipsoid(1.0), _ti_km(60.0, 60.0 - 40.0 * (1 + 1.0e-6), 15.0, 45.0, 20.0, 20.0 * (1 + 1.0e-6))),
        # Three roots near the triple one of an isotropic matrix, away from i.
        (Ellipsoid(1.0, 1.0, 0.5), _ti_km(66.0 + 1.0e-4, 30.0, 30.0, 66.0 + 1.0e-4, 18.0, 18.0)),
        # A cubic matrix around a prolate spheroid.
        (Ellipsoid(3.0, 1.0, 1.0), _ti_km(237.0, 141.0, 141.0, 237.0, 116.0, 116.0)),
    ]
    for (ell, C) in cases
        @test components_canon(hill_tensor(ell, C; method = :residues, tight...)) ≈
            components_canon(hill_tensor(ell, C; method = :nestedquadgk, tight...)) rtol = 1.0e-10
    end
end
