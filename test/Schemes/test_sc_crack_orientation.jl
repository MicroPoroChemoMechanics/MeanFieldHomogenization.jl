using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  REGRESSION — self-consistent scheme with NON-COAXIAL crack families.
#
#  `SelfConsistent` used to abort with
#      DomainError: integrand produced NaN in the interval (0.0, 1.5e-152)
#  for *any* RVE holding two crack families that are not parallel, at any
#  density — including densities far below percolation.
#
#  Cause: the SC body assembles the estimate as `𝔹_E : 𝔸_E⁻¹`, a product of two
#  tensors that do not commute. With non-coaxial families the product is not
#  major-symmetric, the asymmetry is amplified by the fixed point (3e-15 at the
#  second iterate, 2e-4 at the third), and a stiffness without major symmetry is
#  not a valid Eshelby reference medium: the anisotropic crack cubature returns a
#  NaN integrand on it. `Schemes._major_symmetrize` now projects each iterate
#  back onto the admissible set, which is also what the reference ECHOES
#  implementation produces (its estimate is exactly major-symmetric).
# =============================================================================

const CS_SCO = TensISO{3}(3 * 30.0, 2 * 18.0)
const CANON_SCO = TensND.CanonicalBasis{3, Float64}()

function _cracked_rve(fams)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_SCO))
    for (i, (θ, d)) in enumerate(fams)
        add_phase!(
            rve, Symbol("F", i), PennyCrack(1.0; euler_angles = (θ, 0.0)),
            Dict(:C => CS_SCO); density = d
        )
    end
    return rve
end

# Components in the CANONICAL frame. Indexing a TensND tensor returns its
# *own*-basis components, and a scheme fed rotated crack families returns an
# estimate carried in a rotated basis — comparing raw components across
# orientations (or against another code) is meaningless without this.
_global(C) = get_array(TensND.change_tens(C, CANON_SCO))

_major_asym(C) = (
    A = _global(C);
    maximum(
        abs(A[i, j, k, l] - A[k, l, i, j])
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    )
)

# ── ECHOES reference ─────────────────────────────────────────────────────────
#
#     from echoes import *
#     Cs = stiff_kmu(30.0, 18.0)
#     r  = rve(matrix="SOLID")
#     r["SOLID"] = ellipsoid(shape=ellipsoidal(1.,1.,1.), prop={"C": Cs})
#     r["F0"]    = crack(shape=ellipsoidal(1.,1.,1e-7, θ, 0., 0.),
#                        density=d, prop={"C": tZ4})
#     homogenize(rve=r, scheme=SC, prop="C", epsrel=1e-10, maxnb=2000).array
#
# ECHOES needs a small but NON-ZERO aspect ratio (1e-7 here) where MFH takes the
# exact ω → 0 limit, which is what sets the 1e-4 rtol on the densest case.
# Values are (C₁₁₁₁, C₃₃₃₃) in the canonical frame.
const ECHOES_SC = (
    (:one_aligned, [(0.0, 0.01)], (53.656353, 50.907175)),
    (:two_tilted, [(0.0, 0.01), (π / 4, 0.01)], (51.899760, 49.286233)),
    (:arma3, [(0.0, 0.10), (π / 4, 0.25), (π / 4, 0.25)], (20.238537, 15.537175)),
)

const ECHOES_MT = (
    (:one_aligned, [(0.0, 0.01)], (53.660377, 50.943395)),
    (:two_tilted, [(0.0, 0.01), (π / 4, 0.01)], (51.933661, 49.372944)),
    (:arma3, [(0.0, 0.10), (π / 4, 0.25), (π / 4, 0.25)], (24.863175, 19.883190)),
)

@testset "SC converges on non-coaxial crack families" begin
    # The bug fired at any density; 0.01 is far below any percolation threshold,
    # so a failure here is structural rather than physical.
    for fams in ([(0.0, 0.01), (π / 4, 0.01)], [(0.0, 0.05), (π / 3, 0.05), (π / 6, 0.05)])
        C = homogenize(_cracked_rve(fams), SelfConsistent())
        @test all(isfinite, _global(C))
    end
end

@testset "the SC estimate is major-symmetric" begin
    sc = SelfConsistent(; abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 3000)
    for (_, fams, _) in ECHOES_SC
        C = homogenize(_cracked_rve(fams), sc)
        @test _major_asym(C) < 1.0e-12
    end
    # …and so is Mori-Tanaka, which never had the problem — a guard against the
    # projection being applied in the wrong place.
    for (_, fams, _) in ECHOES_MT
        @test _major_asym(homogenize(_cracked_rve(fams), MoriTanaka())) < 1.0e-12
    end
end

@testset "SC and MT agree with ECHOES on tilted families" begin
    sc = SelfConsistent(; abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 3000)
    for (name, fams, (c11, c33)) in ECHOES_SC
        A = _global(homogenize(_cracked_rve(fams), sc))
        rtol = name === :arma3 ? 1.0e-4 : 1.0e-6
        @test A[1, 1, 1, 1] ≈ c11 rtol = rtol
        @test A[3, 3, 3, 3] ≈ c33 rtol = rtol
    end
    for (name, fams, (c11, c33)) in ECHOES_MT
        A = _global(homogenize(_cracked_rve(fams), MoriTanaka()))
        rtol = name === :arma3 ? 1.0e-4 : 1.0e-6
        @test A[1, 1, 1, 1] ≈ c11 rtol = rtol
        @test A[3, 3, 3, 3] ≈ c33 rtol = rtol
    end
end

@testset "a single tilted family respects the mirror symmetry" begin
    # A crack family whose normal lies at 45° in the x-z plane leaves the
    # reflection swapping e₁ and e₃ invariant, so C₁₁₁₁ must equal C₃₃₃₃ in the
    # canonical frame. This is a reference-free statement, and it is the check
    # that exposes an orientation dropped anywhere in the assembly.
    for scheme in (MoriTanaka(), SelfConsistent(), AsymmetricSelfConsistent())
        A = _global(homogenize(_cracked_rve([(π / 4, 0.3)]), scheme))
        @test A[1, 1, 1, 1] ≈ A[3, 3, 3, 3] rtol = 1.0e-9
        @test A[1, 1, 2, 2] ≈ A[2, 2, 3, 3] rtol = 1.0e-9
    end

    # Splitting one family into two identical halves changes nothing.
    C_one = _global(homogenize(_cracked_rve([(π / 4, 0.5)]), MoriTanaka()))
    C_two = _global(homogenize(_cracked_rve([(π / 4, 0.25), (π / 4, 0.25)]), MoriTanaka()))
    @test C_one ≈ C_two rtol = 1.0e-12
end

@testset "_major_symmetrize" begin
    # NOTE: deliberately built from plain arrays rather than
    # `Tensors.SymmetricTensor`. `Tensors` is not in scope in this suite unless
    # some *other* test happens to pull it in (Ferrite re-exports it), and
    # `using Tensors` here would make the exported `gradient` ambiguous with
    # MeanFieldHomogenization's, which fails only when the whole suite runs.
    sym = MeanFieldHomogenization.Schemes._major_symmetrize

    # Structured, major-symmetric by construction: returned untouched, and — the
    # point of the type-preserving methods — with its concrete type intact, which
    # the Newton parameterization and the symmetry-class dispatch both need.
    for t in (
            CS_SCO, TensISO{3}(2.0),
            TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0.0, 0.0, 1.0)),
        )
        @test sym(t) === t
    end

    # A minor-symmetric but NOT major-symmetric probe: A_ijkl = (i+j)·(k·l).
    A = [Float64((i + j) * k * l) for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
    T4 = Tens(A)
    @test _major_asym(T4) > 1.0e-3        # the probe really is asymmetric
    S1 = sym(T4)
    @test _major_asym(S1) < 1.0e-13       # …and gets projected
    @test get_array(sym(S1)) ≈ get_array(S1) atol = 1.0e-14   # idempotent
    # The projection preserves the major-symmetric part it started from.
    Aexp = [(A[i, j, k, l] + A[k, l, i, j]) / 2 for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
    @test get_array(S1) ≈ Aexp atol = 1.0e-12

    # 2nd-order (conductivity) counterpart.
    K = Tens([1.0 * i + 2.0 * j for i in 1:3, j in 1:3])
    KS = sym(K)
    @test get_array(KS) ≈ [(3.0 * (i + j)) / 2 for i in 1:3, j in 1:3] atol = 1.0e-12
end
