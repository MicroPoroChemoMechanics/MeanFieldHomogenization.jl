# =============================================================================
#  test_H_oracle.jl
#
#  Independent oracle for the crack compliance contribution tensor ℍ.
#
#  The other crack tests compare ℍ against the closed form of the COD tensor
#  𝐁 — which is precisely what `src/Cracks/cod_analytical.jl` implements, so
#  they cannot detect an error in that closed form itself.  This file closes
#  the loop by evaluating the *definition* instead
#  (see the `cod_tensor` docstring in `src/Cracks/api.jl`):
#
#      ℍ = lim_{c/b→0} (c/b) ℚ⁻¹ ,      ℚ = ℂ − ℂ:ℙ:ℂ
#
#  starting from `hill_tensor` on a flattened `Ellipsoid(a, b, ω·b)`.  The
#  path ℙ → ℚ → ℚ⁻¹ → limit never touches the COD closed form, so agreement
#  is genuine evidence and not a tautology.  This is the exercise of the
#  Echoes manual appendix (`crack_compliance.qmd`, figure `fig-errorH`).
#
#  What is actually asserted:  a single ω would not distinguish a true limit
#  from a coincidence, so the test checks that the error *decreases at the
#  expected order*.  Convergence is O(ω) (first-order Taylor term of the Hill
#  tensor, Barthélémy 2009), hence a factor ≈ 10 between ω = 1e-2 and
#  ω = 1e-3; the threshold is set at 5 to leave room for the quadrature.
#
#  Backend note: the nested-QuadGK backend is validated down to ω = 1e-3 in
#  `test/Elasticity/test_hill_nestedquadgk_oblate.jl`, which is why the sweep
#  stops there.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# ω-family of the definition: (c/b) ℚ⁻¹ evaluated at finite flatness ω = c/b.
function _H_from_hill(a, b, ω, C₀; kw...)
    P = hill_tensor(Ellipsoid(a, b, ω * b), C₀; kw...)
    Q = C₀ - C₀ ⊡ P ⊡ C₀
    return ω * inv(Q)
end

_relerr(A, B) = maximum(abs, components_canon(A - B)) / maximum(abs, components_canon(B))

# Crack of in-plane aspect ratio η, semi-major 1 — normal along e₃ by default.
_crack(η) = η == 1.0 ? PennyCrack(1.0) : EllipticCrack(1.0, η)

const ORACLE_ω = (1.0e-2, 1.0e-3)   # ratio 10 expected between the two

@testset "Cracks — ℍ oracle: (c/b)·ℚ⁻¹ → ℍ (isotropic matrix)" begin
    E, ν = 210.0, 0.3
    C₀ = TensISO{3}(E / (1 - 2ν), E / (1 + ν))     # (3k, 2μ)

    for η in (1.0, 0.7, 0.5, 0.3)
        crack = _crack(η)
        H_ref = inverse_property_contribution(crack, C₀)
        errs = [_relerr(_H_from_hill(1.0, η, ω, C₀), H_ref) for ω in ORACLE_ω]

        @test errs[2] < 2.0e-3              # the limit is actually approached
        @test errs[1] / errs[2] > 5         # …and at the O(ω) rate
    end
end

@testset "Cracks — ℍ oracle: transversely isotropic matrix, axis = crack normal" begin
    # Sevostianov-Yilmaz-Kushch-Levin (2005) test stiffness, as in
    # `test_cod_ti_aligned.jl`.
    C₀ = tens_TI(2.179, 0.579, 0.689, 10.345, 1.0, [0.0, 0.0, 1.0])

    for η in (1.0, 0.5)
        H_ref = inverse_property_contribution(_crack(η), C₀)
        errs = [
            _relerr(_H_from_hill(1.0, η, ω, C₀; method = :nestedquadgk, reltol = 1.0e-8), H_ref)
                for ω in ORACLE_ω
        ]

        @test errs[2] < 5.0e-3
        @test errs[1] / errs[2] > 5
    end
end

@testset "Cracks — ℍ oracle: fully anisotropic (triclinic) matrix" begin
    # Same triclinic stiffness as `test_residue_accuracy.jl`.
    basis = TensND.CanonicalBasis{3, Float64}()
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C₀ = TensND.inv_KM(KM_tri, basis)

    for η in (1.0, 0.5)
        H_ref = inverse_property_contribution(_crack(η), C₀)
        errs = [
            _relerr(_H_from_hill(1.0, η, ω, C₀; method = :nestedquadgk, reltol = 1.0e-8), H_ref)
                for ω in ORACLE_ω
        ]

        @test errs[2] < 5.0e-3
        @test errs[1] / errs[2] > 5
    end
end

@testset "Cracks — every route to ℍ and 𝐁 agrees (triclinic matrix)" begin
    # 𝐁 from the reduced kernel (Barthélémy 2026) by three integrators, turned
    # into ℍ by `compliance_from_cod`; ℍ from the flattening of an ellipsoid
    # (Barthélémy 2009), with the Hill tensor by three integrators, extrapolated
    # to ω → 0 from two flatnesses — the error is O(ω), Richardson's combination
    # O(ω²) — and turned back into 𝐁 by `cod_from_compliance`.
    basis = TensND.CanonicalBasis{3, Float64}()
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C₀ = TensND.inv_KM(KM_tri, basis)
    tol = (; abstol = 1.0e-12, reltol = 1.0e-10)

    for η in (1.0, 0.5)
        crack = _crack(η)
        Bs = [cod_tensor(crack, C₀; method = m, tol...) for m in (:nestedquadgk, :decuhr, :residues)]
        H_ref = compliance_from_cod(Bs[1], crack)
        @test all(B -> _relerr(B, Bs[1]) < 1.0e-12, Bs)
        @test _relerr(inverse_property_contribution(crack, C₀; tol...), H_ref) < 1.0e-12
        # Measured: 1.5e-9 with the cubature and the residues from ω = 1e-4 and
        # 1e-5, 1.5e-5 with DECUHR from ω = 1e-2 and 1e-3.
        for (m, ω, rtol) in ((:nestedquadgk, 1.0e-4, 1.0e-8), (:residues, 1.0e-4, 1.0e-8), (:decuhr, 1.0e-2, 1.0e-4))
            H = (10 * _H_from_hill(1.0, η, ω / 10, C₀; method = m, tol...) - _H_from_hill(1.0, η, ω, C₀; method = m, tol...)) / 9
            @test _relerr(H, H_ref) < rtol
            @test _relerr(cod_from_compliance(H, crack), Bs[1]) < rtol
        end
    end
end
