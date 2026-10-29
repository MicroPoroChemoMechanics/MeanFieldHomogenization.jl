# PR-4 validation: power change-of-variable in nestedquadgk for flat ellipsoids.
#
# Before this PR the nestedquadgk backend used a plain z ∈ [0, 1] outer
# integral.  For flat ellipsoids (ω ≪ 1) the integrand has a steep gradient
# near z = 1 that requires many subdivisions — or fails to converge at default
# maxiters.  The new path applies z(u) = 1 − (1 − u)^α with
# α = max(1, log₁₀(1/ω)), concentrating Gauss-Kronrod nodes near z = 1.
#
# Note: the Ellipsoid constructor already sorts semi-axes in descending order
# for real-valued types (via _sort_axes_and_basis), so η = a₂/a₁ ≤ 1 and
# ω = a₃/a₁ ≤ 1 are guaranteed on entry — the axis-sort inside the backend
# is always a no-op for Float64 inputs.
using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

const ATOL4 = 5.0e-6   # looser than spheroidal due to near-singular kernel at z≈1
const basis4 = TensND.CanonicalBasis{3, Float64}()

# Triclinic reference stiffness (same hand-picked matrix as test_anisotropic.jl)
const _KM_obl = [
    210.0 80.0 75.0 5.0 4.0 3.0;
    80.0 195.0 90.0 -2.0 3.0 -1.0;
    75.0 90.0 220.0 1.0 -2.0 2.0;
    5.0 -2.0 1.0 60.0 2.5 1.5;
    4.0 3.0 -2.0 2.5 65.0 -1.0;
    3.0 -1.0 2.0 1.5 -1.0 55.0
]


@testset "nestedquadgk — oblate ellipsoid ω = 0.01 (α = 2)" begin
    # Stored as (1.0, 1.0, 0.01) after constructor sort; ω = 0.01 → α = 2.
    # The change-of-variable concentrates quadrature near z = 1.
    C = TensND.from_mandel(_KM_obl, basis4)
    ell = Ellipsoid(1.0, 1.0, 0.01)

    P_nqg = hill_tensor(ell, C; method = :nestedquadgk)
    P_dec = hill_tensor(ell, C; method = :decuhr, reltol = 1.0e-9)

    diff = maximum(abs, components_canon(P_nqg - P_dec))
    @test diff < ATOL4
    @test all(isfinite, P_nqg)
end


@testset "nestedquadgk — very oblate ellipsoid ω = 0.001 (α = 3)" begin
    # ω = 0.001 → α = 3; tightest practical test for the change-of-variable.
    C = TensND.from_mandel(_KM_obl, basis4)
    ell = Ellipsoid(1.0, 1.0, 0.001)

    P_nqg = hill_tensor(ell, C; method = :nestedquadgk)
    P_dec = hill_tensor(ell, C; method = :decuhr, reltol = 1.0e-9)

    diff = maximum(abs, components_canon(P_nqg - P_dec))
    @test diff < ATOL4
    @test all(isfinite, P_nqg)
end


@testset "nestedquadgk — regular triaxial (α = 1, regression guard)" begin
    # ω = 1/3 → α = max(1, 0.48) = 1 (identity change-of-variable).
    # Checks that the new path does not regress accuracy on well-conditioned cases.
    C = TensND.from_mandel(_KM_obl, basis4)
    ell = Ellipsoid(3.0, 2.0, 1.0)

    P_nqg = hill_tensor(ell, C; method = :nestedquadgk, reltol = 1.0e-12)
    P_res = hill_tensor(ell, C; method = :residues)

    scale = maximum(abs, components_canon(P_nqg))
    diff = maximum(abs, components_canon(P_nqg - P_res))
    @test diff < 1.0e-7 * scale
end


@testset "nestedquadgk — derivative with respect to a semi-axis" begin
    # The exponent α was read as `Float64(ω)`, which a dual semi-axis cannot be
    # converted to: the derivative threw. Reference: the closed form of the
    # spheroid in an isotropic medium, differentiated the same way; the same
    # medium is handed to the quadrature as a general tensor.
    C_iso = TensISO{3}(300.0, 80.0)
    C_gen = TensND.Tens(Array(get_array(C_iso)), basis4)
    P(ω, C; kw...) = Array(get_array(hill_tensor(Ellipsoid(1.0, 1.0, ω), C; kw...)))
    for ω in (0.2, 0.02)    # α = 1, then α = 2 − log₁₀ 2
        d_ref = ForwardDiff.derivative(x -> P(x, C_iso), ω)
        d_nqg = ForwardDiff.derivative(x -> P(x, C_gen; method = :nestedquadgk, reltol = 1.0e-10), ω)
        @test d_nqg ≈ d_ref rtol = 1.0e-7
    end
end

@testset "nestedquadgk — the boundary layer of a very flat ellipsoid" begin
    # Isotropic values in a fully anisotropic type, so that the closed form is
    # the exact reference. Below ω ≈ 1e-4, QuadGK used to miss the layer z ~ ω
    # near z = 0 and stop with an error of 2e-5 at ω = 1e-5 for the default
    # `reltol = 1e-6`.
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_full = TensND.change_tens_canon(C_iso)
    canon(P) = TensND.get_array(TensND.change_tens_canon(P))
    for ω in (1.0e-5, 1.0e-6), ell in (Ellipsoid(1.0, 1.0, ω), Ellipsoid(1.0, 0.5, ω))
        P = canon(hill_tensor(ell, C_full; method = :nestedquadgk))
        R = canon(hill_tensor(ell, C_iso))
        @test maximum(abs, P .- R) / maximum(abs, R) < 1.0e-6
    end
end

@testset "nestedquadgk — a very flat ellipsoid: derivative and inference" begin
    # The breakpoints of the boundary layer go through a separate QuadGK call,
    # compiled on demand: it keeps the derivatives and an inferred result. Exact
    # identity: ℙ(λℂ) = ℙ(ℂ)/λ, so dℙ/dλ = −ℙ at λ = 1.
    C = TensND.change_tens_canon(TensISO{3}(3 * 30.0, 2 * 18.0))
    ell = Ellipsoid(1.0, 0.5, 1.0e-5)
    canon(P) = TensND.get_array(TensND.change_tens_canon(P))
    P = canon(hill_tensor(ell, C; method = :nestedquadgk))
    dP = ForwardDiff.derivative(λ -> canon(hill_tensor(ell, λ * C; method = :nestedquadgk)), 1.0)
    @test dP ≈ -P rtol = 1.0e-6
    nested = MeanFieldHomogenization.Elasticity._hill_3d_aniso_nestedquadgk
    @test only(Base.return_types(nested, (typeof(ell), typeof(C)))) <: TensND.AbstractTens{4, 3, Float64}
end
