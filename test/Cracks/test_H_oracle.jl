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
using ForwardDiff

# ω-family of the definition: (c/b) ℚ⁻¹ evaluated at finite flatness ω = c/b.
function _H_from_hill(a, b, ω, C₀; kw...)
    Q = second_hill_tensor(Ellipsoid(a, b, ω * b), C₀; kw...)   # ℚ = ℂ₀ − ℂ₀:ℙ:ℂ₀
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
    C₀ = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, [0.0, 0.0, 1.0])

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
    C₀ = TensND.from_mandel(KM_tri, basis)

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
    C₀ = TensND.from_mandel(KM_tri, basis)
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

# =============================================================================
#  An analytic anchor in anisotropic matrices: the linear transformation of the
#  crack problem (Barthélémy, 2026, "Transformation of the crack problem").
#  With x = P x̃, the matrix C = (P ⊠ˢ P) : C̃ : (P ⊠ˢ P)ᵀ and the crack (a, b, ℓ,
#  m, n) are the images of C̃ and of the crack (ã, b̃, ℓ̃, m̃, ñ), and the COD
#  tensors, both normalized by the half-width as everywhere in MFH, satisfy
#      𝐁ᴱ = (b̃/b) / ‖Pᵀn‖ · P⁻ᵀ · 𝐁̃ᴱ · P⁻¹,    𝐁ᴿ = P⁻ᵀ · 𝐁̃ᴿ · P⁻¹ / (‖P⁻¹ℓ‖ |det P|).
#  An isotropic C̃ gives an elliptically orthotropic matrix (EO), a transversely
#  isotropic one whose axis is the transformed normal a transformed transversely
#  isotropic one (TraTI, monoclinic here): 𝐁̃ then has a closed form, and the
#  three integrators of the anisotropic 𝐁 are held to it.
# =============================================================================

# The transformed crack, radii and axes in closed form (same reference).
function _transformed_crack(P, a, b, ℓ, m, n)
    g, h = P \ ℓ, P \ m
    area = norm(cross(g, h))
    s₊ = sqrt(a^2 * norm(g)^2 + b^2 * norm(h)^2 + 2a * b * area)
    s₋ = sqrt(a^2 * norm(g)^2 + b^2 * norm(h)^2 - 2a * b * area)
    ã, b̃ = (s₊ + s₋) / 2, (s₊ - s₋) / 2
    ñ = normalize(transpose(P) * n)
    w_ℓ = (a^2 * norm(g)^2 - b̃^2) * g + b^2 * dot(g, h) * h
    w_m = a^2 * dot(g, h) * g + (b^2 * norm(h)^2 - b̃^2) * h
    ℓ̃ = normalize(norm(w_ℓ) ≥ norm(w_m) ? w_ℓ : w_m)
    return ã, b̃, ℓ̃, cross(ñ, ℓ̃), ñ
end

# A right-handed crack frame (ℓ, m, n) of normal n, turned by ψ in its plane.
function _crack_frame(n, ψ)
    n = normalize(n)
    t = abs(n[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]
    e₁ = normalize(cross(t, n))
    ℓ = cos(ψ) * e₁ + sin(ψ) * cross(n, e₁)
    return ℓ, cross(n, ℓ), n
end

@testset "Cracks — ℍ = k n ⊗ˢ 𝐁 ⊗ˢ n from the Kelvin–Mandel block, any frame" begin
    # In the frame of the crack, ℍ is read off the out-of-plane block of 𝐁;
    # against the product of the components, in a general basis that takes the
    # other route, for 𝐁 written in another frame, an elliptic and a ribbon
    # crack, and with a dual entry of 𝐁.
    ℬ = TensND.RotatedBasis(0.7, 1.9, -0.4)
    n̂ = TensND.tens_basis(ℬ, 3)
    B = TensND.change_tens(TensND.Tens([2.0 0.3 -0.2; 0.3 1.5 0.4; -0.2 0.4 0.8]), TensND.RotatedBasis(0.2, 0.5, 1.1))
    for (crack, k) in ((EllipticCrack(1.0, 0.6; euler_angles = (0.7, 1.9, -0.4)), 3 / 4), (RibbonCrack(1.0; euler_angles = (0.7, 1.9, -0.4)), 2 / π))
        H = compliance_from_cod(B, crack, ℬ)
        @test TensND.get_array(H) isa TensND.Tensors.SymmetricTensor{4, 3}
        @test Array(TensND.components_canon(H)) ≈ Array(TensND.components_canon(k * (n̂ ⊗ˢ B ⊗ˢ n̂))) atol = 1.0e-14
        @test cod_from_compliance(H, crack, ℬ) ≈ B
        @test compliance_from_cod(B, crack, TensND.Basis(Matrix(TensND.vecbasis(ℬ)))) ≈ H
    end
    dH = ForwardDiff.derivative(x -> Array(TensND.components_canon(compliance_from_cod(x * B, ℬ))), 1.0)
    @test dH ≈ Array(TensND.components_canon(compliance_from_cod(B, ℬ)))
end

@testset "Cracks — 𝐁 in EO and TraTI matrices against the transformation rule" begin
    canonical = TensND.CanonicalBasis{3, Float64}()
    P = [1.3 0.2 -0.1; 0.15 0.8 0.25; -0.05 0.3 1.1]
    PP = TensND.mandel(Tens(P) ⊠ˢ Tens(P))
    image(C̃) = TensND.from_mandel(PP * TensND.mandel(C̃) * transpose(PP), canonical)
    tol = (; abstol = 1.0e-12, reltol = 1.0e-10)
    references = (
        (TensISO{3}(3 * 20.0, 2 * 9.0), [0.3, -0.5, 0.8]),                                       # EO
        (TensTI{4}(30.0, 2 * 25.0, sqrt(2) * 8.0, 2 * 10.0, 2 * 6.0, (0.0, 0.0, 1.0)), nothing),  # TraTI
    )
    for (C̃, n₀) in references
        C = image(C̃)
        # TraTI: the normal whose image is the axis of C̃, for the closed form.
        n = n₀ === nothing ? normalize(transpose(P) \ [0.0, 0.0, 1.0]) : normalize(n₀)
        ℓ, m, _ = _crack_frame(n, 0.6)

        a, b = 1.0, 0.45
        ã, b̃, ℓ̃, m̃, ñ = _transformed_crack(P, a, b, ℓ, m, n)
        B̃ = components_canon(cod_tensor(EllipticCrack(ã, b̃, hcat(ℓ̃, m̃, ñ)), C̃))
        Bᴱ = (b̃ / b) / norm(transpose(P) * n) * (transpose(P) \ B̃ / P)

        g = P \ ℓ
        ℓ̃ᴿ = normalize(g)
        B̃ᴿ = components_canon(cod_tensor(RibbonCrack(1.0, hcat(ℓ̃ᴿ, cross(ñ, ℓ̃ᴿ), ñ)), C̃))
        Bᴿ = (transpose(P) \ B̃ᴿ / P) / (norm(g) * abs(det(P)))

        # Measured: 4e-16 to 2e-13 for the ellipse, 2e-15 to 6e-15 for the ribbon.
        for method in (:nestedquadgk, :residues, :decuhr)
            Bₑ = components_canon(cod_tensor(EllipticCrack(a, b, hcat(ℓ, m, n)), C; method, tol...))
            Bᵣ = components_canon(cod_tensor(RibbonCrack(0.7, hcat(ℓ, m, n)), C; method, tol...))
            @test norm(Bₑ - Bᴱ) / norm(Bᴱ) < 1.0e-11
            @test norm(Bᵣ - Bᴿ) / norm(Bᴿ) < 1.0e-11
        end
    end
end

@testset "Cracks — derivatives of 𝐁 through the transformation rule" begin
    # d𝐁/dμ̃ by ForwardDiff through the cubature of the anisotropic 𝐁, against the
    # same derivative through the closed form of the isotropic one and the
    # transformation rule: two independent paths.
    canonical = TensND.CanonicalBasis{3, Float64}()
    P = [1.3 0.2 -0.1; 0.15 0.8 0.25; -0.05 0.3 1.1]
    PP = TensND.mandel(Tens(P) ⊠ˢ Tens(P))
    image(C̃) = TensND.from_mandel(PP * TensND.mandel(C̃) * transpose(PP), canonical)
    ℓ, m, n = _crack_frame([0.3, -0.5, 0.8], 0.6)
    a, b = 1.0, 0.45
    ã, b̃, ℓ̃, m̃, ñ = _transformed_crack(P, a, b, ℓ, m, n)
    crack, crack̃ = EllipticCrack(a, b, hcat(ℓ, m, n)), EllipticCrack(ã, b̃, hcat(ℓ̃, m̃, ñ))
    B_rule(μ) = (b̃ / b) / norm(transpose(P) * n) *
        (transpose(P) \ components_canon(cod_tensor(crack̃, TensISO{3}(3 * 20.0, 2μ))) / P)
    B_cubature(μ) = components_canon(
        cod_tensor(crack, image(TensISO{3}(3 * 20.0, 2μ)); method = :nestedquadgk, abstol = 1.0e-12, reltol = 1.0e-10)
    )
    dB = ForwardDiff.derivative(B_rule, 9.0)
    @test norm(ForwardDiff.derivative(B_cubature, 9.0) - dB) / norm(dB) < 1.0e-9

    # The residues work in Float64 and say so, rather than failing deep inside.
    μ = ForwardDiff.Dual(9.0, 1.0)
    C_dual = image(TensISO{3}(3 * 20.0, 2μ))
    @test_throws r"method = :nestedquadgk" cod_tensor(crack, C_dual; method = :residues)
    @test_throws r"method = :nestedquadgk" cod_tensor(RibbonCrack(1.0, hcat(ℓ, m, n)), C_dual; method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(Ellipsoid(1.0, 0.6, 0.3), C_dual; method = :residues)
end

@testset "Cracks — ℍ from the first-order term of the Hill tensor, by ForwardDiff" begin
    # ℙ(ω) = ℙ₀ + ω ℙ₁ for the ellipsoid (a, b, ω b); ℚ₁ = −ℂ:ℙ₁:ℂ, and ℍ is the
    # inverse of the block of ℚ₁ on the kernel of ℚ₀, the Kelvin–Mandel components
    # 33, 23, 13 of the crack frame (Barthélémy 2009). ℙ₁ is the derivative of the
    # cubature, at ω₀/2 and ω₀ combined to cancel its own first-order term; at
    # ω₀/2 = 2e-6 that derivative was NaN, √(1 − z²) having been promoted to a
    # dual where 1 − z² rounds to zero.
    canonical = TensND.CanonicalBasis{3, Float64}()
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C = TensND.from_mandel(KM_tri, canonical)
    tol = (; abstol = 1.0e-12, reltol = 1.0e-10)
    R = hcat(_crack_frame([0.3, -0.5, 0.8], 0.6)...)
    for η in (1.0, 0.5)
        P(ω) = TensND.mandel(hill_tensor(Ellipsoid(1.0, η, ω * η, R), C; tol...), canonical)
        ω₀ = 4.0e-6
        P₁ = 2 * ForwardDiff.derivative(P, ω₀ / 2) - ForwardDiff.derivative(P, ω₀)
        @test all(isfinite, P₁)
        crack = EllipticCrack(1.0, η, R)
        Y = TensND.mandel(TensND.from_mandel(-TensND.mandel(C) * P₁ * TensND.mandel(C), canonical), crack_basis(crack))
        H = zeros(6, 6)
        H[3:5, 3:5] = inv(Y[3:5, 3:5])
        H_ref = compliance_from_cod(cod_tensor(crack, C; method = :nestedquadgk, tol...), crack)
        # Measured: 1.0e-8, the accuracy of the derivative of the cubature there.
        @test _relerr(TensND.from_mandel(H, crack_basis(crack)), H_ref) < 5.0e-8
    end
end
