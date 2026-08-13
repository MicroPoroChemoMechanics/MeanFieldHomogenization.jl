using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  Thermal COD scalar `b` and resistivity contribution `R`.
#
#  ── Why this file was rewritten for v0.4.0 ─────────────────────────────────
#  Most assertions here used to restate the very closed form the code
#  evaluates, which cannot detect a wrong prefactor. The one non-tautological
#  test — the Hill-tensor limit — compared its oracle against
#  `delta_resistivity(crack, R, 1) = (4π/3) R` instead of against `R` itself.
#  A `b` too small by exactly `4π/(3η)` is invisible to that comparison,
#  because the spurious `4π/3` on one side cancels it. That is how the error
#  survived 37 green tests.
#
#  The oracle below is therefore anchored where the definition puts it,
#
#       R = lim_{ω→0} ω · (K₀ − K₀ P(ω) K₀)⁻¹ ,      ω = c/b ,
#
#  and its convergence *rate* is checked, not just one value: the truncation
#  is O(ω), so halving ω must halve the error. A constant offset — which is
#  what a wrong prefactor looks like — fails that even when a single-ω
#  tolerance would pass.
# =============================================================================

const ELL_T = MeanFieldHomogenization.Elliptic

"`R₃₃` from the flattening route at finite flatness `ω = c/b`, in the crack frame."
function R_from_hill_thermal(crack, K₀, ω)
    a, bmin = crack isa RibbonCrack ? (1.0e8 * crack.b, crack.b) : (crack.a, crack.b)
    P = hill_tensor(Ellipsoid(a, bmin, ω * bmin, crack_basis(crack)), K₀)
    Pa = [P[i, j] for i in 1:3, j in 1:3]
    Ka = [K₀[i, j] for i in 1:3, j in 1:3]
    return ω * inv(Ka - Ka * Pa * Ka)
end

@testset "Cracks / conductivity — thermal COD scalar b and R" begin

    @testset "Iso matrix, penny crack" begin
        for k₀ in (1.0, 2.5, 7.3)
            K₀ = TensISO{3}(k₀)
            crack = PennyCrack(1.0)
            b = cod_tensor(crack, K₀)
            # Surface average of the textbook jump 4σₙa/(πk₀)·√(1-r²/a²), over a.
            @test b ≈ 8 / (3 * π * k₀) rtol = 1.0e-13
            R = compliance_contribution(crack, K₀)
            @test R[3, 3] ≈ 2 / (π * k₀) rtol = 1.0e-13      # (3/4)·b
            @test R[1, 1] ≈ 0 atol = 1.0e-14
            @test R[2, 2] ≈ 0 atol = 1.0e-14
            ΔR = delta_resistivity(crack, R, 1.0)
            @test ΔR[3, 3] ≈ (4π / 3) * R[3, 3] rtol = 1.0e-13
        end
    end

    @testset "Iso matrix, non-penny elliptic" begin
        k₀ = 2.0
        K₀ = TensISO{3}(k₀)
        for η in (0.75, 0.5, 0.3)
            crack = EllipticCrack(1.0, η)
            ℰ = ELL_T.ell_E(1 - η^2)
            @test cod_tensor(crack, K₀) ≈ 4 / (3 * k₀ * ℰ) rtol = 1.0e-13
        end
    end

    @testset "Iso matrix, ribbon crack" begin
        for k₀ in (1.0, 2.5)
            K₀ = TensISO{3}(k₀)
            crack = RibbonCrack(0.5)
            b = cod_tensor(crack, K₀)
            @test b ≈ π / (2 * k₀) rtol = 1.0e-13
            R = compliance_contribution(crack, K₀)
            @test R[3, 3] ≈ 1 / k₀ rtol = 1.0e-13            # (2/π)·b
            ΔR = delta_resistivity(crack, R, 1.0)
            @test ΔR[3, 3] ≈ π * R[3, 3] rtol = 1.0e-13
        end
    end

    @testset "Aligned TI — the effective conductivity is the geometric mean" begin
        k_t, k_n = 1.0, 4.0
        K₀ = TensND.Tens(Matrix(Diagonal([k_t, k_t, k_n])))
        @test cod_tensor(PennyCrack(1.0), K₀) ≈ 8 / (3π * sqrt(k_t * k_n)) rtol = 1.0e-13
        for η in (0.7, 0.4)
            ℰ = ELL_T.ell_E(1 - η^2)
            @test cod_tensor(EllipticCrack(1.0, η), K₀) ≈
                4 / (3 * sqrt(k_t * k_n) * ℰ) rtol = 1.0e-13
        end
    end

    @testset "Aniso ribbon — only the transverse block enters" begin
        K₀ = TensND.Tens(Matrix(Diagonal([3.0, 4.0, 1.5])))
        # K₀[1,1] is deliberately different: it must not appear in the answer.
        @test cod_tensor(RibbonCrack(0.8), K₀) ≈ π / (2 * sqrt(4.0 * 1.5)) rtol = 1.0e-13
    end

    @testset "The anisotropic route reproduces the isotropic one" begin
        # Same medium, presented as TensISO and as a dense Tens: the adjugate
        # branch and the k₀ branch must agree to machine precision.
        k₀ = 2.5
        K_iso = TensISO{3}(k₀)
        K_dense = TensND.Tens(Matrix(k₀ * I(3)))
        for crack in (PennyCrack(1.0), EllipticCrack(1.0, 0.4), RibbonCrack(0.7))
            @test cod_tensor(crack, K_dense) ≈ cod_tensor(crack, K_iso) rtol = 1.0e-12
        end
    end

    @testset "ORACLE — R is the flattening limit, and the error decays like ω" begin
        # The check the previous version of this file got wrong. `R`, not `ΔR`.
        cases = (
            (TensISO{3}(1.0), PennyCrack(1.0)),
            (TensISO{3}(2.0), EllipticCrack(1.0, 0.5)),
            # n̂ is NOT an eigenvector of K₀ here, so this also exercises the
            # adjugate branch and its frame handling.
            (TensND.Tens([3.0 0.5 0.3; 0.5 2.0 0.2; 0.3 0.2 1.5]), PennyCrack(1.0)),
            (TensND.Tens([3.0 0.5 0.3; 0.5 2.0 0.2; 0.3 0.2 1.5]), EllipticCrack(1.0, 0.6)),
        )
        for (K₀, crack) in cases
            R = compliance_contribution(crack, K₀)
            errs = map((1.0e-2, 1.0e-3)) do ω
                O = R_from_hill_thermal(crack, K₀, ω)
                maximum(abs, [R[i, j] - O[i, j] for i in 1:3, j in 1:3])
            end
            # Converging at all …
            @test errs[2] < errs[1]
            # … and at order ω: a tenfold smaller ω must cut the error ~tenfold.
            # A wrong prefactor would leave a constant offset and flunk this.
            @test errs[1] / errs[2] > 5
            @test errs[2] / maximum(abs, [R[i, j] for i in 1:3, j in 1:3]) < 5.0e-3
        end
    end

    @testset "Rotated crack basis — invariance of the scalar b" begin
        K₀ = TensISO{3}(1.0)
        b_canonical = cod_tensor(PennyCrack(1.0), K₀)
        rot = TensND.RotatedBasis(0.3, 0.4, 0.2)
        @test cod_tensor(EllipticCrack(1.0, 1.0, rot), K₀) ≈ b_canonical rtol = 1.0e-13
    end

    @testset "Rotating the whole problem changes nothing" begin
        # An anisotropic conductor and a crack, both rotated together: b is a
        # scalar, so it must be strictly invariant. Catches frame mixing in the
        # adjugate branch, the failure mode the elastic kernels once had.
        K_arr = [3.0 0.5 0.3; 0.5 2.0 0.2; 0.3 0.2 1.5]
        b_ref = cod_tensor(EllipticCrack(1.0, 0.6), TensND.Tens(K_arr))
        for angles in ((0.3, 0.4, 0.2), (1.1, -0.5, 0.7))
            rot = TensND.RotatedBasis(angles...)
            Q = [rot[i, j] for i in 1:3, j in 1:3]
            K_rot = TensND.Tens(Q * K_arr * transpose(Q))
            b_rot = cod_tensor(EllipticCrack(1.0, 0.6, rot), K_rot)
            @test b_rot ≈ b_ref rtol = 1.0e-11
        end
    end

end

@testset "Cracks / conductivity — ForwardDiff through the thermal COD" begin
    # The adjugate route is plain arithmetic plus `sqrt` and `ell_E`, so it must
    # differentiate. Checked against central differences, including a *gradient*
    # with respect to all six components of an anisotropic K₀ — that exercises
    # the branch the `K₀^{-1/2}` route could not reach at all, `eigen` and
    # `svdvals` being Float64-only.
    h = 1.0e-6

    # b ∝ 1/k₀ exactly, so the derivative is known in closed form.
    for k₀ in (1.0, 2.5)
        d = ForwardDiff.derivative(k -> cod_tensor(PennyCrack(1.0), TensISO{3}(k)), k₀)
        @test d ≈ -cod_tensor(PennyCrack(1.0), TensISO{3}(k₀)) / k₀ rtol = 1.0e-12
    end

    fη(η) = cod_tensor(EllipticCrack(1.0, η), TensISO{3}(1.5))
    @test ForwardDiff.derivative(fη, 0.6) ≈ (fη(0.6 + h) - fη(0.6 - h)) / (2h) rtol = 1.0e-6

    fr(k) = cod_tensor(RibbonCrack(0.7), TensISO{3}(k))
    @test ForwardDiff.derivative(fr, 2.0) ≈ (fr(2.0 + h) - fr(2.0 - h)) / (2h) rtol = 1.0e-6

    fR(k) = compliance_contribution(PennyCrack(1.0), TensISO{3}(k))[3, 3]
    @test ForwardDiff.derivative(fR, 2.0) ≈ (fR(2.0 + h) - fR(2.0 - h)) / (2h) rtol = 1.0e-6

    # Gradient through the anisotropic adjugate branch.
    packK(v) = TensND.Tens([v[1] v[4] v[5]; v[4] v[2] v[6]; v[5] v[6] v[3]])
    fK(v) = cod_tensor(EllipticCrack(1.0, 0.6), packK(v))
    v0 = [3.0, 2.0, 1.5, 0.5, 0.3, 0.2]
    g = ForwardDiff.gradient(fK, v0)
    fd = map(eachindex(v0)) do i
        vp = copy(v0); vm = copy(v0); vp[i] += h; vm[i] -= h
        (fK(vp) - fK(vm)) / (2h)
    end
    @test maximum(abs, g .- fd) < 1.0e-8
    @test maximum(abs, g) > 1.0e-3          # not trivially zero
end

@testset "Cracks / conductivity — SIF and DIF" begin

    @testset "Ribbon thermal SIF" begin
        K₀ = TensISO{3}(2.5)
        b_ribbon = 0.8
        crack = RibbonCrack(b_ribbon)
        q∞ = TensND.Tens([1.0, 0.0, 3.0])
        # Stiffness-independent, exactly as in elasticity.
        @test sif(crack, K₀, q∞) ≈ sqrt(π * b_ribbon) * 3.0 rtol = 1.0e-13
    end

    @testset "Penny thermal DIF: [T] = b · (n̂·q∞)" begin
        K₀ = TensISO{3}(1.5)
        crack = PennyCrack(1.0)
        b = cod_tensor(crack, K₀)
        q∞ = TensND.Tens([2.0, 1.0, 4.0])
        @test dif(crack, K₀, q∞) ≈ b * 4.0 rtol = 1.0e-13
    end

end
