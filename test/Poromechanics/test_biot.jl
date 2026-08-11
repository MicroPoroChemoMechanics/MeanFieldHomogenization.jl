using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# Reference solid phase, reused everywhere below.
const KS_PORO, MUS_PORO = 20.0, 12.0
const CS_PORO = TensISO{3}(3 * KS_PORO, 2 * MUS_PORO)

# Mori-Tanaka bulk modulus of a solid with spherical pores, closed form:
#   k_MT = 4 μ_s k_s (1-φ) / (4 μ_s + 3 k_s φ)
_k_mt_porous(ks, μs, φ) = 4 * μs * ks * (1 - φ) / (4 * μs + 3 * ks * φ)

function _porous_rve(φ; C_s = CS_PORO)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_s))
    add_phase!(
        rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9));
        fraction = φ
    )
    return rve
end

@testset "isotropic closed forms" begin
    φ = 0.2
    rve = _porous_rve(φ)
    C_hom = homogenize(rve, MoriTanaka())

    B = biot_tensor(C_hom, CS_PORO)
    @test B isa TensND.AbstractTens{2, 3}
    # b = 1 - k_hom / k_s, and B = b δ for an isotropic pore space.
    b_ref = 1 - _k_mt_porous(KS_PORO, MUS_PORO, φ) / KS_PORO
    @test B[1, 1] ≈ b_ref
    @test B[2, 2] ≈ b_ref
    @test B[3, 3] ≈ b_ref
    @test B[1, 2] ≈ 0.0 atol = 1.0e-14

    # 1/M = (b - φ) / k_s
    invM = inverse_biot_modulus(CS_PORO, B, φ)
    @test invM ≈ (b_ref - φ) / KS_PORO
    @test biot_modulus(CS_PORO, B, φ) ≈ 1 / invM

    # The RVE convenience layer must agree exactly (matrix = solid phase).
    @test get_array(biot_tensor(rve, C_hom)) == get_array(B)
    par = poroelastic_parameters(rve, C_hom, φ)
    @test get_array(par.B) == get_array(B)
    @test par.inverse_modulus == invM
    @test par.modulus == 1 / invM
end

@testset "limits of the Biot tensor" begin
    # A homogeneous medium has no pore space: B = 0.
    B_hom = biot_tensor(CS_PORO, CS_PORO)
    @test maximum(abs, get_array(B_hom)) < 1.0e-14

    # A vanishing drained stiffness gives B = δ.
    B_soft = biot_tensor(TensISO{3}(1.0e-14, 1.0e-14), CS_PORO)
    @test B_soft[1, 1] ≈ 1.0 atol = 1.0e-12
    @test B_soft[3, 3] ≈ 1.0 atol = 1.0e-12

    # An incompressible solid phase gives 1/M = 0 hence M = Inf, whatever the
    # drained stiffness and the porosity: δ : s_s : δ vanishes and so does the
    # whole expression once B = δ.
    C_incomp = TensISO{3}(3 * 1.0e14, 2 * MUS_PORO)
    rve = _porous_rve(0.15; C_s = C_incomp)
    C_hom = homogenize(rve, MoriTanaka())
    par = poroelastic_parameters(C_hom, C_incomp, 0.15)
    @test par.inverse_modulus ≈ 0.0 atol = 1.0e-13
end

@testset "the two contraction routes agree" begin
    # B = δ : (𝕀 - s_s : C_hom) and B = δ - C_hom : (s_s : δ) are the same
    # tensor by the major symmetry of s_s and C_hom. Checked on a genuinely
    # anisotropic C_hom (aligned penny cracks), where a wrong contraction order
    # would show up.
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_PORO))
    add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => CS_PORO); density = 0.08)
    C_hom = homogenize(rve, MoriTanaka())

    T = Float64
    δ = TensND.tens_Id2(Val(3), Val(T))
    Id4 = TensND.tens_Id4(Val(3), Val(T))
    s_s = inv(CS_PORO)
    B1 = biot_tensor(C_hom, CS_PORO)
    B2 = δ - C_hom ⊡ (s_s ⊡ δ)
    @test get_array(B1) ≈ get_array(B2) atol = 1.0e-14

    # Aligned cracks make the pore space transversely isotropic: the normal
    # component must dominate the in-plane ones.
    @test B1[3, 3] > B1[1, 1]
    @test B1[1, 1] ≈ B1[2, 2]
end

@testset "undrained ↔ drained round trip and Skempton" begin
    φ = 0.2
    rve = _porous_rve(φ)
    C_hom = homogenize(rve, MoriTanaka())
    par = poroelastic_parameters(C_hom, CS_PORO, φ)
    B, M = par.B, par.modulus

    C_u = undrained_stiffness(C_hom, B, M)
    @test get_array(drained_stiffness(C_u, B, M)) ≈ get_array(C_hom) atol = 1.0e-10

    k_hom, μ_hom = k_mu(C_hom)
    k_u, μ_u = k_mu(C_u)
    b = B[1, 1]
    # `B ⊗ B` is purely volumetric for an isotropic Biot tensor, so the
    # undrained shear modulus is the drained one — *not* the solid's.
    @test μ_u ≈ μ_hom rtol = 1.0e-12
    @test k_u ≈ k_hom + M * b^2 rtol = 1.0e-12

    Bsk = skempton_tensor(C_hom, B, M)
    S_scalar = tr(get_array(Bsk))
    @test S_scalar ≈ M * b / k_u rtol = 1.0e-10

    # Independent cross-check of the whole chain (b → 1/M → k_u → Skempton)
    # against the classical closed form for an *incompressible* fluid
    # (k_f = ∞), which is the assumption behind `inverse_biot_modulus`:
    #   B = (1/k - 1/k_s) / (1/k - 1/k_s + φ (1/k_f - 1/k_s))
    S_ref = (1 / k_hom - 1 / KS_PORO) / (1 / k_hom - 1 / KS_PORO - φ / KS_PORO)
    @test S_scalar ≈ S_ref rtol = 1.0e-12
    # An incompressible fluid is stiffer than the compressible grains, so the
    # textbook bound `B ≤ 1` legitimately fails here. Guard the documented
    # behavior rather than an inapplicable bound.
    @test S_scalar > 1

    # Defining property: under an undrained isotropic stress -p₀ δ the pore
    # pressure is p = -Bsk : Σ.
    p₀ = 3.0
    Σ = -p₀ * TensND.tens_Id2(Val(3), Val(Float64))
    @test -(Bsk ⊡ Σ) ≈ p₀ * S_scalar rtol = 1.0e-12
end

@testset "effective stresses" begin
    Σ = Tens(TensND.SymmetricTensor{2, 3}((i, j) -> i == j ? -10.0 * i : 1.0))
    p = 4.0
    Σ_t = terzaghi_stress(Σ, p)
    @test Σ_t[1, 1] ≈ Σ[1, 1] + p
    @test Σ_t[1, 2] ≈ Σ[1, 2]

    # Biot effective stress coincides with Terzaghi iff B = δ.
    δ = TensND.tens_Id2(Val(3), Val(Float64))
    @test get_array(biot_effective_stress(Σ, p, δ)) ≈ get_array(Σ_t)

    B = TensISO{3}(0.6)
    @test biot_effective_stress(Σ, p, B)[1, 1] ≈ Σ[1, 1] + 0.6 * p
end

@testset "pore_volume_fraction" begin
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_PORO))
    add_phase!(rve, :P1, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9)); fraction = 0.1)
    add_phase!(rve, :P2, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9)); fraction = 0.05)
    add_phase!(rve, :SOLID, Ellipsoid(1.0), Dict(:C => CS_PORO); fraction = 0.2)
    add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => CS_PORO); density = 0.05)

    @test pore_volume_fraction(rve, (:P1, :P2)) ≈ 0.15
    @test pore_volume_fraction(rve, :P1) ≈ 0.1
    # A crack phase carries no volume: it must contribute exactly zero.
    @test pore_volume_fraction(rve, :CR) == 0.0
    @test pore_volume_fraction(rve, (:P1, :P2, :CR)) ≈ 0.15
end

@testset "number types are part of the contract" begin
    φ = 0.2

    # BigFloat
    C_big = TensISO{3}(big(3 * KS_PORO), big(2 * MUS_PORO))
    rve_big = _porous_rve(big(φ); C_s = C_big)
    C_hom_big = homogenize(rve_big, MoriTanaka())
    par_big = poroelastic_parameters(C_hom_big, C_big, big(φ))
    @test eltype(par_big.B) === BigFloat
    @test eltype(par_big.inverse_modulus) === BigFloat
    # The empty-pore closed form is only approached to the accuracy of the
    # `1e-9` stiffness used to regularize the pore — that is a property of the
    # RVE, not of the arithmetic, so the tolerance is set by the regularization.
    @test par_big.B[1, 1] ≈
        big(1) - _k_mt_porous(big(KS_PORO), big(MUS_PORO), big(φ)) / KS_PORO rtol = 1.0e-9

    # BigFloat *precision* is exercised instead by an exact identity: the two
    # contraction routes for `B` must agree to full working precision.
    δ_big = TensND.tens_Id2(Val(3), Val(BigFloat))
    B_alt = δ_big - C_hom_big ⊡ (inv(C_big) ⊡ δ_big)
    @test get_array(par_big.B) ≈ get_array(B_alt) rtol = 1.0e-60

    # ForwardDiff through the porosity: d(1/M)/dφ = -1/k_s at fixed C_hom.
    C_hom = homogenize(_porous_rve(φ), MoriTanaka())
    B = biot_tensor(C_hom, CS_PORO)
    d = ForwardDiff.derivative(x -> inverse_biot_modulus(CS_PORO, B, x), φ)
    @test d ≈ -1 / KS_PORO rtol = 1.0e-12

    # ForwardDiff all the way through the scheme: db/dφ must be positive
    # (more pores, softer medium, larger Biot coefficient).
    function b_of_φ(x)
        rve = _porous_rve(x)
        return biot_tensor(homogenize(rve, MoriTanaka()), CS_PORO)[1, 1]
    end
    @test ForwardDiff.derivative(b_of_φ, φ) > 0
end
