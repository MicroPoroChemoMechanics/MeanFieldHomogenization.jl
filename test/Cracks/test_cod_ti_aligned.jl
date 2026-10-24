# =============================================================================
#  test_cod_ti_aligned.jl
#
#  Validates the closed-form COD tensor for an elliptic / penny / ribbon
#  crack whose normal coincides with the symmetry axis of a transversely
#  isotropic matrix (Hoenig 1979 / Kanaun-Levin 2009 / Barthélémy 2021).
#
#  Coverage:
#   1. Iso-as-TI degenerate limit: the TI analytical branch must reproduce
#      the isotropic analytical result.
#   2. Cross-validation against the residue and DECUHR numerical backends
#      on a true TI matrix (Sevostianov-Yilmaz-Kushch-Levin 2005 stiffness).
#   3. Penny crack (η=1) and ribbon crack — specialized closed forms.
#   4. Dispatcher: Analytical when TI axis = crack normal, Residue otherwise.
#   5. ForwardDiff compatibility through the analytical branch.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const ATOL_COD_ISO = 1.0e-10
const ATOL_COD_RES = 1.0e-9
const ATOL_COD_DEC = 1.0e-7

# Helper: convert TI stiffness to engineering compliance moduli.
# (TensND.ti_args_hoenig, as the analytical kernel reads them.)

@testset "COD TI aligned — iso-as-TI recovers isotropic builder" begin
    # Build an isotropic stiffness, then re-express it as a degenerate TI
    # tensor with axis along the crack normal e₃.
    n_axis = [0.0, 0.0, 1.0]
    E, ν = 210.0, 0.3
    k = E / (3 * (1 - 2ν)); μ = E / (2 * (1 + ν))
    C_iso = TensISO{3}(3k, 2μ)
    C_ti = iso_to_ti(C_iso, n_axis)
    @test C_ti isa TensND.TensTI{4, Float64, 5}

    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5), RibbonCrack(1.0))
        B_iso = cod_tensor(c, C_iso)
        B_ti = cod_tensor(c, C_ti)
        @test maximum(abs, components_canon(B_ti - B_iso)) < ATOL_COD_ISO
    end
end

@testset "COD TI aligned — cross-validation vs residue (true TI)" begin
    # Sevostianov-Yilmaz-Kushch-Levin (2005) test stiffness.
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5), EllipticCrack(1.0, 0.2))
        B_ana = cod_tensor(c, C_TI)                                   # analytical
        B_res = cod_tensor(c, C_TI; method = :residues)
        diff = maximum(abs, components_canon(B_ana - B_res))
        @test diff < ATOL_COD_RES
    end
end

@testset "COD TI aligned — cross-validation vs DECUHR (true TI)" begin
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5))
        B_ana = cod_tensor(c, C_TI)
        B_dec = cod_tensor(c, C_TI; method = :decuhr)
        diff = maximum(abs, components_canon(B_ana - B_dec))
        @test diff < ATOL_COD_DEC
    end
end

@testset "COD TI aligned — dispatcher selects Analytical when aligned" begin
    # Crack basis defaults to canonical with normal = e₃.
    n_axis = [0.0, 0.0, 1.0]
    C_TI = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)

    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5), RibbonCrack(1.0))
        algo = MeanFieldHomogenization.Core._resolve_algo(Val(:auto), c, C_TI)
        @test algo isa MeanFieldHomogenization.Core.Analytical
    end
end

@testset "COD TI aligned — dispatcher falls back when non-aligned" begin
    # TI axis = e₁ but the crack normal stays e₃ → not aligned.
    n_axis = [1.0, 0.0, 0.0]
    C_TI = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)
    c = EllipticCrack(1.0, 0.5)
    # `:auto` falls back to a CUBATURE, never to the residue algorithm: the
    # latter degenerates on a reference that is anisotropic in type and
    # isotropic in value, so it is reachable on `:residues` only.
    algo = MeanFieldHomogenization.Core._resolve_algo(Val(:auto), c, C_TI)
    @test algo isa Union{MeanFieldHomogenization.Core.DECUHR, MeanFieldHomogenization.Core.NestedQuadGK}
    @test MeanFieldHomogenization.Core._resolve_algo(Val(:residues), c, C_TI) isa
        MeanFieldHomogenization.Core.Residue
    # And the residue path still produces a finite COD tensor.
    B = cod_tensor(c, C_TI; method = :residues)
    @test all(isfinite, get_array(B))
end

@testset "COD TI aligned — ForwardDiff compatibility" begin
    using ForwardDiff
    n_axis = [0.0, 0.0, 1.0]
    c_penny = PennyCrack(1.0)
    C0 = [2.179, 0.579, 0.689, 10.345, 1.0]

    f = C -> begin
        C_TI = tens_ti(C[1], C[2], C[3], C[4], C[5], n_axis)
        get_array(cod_tensor(c_penny, C_TI))[3, 3]
    end
    grad = ForwardDiff.gradient(f, C0)
    @test length(grad) == 5
    @test all(isfinite, grad)

    # Also through the elliptic-crack aspect ratio
    g = b_minor -> begin
        c_e = EllipticCrack(1.0, b_minor)
        C_TI = tens_ti(C0..., n_axis)
        get_array(cod_tensor(c_e, C_TI))[3, 3]
    end
    dval = ForwardDiff.derivative(g, 0.5)
    @test isfinite(dval)
end

@testset "COD TI aligned — complex moduli (frequency-domain viscoelasticity)" begin
    n_axis = [0.0, 0.0, 1.0]
    δ = 0.05
    C_c = tens_ti(
        2.179 + δ * im, 0.579 + 0.0im, 0.689 + 0.0im,
        10.345 + δ * im, 1.0 + 0.5δ * im, n_axis
    )
    @test eltype(C_c) <: Complex

    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5), RibbonCrack(1.0))
        B_c = cod_tensor(c, C_c)
        @test eltype(B_c) <: Complex
        @test all(isfinite, get_array(B_c))
    end

    # Limit Im → 0 must reproduce the real result exactly.
    C_r = tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, n_axis)
    C_0 = tens_ti(
        2.179 + 0.0im, 0.579 + 0.0im, 0.689 + 0.0im,
        10.345 + 0.0im, 1.0 + 0.0im, n_axis
    )
    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.5), RibbonCrack(1.0))
        B_r = cod_tensor(c, C_r)
        B_0 = cod_tensor(c, C_0)
        @test maximum(abs.(real.(get_array(B_0)) .- get_array(B_r))) < 1.0e-14
        @test maximum(abs.(imag.(get_array(B_0)))) < 1.0e-14
    end
end
