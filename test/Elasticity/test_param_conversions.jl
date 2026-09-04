# =============================================================================
#  test_param_conversions.jl — physical readings of the symmetry-class
#  coefficients (`src/Elasticity/param_conversions.jl`).
#
#  Covers the two exported families:
#    • isotropic   : k_mu / iso_stiffness, E_nu / iso_stiffness_E_nu
#    • TI (Hoenig) : hoenig_params / hoenig_stiffness, including the
#      two-argument form that projects onto the TI span first.
#
#  Every pair is checked both ways (round trip), against the closed-form
#  relations of isotropic elasticity, and on the degenerate case of a TI
#  tensor that is in fact isotropic.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

@testset "k_mu / iso_stiffness — round trip" begin
    k, mu = 30.0, 12.0
    C = iso_stiffness(k, mu)
    @test C isa TensND.TensISO{4}

    # C = 3k·𝕁 + 2μ·𝕂: the raw coefficients are (3k, 2μ).
    @test collect(TensND.get_data(C)) ≈ [3k, 2mu]

    k_back, mu_back = k_mu(C)
    @test k_back ≈ k
    @test mu_back ≈ mu

    # The compliance S = C⁻¹ is read with the same call, through inv (see the
    # header of param_conversions.jl: there is no `_compliance` variant).
    S = inv(C)
    k_S, mu_S = k_mu(inv(S))
    @test k_S ≈ k
    @test mu_S ≈ mu
end

@testset "E_nu / iso_stiffness_E_nu — round trip and consistency with (k, μ)" begin
    E, nu = 210.0, 0.3
    C = iso_stiffness_E_nu(E, nu)
    @test C isa TensND.TensISO{4}

    E_back, nu_back = E_nu(C)
    @test E_back ≈ E
    @test nu_back ≈ nu

    # Consistency with the closed forms k = E/(3(1-2ν)), μ = E/(2(1+ν)).
    k, mu = k_mu(C)
    @test k ≈ E / (3 * (1 - 2nu))
    @test mu ≈ E / (2 * (1 + nu))

    # And the other way round, starting from (k, μ).
    E2, nu2 = E_nu(iso_stiffness(k, mu))
    @test E2 ≈ E
    @test nu2 ≈ nu

    # ν = 0: k = E/3 and μ = E/2.
    C0 = iso_stiffness_E_nu(100.0, 0.0)
    k0, mu0 = k_mu(C0)
    @test k0 ≈ 100.0 / 3
    @test mu0 ≈ 50.0
    @test all(isapprox.(E_nu(C0), (100.0, 0.0); atol = 1.0e-12))
end

@testset "hoenig_params / hoenig_stiffness — round trip" begin
    axis = [0.0, 0.0, 1.0]
    E1, h, nu1, nu2, gamma = 12.0, 2.5, 0.25, 0.2, 1.4

    C = hoenig_stiffness(E1, h, nu1, nu2, gamma, axis)
    @test C isa TensND.TensTI{4}

    p = hoenig_params(C)
    @test p.E1 ≈ E1
    @test p.h ≈ h
    @test p.nu1 ≈ nu1
    @test p.nu2 ≈ nu2
    @test p.gamma ≈ gamma

    # The parameters are named: the documented order must be honored.
    @test collect(p) ≈ [E1, h, nu1, nu2, gamma]

    # Full round trip on the tensor itself, not only on the parameters.
    C_back = hoenig_stiffness(p.E1, p.h, p.nu1, p.nu2, p.gamma, axis)
    @test Array(C_back) ≈ Array(C)
end

@testset "hoenig_params — non-canonical axis" begin
    axis = normalize([1.0, 1.0, 0.0])
    E1, h, nu1, nu2, gamma = 20.0, 1.8, 0.3, 0.15, 0.9

    C = hoenig_stiffness(E1, h, nu1, nu2, gamma, axis)
    p = hoenig_params(C)

    # Hoenig's parameters are intrinsic: they do not depend on the
    # orientation of the symmetry axis.
    @test p.E1 ≈ E1
    @test p.h ≈ h
    @test p.nu1 ≈ nu1
    @test p.nu2 ≈ nu2
    @test p.gamma ≈ gamma
end

@testset "hoenig_params — two-argument form (projection first)" begin
    axis = [0.0, 0.0, 1.0]
    E1, h, nu1, nu2, gamma = 15.0, 2.0, 0.28, 0.18, 1.1

    C_ti = hoenig_stiffness(E1, h, nu1, nu2, gamma, axis)

    # A TensTI already in the TI span must project onto itself: the
    # two-argument form must then return exactly the same parameters as the
    # one-argument form.
    C_full = Tens(Array(C_ti))
    p2 = hoenig_params(C_full, axis)
    p1 = hoenig_params(C_ti)

    @test p2.E1 ≈ p1.E1
    @test p2.h ≈ p1.h
    @test p2.nu1 ≈ p1.nu1
    @test p2.nu2 ≈ p1.nu2
    @test p2.gamma ≈ p1.gamma
end

@testset "hoenig_params — degenerate isotropic case" begin
    # A TI built from an isotropic tensor must return γ = 1 (no shear
    # anisotropy) and fall back on the isotropic ν.
    E, nu = 70.0, 0.25
    C_iso = iso_stiffness_E_nu(E, nu)
    axis = [0.0, 0.0, 1.0]

    C_ti, = TensND.proj_tens(Val(:TI), Array(C_iso), axis)
    p = hoenig_params(C_ti)

    @test p.gamma ≈ 1.0
    @test p.h ≈ 1.0
    @test p.nu1 ≈ nu
    @test p.nu2 ≈ nu
    @test p.E1 ≈ E
end

@testset "param_conversions — TI compliance through inv" begin
    axis = [0.0, 0.0, 1.0]
    C = hoenig_stiffness(18.0, 2.2, 0.26, 0.17, 1.25, axis)
    S = inv(C)

    # inv of a TensTI stays in the TI class, so hoenig_params(inv(S)) must
    # return the parameters of C.
    p = hoenig_params(inv(S))
    q = hoenig_params(C)
    @test p.E1 ≈ q.E1
    @test p.h ≈ q.h
    @test p.nu1 ≈ q.nu1
    @test p.nu2 ≈ q.nu2
    @test p.gamma ≈ q.gamma
end
