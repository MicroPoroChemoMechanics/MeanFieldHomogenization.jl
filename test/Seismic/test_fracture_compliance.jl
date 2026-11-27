# =============================================================================
#  test_fracture_compliance.jl — the compliance of linear slip of a crack
#  family (`slip_compliance`, `fracture_compliance`).
#
#  References, independent of the code under test:
#  - dry penny cracks of density ε, dilute, in an isotropic matrix of Young's
#    modulus E and Poisson's ratio ν: Z_N = 16(1 − ν²)ε/(3E) and
#    Z_T = 32(1 − ν²)ε/(3(2 − ν)E) (Kachanov 1992, Appl. Mech. Rev. 45);
#  - the change of compliance written index by index from Z,
#    ΔS_ijkl = ¼(n_i Z_jk n_l + n_j Z_ik n_l + n_i Z_jl n_k + n_j Z_il n_k);
#  - the share of the family of `crack_family_compliances`, recovered.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using Random
import Symbolics

const FC_K, FC_MU = 20.0, 12.0
const FC_E = 9FC_K * FC_MU / (3FC_K + FC_MU)
const FC_NU = (3FC_K - 2FC_MU) / (2 * (3FC_K + FC_MU))

function _fc_cell(ε; angles = (0.4, 0.9, 0.0), μ = FC_MU)
    C_s = iso_stiffness(FC_K, μ)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0; euler_angles = angles), Dict(:C => 0 * C_s); density = ε)
    return rve
end

_fc_change(n, Z) = [
    (n[i] * Z[j, k] * n[l] + n[j] * Z[i, k] * n[l] + n[i] * Z[j, l] * n[k] + n[j] * Z[i, l] * n[k]) / 4
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
]

@testset "Compliance of linear slip" begin
    @testset "Z read on a change of compliance" begin
        rng = Xoshiro(20261128)
        for _ in 1:3
            n = normalize(randn(rng, 3))
            A = randn(rng, 3, 3)
            Z = A + transpose(A)
            ΔS = express(Tens(_fc_change(n, Z)), random_basis(rng))      # written in a basis drawn at random
            @test Array(canonical_components(slip_compliance(ΔS, n))) ≈ Z rtol = 1.0e-12
            @test Array(canonical_components(linear_slip_change(n, Z))) ≈ _fc_change(n, Z) rtol = 1.0e-12
        end
        n = [0.0, 0.0, 1.0]
        @test Array(canonical_components(linear_slip_change(n, (2.0, 3.0)))) ≈ _fc_change(n, Diagonal([3.0, 3.0, 2.0])) rtol = 1.0e-14
    end

    @testset "dilute penny cracks against the closed forms" begin
        ε = 0.05
        for scheme in (DiluteDual(:M), MoriTanaka(:M))
            f = fracture_compliance(_fc_cell(ε), scheme, :F)
            @test f.Z_N ≈ 16 * (1 - FC_NU^2) * ε / (3FC_E) rtol = 1.0e-12
            @test f.Z_T ≈ 32 * (1 - FC_NU^2) * ε / (3 * (2 - FC_NU) * FC_E) rtol = 1.0e-12
            # Z is that of a set invariant by rotation about its normal
            n = Array(canonical_components(f.normal))
            @test Array(canonical_components(f.Z)) ≈ f.Z_N * n * transpose(n) + f.Z_T * (I - n * transpose(n)) rtol = 1.0e-12
        end
        # the other schemes: their own share, Z read on it exactly
        for scheme in (Dilute(:M), SelfConsistent(; abstol = 1.0e-14, reltol = 1.0e-14))
            rve = _fc_cell(ε)
            C_hom = homogenize(rve, scheme)
            share = crack_family_compliances(rve, scheme, C_hom).families[:F]
            ΔS = delta_inverse_property(rve.phases[:F].geometry, share, ε)
            f = fracture_compliance(rve, scheme, :F; C_hom)
            n = Array(canonical_components(f.normal))
            @test Array(canonical_components(ΔS)) ≈ _fc_change(n, Array(canonical_components(f.Z))) rtol = 1.0e-9
        end
        @test_throws ArgumentError fracture_compliance(_fc_cell(ε), DiluteDual(:M), :M)
    end

    @testset "derivatives and symbols" begin
        # dZ_N/dε at fixed matrix: the closed form is linear in ε
        @test ForwardDiff.derivative(ε -> fracture_compliance(_fc_cell(ε), DiluteDual(:M), :F).Z_N, 0.05) ≈
            16 * (1 - FC_NU^2) / (3FC_E) rtol = 1.0e-12
        # with respect to the shear modulus of the matrix, nested
        ZN(μ) = fracture_compliance(_fc_cell(0.05; μ), DiluteDual(:M), :F).Z_N
        ref(μ) = (E = 9FC_K * μ / (3FC_K + μ); ν = (3FC_K - 2μ) / (2 * (3FC_K + μ)); 16 * (1 - ν^2) * 0.05 / (3E))
        d²(f) = ForwardDiff.derivative(a -> ForwardDiff.derivative(f, a), FC_MU)
        @test d²(ZN) ≈ d²(ref) rtol = 1.0e-9
        Symbolics.@variables a b c
        Zs = [a b 0; b c 0; 0 0 a]
        n = [0, 0, 1]
        Zr = slip_compliance(linear_slip_change(n, Zs), n)
        @test all(isequal.(Symbolics.simplify.(Array(canonical_components(Zr)) .- Zs), 0))
    end
end
