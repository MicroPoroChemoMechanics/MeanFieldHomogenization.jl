# =============================================================================
#  test_fracture_sets.jl — a background crossed by sets of fractures
#  (`fractured_stiffness`, `orthogonal_fracture_compliances`).
#
#  References, independent of the code under test:
#  - one set: the periodic laminate of the background and a spring interface
#    of compliance L Z every length L (`Laminate`, `AnisotropicSpringInterface`);
#  - the compliance of the medium written as the sum of the changes;
#  - the compliance of linear slip recovered by `fracture_compliance` of the
#    DiluteDual scheme, which adds the same changes;
#  - for three orthogonal sets (`orthogonal_fracture_compliances`): the changes
#    built from given compliances, in a frame drawn at random, and the values of
#    Rasolofosaon et al. (2000) for the KTB gneiss (data/literature).
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Random: MersenneTwister
using JSON
import Symbolics

const FS_CB = iso_stiffness(20.0, 12.0)

@testset "Sets of fractures" begin
    L = 0.5
    𝕂 = SMatrix{3, 3}([3.0e-3 5.0e-4 7.0e-4; 5.0e-4 8.0e-3 2.0e-4; 7.0e-4 2.0e-4 4.0e-3])
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => FS_CB); thickness = L, interface = AnisotropicSpringInterface(𝕂))
    C_lam = homogenize(lam, Laminated(), :C)
    C_fs = @inferred fractured_stiffness(FS_CB, [0.0, 0.0, 1.0] => Matrix(𝕂) / L)
    @test Matrix(mandel(C_fs)) ≈ Matrix(mandel(C_lam)) rtol = 1.0e-12
    # two sets: the compliances add
    n₁, n₂ = [0.0, 0.0, 1.0], normalize([1.0, 1.0, 0.5])
    C₂ = fractured_stiffness(FS_CB, n₁ => (0.01, 0.02), n₂ => (0.005, 0.008))
    S_ref = inv(FS_CB) + linear_slip_change(n₁, (0.01, 0.02)) + linear_slip_change(n₂, (0.005, 0.008))
    @test Matrix(mandel(inv(C₂))) ≈ Matrix(mandel(S_ref)) rtol = 1.0e-12
    # the derivative of a stiffness component with respect to Z_N, against the
    # derivative of the inverse: dC/dZ = −C : dS/dZ : C
    C33(z) = Matrix(mandel(fractured_stiffness(FS_CB, n₁ => (z, 0.02))))[3, 3]
    C = Matrix(mandel(fractured_stiffness(FS_CB, n₁ => (0.01, 0.02))))
    dS = Matrix(mandel(linear_slip_change(n₁, (1.0, 0.0))))
    @test ForwardDiff.derivative(C33, 0.01) ≈ -(C * dS * C)[3, 3] rtol = 1.0e-12
end

# Three orthogonal sets, drawn in a frame drawn at random: the six compliances
# come back exactly, whatever couplings outside the pattern of the sets are added.
@testset "Three orthogonal sets: compliances from a change of compliance" begin
    R = Matrix(qr(randn(MersenneTwister(2000), 3, 3)).Q)
    Z_N, Z_T = (0.011, 0.052, 0.003), (0.017, 0.046, 0.006)
    ΔS = sum(linear_slip_change(R[:, i], (Z_N[i], Z_T[i])) for i in 1:3)
    f = orthogonal_fracture_compliances(ΔS, R)
    @test all(f.Z_N .≈ Z_N) && all(f.Z_T .≈ Z_T)
    @test all(isapprox(canonical_components(f.normals[i]), R[:, i]) for i in 1:3)
    # the converse
    C = fractured_stiffness(FS_CB, (R[:, i] => (Z_N[i], Z_T[i]) for i in 1:3)...)
    @test Matrix(mandel(inv(C) - inv(FS_CB))) ≈ Matrix(mandel(ΔS)) rtol = 1.0e-12
    # a coupling between the normal compliances of the frame is left out
    P(a, b) = Tens(R[:, a] * transpose(R[:, a])) ⊗ Tens(R[:, b] * transpose(R[:, b]))
    g = orthogonal_fracture_compliances(ΔS + 0.004 * (P(1, 2) + P(2, 1)), R)
    @test all(g.Z_N .≈ Z_N) && all(g.Z_T .≈ Z_T)
    # a basis of TensND as the frame
    h = orthogonal_fracture_compliances(ΔS, TensND.Basis(R))
    @test all(h.Z_N .≈ Z_N) && all(h.Z_T .≈ Z_T)
    # dual numbers: the derivative of the compliance read with respect to the one given
    @test ForwardDiff.derivative(z -> orthogonal_fracture_compliances(linear_slip_change(R[:, 2], (z, 0.01)), R).Z_N[2], 0.03) ≈ 1
    @test ForwardDiff.derivative(z -> orthogonal_fracture_compliances(linear_slip_change(R[:, 2], (0.03, z)), R).Z_T[2], 0.01) ≈ 1
    # symbols, in the canonical frame
    Symbolics.@variables zn zt
    s = orthogonal_fracture_compliances(linear_slip_change([0, 0, 1], (zn, zt)), Matrix{Int}(I, 3, 3))
    @test isequal(Symbolics.simplify(s.Z_N[3] - zn), 0) && isequal(Symbolics.simplify(s.Z_T[3] - zt), 0)
    @test isequal(Symbolics.simplify(s.Z_T[1]), 0)
end

# The KTB gneiss of Rasolofosaon et al. (2000): their excess compliance at
# 10 MPa, written in the crack frame (eq. 2), gives their compliances of the
# three sets (eq. 3) and, normalized, their eq. (4).
@testset "Three orthogonal sets: the KTB gneiss" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "rasolofosaonGJI2000.json"))
    v = Dict(r["component"] => r["value"] for r in lit["tables"]["excess_compliance_ort_10MPa"])
    V = zeros(6, 6)
    for (c, x) in v
        i, j = parse(Int, c[1]), parse(Int, c[2])
        V[i, j] = V[j, i] = x
    end
    ΔS = TensND.compliance_from_voigt(V)                        # 1e-12 Pa⁻¹
    f = orthogonal_fracture_compliances(ΔS, Matrix{Float64}(I, 3, 3))
    for (i, row) in enumerate(lit["tables"]["fracture_compliances_10MPa"])
        @test f.Z_N[i] ≈ row["S_N"] atol = 0.006                 # printed with two decimals
        @test f.Z_T[i] ≈ row["S_T"] atol = 0.006
    end
    q = lit["quantities"]
    C11, C44 = q["iso_C11_intact"]["value"], q["iso_C44_intact"]["value"]
    for (i, row) in enumerate(lit["tables"]["normalized_compliances_10MPa"])
        @test 100 * f.Z_N[i] * C11 * 1.0e-3 ≈ row["E_N_percent"] atol = 0.06
        @test 100 * f.Z_T[i] * C44 * 1.0e-3 ≈ row["E_T_percent"] atol = 0.06
    end
    # the isotropic constants of the intact rock are the projection of its
    # stiffness (Fig. 5a), not of its compliance
    M = [r["c$j"] for r in lit["tables"]["stiffness_400MPa"], j in 1:6]
    iso = best_fit_iso(TensND.stiffness_from_voigt(M))
    λμ = Matrix(TensND.voigt_stiffness(iso))
    @test λμ[1, 1] ≈ C11 atol = 0.005
    @test λμ[4, 4] ≈ C44 atol = 0.005
end
