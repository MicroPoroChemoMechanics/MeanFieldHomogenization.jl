# =============================================================================
#  test_fracture_sets.jl — a background crossed by sets of fractures
#  (`fractured_stiffness`).
#
#  References, independent of the code under test:
#  - one set: the periodic laminate of the background and a spring interface
#    of compliance L Z every length L (`Laminate`, `AnisotropicSpringInterface`);
#  - the compliance of the medium written as the sum of the changes;
#  - the compliance of linear slip recovered by `fracture_compliance` of the
#    DiluteDual scheme, which adds the same changes.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff

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
