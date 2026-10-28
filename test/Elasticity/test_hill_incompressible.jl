# The Hill tensor of an incompressible anisotropic matrix, by the Green kernel
# inverted on the plane orthogonal to the direction: the closed form of a sphere
# in an isotropic matrix and its derivative, and the limit of the compressible
# Hill tensor as the bulk stiffness grows, its error decreasing as 1/κ.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using ForwardDiff

const EL_INC = MeanFieldHomogenization.Elasticity

# In the canonical basis, whatever the basis of the ellipsoid.
_hill_inc(ell, L; kw...) = TensND.change_tens_canon(EL_INC._hill_3d_aniso_nestedquadgk(ell, L; kernel = EL_INC._sym3_inv_acoustic_incompressible, kw...))

@testset "a sphere in an isotropic incompressible matrix" begin
    μ = 3.0
    P = _hill_inc(Ellipsoid(1.0), TensISO{3}(0.0, 2μ); reltol = 1.0e-10, abstol = 1.0e-13)
    @test KM(P) ≈ KM(TensISO{3}(0.0, 1 / (5μ))) atol = 1.0e-14
    dP = ForwardDiff.derivative(m -> KM(_hill_inc(Ellipsoid(1.0), TensISO{3}(0.0, 2m); reltol = 1.0e-10, abstol = 1.0e-13)), μ)
    @test dP ≈ KM(TensISO{3}(0.0, -1 / (5μ^2))) atol = 1.0e-12
end

@testset "an ellipsoid: the limit of the compressible Hill tensor" begin
    𝕁, 𝕂 = KM(TensISO{3}(1.0, 0.0)), KM(TensISO{3}(0.0, 1.0))
    Ldev = SMatrix{6, 6}(𝕂 * KM(TensND.tens_cubic(2.0, -1.0, 0.7, TensND.CanonicalBasis{3, Float64}())) * 𝕂)
    ell = Ellipsoid(2.0, 1.0, 0.5)
    kw = (reltol = 1.0e-10, abstol = 1.0e-13)
    Pi = KM(_hill_inc(ell, TensND.from_mandel(Ldev); kw...))
    gap(κ) = KM(TensND.change_tens_canon(EL_INC._hill_3d_aniso_nestedquadgk(ell, TensND.from_mandel(Ldev + 3κ * 𝕁); kw...))) - Pi
    g4, g5 = gap(1.0e4), gap(1.0e5)
    # P(κ) − P_inc = c/κ + O(1/κ²): κ·gap is the same at both bulk stiffnesses.
    @test maximum(abs, 1.0e4 * g4 - 1.0e5 * g5) < 1.0e-3 * maximum(abs, 1.0e4 * g4)
    @test maximum(abs, g5) < 1.0e-5
    # The kernel is the inverse of the acoustic tensor on the plane only.
    A = Array(TensND.components(TensND.from_mandel(Ldev), TensND.CanonicalBasis{3, Float64}()))
    ζ = (0.3, -0.5, 0.8)
    N = SMatrix{3, 3}(EL_INC._sym3_inv_acoustic_incompressible(A, ζ))
    n = SVector(ζ) / sqrt(sum(abs2, ζ))
    @test maximum(abs, N * n) < 1.0e-14
    @test N ≈ transpose(N) atol = 1.0e-14
end
