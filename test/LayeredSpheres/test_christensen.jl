using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  Multi-layer shear recurrence — validation against known limiting cases
#  (Christensen-Lo 1979 generalized self-consistent / Eshelby single-layer).
# =============================================================================

@testset "Shear — N=2 homogeneous gives β_k = 1" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    # All layers identical to matrix.
    s = LayeredSphere((0.5, 1.0), (C₀, C₀))
    β = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s, C₀)
    @test length(β) == 2
    for k in 1:2
        @test β[k] ≈ 1.0 rtol = 1.0e-10
    end
end

@testset "Shear — N=3 homogeneous gives β_k = 1" begin
    κ₀, μ₀ = 150.0, 90.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    s = LayeredSphere((0.3, 0.7, 1.0), (C₀, C₀, C₀))
    β = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s, C₀)
    for k in 1:3
        @test β[k] ≈ 1.0 rtol = 1.0e-10
    end
end

@testset "Shear — N=2 with outer shell ≡ matrix ↔ N=1 Eshelby" begin
    # If shell-layer moduli equal the matrix, the outer shell is
    # "invisible" and the effective inclusion reduces to the inner
    # sphere of radius a in matrix C₀.
    κ₀, μ₀ = 100.0, 70.0
    κ₁, μ₁ = 250.0, 160.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₁, 2μ₁)

    a, b = 0.4, 1.0
    s2 = LayeredSphere((a, b), (C₁, C₀))
    β2 = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s2, C₀)

    s1 = LayeredSphere((a,), (C₁,))
    β1 = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s1, C₀)[1]

    @test β2[1] ≈ β1        rtol = 1.0e-9
    @test β2[2] ≈ 1.0       rtol = 1.0e-10  # shell identical to matrix: β = 1
end

@testset "Shear — N=2 with core ≡ shell ↔ N=1 Eshelby (full sphere)" begin
    # Core and shell share moduli → behaves as a single-material sphere
    # of outer radius b.
    κ₀, μ₀ = 100.0, 70.0
    κ₁, μ₁ = 300.0, 200.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₁, 2μ₁)

    a, b = 0.6, 1.0
    s2 = LayeredSphere((a, b), (C₁, C₁))
    β2 = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s2, C₀)

    s1 = LayeredSphere((b,), (C₁,))
    β1 = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s1, C₀)[1]

    @test β2[1] ≈ β1 rtol = 1.0e-9
    @test β2[2] ≈ β1 rtol = 1.0e-9
end

@testset "Shear — stiffness_contribution respects single-layer Eshelby" begin
    κ₀, μ₀ = 100.0, 70.0
    κ₁, μ₁ = 250.0, 160.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₁, 2μ₁)

    # N=2 with outer shell = matrix ⇒ same μ contribution as a N=1
    # sphere of radius a containing C₁ at volume fraction (a/b)³.
    a, b = 0.4, 1.0
    s2 = LayeredSphere((a, b), (C₁, C₀))
    N2 = stiffness_contribution(s2, C₀)

    # Reference: single-sphere Eshelby result scaled by (a/b)³.
    ell = Ellipsoid(a)
    N_ref_ell = stiffness_contribution(ell, C₁, C₀)
    scale = (a / b)^3
    # Compare shear coefficients.
    @test N2.data[2] ≈ scale * N_ref_ell.data[2] rtol = 1.0e-8
    @test N2.data[1] ≈ scale * N_ref_ell.data[1] rtol = 1.0e-8
end

@testset "Shear — quasi-incompressible shell stays finite" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    μ₁ = 80.0
    for ν₁ in (0.3, 0.49, 0.499, 0.4999)
        κ₁ = 2 * μ₁ * (1 + ν₁) / (3 * (1 - 2 * ν₁))
        C₁ = TensISO{3}(3κ₁, 2μ₁)
        s = LayeredSphere((0.5, 1.0), (C₁, C₀))
        β = MeanFieldHomogenization.LayeredSpheres._shear_concentration(s, C₀)
        @test all(isfinite, β)
    end
end
