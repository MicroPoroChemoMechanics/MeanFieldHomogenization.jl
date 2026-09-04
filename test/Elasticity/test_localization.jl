using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy   # top level on purpose: `@syms` below is expanded before any
# `using` nested in a @testset body would have run.

# =============================================================================
#  Localization tensors for ellipsoids (dilute Eshelby).
# =============================================================================

@testset "strain_strain_loc — iso sphere, iso matrix" begin
    # Analytical Eshelby for a sphere in iso matrix: A_εε is isotropic,
    # its two moduli are α = 1/(1 + 3k₀ₚ·Δk) and β similar for shear.
    # We cross-check the diagonal (1,1,1,1) entry and identity limit.
    E₀, ν₀ = 200.0, 0.3
    k₀ = E₀ / (3 * (1 - 2ν₀))
    μ₀ = E₀ / (2 * (1 + ν₀))
    C₀ = TensISO{3}(3k₀, 2μ₀)

    ell = Ellipsoid(1.0)

    # C₁ = C₀ → A = 𝕀
    A₀ = strain_strain_loc(ell, C₀, C₀)
    @test A₀[1, 1, 1, 1] ≈ 1.0 rtol = 1.0e-12
    @test A₀[1, 2, 1, 2] ≈ 0.5 rtol = 1.0e-12

    # Stiffer inclusion: A_εε[1,1,1,1] < 1
    C₁ = TensISO{3}(3k₀ * 2, 2μ₀ * 2)
    A = strain_strain_loc(ell, C₁, C₀)
    @test 0 < A[1, 1, 1, 1] < 1
end

@testset "4 localization tensors are consistent" begin
    E₀, ν₀ = 1.0, 0.25
    k₀ = E₀ / (3 * (1 - 2ν₀))
    μ₀ = E₀ / (2 * (1 + ν₀))
    C₀ = TensISO{3}(3k₀, 2μ₀)
    C₁ = TensISO{3}(3k₀ * 1.8, 2μ₀ * 1.4)
    ell = Ellipsoid(1.0)

    A_εε = strain_strain_loc(ell, C₁, C₀)
    A_σε = stress_strain_loc(ell, C₁, C₀)
    A_εσ = strain_stress_loc(ell, C₁, C₀)
    A_σσ = stress_stress_loc(ell, C₁, C₀)

    # A_σε = C₁ : A_εε
    lhs = A_σε
    rhs = C₁ ⊡ A_εε
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        @test lhs[i, j, k, l] ≈ rhs[i, j, k, l] rtol = 1.0e-10
    end
    # A_εσ = A_εε : S₀
    S₀ = inv(C₀)
    rhs2 = A_εε ⊡ S₀
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        @test A_εσ[i, j, k, l] ≈ rhs2[i, j, k, l] rtol = 1.0e-10
    end
    # A_σσ = C₁ : A_εε : S₀
    rhs3 = C₁ ⊡ A_εε ⊡ S₀
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        @test A_σσ[i, j, k, l] ≈ rhs3[i, j, k, l] rtol = 1.0e-10
    end
end

@testset "strain_strain_loc — triaxial ellipsoid, aniso stiffness" begin
    # Hand-picked triclinic stiffness (same reference as test_anisotropic.jl)
    ℬ = CanonicalBasis{3, Float64}()
    _KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C₀ = TensND.inv_KM(_KM_tri, ℬ)
    C₁ = 2.0 * C₀   # same symmetry, doubled
    ell = Ellipsoid(3.0, 2.0, 1.0)

    A = strain_strain_loc(ell, C₁, C₀; method = :residues)
    A_nqg = strain_strain_loc(ell, C₁, C₀; method = :nestedquadgk, reltol = 1.0e-12)
    scale = maximum(abs(A[i, j, k, l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        @test abs(A[i, j, k, l] - A_nqg[i, j, k, l]) < 1.0e-7 * scale
    end
end

@testset "Localization — symbolic propagation (SymPy)" begin
    @syms k0 m0 kk1 mm1
    C₀ = TensISO{3}(3k0, 2m0)
    C₁ = TensISO{3}(3kk1, 2mm1)
    ell = Ellipsoid(1.0)
    A = strain_strain_loc(ell, C₁, C₀)
    # Expression should be symbolic in k0, m0, kk1, mm1
    @test string(A[1, 1, 1, 1]) isa String    # smoke
    # A collapses to identity at C₁ = C₀
    A_id = strain_strain_loc(ell, C₀, C₀)
    @test simplify(A_id[1, 1, 1, 1] - 1) == 0
end

@testset "Localization — ForwardDiff.Dual" begin
    # Derivative of A_εε[3,3,3,3] w.r.t. Poisson's ratio of matrix
    ell = Ellipsoid(1.0)
    f = ν -> begin
        E = 1.0
        k = E / (3 * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        C₀ = TensISO{3}(3k, 2μ)
        C₁ = 2.0 * C₀
        strain_strain_loc(ell, C₁, C₀)[3, 3, 3, 3]
    end
    d = ForwardDiff.derivative(f, 0.3)
    # Finite-difference check
    ε = 1.0e-6
    fd = (f(0.3 + ε) - f(0.3 - ε)) / (2ε)
    @test d ≈ fd rtol = 1.0e-5
end
