using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  LayeredSphere — tests.
# =============================================================================

@testset "LayeredSphere — geometry accessors" begin
    C_a = TensISO{3}(300.0, 200.0)
    C_b = TensISO{3}(100.0, 80.0)

    s = LayeredSphere((0.5, 0.8, 1.0), (C_a, C_b, C_a))
    @test layer_count(s) == 3
    @test layer_radius(s, 1) == 0.5
    @test layer_radius(s, 3) == 1.0
    @test outer_radius(s) == 1.0
    @test layer_modulus(s, 1) === C_a
    @test layer_modulus(s, 2) === C_b
    @test layer_volume_fraction(s, 1) ≈ 0.5^3 rtol = 1.0e-12
    @test layer_volume_fraction(s, 2) ≈ (0.8^3 - 0.5^3) rtol = 1.0e-12
    @test layer_volume_fraction(s, 3) ≈ (1.0^3 - 0.8^3) rtol = 1.0e-12

    # Interfaces default to perfect
    for k in 1:3
        @test layer_interface(s, k) isa PerfectInterface
    end

    # Ascending-radii invariant
    @test_throws ArgumentError LayeredSphere((1.0, 0.5), (C_a, C_b))
end

@testset "LayeredSphere — single-layer ≡ Ellipsoid Eshelby" begin
    κ₀, μ₀ = 100.0, 70.0
    κ₁, μ₁ = 200.0, 140.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₁, 2μ₁)

    s = LayeredSphere((1.0,), (C₁,))
    A_layered = strain_strain_concentration(s, C₀; layer = 1)
    A_ell = strain_strain_concentration(Ellipsoid(1.0), C₁, C₀)
    # Both should match to floating-point precision (the two paths
    # differ only in round-off — bulk via (u,σ) propagation vs direct
    # Eshelby iso inversion).
    @test A_layered.data[1] ≈ A_ell.data[1] rtol = 1.0e-14
    @test A_layered.data[2] ≈ A_ell.data[2] rtol = 1.0e-14

    # Bulk concentration matches Eshelby sphere formula
    α_bulk = (3κ₀ + 4μ₀) / (3κ₁ + 4μ₀)
    @test A_layered[1, 1, 1, 1] ≈ α_bulk / 3 + 2 * A_layered.data[2] / 3 rtol = 1.0e-12
end

@testset "LayeredSphere — bulk concentration and effective bulk" begin
    # Homogeneous composite (all layers = matrix): α_k ≡ 1 and κ_eff = κ₀.
    κ₀, μ₀ = 120.0, 80.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    s = LayeredSphere((0.4, 0.7, 1.0), (C₀, C₀, C₀))
    α = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s, κ₀, μ₀)
    @test all(≈(1), α)
    κ_eff = MeanFieldHomogenization.LayeredSpheres._effective_bulk(s, κ₀, μ₀)
    @test κ_eff ≈ κ₀ rtol = 1.0e-12

    # Monolithic composite (all layers = C₁ ≠ C₀): reduces to single-layer
    κ₁, μ₁ = 3κ₀, 2μ₀
    C₁ = TensISO{3}(3κ₁, 2μ₁)
    s_mono = LayeredSphere((0.4, 0.7, 1.0), (C₁, C₁, C₁))
    α_mono = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s_mono, κ₀, μ₀)
    α_expected = (3κ₀ + 4μ₀) / (3κ₁ + 4μ₀)
    for k in 1:3
        @test α_mono[k] ≈ α_expected rtol = 1.0e-10
    end
end

@testset "LayeredSphere — layer and sphere averages (hydrostatic loading)" begin
    κ₀, μ₀ = 100.0, 70.0
    κ_in = 200.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C_in = TensISO{3}(3κ_in, 2μ₀)  # same shear, stiffer bulk (core)
    s = LayeredSphere((0.5, 1.0), (C_in, C₀))
    ε∞ = TensND.Tens(Matrix(Diagonal([0.1, 0.1, 0.1])))

    α = MeanFieldHomogenization.LayeredSpheres._bulk_concentration(s, κ₀, μ₀)
    # Layer 2 (shell = matrix): α[2] should be close to 1 for this case
    # (consistent with the generalized self-consistent ... the shell's
    # average strain is not exactly 1 because the core perturbs it,
    # but the trace average across the whole composite must equal ε∞ when
    # the matrix surrounds a composite of the same material).
    # We just check sphere_strain_average trace vs exact dilute expectation.
    sph_avg = sphere_strain_average(s, C₀, ε∞)
    # Sum of per-layer averages weighted by volume fraction equals sphere_avg
    f1 = layer_volume_fraction(s, 1)
    f2 = layer_volume_fraction(s, 2)
    avg1 = layer_strain_average(s, C₀, ε∞, 1)
    avg2 = layer_strain_average(s, C₀, ε∞, 2)
    @test sph_avg ≈ f1 * avg1 + f2 * avg2 rtol = 1.0e-12

    # Cumulative at r = outer equals sphere_avg
    cum = cumulative_strain_average(s, C₀, ε∞, outer_radius(s))
    @test cum ≈ sph_avg rtol = 1.0e-12

    # Cumulative at r = r_1 equals layer_1 average
    cum1 = cumulative_strain_average(s, C₀, ε∞, layer_radius(s, 1))
    @test cum1 ≈ avg1 rtol = 1.0e-12
end

@testset "LayeredSphere — stiffness_contribution for single-layer" begin
    κ₀, μ₀ = 100.0, 70.0
    κ₁, μ₁ = 250.0, 180.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₁, 2μ₁)

    s = LayeredSphere((1.0,), (C₁,))
    N_layered = stiffness_contribution(s, C₀)
    N_ell = stiffness_contribution(Ellipsoid(1.0), C₁, C₀)
    # For single layer (f = 1), both should agree on the iso coefficients.
    @test N_layered.data[1] ≈ N_ell.data[1] rtol = 1.0e-10
    @test N_layered.data[2] ≈ N_ell.data[2] rtol = 1.0e-10
end

@testset "LayeredSphere — multi-layer strain average under deviatoric loading" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(200.0, 140.0)
    s2 = LayeredSphere((0.5, 1.0), (C₁, C₀))
    ε∞_sph = TensND.Tens(Matrix(Diagonal([0.1, 0.1, 0.1])))
    @test_nowarn layer_strain_average(s2, C₀, ε∞_sph, 1)

    # Deviatoric component ≠ 0: combines bulk α_k and shear β_k.
    ε∞_dev = TensND.Tens([0.1 0.0 0.0; 0.0 -0.05 0.0; 0.0 0.0 -0.05])
    avg = layer_strain_average(s2, C₀, ε∞_dev, 1)
    @test all(isfinite, Matrix(avg))
end
