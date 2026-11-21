# =============================================================================
#  test_neutral.jl — neutral inhomogeneities.
#
#  An inhomogeneity is neutral when it leaves the remote field undisturbed: its
#  contribution tensor vanishes. Benveniste & Miloh (1999) study them in
#  conduction. For a sphere of radius a the condition on the dipole is
#  elementary, and independent of the recurrence over the layers:
#   - Kapitza interface of resistance ρ:   k_m = k_i / (1 + ρ k_i / a);
#   - surface of conductance kˢ:           k_m = k_i + 2 kˢ / a;
#   - perfect coating (Hashin's composite sphere), in conduction and for the
#     bulk modulus.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

@testset "Neutral inhomogeneities" begin
    a, k_i = 0.7, 3.0
    iso(k) = TensISO{3}(k)
    contribution(sphere, k_m) = canonical_components(property_contribution(sphere, iso(k_m), iso(k_m)))

    @testset "a resistive interface hides a conducting sphere" begin
        ρ = 0.4
        k_m = k_i / (1 + ρ * k_i / a)
        sphere = LayeredSphere((a,), (iso(k_i),); interfaces = (KapitzaInterface(ρ),))
        @test maximum(abs, contribution(sphere, k_m)) < 1.0e-13
        # not neutral in another matrix, and the sign follows the contrast
        @test contribution(sphere, 0.9 * k_m)[1, 1] > 1.0e-2
        @test contribution(sphere, 1.1 * k_m)[1, 1] < -1.0e-2
        # more resistance lowers the contribution, through its neutral value
        slope = ForwardDiff.derivative(x -> contribution(LayeredSphere((a,), (iso(k_i),); interfaces = (KapitzaInterface(x),)), k_m)[1, 1], ρ)
        @test slope < 0
    end

    @testset "a conducting surface hides an insulating sphere" begin
        kˢ = 0.9
        k_m = k_i + 2kˢ / a
        sphere = LayeredSphere((a,), (iso(k_i),); interfaces = (SurfaceConductiveInterface(kˢ),))
        @test maximum(abs, contribution(sphere, k_m)) < 1.0e-13
        @test contribution(sphere, 0.9 * k_m)[1, 1] > 1.0e-2
    end

    @testset "a coating that hides its core" begin
        k_c, k_s, b = 9.0, 0.5, 1.0
        φ = (a / b)^3
        k_m = k_s * (2k_s + k_c + 2φ * (k_c - k_s)) / (2k_s + k_c - φ * (k_c - k_s))
        @test maximum(abs, contribution(LayeredSphere((a, b), (iso(k_c), iso(k_s))), k_m)) < 1.0e-13
        # bulk modulus: neutral under a hydrostatic loading, not under shear
        K_c, μ_c, K_s, μ_s, μ_m = 12.0, 5.0, 1.0, 0.6, 1.1
        K_m = K_s + φ * (K_c - K_s) * (3K_s + 4μ_s) / (3K_s + 4μ_s + 3 * (1 - φ) * (K_c - K_s))
        C_m = iso_stiffness(K_m, μ_m)
        N = canonical_components(property_contribution(LayeredSphere((a, b), (iso_stiffness(K_c, μ_c), iso_stiffness(K_s, μ_s))), C_m, C_m))
        @test abs(sum(N[i, i, j, j] for i in 1:3, j in 1:3)) < 1.0e-12
        @test abs(N[1, 2, 1, 2]) > 1.0e-2
    end
end
