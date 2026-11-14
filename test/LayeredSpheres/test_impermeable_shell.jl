using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_impermeable_shell.jl — a layer of zero conductivity at any rank of a
#  layered sphere. No flux crosses it: the layers it encloses carry no field,
#  and its own field is the limit of a vanishing conductivity, which is the
#  reference here (a conductivity of 1e-10), with perfect, Kapitza and
#  surface-conductive interfaces on either side.
# =============================================================================

@testset "LayeredSphere conduction — impermeable shell" begin
    K(x) = TensISO{3}(x)
    K₀ = K(2.0)
    radii = (0.4, 0.7, 1.0)
    pc = MeanFieldHomogenization.property_contribution
    α(s, k; kw...) = TensND.get_data(gradient_gradient_concentration(s, K₀; layer = k, kw...))[1]
    for interfaces in (
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
            (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3), KapitzaInterface(0.1)),
            (SurfaceConductiveInterface(0.3), KapitzaInterface(0.2), SurfaceConductiveInterface(0.1)),
        )
        zero_k = LayeredSphere(radii, (K(5.0), K(0.0), K(1.0)); interfaces)
        small_k = LayeredSphere(radii, (K(5.0), K(1.0e-10), K(1.0)); interfaces)
        @test gradient_gradient_concentration(zero_k, K₀, K₀) ≈ gradient_gradient_concentration(small_k, K₀, K₀) rtol = 1.0e-7
        @test pc(zero_k, K₀) ≈ pc(small_k, K₀) rtol = 1.0e-7
        @test flux_gradient_concentration(zero_k, K₀, K₀) ≈ flux_gradient_concentration(small_k, K₀, K₀) rtol = 1.0e-7
        @test all(isapprox(α(zero_k, k; external = true), α(small_k, k; external = true); rtol = 1.0e-6, atol = 1.0e-7) for k in 1:3)
        # the enclosed core carries no gradient at all
        @test α(zero_k, 1) == 0.0
        # pointwise: the same limit in every region
        G = TensND.Vec{3}((0.3, -0.2, 0.9))
        f_zero, f_small = LayeredSphereTransportFields(zero_k, K₀), LayeredSphereTransportFields(small_k, K₀)
        for r in (0.2, 0.55, 0.85, 1.6)
            x = r .* (0.48, 0.6, 0.64)
            @test isapprox(collect(local_gradient(f_zero, x, G)), collect(local_gradient(f_small, x, G)); rtol = 1.0e-6, atol = 1.0e-7)
            @test isapprox(local_temperature(f_zero, x, G), local_temperature(f_small, x, G); rtol = 1.0e-6, atol = 1.0e-7)
        end
        # the shell is insulated on both faces, and its field vanishes on the inner one
        @test local_temperature(f_zero, 0.4, 0.3, 0.2, G; side = :outer) ≈ 0 atol = 1.0e-14
        @test norm(collect(local_flux(f_zero, 0.55, 0.3, 0.2, G))) == 0.0
    end
    # two adjacent impermeable layers leave the inner one undefined
    @test_throws "merge the two layers" gradient_gradient_concentration(LayeredSphere(radii, (K(0.0), K(0.0), K(1.0))), K₀, K₀)
    # an impermeable core keeps the gradient its conducting shell imposes on it
    core = LayeredSphere((0.5, 1.0), (K(0.0), K(3.0)))
    @test α(core, 1) ≈ α(LayeredSphere((0.5, 1.0), (K(1.0e-12), K(3.0))), 1) rtol = 1.0e-8
end
