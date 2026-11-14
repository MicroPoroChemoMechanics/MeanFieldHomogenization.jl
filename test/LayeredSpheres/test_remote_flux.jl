using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_remote_flux.jl — the pointwise concentrations of a layered sphere in
#  conduction under a remote FLUX. The remote gradient of a remote flux is
#  ∇T∞ = −K₀⁻¹·q∞: in a homogeneous medium the flux-flux concentration is the
#  identity and the gradient-flux one is −K₀⁻¹, and in any sphere they are the
#  gradient and the flux of that remote gradient, as for the layered spheroid.
# =============================================================================

@testset "LayeredSphere conduction — concentrations under a remote flux" begin
    K₀ = TensISO{3}(2.0)
    m(t) = collect(TensND.canonical_components(t))
    neutral = LayeredSphereTransportFields(LayeredSphere((0.6, 1.0), (K₀, K₀)), K₀)
    sphere = LayeredSphereTransportFields(
        LayeredSphere((0.6, 1.0), (TensISO{3}(5.0), TensISO{3}(1.0)); interfaces = (KapitzaInterface(0.2), PerfectInterface())), K₀
    )
    q∞ = [0.4, -0.3, 0.8]
    ∇T∞ = TensND.Vec{3}(Tuple(-(m(K₀) \ q∞)))
    for x in ((0.2, 0.1, 0.3), (0.5, -0.4, 0.2), (1.1, 0.6, -0.4))
        @test m(local_flux_flux_concentration(neutral, x)) ≈ I(3)
        @test m(local_gradient_flux_concentration(neutral, x)) ≈ -inv(m(K₀))
        @test m(local_gradient_flux_concentration(sphere, x)) * q∞ ≈ collect(local_gradient(sphere, x, ∇T∞))
        @test m(local_flux_flux_concentration(sphere, x)) * q∞ ≈ collect(local_flux(sphere, x, ∇T∞))
    end
end
