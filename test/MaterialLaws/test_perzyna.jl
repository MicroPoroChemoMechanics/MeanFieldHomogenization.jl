# The Perzyna law of Pierard and Doghri: under a constant uniaxial stress above
# the yield stress, without hardening, the inelastic strain grows at the
# constant rate (σ_Y/η)((σ − σ_Y)/σ_Y)^m of the law, which the backward Euler
# scheme reproduces exactly, its derivative in η included; with linear hardening
# and m = 1 the response to a constant strain rate is that of a linear
# differential system, solved here in closed form, to which the steps converge
# at first order; below the yield stress the response is elastic.

using Test
using MeanFieldHomogenization
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_PZ = MeanFieldHomogenization.MaterialLaws

@testset "creep at a constant stress" begin
    σ, σY, m, E = 150.0, 100.0, 2.5, 100.0e3
    rate(η) = σY / η * ((σ - σY) / σY)^m
    creep(η) = point_response(PerzynaViscoplasticity(; E, ν = 0.3, σY, η, m), LoadingPath(range(0, 50; length = 11); σ11 = σ))
    r = creep(2.0e5)
    @test r[:ε11] ≈ σ / E .+ rate(2.0e5) .* r.t rtol = 1.0e-10
    # Isochoric flow: the lateral strains are elastic plus half the axial creep.
    @test r[:ε22] ≈ -0.3 * σ / E .- rate(2.0e5) .* r.t ./ 2 rtol = 1.0e-10
    @test ForwardDiff.derivative(η -> creep(η)[:ε11][end], 2.0e5) ≈ ForwardDiff.derivative(rate, 2.0e5) * 50 rtol = 1.0e-10
    # Below the yield stress nothing flows.
    @test point_response(PerzynaViscoplasticity(; E, ν = 0.3, σY, η = 2.0e5, m), LoadingPath(range(0, 50; length = 11); σ11 = 90.0))[:ε11] ≈
        fill(90.0 / E, 11) rtol = 1.0e-12
end

@testset "linear hardening at a constant strain rate" begin
    # Uniaxial stress, m = 1, R = H p: with the axial strain rate ε̇,
    # σ̇ = E(ε̇ − ṗ), η ṗ = σ − σ_Y − H p once yielded, a linear system whose
    # solution from the onset of flow is a constant plus an exponential.
    E, σY, η, H, ε̇ = 100.0e3, 100.0, 1.0e5, 5.0e3, 1.0e-3
    law = PerzynaViscoplasticity(; E, ν = 0.3, σY, η, m = 1.0, hardening = p -> H * p)
    t₀ = σY / (E * ε̇)
    λ = (E + H) / η
    exact(t) = t ≤ t₀ ? E * ε̇ * t : begin
            # ṗ_∞ = E ε̇/(E + H): the plastic rate the flow tends to.
            ṗ∞ = E * ε̇ / (E + H)
            τ = t - t₀
            p = ṗ∞ * (τ - (1 - exp(-λ * τ)) / λ)
            σY + E * (ε̇ * τ - p)
        end
    T = 2.0
    error(n) = (r = point_response(law, LoadingPath(range(0, T; length = n + 1); ε11 = t -> ε̇ * t)); abs(r[:σ11][end] - exact(T)))
    e₁, e₂ = error(200), error(400)
    @test e₂ < 1.0e-3 * exact(T)
    @test 1.8 < e₁ / e₂ < 2.2
    @test_throws DomainError PerzynaViscoplasticity(; E, ν = 0.3, σY, η, m = 0.5)
    st = initial_state(law)
    @inferred point_update(law, MP_PZ.Mandel(2.0e-3, -1.0e-3, -1.0e-3, 0, 0, 0), st, 0.1)
end
