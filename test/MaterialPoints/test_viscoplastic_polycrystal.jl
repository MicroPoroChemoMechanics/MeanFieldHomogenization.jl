# The steady flow of rigid–viscoplastic polycrystals: the Taylor and the tangent
# self-consistent models. Linear grains (n = 1) against an independent solution
# of the linear self-consistent problem; the tangent estimate between the Sachs
# and the Taylor ones; the averages; homogeneity of degree 1/n in the strain
# rate and of degree one in τ₀, with their ForwardDiff derivatives.

using Test
using MeanFieldHomogenization
using StaticArrays
using LinearAlgebra
using Random
using ForwardDiff
import Symbolics

const MP_VP = MeanFieldHomogenization.MaterialPoints

_aggregate_vp(n; τ₀ = 1.0, count = 40) =
    ViscoplasticPolycrystal([PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀, n, frame = f) for f in random_orientations(count; rng = MersenneTwister(1))])

const D_VP = SVector(1.0, -0.5, -0.5, 0.2, 0, 0.1) * 1.0e-3

@testset "linear grains: the linear self-consistent solution" begin
    p = _aggregate_vp(1.0)
    r = flow_stress(p, D_VP)
    # Linear grains d = M_g σ; the interaction (M_g + 𝕂/(3η) + 𝕁) σ_g = D + 𝕂Σ/(3η)
    # with η = Σ_eq/(3 D_eq): a fixed point on η, solved by plain substitution.
    M = [c.γ̇₀ / c.τ₀ * c.P * transpose(c.P) for c in p.grains]
    𝕁, 𝕂 = MP_VP._𝕁, MP_VP._𝕂
    Σ = sum((m + 𝕁) \ D_VP for m in M) / length(M)
    for _ in 1:200
        a = equivalent_strain(D_VP) / von_mises(Σ)
        Σ = sum((m + a * 𝕂 + 𝕁) \ (D_VP + a * (𝕂 * Σ)) for m in M) / length(M)
    end
    @test r.σ ≈ Σ rtol = 1.0e-10
end

@testset "between Sachs and Taylor; averages" begin
    for n in (3.0, 10.0)
        p = _aggregate_vp(n)
        D = SVector(1.0, -0.5, -0.5, 0, 0, 0) * 1.0e-3
        rT, rt = flow_stress(p, D; model = TaylorModel()), flow_stress(p, D)
        # Sachs: every grain at the uniaxial stress s whose average axial rate is D₁₁.
        sachs(s) = sum(crystal_strain_rate(c, SVector(2s / 3, -s / 3, -s / 3, 0, 0, 0))[1] for c in p.grains) / length(p.grains) - D[1]
        s = 3.0
        for _ in 1:60
            s -= sachs(s) / ForwardDiff.derivative(sachs, s)
        end
        @test s < von_mises(rt.σ) < von_mises(rT.σ)
        @test sum(crystal_strain_rate(c, σ) for (c, σ) in zip(p.grains, rt.grains)) / length(p.grains) ≈ D atol = 1.0e-15
        @test sum(rt.grains) / length(p.grains) ≈ rt.σ rtol = 1.0e-12
        @test abs(trace_mandel(rt.σ)) < 1.0e-12 * von_mises(rt.σ)
        @test all(crystal_strain_rate(c, σ) ≈ D for (c, σ) in zip(p.grains, rT.grains))
    end
end

@testset "homogeneity and derivatives" begin
    n = 5.0
    Σ = flow_stress(_aggregate_vp(n), D_VP).σ
    @test flow_stress(_aggregate_vp(n), 2D_VP).σ ≈ 2^(1 / n) * Σ rtol = 1.0e-12
    # Degree 1/n in the strain rate, degree one in τ₀.
    @test ForwardDiff.derivative(λ -> flow_stress(_aggregate_vp(n), λ * D_VP).σ, 1.0) ≈ Σ / n rtol = 1.0e-10
    @test ForwardDiff.derivative(τ -> flow_stress(_aggregate_vp(n; τ₀ = τ), D_VP).σ, 1.0) ≈ Σ rtol = 1.0e-10
    @test ForwardDiff.derivative(τ -> flow_stress(_aggregate_vp(n; τ₀ = τ), D_VP; model = TaylorModel()).σ, 1.0) ≈
        flow_stress(_aggregate_vp(n), D_VP; model = TaylorModel()).σ rtol = 1.0e-10
    @inferred flow_stress(_aggregate_vp(n), D_VP)
end

@testset "one grain, rest, refusals" begin
    c = PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 4.0)
    one_grain = ViscoplasticPolycrystal([c])
    @test flow_stress(one_grain, D_VP).σ ≈ flow_stress(one_grain, D_VP; model = TaylorModel()).σ rtol = 1.0e-12
    @test crystal_strain_rate(c, flow_stress(one_grain, D_VP).σ) ≈ D_VP rtol = 1.0e-12
    @test iszero(flow_stress(one_grain, zero(D_VP)).σ)
    @test_throws ArgumentError flow_stress(one_grain, SVector(1.0, 0, 0, 0, 0, 0))
    @test_throws ArgumentError flow_stress(one_grain, Symbolics.Num.(D_VP))
    @test_throws ArgumentError ViscoplasticPolycrystal(PowerLawCrystal[])
    @test_throws ArgumentError ViscoplasticPolycrystal([c, PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 3.0)])
    @test_throws ArgumentError ViscoplasticPolycrystal([c, c], [0.5, 0.6])
    @test_throws ArgumentError ViscoplasticPolycrystal([c, c], [0.5])
    @test_throws DomainError PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 0.5)
    @test_throws DomainError PowerLawCrystal(; γ̇₀ = -1.0, τ₀ = 1.0, n = 2.0)
    @test_throws DomainError PowerLawCrystal(; γ̇₀ = 1.0, τ₀ = 0.0, n = 2.0)
    @test_throws ArgumentError PowerLawCrystal(; γ̇₀ = Symbolics.Num(1.0), τ₀ = 1.0, n = 2.0)
    # No convergence within the iterations allowed: a failed step.
    @test_throws LocalIntegrationError flow_stress(one_grain, D_VP; model = TaylorModel(), algorithm = LocalNewton(; maxiters = 1))
    # A Jacobian that vanishes past the origin, which the trust-region iteration
    # reaches: a failed step, not an exception of the factorization.
    r(x) = SVector(x[1] > 0 ? x[1] + 3 : 0.5 + 0 * x[1], x[2], x[3], x[4])
    @test_throws LocalIntegrationError MP_VP._trust_region_values(r, x -> ForwardDiff.jacobian(r, x), SVector(1.0, 1.0, 1.0, 1.0), LocalNewton(), :test)
end
