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
using TensND
import Symbolics

const MP_VP = MeanFieldHomogenization.MaterialLaws

_aggregate_vp(n; τ₀ = 1.0, count = 40) =
    ViscoplasticPolycrystal(
    [
        PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀, n, frame = f) for f in let rng = MersenneTwister(1)
                [random_basis(rng) for _ in 1:count]
        end
    ]
)

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
    # The tangent models converge their fixed point to 1e-9 relative.
    @test r.σ ≈ Σ rtol = 1.0e-8
end

@testset "between Sachs and Taylor; averages" begin
    for n in (3.0, 10.0, 20.0)
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
        @test sum(crystal_strain_rate(c, σ) for (c, σ) in zip(p.grains, rt.grains)) / length(p.grains) ≈ D rtol = 1.0e-8
        @test sum(rt.grains) / length(p.grains) ≈ rt.σ rtol = 1.0e-8
        @test abs(trace_mandel(rt.σ)) < 1.0e-12 * von_mises(rt.σ)
        @test all(crystal_strain_rate(c, σ) ≈ D for (c, σ) in zip(p.grains, rT.grains))
    end
end

@testset "homogeneity and derivatives" begin
    n = 5.0
    Σ = flow_stress(_aggregate_vp(n), D_VP).σ
    @test flow_stress(_aggregate_vp(n), 2D_VP).σ ≈ 2^(1 / n) * Σ rtol = 1.0e-8
    # Degree 1/n in the strain rate, degree one in τ₀.
    @test ForwardDiff.derivative(λ -> flow_stress(_aggregate_vp(n), λ * D_VP).σ, 1.0) ≈ Σ / n rtol = 1.0e-8
    @test ForwardDiff.derivative(τ -> flow_stress(_aggregate_vp(n; τ₀ = τ), D_VP).σ, 1.0) ≈ Σ rtol = 1.0e-8
    @test ForwardDiff.derivative(τ -> flow_stress(_aggregate_vp(n; τ₀ = τ), D_VP; model = TaylorModel()).σ, 1.0) ≈
        flow_stress(_aggregate_vp(n), D_VP; model = TaylorModel()).σ rtol = 1.0e-10
    @inferred flow_stress(_aggregate_vp(n), D_VP)
end

@testset "one grain, rest, refusals" begin
    c = PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 4.0)
    one_grain = ViscoplasticPolycrystal([c])
    @test flow_stress(one_grain, D_VP).σ ≈ flow_stress(one_grain, D_VP; model = TaylorModel()).σ rtol = 1.0e-8
    @test crystal_strain_rate(c, flow_stress(one_grain, D_VP).σ) ≈ D_VP rtol = 1.0e-8
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

@testset "the anisotropic matrix of Lebensohn and Tomé" begin
    𝕁 = SMatrix{6, 6}(KM(TensISO{3}(1.0, 0.0)))
    # Linear grains: the tangent model is the linear self-consistent scheme,
    # which the package computes otherwise (its own anisotropic Hill tensors) on
    # grains made compressible by a bulk stiffness κ, the gap closing as 1/κ.
    grains = [
        PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 1.0, frame = f) for f in let rng = MersenneTwister(1)
                [random_basis(rng) for _ in 1:6]
        end
    ]
    r = flow_stress(ViscoplasticPolycrystal(grains), D_VP; model = AnisotropicTangentModel(; reltol = 1.0e-10))
    gap(κ) = begin
        rve = RVE()
        for (i, c) in enumerate(grains)
            Lg = deviatoric_pinv(c.γ̇₀ / c.τ₀ * c.P * transpose(c.P)) + 3κ * 𝕁
            add_phase!(rve, Symbol(:g, i), Ellipsoid(1.0), Dict(:C => TensND.from_mandel(Lg)); fraction = i == 1 ? :rest : 1 / 6)
        end
        C = SMatrix{6, 6}(KM(TensND.change_tens_canon(homogenize(rve, SelfConsistent(), :C; reltol = 1.0e-9))))
        maximum(abs, C * D_VP - r.σ) / maximum(abs, r.σ)
    end
    g4, g5 = gap(1.0e4), gap(1.0e5)
    @test 8 < g4 / g5 < 12
    @test g5 < 1.0e-3
    # Power-law grains: one grain is the aggregate; the averages; homogeneity and
    # derivatives, to the accuracy of the Hill tensor (reltol 1e-10, the fixed
    # point converged to 1e-9).
    c = PowerLawCrystal(; γ̇₀ = 1.0e-3, τ₀ = 1.0, n = 5.0, frame = random_basis(MersenneTwister(4)))
    @test crystal_strain_rate(c, flow_stress(ViscoplasticPolycrystal([c]), D_VP; model = AnisotropicTangentModel()).σ) ≈ D_VP rtol = 1.0e-8
    n = 5.0
    model = AnisotropicTangentModel()
    p = _aggregate_vp(n; count = 10)
    ra = flow_stress(p, D_VP; model)
    @test sum(crystal_strain_rate(g, σ) for (g, σ) in zip(p.grains, ra.grains)) / length(p.grains) ≈ D_VP rtol = 1.0e-8
    @test sum(ra.grains) / length(p.grains) ≈ ra.σ rtol = 1.0e-8
    @test flow_stress(p, 2D_VP; model).σ ≈ 2^(1 / n) * ra.σ rtol = 1.0e-8
    @test ForwardDiff.derivative(τ -> flow_stress(_aggregate_vp(n; τ₀ = τ, count = 10), D_VP; model).σ, 1.0) ≈ ra.σ rtol = 1.0e-8
    @test von_mises(flow_stress(p, D_VP).σ) < von_mises(ra.σ) < von_mises(flow_stress(p, D_VP; model = TaylorModel()).σ)
    @inferred flow_stress(p, D_VP; model)
    @test_throws DomainError AnisotropicTangentModel(; reltol = 0.0)
end
