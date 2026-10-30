# The "quasi-elastic" affine formulation of Brenner et al. Its direct inversion
# p = 1/t is exact for the creep of a single phase, which two identical phases
# reproduce; with h(t) built on a relaxation time it is exact for the
# relaxation of a single linear Maxwell phase; for a composite of linear
# Maxwell phases in creep it is exact at the start (the elastic Mori–Tanaka
# estimate) and in the steady regime (the viscous one), as the paper shows; the
# response is homogeneous of degree one in the moduli and stresses of the
# phases, derivative included; the tangent is that of the step.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_QE = MeanFieldHomogenization.MaterialLaws

function _rve_qe(f)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
    return rve
end
_step_qe(c, E, st, Δt) = MP_QE._mandel_update(c, E, st, Δt; algorithm = LocalNewton(), external = (;))

@testset "a single phase: creep and relaxation exact" begin
    E, K, n, Σ = 70.0e3, 300.0, 3, 120.0
    law = NortonViscoplasticity(; E, ν = 0.3, K, n)
    r = point_response(QuasiElasticComposite(_rve_qe(0.3), (M = law, I = law)), LoadingPath(range(0, 10; length = 11); σ11 = Σ))
    @test r[:ε11] ≈ Σ / E .+ r.t .* (Σ / K)^n rtol = 1.0e-10
    # A linear Maxwell phase (n = 1, viscosity K/3 in shear) held at a strain:
    # σ = 3k ε_v + 2μ e^{−t/τ} e, τ = (K/3)/μ, with h(t) built on τ.
    lin = NortonViscoplasticity(; E, ν = 0.3, K, n = 1)
    k, μ = E / (3 * (1 - 0.6)), E / 2.6
    τ = K / 3 / μ
    D = SVector(1.0e-3, -2.0e-4, 3.0e-4, 2.0e-4, 0, -1.0e-4)
    Dv = SVector(1, 1, 1, 0, 0, 0) * sum(D[1:3]) / 3
    path = LoadingPath(range(0, 3τ; length = 7); ε11 = D[1], ε22 = D[2], ε33 = D[3], ε23 = D[4] / sqrt(2), ε13 = D[5] / sqrt(2), ε12 = D[6] / sqrt(2))
    r = point_response(QuasiElasticComposite(_rve_qe(0.3), (M = lin, I = lin); relaxation_time = τ), path)
    @test all(st.σ ≈ 3k * Dv + 2μ * exp(-t / τ) * (D - Dv) for (st, t) in zip(r.states, r.t))
end

@testset "linear Maxwell phases in creep: exact at the start and in the steady regime" begin
    f = 0.3
    E₀, K₀, E₁, K₁ = 70.0e3, 3 * 70.0e3 / 2.6, 400.0e3, 3 * 400.0e3 / 2.5 * 5
    laws = (M = NortonViscoplasticity(; E = E₀, ν = 0.3, K = K₀, n = 1), I = NortonViscoplasticity(; E = E₁, ν = 0.25, K = K₁, n = 1))
    c = QuasiElasticComposite(_rve_qe(f), laws)
    Σ = 100.0
    times = vcat(0.0, exp10.(range(-6, 4; length = 61)))
    r = point_response(c, LoadingPath(times; σ11 = Σ))
    # At the start, the elastic Mori–Tanaka estimate.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(E₀, 0.3)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(E₁, 0.25)); fraction = f)
    C = mandel(homogenize(rve, MoriTanaka(), :C))
    @test r[:ε11][2] ≈ (C \ SVector(Σ, 0, 0, 0, 0, 0))[1] rtol = 1.0e-4
    # In the steady regime, the creep rate of the Mori–Tanaka estimate of the
    # viscous, incompressible phases: Σ/(3η) with η its viscosity in shear.
    η₀, η₁ = K₀ / 3, K₁ / 3
    η = η₀ + f * (η₁ - η₀) / (1 + (1 - f) * (η₁ - η₀) / (η₀ + 3η₀ / 2))
    rate = (r[:ε11][end] - r[:ε11][end - 1]) / (r.t[end] - r.t[end - 1])
    @test rate ≈ Σ / (3η) rtol = 1.0e-3
end

@testset "homogeneity, derivative, tangent, refusals" begin
    rve = _rve_qe(0.2)
    composite(λ) = QuasiElasticComposite(rve, (M = NortonViscoplasticity(; E = λ * 70.0e3, ν = 0.3, K = λ * 300.0, n = 3), I = NortonViscoplasticity(; E = λ * 400.0e3, ν = 0.25, K = λ * 900.0, n = 3)))
    path = LoadingPath(range(0, 1; length = 5); ε11 = t -> 2.0e-3 * t)
    stress(λ) = point_response(composite(λ), path).states[end].σ
    Σ = stress(1.0)
    # Homogeneous of degree one in the moduli, to the accuracy of the solves:
    # each response is converged to the relative tolerance of the driver
    # (1e-10 by default), so two of them agree to a margin above it, not to
    # it — at 1e-10 the check failed on one CI machine (1.5e-10).
    @test stress(3.0) ≈ 3Σ rtol = 1.0e-8
    @test ForwardDiff.derivative(stress, 1.0) ≈ Σ rtol = 1.0e-8
    c = composite(1.0)
    st = point_response(c, path).states[end]
    E = st.ε + SVector(1.0e-4, -2.0e-5, -3.0e-5, 1.0e-5, 0, 0)
    @test _step_qe(c, E, st, 0.1)[2] ≈ ForwardDiff.jacobian(e -> _step_qe(c, e, st, 0.1)[1], E) rtol = 1.0e-8
    @inferred _step_qe(c, E, st, 0.1)
    @test_throws ArgumentError _step_qe(c, E, st, -0.1)
    @test_throws ArgumentError QuasiElasticComposite(rve, (M = c.phases[1],))
    @test_throws ArgumentError QuasiElasticComposite(rve, (M = c.phases[1], X = c.phases[2]))
    @test_throws ArgumentError QuasiElasticComposite(rve, (M = c.phases[1], I = PerzynaViscoplasticity(; E = 1.0e5, ν = 0.3, σY = 100.0, η = 1.0e5, m = 1.0)))
    @test_throws DomainError QuasiElasticComposite(rve, (M = c.phases[1], I = c.phases[2]); relaxation_time = -1.0)
end

@testset "a tolerance out of the reach of the material is said" begin
    rve = _rve_qe(0.2)
    c = QuasiElasticComposite(rve, (M = NortonViscoplasticity(; E = 70.0e3, ν = 0.3, K = 300.0, n = 3), I = NortonViscoplasticity(; E = 400.0e3, ν = 0.25, K = 900.0, n = 3)))
    path = LoadingPath(range(0, 1; length = 5); ε11 = t -> 2.0e-3 * t)
    # Finer than the solves of the composite resolve its stress: the steps are
    # cut, and the driver says why, once.
    @test_logs (:warn, r"stalls") match_mode = :any point_response(c, path; rtol = 1.0e-14)
    # A residual far from the tolerance is a failure to converge, which a cut
    # may cure: nothing to say.
    @test_logs MP_QE._warn_unreachable(MP_QE.PointDriverError(0.0, 25, 1.0e-2), 1.0e-10)
    @test_logs MP_QE._warn_unreachable(MP_QE.LocalIntegrationError(:law, 50, 1.0, "no convergence"), 1.0e-10)
end
