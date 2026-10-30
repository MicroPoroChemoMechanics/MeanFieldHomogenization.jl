# The affine formulation of Pierard and Doghri for elasto-viscoplastic
# composites. Two identical phases are the phase itself; phases that stay
# elastic give the elastic Mori–Tanaka estimate; linear Maxwell phases give the
# linear viscoelastic Mori–Tanaka estimate, computed here in closed form in the
# Laplace–Carson domain and inverted numerically, which the steps approach at
# the first order of their backward Euler scheme; the response is homogeneous
# of degree one in the moduli, stresses and viscosities of the phases,
# derivative included; the tangent is that of the step.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_EV = MeanFieldHomogenization.MaterialLaws

function _rve_ev(f; M = TensISO{3}(1.0, 1.0), I = TensISO{3}(1.0, 1.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => M); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => I); fraction = f)
    return rve
end

const θ_EV = exp10.(range(-3, 3; length = 20))
const D_EV = SVector(1.0e-3, -2.0e-4, 3.0e-4, 2.0e-4, 0, -1.0e-4)
_ramp_ev(T, n) = LoadingPath(
    range(0, T; length = n); ε11 = t -> D_EV[1] * t, ε22 = t -> D_EV[2] * t, ε33 = t -> D_EV[3] * t,
    ε23 = t -> D_EV[4] / sqrt(2) * t, ε13 = t -> D_EV[5] / sqrt(2) * t, ε12 = t -> D_EV[6] / sqrt(2) * t
)
_perzyna_ev(λ = 1.0; E = 100.0e3, σY = 100.0) =
    PerzynaViscoplasticity(; E = λ * E, ν = 0.3, σY = λ * σY, η = λ * 1.0e5, m = 2.0, hardening = p -> λ * 5.0e3 * p)
_step_ev(c, E, st, Δt) = MP_EV._mandel_update(c, E, st, Δt; algorithm = LocalNewton(), external = (;))

@testset "identical phases: the phase itself" begin
    law = _perzyna_ev()
    path = LoadingPath(range(0, 10; length = 41); ε11 = t -> 1.0e-3 * t)
    r = point_response(AffineEVPComposite(_rve_ev(0.1), (M = law, I = law); relaxation_times = θ_EV), path)
    @test r[:σ11] ≈ point_response(law, path)[:σ11] rtol = 1.0e-12
end

@testset "elastic phases: the elastic Mori–Tanaka estimate" begin
    laws = (M = _perzyna_ev(; σY = 1.0e6), I = _perzyna_ev(; E = 400.0e3, σY = 1.0e6))
    r = point_response(AffineEVPComposite(_rve_ev(0.3), laws; relaxation_times = θ_EV), _ramp_ev(2.0, 5))
    C = mandel(homogenize(_rve_ev(0.3; M = iso_stiffness_young_poisson(100.0e3, 0.3), I = iso_stiffness_young_poisson(400.0e3, 0.3)), MoriTanaka(), :C))
    @test r.states[end].σ ≈ C * (2.0 * D_EV) rtol = 1.0e-10
    # The whole modulus of the matrix, isotropic here, gives the same.
    r = point_response(AffineEVPComposite(_rve_ev(0.3), laws; relaxation_times = θ_EV, isotropic = false), _ramp_ev(2.0, 5))
    @test r.states[end].σ ≈ C * (2.0 * D_EV) rtol = 1.0e-6
end

@testset "linear Maxwell phases: the viscoelastic Mori–Tanaka estimate" begin
    # Norton's law with n = 1 is a Maxwell solid in shear, of viscosity K/3, and
    # elastic in volume: μ*(s) = μηs/(ηs + μ) in the Laplace–Carson domain, and
    # the Mori–Tanaka estimate of spheres is in closed form.
    f = 0.3
    E₀, ν₀, K₀ = 70.0e3, 0.3, 3 * 70.0e3 / 2.6
    E₁, ν₁, K₁ = 400.0e3, 0.25, 3 * 400.0e3 / 2.5 * 5
    laws = (M = NortonViscoplasticity(; E = E₀, ν = ν₀, K = K₀, n = 1), I = NortonViscoplasticity(; E = E₁, ν = ν₁, K = K₁, n = 1))
    k₀, μ₀ = E₀ / (3 * (1 - 2ν₀)), E₀ / (2 * (1 + ν₀))
    k₁, μ₁ = E₁ / (3 * (1 - 2ν₁)), E₁ / (2 * (1 + ν₁))
    maxwell(μ, η, s) = μ * η * s / (η * s + μ)
    function mori_tanaka(s)
        m₀, m₁ = maxwell(μ₀, K₀ / 3, s), maxwell(μ₁, K₁ / 3, s)
        ζ = m₀ * (9k₀ + 8m₀) / (6 * (k₀ + 2m₀))
        return k₀ + f * (k₁ - k₀) / (1 + (1 - f) * (k₁ - k₀) / (k₀ + 4m₀ / 3)),
            m₀ + f * (m₁ - m₀) / (1 + (1 - f) * (m₁ - m₀) / (m₀ + ζ))
    end
    # Under E(t) = D t, E*(s) = D/s.
    Dm = sum(D_EV[1:3]) / 3
    Dv, Dd = SVector(Dm, Dm, Dm, 0, 0, 0), D_EV - SVector(Dm, Dm, Dm, 0, 0, 0)
    exact(t) = 3 * inverse_carson(s -> first(mori_tanaka(s)) / s, t) * Dv + 2 * inverse_carson(s -> last(mori_tanaka(s)) / s, t) * Dd
    c = AffineEVPComposite(_rve_ev(f), laws; relaxation_times = θ_EV)
    error(n) = (r = point_response(c, _ramp_ev(3.0, n)); maximum(abs, r.states[(n + 1) ÷ 2].σ - exact(1.5)) / maximum(abs, exact(1.5)))
    e₁, e₂ = error(121), error(241)
    @test e₂ < 1.0e-3
    @test 1.9 < e₁ / e₂ < 2.1
end

@testset "homogeneity, derivative, tangent, refusals" begin
    rve = _rve_ev(0.2)
    composite(λ) = AffineEVPComposite(rve, (M = _perzyna_ev(λ), I = _perzyna_ev(λ; E = 400.0e3, σY = 400.0)); relaxation_times = θ_EV)
    path = _ramp_ev(1.0, 5)
    stress(λ) = point_response(composite(λ), path).states[end].σ
    Σ = stress(1.0)
    @test stress(3.0) ≈ 3Σ rtol = 1.0e-10
    @test ForwardDiff.derivative(stress, 1.0) ≈ Σ rtol = 1.0e-8
    # The tangent of a step from a flowing state.
    c = composite(1.0)
    st = point_response(c, path).states[end]
    E = st.ε + 1.0e-4 * D_EV
    @test _step_ev(c, E, st, 0.1)[2] ≈ ForwardDiff.jacobian(e -> _step_ev(c, e, st, 0.1)[1], E) rtol = 1.0e-8
    @inferred _step_ev(c, E, st, 0.1)
    @test_throws ArgumentError _step_ev(c, E, st, -0.1)
    @test_throws ArgumentError AffineEVPComposite(rve, (M = _perzyna_ev(),); relaxation_times = θ_EV)
    @test_throws ArgumentError AffineEVPComposite(rve, (M = _perzyna_ev(), X = _perzyna_ev()); relaxation_times = θ_EV)
    @test_throws ArgumentError AffineEVPComposite(rve, (M = _perzyna_ev(), I = J2Plasticity(; E = 70.0e3, ν = 0.3, σ₀ = 100.0)); relaxation_times = θ_EV)
    @test_throws ArgumentError AffineEVPComposite(rve, (M = _perzyna_ev(), I = _perzyna_ev()); relaxation_times = [1.0])
end
