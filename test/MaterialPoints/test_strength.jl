# The strength of a porous von Mises solid by the modified secant method: a
# rigid–perfectly plastic solid (a power law of exponent zero) gives, at any
# macroscopic strain, a stress on the macroscopic strength domain. Checked
# against the closed-form ellipse of Mori–Tanaka with spherical pores, against
# the construction from the derivatives of the linear moduli (another code
# path) for the self-consistent scheme with spheroidal pores, for the normality
# of the strain to the domain, and for the convergence of a power law of small
# exponent to that limit.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const 𝟏_ST = SVector(1.0, 1, 1, 0, 0, 0)
# The mean stress and the norm of the stress deviator (no factor under the root).
_mean_dev(σ) = (m = sum(σ[1:3]) / 3; (m, norm(σ - m * 𝟏_ST)))
# A macroscopic strain of deviatoric part (1, -1/2, -1/2) and volumetric part α.
_direction(α) = α * 𝟏_ST / 3 + SVector(1.0, -0.5, -0.5, 0, 0, 0)

# σo bounds the norm of the stress deviator of the solid: a power law caps the
# von Mises stress at σ₀ = √(3/2) σo.
_rigid_plastic(σo; k = 1.0e6) = PowerLawElasticity(; k, σ₀ = sqrt(3 / 2) * σo, ε₀ = 1.0, m = 0.0)

function _porous_spheres(φ)
    rve = RVE()
    add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-12, 1.0e-12)); fraction = φ)
    return rve
end

@testset "Mori–Tanaka, spherical pores: the closed-form ellipse" begin
    φ, σo = 0.15, 1.0
    h = NonlinearHomogenized(_porous_spheres(φ), Dict(:S => _rigid_plastic(σo)))
    # (3/2) φ Σm² + (1 + 2φ/3) Σd² = (1 − φ)² σo²
    ellipse(Σm, Σd) = (1.5φ * Σm^2 + (1 + 2φ / 3) * Σd^2) / ((1 - φ)^2 * σo^2)
    for α in (0.0, 0.3, 1.0, 3.0, 10.0)
        σ = point_update(h, _direction(α), initial_state(h), 0.0)[1]
        @test ellipse(_mean_dev(σ)...) ≈ 1 rtol = 1.0e-6
    end
    # A strength: the stress does not depend on the size of the strain. A larger
    # strain lowers the secant shear modulus against the bulk one, and the
    # residual, then accurate to about ε k/μ, needs a looser tolerance.
    E = _direction(1.0)
    alg = LocalNewton(; reltol = 1.0e-8)
    @test point_update(h, 7.5 * E, initial_state(h), 0.0; algorithm = alg)[1] ≈ point_update(h, E, initial_state(h), 0.0)[1] rtol = 1.0e-6
    # Normality: the strain is along the normal to the domain at the stress,
    # the gradient of the ellipse, φ Σm 𝟏 + 2 (1 + 2φ/3) Σᵈ (up to a factor).
    σ = point_update(h, E, initial_state(h), 0.0)[1]
    Σm, _ = _mean_dev(σ)
    normal = φ * Σm * 𝟏_ST + 2 * (1 + 2φ / 3) * (σ - Σm * 𝟏_ST)
    @test dot(normal, E) / (norm(normal) * norm(E)) ≈ 1 rtol = 1.0e-6
end

@testset "a small exponent converges to the strength, at first order" begin
    φ, σo = 0.15, 1.0
    rve = _porous_spheres(φ)
    limit = point_update(NonlinearHomogenized(rve, Dict(:S => _rigid_plastic(σo))), _direction(1.0), initial_state(NonlinearHomogenized(rve, Dict(:S => _rigid_plastic(σo)))), 0.0)[1]
    gap(m) = (h = NonlinearHomogenized(rve, Dict(:S => PowerLawElasticity(; k = 1.0e6, σ₀ = sqrt(3 / 2) * σo, ε₀ = 1.0, m))); norm(point_update(h, 2 * _direction(1.0), initial_state(h), 0.0)[1] - limit))
    @test gap(0.02) / gap(0.04) ≈ 0.5 rtol = 0.05
end

@testset "self-consistent, spheroidal pores: against the derivatives of the moduli" begin
    k_s, ω, φ, σo = 1.0e6, 0.1, 0.15, 1.0
    scheme = SelfConsistent(; abstol = 1.0e-12, maxiters = 500)
    function rve_of(μ_s::T) where {T}
        r = RVE(; T = T, distribution_shape = Ellipsoid(1.0))
        add_phase!(r, :S, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, k_s), μ_s)); fraction = :rest, symmetrize = IsoSymmetrize())
        add_phase!(r, :P, Spheroid(ω), Dict(:C => iso_stiffness(convert(T, 1.0e-12), convert(T, 1.0e-12))); fraction = convert(T, φ), symmetrize = IsoSymmetrize())
        return r
    end
    # The ellipse 2A Σm² + B Σd² = (1 − φ) σo² from the sensitivities of the
    # linear moduli to the shear modulus of the solid.
    kμ(μ) = [k_mu(best_fit_iso(homogenize(rve_of(μ), scheme, :C)))...]
    k, μ = kμ(1.0)
    dk, dμ = ForwardDiff.derivative(kμ, 1.0)
    A, B = dk / k^2, dμ / μ^2
    h = NonlinearHomogenized(rve_of(1.0), Dict(:S => _rigid_plastic(σo; k = k_s)); scheme)
    # The outer Newton iteration cannot go below the tolerance of the inner
    # self-consistent fixed point.
    alg = LocalNewton(; reltol = 1.0e-9)
    for α in (0.0, 0.5, 2.0)
        Σm, Σd = _mean_dev(point_update(h, _direction(α), initial_state(h), 0.0; algorithm = alg)[1])
        @test (2A * Σm^2 + B * Σd^2) / ((1 - φ) * σo^2) ≈ 1 rtol = 1.0e-8
    end
end
