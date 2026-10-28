# The limit stress states of a composite with a Drucker–Prager matrix and a
# non-associated flow rule, by the modified secant method, against the closed
# forms of Maghous, Dormieux and Barthélémy (2009) for spherical heterogeneities
# and the Mori–Tanaka scheme: the elliptic domain of a porous medium, which does
# not depend on the dilatancy, and its plastic potential, which does; the
# Drucker–Prager cone of a matrix with rigid inclusions. The computed states
# approach the limit at first order in the regularization a, so that the error
# halves with a and a Richardson extrapolation meets the closed forms.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff
import Symbolics

const 𝟏_FS = SVector(1.0, 1, 1, 0, 0, 0)
_mean_dev_fs(σ) = (m = sum(σ[1:3]) / 3; (m, norm(σ - m * 𝟏_FS)))
_rate(Dv) = Dv * 𝟏_FS / 3 + SVector(1.0, -0.5, -0.5, 0, 0, 0)       # D_d = √(3/2)
const DD_FS = sqrt(1.5)

function _two_phase_cell(φ)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
    add_phase!(rve, :H, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = φ)
    return rve
end

_porous(φ, dp; a = 1.0e-3) = FrictionalComposite(_two_phase_cell(φ), dp; heterogeneities = Pores(), regularization = a)
_reinforced(φ, dp; a = 1.0e-3) = FrictionalComposite(_two_phase_cell(φ), dp; heterogeneities = RigidInclusions(), regularization = a)

@testset "refusals" begin
    @test_throws ArgumentError DruckerPrager(; h = -1.0, friction = 0.3)
    @test_throws ArgumentError DruckerPrager(; h = 1.0, friction = 0.3, dilatancy = 0.0)
    @test_throws ArgumentError DruckerPrager(; h = 1.0, friction = 0.3, dilatancy = 0.4)
    @test_throws ArgumentError DruckerPrager(; h = Symbolics.variable(:h), friction = 0.3)
    dp = DruckerPrager(; h = 1.0, friction = 0.3)
    @test_throws ArgumentError FrictionalComposite(_two_phase_cell(0.1), dp; matrix_phase = :X, heterogeneities = Pores())
    @test_throws ArgumentError FrictionalComposite(_two_phase_cell(0.1), dp; heterogeneities = Pores(), regularization = 0.0)
    single = RVE()
    add_phase!(single, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
    @test_throws ArgumentError FrictionalComposite(single, dp; heterogeneities = Pores())
end

@testset "porous medium: the elliptic domain, whatever the dilatancy" begin
    φ, h, T = 0.15, 1.0, 0.3
    # Eq. (50) of the paper, divided by h².
    F(Σm, Σd) = ((1 + 2φ / 3) / T^2 * Σd^2 + (3φ / (2T^2) - 1) * Σm^2 + 2(1 - φ) * h * Σm - (1 - φ)^2 * h^2) / h^2
    for t in (0.15, 0.3), Dv in (-0.6, 0.0, 0.6)
        dp = DruckerPrager(; h, friction = T, dilatancy = t)
        F_a, F_a2 = (F(_mean_dev_fs(limit_stress(_porous(φ, dp; a), _rate(Dv)).σ)...) for a in (1.0e-2, 5.0e-3))
        @test F_a2 / F_a ≈ 0.5 rtol = 0.05
        @test abs(2F_a2 - F_a) < 1.0e-6
    end
end

@testset "porous medium: the non-associated flow rule" begin
    φ, h, T, t = 0.15, 1.0, 0.3, 0.15
    dp = DruckerPrager(; h, friction = T, dilatancy = t)
    # The plastic potential of eq. (53) with the coefficients of eq. (55): the
    # macroscopic strain rate is normal to it, not to the strength domain.
    α, β, γ = (1 + 2φ / 3) / (T * t), 3φ / (2T * t) - 1, 2(1 - φ) * h
    for Dv in (-0.6, 0.6)
        ratio(a) = ((Σm, Σd) = _mean_dev_fs(limit_stress(_porous(φ, dp; a), _rate(Dv)).σ); (2β * Σm + γ) / (2α * Σd))
        @test 2ratio(5.0e-3) - ratio(1.0e-2) ≈ Dv / DD_FS rtol = 1.0e-4
    end
end

@testset "rigid inclusions: a Drucker–Prager cone" begin
    φ, h, T, t = 0.15, 1.0, 0.3, 0.15
    dp = DruckerPrager(; h, friction = T, dilatancy = t)
    # Eqs. (39) and (40): the macroscopic dilatancy and friction coefficients.
    t_hom = t * sqrt(1 + 1.5φ) / sqrt(1 - 2φ * t^2 / 3)
    T_hom = T * sqrt(1 + 1.5φ) * sqrt(1 - 2φ * t^2 / 3) / (1 - 2φ * t * T / 3)
    # A strain rate within the regularization of the flow rule: its offset,
    # relative to a t |D|, sets the point on the cone.
    D_on = _rate(t_hom * DD_FS)
    rate(a, offset) = _rate(t_hom * DD_FS - offset * a * t * norm(D_on))
    friction(a, offset) = ((Σm, Σd) = _mean_dev_fs(limit_stress(_reinforced(φ, dp; a), rate(a, offset)).σ); Σd / (h - Σm))
    for offset in (0.4, 0.6)
        @test 2friction(5.0e-3, offset) - friction(1.0e-2, offset) ≈ T_hom rtol = 1.0e-5
    end
    # The two offsets reach two different points of the cone.
    Σ1 = limit_stress(_reinforced(φ, dp; a = 1.0e-2), rate(1.0e-2, 0.4)).σ
    Σ2 = limit_stress(_reinforced(φ, dp; a = 1.0e-2), rate(1.0e-2, 0.6)).σ
    @test norm(Σ1 - Σ2) > 1
    # Too far from the flow rule there is no limit state: refused, not a stress
    # of the edge of the regularized matrix.
    @test_throws ArgumentError limit_stress(_reinforced(φ, dp; a = 1.0e-2), rate(1.0e-2, 1.0))
end

@testset "only the direction of the strain rate matters; derivatives" begin
    dp = DruckerPrager(; h = 1.0, friction = 0.3, dilatancy = 0.2)
    fc = _porous(0.15, dp)
    D = _rate(0.4)
    @test limit_stress(fc, 3.7 * D).σ ≈ limit_stress(fc, D).σ rtol = 1.0e-9
    @inferred limit_stress(fc, D)
    t_hom = 0.2 * sqrt(1 + 1.5 * 0.15) / sqrt(1 - 2 * 0.15 * 0.2^2 / 3)
    D_on = _rate(t_hom * DD_FS)
    @inferred limit_stress(_reinforced(0.15, dp; a = 1.0e-2), _rate(t_hom * DD_FS - 0.4e-2 * 0.2 * norm(D_on)))
    # The fixed point is solved by comparing values: symbols are refused.
    x = Symbolics.variable(:x)
    @test_throws ArgumentError limit_stress(fc, x * D)
    # The stress is proportional to the tensile strength h: its derivative is Σ/h.
    Σh(h) = limit_stress(_porous(0.15, DruckerPrager(; h, friction = 0.3, dilatancy = 0.2)), D).σ
    @test ForwardDiff.derivative(Σh, 2.0) ≈ Σh(2.0) / 2.0 rtol = 1.0e-8
    # Along the friction coefficient, against a central difference (a reference).
    ΣT(T) = limit_stress(_porous(0.15, DruckerPrager(; h = 1.0, friction = T, dilatancy = 0.2)), D).σ[1]
    @test ForwardDiff.derivative(ΣT, 0.3) ≈ (ΣT(0.3 + 1.0e-6) - ΣT(0.3 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-5
end

@testset "the von Mises limit meets the rigid-plastic secant composite" begin
    # h → ∞ and T → 0 with T h = σo: a von Mises matrix bounding σ_d by σo, whose
    # porous domain is eq. (56), reached at first order in T; NonlinearHomogenized
    # with a rigid–perfectly plastic solid computes it by another route.
    φ, σo = 0.15, 1.0
    pores = RVE()
    add_phase!(pores, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(pores, :H, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-12, 1.0e-12)); fraction = φ)
    rp = NonlinearHomogenized(pores, Dict(:M => PowerLawElasticity(; k = 1.0e6, σ₀ = sqrt(1.5) * σo, ε₀ = 1.0, m = 0.0)))
    gap(T, Dv) = (
        Σdp = limit_stress(_porous(φ, DruckerPrager(; h = σo / T, friction = T)), _rate(Dv)).σ;
        Σvm = point_update(rp, _rate(Dv), initial_state(rp), 0.0; algorithm = LocalNewton(; reltol = 1.0e-8))[1];
        norm(Σdp - Σvm) / norm(Σvm)
    )
    for Dv in (-0.6, 0.6)
        @test gap(5.0e-3, Dv) < 2.0e-2
        @test gap(2.5e-3, Dv) / gap(5.0e-3, Dv) ≈ 0.5 rtol = 0.1
    end
end
