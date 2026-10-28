# The laws shipped with MaterialLaws, each against a closed-form response that
# the test computes on its own: Hooke's law, the uniaxial power law, the discrete
# and continuous Maxwell relaxation, Norton's steady creep, the radial return of
# von Mises plasticity (3-D closed form and a 1-D return mapping written here),
# and Biot's drained and undrained responses.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra

const E_MP, ν_MP = 200.0e3, 0.3                     # MPa
const μ_MP = E_MP / (2 * (1 + ν_MP))
const k_MP = E_MP / (3 * (1 - 2ν_MP))
const SQ2_MP = sqrt(2.0)

# A Mandel vector from tensor components (11, 22, 33, 23, 13, 12).
_mv(a, b, c, d = 0.0, e = 0.0, f = 0.0) = SVector(a, b, c, SQ2_MP * d, SQ2_MP * e, SQ2_MP * f)

@testset "LinearElasticity is Hooke's law" begin
    ℂ = TensND.iso_stiffness_young_poisson(E_MP, ν_MP)
    for law in (LinearElasticity(ℂ), LinearElasticity(; E = E_MP, ν = ν_MP))
        ε = _mv(1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4, -5.0e-5, 2.0e-4)
        σ, 𝕂, st = point_update(law, ε, initial_state(law), 0.0)
        A = mandel(ℂ)
        @test σ ≈ A * ε rtol = 1.0e-13
        @test 𝕂 ≈ A rtol = 1.0e-13
        @test st.g == ε
        # The same as the homogenized material built on the same stiffness.
        r = material_response(HomogenizedElastic(ℂ), from_mandel(ε), NoState(), 0.0)
        @test mandel(stress(r)) ≈ σ rtol = 1.0e-13
    end
    # An anisotropic stiffness written in a rotated basis: the law reads it in
    # the canonical frame.
    B = SMatrix{6, 6}([i == j ? 300.0 + 10i : 20.0 / (i + j) for i in 1:6, j in 1:6])
    ℂrot = change_tens(from_mandel(B), TensND.RotatedBasis(0.1, 0.2, 0.3))
    law = LinearElasticity(ℂrot)
    @test point_update(law, _mv(1.0e-3, 0, 0), initial_state(law), 0.0)[2] ≈ B rtol = 1.0e-12
end

@testset "PowerLawElasticity" begin
    k, σ₀, ε₀, m = 1.0e5, 100.0, 1.0e-3, 0.3
    law = PowerLawElasticity(; k, σ₀, ε₀, m)
    # An isochoric extension ε = e (1, −½, −½): ε_eq = e, so σ₁₁ − σ₂₂ = σ₀ (e/ε₀)^m
    # and the mean stress is zero.
    es = range(0, 3.0e-3; length = 7)
    r = point_response(
        law, LoadingPath(
            range(0, 1; length = 7); ε11 = t -> 3.0e-3 * t, ε22 = t -> -1.5e-3 * t,
            ε33 = t -> -1.5e-3 * t, ε23 = 0, ε13 = 0, ε12 = 0
        )
    )
    @test r[:σ11] - r[:σ22] ≈ [σ₀ * (e / ε₀)^m for e in es] rtol = 1.0e-12
    @test maximum(abs, r[:σ11] + r[:σ22] + r[:σ33]) < 1.0e-9
    # At zero strain the stress is zero and finite, not 0·∞.
    @test iszero(r[:σ11][1])
    # A volume change alone: σ = k tr ε 1.
    σ, _, _ = point_update(law, _mv(1.0e-3, 1.0e-3, 1.0e-3), initial_state(law), 0.0)
    @test σ ≈ 3k * 1.0e-3 * SVector(1.0, 1, 1, 0, 0, 0) rtol = 1.0e-13
    # m = 1 is linear elasticity of shear modulus σ₀/(3ε₀).
    lin = PowerLawElasticity(; k, σ₀, ε₀, m = 1.0)
    ε = _mv(1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4)
    @test point_update(lin, ε, initial_state(lin), 0.0)[2] ≈
        mandel(TensISO{3}(3k, 2 * σ₀ / (3ε₀))) rtol = 1.0e-12
end

@testset "MaxwellViscoelasticity relaxes" begin
    τ = 2.0
    law = MaxwellViscoelasticity(; E = E_MP, ν = ν_MP, τ)
    ε₀ = 1.0e-3
    for n in (40, 400)
        ts = range(0, 4; length = n + 1)
        r = point_response(law, LoadingPath(ts; ε11 = ε₀, ε22 = 0, ε33 = 0))
        σ₀ = (k_MP + 4μ_MP / 3) * ε₀                 # the instantaneous (oedometric) response
        @test r[:σ11][1] ≈ σ₀ rtol = 1.0e-12
        # Backward Euler of σ̇ = −σ/τ at fixed strain, exactly.
        @test r[:σ11] ≈ σ₀ .* (1 + step(ts) / τ) .^ (-(0:n)) rtol = 1.0e-10
        # And the continuous relaxation, to the order of the scheme.
        @test abs(r[:σ11][end] / σ₀ - exp(-4 / τ)) < 10 * step(ts) / τ * exp(-4 / τ)
    end
end

@testset "NortonViscoplasticity creeps at the steady rate" begin
    K, n, σ = 500.0, 3.0, 150.0
    law = NortonViscoplasticity(; E = E_MP, ν = ν_MP, K, n)
    ts = range(0, 100; length = 11)
    r = point_response(law, LoadingPath(ts; σ11 = σ))
    rate = (σ / K)^n
    # Constant stress: the backward Euler step is exact.
    @test r[:ε11] ≈ [σ / E_MP + rate * t for t in ts] rtol = 1.0e-10
    @test r[:ε22] ≈ [-ν_MP * σ / E_MP - rate * t / 2 for t in ts] rtol = 1.0e-10
    @test r[:εᵖ11][end] ≈ rate * 100 rtol = 1.0e-10
    # At zero stress, no creep and no division by zero.
    st = initial_state(law)
    @test point_update(law, zero(SVector{6}), st, 1.0)[3].y == st.y
    @test_throws DomainError NortonViscoplasticity(; E = E_MP, ν = ν_MP, K, n = 0.5)
end

# ── Von Mises plasticity ─────────────────────────────────────────────────────

# The 3-D radial return for linear isotropic and kinematic hardening, in closed
# form (one plastic step from a virgin state).
function _radial_return(ε, σ₀, H, Hkin)
    σtr = (3k_MP) * (sum(ε[1:3]) / 3) * SVector(1.0, 1, 1, 0, 0, 0) + 2μ_MP * (ε - (sum(ε[1:3]) / 3) * SVector(1.0, 1, 1, 0, 0, 0))
    s = σtr - (sum(σtr[1:3]) / 3) * SVector(1.0, 1, 1, 0, 0, 0)
    q = sqrt(1.5 * sum(abs2, s))
    Δp = (q - σ₀) / (3μ_MP + H + Hkin)
    return σtr - 2μ_MP * Δp * 1.5 * s / q, Δp
end

# The 1-D return mapping of uniaxial stress, written independently: the
# effective stress is |σ − H_kin εᵖ| and the yield stress R(p).
function _uniaxial_history(εs, R, Hkin)
    εᵖ, p, out = 0.0, 0.0, Float64[]
    for ε in εs
        ξ = E_MP * (ε - εᵖ) - Hkin * εᵖ
        f = abs(ξ) - R(p)
        if f > 0
            # Solve |ξ| − (E + H_kin) Δp − R(p + Δp) = 0 by bisection.
            lo, hi = 0.0, f / E_MP
            for _ in 1:200
                mid = (lo + hi) / 2
                abs(ξ) - (E_MP + Hkin) * mid - R(p + mid) > 0 ? (lo = mid) : (hi = mid)
            end
            Δp = (lo + hi) / 2
            εᵖ += Δp * sign(ξ); p += Δp
        end
        push!(out, E_MP * (ε - εᵖ))
    end
    return out
end

@testset "J2Plasticity: radial return in 3-D" begin
    σ₀, H, Hkin = 250.0, 2.0e3, 5.0e3
    law = J2Plasticity(; E = E_MP, ν = ν_MP, σ₀, H, Hkin)
    for ε in (_mv(4.0e-3, -1.0e-3, -1.0e-3), _mv(1.0e-3, 2.0e-3, -2.5e-3, 1.0e-3, -2.0e-4, 6.0e-4))
        σ, 𝕂, st = point_update(law, ε, initial_state(law), 0.0)
        σref, Δp = _radial_return(ε, σ₀, H, Hkin)
        @test σ ≈ σref rtol = 1.0e-11
        @test st.y[7] ≈ Δp rtol = 1.0e-11
    end
    # An elastic step keeps the state and returns the elastic stiffness.
    σ, 𝕂, st = point_update(law, _mv(5.0e-4, 0, 0), initial_state(law), 0.0)
    @test 𝕂 ≈ mandel(TensND.iso_stiffness_young_poisson(E_MP, ν_MP)) rtol = 1.0e-13
    @test iszero(st.y)
end

@testset "J2Plasticity: uniaxial cycles against a 1-D return mapping" begin
    σ₀ = 250.0
    εs = vcat(range(0, 4.0e-3; length = 21), range(4.0e-3, -4.0e-3; length = 41)[2:end], range(-4.0e-3, 2.0e-3; length = 31)[2:end])
    ts = range(0, 1; length = length(εs))
    εpath = t -> εs[clamp(round(Int, t * (length(εs) - 1)) + 1, 1, length(εs))]
    cases = (
        (law = J2Plasticity(; E = E_MP, ν = ν_MP, σ₀, H = 3.0e3), R = p -> σ₀ + 3.0e3 * p, Hkin = 0.0),
        (law = J2Plasticity(; E = E_MP, ν = ν_MP, σ₀, Hkin = 8.0e3), R = p -> σ₀, Hkin = 8.0e3),
        # A saturating isotropic hardening given as a function.
        (
            law = J2Plasticity(; E = E_MP, ν = ν_MP, σ₀, Hkin = 2.0e3, hardening = p -> σ₀ + 120.0 * (1 - exp(-300.0 * p))),
            R = p -> σ₀ + 120.0 * (1 - exp(-300.0 * p)), Hkin = 2.0e3,
        ),
    )
    for c in cases
        r = point_response(c.law, LoadingPath(ts; ε11 = εpath))
        @test r[:σ11] ≈ _uniaxial_history(εs, c.R, c.Hkin) rtol = 1.0e-9 atol = 1.0e-9
        @test maximum(abs, r[:σ22]) < 1.0e-9
        @test r[:εᵖ22] ≈ -r[:εᵖ11] / 2 rtol = 1.0e-9 atol = 1.0e-15
    end
end

# ── Biot poroelasticity: a law of two coupled fields ────────────────────────

@testset "BiotPoroelasticity: blocks, drained and undrained" begin
    b, M = 0.8, 1.0e4
    law = BiotPoroelasticity(; E = E_MP, ν = ν_MP, b, M)
    @test gradient_layout(law) == (ε = 6, p = 1)
    @test flux_layout(law) == (σ = 6, φ = 1)
    @test gradient_names(law) == (:ε, :p) && flux_names(law) == (:σ, :φ)
    @test tangent_blocks(law) == (:σε, :σp, :φε, :φp)

    ε, p = _mv(1.0e-3, -2.0e-4, 5.0e-4, 1.0e-4), 2.0
    r = material_response(law, (ε = from_mandel(ε), p = p), initial_state(law), 0.0)
    ℂ = TensND.iso_stiffness_young_poisson(E_MP, ν_MP)
    @test mandel(stress(r)) ≈ mandel(ℂ) * ε - b * p * SVector(1.0, 1, 1, 0, 0, 0) rtol = 1.0e-12
    @test r.fluxes.φ ≈ b * sum(ε[1:3]) + p / M rtol = 1.0e-12
    @test mandel(r.tangents.σε) ≈ mandel(ℂ) rtol = 1.0e-13
    @test mandel(r.tangents.σp) ≈ -b * SVector(1.0, 1, 1, 0, 0, 0) atol = 1.0e-15
    @test mandel(r.tangents.φε) ≈ b * SVector(1.0, 1, 1, 0, 0, 0) atol = 1.0e-15
    @test r.tangents.φp ≈ 1 / M rtol = 1.0e-14
    @test check_material_interface(law; verbose = false)

    # Oedometric compression: drained (p = 0) and undrained (φ = 0).
    ts = 0:0.5:1
    drained = point_response(law, LoadingPath(ts; ε11 = t -> -1.0e-3 * t, ε22 = 0, ε33 = 0, p = 0))
    undrained = point_response(law, LoadingPath(ts; ε11 = t -> -1.0e-3 * t, ε22 = 0, ε33 = 0))
    Moed = k_MP + 4μ_MP / 3
    @test drained[:σ11][end] ≈ -Moed * 1.0e-3 rtol = 1.0e-12
    @test drained[:φ][end] ≈ -b * 1.0e-3 rtol = 1.0e-12
    @test undrained[:p][end] ≈ M * b * 1.0e-3 rtol = 1.0e-12
    @test undrained[:σ11][end] ≈ -(Moed + b^2 * M) * 1.0e-3 rtol = 1.0e-12
    @test abs(undrained[:φ][end]) < 1.0e-15
end
