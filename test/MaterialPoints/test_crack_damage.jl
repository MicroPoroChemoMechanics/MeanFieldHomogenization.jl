# Damage by the growth of microcracks: below the threshold the law is the linear
# homogenized solid; the driving force is the derivative of the energy (checked
# against a central difference of the homogenized energy, a reference only); a
# growing family sits on its criterion; the viscous law obeys its rate equation;
# the tangent is the derivative of the whole update; families loaded differently
# grow differently. And the local Newton iteration shortens a correction that
# leaves the domain of a law.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_CD = MeanFieldHomogenization.MaterialPoints
const SC_CD = MeanFieldHomogenization.Schemes
const C0_CD = TensISO{3}(3 * 30.0e3, 2 * 20.0e3)

function _cracked_cell(; d = 0.05, oriented = false)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_CD); fraction = :rest)
    if oriented
        add_phase!(rve, :F3, PennyCrack(1.0), Dict(:C => C0_CD); density = d)
        add_phase!(rve, :F1, PennyCrack(1.0; euler_angles = (π / 2, 0.0)), Dict(:C => C0_CD); density = d)
    else
        add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C0_CD); density = d, symmetrize = :iso)
    end
    return rve
end

_stiffness_at(rve, φ, d) = KM(homogenize(SC_CD.set_param(rve, SC_CD.AmountParameter(φ), d), MoriTanaka(), :C))
# −∂W/∂d by a central difference of the homogenized energy: a reference only.
_force_fd(rve, φ, d, ε; h = 1.0e-6) =
    -(dot(ε, _stiffness_at(rve, φ, d + h) * ε) - dot(ε, _stiffness_at(rve, φ, d - h) * ε)) / (4h)

const RISING_CD = d -> 0.04 * (1 + 20 * (d - 0.05))
const GRIFFITH_CD = CrackGrowthResistance(; Gc = 0.02, a₀ = 1.0, d₀ = 0.05)

@testset "the resistance of growing cracks" begin
    @test GRIFFITH_CD(0.05) ≈ 2π * 0.02 / 3 rtol = 1.0e-14
    @test GRIFFITH_CD(0.4) ≈ GRIFFITH_CD(0.05) / 2 rtol = 1.0e-14          # (d₀/d)^(1/3) = 1/2
    @test ForwardDiff.derivative(GRIFFITH_CD, 0.4) ≈ -GRIFFITH_CD(0.4) / (3 * 0.4) rtol = 1.0e-12
    @test_throws ArgumentError CrackGrowthResistance(; Gc = 0.0, a₀ = 1.0, d₀ = 0.05)
end

@testset "below the threshold: the linear cracked solid" begin
    rve = _cracked_cell()
    law = CrackGrowthDamage(rve; resistance = RISING_CD)
    @test initial_state(law).y == SVector(0.05)
    ε = SVector(2.0e-4, -1.0e-4, 0.5e-4, 1.0e-4, 0, 0)
    σ, 𝕂, st = point_update(law, ε, initial_state(law), 1.0)
    @test st.y == SVector(0.05)
    @test 𝕂 ≈ _stiffness_at(rve, :F, 0.05) rtol = 1.0e-12
    @test σ ≈ 𝕂 * ε rtol = 1.0e-12
    @test MP_CD._driving_forces(ε, (F = 0.05,), law.p).F ≈ _force_fd(rve, :F, 0.05, ε) rtol = 1.0e-6
end

@testset "a growing family sits on its criterion" begin
    rve = _cracked_cell()
    law = CrackGrowthDamage(rve; resistance = RISING_CD)
    r = point_response(law, LoadingPath(range(0, 1; length = 21); ε11 = t -> 2.0e-3 * t))
    @test issorted(r[:F]) && r[:F][end] > 0.1
    k = findlast(==(0.05), r[:F])
    for i in (k + 1):length(r[:F])
        ε = SVector(r[:ε11][i], r[:ε22][i], r[:ε33][i], r[:ε23][i], r[:ε13][i], r[:ε12][i])
        @test _force_fd(rve, :F, r[:F][i], ε) ≈ RISING_CD(r[:F][i]) rtol = 1.0e-6
    end
    # The consistent tangent of a damaging step is the derivative of the update.
    prev = initial_state(law)
    ε = SVector(1.6e-3, -0.2e-3, -0.3e-3, 1.0e-4, 0, -2.0e-4)
    σ, 𝕂, st = point_update(law, ε, prev, 1.0)
    @test st.y[1] > 0.05
    @test 𝕂 ≈ ForwardDiff.jacobian(e -> point_update(law, e, prev, 1.0)[1], ε) rtol = 1.0e-8
end

@testset "unstable growth, regularized by a viscosity" begin
    rve = _cracked_cell()
    @test_throws LocalIntegrationError point_update(
        CrackGrowthDamage(rve; resistance = GRIFFITH_CD), SVector(1.0e-3, 0, 0, 0, 0, 0),
        initial_state(CrackGrowthDamage(rve; resistance = GRIFFITH_CD)), 1.0
    )
    τ = 0.1
    law = CrackGrowthDamage(rve; resistance = GRIFFITH_CD, τ)
    prev = initial_state(law)
    ε, Δt = SVector(1.0e-3, 0, 0, 0, 0, 0), 0.005
    σ, 𝕂, st = point_update(law, ε, prev, Δt)
    d = st.y[1]
    @test (d - 0.05) / Δt ≈ (_force_fd(rve, :F, d, ε) / GRIFFITH_CD(d) - 1) / τ rtol = 1.0e-5
    @test 𝕂 ≈ ForwardDiff.jacobian(e -> point_update(law, e, prev, Δt)[1], ε) rtol = 1.0e-8
    # A smaller time comes closer to the rate-independent jump: more damage, less stress.
    path = LoadingPath(range(0, 1; length = 21); ε11 = t -> 2.0e-3 * t)
    slow, fast = point_response(law, path), point_response(CrackGrowthDamage(rve; resistance = GRIFFITH_CD, τ = 0.01), path)
    @test fast[:F][end] > slow[:F][end] && fast[:σ11][end] < slow[:σ11][end]
    @test check_material_interface(law; verbose = false)
end

@testset "families loaded differently grow differently" begin
    rve = _cracked_cell(; oriented = true)
    law = CrackGrowthDamage(rve; resistance = (F3 = RISING_CD, F1 = d -> 2 * RISING_CD(d)))
    @test MP_CD.state_layout(law) == (F1 = 1, F3 = 1) || MP_CD.state_layout(law) == (F3 = 1, F1 = 1)
    r = point_response(law, LoadingPath(range(0, 1; length = 5); ε33 = t -> 2.0e-3 * t))
    @test r[:F3][end] > 0.05                  # normal to the load: opened, grows
    @test r[:F1][end] == 0.05                 # parallel to the load: no driving force
end

@testset "refusals" begin
    rve = _cracked_cell()
    @test_throws ArgumentError CrackGrowthDamage(rve; resistance = RISING_CD, τ = 0.0)
    @test_throws ArgumentError CrackGrowthDamage(rve; resistance = RISING_CD, families = (:M,))
    plain = RVE()
    add_phase!(plain, :M, Ellipsoid(1.0), Dict(:C => C0_CD); fraction = :rest)
    @test_throws ArgumentError CrackGrowthDamage(plain; resistance = RISING_CD)
end

@testset "a correction leaving the domain of a law is shortened" begin
    # r = log y + 3 is defined for y > 0 only: the first full correction from
    # y = 1 lands at y = -2.
    inside = ResidualLaw(;
        stress = (ε, y, p) -> ε, layout = (y = 1,), y₀ = [1.0],
        residual = (y, g, prev, Δt, p) -> (y = y.y > 0 ? log(y.y) + 3 : oftype(y.y, NaN),)
    )
    @test point_update(inside, zero(SVector{6}), initial_state(inside), 1.0)[3].y[1] ≈ exp(-3) rtol = 1.0e-12
    # No fraction of the correction stays where the residual is finite.
    nowhere = ResidualLaw(;
        stress = (ε, y, p) -> ε, layout = (y = 1,), y₀ = [1.0],
        residual = (y, g, prev, Δt, p) -> (y = y.y >= 1 ? y.y - 0.5 : oftype(y.y, NaN),)
    )
    @test_throws LocalIntegrationError point_update(nowhere, zero(SVector{6}), initial_state(nowhere), 1.0)
end
