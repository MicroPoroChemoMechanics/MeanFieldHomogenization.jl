# The Taylor polycrystal: equal to the uniform-strain composite of the same
# grains (another code path); its elastic tangent is the Voigt average of the
# rotated cubic stiffnesses; a random texture makes that average isotropic, with
# the closed-form Voigt moduli of a cubic crystal; the flow stress of a random
# FCC aggregate in tension is the Taylor factor (about 3.06, Bishop and Hill)
# times the critical resolved shear stress.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using Random
import Symbolics

const MP_PC = MeanFieldHomogenization.MaterialLaws
const C11_PC, C12_PC, C44_PC = 168.4e3, 121.4e3, 75.4e3          # MPa, a copper-like crystal

_grain_pc(frame; τ₀ = 10.0, K = 0.2) = CrystalViscoplasticity(; C₁₁ = C11_PC, C₁₂ = C12_PC, C₄₄ = C44_PC, K, n = 10, τ₀, frame)

@testset "the same as the uniform-strain composite of its grains" begin
    frames = let rng = MersenneTwister(3)
        [random_basis(rng) for _ in 1:2]
    end
    grains = _grain_pc.(frames)
    # Steps short enough that neither path cuts them: a viscous law integrated
    # on other substeps differs at first order in the step.
    path = LoadingPath(range(0, 1; length = 81); ε11 = t -> 4.0e-3 * t, ε22 = t -> -1.0e-3 * t, ε33 = t -> -1.0e-3 * t, ε23 = 0, ε13 = 0, ε12 = t -> 1.0e-3 * t)
    r = point_response(Polycrystal(grains, [0.4, 0.6]), path)
    rc = point_response(UniformStrain((a = grains[1], b = grains[2]), (a = 0.4, b = 0.6)), path)
    @test r[:σ11] ≈ rc[:σ11] rtol = 1.0e-9
    @test r[:σ12] ≈ rc[:σ12] rtol = 1.0e-9
    @test grain_states(r.states[end])[2].y ≈ rc.states[end].y[25:48] rtol = 1.0e-9
end

@testset "elastic tangent: the Voigt average of the rotated crystals" begin
    frames = let rng = MersenneTwister(5)
        [random_basis(rng) for _ in 1:5]
    end
    pc = Polycrystal(_grain_pc.(frames))
    E = SVector(1.0e-5, 0, 0, 0, 0, 0)
    𝕂 = MP_PC._mandel_update(pc, E, initial_state(pc), 1.0; algorithm = LocalNewton(), external = (;))[2]
    voigt = sum(MP_PC._mandel(TensND.change_tens_canon(TensND.tens_cubic(C11_PC, C12_PC, C44_PC, f))) for f in frames) / 5
    @test 𝕂 ≈ voigt rtol = 1.0e-12
    # The contract material gives the same.
    r = material_response(pc, (ε = TensND.from_mandel(E),), initial_state(pc), 1.0)
    @test MP_PC._mandel(TensND.change_tens_canon(r.tangents.σε)) ≈ voigt rtol = 1.0e-12
end

@testset "a random texture is isotropic on average" begin
    frames = let rng = MersenneTwister(11)
        [random_basis(rng) for _ in 1:4000]
    end
    C = sum(MP_PC._mandel(TensND.change_tens_canon(TensND.tens_cubic(C11_PC, C12_PC, C44_PC, f))) for f in frames) / length(frames)
    # The Voigt moduli of a cubic crystal.
    k_V, μ_V = (C11_PC + 2C12_PC) / 3, (C11_PC - C12_PC + 3C44_PC) / 5
    J = SMatrix{6, 6}([i <= 3 && j <= 3 ? 1 / 3 : 0.0 for i in 1:6, j in 1:6])
    iso = 3k_V * J + 2μ_V * (I - J)
    @test norm(C - iso) / norm(iso) < 1.0e-2
    # The bulk modulus is the same in every orientation.
    @test sum(C[1:3, 1:3]) / 9 ≈ k_V rtol = 1.0e-12
end

@testset "the Taylor factor of a random FCC aggregate" begin
    τ₀ = 10.0
    pc = Polycrystal(
        _grain_pc.(
            let rng = MersenneTwister(1)
                [random_basis(rng) for _ in 1:100]
            end; τ₀, K = 0.05
        )
    )
    r = point_response(pc, LoadingPath(range(0, 1; length = 26); ε11 = t -> 1.0e-2 * t))
    # Bishop and Hill (1951): M ≈ 3.06; sampling (100 grains) and the small
    # viscous overstress leave a few percent.
    @test r[:σ11][end] / τ₀ ≈ 3.06 rtol = 0.03
    @test maximum(abs, r[:σ22]) < 1.0e-6 * r[:σ11][end]
end

@testset "refusals" begin
    g = _grain_pc(TensND.CanonicalBasis{3, Float64}())
    @test_throws ArgumentError Polycrystal(typeof(g)[])
    @test_throws ArgumentError Polycrystal([g, g], [0.5, 0.6])
    @test_throws ArgumentError Polycrystal([g], [1.0, 0.0])
    @test_throws ArgumentError Polycrystal([g], [Symbolics.variable(:w)])
    @test_throws ArgumentError Polycrystal([BiotPoroelasticity(; E = 1.0, ν = 0.2, b = 0.5, M = 1.0)])
end

@testset "a runaway iteration is a failed step, not a result" begin
    # A crystal pushed in one long step far beyond its critical stress: Newton's
    # iterates run away (residual growing to 1e20 and beyond) while their last
    # correction looks small against their own size. The step must fail, for the
    # driver to cut it, and not return them.
    g = _grain_pc(random_basis(MersenneTwister(3)))
    E = SVector(1.0e-3, -2.5e-4, -2.5e-4, 0, 0, sqrt(2) * 2.5e-4)
    @test_throws LocalIntegrationError point_update(g, E, initial_state(g), 0.125)
    # Cut into short steps, the same loading integrates.
    r = point_response(g, LoadingPath(range(0, 0.125; length = 2); ε11 = t -> 8.0e-3 * t, ε22 = t -> -2.0e-3 * t, ε33 = t -> -2.0e-3 * t, ε23 = 0, ε13 = 0, ε12 = t -> 2.0e-3 * t))
    @test all(isfinite, r[:σ11]) && abs(r[:σ11][end]) < 1.0e3
end
