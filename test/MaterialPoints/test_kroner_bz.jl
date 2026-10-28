# Polycrystals of grains interacting with a matrix: Kröner (elastic matrix) and
# Berveiller–Zaoui (secant elastoplastic matrix). Every grain obeys the
# interaction law of the papers, written here independently in their own form,
# σ_g = Σ + 2αμ(1 − β)(Eᵖ − εᵖ_g) with α of equation (3.15) of Berveiller and
# Zaoui; the strains average to E; the responses are ordered under Taylor's;
# the tangent is that of the step, by ForwardDiff.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using Random
using ForwardDiff
import Symbolics

const MP_KB = MeanFieldHomogenization.MaterialPoints

const E_KB, ν_KB = 120.0e3, 0.3

# Grains of isotropic elasticity, as in the papers.
function _grains_kb(count)
    C₁₁ = E_KB * (1 - ν_KB) / ((1 + ν_KB) * (1 - 2ν_KB))
    C₁₂ = E_KB * ν_KB / ((1 + ν_KB) * (1 - 2ν_KB))
    return [
        CrystalViscoplasticity(; C₁₁, C₁₂, C₄₄ = (C₁₁ - C₁₂) / 2, K = 1.0, n = 5, τ₀ = 10.0, frame = f)
            for f in random_orientations(count; rng = MersenneTwister(2))
    ]
end

_step_kb(m, E, st) = MP_KB._mandel_update(m, E, st, 0.125; algorithm = LocalNewton(), external = (;))

# The interaction law of the papers, against the grain states of a step.
function _interaction_error(model, m, st)
    μ, k = model.μ, model.k
    ℂ = MP_KB._mandel(TensISO{3}(3k, 2μ))
    σs = [MP_KB._fluxes(law, s.g, s.y) for (law, s) in zip(m.grains, st.grains)]
    εᵖs = [s.g - ℂ \ σ for (s, σ) in zip(st.grains, σs)]
    Eᵖ, Σ = sum(εᵖs) / length(εᵖs), sum(σs) / length(σs)
    β = 2 * (4 - 5ν_KB) / (15 * (1 - ν_KB))
    x = μ * norm(Eᵖ) / norm(deviator(Σ))
    ν = ν_KB
    α = model isa KronerModel ? 1.0 :
        (1 + 6x * (1 + ν) / (7 - 5ν)) / (1 + 2x * (13 - 5ν) / (15 * (1 - ν)) + 8x^2 * (1 + ν) / (15 * (1 - ν)))
    return maximum(maximum(abs, σ - (Σ + 2α * μ * (1 - β) * (Eᵖ - εᵖ))) for (σ, εᵖ) in zip(σs, εᵖs)) / maximum(abs, Σ), α
end

@testset "the interaction laws of the papers" begin
    path = LoadingPath(range(0, 1; length = 9); ε11 = t -> 2.0e-3 * t)
    for model in (KronerModel(; E = E_KB, ν = ν_KB), BerveillerZaouiModel(; E = E_KB, ν = ν_KB))
        m = Polycrystal(_grains_kb(8); model)
        r = point_response(m, path)
        for i in (3, 9)
            st = r.states[i]
            err, α = _interaction_error(model, m, st)
            @test err < 1.0e-9
            # The strains average to E to the absolute tolerance of the solver.
            @test sum(s.g for s in st.grains) / length(st.grains) ≈ st.ε atol = 1.0e-11
            @test st.σ ≈ sum(MP_KB._fluxes(l, s.g, s.y) for (l, s) in zip(m.grains, st.grains)) / length(m.grains) rtol = 1.0e-10
            # The plastic accommodation has set in at the end.
            i == 9 && model isa BerveillerZaouiModel && @test α < 0.5
        end
        # The tangent of a step, from the state reached, by a small increment.
        st = r.states[8]
        E = st.ε + SVector(1.0e-5, -5.0e-6, -5.0e-6, 0, 0, 0)
        𝕂 = _step_kb(m, E, st)[2]
        @test 𝕂 ≈ ForwardDiff.jacobian(e -> _step_kb(m, e, st)[1], E) rtol = 1.0e-7
    end
end

@testset "elastic step and ordering" begin
    grains = _grains_kb(8)
    elastic = SVector(1.0e-5, 0, 0, 0, 0, 0)
    ℂ = MP_KB._mandel(iso_stiffness_young_poisson(E_KB, ν_KB))
    path = LoadingPath(range(0, 1; length = 9); ε11 = t -> 3.0e-3 * t)
    final = map((TaylorModel(), KronerModel(; E = E_KB, ν = ν_KB), BerveillerZaouiModel(; E = E_KB, ν = ν_KB))) do model
        m = Polycrystal(grains; model)
        @test _step_kb(m, elastic, initial_state(m))[1] ≈ ℂ * elastic rtol = 1.0e-12
        point_response(m, path)[:σ11][end]
    end
    @test final[1] ≥ final[2] > final[3]
end

@testset "refusals" begin
    @test_throws DomainError KronerModel(; E = -1.0, ν = 0.3)
    @test_throws DomainError BerveillerZaouiModel(; E = 1.0, ν = 0.5)
    @test_throws DomainError KronerModel(; E = 1.0, ν = -1.0)
    @test_throws ArgumentError BerveillerZaouiModel(; E = Symbolics.Num(1.0), ν = 0.3)
end
