# Hill's self-consistent model of elastoplastic polycrystals. Elastic grains give
# the linear self-consistent scheme of the package; on a step, every grain obeys
# the interaction law (2) of Hill (1965) with the constraint tensor of the tangent
# returned, computed here by the public Hill tensor, and that tangent satisfies the
# self-consistency (18) with the consistent tangents of the grains, computed by
# their own laws; one grain is the aggregate; the response lies under Kröner's
# (whose elastic constraint Hill found too stiff) and Taylor's; the stress is
# homogeneous of degree one in the moduli and stresses of the grains, derivative
# included.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using Random
using ForwardDiff

const MP_HM = MeanFieldHomogenization.MaterialLaws

# Grains whose moduli and stresses are scaled by λ.
_grains_hm(count; λ = 1.0) = [
    CrystalViscoplasticity(; C₁₁ = 168.4e3λ, C₁₂ = 121.4e3λ, C₄₄ = 75.4e3λ, K = λ, n = 5, τ₀ = 10λ, Q = 20λ, b = 5.0, frame = f)
        for f in let rng = MersenneTwister(2)
            [random_basis(rng) for _ in 1:count]
    end
]

_step_hm(m, E, st, Δt; algorithm = LocalNewton()) = MP_HM._mandel_update(m, E, st, Δt; algorithm, external = (;))

_canonical_hm(t) = SMatrix{6, 6}(KM(TensND.change_tens_canon(t)))

@testset "elastic grains: the linear self-consistent scheme" begin
    grains = _grains_hm(6)
    m = Polycrystal(grains; model = HillModel(; reltol = 1.0e-10))
    E = SVector(1.0e-5, -3.0e-6, 2.0e-6, 4.0e-6, 0, -1.0e-6)
    Σ, 𝕃, _ = _step_hm(m, E, initial_state(m), 0.1; algorithm = LocalNewton(; abstol = 1.0e-16))
    rve = RVE()
    for (i, f) in enumerate(
            let rng = MersenneTwister(2)
                [random_basis(rng) for _ in 1:6]
            end
        )
        C = _canonical_hm(TensND.tens_cubic(168.4e3, 121.4e3, 75.4e3, f))
        add_phase!(rve, Symbol(:g, i), Ellipsoid(1.0), Dict(:C => TensND.from_mandel(C)); fraction = i == 1 ? :rest : 1 / 6)
    end
    C = _canonical_hm(homogenize(rve, SelfConsistent(), :C; reltol = 1.0e-10))
    @test Σ ≈ C * E rtol = 1.0e-8
    @test 𝕃 ≈ C rtol = 1.0e-8
end

@testset "the interaction law and the self-consistency of Hill" begin
    grains = _grains_hm(6)
    path = LoadingPath(range(0, 1; length = 9); ε11 = t -> 3.0e-3 * t)
    final = map((TaylorModel(), KronerModel(; E = 120.0e3, ν = 0.3), HillModel())) do model
        point_response(Polycrystal(grains; model), path)
    end
    # Hill's constraint, softer than Kröner's elastic one, lowers the response.
    σ11 = [r[:σ11][end] for r in final]
    @test σ11[1] ≥ σ11[2] > σ11[3]
    # One step of the model from a plastic state.
    m = Polycrystal(grains; model = HillModel(; reltol = 1.0e-8))
    prev = final[3].states[7]
    Δt = 1 / 64
    E = prev.ε + SVector(5.0e-5, -2.5e-5, -2.5e-5, 0, 0, 0)
    Σ, 𝕃, st = _step_hm(m, E, prev, Δt)
    σs = [MP_HM._fluxes(l, s.g, s.y) for (l, s) in zip(grains, st.grains)]
    σₙ = [MP_HM._fluxes(l, s.g, s.y) for (l, s) in zip(grains, prev.grains)]
    εs, εₙ = [s.g for s in st.grains], [s.g for s in prev.grains]
    @test sum(εs) / 6 ≈ E atol = 1.0e-11
    @test sum(σs) / 6 ≈ Σ rtol = 1.0e-10
    𝕃s = inv(_canonical_hm(hill_tensor(Ellipsoid(1.0), TensND.from_mandel(𝕃); reltol = 1.0e-10))) - 𝕃
    Σₙ, Eₙ = sum(σₙ) / 6, sum(εₙ) / 6
    gap = maximum(maximum(abs, (σ - Σ) - (σn - Σₙ) + 𝕃s * ((ε - E) - (εn - Eₙ))) for (σ, σn, ε, εn) in zip(σs, σₙ, εs, εₙ))
    @test gap < 1.0e-7 * maximum(abs, Σ)
    # The consistent tangents of the grains on the step, by their own laws.
    𝕂s = [point_update(l, ε, s, Δt)[2] for (l, ε, s) in zip(grains, εs, prev.grains)]
    𝕃18 = inv(sum(inv(𝕃s + 𝕂) for 𝕂 in 𝕂s) / 6) - 𝕃s
    @test (𝕃18 + transpose(𝕃18)) / 2 ≈ 𝕃 rtol = 1.0e-6
    @inferred _step_hm(m, E, prev, Δt)
end

@testset "one grain, homogeneity and its derivative, refusals" begin
    g = _grains_hm(1)
    path = LoadingPath(range(0, 1; length = 5); ε11 = t -> 2.0e-3 * t)
    @test point_response(Polycrystal(g; model = HillModel()), path)[:σ11] ≈ point_response(only(g), path)[:σ11] rtol = 1.0e-8
    # One elastoplastic step from rest, the moduli and stresses scaled by λ.
    E = SVector(1.5e-4, -6.0e-5, -6.0e-5, 2.0e-5, 0, 0)
    stress(λ) = (m = Polycrystal(_grains_hm(4; λ); model = HillModel()); _step_hm(m, E, initial_state(m), 0.01)[1])
    Σ = stress(1.0)
    @test stress(2.5) ≈ 2.5Σ rtol = 1.0e-6
    @test ForwardDiff.derivative(stress, 1.0) ≈ Σ rtol = 1.0e-6
    # The accelerated fixed point: a linear contraction reaches (I − A)⁻¹b; an
    # iteration still unconverged after its last step is a failed step.
    A, b = SMatrix{2, 2}(0.5, 0.3, -0.2, 0.4), SVector(1.0, 2.0)
    x, out = MP_HM._anderson(x -> (A * x + b, :payload), zero(b), 50, (f, x) -> maximum(abs, f) < 1.0e-13, :test)
    @test x ≈ (I - A) \ b rtol = 1.0e-12
    @test out == :payload
    @test_throws LocalIntegrationError MP_HM._anderson(x -> (A * x + b, nothing), zero(b), 2, (f, x) -> false, :test)
    # A state built without a tangent has none yet.
    st = PolycrystalState([initial_state(only(g))], SVector(1.0, 0, 0, 0, 0, 0), zero(SVector{6, Float64}))
    @test iszero(st.tangent) && st.tangent isa SMatrix{6, 6, Float64}
    @test_throws DomainError HillModel(; reltol = 0.0)
    @test_throws DomainError HillModel(; reltol = 1.0)
end
