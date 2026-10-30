# `update_points!`: all the quadrature points of a mesh in one call, for a
# finite-element code that evaluates its strains in one array. Each point gives
# what `point_update` gives it alone, bit for bit; the result does not depend on
# the number of tasks; the new states land in `states.new`; a derivative goes
# through the batch; and a wrong size, a material of other fields or a point that
# fails to integrate is refused with its own error.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using Random
using ForwardDiff

const MP_UP = MeanFieldHomogenization.MaterialLaws

# `ncells` cells of `nqp` points, strains drawn around a uniaxial tension.
function _up_strains(rng, n, scale)
    return scale .* (repeat([1.0, -0.3, -0.3, 0.0, 0.0, 0.0], 1, n) .+ 0.4 .* randn(rng, 6, n))
end

function _up_reference(material, ε, states, Δt)
    σ, 𝕂 = similar(ε), zeros(6, 6, size(ε, 2))
    new = deepcopy(states.old)
    nqp = length(first(states.old))
    for p in axes(ε, 2)
        cell, qp = divrem(p - 1, nqp) .+ (1, 1)
        F, K, st = MP_UP._mandel_update(material, SVector{6}(ε[:, p]), states.old[cell][qp], Δt; algorithm = LocalNewton(), external = (;))
        σ[:, p] .= F
        𝕂[:, :, p] .= K
        new[cell][qp] = st
    end
    return σ, 𝕂, new
end

@testset "update_points!: each point as alone, any number of tasks" begin
    rng = Random.Xoshiro(20261029)
    ncells, nqp = 7, 4
    n = ncells * nqp
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.2)
    laws = (M = NortonViscoplasticity(; E = 70.0e3, ν = 0.3, K = 300.0, n = 3), I = NortonViscoplasticity(; E = 400.0e3, ν = 0.25, K = 900.0, n = 3))
    for (material, scale) in ((J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3), 2.0e-3), (QuasiElasticComposite(rve, laws), 1.0e-3))
        ε = _up_strains(rng, n, scale)
        states = QuadratureStates(material, ncells, nqp)
        σ_ref, 𝕂_ref, new_ref = _up_reference(material, ε, states, 0.1)
        for ntasks in (1, 3, 16)
            σ, 𝕂 = fill(NaN, 6, n), fill(NaN, 6, 6, n)
            @test update_points!(σ, 𝕂, ε, states, 0.1, material; ntasks) === (σ, 𝕂)
            @test σ == σ_ref && 𝕂 == 𝕂_ref
            @test states.new == new_ref
        end
        # The states written are the ones the next step reads once committed.
        commit!(states)
        @test states.old == new_ref
    end
    # A plasticity point against the exported single-point update.
    law = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)
    ε = _up_strains(rng, 4, 3.0e-3)
    states = QuadratureStates(law, 2, 2)
    σ, 𝕂 = zeros(6, 4), zeros(6, 6, 4)
    update_points!(σ, 𝕂, ε, states, 1.0, law)
    F, K, _ = point_update(law, SVector{6}(ε[:, 3]), states.old[2][1], 1.0)
    @test σ[:, 3] == F && 𝕂[:, :, 3] == K
end

@testset "update_points!: a derivative through the batch" begin
    # σ = ℂ : (x ε) for a linear law: dσ/dx = ℂ : ε exactly. The states hold
    # duals, so their vectors are left untyped.
    law = LinearElasticity(; E = 70.0e3, ν = 0.3)
    ε = _up_strains(Random.Xoshiro(1), 6, 1.0e-3)
    st = initial_state(law)
    states_any() = QuadratureStates{Any}([Any[st, st] for _ in 1:3], [Any[st, st] for _ in 1:3])
    function stress(x)
        σ, 𝕂 = zeros(typeof(x), 6, 6), zeros(typeof(x), 6, 6, 6)
        update_points!(σ, 𝕂, x .* ε, states_any(), 1.0, law)
        return σ
    end
    σ₁ = zeros(6, 6)
    update_points!(σ₁, zeros(6, 6, 6), ε, states_any(), 1.0, law)
    @test ForwardDiff.derivative(stress, 1.0) ≈ σ₁ rtol = 1.0e-14
end

@testset "update_points!: refusals" begin
    law = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)
    states = QuadratureStates(law, 2, 3)
    @test_throws DimensionMismatch update_points!(zeros(6, 5), zeros(6, 6, 6), zeros(6, 6), states, 1.0, law)
    @test_throws ArgumentError update_points!(zeros(6, 6), zeros(6, 6, 6), zeros(6, 6), states, 1.0, law; ntasks = 0)
    # A poroelastic material has 7 gradient and 7 flux components: arrays of 6
    # rows are refused with the sizes it needs.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0e3, 2 * 20.0e3)); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => TensISO{3}(3 * 30.0e3, 2 * 20.0e3)); density = 0.1)
    poro = ClosingCrackPoroelasticity(rve; closure = ExponentialClosure(20.0), porosity = 0.005)
    @test_throws DimensionMismatch update_points!(zeros(6, 6), zeros(6, 6, 6), zeros(6, 6), QuadratureStates(poro, 2, 3), 1.0, poro)
    # A point that fails to integrate: its own error, not a task failure.
    ε = zeros(6, 6)
    ε[:, 4] .= (0.05, -0.02, -0.02, 0.0, 0.0, 0.0)
    @test_throws LocalIntegrationError update_points!(zeros(6, 6), zeros(6, 6, 6), ε, states, 1.0, law; algorithm = LocalNewton(; maxiters = 1), ntasks = 2)
    # Any error other than a failed task is passed on as it came.
    @test MP_UP._first_failure(InterruptException()) isa InterruptException
end

@testset "update_points!: a material of several fields (strain and pressure)" begin
    # The gradients of each point stacked in one column, the strain then the
    # pressure; the fluxes the stress then the fluid content; a 7 × 7 tangent.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0e3, 2 * 20.0e3)); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0; euler_angles = (0.3, 1.1)), Dict(:C => TensISO{3}(3 * 30.0e3, 2 * 20.0e3)); density = 0.1)
    rng = Random.Xoshiro(7)
    for material in (
            ClosingCrackPoroelasticity(rve; closure = ExponentialClosure(20.0), porosity = 0.005),
            BiotPoroelasticity(; E = 10.0e3, ν = 0.25, b = 0.8, M = 5.0e3),
        )
        ncells, nqp = 3, 2
        n = ncells * nqp
        G = vcat(1.0e-3 .* randn(rng, 6, n), -5.0 .* rand(rng, 1, n))
        states = QuadratureStates(material, ncells, nqp)
        for ntasks in (1, 3)
            F, 𝕂 = fill(NaN, 7, n), fill(NaN, 7, 7, n)
            update_points!(F, 𝕂, G, states, 0.5, material; ntasks)
            for p in axes(G, 2)
                cell, qp = divrem(p - 1, nqp) .+ (1, 1)
                Fp, Kp, st = MP_UP._mandel_update(material, SVector{7}(G[:, p]), states.old[cell][qp], 0.5; algorithm = LocalNewton(), external = (;))
                @test F[:, p] == Fp && 𝕂[:, :, p] == Kp && states.new[cell][qp] == st
            end
        end
        @test_throws DimensionMismatch update_points!(zeros(7, n), zeros(7, 6, n), G, states, 0.5, material)
    end
end
