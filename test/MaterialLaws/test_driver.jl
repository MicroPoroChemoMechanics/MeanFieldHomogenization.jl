# The point driver: loading paths, mixed control, reading the response by name,
# cutting a step that fails, writing as it goes, and driving the materials of
# the Gauss-point contract that are not laws of MaterialLaws.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const J2_DRV = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)

@testset "loading paths" begin
    @test_throws ArgumentError LoadingPath([0.0]; ε11 = 1.0)
    @test_throws ArgumentError LoadingPath([0.0, 0.0, 1.0]; ε11 = 1.0)
    # Named against the law when run: unknown names and conjugate pairs refused.
    @test_throws ArgumentError point_response(J2_DRV, LoadingPath(0:1; ε44 = 1.0))
    @test_throws ArgumentError point_response(J2_DRV, LoadingPath(0:1; ε11 = 1.0e-3, σ11 = 0.0))
    @test_throws ArgumentError point_response(J2_DRV, LoadingPath(0:1; p = 1.0))
    # A shear named by its twin is the same component.
    # Small enough to stay elastic: √3 · 2μ ε₁₂ < σ₀.
    a = point_response(J2_DRV, LoadingPath(0:0.5:1; ε12 = t -> 5.0e-4 * t))
    b = point_response(J2_DRV, LoadingPath(0:0.5:1; ε21 = t -> 5.0e-4 * t))
    @test a[:σ12] == b[:σ21] == b[:σ12]
    @test a[:ε12][end] ≈ 5.0e-4                    # a tensor component, not 2ε₁₂
    μ = 200.0e3 / 2.6
    @test a[:σ12][end] ≈ 2μ * 5.0e-4 rtol = 1.0e-12
    @test length(a) == 3 && a[:t] == [0.0, 0.5, 1.0]
end

@testset "reading a response" begin
    r = point_response(J2_DRV, LoadingPath(range(0, 1; length = 11); ε11 = t -> 4.0e-3 * t))
    @test r[:p] ≈ r[:εᵖ11] rtol = 1.0e-12          # uniaxial: the axial plastic strain is p
    @test r[:εᵖ] isa Vector{<:SVector{6}}
    @test r[:εᵖ22] ≈ -r[:p] / 2 rtol = 1.0e-10
    @test_throws KeyError r[:x]
    @test_throws KeyError r[:εᵖ44]
    @test_throws KeyError r[:σ44]
end

@testset "writing as it goes, and starting from a state" begin
    seen = Tuple{Int, Float64, Float64}[]
    path = LoadingPath(range(0, 1; length = 6); ε11 = t -> 4.0e-3 * t)
    r = point_response(J2_DRV, path; on_step = (i, t, g, F, st) -> push!(seen, (i, t, F[1])))
    @test first.(seen) == 1:6
    @test last.(seen) == r[:σ11]
    # The same path in two halves: the second starts from the end state of the first.
    first_half = point_response(J2_DRV, LoadingPath(range(0, 0.5; length = 4); ε11 = t -> 4.0e-3 * t))
    second = point_response(J2_DRV, LoadingPath(range(0.5, 1; length = 4); ε11 = t -> 4.0e-3 * t); state = first_half.states[end])
    @test second[:σ11][end] ≈ r[:σ11][end] rtol = 1.0e-10
    @test second[:σ11][1] ≈ first_half[:σ11][end] rtol = 1.0e-12
end

@testset "a step that fails is cut" begin
    # A law whose local equations are singular on steps longer than 0.3 and
    # integrate ȧ = 1 exactly on shorter ones: the step from 0 to 1 is cut twice.
    law = ResidualLaw(;
        stress = (ε, y, p) -> ε, layout = (a = 1,),
        residual = (y, g, prev, Δt, p) -> (a = Δt > 0.3 ? zero(y.a) : y.a - prev.y.a - Δt,),
    )
    path = LoadingPath([0.0, 1.0]; ε11 = 0, ε22 = 0, ε33 = 0, ε23 = 0, ε13 = 0, ε12 = 0)
    @test point_response(law, path)[:a] ≈ [0.0, 1.0]
    @test_throws LocalIntegrationError point_response(law, path; max_cuts = 1)
    # The driver's own iteration, limited to one, on a law whose plastic branch
    # is curved (with linear hardening one correction would be exact).
    sat = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, hardening = p -> 250.0 + 150.0 * (1 - exp(-400.0 * p)))
    err = (@test_throws PointDriverError point_response(sat, LoadingPath([0.0, 1.0]; ε11 = t -> 5.0e-3 * t); maxiters = 1, max_cuts = 0)).value
    @test err isa PointDriverError && err.time == 1.0
    @test occursin("prescribed fluxes", sprint(showerror, err))
end

@testset "derivatives through the driver" begin
    endσ(E) = point_response(
        J2Plasticity(; E, ν = 0.3, σ₀ = 250.0, H = 2.0e3),
        LoadingPath(range(0, 1; length = 11); ε11 = t -> 5.0e-3 * t)
    )[:σ11][end]
    ref(E) = 250.0 + E * 2.0e3 / (E + 2.0e3) * (5.0e-3 - 250.0 / E)
    @test endσ(200.0e3) ≈ ref(200.0e3) rtol = 1.0e-11
    @test ForwardDiff.derivative(endσ, 200.0e3) ≈ ForwardDiff.derivative(ref, 200.0e3) rtol = 1.0e-9
end

@testset "driving the materials of the contract" begin
    C0, C1 = TensISO{3}(3 * 30.0, 2 * 18.0), TensISO{3}(3 * 90.0, 2 * 60.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C1); fraction = 0.2)
    he = HomogenizedElastic(rve, MoriTanaka())
    r = point_response(he, LoadingPath(0:0.5:1; ε11 = t -> 1.0e-3 * t))
    # Uniaxial stress: σ₁₁ = ε₁₁ / S₁₁₁₁, S the homogenized compliance.
    S = inv(mandel(homogenize(rve, MoriTanaka()), TensND.CanonicalBasis{3, Float64}()))
    @test r[:σ11][end] ≈ 1.0e-3 / S[1, 1] rtol = 1.0e-10
    @test abs(r[:σ22][end]) < 1.0e-12

    # A poroelastic material of the contract: two gradients, four blocks.
    prve = RVE()
    add_phase!(prve, :M, Ellipsoid(1.0), Dict(:C => C0, :K => TensISO{3}(1.0e-18)); fraction = :rest)
    add_phase!(prve, :F, PennyCrack(1.0), Dict(:C => C0, :K => TensISO{3}(1.0e-18)); density = 0.1)
    rock = FracturedPoroelasticRock(prve, MoriTanaka(); ω₀ = (1.0e-3,), k_matrix = 1.0e-18)
    path = LoadingPath(0:0.5:1; ε33 = t -> -1.0e-5 * t, ε11 = 0, ε22 = 0, ε23 = 0, ε13 = 0, ε12 = 0, p = t -> 0.1 * t)
    rr = point_response(rock, path)
    direct = material_response(rock, (ε = from_mandel(SVector(0, 0, -1.0e-5, 0, 0, 0.0)), p = 0.1), rr.states[2], 0.5)
    @test rr[:σ33][end] ≈ canonical_components(stress(direct))[3, 3] rtol = 1.0e-10
    @test rr[:φ][end] ≈ direct.fluxes.φ rtol = 1.0e-10
end

@testset "external variables" begin
    # A thermoelastic law written by a user: the temperature T is read, not
    # returned; free expansion α (T − T₀) 1.
    k, μ, α, T₀ = 160.0e3, 80.0e3, 1.2e-5, 293.0
    J = identity_mandel() * transpose(identity_mandel()) / 3
    ℂ = 3k * J + 2μ * (one(SMatrix{6, 6, Float64}) - J)
    elastic_part(g, p) = g.ε - p.α * (p.T - p.T₀) * identity_mandel()
    thermo = PotentialLaw(; energy = (g, a, p) -> sum(elastic_part(g, p) .* (p.ℂ * elastic_part(g, p))) / 2, p = (; ℂ, α, T₀))
    zero6 = zero(SVector{6, Float64})
    σ, _, _ = point_update(thermo, zero6, initial_state(thermo), 0.0; external = (T = 343.0,))
    @test σ ≈ -3k * α * 50 * identity_mandel() rtol = 1.0e-12
    @test ForwardDiff.derivative(T -> point_update(thermo, zero6, initial_state(thermo), 0.0; external = (T = T,))[1][1], 343.0) ≈
        -3k * α rtol = 1.0e-12
    r = material_response(thermo, (ε = from_mandel(zero6),), initial_state(thermo), 0.0; external = (T = 343.0,))
    @test canonical_components(stress(r))[1, 1] ≈ -3k * α * 50 rtol = 1.0e-12
    # Along a path: free expansion under a temperature ramp, then blocked.
    ramp = (T = t -> 293.0 + 100.0 * t,)
    free = point_response(thermo, LoadingPath(0:0.5:1; external = ramp))
    @test free[:ε11] ≈ α * 100 .* [0.0, 0.5, 1.0] atol = 1.0e-15
    @test free[:ε12] ≈ zeros(3) atol = 1.0e-15
    blocked = point_response(thermo, LoadingPath(0:0.5:1; ε11 = 0, ε22 = 0, ε33 = 0, ε23 = 0, ε13 = 0, ε12 = 0, external = ramp))
    @test blocked[:σ22] ≈ -3k * α * 100 .* [0.0, 0.5, 1.0] atol = 1.0e-9
    # A name already in `p` takes the value given: a yield stress varying along a path.
    soft = point_response(J2_DRV, LoadingPath(0:0.5:1; ε11 = t -> 3.0e-3 * t, external = (σ₀ = 200.0,)))
    @test soft[:σ11][end] ≈ 200.0 + 200.0e3 * 2.0e3 / 202.0e3 * (3.0e-3 - 200.0 / 200.0e3) rtol = 1.0e-10
    # A material of the contract that reads no external variable refuses them.
    @test_throws ArgumentError point_response(HomogenizedElastic(TensISO{3}(3 * 30.0, 2 * 18.0)), LoadingPath(0:1; ε11 = 1.0e-3, external = (T = 300.0,)))
end
