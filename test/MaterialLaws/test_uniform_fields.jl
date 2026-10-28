# The composites with uniform fields: linear elastic phases give the Voigt and
# Reuss bounds of the linear schemes; with plastic phases, the uniform-strain
# composite averages the responses of the phases along the same strain path,
# and the uniform-stress one follows the closed form of linear hardening in
# uniaxial stress; the two bound the secant estimate of a power-law composite;
# the tangents are consistent.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_UF = MeanFieldHomogenization.MaterialLaws

_canon_uf(t) = MP_UF._mandel(TensND.change_tens_canon(t))

@testset "linear elastic phases: the Voigt and Reuss bounds" begin
    soft, stiff = LinearElasticity(; E = 10.0e3, ν = 0.2), LinearElasticity(; E = 70.0e3, ν = 0.3)
    laws, fractions = (M = soft, I = stiff), (M = 0.6, I = 0.4)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(10.0e3, 0.2)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70.0e3, 0.3)); fraction = 0.4)
    E = SVector(1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4, 0, -2.0e-4)
    for (composite, scheme) in ((UniformStrain(laws, fractions), Voigt()), (UniformStress(laws, fractions), Reuss()))
        σ, 𝕂, _ = point_update(composite, E, initial_state(composite), 1.0)
        C = _canon_uf(homogenize(rve, scheme, :C))
        @test 𝕂 ≈ C rtol = 1.0e-10
        @test σ ≈ C * E rtol = 1.0e-10
    end
end

const STEEL_UF = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 10.0e3)
const ALLOY_UF = J2Plasticity(; E = 70.0e3, ν = 0.33, σ₀ = 150.0, H = 2.0e3)

@testset "uniform strain: the average of the phases along the same path" begin
    taylor = UniformStrain((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.7))
    r = point_response(taylor, LoadingPath(range(0, 1; length = 21); ε11 = t -> 4.0e-3 * t))
    # The strain path the composite followed, imposed on each phase.
    comps = (:ε11, :ε22, :ε33, :ε23, :ε13, :ε12)
    path = LoadingPath(r.t; NamedTuple{comps}(map(c -> (v = r[c]; t -> v[searchsortedfirst(r.t, t)]), comps))...)
    rS, rA = point_response(STEEL_UF, path), point_response(ALLOY_UF, path)
    @test r[:σ11] ≈ 0.3 * rS[:σ11] + 0.7 * rA[:σ11] rtol = 1.0e-8
    @test r[:A_p] ≈ rA[:p] rtol = 1.0e-8
    @test maximum(abs, r[:σ22]) < 1.0e-6 * maximum(abs, r[:σ11])
    @test check_material_interface(taylor; verbose = false)
end

@testset "uniform stress: linear hardening in uniaxial stress" begin
    sachs = UniformStress((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.7))
    r = point_response(sachs, LoadingPath(range(0, 1; length = 31); ε11 = t -> 6.0e-3 * t))
    # Every phase carries the uniaxial stress σ: its axial strain is σ/E plus
    # (σ − σ₀)/(H) once it yields (linear isotropic hardening, p = ε₁₁ᵖ).
    axial(σ, E, σ₀, H) = σ / E + max(σ - σ₀, 0) / H
    for i in eachindex(r.t)
        σ = r[:σ11][i]
        @test r[:ε11][i] ≈ 0.3 * axial(σ, 200.0e3, 250.0, 10.0e3) + 0.7 * axial(σ, 70.0e3, 150.0, 2.0e3) atol = 1.0e-9
    end
    @test r[:S_ε11][end] < r[:ε11][end] < r[:A_ε11][end]          # the soft phase strains more
    @test r[:Σ_σ11] ≈ r[:σ11]
    @test check_material_interface(sachs; verbose = false)
end

@testset "the two composites bound the secant estimate" begin
    matrix, inclusion = PowerLawElasticity(; k = 50.0e3, σ₀ = 100.0, ε₀ = 1.0e-3, m = 0.2), PowerLawElasticity(; k = 100.0e3, σ₀ = 400.0, ε₀ = 1.0e-3, m = 0.4)
    laws, fractions = (M = matrix, I = inclusion), (M = 0.7, I = 0.3)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.3)
    # Along an isochoric extension, driven in steps.
    path = LoadingPath(range(0, 1; length = 11); ε11 = t -> 2.0e-3 * t, ε22 = t -> -1.0e-3 * t, ε33 = t -> -1.0e-3 * t, ε23 = 0, ε13 = 0, ε12 = 0)
    q(c) = (r = point_response(c, path); r[:σ11][end] - r[:σ22][end])
    @test q(UniformStress(laws, fractions)) < q(NonlinearHomogenized(rve, Dict(pairs(laws)))) < q(UniformStrain(laws, fractions))
end

@testset "derivatives and refusals" begin
    # The tangent of a plastic step is the derivative of the update.
    for c in (UniformStrain((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.7)), UniformStress((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.7)))
        prev = initial_state(c)
        E = SVector(3.0e-3, -1.0e-3, -1.2e-3, 2.0e-4, 0, 1.0e-4)
        @test point_update(c, E, prev, 1.0)[2] ≈ ForwardDiff.jacobian(e -> point_update(c, e, prev, 1.0)[1], E) rtol = 1.0e-7
    end
    @test_throws ArgumentError UniformStrain((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.6))
    @test_throws ArgumentError UniformStrain((S = STEEL_UF, A = ALLOY_UF), (A = 0.7, S = 0.3))
    @test_throws ArgumentError UniformStrain((S = STEEL_UF, P = BiotPoroelasticity(; E = 1.0e3, ν = 0.2, b = 0.5, M = 1.0e3)), (S = 0.5, P = 0.5))
    @test_throws ArgumentError UniformStrain((S = STEEL_UF, X = 1.0), (S = 0.5, X = 0.5))
    @test sprint(show, UniformStress((S = STEEL_UF, A = ALLOY_UF), (S = 0.3, A = 0.7))) == "UniformStress of 2 phases (S, A)"
    # A law without a finite derivative at the start of the iteration fails the
    # step (for the driver to cut), it does not break the linear solve.
    cusp = ResidualLaw(; stress = (ε, y, p) -> ε, layout = (y = 1,), y₀ = [0.0], residual = (y, g, prev, Δt, p) -> (y = sqrt(abs(y.y)) + 1,))
    @test_throws LocalIntegrationError point_update(cusp, zero(SVector{6}), initial_state(cusp), 1.0)
    # External variables reach the phases.
    thermal = ResidualLaw(; stress = (ε, y, p) -> 1.0e3 * (ε - p.α * p.T * identity_mandel()), residual = (y, g, prev, Δt, p) -> (d = y.d,), layout = (d = 1,), p = (α = 1.0e-5, T = 0.0))
    c = UniformStrain((A = thermal, B = thermal), (A = 0.5, B = 0.5))
    @test point_update(c, zero(SVector{6}), initial_state(c), 1.0; external = (T = 100.0,))[1] ≈ -1.0 * identity_mandel() rtol = 1.0e-12
end
