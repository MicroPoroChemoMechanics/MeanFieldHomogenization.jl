# A composite of nonlinear phases homogenized by a secant method: linear phases
# give back the linear scheme; two phases of the same law give back the law;
# the uniform-strain (Voigt) bound holds; the consistent tangent is the
# derivative of the whole fixed point; the law runs in the point driver.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_NH = MeanFieldHomogenization.MaterialPoints
const NONE_NH = SVector{0, Float64}()

function _two_phase_rve(f; Cm = TensISO{3}(3 * 50.0e3, 2 * 1.0e4), Ci = TensISO{3}(3 * 100.0e3, 2 * 4.0e4))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => Ci); fraction = f)
    return rve
end

power(; k = 50.0e3, σ₀ = 100.0, m = 0.2) = PowerLawElasticity(; k, σ₀, ε₀ = 1.0e-3, m)

@testset "secant moduli read on the law" begin
    law = power()
    s = secant_moduli(law, 2.0e-3)
    @test s.k ≈ 50.0e3 rtol = 1.0e-12
    @test s.μ ≈ 100.0 * 2.0^0.2 / (3 * 2.0e-3) rtol = 1.0e-12       # σ_eq / (3 ε_eq)
end

@testset "linear phases give back the linear scheme" begin
    lin(μ, k) = PowerLawElasticity(; k, σ₀ = 3μ * 1.0e-3, ε₀ = 1.0e-3, m = 1.0)
    rve = _two_phase_rve(0.3)
    C = MP_NH._mandel(homogenize(rve, MoriTanaka(), :C))
    E = SVector(1.0e-3, 2.0e-4, -3.0e-4, 1.0e-4, 0, 2.0e-4)
    for method in (ModifiedSecant(), ClassicalSecant())
        law = NonlinearHomogenized(rve, Dict(:M => lin(1.0e4, 50.0e3), :I => lin(4.0e4, 100.0e3)); method)
        σ, 𝕂, _ = point_update(law, E, initial_state(law), 0.0)
        @test σ ≈ C * E rtol = 1.0e-12
        @test 𝕂 ≈ C rtol = 1.0e-10
    end
end

@testset "two phases of the same law give back the law" begin
    law = power()
    rve = _two_phase_rve(0.4)
    E = SVector(2.0e-3, -1.4e-3, -0.6e-3, 3.0e-4, 0, -2.0e-4)
    σ_law = MP_NH._fluxes(law, E, NONE_NH)
    for method in (ModifiedSecant(), ClassicalSecant())
        h = NonlinearHomogenized(rve, Dict(:M => law, :I => law); method)
        σ, _, st = point_update(h, E, initial_state(h), 0.0)
        @test σ ≈ σ_law rtol = 1.0e-10
        @test st.y ≈ equivalent_strain(E) * SVector(1.0, 1.0) rtol = 1.0e-10
    end
end

@testset "a stiffer inclusion, the Voigt bound and the two methods" begin
    rve = _two_phase_rve(0.3)
    matrix, inclusion = power(), power(; k = 100.0e3, σ₀ = 400.0)
    E = SVector(2.0e-3, -1.0e-3, -1.0e-3, 0, 0, 0)          # isochoric extension
    # Uniform strain: the phase stresses averaged, an upper bound.
    voigt = 0.7 * MP_NH._fluxes(matrix, E, NONE_NH) + 0.3 * MP_NH._fluxes(inclusion, E, NONE_NH)
    q(σ) = σ[1] - σ[2]
    estimates = map((ModifiedSecant(), ClassicalSecant())) do method
        h = NonlinearHomogenized(rve, Dict(:M => matrix, :I => inclusion); method)
        q(point_update(h, E, initial_state(h), 0.0)[1])
    end
    @test all(<(q(voigt)), estimates)
    @test q(MP_NH._fluxes(matrix, E, NONE_NH)) < minimum(estimates)
    @test estimates[1] < estimates[2]                         # the classical secant is the stiffer
    # A phase without a law keeps its linear stiffness.
    h = NonlinearHomogenized(rve, Dict(:M => matrix))
    @test point_update(h, E, initial_state(h), 0.0)[3].y isa SVector{1}
end

@testset "the tangent is the derivative of the whole fixed point" begin
    h = NonlinearHomogenized(_two_phase_rve(0.3), Dict(:M => power(), :I => power(; m = 0.5, σ₀ = 300.0)))
    E = SVector(1.5e-3, -0.5e-3, -0.8e-3, 2.0e-4, -1.0e-4, 3.0e-4)
    σ, 𝕂, _ = point_update(h, E, initial_state(h), 0.0)
    @test 𝕂 ≈ ForwardDiff.jacobian(e -> point_update(h, e, initial_state(h), 0.0)[1], E) rtol = 1.0e-8
    @inferred point_update(h, E, initial_state(h), 0.0)
    # And with respect to a parameter of a phase law.
    σ₀M(s) = point_update(NonlinearHomogenized(_two_phase_rve(0.3), Dict(:M => power(; σ₀ = s), :I => power(; m = 0.5, σ₀ = 300.0))), E, initial_state(h), 0.0)[1][1]
    @test ForwardDiff.derivative(σ₀M, 100.0) ≈ (σ₀M(100.0 + 1.0e-3) - σ₀M(100.0 - 1.0e-3)) / 2.0e-3 rtol = 1.0e-6
end

@testset "in the point driver" begin
    h = NonlinearHomogenized(_two_phase_rve(0.3), Dict(:M => power(), :I => power(; σ₀ = 400.0)))
    r = point_response(h, LoadingPath(range(0, 1; length = 5); ε11 = t -> 3.0e-3 * t))
    @test abs(r[:σ22][end]) < 1.0e-8 * abs(r[:σ11][end])
    @test issorted(r[:σ11]) && r[:M][end] > r[:I][end] > 0          # the soft matrix strains more
end

@testset "refusals" begin
    rve = _two_phase_rve(0.3)
    @test_throws ArgumentError NonlinearHomogenized(rve, Dict(:M => J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0)))
    # An anisotropic stiffness has no secant counterpart here.
    aniso = from_mandel(SMatrix{6, 6}(Diagonal(SVector(300.0, 280.0, 260.0, 160.0, 150.0, 140.0) .* 1.0e3)))
    @test_throws ArgumentError NonlinearHomogenized(_two_phase_rve(0.3; Ci = aniso), Dict(:I => power()))
end
