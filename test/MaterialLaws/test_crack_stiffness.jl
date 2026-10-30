# The stiffness of a cracked solid as a function of its crack densities: the
# affine compliance of Mori–Tanaka equals the generic homogenization at any
# densities and orientations, and so do the driving forces; the generic route
# is kept wherever the identity does not hold; derivatives go through the
# construction of the affine form.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Random

const MP_KS = MeanFieldHomogenization.MaterialLaws
const SC_KS = MeanFieldHomogenization.Schemes

function _cracked_rve(μ = 20.0e3; extra = false, oriented = true)
    C = TensISO{3}(3 * 30.0e3, 2μ)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C); fraction = :rest)
    if oriented
        rng = MersenneTwister(7)
        add_phase!(rve, :A, PennyCrack(1.0; euler_angles = (π * rand(rng), 2π * rand(rng))), Dict(:C => C); density = 0.05)
        add_phase!(rve, :B, PennyCrack(1.0; euler_angles = (π * rand(rng), 2π * rand(rng))), Dict(:C => C); density = 0.08)
    else
        add_phase!(rve, :A, PennyCrack(1.0), Dict(:C => C); density = 0.05, symmetrize = :iso)
    end
    extra && add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60.0e3, 2 * 40.0e3)); fraction = 0.1)
    return rve
end

@testset "the affine compliance of Mori–Tanaka equals the generic route" begin
    rng = MersenneTwister(11)
    for (rve, fams) in ((_cracked_rve(), (:A, :B)), (_cracked_rve(; oriented = false), (:A,)))
        model = MP_KS.cracked_stiffness(rve, MoriTanaka(), fams)
        @test model isa MP_KS.AffineCrackCompliance
        generic = MP_KS.GenericCrackedStiffness(rve, MoriTanaka(), fams)
        ε = SVector{6}(1.0e-3 * randn(rng, 6))
        for _ in 1:4
            d = SVector{length(fams)}(0.3 * rand(rng, length(fams)))
            @test MP_KS._stiffness_at(model, d) ≈ MP_KS._stiffness_at(generic, d) rtol = 1.0e-12
            @test MP_KS._driving_forces_at(model, ε, d) ≈ MP_KS._driving_forces_at(generic, ε, d) rtol = 1.0e-10
        end
    end
end

@testset "the generic route where the identity does not hold" begin
    @test MP_KS.cracked_stiffness(_cracked_rve(; extra = true), MoriTanaka(), (:A, :B)) isa MP_KS.GenericCrackedStiffness
    @test MP_KS.cracked_stiffness(_cracked_rve(), SelfConsistent(), (:A, :B)) isa MP_KS.GenericCrackedStiffness
    # A damage law on the self-consistent scheme: its stiffness is the scheme's.
    rve = _cracked_rve(; oriented = false)
    law = CrackGrowthDamage(rve; scheme = SelfConsistent(), resistance = d -> 0.04 * (1 + 20 * (d - 0.05)))
    ε = SVector(2.0e-4, -1.0e-4, 0, 0, 0, 0)
    @test point_update(law, ε, initial_state(law), 1.0)[2] ≈ MP_KS._mandel(TensND.canonical(homogenize(rve, SelfConsistent(), :C))) rtol = 1.0e-8
end

@testset "derivatives through the affine form" begin
    d = SVector(0.12, 0.2)
    C_aff(μ) = MP_KS._stiffness_at(MP_KS.cracked_stiffness(_cracked_rve(μ), MoriTanaka(), (:A, :B)), d)
    C_gen(μ) = MP_KS._stiffness_at(MP_KS.GenericCrackedStiffness(_cracked_rve(μ), MoriTanaka(), (:A, :B)), d)
    @test ForwardDiff.derivative(C_aff, 20.0e3) ≈ ForwardDiff.derivative(C_gen, 20.0e3) rtol = 1.0e-10
end
