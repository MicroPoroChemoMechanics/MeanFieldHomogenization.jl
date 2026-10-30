# The incremental tangent approach: elastic phases give the linear Mori–Tanaka
# estimate exactly; with a plastic matrix the response lies between the
# uniform-stress and uniform-strain composites; the isotropic projection of a
# cubic stiffness is the Voigt average of its orientations; the tangent of the
# composite is consistent.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra

const MP_IT = MeanFieldHomogenization.MaterialLaws

function _cell_it(f = 0.3)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
    return rve
end

const PLASTIC_IT = (M = J2Plasticity(; E = 70.0e3, ν = 0.33, σ₀ = 150.0, H = 1.0e3), I = LinearElasticity(; E = 400.0e3, ν = 0.2))

@testset "the isotropic projection" begin
    k, μ = TensND.bulk_shear(TensND.isotropify(TensND.from_mandel(MP_IT._mandel(TensISO{3}(3 * 40.0, 2 * 25.0)))))
    @test k ≈ 40.0 rtol = 1.0e-14
    @test μ ≈ 25.0 rtol = 1.0e-14
    C11, C12, C44 = 168.4, 121.4, 75.4
    k, μ = TensND.bulk_shear(TensND.isotropify(TensND.from_mandel(MP_IT._mandel(TensND.canonical(TensND.cubic_from_constants(C11, C12, C44, TensND.RotatedBasis(0.4, 1.1, 0.3)))))))
    @test k ≈ (C11 + 2C12) / 3 rtol = 1.0e-12
    @test μ ≈ (C11 - C12 + 3C44) / 5 rtol = 1.0e-12
end

@testset "elastic phases: the Mori–Tanaka estimate" begin
    elastic = (M = LinearElasticity(; E = 70.0e3, ν = 0.33), I = LinearElasticity(; E = 400.0e3, ν = 0.2))
    c = IncrementalTangent(_cell_it(), elastic)
    E = SVector(1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4, 0, -2.0e-4)
    σ, 𝕂, _ = point_update(c, E, initial_state(c), 1.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70.0e3, 0.33)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(400.0e3, 0.2)); fraction = 0.3)
    C = mandel(homogenize(rve, MoriTanaka(), :C))
    @test 𝕂 ≈ C rtol = 1.0e-12
    @test σ ≈ C * E rtol = 1.0e-12
end

@testset "a plastic matrix: between the uniform-field composites" begin
    path = LoadingPath(range(0, 1; length = 21); ε11 = t -> 1.0e-2 * t)
    c = IncrementalTangent(_cell_it(), PLASTIC_IT)
    r = point_response(c, path)
    rT = point_response(UniformStrain(PLASTIC_IT, (M = 0.7, I = 0.3)), path)
    rS = point_response(UniformStress(PLASTIC_IT, (M = 0.7, I = 0.3)), path)
    @test all(rS[:σ11][2:end] .< r[:σ11][2:end] .< rT[:σ11][2:end])
    # The mixture rule holds, and the particles strain less than the matrix.
    @test 0.7 * r[:M_ε11][end] + 0.3 * r[:I_ε11][end] ≈ r[:ε11][end] rtol = 1.0e-10
    @test r[:I_ε11][end] < r[:ε11][end] < r[:M_ε11][end]
    @test check_material_interface(c; verbose = false)
    @inferred point_update(c, SVector(4.0e-3, -1.0e-3, -1.0e-3, 0, 0, 0), initial_state(c), 1.0)
end

@testset "refusals" begin
    @test_throws ArgumentError IncrementalTangent(_cell_it(), (M = PLASTIC_IT.M,))
    @test_throws ArgumentError IncrementalTangent(_cell_it(), (M = PLASTIC_IT.M, I = BiotPoroelasticity(; E = 1.0, ν = 0.2, b = 0.5, M = 1.0)))
    aniso = RVE()
    add_phase!(aniso, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
    add_phase!(aniso, :I, Ellipsoid(1.0), Dict(:C => TensND.cubic_from_constants(3.0, 1.0, 1.0, TensND.CanonicalBasis{3, Float64}())); fraction = 0.3)
    @test_throws ArgumentError IncrementalTangent(aniso, PLASTIC_IT)
    c = IncrementalTangent(_cell_it(), PLASTIC_IT)
    @test MP_IT._with_external(c, (;)) === c
    @test MP_IT._with_external(c, (T = 1.0,)) isa IncrementalTangent
end
