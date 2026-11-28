using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
import Symbolics

# A solid of moduli (k₀, μ₀) with spherical pores filled with a fluid of bulk
# modulus k_f, sealed in each pore (the fluid phase has no shear stiffness).
const K0_CP, MU0_CP = 30.0, 18.0
const C0_CP = TensISO{3}(3 * K0_CP, 2 * MU0_CP)

function _sealed_rve(φ, k_f; C_0 = C0_CP, shape = Ellipsoid(1.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_0); fraction = :rest)
    add_phase!(rve, :F, shape, Dict(:C => TensISO{3}(3 * k_f, zero(k_f))); fraction = φ)
    return rve
end

# Components of a second-order tensor in the canonical basis, and its isotropic part.
_canon2(t) = get_array(TensND.express(t, TensND.CanonicalBasis{3, Float64}()))
_iso2(t) = tr(_canon2(t)) / 3

# The stress-driven dilute scheme relates the field in a pore to the remote
# stress through the compliance of the solid: an isolated pore, exactly.
@testset "an isolated spherical pore: Eshelby and Skempton" begin
    k_f = 2.2
    ct = concentration_tensors(_sealed_rve(1.0e-3, k_f), DiluteDual())
    Bsk = pore_pressure_tensor(ct, :F)
    # Eshelby: the volume strain of the fluid sphere is (3k₀ + 4μ₀)/(3k_f + 4μ₀)
    # times the remote one, tr Σ/(3k₀), and its pressure is −k_f times it.
    ref = k_f * (3K0_CP + 4MU0_CP) / (3K0_CP * (3k_f + 4MU0_CP))
    @test _iso2(Bsk) ≈ ref rtol = 1.0e-12
    @test norm(_canon2(Bsk) - ref * I) < 1.0e-12
    # Skempton's coefficient as Shafiro and Kachanov (1997, eq. 14) write it,
    # B = C_p/(C_p + 1/k_f − 1/k₀), with C_p = 1/k₀ + 3/(4μ₀) the compressibility
    # of the dry spherical pore (their R/E₀ with R = 9(1 − ν₀)/2).
    C_p = 1 / K0_CP + 3 / (4MU0_CP)
    @test 3 * _iso2(Bsk) ≈ C_p / (C_p + 1 / k_f - 1 / K0_CP) rtol = 1.0e-12
    # a fluid as stiff as the solid in compression carries the load as the solid: B = 1
    @test 3 * _iso2(pore_pressure_tensor(concentration_tensors(_sealed_rve(1.0e-3, K0_CP), DiluteDual()), :F)) ≈ 1 rtol = 1.0e-12
end

@testset "derivative with respect to the fluid modulus" begin
    f(k) = _iso2(pore_pressure_tensor(concentration_tensors(_sealed_rve(1.0e-3, k), DiluteDual()), :F))
    g(k) = k * (3K0_CP + 4MU0_CP) / (3K0_CP * (3k + 4MU0_CP))
    @test ForwardDiff.derivative(f, 2.2) ≈ ForwardDiff.derivative(g, 2.2) rtol = 1.0e-10
end

# One family of pores under Mori-Tanaka: every pore has the same pressure, so
# sealed pores and connected ones (Gassmann, through the Biot coefficients of
# the drained cell) give the same undrained bulk modulus and the same Skempton
# coefficient.
@testset "Mori-Tanaka: sealed pores agree with Gassmann" begin
    φ, k_f = 0.2, 2.2
    k_of(C) = (m = Matrix(mandel(C, TensND.CanonicalBasis{3, Float64}())); (m[1, 1] + 2m[1, 2]) / 3)
    dry, wet = _sealed_rve(φ, 0.0), _sealed_rve(φ, k_f)
    C_d = homogenize(dry, MoriTanaka())
    k_d = k_of(C_d)
    gassmann = k_d + (1 - k_d / K0_CP)^2 / (φ / k_f + (1 - φ) / K0_CP - k_d / K0_CP^2)
    @test k_of(homogenize(wet, MoriTanaka())) ≈ gassmann rtol = 1.0e-12
    par = poroelastic_parameters(C_d, C0_CP, φ; k_f)
    Bsk = pore_pressure_tensor(concentration_tensors(wet, MoriTanaka()), :F)
    @test _canon2(Bsk) ≈ _canon2(skempton_tensor(C_d, par.B, par.modulus)) rtol = 1.0e-12
end

# A crack-like spheroid: the pressure answers the normal stress mostly, and its
# tensor turns with the pore.
@testset "a flat pore turns with its normal" begin
    k_f = 2.2
    θ, ϕ = 0.7, 2.1
    n = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    flat(angles) = pore_pressure_tensor(
        concentration_tensors(_sealed_rve(1.0e-3, k_f; shape = Spheroid(1.0e-3; euler_angles = angles)), DiluteDual()), :F
    )
    B₃, Bₙ = _canon2(flat(())), _canon2(flat((θ, ϕ, 0.0)))
    # the same eigenvalues, the normal carrying the largest one
    @test eigvals(Symmetric(Matrix(Bₙ))) ≈ eigvals(Symmetric(Matrix(B₃))) rtol = 1.0e-10
    @test Bₙ * n ≈ B₃[3, 3] * n rtol = 1.0e-10
    # transversely isotropic about the normal, dominated by the normal component
    @test B₃[1, 1] ≈ B₃[2, 2] rtol = 1.0e-12
    @test B₃[3, 3] > 100 * B₃[1, 1]
end

@testset "symbolic fluid modulus" begin
    Symbolics.@variables kf
    ct = concentration_tensors(_sealed_rve(1.0e-3, kf), DiluteDual())
    expr = _iso2(pore_pressure_tensor(ct, :F))
    for k in (0.5, 2.2, 40.0)
        @test Symbolics.value(Symbolics.substitute(expr, Dict(kf => k))) ≈ k * (3K0_CP + 4MU0_CP) / (3K0_CP * (3k + 4MU0_CP)) rtol = 1.0e-12
    end
end
