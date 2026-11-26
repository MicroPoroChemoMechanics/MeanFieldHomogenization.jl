# =============================================================================
#  test_two_strategies.jl — the layered sphere solved by its radial modes and
#  by its systems of the first order, and its layers entered in every notation
#  of a transversely isotropic material.
#
#  References, independent of what is tested:
#   1. the transfer matrices of the radial modes of an isotropic layer
#      (`_bulk_layer_transfer`, `_shear_layer_transfer`, the modes r and r⁻² of
#      the temperature): the power ξ^𝐌 of the system of the same layer is the
#      same matrix, on the state whose tractions are multiplied by r;
#   2. the matrices of the theory page, written with ω = 3κ/(3κ + 4μ), and the
#      exponents 1, −2 and 1, 3, −4, −2 of the modes;
#   3. the engineering constants of a stiffness of Hill's moduli (k, l, m, n, p)
#      in closed form, E₃ = n − l²/k, ν₃₁ = l/2k, …: a shell entered by them is
#      the shell entered by Hill's moduli.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import SymPyPythonCall

const TwoS = MeanFieldHomogenization.LayeredSpheres
const TwoS_Core = MeanFieldHomogenization.Core

# 𝐃(r): ones on the displacements, r on the tractions of a state of size 2n.
ts_D(r, n) = Diagonal(vcat(fill(one(r), n), fill(r, n)))
ts_bulk(κ, μ) = TwoS._bulk_system(hill_moduli(iso_stiffness(κ, μ)))
ts_shear(κ, μ) = TwoS._shear_system(hill_moduli(iso_stiffness(κ, μ)))
ts_data(t) = collect(TensND.get_data(t))

@testset "LayeredSphere: modes and systems of the first order" begin
    κ, μ, r₁, r₂ = 30.0, 12.0, 0.7, 1.3
    ξ, ω = r₂ / r₁, 3κ / (3κ + 4μ)

    @testset "the matrices of an isotropic layer and their exponents" begin
        @test ts_bulk(κ, μ) ≈ [1 - 3ω 3 * (1 - ω) / (4μ); 12μ * ω 3ω - 2]
        @test ts_shear(κ, μ) ≈ [
            1 - 3ω 9ω - 3 3 * (1 - ω) / (4μ) 0
            -1 1 0 1 / μ
            12μ * ω -36μ * ω 3ω - 2 6
            -6μ * ω 18μ * ω + 4μ (1 - 3ω) / 2 -2
        ]
        @test sort(real(eigvals(Matrix(ts_bulk(κ, μ))))) ≈ [-2, 1]
        @test sort(real(eigvals(Matrix(ts_shear(κ, μ))))) ≈ [-4, -2, 1, 3]
        # The states of the modes u_r = r and u_r = r⁻² are eigenvectors.
        @test ts_bulk(κ, μ) * [1, 3κ] ≈ [1, 3κ]
        @test ts_bulk(κ, μ) * [1, -4μ] ≈ -2 * [1, -4μ]
    end

    @testset "the power of the system is the transfer matrix of the modes" begin
        @test TwoS_Core.radial_power(ts_bulk(κ, μ), ξ) ≈ ts_D(r₂, 1) * TwoS._bulk_layer_transfer(r₂, r₁, κ, μ) / ts_D(r₁, 1)
        @test TwoS_Core.radial_power(ts_shear(κ, μ), ξ) ≈ ts_D(r₂, 2) * TwoS._shear_layer_transfer(r₂, r₁, κ, μ) / ts_D(r₁, 2) rtol = 1.0e-10
        # The closed form of the theory page.
        @test TwoS_Core.radial_power(ts_bulk(κ, μ), ξ) ≈ [
            (1 - ω) * ξ + ω / ξ^2 (1 - ω) * (ξ - 1 / ξ^2) / (4μ)
            4μ * ω * (ξ - 1 / ξ^2) ω * ξ + (1 - ω) / ξ^2
        ]
        # Conduction: the modes T̂ = r and T̂ = r⁻², of outward flux −k and 2k r⁻³.
        kc = 3.0
        Φ(r) = [r r^-2; -kc 2kc * r^-3]
        M = TwoS._cond_system(radial_ti(SphericalChart(), kc, kc))
        @test M ≈ [0 -1 / kc; -2kc -1]
        @test TwoS_Core.radial_power(M, ξ) ≈ ts_D(r₂, 1) * (Φ(r₂) / Φ(r₁)) / ts_D(r₁, 1)
    end

    @testset "derivatives through both strategies" begin
        by_power(κ) = TwoS_Core.radial_power(ts_bulk(κ, μ), ξ)
        by_modes(κ) = ts_D(r₂, 1) * TwoS._bulk_layer_transfer(r₂, r₁, κ, μ) / ts_D(r₁, 1)
        @test ForwardDiff.derivative(by_power, κ) ≈ ForwardDiff.derivative(by_modes, κ)
        @test ForwardDiff.derivative(k -> ForwardDiff.derivative(by_power, k), κ) ≈
            ForwardDiff.derivative(k -> ForwardDiff.derivative(by_modes, k), κ)
        shear_by_power(μ) = TwoS_Core.radial_power(ts_shear(κ, μ), ξ)
        shear_by_modes(μ) = ts_D(r₂, 2) * TwoS._shear_layer_transfer(r₂, r₁, κ, μ) / ts_D(r₁, 2)
        @test ForwardDiff.derivative(shear_by_power, μ) ≈ ForwardDiff.derivative(shear_by_modes, μ) rtol = 1.0e-9
    end

    # ξ^𝐌 is the solution of ξ d𝐏/dξ = 𝐌 𝐏 equal to the identity at ξ = 1: the
    # power of the system has this property for any moduli, and so has the
    # transfer matrix of the modes of an isotropic layer, for its own matrix 𝐌.
    @testset "with symbols, at size two" begin
        S = SymPyPythonCall
        κˢ, μˢ, a, x = S.symbols("kappa mu a xi", positive = true)
        n, l, k, m, p = S.symbols("n l k m p", positive = true)
        𝟙 = S.Sym[1 0; 0 1]
        solves(P, M) = all(iszero, S.simplify.(x .* S.diff.(P, x) .- M * P)) && all(iszero, S.simplify.(S.subs.(P, x => 1) .- 𝟙))
        general = TwoS._bulk_system((; k, l, m, n, p))
        @test solves(Matrix(TwoS_Core.radial_power(general, x)), Matrix(general))
        by_modes = ts_D(x * a, 1) * TwoS._bulk_layer_transfer(x * a, a, κˢ, μˢ) * inv(ts_D(a, 1))
        @test solves(by_modes, Matrix(ts_bulk(κˢ, μˢ)))
        ω = 3κˢ / (3κˢ + 4μˢ)
        @test all(iszero, S.simplify.(ts_bulk(κˢ, μˢ) .- [1 - 3ω 3 * (1 - ω) / (4μˢ); 12μˢ * ω 3ω - 2]))
    end

    @testset "a shell in every notation of a transversely isotropic material" begin
        k, l, m, n, p = 19.0, 5.0, 11.0, 12.0, 4.0
        chart = SphericalChart()
        C₀, core = iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0)
        contribution(shell) = ts_data(property_contribution(LayeredSphere((0.6, 1.0), (core, shell)), C₀, C₀))
        reference = contribution(radial_ti(chart; n, l, k, m, p))
        # Engineering constants of the stiffness, in closed form (r is the axis 3).
        Eᵣ, νᵣₜ, G = n - l^2 / k, l / (2k), p
        Eₜ = 4m * (k * n - l^2) / ((k + m) * n - l^2)
        νₜ = ((k - m) * n - l^2) / ((k + m) * n - l^2)
        hoenig = (Eₜ, νₜ, Eₜ * νᵣₜ / Eᵣ, Eᵣ / Eₜ, 2 * (1 + νₜ) * G / Eₜ)
        shells = (
            radial_ti(chart, k, l, m, n, p; form = :hill),
            radial_ti(chart, k + m, k - m, l, n, p; form = :components),
            radial_ti(chart, Eₜ, Eᵣ, νₜ, νᵣₜ, G; form = :engineering, role = :stiffness),
            radial_ti(chart, hoenig...; form = :hoenig, role = :stiffness),
            inv(radial_ti(chart, Eₜ, Eᵣ, νₜ, νᵣₜ, G; form = :engineering)),       # the compliance, inverted
            radial_ti(chart, n, 2k, √2 * l, 2m, 2p),                               # Walpole coefficients
        )
        for shell in shells
            @test contribution(shell) ≈ reference rtol = 1.0e-10
            @test collect(values(TwoS_Core.radial_hill_moduli(shell))) ≈ [k, l, m, n, p]
        end
        @test TwoS_Core.radial_hill_moduli(shells[1]) === (k = k, l = l, m = m, n = n, p = p)
    end
end
