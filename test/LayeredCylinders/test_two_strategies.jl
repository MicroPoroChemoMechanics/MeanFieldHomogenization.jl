# =============================================================================
#  test_two_strategies.jl — the layered cylinder solved by its radial modes and
#  by its systems of the first order, and its layers entered in every notation
#  of a transversely isotropic material.
#
#  References, independent of what is tested:
#   1. the radial modes of a layer transversely isotropic about the axis
#      (`_pot_cross`, `_axi_cross`, `_shear_cross`, which go through the
#      amplitudes): the power ξ^𝐌 of the systems of the theory page, written
#      here from their formulas, carries the same state;
#   2. for an isotropic layer, the systems of the package, which are written
#      for a layer transversely isotropic about the radius;
#   3. the components C₁₁₁₁ = k + m, C₁₁₂₂ = k − m, C₁₁₃₃ = l, C₃₃₃₃ = n,
#      C₂₃₂₃ = p of a stiffness of Hill's moduli, and its engineering constants
#      in closed form.
# =============================================================================

isdefined(@__MODULE__, :LCyl) || include(joinpath(@__DIR__, "helpers.jl"))
using StaticArrays: SMatrix, SVector, @SMatrix
import SymPyPythonCall

const TwoC_Core = MeanFieldHomogenization.Core

# The systems of a layer transversely isotropic about the axis, of Hill's
# moduli (k, l, m, n, p): antiplane shear, in-plane dilatation with the axial
# strain appended to the state, in-plane shear.
tc_antiplane(L) = @SMatrix [0 1 / L.p; L.p 0]
function tc_dilatation(L)
    (; k, l, m) = L
    η = (k - m) / (k + m)
    return @SMatrix [-η 1 / (k + m) -l / (k + m); 4k * m / (k + m) η 2m * l / (k + m); 0 0 1]
end
function tc_shear(L)
    (; k, m) = L
    η, c = (k - m) / (k + m), 4k * m / (k + m)
    return @SMatrix [-η -2η 1 / (k + m) 0; 2 1 0 1 / m; c 2c η -2; 2c 4c 2η -1]
end

@testset "LayeredCylinder: modes and systems of the first order" begin
    r₁, r₂ = 0.6, 1.1
    ξ = r₂ / r₁
    ez = (0.0, 0.0, 1.0)

    @testset "exponents of a layer transversely isotropic about the axis" begin
        @test sort(real(eigvals(Matrix(tc_antiplane(LC_F))))) ≈ [-1, 1]
        @test sort(real(eigvals(Matrix(tc_dilatation(LC_F))))) ≈ [-1, 1, 1]
        @test sort(real(eigvals(Matrix(tc_shear(LC_F))))) ≈ [-3, -1, 1, 3]
    end

    @testset "the power of the system carries the state of the modes" begin
        φ, τ = 0.7, -1.3
        Y = TwoC_Core.radial_power(tc_antiplane(LC_F), ξ) * SVector(φ, r₁ * τ)
        @test collect(LCyl._pot_cross(r₁, r₂, LC_F.p, (φ, τ))[2]) ≈ [Y[1], Y[2] / r₂]

        u, σ, ε = 0.4, 2.5, 0.3
        Y = TwoC_Core.radial_power(tc_dilatation(LC_F), ξ) * SVector(u, r₁ * σ, ε * r₁)
        @test collect(LCyl._axi_cross(r₁, r₂, LC_F, (u, σ), ε)[2]) ≈ [Y[1], Y[2] / r₂]
        @test Y[3] ≈ ε * r₂

        state = (1.0, -0.4, 2.0, 0.7)
        Y = TwoC_Core.radial_power(tc_shear(LC_F), ξ) * SVector(state[1], state[2], r₁ * state[3], r₁ * state[4])
        @test collect(LCyl._shear_cross(r₁, r₂, LC_F, state)[2]) ≈ [Y[1], Y[2], Y[3] / r₂, Y[4] / r₂] rtol = 1.0e-10
    end

    @testset "an isotropic layer belongs to both classes" begin
        C = iso_stiffness(4.0, 1.5)
        L, radial = LCyl._hill_moduli(C, ez), LCyl._radial_moduli(C)
        @test tc_antiplane(L) ≈ LCyl._pot_system(LCyl._antiplane_modulus(radial))
        @test tc_dilatation(L) ≈ Matrix(LCyl._axi_system(radial))[1:3, 1:3]
        @test tc_shear(L) ≈ LCyl._shear_system(radial)
    end

    @testset "derivatives through both strategies" begin
        state = (1.0, -0.4, 2.0, 0.7)
        layer(m) = (; LC_F..., m)
        by_modes(m) = collect(LCyl._shear_cross(r₁, r₂, layer(m), state)[2])
        function by_power(m)
            Y = TwoC_Core.radial_power(tc_shear(layer(m)), ξ) * SVector(state[1], state[2], r₁ * state[3], r₁ * state[4])
            return [Y[1], Y[2], Y[3] / r₂, Y[4] / r₂]
        end
        @test ForwardDiff.derivative(by_power, LC_F.m) ≈ ForwardDiff.derivative(by_modes, LC_F.m) rtol = 1.0e-9
        anti_modes(p) = collect(LCyl._pot_cross(r₁, r₂, p, (0.7, -1.3))[2])
        anti_power(p) = (Y = TwoC_Core.radial_power(tc_antiplane((; p)), ξ) * SVector(0.7, r₁ * -1.3); [Y[1], Y[2] / r₂])
        @test ForwardDiff.derivative(anti_power, LC_F.p) ≈ ForwardDiff.derivative(anti_modes, LC_F.p)
        @test ForwardDiff.derivative(p -> ForwardDiff.derivative(anti_power, p), LC_F.p) ≈
            ForwardDiff.derivative(p -> ForwardDiff.derivative(anti_modes, p), LC_F.p)
    end

    # ξ^𝐌 is the solution of ξ d𝐏/dξ = 𝐌 𝐏 equal to the identity at ξ = 1: the
    # power of the system and the transfer by the modes both have this property.
    @testset "with symbols, at size two" begin
        S = SymPyPythonCall
        p, a, x = S.symbols("p a xi", positive = true)
        M = S.Sym[0 1 / p; p 0]
        𝟙 = S.Sym[1 0; 0 1]
        solves(P) = all(iszero, S.simplify.(x .* S.diff.(P, x) .- M * P)) && all(iszero, S.simplify.(S.subs.(P, x => 1) .- 𝟙))
        @test solves(Matrix(TwoC_Core.radial_power(tc_antiplane((; p)), x)))
        # The state (φ, r σ_rz) at ξ a, from the state (φ, a σ_rz) at a.
        cross(φ, f) = (s = LCyl._pot_cross(a, x * a, p, (φ, f / a))[2]; [s[1], x * a * s[2]])
        @test solves(hcat(cross(S.Sym(1), S.Sym(0)), cross(S.Sym(0), S.Sym(1))))
    end

    @testset "Hill's moduli of a layer, and its stiffness" begin
        axis = normalize([0.3, -0.5, 0.8])
        C = LCyl._stiffness_tensor(LC_F, axis)
        @test lc_data(C) ≈ lc_data(lc_tensor(LC_F; axis))
        L = LCyl._hill_moduli(C, axis)
        @test [L.k, L.l, L.m, L.n, L.p, L.lt] ≈ [LC_F.k, LC_F.l, LC_F.m, LC_F.n, LC_F.p, LC_F.l]
        κ, μ = 4.0, 1.5
        L = LCyl._hill_moduli(iso_stiffness(κ, μ), axis)
        @test keys(L) == (:k, :l, :m, :n, :p, :lt)
        @test [L.k, L.l, L.m, L.n, L.p, L.lt] ≈ [κ + μ / 3, κ - 2μ / 3, μ, κ + 4μ / 3, μ, κ - 2μ / 3]
    end

    @testset "a layer in every notation of a transversely isotropic material" begin
        (; k, l, m, n, p) = LC_F
        C₀ = lc_tensor(LC_0)
        coating = iso_stiffness(4.0, 1.5)
        E₃, ν₃₁, G₃₁ = n - l^2 / k, l / (2k), p
        E₁ = 4m * (k * n - l^2) / ((k + m) * n - l^2)
        ν₁₂ = ((k - m) * n - l^2) / ((k + m) * n - l^2)
        hoenig = (E₁, ν₁₂, E₁ * ν₃₁ / E₃, E₃ / E₁, 2 * (1 + ν₁₂) * G₃₁ / E₁)
        # About the axis of the fiber.
        contribution(fiber) = lc_data(property_contribution(LayeredCylinder((0.6, 1.0), (fiber, coating)), C₀))
        reference = contribution(lc_tensor(LC_F))
        fibers = (
            ti_from_constants(k, l, m, n, p, ez; form = :hill),
            ti_from_constants(E₁, E₃, ν₁₂, ν₃₁, G₃₁, ez; form = :engineering, role = :stiffness),
            ti_from_constants(hoenig..., ez; form = :hoenig, role = :stiffness),
            inv(ti_from_constants(E₁, E₃, ν₁₂, ν₃₁, G₃₁, ez; form = :engineering)),
        )
        for fiber in fibers
            @test contribution(fiber) ≈ reference rtol = 1.0e-10
        end
        # About the radius of the section.
        chart = CylindricalChart()
        wound_contribution(layer) = lc_data(property_contribution(LayeredCylinder((0.6, 1.0), (coating, layer)), C₀))
        wound_reference = wound_contribution(radial_ti(chart; n, l, k, m, p))
        wound = (
            radial_ti(chart, k, l, m, n, p; form = :hill),
            radial_ti(chart, k + m, k - m, l, n, p; form = :components),
            radial_ti(chart, E₁, E₃, ν₁₂, ν₃₁, G₃₁; form = :engineering, role = :stiffness),
            radial_ti(chart, hoenig...; form = :hoenig, role = :stiffness),
        )
        for layer in wound
            @test wound_contribution(layer) ≈ wound_reference rtol = 1.0e-10
        end
    end
end
