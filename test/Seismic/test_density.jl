# =============================================================================
#  test_density.jl — the mass density of a cell (`average_density`).
#
#  Reference: the mass of the cell divided by its volume, Σ fᵢ ρᵢ written out
#  over the phases that carry volume; a crack family adds nothing.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff
import Symbolics

function _density_cell(ρ_M, ρ_I, f)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(20.0, 12.0), :ρ => ρ_M); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 0.5, 0.3), Dict(:C => iso_stiffness(45.0, 30.0), :ρ => ρ_I); fraction = f)
    add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => iso_stiffness(0.0, 0.0), :ρ => 1.0); density = 0.1)
    return rve
end

@testset "Mass density of a cell" begin
    @test average_density(_density_cell(2.5, 3.0, 0.3)) ≈ 0.7 * 2.5 + 0.3 * 3.0
    @test average_density(_density_cell(2.5, 3.0, 0.3)) isa Float64
    # derivatives: with respect to a density and to a fraction, nested
    @test ForwardDiff.derivative(x -> average_density(_density_cell(2.5, x, 0.3)), 3.0) ≈ 0.3
    @test ForwardDiff.derivative(f -> average_density(_density_cell(2.5, 3.0, f)), 0.3) ≈ 3.0 - 2.5
    @test ForwardDiff.derivative(f -> ForwardDiff.derivative(g -> average_density(_density_cell(2.5, 3.0, f * g)), 1.0), 0.3) ≈ 3.0 - 2.5
    # symbols
    Symbolics.@variables ρ₁ ρ₂ φ
    ρ̄ = average_density(_density_cell(ρ₁, ρ₂, φ))
    @test isequal(Symbolics.simplify(ρ̄ - ((1 - φ) * ρ₁ + φ * ρ₂); expand = true), 0)
end
