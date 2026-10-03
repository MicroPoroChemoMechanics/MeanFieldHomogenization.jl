using Test
using MeanFieldHomogenization
using TensND

# =============================================================================
#  test_visco_law.jl — ViscoLaw type & convenience constructors
# =============================================================================

@testset "ViscoLaw — Heaviside (elastic limit)" begin
    law = heaviside_law(2.5)
    @test visco_mode(law) == :relaxation
    @test law(1.0, 0.5) == 2.5
    @test law(0.5, 1.0) == 0.0   # t < t' ⇒ 0
    @test visco_eval(law, 2.0, 1.0) == 2.5

    # Tensor heaviside
    C = TensISO{3}(30.0, 8.0)
    law_C = heaviside_law(C)
    @test law_C(1.0, 0.5) === C
    @test iszero(TensND.get_data(law_C(0.0, 1.0))[1])
end

@testset "ViscoLaw — Maxwell scalar relaxation" begin
    law = maxwell_relaxation(0.0, [1.0, 0.5], [0.5, 2.0])
    @test visco_mode(law) == :relaxation
    @test law(1.0, 0.0) ≈ 1.0 * exp(-2.0) + 0.5 * exp(-0.5)
    @test law(0.0, 1.0) == 0.0
    # Length mismatch must throw.
    @test_throws ArgumentError maxwell_relaxation(0.0, [1.0], [0.5, 1.0])
end

@testset "ViscoLaw — Kelvin scalar creep" begin
    law = kelvin_creep(0.5, [0.2, 0.1], [1.0, 4.0])
    @test visco_mode(law) == :creep
    expected = 0.5 + 0.2 * (1 - exp(-1.0)) + 0.1 * (1 - exp(-0.25))
    @test law(1.0, 0.0) ≈ expected
    @test law(0.0, 1.0) == 0.0
end

@testset "ViscoLaw — maxwell_iso (4-tensor)" begin
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    @test visco_mode(law) == :relaxation
    out0 = law(0.0, 0.0)
    @test out0 isa TensISO{4, 3}
    α0, β0 = TensND.get_data(out0)
    @test α0 ≈ 30.0   # 3k
    @test β0 ≈ 8.0    # 2μ

    out1 = law(1.0, 0.0)
    α1, β1 = TensND.get_data(out1)
    @test α1 ≈ 30.0 * exp(-1.0)
    @test β1 ≈ 8.0 * exp(-2.0)

    # Causality: t < t' should return 0.
    out_zero = law(0.0, 1.0)
    @test all(iszero, TensND.get_data(out_zero))
end

@testset "ViscoLaw — kelvin_iso (4-tensor) instantaneous + branches" begin
    # Instantaneous-only case (no branches).
    # The instantaneous compliance is the inverse of the stiffness with the
    # same bulk and shear moduli, (1/(3k), 1/(2μ)) on (𝕁, 𝕂).
    law0 = kelvin_iso(10.0, 4.0)
    @test visco_mode(law0) == :creep
    α, β = TensND.get_data(law0(0.0, 0.0))
    @test collect(TensND.get_data(inv(iso_stiffness(10.0, 4.0)))) ≈ [α, β]
    @test α ≈ 1 / 30.0
    @test β ≈ 1 / 8.0

    # With one Kelvin branch on the shear axis.
    law1 = kelvin_iso(10.0, 4.0, Float64[], [2.0], Float64[], [1.0])
    α2, β2 = TensND.get_data(law1(1.0, 0.0))
    @test α2 ≈ 1 / 30.0   # bulk unchanged (no k branches)
    @test β2 ≈ 1 / 8.0 + (1 / 4.0) * (1 - exp(-1.0))

    # And one on the spherical axis.
    law2 = kelvin_iso(10.0, 4.0, [5.0], Float64[], [2.0], Float64[])
    α3, β3 = TensND.get_data(law2(1.0, 0.0))
    @test α3 ≈ 1 / 30.0 + (1 / 15.0) * (1 - exp(-0.5))
    @test β3 ≈ 1 / 8.0

    # Length mismatch must throw.
    @test_throws ArgumentError kelvin_iso(10.0, 4.0, [1.0], Float64[], [1.0, 2.0], Float64[])
end

@testset "ViscoLaw — show / print" begin
    law = heaviside_law(1.0)
    s = sprint(show, law)
    @test occursin("ViscoLaw", s)
    @test occursin(":relaxation", s)
end
