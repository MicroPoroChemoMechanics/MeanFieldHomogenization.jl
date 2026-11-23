# =============================================================================
#  The two stiffnesses of the plane of an isotropic aging material: plane
#  strain (the in-plane block) and plane stress (ε₃₃ condensed out with Volterra
#  products and inverse).
#
#  References, independent of the code under test:
#   1. constant moduli: the elastic plane-strain and plane-stress stiffnesses
#      of TensND at every pair of instants;
#   2. the definition: the full three-dimensional operator applied to a strain
#      history whose ε₃₃ is the one plane stress gives returns σ₃₃ = 0 and the
#      in-plane stress of the plane-stress operator;
#   3. a cell of the plane: holes in a matrix of constant moduli in plane stress
#      against the elastic estimate of the same cell.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff

const PS_TIMES = [0.0, 0.3, 0.8, 1.5, 3.0, 6.0]
ps_decay(t, tp) = exp(-(t - tp) / 1.5)
# An aging isotropic material of space: k and μ relax, and depend on the age.
ps_law(k₀ = 10.0) = ViscoLaw(
    (t, tp) -> t ≥ tp ? iso_stiffness(k₀ + 4 * ps_decay(t, tp) + 0.5 * tp, 3 + 2 * ps_decay(t, tp) * (1 + tp)) :
        iso_stiffness(0.0, 0.0)
)

@testset "Plane states of an aging material" begin
    R = VolterraOperator(ps_law(), PS_TIMES)
    n = length(PS_TIMES)

    @testset "constant moduli: the elastic stiffnesses of the plane" begin
        C = iso_stiffness(12.0, 5.0)
        R₀ = VolterraOperator(C, PS_TIMES)
        for (operator, elastic) in ((plane_strain_stiffness, plane_strain_stiffness), (plane_stress_stiffness, plane_stress_stiffness))
            P = operator(R₀)
            @test P.class isa TensISO{4, 2}
            @test Matrix(P) ≈ Matrix(VolterraOperator(elastic(C), PS_TIMES)) rtol = 1.0e-12
        end
    end

    @testset "plane stress: the definition, on the full operator of space" begin
        # In-plane strain history; ε₃₃ from σ₃₃ = 0, as plane stress takes it.
        ε₁₁ = [0.0, 1.0, 1.0, 0.7, 0.7, 0.2]
        ε₂₂ = [0.0, -0.3, 0.4, 0.4, 0.1, 0.1]
        α, β = R.blocks
        λ = (α .- β) ./ 3
        ε₃₃ = -(LowerTriangular(λ .+ β) \ (λ * (ε₁₁ .+ ε₂₂)))
        # The history in Kelvin-Mandel components, instant by instant.
        h = zeros(6, n)
        h[1, :] .= ε₁₁
        h[2, :] .= ε₂₂
        h[3, :] .= ε₃₃
        σ = Matrix(R * TimeHistory(h, PS_TIMES))
        @test maximum(abs, σ[3, :]) < 1.0e-12 * maximum(abs, σ)
        P = plane_stress_stiffness(R)
        σₚ = Matrix(P * TimeHistory(vcat(h[1:2, :], zeros(1, n)), PS_TIMES))
        @test σₚ[1:2, :] ≈ σ[1:2, :] rtol = 1.0e-12
    end

    @testset "the order of the factors: not the pointwise conversion" begin
        # Converting the moduli instant by instant gives another operator for an aging material.
        pointwise = VolterraOperator(
            ViscoLaw((t, tp) -> plane_stress_stiffness(visco_eval(ps_law(), t, tp))), PS_TIMES
        )
        gap = norm(Matrix(plane_stress_stiffness(R)) - Matrix(pointwise)) / norm(Matrix(pointwise))
        @test gap > 1.0e-3
        # Plane strain is the block, pointwise as well.
        block = VolterraOperator(ViscoLaw((t, tp) -> plane_strain_stiffness(visco_eval(ps_law(), t, tp))), PS_TIMES)
        @test Matrix(plane_strain_stiffness(R)) ≈ Matrix(block) rtol = 1.0e-12
    end

    @testset "a sheet with holes in plane stress" begin
        C = iso_stiffness(12.0, 5.0)
        f = 0.15
        function sheet(matrix)
            rve = RVE()
            add_phase!(rve, :matrix, Ellipsoid(1.0; dim = 2), Dict(:C => matrix); fraction = :rest)
            add_phase!(rve, :holes, Ellipsoid(1.0; dim = 2), Dict(:C => 0 * plane_stress_stiffness(C)); fraction = f)
            return rve
        end
        R₀ = VolterraOperator(C, PS_TIMES)
        H = homogenize_alv(sheet(plane_stress_stiffness(R₀)), MoriTanaka(), :C; times = PS_TIMES)
        E = homogenize(sheet(plane_stress_stiffness(C)), MoriTanaka(), :C)
        @test Matrix(H) ≈ Matrix(VolterraOperator(E, PS_TIMES)) rtol = 1.0e-9
    end

    @testset "derivatives with respect to a modulus" begin
        g(k₀) = Matrix(plane_stress_stiffness(VolterraOperator(ps_law(k₀), PS_TIMES)))[end, end]
        # An independent check of the derivative: the elastic case, λ* = 2λμ/(λ + 2μ), α₂ = 2λ* + 2μ.
        h(k) = (C = iso_stiffness(k, 5.0); first(TensND.get_data(plane_stress_stiffness(C))))
        # The same coefficient α = 2λ* + 2μ of the plane, read from the first block of the operator.
        hv(k) = first(plane_stress_stiffness(VolterraOperator(iso_stiffness(k, 5.0), PS_TIMES)).blocks)[1, 1]
        @test ForwardDiff.derivative(hv, 12.0) ≈ ForwardDiff.derivative(h, 12.0) rtol = 1.0e-12
        # The aging case, against a centered difference used as the reference of the derivative.
        @test ForwardDiff.derivative(g, 10.0) ≈ (g(10.0 + 1.0e-5) - g(10.0 - 1.0e-5)) / 2.0e-5 rtol = 1.0e-7
    end

    @test_throws MethodError plane_stress_stiffness(VolterraOperator(TensISO{2}(3.0, 2.0), PS_TIMES))
end
