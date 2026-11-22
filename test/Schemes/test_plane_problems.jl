# =============================================================================
#  test_plane_problems.jl — two-dimensional cells, in plane strain and in plane
#  stress: the schemes run on tensors of the plane as they do in space.
#
#  References, independent of the schemes:
#   1. the scheme of Mori and Tanaka written by hand with the Hill tensor of
#      the disk, in elasticity and in conduction;
#   2. the classical estimates of a sheet with circular holes, which do not
#      depend on Poisson's ratio: E/E₀ = (1 − f)/(1 + 2f) (Mori-Tanaka),
#      1 − 3f (self-consistent), (1 − f)³ (differential);
#   3. the same medium seen in three dimensions, as aligned cylinders in plane
#      strain: the block of the plane of its estimate.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

_pp_E, _pp_ν = 70.0, 0.25
_pp_C = iso_stiffness_young_poisson(_pp_E, _pp_ν)
_pp_stiff = iso_stiffness_young_poisson(200.0, 0.3)
_pp_disk = Ellipsoid(1.0; dim = 2)
function _pp_cell(c₀, c₁, f; inclusion = _pp_disk, matrix = _pp_disk, key = :C, kw...)
    rve = RVE()
    add_phase!(rve, :matrix, matrix, Dict(key => c₀); fraction = :rest)
    add_phase!(rve, :inclusions, inclusion, Dict(key => c₁); fraction = f, kw...)
    return rve
end
# Young's modulus of a tensor of the plane read as a plane-stress stiffness
_pp_plate_modulus(t) = first(plane_strain_to_plane_stress(young_poisson(t)...))

@testset "Plane problems" begin

    @testset "against the scheme written by hand" begin
        f = 0.2
        for state in (plane_strain_stiffness, plane_stress_stiffness)
            c₀, c₁ = state(_pp_C), state(_pp_stiff)
            A = inv(one(c₀) + hill_tensor(_pp_disk, c₀) ⊡ (c₁ - c₀))
            reference = c₀ + f * ((c₁ - c₀) ⊡ A) ⊡ inv((1 - f) * one(c₀) + f * A)
            estimate = homogenize(_pp_cell(c₀, c₁, f), MoriTanaka(), :C)
            @test estimate isa TensISO{4, 2}
            @test canonical_components(estimate) ≈ canonical_components(reference) rtol = 1.0e-12
            # the tensor of one inclusion, and its contribution
            @test canonical_components(@inferred gradient_gradient_concentration(_pp_disk, c₁, c₀)) ≈ canonical_components(A) rtol = 1.0e-13
            @test canonical_components(property_contribution(_pp_disk, c₁, c₀)) ≈ canonical_components((c₁ - c₀) ⊡ A) rtol = 1.0e-13
        end
        # conduction: k₀ (1 + 2f (k₁ − k₀)/(k₁ + k₀ − f (k₁ − k₀)))
        k₀, k₁ = 1.0, 5.0
        K = homogenize(_pp_cell(TensISO{2}(k₀), TensISO{2}(k₁), f; key = :K), MoriTanaka(), :K)
        @test only(TensND.get_data(K)) ≈ k₀ * (1 + 2 * f * (k₁ - k₀) / (k₁ + k₀ - f * (k₁ - k₀)))
    end

    @testset "a sheet with circular holes" begin
        f = 0.1
        for ν in (0.1, 0.25, 0.4), state in (plane_strain_stiffness, plane_stress_stiffness)
            c = state(iso_stiffness_young_poisson(_pp_E, ν))
            E₀ = _pp_plate_modulus(c)
            modulus(scheme) = _pp_plate_modulus(homogenize(_pp_cell(c, 0 * c, f), scheme, :C))
            @test modulus(MoriTanaka()) ≈ E₀ * (1 - f) / (1 + 2 * f) rtol = 1.0e-10
            @test modulus(SelfConsistent()) ≈ E₀ * (1 - 3 * f) rtol = 1.0e-6
            @test modulus(DifferentialScheme()) ≈ E₀ * (1 - f)^3 rtol = 1.0e-6
        end
        # with respect to Young's modulus of the sheet
        plate(E) = (c = plane_stress_stiffness(iso_stiffness_young_poisson(E, _pp_ν)); _pp_plate_modulus(homogenize(_pp_cell(c, 0 * c, f), MoriTanaka(), :C)))
        @test ForwardDiff.derivative(plate, _pp_E) ≈ (1 - f) / (1 + 2 * f) rtol = 1.0e-10
    end

    @testset "the same medium in three dimensions" begin
        f = 0.15
        c = plane_strain_stiffness(_pp_C)
        function space_cell(inclusion; kw...)
            rve = RVE()
            add_phase!(rve, :matrix, Cylinder(1.0, 1.0), Dict(:C => _pp_C); fraction = :rest)
            add_phase!(rve, :inclusions, inclusion, Dict(:C => 0 * _pp_C); fraction = f, kw...)
            return rve
        end
        # the plane (2, 3) of the cylinders of axis 1 is the plane (1, 2) of the cell
        plane_block(C) = canonical_components(C)[2:3, 2:3, 2:3, 2:3]
        triangle = ConformalHole(polygon_map(3, 4)...)
        ellipse = (plane = Ellipsoid(3.0, 1.0), space = Cylinder(3.0, 1.0))
        for scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
            @test canonical_components(homogenize(_pp_cell(c, 0 * c, f), scheme, :C)) ≈ plane_block(homogenize(space_cell(Cylinder(1.0, 1.0)), scheme, :C)) rtol = 1.0e-6
            @test canonical_components(homogenize(_pp_cell(c, 0 * c, f; inclusion = triangle), scheme, :C)) ≈ plane_block(homogenize(space_cell(triangle), scheme, :C)) rtol = 1.0e-6
        end
        # aligned ellipses: an estimate that is anisotropic in the plane
        @test canonical_components(homogenize(_pp_cell(c, 0 * c, f; inclusion = ellipse.plane), MoriTanaka(), :C)) ≈ plane_block(homogenize(space_cell(ellipse.space), MoriTanaka(), :C)) rtol = 1.0e-9
        # ellipses at random in the plane: the average about the axis, in space
        random_plane = homogenize(_pp_cell(c, 0 * c, f; inclusion = ellipse.plane, symmetrize = IsoSymmetrize()), MoriTanaka(), :C)
        random_space = homogenize(space_cell(ellipse.space; symmetrize = TISymmetrize((1.0, 0.0, 0.0))), MoriTanaka(), :C)
        @test random_plane isa TensISO{4, 2}
        @test canonical_components(random_plane) ≈ plane_block(random_space) rtol = 1.0e-9
    end

    @testset "holes given by a conformal map" begin
        c = plane_stress_stiffness(_pp_C)
        k, μ = bulk_shear(c)
        # the circle: each strain of the hole is (κ + 1)/(4μ) = (1/μ + 1/k)/2 per unit hydrostatic stress
        circle = ConformalHole(())
        H = canonical_components(gradient_gradient_concentration(circle, 0 * c, c) ⊡ inv(c))
        @test H[1, 1, 1, 1] + H[1, 1, 2, 2] ≈ (1 / μ + 1 / k) / 2
        # an ellipse: the Hill tensor of the plane
        ellipse = ConformalHole((0.5,))
        @test canonical_components(gradient_gradient_concentration(ellipse, 0 * c, c)) ≈
            canonical_components(gradient_gradient_concentration(Ellipsoid(3.0, 1.0), 0 * c, c)) rtol = 1.0e-10
        # the reference as a general tensor of the plane that is isotropic
        square = ConformalHole(polygon_map(4, 3)...)
        @test canonical_components(gradient_gradient_concentration(square, 0 * c, Tens(canonical_components(c)))) ≈
            canonical_components(gradient_gradient_concentration(square, 0 * c, c)) rtol = 1.0e-12
        @test iszero(canonical_components(flux_gradient_concentration(square, 0 * c, c)))
        @inferred gradient_gradient_concentration(square, 0 * c, c)
        # conduction: the insulating section, whatever the conductivity
        κ = TensISO{2}(1.3)
        A = canonical_components(gradient_gradient_concentration(ellipse, 0 * κ, κ))
        @test A ≈ canonical_components(gradient_gradient_concentration(Ellipsoid(3.0, 1.0), 0 * κ, κ)) rtol = 1.0e-12
        @test iszero(canonical_components(flux_gradient_concentration(ellipse, 0 * κ, κ)))
        @inferred gradient_gradient_concentration(ellipse, 0 * κ, κ)
        # derivative with respect to a modulus of the plane, against the ellipse of the package
        modulus(inclusion) = x -> canonical_components(gradient_gradient_concentration(inclusion, 0 * c, iso_stiffness(Val(2), x, μ)))[1, 1, 2, 2]
        @test ForwardDiff.derivative(modulus(ellipse), k) ≈ ForwardDiff.derivative(modulus(Ellipsoid(3.0, 1.0)), k) rtol = 1.0e-9
        # a reference that is anisotropic in the plane is refused
        skewed = Tens(canonical_components(c) .+ [(i == j == k == l == 1) * 5.0 for i in 1:2, j in 1:2, k in 1:2, l in 1:2])
        @test_throws ArgumentError gradient_gradient_concentration(square, 0 * c, skewed)
        @test_throws ArgumentError gradient_gradient_concentration(square, 0 * κ, Tens([1.0 0.0; 0.0 2.0]))
    end

    @testset "what is refused" begin
        c = plane_strain_stiffness(_pp_C)
        @test_throws ArgumentError homogenize(_pp_cell(c, 0 * c, 0.1; symmetrize = TISymmetrize()), MoriTanaka(), :C)
    end
end
