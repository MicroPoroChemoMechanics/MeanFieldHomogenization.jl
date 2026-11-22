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
#      strain: the block of the plane of its estimate;
#   4. the classical estimates of a sheet with cracks of density ρ = N b²
#      (Kachanov 1992): E/E₀ = 1/(1 + πρ) without interaction and e^{-πρ} by the
#      differential scheme for cracks at random, and the compliance
#      (1 + 2πρ)/E₀ across parallel cracks.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using StaticArrays: SMatrix

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

    @testset "coated disks" begin
        # concentric rings: the layered cylinder in a cell of the plane, against the
        # block of the plane of the same cylinder in three dimensions
        f = 0.2
        coat = iso_stiffness_young_poisson(20.0, 0.35)
        plane_block(C) = canonical_components(C)[2:3, 2:3, 2:3, 2:3]
        for state in (plane_strain_stiffness, plane_stress_stiffness),
                interfaces in ((PerfectInterface(), PerfectInterface()), (PerfectInterface(), SpringInterface(50.0, 30.0)), (MembraneInterface(4.0, 3.0), PerfectInterface()))
            c₀, c₁, c₂ = state(_pp_C), state(_pp_stiff), state(coat)
            # the three-dimensional materials that have these stiffnesses in plane strain
            C₀, C₁, C₂ = iso_from_plane_strain.((c₀, c₁, c₂))
            rings = LayeredCylinder((0.8, 1.0), (c₁, c₂); interfaces)
            cylinder = LayeredCylinder((0.8, 1.0), (C₁, C₂); interfaces, axis = (1.0, 0.0, 0.0))
            for whole in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test canonical_components(whole(rings, c₁, c₀)) ≈ plane_block(whole(cylinder, C₁, C₀)) rtol = 1.0e-11
                @test canonical_components(whole(rings, c₁, c₀; external = false)) ≈ plane_block(whole(cylinder, C₁, C₀; external = false)) rtol = 1.0e-11
            end
            @test canonical_components(property_contribution(rings, c₀)) ≈ plane_block(property_contribution(cylinder, C₀)) rtol = 1.0e-11
            for layer in 1:2
                @test canonical_components(gradient_gradient_concentration(rings, c₀; layer, external = true)) ≈
                    plane_block(gradient_gradient_concentration(cylinder, C₀; layer, external = true)) rtol = 1.0e-11
                @test canonical_components(flux_gradient_concentration(rings, c₀; layer)) ≈ plane_block(flux_gradient_concentration(cylinder, C₀; layer)) rtol = 1.0e-11
            end
            A, N = MeanFieldHomogenization.Core.concentration_and_contribution(rings, c₁, c₀)
            _, B = MeanFieldHomogenization.Core.concentration_and_flux_average(rings, c₁, c₀)
            @test A ≈ gradient_gradient_concentration(rings, c₁, c₀)
            @test N ≈ property_contribution(rings, c₁, c₀)
            @test B ≈ flux_gradient_concentration(rings, c₁, c₀)
            space = RVE()
            add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:C => C₀); fraction = :rest)
            add_phase!(space, :inclusions, cylinder, Dict(:C => C₁); fraction = f)
            for scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
                estimate = homogenize(_pp_cell(c₀, c₁, f; inclusion = rings), scheme, :C)
                @test estimate isa TensISO{4, 2}
                @test canonical_components(estimate) ≈ plane_block(homogenize(space, scheme, :C)) rtol = 1.0e-6
            end
        end
        c₀, c₁, c₂ = plane_stress_stiffness.((_pp_C, _pp_stiff, coat))
        rings = LayeredCylinder((0.8, 1.0), (c₁, c₂))
        @inferred gradient_gradient_concentration(rings, c₁, c₀)
        # two rings of the same material are a disk
        @test gradient_gradient_concentration(LayeredCylinder((0.8, 1.0), (c₁, c₁)), c₁, c₀) ≈ gradient_gradient_concentration(_pp_disk, c₁, c₀) rtol = 1.0e-12
        # the bounds: averages of the stiffnesses of the plane and of their inverses
        f₁, f₂ = 0.8^2, 1 - 0.8^2
        bounds = _pp_cell(c₀, c₁, f; inclusion = rings)
        @test homogenize(bounds, Voigt(), :C) ≈ (1 - f) * c₀ + f * (f₁ * c₁ + f₂ * c₂)
        @test homogenize(bounds, Reuss(), :C) ≈ inv((1 - f) * inv(c₀) + f * (f₁ * inv(c₁) + f₂ * inv(c₂)))
        # with respect to a modulus of a ring, against the disk when the rings are alike
        coated(x) = bulk_modulus(homogenize(_pp_cell(c₀, c₁, f; inclusion = LayeredCylinder((0.8, 1.0), (iso_stiffness(Val(2), x, 80.0), iso_stiffness(Val(2), x, 80.0)))), MoriTanaka(), :C))
        plain(x) = bulk_modulus(homogenize(_pp_cell(c₀, iso_stiffness(Val(2), x, 80.0), f), MoriTanaka(), :C))
        @test ForwardDiff.derivative(coated, 150.0) ≈ ForwardDiff.derivative(plain, 150.0) rtol = 1.0e-9
        # conduction, against the same cylinder in three dimensions (transverse entries)
        κ₀, κ₁, κ₂ = 1.0, 5.0, 0.3
        for interfaces in ((PerfectInterface(), PerfectInterface()), (PerfectInterface(), KapitzaInterface(0.4)), (SurfaceConductiveInterface(2.0), PerfectInterface()))
            disks = LayeredCylinder((0.8, 1.0), (TensISO{2}(κ₁), TensISO{2}(κ₂)); interfaces)
            cylinder = LayeredCylinder((0.8, 1.0), (TensISO{3}(κ₁), TensISO{3}(κ₂)); interfaces, axis = (1.0, 0.0, 0.0))
            transverse(K) = canonical_components(K)[2, 2]
            k₂, k₃ = TensISO{2}(κ₀), TensISO{3}(κ₀)
            for whole in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test only(TensND.get_data(whole(disks, k₂, k₂))) ≈ transverse(whole(cylinder, k₃, k₃)) rtol = 1.0e-12
            end
            @test only(TensND.get_data(property_contribution(disks, k₂))) ≈ transverse(property_contribution(cylinder, k₃)) rtol = 1.0e-12
            for layer in 1:2
                @test only(TensND.get_data(gradient_gradient_concentration(disks, k₂; layer, external = true))) ≈
                    transverse(gradient_gradient_concentration(cylinder, k₃; layer, external = true)) rtol = 1.0e-12
                @test only(TensND.get_data(flux_gradient_concentration(disks, k₂; layer))) ≈ transverse(flux_gradient_concentration(cylinder, k₃; layer)) rtol = 1.0e-12
            end
            G, Nκ = MeanFieldHomogenization.Core.concentration_and_contribution(disks, k₂, k₂)
            _, Q = MeanFieldHomogenization.Core.concentration_and_flux_average(disks, k₂, k₂)
            @test G ≈ gradient_gradient_concentration(disks, k₂, k₂)
            @test Nκ ≈ property_contribution(disks, k₂, k₂)
            @test Q ≈ flux_gradient_concentration(disks, k₂, k₂)
            space = RVE()
            add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:K => k₃); fraction = :rest)
            add_phase!(space, :inclusions, cylinder, Dict(:K => TensISO{3}(κ₁)); fraction = f)
            plane = homogenize(_pp_cell(k₂, TensISO{2}(κ₁), f; inclusion = disks, key = :K), MoriTanaka(), :K)
            @test only(TensND.get_data(plane)) ≈ transverse(homogenize(space, MoriTanaka(), :K)) rtol = 1.0e-10
        end
        # what is refused: layers that are not tensors of the plane, a void layer, a
        # reference that is anisotropic in the plane
        @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((0.8, 1.0), (_pp_stiff, coat)), c₁, c₀)
        @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((0.8, 1.0), (c₁, 0 * c₂)), c₁, c₀)
        @test_throws ArgumentError gradient_gradient_concentration(rings, c₁, Tens(canonical_components(c₀) .+ [(i == j == k == l == 1) * 5.0 for i in 1:2, j in 1:2, k in 1:2, l in 1:2]))
        @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((0.8, 1.0), (TensISO{2}(κ₁), TensISO{2}(-1.0))), TensISO{2}(κ₁), TensISO{2}(κ₀))
    end

    @testset "cracks of the plane" begin
        ρ = 0.1
        plane_block(X) = canonical_components(X)[2:3, 2:3, 2:3, 2:3]
        crack = RibbonCrack(1.0; angle = 0.3)
        # the frame of the crack: unbounded along e₁, its trace at 0.3 from e₂
        frame = TensND.vecbasis(MeanFieldHomogenization.Cracks.crack_basis(crack))
        @test frame[:, 1] ≈ [1, 0, 0]
        @test frame[:, 2] ≈ [0, cos(0.3), sin(0.3)]
        @test_throws ArgumentError RibbonCrack(1.0; angle = 0.3, euler_angles = (0.1,))
        for (state, χ) in ((plane_strain_stiffness, π * (1 - _pp_ν^2) / _pp_E), (plane_stress_stiffness, π / _pp_E))
            c = state(_pp_C)
            C = iso_from_plane_strain(c)                       # the same plane, seen in three dimensions
            # the opening tensor: χ 𝟏, and the block of the plane of that of the ribbon
            B = @inferred cod_tensor(crack, c)
            @test canonical_components(B) ≈ χ * I(2)
            @test canonical_components(B) ≈ canonical_components(cod_tensor(crack, C))[2:3, 2:3] rtol = 1.0e-12
            # in a reference given as a general tensor of the plane
            @test canonical_components(cod_tensor(crack, Tens(canonical_components(c)))) ≈ canonical_components(B) rtol = 1.0e-10
            for contribution in (inverse_property_contribution, property_contribution)
                @test canonical_components(contribution(crack, c)) ≈ plane_block(contribution(crack, C)) rtol = 1.0e-12
            end
            H, N = MeanFieldHomogenization.Cracks.inverse_and_property_contribution(crack, c)
            @test H ≈ inverse_property_contribution(crack, c)
            @test N ≈ property_contribution(crack, c)
            # with a stiffness between the faces
            Kᵢ = Tens([3.0 0.5; 0.5 8.0])
            Kᵢ₃ = Tens([1.0 0.0 0.0; 0.0 3.0 0.5; 0.0 0.5 8.0])
            @test canonical_components(inverse_property_contribution(crack, c; K_interface = Kᵢ)) ≈
                plane_block(inverse_property_contribution(crack, C; K_interface = Kᵢ₃)) rtol = 1.0e-10
            # the classical estimates of a cracked sheet
            E₀ = _pp_plate_modulus(c)
            cell(; kw...) = (r = RVE(); add_phase!(r, :matrix, _pp_disk, Dict(:C => c); fraction = :rest); add_phase!(r, :cracks, RibbonCrack(1.0; angle = 0.0), Dict(:C => 0 * c); density = ρ, kw...); r)
            at_random = cell(; symmetrize = IsoSymmetrize())
            @test _pp_plate_modulus(homogenize(at_random, DiluteDual(), :C)) ≈ E₀ / (1 + π * ρ) rtol = 1.0e-12
            @test _pp_plate_modulus(homogenize(at_random, MoriTanaka(), :C)) ≈ E₀ / (1 + π * ρ) rtol = 1.0e-12
            @test _pp_plate_modulus(homogenize(at_random, DifferentialScheme(), :C)) ≈ E₀ * exp(-π * ρ) rtol = 1.0e-6
            S = canonical_components(inv(homogenize(cell(), MoriTanaka(), :C)))
            @test E₀ * S[2, 2, 2, 2] ≈ 1 + 2π * ρ rtol = 1.0e-12               # across the cracks
            @test E₀ * S[1, 1, 1, 1] ≈ 1 rtol = 1.0e-12                        # along them
            # against the ribbons in three dimensions, the estimate being anisotropic or not
            for scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme()), random in (false, true)
                space = RVE()
                add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:C => C); fraction = :rest)
                add_phase!(space, :cracks, RibbonCrack(1.0; angle = 0.0), Dict(:C => 0 * C); density = ρ, symmetrize = random ? TISymmetrize((1.0, 0.0, 0.0)) : nothing)
                plane = homogenize(random ? at_random : cell(), scheme, :C)
                @test canonical_components(plane) ≈ plane_block(homogenize(space, scheme, :C)) rtol = 1.0e-6
            end
        end
        # conduction: b = π/(2k₀), and the resistivity along the normal
        κ = TensISO{2}(2.0)
        @test cod_tensor(crack, κ) ≈ π / 4
        @test cod_tensor(crack, Tens([2.0 0.0; 0.0 2.0])) ≈ π / 4
        @test cod_tensor(crack, κ; α_interface = 3.0) ≈ (π / 4) / (1 + 3.0 * π / 4)
        R = canonical_components(inverse_property_contribution(crack, κ))
        @test R ≈ canonical_components(inverse_property_contribution(crack, TensISO{3}(2.0)))[2:3, 2:3] rtol = 1.0e-12
        Rκ, Nκ = MeanFieldHomogenization.Cracks.inverse_and_property_contribution(crack, κ)
        @test canonical_components(Nκ) ≈ -4 * R
        @test canonical_components(property_contribution(crack, κ)) ≈ -4 * R
        conducting = RVE()
        add_phase!(conducting, :matrix, _pp_disk, Dict(:K => κ); fraction = :rest)
        add_phase!(conducting, :cracks, crack, Dict(:K => 0 * κ); density = ρ, symmetrize = IsoSymmetrize())
        @test only(TensND.get_data(homogenize(conducting, DiluteDual(), :K))) ≈ 2.0 / (1 + π * ρ / 2) rtol = 1.0e-12
        # with respect to a modulus of the plane
        @test ForwardDiff.derivative(k -> canonical_components(cod_tensor(crack, iso_stiffness(Val(2), k, 28.0)))[1, 1], 56.0) ≈ -π / (4 * 56.0^2)
        # what is refused: a crack that is not a segment of the plane
        c = plane_strain_stiffness(_pp_C)
        @test_throws ArgumentError cod_tensor(EllipticCrack(1.0, 0.5), c)
        @test_throws ArgumentError cod_tensor(RibbonCrack(1.0; euler_angles = (0.4, 0.3)), c)
        @test_throws ArgumentError cod_tensor(EllipticCrack(1.0, 0.5), κ)
    end

    @testset "the other schemes and tensors" begin
        f = 0.2
        c₀, c₁ = plane_strain_stiffness(_pp_C), plane_strain_stiffness(_pp_stiff)
        cell(; kw...) = (r = RVE(; kw...); add_phase!(r, :matrix, _pp_disk, Dict(:C => c₀); fraction = :rest); add_phase!(r, :inclusions, _pp_disk, Dict(:C => c₁); fraction = f); r)
        reference = homogenize(cell(), MoriTanaka(), :C)
        # disks distributed as disks: the estimate of Mori and Tanaka, as spheres in space
        for scheme in (Maxwell(), PonteCastanedaWillis())
            @test homogenize(cell(; distribution_shape = _pp_disk), scheme, :C) ≈ reference rtol = 1.0e-12
            @test_throws ArgumentError homogenize(cell(; distribution_shape = Ellipsoid(1.0)), scheme, :C)
        end
        # every estimate between the bounds; with stiff disks the estimate of Mori and
        # Tanaka is the lower bound of Hashin and Shtrikman, below the self-consistent one
        k(scheme) = bulk_modulus(isotropify(homogenize(cell(), scheme, :C)))
        @test k(Reuss()) < k(MoriTanaka()) < k(SelfConsistent()) < k(Voigt())
        for scheme in (Dilute(), DiluteDual(), AsymmetricSelfConsistent(), DifferentialScheme())
            @test k(Reuss()) < k(scheme) < k(Voigt())
        end
        # concentration tensors of the estimate: ⟨A⟩ = 𝕀 and C = ⟨C : A⟩
        ct = concentration_tensors(cell(), MoriTanaka(), :C)
        A = Dict(name => gradient_gradient_concentration(ct, name) for name in (:matrix, :inclusions))
        @test (1 - f) * A[:matrix] + f * A[:inclusions] ≈ one(c₀)
        @test (1 - f) * (c₀ ⊡ A[:matrix]) + f * (c₁ ⊡ A[:inclusions]) ≈ reference
        # Eshelby and second Hill tensors of the disk: 𝕊 = ℙ : ℂ₀, ℚ = ℂ₀ − ℂ₀ : ℙ : ℂ₀
        P = hill_tensor(_pp_disk, c₀)
        @test eshelby_tensor(_pp_disk, c₀) ≈ P ⊡ c₀
        @test second_hill_tensor(_pp_disk, c₀) ≈ c₀ - c₀ ⊡ P ⊡ c₀
        κ = TensISO{2}(2.0)
        @test only(TensND.get_data(eshelby_tensor(_pp_disk, κ))) ≈ 1 / 2
        @test only(TensND.get_data(second_hill_tensor(_pp_disk, κ))) ≈ 1.0
    end

    @testset "families of cracks and poromechanics in the plane" begin
        c = plane_stress_stiffness(_pp_C)
        # the decomposition of the compliance over the families of cracks: an
        # identity of each scheme, S = S_solid + Σ π ρᵢ 𝕊ᵢ
        cracked = RVE()
        add_phase!(cracked, :matrix, _pp_disk, Dict(:C => c); fraction = :rest)
        add_phase!(cracked, :F1, RibbonCrack(1.0; angle = 0.0), Dict(:C => 0 * c); density = 0.06)
        add_phase!(cracked, :F2, RibbonCrack(1.0; angle = 0.9), Dict(:C => 0 * c); density = 0.04)
        for scheme in (MoriTanaka(), SelfConsistent(; abstol = 1.0e-13, reltol = 1.0e-13))
            estimate = homogenize(cracked, scheme, :C)
            families = crack_family_compliances(cracked, scheme, estimate)
            S = families.solid
            for (name, Sᵢ) in families.families
                S += delta_inverse_property(cracked.phases[name].geometry, Sᵢ, crack_density(cracked, name))
            end
            @test canonical_components(S) ≈ canonical_components(inv(estimate)) rtol = 1.0e-8
        end
        # a porous sheet: the Biot coefficient is 1 − k/kₛ on the bulk moduli of the plane,
        # and the entry of the plane of the Biot tensor of the same medium in space
        f = 0.2
        solid = plane_strain_stiffness(_pp_C)
        porous = _pp_cell(solid, 0 * solid, f)
        drained = homogenize(porous, MoriTanaka(), :C)
        B = @inferred biot_tensor(drained, solid)
        @test B isa TensISO{2, 2}
        b = only(TensND.get_data(B))
        @test b ≈ 1 - bulk_modulus(drained) / bulk_modulus(solid)
        @test canonical_components(biot_tensor(porous, drained)) ≈ b * I(2)
        space = RVE()
        add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:C => _pp_C); fraction = :rest)
        add_phase!(space, :inclusions, Cylinder(1.0, 1.0), Dict(:C => 0 * _pp_C); fraction = f)
        @test b ≈ canonical_components(biot_tensor(homogenize(space, MoriTanaka(), :C), _pp_C))[2, 2] rtol = 1.0e-12
        # the modulus: 1/M = 𝟏 : 𝕊ₛ : (𝐁 − f 𝟏) = (b − f)/kₛ in the plane
        parameters = poroelastic_parameters(drained, solid, f)
        @test parameters.inverse_modulus ≈ (b - f) / bulk_modulus(solid)
        @test parameters.modulus ≈ biot_modulus(solid, B, f)
        # the same on Kelvin-Mandel matrices, of size 3 in the plane
        km(t) = SMatrix{3, 3}(TensND.mandel(t, CanonicalBasis{2, Float64}()))
        on_matrices = poroelastic_parameters(km(drained), km(solid), f)
        @test on_matrices.B ≈ [b, b, 0]
        @test on_matrices.modulus ≈ parameters.modulus
        # the undrained stiffness and back, the effective stresses
        M = parameters.modulus
        undrained = undrained_stiffness(drained, B, M)
        @test bulk_modulus(isotropify(undrained)) ≈ bulk_modulus(drained) + M * b^2
        @test canonical_components(drained_stiffness(undrained, B, M)) ≈ canonical_components(drained)
        Σ = Tens([1.0 0.2; 0.2 3.0])
        @test canonical_components(terzaghi_stress(Σ, 0.5)) ≈ [1.5 0.2; 0.2 3.5]
        @test canonical_components(biot_effective_stress(Σ, 0.5, B)) ≈ [1.0 0.2; 0.2 3.0] + 0.5 * b * I(2)
    end

    @testset "what is refused" begin
        c = plane_strain_stiffness(_pp_C)
        # an inclusion and a property of different dimensions
        for inclusion in (Ellipsoid(1.0), Cylinder(1.0, 1.0), LayeredSphere((0.5, 1.0), (_pp_C, _pp_stiff)), EllipticCrack(1.0, 0.5))
            @test_throws ArgumentError gradient_gradient_concentration(inclusion, 0 * c, c)
        end
        @test_throws ArgumentError gradient_gradient_concentration(_pp_disk, 0 * _pp_C, _pp_C)
        @test_throws ArgumentError gradient_gradient_concentration(_pp_disk, TensISO{3}(1.0), TensISO{3}(2.0))
        @test_throws ArgumentError homogenize(_pp_cell(c, 0 * c, 0.1; inclusion = Ellipsoid(1.0)), MoriTanaka(), :C)
        @test admits_dimension(RibbonCrack(1.0), 2) && admits_dimension(ConformalHole(()), 2) && admits_dimension(LayeredCylinder((1.0,), (c,)), 2)
        @test !admits_dimension(Ellipsoid(1.0), 2) && !admits_dimension(EllipticCrack(1.0, 0.5), 2) && admits_dimension(_pp_disk, 2)
        @test_throws ArgumentError homogenize(_pp_cell(c, 0 * c, 0.1; symmetrize = TISymmetrize()), MoriTanaka(), :C)
    end
end
