# =============================================================================
#  test_layered_ellipsoid.jl — the confocal layered ellipsoid in conduction.
#
#  Independent references: the confocal layered spheroid (spheroidal
#  harmonics), the layered sphere, the homogeneous ellipsoid (Hill tensor),
#  and the additivity of the contribution of a core in a coating of the
#  conductivity of the medium.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using Random

const _LE = MeanFieldHomogenization.LayeredEllipsoids
_le_c(t) = Matrix(TensND.canonical_components(t))
_le_k(x) = TensISO{3}(x)

@testset "Layered ellipsoid (conduction)" begin
    K₀ = _le_k(1.0)
    ks = (_le_k(50.0), _le_k(0.2), _le_k(4.0))

    @testset "against the confocal spheroid" begin
        for (disks, focal²) in (((0.5, 0.8, 1.0), 3.0), ((1.2, 1.5, 2.0), -1.0))     # prolate, oblate
            polar = sqrt.(disks .^ 2 .+ focal²)
            spheroid = LayeredSpheroid(polar, disks, ks; Nseries = 10)
            ellipsoid = LayeredEllipsoid(map((d, a) -> (d, d, a), disks, polar), ks)
            for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test _le_c(f(ellipsoid, K₀, K₀)) ≈ _le_c(f(spheroid, K₀, K₀)) rtol = 1.0e-9
            end
            @test all(k -> layer_volume_fraction(ellipsoid, k) ≈ layer_volume_fraction(spheroid, k), 1:3)
        end
    end

    @testset "against the layered sphere" begin
        radii = (0.4, 0.7, 1.0)
        sphere = LayeredSphere(radii, ks)
        ellipsoid = LayeredEllipsoid(map(r -> (r, r, r), radii), ks)
        @test _le_c(gradient_gradient_concentration(ellipsoid, K₀, K₀)) ≈ _le_c(gradient_gradient_concentration(sphere, K₀, K₀)) rtol = 1.0e-12
        @test _le_c(property_contribution(ellipsoid, K₀, K₀)) ≈ _le_c(property_contribution(sphere, K₀, K₀)) rtol = 1.0e-12
    end

    # Three unequal axes from here on.
    core = (0.5, 0.8, 1.3)
    fractions = (0.3, 0.25, 0.45)
    ellipsoid = layered_ellipsoid_from_fractions(core, fractions, ks)

    @testset "the confocal family" begin
        @test all(k -> layer_volume_fraction(ellipsoid, k) ≈ fractions[k], 1:3)
        shifts = [collect(layer_semiaxes(ellipsoid, k)) .^ 2 .- collect(core) .^ 2 for k in 1:3]
        @test all(s -> maximum(s) - minimum(s) < 1.0e-12, shifts)
        @test layer_semiaxes(ellipsoid, 1) == core
        @test outer_semiaxes(ellipsoid) == layer_semiaxes(ellipsoid, 3)
        @test layer_count(ellipsoid) == 3 && layer_modulus(ellipsoid, 2) == ks[2]
        @test !is_homogeneous_inclusion(ellipsoid) && admits_dimension(ellipsoid, 3) && !admits_dimension(ellipsoid, 2)
        @test ellipsoid == layered_ellipsoid_from_fractions(core, fractions, ks)
        @test hash(ellipsoid) == hash(layered_ellipsoid_from_fractions(core, fractions, ks))
        @test occursin("3 layer(s)", sprint(show, ellipsoid))
        @test _le_c(MeanFieldHomogenization.Core.shape_tensor(ellipsoid)) ≈
            _le_c(MeanFieldHomogenization.Core.shape_tensor(Ellipsoid(outer_semiaxes(ellipsoid)...)))
    end

    @testset "limits with a known answer" begin
        # One layer: the homogeneous ellipsoid and its Hill tensor.
        single = LayeredEllipsoid((core,), (ks[1],))
        homogeneous = Ellipsoid(core...)
        @test _le_c(gradient_gradient_concentration(single, K₀, K₀)) ≈ _le_c(gradient_gradient_concentration(homogeneous, ks[1], K₀)) rtol = 1.0e-12
        @test _le_c(property_contribution(single, K₀, K₀)) ≈ _le_c(property_contribution(homogeneous, ks[1], K₀)) rtol = 1.0e-12
        # Layers of one conductivity: the homogeneous outer ellipsoid.
        uniform = layered_ellipsoid_from_fractions(core, fractions, (ks[1], ks[1], ks[1]))
        outer = Ellipsoid(outer_semiaxes(uniform)...)
        @test _le_c(gradient_gradient_concentration(uniform, K₀, K₀)) ≈ _le_c(gradient_gradient_concentration(outer, ks[1], K₀)) rtol = 1.0e-11
        # Coatings of the conductivity of the medium: the core alone, its
        # contribution counted on the volume of the inclusion.
        bare = layered_ellipsoid_from_fractions(core, fractions, (ks[1], K₀, K₀))
        @test _le_c(property_contribution(bare, K₀, K₀)) ≈ fractions[1] .* _le_c(property_contribution(homogeneous, ks[1], K₀)) rtol = 1.0e-11
        # The mean gradient of the core of that inclusion is that of the core alone.
        @test _le_c(gradient_gradient_concentration(bare, K₀; layer = 1)) ≈ _le_c(gradient_gradient_concentration(homogeneous, ks[1], K₀)) rtol = 1.0e-11
    end

    @testset "the tensors of one solution" begin
        A = _le_c(gradient_gradient_concentration(ellipsoid, K₀, K₀))
        B = _le_c(flux_gradient_concentration(ellipsoid, K₀, K₀))
        @test _le_c(property_contribution(ellipsoid, K₀, K₀)) ≈ B - A
        layers = [_le_c(gradient_gradient_concentration(ellipsoid, K₀; layer)) for layer in 1:3]
        @test sum(fractions .* layers) ≈ A
        @test sum(fractions .* (50.0, 0.2, 4.0) .* layers) ≈ B
        @test A ≈ Diagonal(diag(A)) atol = 1.0e-14                     # the axes of the ellipsoid
        @test length(unique(round.(diag(A); digits = 8))) == 3         # three distinct directions
        a, n = MeanFieldHomogenization.Core.concentration_and_contribution(ellipsoid, K₀, K₀)
        @test _le_c(a) ≈ A && _le_c(n) ≈ B - A
        a, b = MeanFieldHomogenization.Core.concentration_and_flux_average(ellipsoid, K₀, K₀)
        @test _le_c(a) ≈ A && _le_c(b) ≈ B
        @test @inferred(gradient_gradient_concentration(ellipsoid, K₀, K₀)) isa TensND.AbstractTens{2, 3}
        @test first(TensND.get_data(MeanFieldHomogenization.LayeredSpheres.layer_property_average(ellipsoid, K₀))) ≈ sum(fractions .* (50.0, 0.2, 4.0))
        @test first(TensND.get_data(MeanFieldHomogenization.LayeredSpheres.layer_inverse_property_average(ellipsoid, Val(2)))) ≈ sum(fractions ./ (50.0, 0.2, 4.0))
    end

    @testset "in a rotated basis" begin
        rng = Xoshiro(20261123)
        basis = TensND.random_basis(rng)
        rotated = layered_ellipsoid_from_fractions(core, fractions, ks; basis)
        R = Matrix(TensND.vecbasis(basis))
        for f in (gradient_gradient_concentration, property_contribution)
            @test _le_c(f(rotated, K₀, K₀)) ≈ R * _le_c(f(ellipsoid, K₀, K₀)) * transpose(R) rtol = 1.0e-11
        end
    end

    @testset "derivatives" begin
        # The conductivity of a coating, against the confocal spheroid differentiated in the same way.
        disks, focal² = (0.5, 0.8, 1.0), 3.0
        polar = sqrt.(disks .^ 2 .+ focal²)
        axial(make) = x -> _le_c(gradient_gradient_concentration(make((_le_k(50.0), _le_k(x), _le_k(4.0))), K₀, K₀))[3, 3]
        by_ellipsoid = ForwardDiff.derivative(axial(k -> LayeredEllipsoid(map((d, a) -> (d, d, a), disks, polar), k)), 0.2)
        by_spheroid = ForwardDiff.derivative(axial(k -> LayeredSpheroid(polar, disks, k; Nseries = 10)), 0.2)
        @test by_ellipsoid ≈ by_spheroid rtol = 1.0e-8
        # The conductivity of the medium and a volume fraction, against a centered difference.
        g(x) = _le_c(property_contribution(ellipsoid, _le_k(x), _le_k(x)))[1, 1]
        @test ForwardDiff.derivative(g, 1.0) ≈ (g(1.0 + 1.0e-6) - g(1.0 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-7
        h(x) = _le_c(property_contribution(layered_ellipsoid_from_fractions(core, (x, 0.55 - x, 0.45), ks), K₀, K₀))[2, 2]
        @test ForwardDiff.derivative(h, 0.3) ≈ (h(0.3 + 1.0e-6) - h(0.3 - 1.0e-6)) / 2.0e-6 rtol = 1.0e-6
    end

    @testset "in the schemes" begin
        disks, focal² = (0.5, 0.8, 1.0), 3.0
        polar = sqrt.(disks .^ 2 .+ focal²)
        pair = (LayeredSpheroid(polar, disks, ks; Nseries = 10), LayeredEllipsoid(map((d, a) -> (d, d, a), disks, polar), ks))
        function effective(inclusion, scheme)
            rve = RVE()
            add_phase!(rve, :matrix, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K₀); fraction = :rest)
            add_phase!(rve, :particles, inclusion, Dict(:K => K₀); fraction = 0.2)
            return _le_c(homogenize(rve, scheme, :K))
        end
        for scheme in (Dilute(), MoriTanaka(), Voigt(), Reuss())
            @test effective(pair[2], scheme) ≈ effective(pair[1], scheme) rtol = 1.0e-9
        end
    end

    @testset "what is refused" begin
        @test_throws "is not confocal with the core" LayeredEllipsoid(((0.5, 0.8, 1.3), (0.6, 0.9, 1.5)), (ks[1], ks[2]))
        @test_throws "must grow from the core outward" LayeredEllipsoid(((0.5, 0.8, 1.3), (0.5, 0.8, 1.3)), (ks[1], ks[2]))
        @test_throws "finite and positive" LayeredEllipsoid(((0.5, -0.8, 1.3),), (ks[1],))
        @test_throws "one isotropic conductivity" LayeredEllipsoid((core, core), (ks[1],))
        @test_throws "positive and sum to 1" layered_ellipsoid_from_fractions(core, (0.5, 0.6), (ks[1], ks[2]))
        @test_throws "must be isotropic" gradient_gradient_concentration(ellipsoid, K₀, TensND.Tens([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]))
        @test_throws "does not exist" gradient_gradient_concentration(ellipsoid, K₀; layer = 4)
    end
end
