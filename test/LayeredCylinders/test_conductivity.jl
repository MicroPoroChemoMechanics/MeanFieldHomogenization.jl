# =============================================================================
#  test_conductivity.jl — conduction in the layered cylinder. References: the
#  closed form of a cylinder in a transverse gradient, the equivalent fiber of
#  a Kapitza resistance (Hasselman & Johnson 1987, cylinders) and of a
#  conducting surface, the identity between antiplane shear and transverse
#  conduction, and the limit of a vanishing conductivity for an impermeable
#  layer.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

const LC_PC = MeanFieldHomogenization.property_contribution

@testset "LayeredCylinder conduction — closed forms" begin
    K0 = lc_K(2.0, 3.0)
    c1 = LayeredCylinder((1.0,), (lc_K(5.0, 7.0),))
    A = gradient_gradient_concentration(c1, K0, K0)
    @test A isa TensND.TensTI{2}
    @test lc_data(A) ≈ [2 * 2.0 / (2.0 + 5.0), 1.0]
    @test lc_data(LC_PC(c1, K0)) ≈ [(5.0 - 2.0) * 4 / 7, 7.0 - 3.0]
    @test lc_data(flux_gradient_concentration(c1, K0, K0)) ≈ [5.0 * 4 / 7, 7.0]
    # isotropic layers and reference
    ci = LayeredCylinder((1.0,), (TensISO{3}(5.0),))
    @test lc_data(gradient_gradient_concentration(ci, TensISO{3}(2.0), TensISO{3}(2.0))) ≈ [4 / 7, 1.0]
    # Kapitza resistance ρ: the fiber of transverse conductivity k₁/(1 + ρk₁/R)
    ρ, R, ks = 0.3, 0.8, 0.9
    ck = LayeredCylinder((R,), (lc_K(5.0, 7.0),); interfaces = (KapitzaInterface(ρ),))
    @test LC_PC(ck, K0) ≈ LC_PC(LayeredCylinder((R,), (lc_K(5.0 / (1 + ρ * 5.0 / R), 7.0),)), K0)
    @test !(gradient_gradient_concentration(ck, K0, K0) ≈ gradient_gradient_concentration(ck, K0, K0; external = false))
    # conducting surface kˢ: k₁ + kˢ/R across, and 2kˢ/R more along the axis
    cs = LayeredCylinder((R,), (lc_K(5.0, 7.0),); interfaces = (SurfaceConductiveInterface(ks),))
    eq = LayeredCylinder((R,), (lc_K(5.0 + ks / R, 7.0 + 2ks / R),))
    @test LC_PC(cs, K0) ≈ LC_PC(eq, K0)
    @test flux_gradient_concentration(cs, K0, K0) ≈ flux_gradient_concentration(eq, K0, K0)
    @test lc_data(flux_gradient_concentration(cs, K0, K0; external = false))[2] ≈ 7.0
    for c in (ck, cs), ext in (true, false)
        A2, N2 = MeanFieldHomogenization.Core.concentration_and_contribution(c, K0, K0; external = ext)
        A3, B3 = MeanFieldHomogenization.Core.concentration_and_flux_average(c, K0, K0; external = ext)
        @test A2 ≈ A3 ≈ gradient_gradient_concentration(c, K0, K0; external = ext)
        @test N2 ≈ B3 - K0 ⋅ A3 ≈ LC_PC(c, K0, K0; external = ext)
    end
end

@testset "LayeredCylinder conduction — layers, and antiplane shear" begin
    K0 = lc_K(2.0, 3.0)
    itfs = (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3), KapitzaInterface(0.1))
    cyl = LayeredCylinder((0.4, 0.7, 1.0), (lc_K(5.0, 7.0), lc_K(1.0, 2.0), lc_K(3.0, 1.0)); interfaces = itfs)
    f = ntuple(k -> layer_volume_fraction(cyl, k), 3)
    @test sum(f[k] * gradient_gradient_concentration(cyl, K0; layer = k, external = true) for k in 1:3) ≈ gradient_gradient_concentration(cyl, K0, K0)
    @test all(flux_gradient_concentration(cyl, K0; layer = k) ≈ layer_modulus(cyl, k) ⋅ gradient_gradient_concentration(cyl, K0; layer = k) for k in 1:3)
    @test_throws BoundsError gradient_gradient_concentration(cyl, K0; layer = 4)
    @test_throws BoundsError flux_gradient_concentration(cyl, K0; layer = 4)
    # antiplane shear with axial shear moduli p is transverse conduction with k_T = p,
    # a spring sₜ being a Kapitza resistance and a membrane μˢ a conducting surface
    p = (5.0, 1.0, 3.0)
    el = LayeredCylinder(
        (0.4, 0.7, 1.0), ntuple(k -> lc_ti(10.0, 2.0, 4.0, 30.0, p[k]), 3);
        interfaces = (SpringInterface(; sn = 0.0, st = 0.2), MembraneInterface(7.0, 0.3), SpringInterface(; sn = 0.3, st = 0.1))
    )
    C0 = lc_ti(10.0, 2.0, 4.0, 30.0, 2.0)
    @test lc_data(gradient_gradient_concentration(el, C0, C0))[6] ≈ lc_data(gradient_gradient_concentration(cyl, K0, K0))[1]
    @test lc_data(flux_gradient_concentration(el, C0, C0))[6] ≈ 2 * lc_data(flux_gradient_concentration(cyl, K0, K0))[1]
end

@testset "LayeredCylinder conduction — impermeable layers" begin
    K0 = lc_K(2.0, 3.0)
    radii = (0.4, 0.7, 1.0)
    for itfs in (
            (PerfectInterface(), PerfectInterface(), PerfectInterface()),
            (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3), KapitzaInterface(0.1)),
            (SurfaceConductiveInterface(0.3), KapitzaInterface(0.2), SurfaceConductiveInterface(0.1)),
        )
        zero_k = LayeredCylinder(radii, (lc_K(5.0, 7.0), lc_K(0.0, 0.0), lc_K(1.0, 2.0)); interfaces = itfs)
        small_k = LayeredCylinder(radii, (lc_K(5.0, 7.0), lc_K(1.0e-10, 0.0), lc_K(1.0, 2.0)); interfaces = itfs)
        @test gradient_gradient_concentration(zero_k, K0, K0) ≈ gradient_gradient_concentration(small_k, K0, K0) rtol = 1.0e-7
        @test LC_PC(zero_k, K0) ≈ LC_PC(small_k, K0) rtol = 1.0e-7
        for k in 1:3
            @test isapprox(
                lc_data(gradient_gradient_concentration(zero_k, K0; layer = k, external = true)),
                lc_data(gradient_gradient_concentration(small_k, K0; layer = k, external = true)); rtol = 1.0e-6, atol = 1.0e-7
            )
        end
        # the layers it encloses carry no transverse gradient
        @test lc_data(gradient_gradient_concentration(zero_k, K0; layer = 1))[1] == 0.0
    end
    # an impermeable core keeps the gradient the conducting shell imposes on it
    core = LayeredCylinder((0.5, 1.0), (lc_K(0.0, 0.0), lc_K(3.0, 3.0)))
    @test lc_data(gradient_gradient_concentration(core, K0; layer = 1))[1] ≈
        lc_data(gradient_gradient_concentration(LayeredCylinder((0.5, 1.0), (lc_K(1.0e-12, 0.0), lc_K(3.0, 3.0))), K0; layer = 1))[1] rtol = 1.0e-8
    @test_throws "merge the two layers" gradient_gradient_concentration(LayeredCylinder(radii, (lc_K(0.0, 0.0), lc_K(0.0, 0.0), lc_K(1.0, 2.0))), K0, K0)
end

@testset "LayeredCylinder conduction — refused inputs" begin
    K0 = lc_K(2.0, 3.0)
    c = LayeredCylinder((1.0,), (lc_K(5.0, 7.0),))
    @test_throws ArgumentError gradient_gradient_concentration(c, K0, lc_K(2.0, 3.0; axis = (1.0, 0.0, 0.0)))
    @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((1.0,), (lc_K(-1.0, 7.0),)), K0, K0)
    @test_throws "does not apply" gradient_gradient_concentration(LayeredCylinder((1.0,), (lc_K(5.0, 7.0),); interfaces = (SpringInterface(1.0, 1.0),)), K0, K0)
    # a reference given as a general tensor is read when it has the symmetry
    Kgen = MeanFieldHomogenization.Core._tens_in([2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0], TensND.CanonicalBasis{3, Float64}())
    @test gradient_gradient_concentration(c, K0, Kgen) ≈ gradient_gradient_concentration(c, K0, K0)
    Kno = MeanFieldHomogenization.Core._tens_in([2.0 0.0 0.0; 0.0 2.5 0.0; 0.0 0.0 3.0], TensND.CanonicalBasis{3, Float64}())
    @test_throws "relative distance" gradient_gradient_concentration(c, K0, Kno)
end
