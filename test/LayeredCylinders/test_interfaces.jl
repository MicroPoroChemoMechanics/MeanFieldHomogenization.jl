# =============================================================================
#  test_interfaces.jl — imperfect interfaces of the layered cylinder in
#  elasticity. An interface is the limit of a thin layer: a spring that of a
#  soft one (sₙ = h/(k + m), sₜ = h/m = h/p), a membrane that of a stiff
#  isotropic one (surface Lamé moduli h·2λμ/(λ + 2μ) and h·μ). The layered
#  cylinder with perfect interfaces, checked on its own, is the reference.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

@testset "LayeredCylinder — interfaces as thin layers" begin
    Cf, Cm, C0 = lc_tensor(LC_F), lc_tensor(LC_M), lc_tensor(LC_0)
    h, r₁ = 1.0e-6, 0.6
    κs, μs, sn, st = 3.0, 1.2, 0.05, 0.08
    quantities = (gradient_gradient_concentration, flux_gradient_concentration, MeanFieldHomogenization.property_contribution)
    for (itf, thin) in ((MembraneInterface(κs, μs), lc_thin_membrane(κs, μs, h)), (SpringInterface(; sn, st), lc_thin_spring(sn, st, h)))
        # the interface inside the cylinder
        ref = LayeredCylinder((r₁, 1.0), (Cf, Cm); interfaces = (itf, PerfectInterface()))
        lay = LayeredCylinder((r₁ - h / 2, r₁ + h / 2, 1.0), (Cf, thin, Cm))
        @test all(isapprox(lc_data(g(ref, Cm, C0)), lc_data(g(lay, Cm, C0)); rtol = 1.0e-4) for g in quantities)
        # the interface on its outer boundary, counted with the cylinder
        ref = LayeredCylinder((r₁, 1.0), (Cf, Cm); interfaces = (PerfectInterface(), itf))
        lay = LayeredCylinder((r₁, 1.0 - h, 1.0), (Cf, Cm, thin))
        @test all(isapprox(lc_data(g(ref, Cm, C0)), lc_data(g(lay, Cm, C0)); rtol = 1.0e-4) for g in quantities)
        # … or left to the matrix: the cylinder then ends on its inner side
        inner = LayeredCylinder((r₁, 1.0), (Cf, Cm))
        @test !(gradient_gradient_concentration(ref, Cm, C0) ≈ gradient_gradient_concentration(ref, Cm, C0; external = false)) ||
            !(flux_gradient_concentration(ref, Cm, C0) ≈ flux_gradient_concentration(ref, Cm, C0; external = false))
        for ext in (true, false)
            N = MeanFieldHomogenization.property_contribution(ref, Cm, C0; external = ext)
            @test N ≈ flux_gradient_concentration(ref, Cm, C0; external = ext) - C0 ⊡ gradient_gradient_concentration(ref, Cm, C0; external = ext)
            # The contribution has the major symmetry when the cylinder carries
            # the whole of the perturbation, its outer interface included.
            d = lc_data(N)
            @test isapprox(d[3], d[4]; rtol = 1.0e-10) == ext
        end
    end
end

@testset "LayeredCylinder — what an average counts" begin
    Cf, Cm, C0 = lc_tensor(LC_F), lc_tensor(LC_M), lc_tensor(LC_0)
    cyl = LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = (SpringInterface(; sn = 0.05, st = 0.08), SpringInterface(; kn = 30.0, kt = 20.0)))
    f = (0.36, 0.64)
    Aₖ(k; kw...) = gradient_gradient_concentration(cyl, C0; layer = k, kw...)
    # with their outer interface the layers partition the cylinder
    @test f[1] * Aₖ(1; external = true) + f[2] * Aₖ(2; external = true) ≈ gradient_gradient_concentration(cyl, Cm, C0)
    # with the inner one of layer 2 instead, the cylinder without its outer interface
    @test f[1] * Aₖ(1) + f[2] * Aₖ(2; internal = true) ≈ gradient_gradient_concentration(cyl, Cm, C0; external = false)
    # the material alone carries the stress: a jump is strain without stress
    @test f[1] * (Cf ⊡ Aₖ(1)) + f[2] * (Cm ⊡ Aₖ(2)) ≈ flux_gradient_concentration(cyl, Cm, C0)
    @test !(Aₖ(1) ≈ Aₖ(1; external = true))
    # a spring of infinite stiffness is a perfect interface
    stiff = LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = (SpringInterface(; sn = 0.0, st = 0.0), PerfectInterface()))
    @test MeanFieldHomogenization.property_contribution(stiff, C0) ≈ MeanFieldHomogenization.property_contribution(LayeredCylinder((0.6, 1.0), (Cf, Cm)), C0)
    # a membrane adds its surface stress to the flux, not to the layers
    mem = LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = (MembraneInterface(3.0, 1.2), PerfectInterface()))
    Bₖ(k) = flux_gradient_concentration(mem, C0; layer = k)
    @test !(f[1] * Bₖ(1) + f[2] * Bₖ(2) ≈ flux_gradient_concentration(mem, Cm, C0))
    @test f[1] * gradient_gradient_concentration(mem, C0; layer = 1) + f[2] * gradient_gradient_concentration(mem, C0; layer = 2) ≈
        gradient_gradient_concentration(mem, Cm, C0)
end
