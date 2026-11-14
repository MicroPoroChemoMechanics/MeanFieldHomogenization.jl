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
    h, r₁ = 1.0e-7, 0.6           # the error of the limit is proportional to h
    κs, μs, sn, st = 3.0, 1.2, 0.05, 0.08
    quantities = (gradient_gradient_concentration, flux_gradient_concentration, MeanFieldHomogenization.property_contribution)
    for (itf, thin) in ((MembraneInterface(κs, μs), lc_thin_membrane(κs, μs, h)), (SpringInterface(; sn, st), lc_thin_spring(sn, st, h)))
        # the interface inside the cylinder
        ref = LayeredCylinder((r₁, 1.0), (Cf, Cm); interfaces = (itf, PerfectInterface()))
        lay = LayeredCylinder((r₁ - h / 2, r₁ + h / 2, 1.0), (Cf, thin, Cm))
        @test all(lc_close(g(ref, Cm, C0), g(lay, Cm, C0)) for g in quantities)
        # the interface on its outer boundary, counted with the cylinder
        ref = LayeredCylinder((r₁, 1.0), (Cf, Cm); interfaces = (PerfectInterface(), itf))
        lay = LayeredCylinder((r₁, 1.0 - h, 1.0), (Cf, Cm, thin))
        @test all(lc_close(g(ref, Cm, C0), g(lay, Cm, C0)) for g in quantities)
        # … or left to the matrix: the thin layer then lies outside the cylinder,
        # whose strain and stress are those of its two layers alone
        out = LayeredCylinder((r₁, 1.0, 1.0 + h), (Cf, Cm, thin))
        f = ntuple(k -> layer_volume_fraction(out, k), 2)
        two_layers(g) = sum(f[k] * g(out, C0; layer = k) for k in 1:2) / sum(f)
        @test lc_close(gradient_gradient_concentration(ref, Cm, C0; external = false), two_layers(gradient_gradient_concentration))
        @test lc_close(flux_gradient_concentration(ref, Cm, C0; external = false), two_layers(flux_gradient_concentration))
        # what the outer interface carries changes sides: the opening of a spring,
        # the surface stress of a membrane (which does not open)
        moved = itf isa SpringInterface ? gradient_gradient_concentration : flux_gradient_concentration
        kept = itf isa SpringInterface ? flux_gradient_concentration : gradient_gradient_concentration
        @test !lc_close(moved(ref, Cm, C0; external = false), moved(ref, Cm, C0))
        @test lc_close(kept(ref, Cm, C0; external = false), kept(ref, Cm, C0))
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

@testset "LayeredCylinder — refused and converted inputs" begin
    Cf, Cm, C0 = lc_tensor(LC_F), lc_tensor(LC_M), lc_tensor(LC_0)
    # vectors are stored as tuples
    c = LayeredCylinder([0.6, 1.0], [Cf, Cm]; interfaces = [SpringInterface(20.0, 10.0), PerfectInterface()])
    @test c.moduli isa Tuple && c.interfaces isa Tuple
    @test gradient_gradient_concentration(c, C0, C0) ≈
        gradient_gradient_concentration(LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = (SpringInterface(20.0, 10.0), PerfectInterface())), C0, C0)
    @test_throws "AbstractInterface" LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = (0.3, PerfectInterface()))
    # a conductivity where a stiffness is expected, and the reverse
    ck = LayeredCylinder((1.0,), (lc_K(5.0, 7.0),))
    @test_throws "same physics" gradient_gradient_concentration(ck, C0, C0)
    @test_throws "same physics" gradient_gradient_concentration(LayeredCylinder((1.0,), (Cf,)), lc_K(2.0, 3.0), lc_K(2.0, 3.0))
    # an operator not invariant under the mirrors through the axis
    skew = TensND.TensTI{4}(9.0, 10.0, 1.0, 1.0, 4.0, 6.0, 0.5, 0.0, (0.0, 0.0, 1.0))
    @test_throws "relative distance" gradient_gradient_concentration(LayeredCylinder((1.0,), (Cf,)), C0, skew)
    # an operator without major symmetry is read as it is: C₃₃₁₁ ≠ C₁₁₃₃
    asym = TensND.TensTI{4}(9.0, 10.0, 1.2, 1.8, 4.0, 6.0, (0.0, 0.0, 1.0))
    L = LCyl._hill_moduli(asym, (0.0, 0.0, 1.0))
    @test [L.n, 2L.k, sqrt(2) * L.lt, sqrt(2) * L.l, 2L.m, 2L.p] ≈ [9.0, 10.0, 1.2, 1.8, 4.0, 6.0]
end
