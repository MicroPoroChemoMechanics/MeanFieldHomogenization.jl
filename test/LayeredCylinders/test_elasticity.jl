# =============================================================================
#  test_elasticity.jl — elasticity of the layered cylinder with perfect
#  interfaces. References: the homogeneous cylinder of the package (Hill
#  tensor), the composite-cylinder closed forms of Hashin and Rosen (1964),
#  Hill's universal relations between the axisymmetric moduli of a two-phase
#  fibrous composite, exact identities (neutrality, partition, symmetry of the
#  contribution), and a solve of all the interface conditions as one system.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

@testset "LayeredCylinder — construction and accessors" begin
    Cf, Cm = lc_tensor(LC_F), lc_tensor(LC_M)
    c = LayeredCylinder((0.6, 1.0), (Cf, Cm))
    @test layer_count(c) == 2
    @test layer_radius(c, 1) == 0.6 && outer_radius(c) == 1.0
    @test layer_modulus(c, 2) === Cm
    @test layer_interface(c, 1) isa PerfectInterface
    @test layer_volume_fraction(c, 1) ≈ 0.36 && layer_volume_fraction(c, 2) ≈ 0.64
    @test MeanFieldHomogenization.Core.dimension(c) == 3
    @test !is_homogeneous_inclusion(c)
    @test MeanFieldHomogenization.Core.shape_trait(c) === LCyl.CylindricalLayered
    @test MeanFieldHomogenization.Core.inclusion_basis(c) isa TensND.CanonicalBasis
    @test c == LayeredCylinder((0.6, 1.0), (Cf, Cm)) && hash(c) == hash(LayeredCylinder((0.6, 1.0), (Cf, Cm)))
    @test c != LayeredCylinder((0.5, 1.0), (Cf, Cm))
    @test occursin("2 layer(s)", sprint(show, c))
    # the axis is normalized; mixed radii are promoted; the positional form rebuilds
    @test collect(LayeredCylinder((1.0,), (Cf,); axis = (0, 0, 2)).axis) ≈ [0.0, 0.0, 1.0]
    @test LayeredCylinder((1, 2.0), (Cf, Cm)).radii === (1.0, 2.0)
    @test LayeredCylinder(c.radii, c.moduli, c.interfaces, c.axis) == c
    @test_throws ArgumentError LayeredCylinder((), ())
    @test_throws ArgumentError LayeredCylinder((1.0, 0.5), (Cf, Cm))
    @test_throws ArgumentError LayeredCylinder((-1.0, 0.5), (Cf, Cm))
    @test_throws ArgumentError LayeredCylinder((1.0,), (Cf,); axis = (0.0, 0.0, 0.0))
    @test_throws DimensionMismatch LayeredCylinder((0.5, 1.0), (Cf,))
    @test_throws DimensionMismatch LayeredCylinder((0.5, 1.0), (Cf, Cm); interfaces = (PerfectInterface(),))
end

@testset "LayeredCylinder — one layer is the homogeneous cylinder" begin
    # `Cylinder` has its axis along the first vector of its basis: align it with e₃.
    R = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
    hom = MeanFieldHomogenization.Elasticity.Cylinder(1.0, 1.0, R)
    Cf = lc_tensor(LC_F)
    lay = LayeredCylinder((1.0,), (Cf,))
    for C0 in (lc_tensor(LC_0), TensISO{3}(3 * 6.0, 2 * 2.5))
        A = gradient_gradient_concentration(lay, Cf, C0)
        @test A isa TensND.TensTI{4, Float64, 6}
        @test lc_cc(A) ≈ lc_cc(gradient_gradient_concentration(hom, Cf, C0)) rtol = 1.0e-8
        @test flux_gradient_concentration(lay, Cf, C0) ≈ Cf ⊡ A
        @test MeanFieldHomogenization.property_contribution(lay, Cf, C0) ≈ (Cf - C0) ⊡ A
        @test MeanFieldHomogenization.property_contribution(lay, C0) ≈ (Cf - C0) ⊡ A
    end
end

@testset "LayeredCylinder — Hashin-Rosen and Hill's relations" begin
    # A fiber coated by the matrix, read as a homogeneous cylinder: B ∘ A⁻¹, which
    # does not depend on the medium it is embedded in.
    F, M = LC_F, LC_M
    c = 0.36
    cyl = LayeredCylinder((sqrt(c), 1.0), (lc_tensor(F), lc_tensor(M)))
    moduli(C0) = LCyl._hill_moduli(
        TensND.best_fit_ti(
            flux_gradient_concentration(cyl, C0, C0) ⊡ inv(gradient_gradient_concentration(cyl, C0, C0)),
            (0.0, 0.0, 1.0)
        ), (0.0, 0.0, 1.0)
    )
    L = moduli(lc_tensor(LC_0))
    k_HR = M.k + c / (1 / (F.k - M.k) + (1 - c) / (M.k + M.m))
    p_HR = M.p * (F.p * (1 + c) + M.p * (1 - c)) / (F.p * (1 - c) + M.p * (1 + c))
    @test L.k ≈ k_HR
    @test L.p ≈ p_HR
    # Hill (1964): (k* − k_f)/(l* − l_f) = (k* − k_m)/(l* − l_m) = (k_f − k_m)/(l_f − l_m),
    # and the same ratio for (l* − l̄)/(n* − n̄), the bars being volume averages.
    ρ = (F.k - M.k) / (F.l - M.l)
    @test (L.k - F.k) / (L.l - F.l) ≈ ρ
    @test (L.k - M.k) / (L.l - M.l) ≈ ρ
    @test (L.l - (c * F.l + (1 - c) * M.l)) / (L.n - (c * F.n + (1 - c) * M.n)) ≈ ρ
    # k, l, n and p do not depend on the surrounding medium. The transverse shear
    # modulus does: under that loading the coated fiber is not equivalent to a
    # homogeneous one (the composite-cylinder model only bounds it).
    L′ = moduli(TensISO{3}(3 * 6.0, 2 * 2.5))
    @test all(getfield(L′, s) ≈ getfield(L, s) for s in (:k, :l, :n, :p))
    @test !(L′.m ≈ L.m)
end

@testset "LayeredCylinder — identities" begin
    C0 = lc_tensor(LC_0)
    # a stack of the reference itself is neutral
    @test maximum(abs, lc_data(MeanFieldHomogenization.property_contribution(LayeredCylinder((0.4, 0.7, 1.0), (C0, C0, C0)), C0))) < 1.0e-12
    cyl = LayeredCylinder((0.4, 0.7, 1.0), (lc_tensor(LC_F), lc_tensor(LC_M), lc_ti(8.0, 3.0, 2.5, 20.0, 4.0)))
    A, B = gradient_gradient_concentration(cyl, C0, C0), flux_gradient_concentration(cyl, C0, C0)
    N = MeanFieldHomogenization.property_contribution(cyl, C0)
    @test N ≈ B - C0 ⊡ A
    d = lc_data(N)
    @test d[3] ≈ d[4] rtol = 1.0e-12                     # major symmetry
    f = ntuple(k -> layer_volume_fraction(cyl, k), 3)
    @test sum(f[k] * gradient_gradient_concentration(cyl, C0; layer = k) for k in 1:3) ≈ A
    @test sum(f[k] * flux_gradient_concentration(cyl, C0; layer = k) for k in 1:3) ≈ B
    @test all(flux_gradient_concentration(cyl, C0; layer = k) ≈ layer_modulus(cyl, k) ⊡ gradient_gradient_concentration(cyl, C0; layer = k) for k in 1:3)
    A2, N2 = MeanFieldHomogenization.Core.concentration_and_contribution(cyl, C0, C0)
    A3, B3 = MeanFieldHomogenization.Core.concentration_and_flux_average(cyl, C0, C0)
    @test A2 ≈ A && N2 ≈ N && A3 ≈ A && B3 ≈ B
    @test_throws BoundsError gradient_gradient_concentration(cyl, C0; layer = 4)
    @test_throws BoundsError flux_gradient_concentration(cyl, C0; layer = 0)
    # an isotropic layer, a reference given in a rotated basis
    ciso = LayeredCylinder((0.5, 1.0), (TensISO{3}(30.0, 8.0), lc_tensor(LC_M)))
    hom = MeanFieldHomogenization.Elasticity.Cylinder(1.0, 1.0, [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0])
    Crot = C0 ⊡ gradient_gradient_concentration(hom, C0, C0)      # C₀ itself, in the basis of `hom`
    @test !(Crot isa TensND.TensTI)
    @test MeanFieldHomogenization.property_contribution(ciso, Crot) ≈ MeanFieldHomogenization.property_contribution(ciso, C0)
end

@testset "LayeredCylinder — transverse shear: the interface conditions as one system" begin
    # Unknown amplitudes (a, b) of the core, (a, b, c, d) of the shell, (c, d) of
    # the matrix for (a, b) = (1, 0): the 8 continuity conditions written on the
    # fields and solved together, without the recurrence.
    F, M, Z = LC_F, LC_M, LC_0
    r₁, r₂ = 0.6, 1.0
    col(L, r, j) = collect(LCyl._shear_state(r, L.k, L.m, ntuple(i -> Float64(i == j), 4)...))
    S = zeros(8, 8)
    rhs = zeros(8)
    S[1:4, 1:2] = hcat(col(F, r₁, 1), col(F, r₁, 2))
    S[1:4, 3:6] = -hcat((col(M, r₁, j) for j in 1:4)...)
    S[5:8, 3:6] = hcat((col(M, r₂, j) for j in 1:4)...)
    S[5:8, 7:8] = -hcat(col(Z, r₂, 3), col(Z, r₂, 4))
    rhs[5:8] = col(Z, r₂, 1)
    x = S \ rhs
    cyl = LayeredCylinder((r₁, r₂), (lc_tensor(F), lc_tensor(M)))
    sol = LCyl._elastic_solution(cyl, lc_tensor(Z))
    @test collect(sol.shear.amps[1])[1:2] ≈ x[1:2]
    @test collect(sol.shear.amps[2]) ≈ x[3:6]
    @test collect(LCyl._matrix_amplitudes(sol).shear) ≈ [1.0, 0.0, x[7], x[8]] atol = 1.0e-12
end

@testset "LayeredCylinder — refused inputs" begin
    Cf = lc_tensor(LC_F)
    cyl = LayeredCylinder((1.0,), (Cf,))
    # a reference of another axis, of another symmetry
    @test_throws ArgumentError gradient_gradient_concentration(cyl, Cf, lc_tensor(LC_0; axis = (1.0, 0.0, 0.0)))
    Cortho = TensND.TensOrtho(10.0, 12.0, 14.0, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0, TensND.CanonicalBasis{3, Float64}())
    @test_throws ArgumentError gradient_gradient_concentration(cyl, Cf, Cortho)
    # an incompressible layer, a void one
    @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((1.0,), (TensISO{3}(Inf, 2.0),)), Cf, Cf)
    @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((1.0,), (TensISO{3}(0.0, 0.0),)), Cf, Cf)
    # an interface of the other physics
    @test_throws ArgumentError gradient_gradient_concentration(LayeredCylinder((1.0,), (Cf,); interfaces = (KapitzaInterface(0.1),)), Cf, Cf)
end
