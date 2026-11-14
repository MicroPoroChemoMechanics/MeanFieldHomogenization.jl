# =============================================================================
#  test_layered_cylinder_alv.jl — the layered cylinder in aging linear
#  viscoelasticity, orders 4 and 2.
#
#  Volterra matrices commute on a uniform grid with non-aging laws, where a
#  misplaced factor is invisible: the grid here is non-uniform and the laws
#  age. References, none of which composes the operators as the recurrences do:
#  the elastic limit; the equilibrium and the constitutive law of the
#  operator-valued modes, checked by differentiating them in the radius; the
#  thin elastic layer an interface is the limit of (an interface does not
#  age), between aging layers; the partition of the cylinder into its layers;
#  the quadrature of the pointwise operator over a layer; the continuity of
#  the traction across a spring.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "..", "LayeredCylinders", "helpers.jl"))

const LCV = MeanFieldHomogenization.Viscoelasticity
const LCV_TIMES = [0.0, 0.3, 0.7, 1.2, 2.0]
const LCV_N = length(LCV_TIMES)
const LCV_CF, LCV_CM, LCV_C0 = lc_tensor(LC_F), lc_tensor(LC_M), lc_tensor(LC_0)
const LCV_ITFS = (SpringInterface(; sn = 0.05, st = 0.08), MembraneInterface(3.0, 1.2))
lcv_law(C, g) = ViscoLaw((t, tp) -> t >= tp ? g(t, tp) * C : 0.0 * C)
lcv_gf(t, tp) = (1 + 0.5tp) * exp(-(t - tp) / 0.8)       # aging
lcv_gm(t, tp) = exp(-(t - tp) / 1.7)
lcv_g0(t, tp) = (1 + 0.2tp) * exp(-(t - tp) / 0.5)       # aging
lcv_mandel(t) = TensND.mandel(TensND.canonical_components(t))

@testset "LayeredCylinder ALV — elastic limit of the whole cylinder" begin
    for itfs in (
                (PerfectInterface(), PerfectInterface()), (SpringInterface(; sn = 0.05, st = 0.08), MembraneInterface(3.0, 1.2)),
                (MembraneInterface(3.0, 1.2), SpringInterface(; sn = 0.05, st = 0.08)),
            ), external in (true, false)
        c = LayeredCylinder((0.6, 1.0), (LCV_CF, LCV_CM); interfaces = itfs)
        sol = LCyl._elastic_solution(c, LCV_C0)
        A = LCV.ti_params_from_blocks(gradient_gradient_concentration_alv(c, heaviside_law(LCV_C0), LCV_TIMES; external))
        Nn = LCV.ti_params_from_blocks(property_contribution_alv(c, heaviside_law(LCV_C0), LCV_TIMES; external))
        Ae = LCyl._whole_strain(sol, external); Ne = LCyl._contribution(sol, external)
        for key in keys(A)
            @test A[key] ≈ Ae[key] * I(LCV_N) atol = 1.0e-12
            @test Nn[key] ≈ Ne[key] * I(LCV_N) atol = 1.0e-10
        end
    end
    K(t, a) = lc_K(t, a)
    for itfs in ((KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)), (SurfaceConductiveInterface(0.3), KapitzaInterface(0.1))), external in (true, false)
        c = LayeredCylinder((0.6, 1.0), (K(5.0, 7.0), K(1.0, 2.0)); interfaces = itfs)
        K0 = K(2.0, 3.0)
        A = LCV.ti_order2_params_from_blocks(gradient_gradient_concentration_alv(c, heaviside_law(K0), LCV_TIMES; external))
        Nn = LCV.ti_order2_params_from_blocks(property_contribution_alv(c, heaviside_law(K0), LCV_TIMES; external))
        a = TensND.get_data(gradient_gradient_concentration(c, K0, K0; external)); b = TensND.get_data(MeanFieldHomogenization.property_contribution(c, K0; external))
        @test A.t ≈ a[1] * I(LCV_N) atol = 1.0e-13
        @test A.a ≈ a[2] * I(LCV_N) atol = 1.0e-13
        @test Nn.t ≈ b[1] * I(LCV_N) atol = 1.0e-12
        @test Nn.a ≈ b[2] * I(LCV_N) atol = 1.0e-12
    end
end
@testset "LayeredCylinder ALV — the operator-valued modes are in equilibrium" begin
    rng = Xoshiro(3)
    op(x) = x * Matrix(1.0I, 3, 3) + LowerTriangular(0.3x * randn(rng, 3, 3)) - Diagonal(zeros(3))
    k = Matrix(op(5.0)); m = Matrix(op(2.0))
    @test norm(k * m - m * k) > 0.1
    amp = ntuple(_ -> randn(rng, 3, 3), 4)
    st(r) = LCyl._shear_state(r, k, m, amp...)
    r0 = 0.7
    U, Vv, σrr, σrθ = st(r0)
    d(i) = ForwardDiff.derivative(r -> st(r)[i], r0)
    dU, dV, dσrr, dσrθ = d(1), d(2), d(3), d(4)
    err = dU; ett = (U + 2Vv) / r0; g = (-2U + r0 * dV - Vv) / r0
    @test σrr ≈ (k + m) * err + (k - m) * ett
    @test σrθ ≈ m * g
    σθθ = (k - m) * err + (k + m) * ett
    @test norm(dσrr + 2σrθ / r0 + (σrr - σθθ) / r0) < 1.0e-10 * norm(σrr)
    @test norm(dσrθ - 2σθθ / r0 + 2σrθ / r0) < 1.0e-10 * norm(σrr)
    back = LCyl._shear_amplitudes(r0, k, m, U, Vv, σrr, σrθ)
    @test all(back[i] ≈ amp[i] for i in 1:4)
end
@testset "LayeredCylinder ALV — aging layers, interfaces as thin layers" begin
    Lf = lcv_law(LCV_CF, lcv_gf); Lm = lcv_law(LCV_CM, lcv_gm); L0 = lcv_law(LCV_C0, lcv_g0)
    c = LayeredCylinder((0.6, 1.0), (Lf, Lm); interfaces = (SpringInterface(; sn = 0.05, st = 0.08), MembraneInterface(3.0, 1.2)))
    A, B = LCV._concentration_and_flux_alv(Val(4), c, L0, LCV_TIMES)
    R0 = LCV._trapezoidal_relaxation(L0, LCV_TIMES, 6)
    @test property_contribution_alv(c, L0, LCV_TIMES) ≈ B - R0 * A
    @test norm(R0 * A - A * R0) > 1.0e-3 * norm(R0 * A)
    h = 1.0e-6; R1 = 0.6
    iso(λ, μ) = TensND.TensISO{3}(3 * (λ + 2μ / 3), 2μ)
    κs, μs = 3.0, 1.2
    μ = μs / h; λs = (κs - μs) / h; λ = 2μ * λs / (2μ - λs)
    ref = LayeredCylinder((R1, 1.0), (Lf, Lm); interfaces = (MembraneInterface(κs, μs), PerfectInterface{Float64}()))
    thin = LayeredCylinder((R1 - h / 2, R1 + h / 2, 1.0), (Lf, iso(λ, μ), Lm))
    for f in (gradient_gradient_concentration_alv, property_contribution_alv)
        a = f(ref, L0, LCV_TIMES); b = f(thin, L0, LCV_TIMES); @test a ≈ b rtol = 1.0e-4
    end
    sn, st = 0.05, 0.08; m_ = h / st; kpm = h / sn
    ref = LayeredCylinder((R1, 1.0), (Lf, Lm); interfaces = (SpringInterface(; sn, st), PerfectInterface{Float64}()))
    thin = LayeredCylinder((R1 - h / 2, R1 + h / 2, 1.0), (Lf, lc_ti(kpm - m_, 0.3h, m_, h, m_), Lm))
    for f in (gradient_gradient_concentration_alv, property_contribution_alv)
        a = f(ref, L0, LCV_TIMES); b = f(thin, L0, LCV_TIMES); @test a ≈ b rtol = 1.0e-4
    end
end

@testset "LayeredCylinder ALV — elastic limit of the layers and of the points" begin
    c = LayeredCylinder((0.6, 1.0), (LCV_CF, LCV_CM); interfaces = LCV_ITFS)
    h0 = heaviside_law(LCV_C0)
    for k in 1:2, (ext, int) in ((false, false), (true, false), (true, true))
        A = gradient_gradient_concentration_alv(c, h0, LCV_TIMES; layer = k, external = ext, internal = int)
        @test A ≈ kron(I(LCV_N), lcv_mandel(gradient_gradient_concentration(c, LCV_C0; layer = k, external = ext, internal = int))) atol = 1.0e-12
    end
    for k in 1:2
        @test LCV.flux_gradient_concentration_alv(c, h0, LCV_TIMES; layer = k) ≈ kron(I(LCV_N), lcv_mandel(flux_gradient_concentration(c, LCV_C0; layer = k))) atol = 1.0e-10
    end
    @test LCV.flux_gradient_concentration_alv(c, h0, LCV_TIMES) ≈ kron(I(LCV_N), lcv_mandel(flux_gradient_concentration(c, LCV_CM, LCV_C0))) atol = 1.0e-10
    f = LCV.LayeredCylinderALVFields(c, h0, LCV_TIMES); fe = LayeredCylinderFields(c, LCV_C0)
    for (x, kw) in (([0.2, 0.1, 0.3], (;)), ([0.5, -0.6, 0.0], (;)), ([1.3, 0.9, -0.2], (;)))
        @test LCV.local_gradient_gradient_concentration_alv(f, x; kw...) ≈ kron(I(LCV_N), lcv_mandel(local_gradient_gradient_concentration(fe, x; kw...))) atol = 1.0e-12
        @test LCV.local_flux_gradient_concentration_alv(f, x; kw...) ≈ kron(I(LCV_N), lcv_mandel(local_flux_gradient_concentration(fe, x; kw...))) atol = 1.0e-10
    end
    for side in (:inner, :outer)
        @test LCV.local_gradient_gradient_concentration_alv(f, 0.6, 0.4, 0.1; side) ≈ kron(I(LCV_N), lcv_mandel(local_gradient_gradient_concentration(fe, 0.6, 0.4, 0.1; side))) atol = 1.0e-12
    end
    K(t, a) = lc_K(t, a)
    c2 = LayeredCylinder((0.6, 1.0), (K(5.0, 7.0), K(1.0, 2.0)); interfaces = (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3))); K0 = K(2.0, 3.0)
    cK(t) = Matrix(TensND.canonical_components(t))
    for k in 1:2
        @test gradient_gradient_concentration_alv(c2, heaviside_law(K0), LCV_TIMES; layer = k, external = true) ≈ kron(I(LCV_N), cK(gradient_gradient_concentration(c2, K0; layer = k, external = true))) atol = 1.0e-12
        @test LCV.flux_gradient_concentration_alv(c2, heaviside_law(K0), LCV_TIMES; layer = k) ≈ kron(I(LCV_N), cK(flux_gradient_concentration(c2, K0; layer = k))) atol = 1.0e-12
    end
    f2 = LCV.LayeredCylinderALVFields(c2, heaviside_law(K0), LCV_TIMES); fe2 = LayeredCylinderTransportFields(c2, K0)
    for x in ([0.2, 0.1, 0.3], [0.5, -0.6, 0.0], [1.3, 0.9, -0.2])
        @test LCV.local_gradient_gradient_concentration_alv(f2, x) ≈ kron(I(LCV_N), cK(local_gradient_gradient_concentration(fe2, x))) atol = 1.0e-12
        @test LCV.local_flux_gradient_concentration_alv(f2, x) ≈ kron(I(LCV_N), cK(local_flux_gradient_concentration(fe2, x))) atol = 1.0e-12
    end
    @test_throws BoundsError gradient_gradient_concentration_alv(c, h0, LCV_TIMES; layer = 3)
    @test_throws BoundsError LCV.flux_gradient_concentration_alv(c2, heaviside_law(K0), LCV_TIMES; layer = 0)
end
@testset "LayeredCylinder ALV — aging: partition, points and layers, traction" begin
    Lf = lcv_law(LCV_CF, lcv_gf); Lm = lcv_law(LCV_CM, lcv_gm); L0 = lcv_law(LCV_C0, lcv_g0)
    c = LayeredCylinder((0.6, 1.0), (Lf, Lm); interfaces = LCV_ITFS)
    fr = (0.36, 0.64)
    A = gradient_gradient_concentration_alv(c, L0, LCV_TIMES)
    @test sum(fr[k] * gradient_gradient_concentration_alv(c, L0, LCV_TIMES; layer = k, external = true) for k in 1:2) ≈ A
    R(L) = LCV._trapezoidal_relaxation(L, LCV_TIMES, 6)
    for k in 1:2
        @test LCV.flux_gradient_concentration_alv(c, L0, LCV_TIMES; layer = k) ≈ R((Lf, Lm)[k]) * gradient_gradient_concentration_alv(c, L0, LCV_TIMES; layer = k)
    end
    f = LCV.LayeredCylinderALVFields(c, L0, LCV_TIMES)
    for lay in 1:2
        ra = lay == 1 ? 0.0 : 0.6; rb = lay == 1 ? 0.6 : 1.0; nr, nθ = 200, 32
        acc = zeros(6LCV_N, 6LCV_N); area = 0.0
        for i in 1:nr, j in 1:nθ
            r = ra + (i - 0.5) * (rb - ra) / nr; θ = 2π * (j - 0.5) / nθ
            acc .+= r .* LCV.local_gradient_gradient_concentration_alv(f, r, θ, 0.0; layer = lay); area += r
        end
        @test acc ./ area ≈ gradient_gradient_concentration_alv(c, L0, LCV_TIMES; layer = lay) rtol = 1.0e-4
    end
    # traction continuity across the spring at r = 0.6 (e_r = e_1 at θ = 0): rows 1 (σ11), 5 (σ13), 6 (σ12)
    Si = LCV.local_flux_gradient_concentration_alv(f, 0.6, 0.0, 0.0; side = :inner); So = LCV.local_flux_gradient_concentration_alv(f, 0.6, 0.0, 0.0; side = :outer)
    rows = vcat([6(i - 1) .+ [1, 5, 6] for i in 1:LCV_N]...)
    @test Si[rows, :] ≈ So[rows, :]
    @test !(Si ≈ So)
    # far field
    @test LCV.local_gradient_gradient_concentration_alv(f, [1.0e4, 0.0, 0.0]) ≈ I(6LCV_N) rtol = 1.0e-6
end

@testset "LayeredCylinder ALV — refused inputs, block conversions" begin
    h0 = heaviside_law(LCV_C0)
    tilted = LayeredCylinder((1.0,), (lc_tensor(LC_F; axis = (1.0, 0.0, 0.0)),); axis = (1.0, 0.0, 0.0))
    @test_throws "its axis must be e₃" gradient_gradient_concentration_alv(tilted, h0, LCV_TIMES)
    # a layer, or the reference, transversely isotropic about another axis
    @test_throws "layer 1" gradient_gradient_concentration_alv(LayeredCylinder((1.0,), (lc_tensor(LC_F; axis = (1.0, 0.0, 0.0)),)), h0, LCV_TIMES)
    @test_throws "the reference medium" gradient_gradient_concentration_alv(
        LayeredCylinder((1.0,), (LCV_CF,)), heaviside_law(lc_tensor(LC_0; axis = (1.0, 0.0, 0.0))), LCV_TIMES
    )
    @test_throws "infinite modulus" gradient_gradient_concentration_alv(LayeredCylinder((1.0,), (TensISO{3}(Inf, 2.0),)), h0, LCV_TIMES)
    @test_throws ArgumentError LCV.ti_params_from_blocks(zeros(5, 5))
    @test_throws ArgumentError LCV.ti_order2_params_from_blocks(zeros(4, 4))
    # the blocks of a stiffness: (n, 2l, l, 2k, 2m, 2p), and back
    c = LCV.ti_params_from_blocks(lcv_mandel(LCV_CF))
    @test [only(c[s]) for s in (:zz, :ze, :mz, :me, :T, :L)] ≈ [150.0, 24.0, 12.0, 60.0, 18.0, 28.0]
    @test LCV.ti_blocks_from_params(c) ≈ lcv_mandel(LCV_CF)
    K = Matrix(lc_cc(lc_K(5.0, 7.0)))
    @test LCV.ti_order2_blocks_from_params(LCV.ti_order2_params_from_blocks(K)) ≈ K
    # a running reference given as a block matrix
    R0 = LCV._trapezoidal_relaxation(lcv_law(LCV_C0, lcv_g0), LCV_TIMES, 6)
    cyl = LayeredCylinder((0.6, 1.0), (lcv_law(LCV_CF, lcv_gf), LCV_CM); interfaces = LCV_ITFS)
    @test property_contribution_alv(cyl, R0, LCV_TIMES) ≈ property_contribution_alv(cyl, lcv_law(LCV_C0, lcv_g0), LCV_TIMES)
    # a dual in one layer only
    d = ForwardDiff.derivative(0.0) do x
        c1 = LayeredCylinder((0.6, 1.0), (lcv_law(lc_ti(30.0 + x, 12.0, 9.0, 150.0, 14.0), lcv_gf), LCV_CM); interfaces = LCV_ITFS)
        property_contribution_alv(c1, R0, LCV_TIMES)
    end
    fd(x) = property_contribution_alv(LayeredCylinder((0.6, 1.0), (lcv_law(lc_ti(30.0 + x, 12.0, 9.0, 150.0, 14.0), lcv_gf), LCV_CM); interfaces = LCV_ITFS), R0, LCV_TIMES)
    @test d ≈ (fd(1.0e-5) - fd(-1.0e-5)) / 2.0e-5 rtol = 1.0e-6
end
