using Test
using MeanFieldHomogenization
using ForwardDiff
import Symbolics

@testset "Core — newton_potential_3d_cylinder" begin
    # Circular base (b = c): I_b = I_c = 2π, I_bb = I_cc = I_bc = π/b²
    b = 1.5
    Iv, IIv = MeanFieldHomogenization.Core.newton_potential_3d_cylinder(b, b)
    @test Iv[1] == 0.0
    @test Iv[2] ≈ 2π atol = 1.0e-12
    @test Iv[3] ≈ 2π atol = 1.0e-12
    @test sum(Iv) ≈ 4π atol = 1.0e-12
    @test IIv[1] == 0.0
    @test IIv[5] == 0.0
    @test IIv[6] == 0.0
    @test IIv[2] ≈ π / b^2 atol = 1.0e-12
    @test IIv[3] ≈ π / b^2 atol = 1.0e-12
    @test IIv[4] ≈ π / b^2 atol = 1.0e-12

    # Elliptic base (b > c): I_b = 4π c/(b+c), I_c = 4π b/(b+c), I_bc = 4π/(b+c)²
    b2, c2 = 2.0, 1.0
    Iv, IIv = MeanFieldHomogenization.Core.newton_potential_3d_cylinder(b2, c2)
    s = b2 + c2
    @test Iv[1] == 0.0
    @test Iv[2] ≈ 4π * c2 / s atol = 1.0e-12
    @test Iv[3] ≈ 4π * b2 / s atol = 1.0e-12
    @test sum(Iv) ≈ 4π atol = 1.0e-12
    @test IIv[1] == 0.0
    @test IIv[4] ≈ 4π / s^2 atol = 1.0e-12
    @test IIv[2] ≈ 4π / 3 * (1 / b2^2 - 1 / s^2) atol = 1.0e-12
    @test IIv[3] ≈ 4π / 3 * (1 / c2^2 - 1 / s^2) atol = 1.0e-12

    # Numerical consistency with the general 3D Newton potential in the limit
    # a → ∞ — use a = 1e6·max(b,c) as a proxy (larger a triggers numerical
    # overflow in the elliptic-integral formula for a triaxial ellipsoid).
    Iv_inf, _ = MeanFieldHomogenization.Core.newton_potential_3d(1.0e6, b2, c2)
    @test Iv_inf[1] ≈ 0.0 atol = 1.0e-3
    @test Iv_inf[2] ≈ 4π * c2 / s rtol = 1.0e-4
    @test Iv_inf[3] ≈ 4π * b2 / s rtol = 1.0e-4

    # BigFloat — high-precision evaluation
    bb = big"1.5"
    cc = big"0.7"
    Iv_big, _ = MeanFieldHomogenization.Core.newton_potential_3d_cylinder(bb, cc)
    @test sum(Iv_big) ≈ 4 * big(π) atol = big"1e-50"

    # ForwardDiff — derivative w.r.t. transverse semi-axis
    f(x) = MeanFieldHomogenization.Core.newton_potential_3d_cylinder(x, 1.0)[1][2]  # I_b
    df = ForwardDiff.derivative(f, 2.0)
    @test isfinite(df)
    @test !isnan(df)
end

@testset "Core — cylinder and ellipse potentials hold down to a circular section" begin
    np_cyl = MeanFieldHomogenization.Core.newton_potential_3d_cylinder
    np_3d = MeanFieldHomogenization.Core.newton_potential_3d
    np_2d = MeanFieldHomogenization.Core.newton_potential_2d
    # The elliptic forms are exact down to b = c and must be used there, in value
    # and in derivative: the circular ones were taken within 1e-6 (relative), at
    # a cost of 2.5e-8 in value and of the whole derivative along c. Reference:
    # the ellipsoid of semi-axes (1e7, 1, c), whose potentials tend to the
    # cylinder's as log(a)/a² — near 1e-13 here.
    I_bc(f, x) = collect(f(x)[1][2:3])
    II_bc(f, x) = collect(f(x)[2][2:4])
    cyl(x) = np_cyl(1.0, x)
    ell(x) = np_3d(1.0e7, 1.0, x)
    for c in (1 - 1.0e-7, 1.0)
        @test I_bc(cyl, c) ≈ I_bc(ell, c) rtol = 1.0e-12
        @test II_bc(cyl, c) ≈ II_bc(ell, c) rtol = 1.0e-12
        @test ForwardDiff.derivative(x -> I_bc(cyl, x), c) ≈ ForwardDiff.derivative(x -> I_bc(ell, x), c) rtol = 1.0e-11
        @test ForwardDiff.derivative(x -> II_bc(cyl, x), c) ≈ ForwardDiff.derivative(x -> II_bc(ell, x), c) rtol = 1.0e-11
        # The cross-section, an ellipse, carries half the transverse potentials.
        @test collect(np_2d(1.0, c)) ≈ I_bc(cyl, c) ./ 2 rtol = 1.0e-15
        @test ForwardDiff.derivative(x -> collect(np_2d(1.0, x)), c) ≈
            ForwardDiff.derivative(x -> I_bc(cyl, x), c) ./ 2 rtol = 1.0e-15
    end
    # One semi-axis: equal numbers with equal derivatives, or the same expression.
    one_axis = MeanFieldHomogenization.Core._one_semi_axis
    d = ForwardDiff.Dual(1.0, 1.0)
    @test one_axis(d, d)
    @test !one_axis(d, ForwardDiff.Dual(1.0, 0.0))
    @test !one_axis(1.0, prevfloat(1.0))
    Symbolics.@variables s
    @test one_axis(s, s)
    @test !one_axis(s, 2s)
end
