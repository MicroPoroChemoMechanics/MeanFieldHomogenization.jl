# =============================================================================
#  test_local_fields.jl — pointwise fields of the layered cylinder, on a
#  cylinder of arbitrary axis with a spring and a membrane. References, none of
#  which uses the recurrences: the strain is the symmetric gradient of the
#  displacement and the stress is divergence-free (automatic differentiation of
#  the fields), the remote fields are the imposed ones, the interface laws hold
#  between the two sides of each interface (the traction jump of the membrane
#  against the surface divergence of its Gurtin-Murdoch stress, itself
#  differentiated), and the quadrature of the strain over a layer is its
#  average.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

@testset "LayeredCylinder — pointwise fields" begin
    rng = Xoshiro(7)
    # random unit axis
    ax = let v = randn(rng, 3)
        Tuple(v ./ norm(v))
    end
    Cf = lc_ti(30.0, 12.0, 9.0, 150.0, 14.0; axis = ax); Cm = lc_ti(4.0, 2.0, 1.5, 7.0, 1.5; axis = ax); C0 = lc_ti(5.0, 1.0, 2.0, 9.0, 3.0; axis = ax)
    itfs = (SpringInterface(; sn = 0.05, st = 0.08), MembraneInterface(3.0, 1.2))
    cyl = LayeredCylinder((0.6, 1.0), (Cf, Cm); interfaces = itfs, axis = ax)
    f = LayeredCylinderFields(cyl, C0)
    E0 = let M = randn(rng, 3, 3)
        (M + M') / 2
    end
    E = MeanFieldHomogenization.Core._tens_in(E0, TensND.CanonicalBasis{3, Float64}())
    n = collect(ax); t1 = collect(LCyl._transverse_direction(ax)); t2 = cross(n, t1)
    pt(r, θ, z) = r * (cos(θ) * t1 + sin(θ) * t2) + z * n
    @testset "compatibility, equilibrium, far field" begin
        for (r, lay) in ((0.3, 1), (0.8, 2), (1.7, 3))
            x = pt(r, 0.7, 0.4)
            u(y) = collect(local_displacement(f, y, E; layer = lay))
            G = ForwardDiff.jacobian(u, x)
            ε = lc_cc(local_gradient(f, x, E))
            @test (G + G') / 2 ≈ ε atol = 1.0e-10
            σ(y) = vec(lc_cc(local_flux(f, y, E; layer = lay)))
            Jσ = reshape(ForwardDiff.jacobian(σ, x), 3, 3, 3)
            div = [sum(Jσ[i, j, j] for j in 1:3) for i in 1:3]
            @test norm(div) < 1.0e-9 * norm(lc_cc(local_flux(f, x, E)))
            @test lc_cc(local_flux(f, x, E)) ≈ lc_cc(region_stiffness(f, lay) ⊡ local_gradient(f, x, E))
        end
        x = pt(1.0e4, 1.1, 0.0)
        @test lc_cc(local_gradient(f, x, E)) ≈ E0 rtol = 1.0e-6
        @test collect(local_displacement(f, x, E)) ≈ E0 * x rtol = 1.0e-6
        # on the axis
        @test all(isfinite, lc_cc(local_gradient(f, 0.2 .* n, E)))
        @test lc_cc(local_gradient(f, 0.2 .* n, E)) ≈ lc_cc(local_gradient(f, pt(1.0e-9, 0.3, 0.2), E)) rtol = 1.0e-6
    end
    @testset "interface laws" begin
        θ = 0.9; z = 0.3
        er = cos(θ) * t1 + sin(θ) * t2; eθ = cross(n, er)
        # spring at r = 0.6
        ui = collect(local_displacement(f, 0.6, θ, z, E; side = :inner)); uo = collect(local_displacement(f, 0.6, θ, z, E; side = :outer))
        σi = lc_cc(local_flux(f, 0.6, θ, z, E; side = :inner)); σo = lc_cc(local_flux(f, 0.6, θ, z, E; side = :outer))
        t = σi * er
        @test σo * er ≈ t
        @test dot(uo - ui, er) ≈ 0.05 * dot(t, er)
        @test dot(uo - ui, eθ) ≈ 0.08 * dot(t, eθ)
        @test dot(uo - ui, n) ≈ 0.08 * dot(t, n)
        # membrane at r = 1: displacement continuous, traction jump = -div_s σ_s
        ui = collect(local_displacement(f, 1.0, θ, z, E; side = :inner)); uo = collect(local_displacement(f, 1.0, θ, z, E; side = :outer))
        @test ui ≈ uo
        @test collect(local_displacement(f, pt(0.8, θ, z), E)) ≈ collect(local_displacement(f, 0.8, θ, z, E))
        κs, μs = 3.0, 1.2; λs = κs - μs
        # surface strain from the displacement on the cylinder r = 1, by AD in (θ, z)
        us(q) = collect(local_displacement(f, pt(1.0, q[1], q[2]), E; layer = 2))
        function σs(q)
            J = ForwardDiff.jacobian(us, q)       # ∂u/∂θ, ∂u/∂z
            erq = cos(q[1]) * t1 + sin(q[1]) * t2; eθq = cross(n, erq)
            εθθ = dot(J[:, 1], eθq) / 1.0; εzz = dot(J[:, 2], n); εθz = (dot(J[:, 1], n) / 1.0 + dot(J[:, 2], eθq)) / 2
            trs = εθθ + εzz
            return [λs * trs + 2μs * εθθ, λs * trs + 2μs * εzz, 2μs * εθz]   # σθθ, σzz, σθz
        end
        q = [θ, z]; s = σs(q); Js = ForwardDiff.jacobian(σs, q)
        # div_s σ_s on a cylinder of radius R = 1: radial −σθθ/R ; θ: ∂θσθθ/R + ∂zσθz ; z: ∂θσθz/R + ∂zσzz
        divs = (-s[1]) * er + (Js[1, 1] + Js[3, 2]) * eθ + (Js[3, 1] + Js[2, 2]) * n
        σi = lc_cc(local_flux(f, 1.0, θ, z, E; side = :inner)); σo = lc_cc(local_flux(f, 1.0, θ, z, E; side = :outer))
        @test (σo - σi) * er ≈ -divs atol = 1.0e-8
    end
    @testset "layer averages by quadrature" begin
        for lay in 1:2
            ra = lay == 1 ? 0.0 : 0.6; rb = lay == 1 ? 0.6 : 1.0
            nr, nθ = 400, 64
            acc = zeros(3, 3); area = 0.0
            for i in 1:nr, j in 1:nθ
                r = ra + (i - 0.5) * (rb - ra) / nr; θ = 2π * (j - 0.5) / nθ
                w = r
                acc .+= w .* lc_cc(local_gradient(f, pt(r, θ, 0.0), E; layer = lay)); area += w
            end
            Ak = gradient_gradient_concentration(cyl, C0; layer = lay)
            @test acc ./ area ≈ lc_cc(Ak ⊡ E) rtol = 1.0e-5
        end
    end
    @testset "conduction" begin
        K(t, a) = TensND.TensTI{2}(t, a, ax)
        c = LayeredCylinder((0.6, 1.0), (K(5.0, 7.0), K(1.0, 2.0)); interfaces = (KapitzaInterface(0.2), SurfaceConductiveInterface(0.3)), axis = ax)
        K0 = K(2.0, 3.0); g = LayeredCylinderTransportFields(c, K0)
        Gv = randn(rng, 3); G = TensND.Vec{3}(Tuple(Gv))
        for (r, lay) in ((0.3, 1), (0.8, 2), (1.7, 3))
            x = pt(r, 0.7, 0.4)
            gT = ForwardDiff.gradient(y -> local_temperature(g, y, G; layer = lay), x)
            @test collect(local_gradient(g, x, G)) ≈ gT atol = 1.0e-10
            q(y) = collect(local_flux(g, y, G; layer = lay))
            @test abs(tr(ForwardDiff.jacobian(q, x))) < 1.0e-9
        end
        θ = 0.9; er = cos(θ) * t1 + sin(θ) * t2
        qi = collect(local_flux(g, 0.6, θ, 0.1, G; side = :inner)); qo = collect(local_flux(g, 0.6, θ, 0.1, G; side = :outer))
        @test dot(qi, er) ≈ dot(qo, er)
        @test local_temperature(g, 0.6, θ, 0.1, G; side = :outer) - local_temperature(g, 0.6, θ, 0.1, G; side = :inner) ≈ -0.2 * dot(qi, er)
        @test collect(local_gradient(g, pt(1.0e4, 0.2, 0.0), G)) ≈ Gv rtol = 1.0e-6
        @test local_temperature(g, 1.0, θ, 0.1, G; side = :outer) ≈ local_temperature(g, 1.0, θ, 0.1, G; side = :inner)
    end
end

@testset "LayeredCylinder — averages under a remote loading, regions" begin
    Cf = lc_ti(30.0, 12.0, 9.0, 150.0, 14.0); Cm = lc_ti(4.0, 2.0, 1.5, 7.0, 1.5); C0 = lc_ti(5.0, 1.0, 2.0, 9.0, 3.0)
    cyl = LayeredCylinder((0.6, 1.0), (Cf, Cm))
    E = MeanFieldHomogenization.Core._tens_in([0.3 0.1 -0.2; 0.1 -0.4 0.25; -0.2 0.25 0.7], TensND.CanonicalBasis{3, Float64}())
    @test lc_cc(cylinder_strain_average(cyl, C0, E)) ≈ lc_cc(gradient_gradient_concentration(cyl, Cm, C0) ⊡ E)
    @test lc_cc(cylinder_stress_average(cyl, C0, E)) ≈ lc_cc(flux_gradient_concentration(cyl, Cm, C0) ⊡ E)
    @test lc_cc(layer_stress_average(cyl, C0, E, 1)) ≈ lc_cc(Cf ⊡ layer_strain_average(cyl, C0, E, 1))
    K(t, a) = TensND.TensTI{2}(t, a, (0.0, 0.0, 1.0))
    c2 = LayeredCylinder((0.6, 1.0), (K(5.0, 7.0), K(1.0, 2.0))); K0 = K(2.0, 3.0)
    G = TensND.Vec{3}((0.3, -0.2, 0.9))
    @test collect(cylinder_gradient_average(c2, K0, G)) ≈ lc_cc(gradient_gradient_concentration(c2, K0, K0)) * collect(G)
    @test collect(cylinder_flux_average(c2, K0, G)) ≈ lc_cc(flux_gradient_concentration(c2, K0, K0)) * collect(G)
    @test_throws ArgumentError cylinder_strain_average(c2, K0, G)
    @test_throws BoundsError layer_gradient_average(cyl, C0, E, 3)
    @test get_layer(cyl, 0.6) == 2 && get_layer(cyl, 0.6; side = :inner) == 1 && get_layer(cyl, 2.0) == 3
    # a point within rounding of the axis, a symbolic radius, a wrong side or region
    f = LayeredCylinderFields(cyl, C0)
    @test lc_cc(local_gradient(f, (1.0e-18, 0.0, 0.3), E)) ≈ lc_cc(local_gradient(f, (0.0, 0.0, 0.3), E))
    @test_throws ArgumentError get_layer(cyl, 0.5; side = :middle)
    @test_throws ArgumentError local_gradient(f, (0.1, 0.0, 0.0), E; layer = 4)
    @test lc_cc(local_strain(cyl, C0, (0.2, 0.1, 0.0), E)) ≈ lc_cc(local_gradient(f, (0.2, 0.1, 0.0), E))
    @test lc_cc(local_stress(cyl, C0, (0.2, 0.1, 0.0), E)) ≈ lc_cc(local_flux(f, (0.2, 0.1, 0.0), E))
    @test collect(local_displacement(cyl, C0, (0.2, 0.1, 0.0), E)) ≈ collect(local_displacement(f, (0.2, 0.1, 0.0), E))
    S0 = inv(C0)
    x = (0.7, 0.2, 0.0)
    @test local_gradient_flux_concentration(f, x) ≈ local_gradient_gradient_concentration(f, x) ⊡ S0
    @test local_flux_flux_concentration(cyl, C0, x) ≈ region_stiffness(f, 2) ⊡ local_gradient_gradient_concentration(f, x) ⊡ S0
    @test local_stress_strain_concentration(f, x) ≈ region_stiffness(f, 2) ⊡ local_strain_strain_concentration(f, x)
    @test region_stiffness(f, 3) ≈ C0
    g = LayeredCylinderTransportFields(c2, K0)
    @test local_temperature(c2, K0, x, G) ≈ local_temperature(g, x, G)
    m(t) = collect(lc_cc(t))
    @test m(local_gradient_flux_concentration(g, x)) ≈ m(local_gradient_gradient_concentration(g, x)) / m(K0)
    @test m(local_flux_flux_concentration(c2, K0, x)) ≈ -m(lc_K(1.0, 2.0)) * m(local_gradient_gradient_concentration(g, x)) / m(K0)
    @test collect(local_flux(c2, K0, x, G)) ≈ -m(lc_K(1.0, 2.0)) * collect(local_gradient(g, x, G))
end
