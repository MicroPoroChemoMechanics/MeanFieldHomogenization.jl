# =============================================================================
#  test_radial_layers.jl — layered cylinder whose layers are transversely
#  isotropic about its radial direction (`radial_ti(CylindricalChart(), …)`).
#
#  References, independent of the solver under test:
#   1. isotropic layers written as radial fields: the amplitude recurrences of
#      the cylinder, for its four problems, with every kind of interface;
#   2. Kreher's theorem in a layer: the derivative of 𝐄:ℕ:𝐄 when the modulus of
#      a layer moves along a tensor 𝔹 that turns with the point is the integral
#      of ε:𝔹(x):ε over the layer, computed by a quadrature of the pointwise
#      field of the isotropic cylinder (`LayeredCylinderFields`), for each of
#      the five moduli and of the two conductivities, under a remote strain
#      that excites the four problems;
#   3. a homogeneous radial cylinder, which its surroundings see as a
#      homogeneous one: of transverse conductivity √(k_r k_t), of antiplane
#      shear modulus √(p m), of plane-strain bulk modulus (√(n (k + m)) + l)/2
#      (the solutions r^s of the three problems of size two);
#   4. the mean over the angle of the tensor of a layer, by quadrature;
#   5. a layer cut into two layers of the same moduli is the same cylinder.
# =============================================================================

isdefined(@__MODULE__, :LCyl) || include(joinpath(@__DIR__, "helpers.jl"))
import QuadGK

rc_chart = CylindricalChart()
rc_C₀ = iso_stiffness(10.0, 6.0)
rc_C = (iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0))
rc_K = (TensISO{3}(5.0), TensISO{3}(0.3))
rc_radii = (0.6, 1.0)
rc_hill = (n = 12.0, l = 5.0, k = 19.0, m = 11.0, p = 4.0)
rc_shell = radial_ti(rc_chart; rc_hill...)
rc_E = Tens(Symmetric([1.0 0.2 0.1; 0.2 -0.5 0.3; 0.1 0.3 0.7]))
rc_km(t) = collect(mandel(t, TensND.CanonicalBasis{3, Float64}()))
rc_radial(P) = radial_ti(rc_chart, P)
rc_unit(i, n) = ntuple(j -> float(i == j), n)
rc_interfaces = (
    (PerfectInterface(), PerfectInterface()),
    (SpringInterface(; sn = 0.02, st = 0.05), MembraneInterface(3.0, 2.0)),
)

# ∫ g(x) dA over the annulus ra < r < rb of the section, divided by the area of the unit disk.
function rc_annulus_integral(g, ra, rb; n = 12, nθ = 24)
    xs, ws = QuadGK.gauss(n)
    total = 0.0
    for (xr, wr) in zip(xs, ws), j in 1:nθ
        r = (ra + rb) / 2 + (rb - ra) / 2 * xr
        θ = 2π * (j - 0.5) / nθ
        total += wr * (2π / nθ) * (rb - ra) / 2 * r * g([r * cos(θ), r * sin(θ), 0.3])
    end
    return total / π
end

@testset "LayeredCylinder — layers transversely isotropic about the radius" begin
    @testset "isotropic layers, written as radial ones" begin
        ti_core = lc_tensor(LC_F)
        stacks = (
            (rc_C, (rc_C[1], rc_radial(rc_C[2]))),
            (rc_C, (rc_radial(rc_C[1]), rc_radial(rc_C[2]))),
            ((ti_core, rc_C[2]), (ti_core, rc_radial(rc_C[2]))),      # a core transversely isotropic about the axis
        )
        for itf in rc_interfaces, (reference, radial) in stacks, external in (true, false)
            iso = LayeredCylinder(rc_radii, reference; interfaces = itf)
            cyl = LayeredCylinder(rc_radii, radial; interfaces = itf)
            for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test rc_km(f(cyl, rc_C₀, rc_C₀; external)) ≈ rc_km(f(iso, rc_C₀, rc_C₀; external)) rtol = 1.0e-10
            end
            for layer in 1:2
                @test rc_km(gradient_gradient_concentration(cyl, rc_C₀; layer)) ≈ rc_km(gradient_gradient_concentration(iso, rc_C₀; layer)) rtol = 1.0e-10
                @test rc_km(flux_gradient_concentration(cyl, rc_C₀; layer)) ≈ rc_km(flux_gradient_concentration(iso, rc_C₀; layer)) rtol = 1.0e-10
            end
        end
        K₀ = TensISO{3}(1.0)
        for itf in ((PerfectInterface(), PerfectInterface()), (KapitzaInterface(0.1), SurfaceConductiveInterface(0.4)))
            iso = LayeredCylinder(rc_radii, rc_K; interfaces = itf)
            for radial in ((rc_K[1], rc_radial(rc_K[2])), (rc_radial(rc_K[1]), rc_radial(rc_K[2])))
                cyl = LayeredCylinder(rc_radii, radial; interfaces = itf)
                for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                    @test rc_km(f(cyl, K₀, K₀)) ≈ rc_km(f(iso, K₀, K₀)) rtol = 1.0e-11
                end
                for layer in 1:2
                    @test rc_km(flux_gradient_concentration(cyl, K₀; layer)) ≈ rc_km(flux_gradient_concentration(iso, K₀; layer)) rtol = 1.0e-11
                end
            end
        end
    end

    @testset "the derivative along a tensor that turns with the point" begin
        iso = LayeredCylinder(rc_radii, rc_C)
        fields = LayeredCylinderFields(iso, rc_C₀)
        energy(cyl) = rc_E ⊡ (property_contribution(cyl, rc_C₀, rc_C₀) ⊡ rc_E)
        for (layer, (ra, rb)) in enumerate(((0.0, 0.6), (0.6, 1.0))), i in 1:5
            𝔹 = radial_ti(rc_chart, rc_unit(i, 5)...)
            moved(t) = LayeredCylinder(rc_radii, Base.setindex(iso.moduli, rc_radial(iso.moduli[layer]) + t * 𝔹, layer))
            form(x) = (ε = local_gradient(fields, x, rc_E; layer); ε ⊡ (tensor_at(𝔹, x) ⊡ ε))
            @test ForwardDiff.derivative(t -> energy(moved(t)), 0.0) ≈ rc_annulus_integral(form, ra, rb) rtol = 1.0e-9
        end
        K₀ = TensISO{3}(1.0)
        isoK = LayeredCylinder(rc_radii, rc_K)
        fieldsK = LayeredCylinderTransportFields(isoK, K₀)
        G = TensND.Vec{3}((0.3, -0.5, 0.8))
        Gt = Tens([0.3, -0.5, 0.8])
        for (layer, (ra, rb)) in enumerate(((0.0, 0.6), (0.6, 1.0))), i in 1:2
            𝑩 = radial_ti(rc_chart, rc_unit(i, 2)...)
            moved(t) = LayeredCylinder(rc_radii, Base.setindex(isoK.moduli, rc_radial(isoK.moduli[layer]) + t * 𝑩, layer))
            form(x) = (g = Tens(collect(local_gradient(fieldsK, x, G; layer))); g ⋅ (tensor_at(𝑩, x) ⋅ g))
            d = ForwardDiff.derivative(t -> Gt ⋅ (property_contribution(moved(t), K₀, K₀) ⋅ Gt), 0.0)
            @test d ≈ rc_annulus_integral(form, ra, rb) rtol = 1.0e-9
        end
    end

    @testset "a homogeneous radial cylinder, and its derivatives" begin
        ∂(f, x) = ForwardDiff.derivative(f, x)
        # Transverse conduction: apparent conductivity √(k_r k_t); a disk of conductivity
        # k✶ in k₀ contributes 2k₀ (k✶ − k₀)/(k✶ + k₀) across the axis.
        k₀ = 1.3
        closed(kₜ, kᵣ) = (k✶ = sqrt(kᵣ * kₜ); 2k₀ * (k✶ - k₀) / (k✶ + k₀))
        computed(kₜ, kᵣ) = first(TensND.get_data(property_contribution(LayeredCylinder((1.0,), (radial_ti(rc_chart, kₜ, kᵣ),)), TensISO{3}(k₀), TensISO{3}(k₀))))
        @test computed(5.0, 2.0) ≈ closed(5.0, 2.0) rtol = 1.0e-12
        @test ∂(x -> computed(x, 2.0), 5.0) ≈ ∂(x -> closed(x, 2.0), 5.0) rtol = 1.0e-12
        @test ∂(y -> ∂(x -> computed(x, y), 5.0), 2.0) ≈ ∂(y -> ∂(x -> closed(x, y), 5.0), 2.0) rtol = 1.0e-10
        # Along the axis: the mean of the tangential conductivity.
        @test last(TensND.get_data(property_contribution(LayeredCylinder((1.0,), (radial_ti(rc_chart, 5.0, 2.0),)), TensISO{3}(k₀), TensISO{3}(k₀)))) ≈ 5.0 - k₀
        # Elasticity: neutral in the reference whose antiplane modulus is √(p m), resp.
        # whose plane-strain bulk modulus is (√(n (k + m)) + l)/2.
        (; n, l, k, m, p) = rc_hill
        ball = LayeredCylinder((1.0,), (rc_shell,))
        antiplane(P₀) = TensND.get_data(property_contribution(ball, P₀, P₀))[6]
        @test abs(antiplane(lc_ti(5.0, 1.0, 2.0, 9.0, sqrt(p * m)))) < 1.0e-12
        @test antiplane(lc_ti(5.0, 1.0, 2.0, 9.0, 0.9 * sqrt(p * m))) > 0
        dilatation(P₀) = TensND.get_data(property_contribution(ball, P₀, P₀))[2]
        @test abs(dilatation(lc_ti((sqrt(n * (k + m)) + l) / 2, 1.0, 2.0, 9.0, 3.0))) < 1.0e-11
        # The contribution is homogeneous of degree one in the moduli and of degree zero
        # in the radii, through the four problems and their interfaces.
        function scaled(t, s)
            itf = (SpringInterface(; sn = 0.02 * s / t, st = 0.05 * s / t), PerfectInterface())
            cyl = LayeredCylinder(s .* rc_radii, (t * rc_C[1], t * rc_shell); interfaces = itf)
            return rc_E ⊡ (property_contribution(cyl, t * rc_C₀, t * rc_C₀) ⊡ rc_E)
        end
        @test ∂(t -> scaled(t, 1.0), 1.0) ≈ scaled(1.0, 1.0) rtol = 1.0e-10
        @test abs(∂(s -> scaled(1.0, s), 1.0)) < 1.0e-10 * abs(scaled(1.0, 1.0))
        @test abs(∂(t -> ∂(u -> scaled(u, 1.0), t), 1.0)) < 1.0e-8 * abs(scaled(1.0, 1.0))
    end

    @testset "a layer cut in two, oscillating modes" begin
        # A positive definite modulus whose transverse shear modes have complex exponents.
        exponents(c) = eigvals(Matrix(LCyl._shear_system(LCyl._hill_moduli(c, (0.0, 0.0, 1.0)))))
        @test sort(real(exponents(rc_radial(rc_C[2])))) ≈ [-3, -1, 1, 3]
        soft = radial_ti(rc_chart; n = 0.8, l = 0.6, k = 1.3, m = 1.2, p = 3.6)
        for shell in (rc_shell, soft)
            whole = LayeredCylinder(rc_radii, (rc_C[1], shell))
            cut = LayeredCylinder((0.6, 0.8, 1.0), (rc_C[1], shell, shell))
            for f in (gradient_gradient_concentration, flux_gradient_concentration)
                @test rc_km(f(cut, rc_C₀, rc_C₀)) ≈ rc_km(f(whole, rc_C₀, rc_C₀)) rtol = 1.0e-10
            end
        end
        @test !isreal(exponents(soft))
    end

    @testset "complex moduli" begin
        z₀ = iso_stiffness(10.0 + 1.0im, 6.0 + 0.5im)
        z = (iso_stiffness(40.0 + 3.0im, 20.0 + 2.0im), iso_stiffness(4.0 + 0.8im, 3.0 + 0.6im))
        zkm(t) = collect(mandel(t, TensND.CanonicalBasis{3, ComplexF64}()))
        for itf in rc_interfaces
            iso = LayeredCylinder(rc_radii, z; interfaces = itf)
            cyl = LayeredCylinder(rc_radii, (z[1], rc_radial(z[2])); interfaces = itf)
            for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test zkm(f(cyl, z₀, z₀)) ≈ zkm(f(iso, z₀, z₀)) rtol = 1.0e-10
            end
        end
        κ₀ = TensISO{3}(1.0 + 0.1im)
        κ = (TensISO{3}(5.0 + 1.0im), TensISO{3}(0.3 + 0.05im))
        @test zkm(property_contribution(LayeredCylinder(rc_radii, (κ[1], rc_radial(κ[2]))), κ₀, κ₀)) ≈
            zkm(property_contribution(LayeredCylinder(rc_radii, κ), κ₀, κ₀)) rtol = 1.0e-11
        # Analytic in the moduli: an imaginary increment gives the derivative of the dual numbers.
        contribution(n) = rc_E ⊡ (property_contribution(LayeredCylinder(rc_radii, (rc_C[1], radial_ti(rc_chart; rc_hill..., n))), rc_C₀, rc_C₀) ⊡ rc_E)
        @test imag(contribution(12.0 + 1.0e-20im)) / 1.0e-20 ≈ ForwardDiff.derivative(contribution, 12.0) rtol = 1.0e-10
    end

    @testset "the mean of the moduli over a layer, and the schemes" begin
        cyl = LayeredCylinder((1.0,), (rc_shell,))
        mean_over_angle(F) = sum(rc_km(tensor_at(F, [cos(θ), sin(θ), 0.0])) for θ in 2π .* ((1:48) .- 0.5) ./ 48) / 48
        @test rc_km(LCyl.layer_property_average(cyl, rc_C₀)) ≈ mean_over_angle(rc_shell) rtol = 1.0e-12
        @test rc_km(LCyl.layer_inverse_property_average(cyl, rc_C₀)) ≈ mean_over_angle(inv(rc_shell)) rtol = 1.0e-12
        κ = radial_ti(rc_chart, 5.0, 2.0)
        cylK = LayeredCylinder((1.0,), (κ,))
        @test rc_km(LCyl.layer_property_average(cylK, TensISO{3}(1.0))) ≈ mean_over_angle(κ) rtol = 1.0e-12
        # In an inclined cylinder the layer turns about its axis.
        axis = Tuple(TensND.normalize([1.0, 2.0, 2.0]))
        inclined = LayeredCylinder(rc_radii, (rc_C[1], rc_shell); axis)
        straight = LayeredCylinder(rc_radii, (rc_C[1], rc_shell))
        @test TensND.get_data(property_contribution(inclined, rc_C₀, rc_C₀)) |> collect ≈ collect(TensND.get_data(property_contribution(straight, rc_C₀, rc_C₀))) rtol = 1.0e-12
        # Voigt, Mori-Tanaka and Reuss, on the energy of a strain.
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => rc_C₀); fraction = :rest)
        add_phase!(rve, :I, straight, Dict(:C => rc_C₀); fraction = 0.3)
        W = [rc_E ⊡ (homogenize(rve, s, :C) ⊡ rc_E) for s in (Reuss(), MoriTanaka(), Voigt())]
        @test issorted(W)
    end

    @testset "the parameters of a layer, and second moments in a layer" begin
        function cell(cyl)
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => rc_C₀); fraction = :rest)
            add_phase!(rve, :I, cyl, Dict(:C => rc_C₀); fraction = 0.3)
            return rve
        end
        rve = cell(LayeredCylinder(rc_radii, (rc_C[1], rc_shell)))
        energy(C) = rc_E ⊡ (C ⊡ rc_E)
        # The names of the frame (e_r, e_θ, e_z), which turns with the point.
        (; n, l, k, m, p) = rc_hill
        for (name, value) in ((:Crrrr, n), (:Cθθθθ, k + m), (:Cθθzz, k - m), (:Crrθθ, l), (:Crθrθ, p))
            lens = layer_param(:I, 2, name)
            @test get_param(rve, lens) ≈ value
            rebuilt(x) = energy(homogenize(cell(LayeredCylinder(rc_radii, (rc_C[1], set_coefficient(rc_shell, name, x)))), MoriTanaka(), :C))
            @test derivative(rve, MoriTanaka(), lens; indexer = energy) ≈ ForwardDiff.derivative(rebuilt, value) rtol = 1.0e-11
        end
        @test_throws ArgumentError get_param(rve, layer_param(:I, 2, :Cθθφφ))       # a name of the sphere
        @test get_param(set_param(rve, layer_param(:I, 2, :ℓ₆), 9.0), layer_param(:I, 2, :Crθrθ)) ≈ 4.5
        κ = cell(LayeredCylinder(rc_radii, (rc_K[1], radial_ti(rc_chart, 0.2, 0.6))))
        @test get_param(κ, layer_param(:I, 2, :radial)) == 0.6
        # ⟨ε_rr²⟩ in the shell of the isotropic cylinder, against its pointwise field.
        iso = LayeredCylinder(rc_radii, rc_C)
        fields = LayeredCylinderFields(iso, rc_C₀)
        𝔹 = radial_ti(rc_chart, 1.0, 0.0, 0.0, 0.0, 0.0)
        form(x) = (ε = local_gradient(fields, x, rc_E; layer = 2); ε ⊡ (tensor_at(𝔹, x) ⊡ ε))
        @test second_moment(cell(iso), Dilute(), (:I, 2), rc_E, 𝔹) ≈ rc_annulus_integral(form, 0.6, 1.0) / (1 - 0.6^2) rtol = 1.0e-9
        @test get_param(cell(iso), layer_param(:I, 2, 𝔹)) == 0
        # An inclined cylinder keeps its axis when a layer is replaced.
        axis = Tuple(TensND.normalize([1.0, 2.0, 2.0]))
        moved = set_param(cell(LayeredCylinder(rc_radii, rc_C; axis)), layer_param(:I, 2, 𝔹), 0.5)
        @test collect(moved.phases[:I].geometry.axis) ≈ collect(axis)
        # A layer transversely isotropic about the axis has no radial writing.
        ti = cell(LayeredCylinder(rc_radii, (lc_tensor(LC_F), rc_C[2])))
        @test_throws ArgumentError set_param(ti, layer_param(:I, 1, 𝔹), 0.5)
    end

    @testset "graded layers: moduli functions of the radius" begin
        data(t) = collect(TensND.get_data(t))
        tensors = (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
        # A constant profile is the uniform layer, in the shell and in the core (which is
        # integrated from the axis over eighteen e-folds of the radius).
        for itf in rc_interfaces, (uniform, graded) in (
                    ((rc_C[1], rc_shell), (rc_C[1], r -> rc_shell)),
                    ((rc_C[1], rc_shell), (r -> rc_C[1], rc_shell)),
                    (rc_C, (GradedLayer(r -> rc_C[1]; reltol = 1.0e-11), r -> rc_C[2])),
                )
            a = LayeredCylinder(rc_radii, uniform; interfaces = itf)
            b = LayeredCylinder(rc_radii, graded; interfaces = itf)
            for f in tensors
                @test rc_km(f(b, rc_C₀, rc_C₀)) ≈ rc_km(f(a, rc_C₀, rc_C₀)) rtol = 1.0e-7
            end
            for layer in 1:2
                @test rc_km(flux_gradient_concentration(b, rc_C₀; layer)) ≈ rc_km(flux_gradient_concentration(a, rc_C₀; layer)) rtol = 1.0e-7
            end
        end
        @test layer_modulus(LayeredCylinder(rc_radii, (rc_C[1], r -> rc_shell)), 2) isa GradedLayer
        # A profile that varies, in the shell then from the axis: thin layers, extrapolated.
        soft(r) = iso_stiffness(40.0 - 20.0 * r^2, 20.0 - 8.0 * r^2)
        wound(r) = radial_ti(rc_chart; n = 12.0 + 3r, l = 5.0, k = 19.0 - 2r, m = 11.0, p = 4.0 + r)
        thin(profile, a, n) = (e = range(a, 1.0; length = n + 1); (Tuple(e[2:end]), ntuple(i -> profile((e[i] + e[i + 1]) / 2), n)))
        for (graded, layered) in (
                (LayeredCylinder(rc_radii, (rc_C[1], wound)), n -> ((r, m) = thin(wound, 0.6, n); LayeredCylinder((0.6, r...), (rc_C[1], m...)))),
                (LayeredCylinder((1.0,), (soft,)), n -> LayeredCylinder(thin(soft, 0.0, n)...)),
            )
            for f in tensors
                @test rc_km(f(graded, rc_C₀, rc_C₀)) ≈ (4 .* rc_km(f(layered(80), rc_C₀, rc_C₀)) .- rc_km(f(layered(40), rc_C₀, rc_C₀))) ./ 3 rtol = 1.0e-6
            end
        end
        # A power of the radius is integrated in a cylinder.
        @test rc_km(property_contribution(LayeredCylinder(rc_radii, (rc_C[1], PowerLawLayer(rc_shell, 1.5))), rc_C₀, rc_C₀)) ≈
            rc_km(property_contribution(LayeredCylinder(rc_radii, (rc_C[1], r -> r^1.5 * rc_shell)), rc_C₀, rc_C₀)) rtol = 1.0e-10
        # Conduction: a disk of conductivity k_s r^β around a core, T = F(r) cos θ with
        # F = B₁ r^p₁ + B₂ r^p₂, p = (−β ± √(β² + 4))/2, between a core A r and r + g/r outside.
        k_c, k_s, β, R, k₀ = 8.0, 0.4, 2.5, 1.6, 1.3
        function power_disk(β, R)
            p = ((-β + sqrt(β^2 + 4)) / 2, (-β - sqrt(β^2 + 4)) / 2)
            q(e, r) = k_s * r^β * e * r^(e - 1)
            S = [1 -1 -1 0; k_c -q(p[1], 1) -q(p[2], 1) 0; 0 R^p[1] R^p[2] -1 / R; 0 q(p[1], R) q(p[2], R) k₀ / R^2]
            return -2k₀ * (S \ [0, 0, R, k₀])[4] / R^2
        end
        conducting(β, R) = first(data(property_contribution(LayeredCylinder((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^β))), TensISO{3}(k₀), TensISO{3}(k₀))))
        @test conducting(β, R) ≈ power_disk(β, R) rtol = 1.0e-9
        @test ForwardDiff.derivative(x -> conducting(x, R), β) ≈ ForwardDiff.derivative(x -> power_disk(x, R), β) rtol = 1.0e-7
        @test ForwardDiff.derivative(x -> conducting(β, x), R) ≈ ForwardDiff.derivative(x -> power_disk(β, x), R) rtol = 1.0e-7
        # Along the axis, and for the bounds: the mean of k_s r^β over the annulus.
        cyl = LayeredCylinder((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^β)))
        mean = (k_c + 2k_s * (R^(β + 2) - 1) / (β + 2)) / R^2
        @test last(data(flux_gradient_concentration(cyl, TensISO{3}(k₀), TensISO{3}(k₀)))) ≈ mean rtol = 1.0e-9
        @test last(data(LCyl.layer_property_average(cyl, TensISO{3}(k₀)))) ≈ mean rtol = 1.0e-9
        averaged(R) = last(data(LCyl.layer_property_average(LayeredCylinder((1.0, R), (TensISO{3}(k_c), r -> TensISO{3}(k_s * r^β))), TensISO{3}(k₀))))
        @test ForwardDiff.derivative(averaged, R) ≈ ForwardDiff.derivative(R -> (k_c + 2k_s * (R^(β + 2) - 1) / (β + 2)) / R^2, R) rtol = 1.0e-8
        @test last(data(flux_gradient_concentration(cyl, TensISO{3}(k₀); layer = 2))) ≈ 2k_s * (R^(β + 2) - 1) / ((β + 2) * (R^2 - 1)) rtol = 1.0e-9
        elastic = LayeredCylinder(rc_radii, (rc_C[1], wound))
        mean_over_angle(F) = sum(rc_km(tensor_at(F, [cos(θ), sin(θ), 0.0])) for θ in 2π .* ((1:48) .- 0.5) ./ 48) / 48
        xs, ws = QuadGK.gauss(12)
        shell_mean = sum(w * 0.2 * (0.8 + 0.2x) * mean_over_angle(wound(0.8 + 0.2x)) for (x, w) in zip(xs, ws)) * 2 / (1 - 0.6^2)
        @test rc_km(LCyl.layer_property_average(elastic, rc_C₀)) ≈ 0.36 .* rc_km(rc_C[1]) .+ 0.64 .* shell_mean rtol = 1.0e-9
        # Complex moduli.
        z(r) = iso_stiffness((4.0 + 0.5im) * (1 + r), (3.0 + 0.2im) * (1 + r))
        zkm(t) = collect(mandel(t, TensND.CanonicalBasis{3, ComplexF64}()))
        zl(n) = ((r, m) = thin(z, 0.6, n); LayeredCylinder((0.6, r...), (rc_C[1], m...)))
        zf(c) = zkm(property_contribution(c, rc_C₀, rc_C₀))
        @test zf(LayeredCylinder(rc_radii, (rc_C[1], z))) ≈ (4 .* zf(zl(80)) .- zf(zl(40))) ./ 3 rtol = 1.0e-6
        # What a graded layer does not have.
        @test_throws ArgumentError LayeredCylinderFields(elastic, rc_C₀)
        @test_throws ArgumentError property_contribution(LayeredCylinder(rc_radii, (rc_C[1], r -> lc_tensor(LC_F))), rc_C₀, rc_C₀)
        @test_throws ArgumentError property_contribution(LayeredCylinder(rc_radii, (rc_K[1], r -> TensND.TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0)))), TensISO{3}(1.0), TensISO{3}(1.0))
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => rc_C₀); fraction = :rest)
        add_phase!(rve, :I, LayeredCylinder(rc_radii, (PowerLawLayer(rc_shell, 1.5), wound)), Dict(:C => rc_C₀); fraction = 0.3)
        @test_throws ArgumentError get_param(rve, layer_param(:I, 2, :Crrrr))
        @test_throws ArgumentError set_param(rve, layer_param(:I, 1, :Crrrr), 1.0)
        @test_throws ArgumentError get_param(rve, layer_param(:I, 1, rc_shell))
        @test_throws ArgumentError set_param(rve, layer_param(:I, 2, rc_shell), 1.0)
    end

    @testset "refused inputs" begin
        cyl = LayeredCylinder(rc_radii, (rc_C[1], rc_shell))
        @test_throws ArgumentError LayeredCylinderFields(cyl, rc_C₀)
        @test_throws ArgumentError LayeredCylinderTransportFields(LayeredCylinder((1.0,), (radial_ti(rc_chart, 5.0, 2.0),)), TensISO{3}(1.0))
        # A field of another axis of the local frame, of another chart, of another order.
        along_θ = ChartTens(TensTI{4}(TensND.get_data(rc_shell.pattern)..., (0.0, 1.0, 0.0)), rc_chart)
        @test_throws ArgumentError property_contribution(LayeredCylinder(rc_radii, (rc_C[1], along_θ)), rc_C₀, rc_C₀)
        @test_throws MethodError property_contribution(LayeredCylinder(rc_radii, (rc_C[1], radial_ti(SphericalChart(); rc_hill...))), rc_C₀, rc_C₀)
        @test_throws ArgumentError property_contribution(cyl, TensISO{3}(1.0), TensISO{3}(1.0))
        @test_throws ArgumentError property_contribution(LayeredCylinder((1.0,), (radial_ti(rc_chart, 5.0, 2.0),)), rc_C₀, rc_C₀)
        # Moduli that are not those of a solid.
        @test_throws ArgumentError property_contribution(LayeredCylinder(rc_radii, (rc_C[1], radial_ti(rc_chart; rc_hill..., p = -1.0))), rc_C₀, rc_C₀)
        @test_throws ArgumentError property_contribution(LayeredCylinder((1.0,), (radial_ti(rc_chart, 5.0, -2.0),)), TensISO{3}(1.0), TensISO{3}(1.0))
        # A two-dimensional cell.
        c₀ = plane_strain_stiffness(rc_C₀)
        @test_throws ArgumentError property_contribution(cyl, c₀, c₀)
        @test_throws ArgumentError property_contribution(LayeredCylinder((1.0,), (radial_ti(rc_chart, 5.0, 2.0),)), TensISO{2}(1.0), TensISO{2}(1.0))
        # Aging viscoelasticity.
        @test_throws ArgumentError MeanFieldHomogenization.Viscoelasticity._alv_operator(rc_shell, [0.0, 1.0])
        @test_throws ArgumentError MeanFieldHomogenization.Viscoelasticity._layer_iso_volterra(radial_ti(SphericalChart(); rc_hill...), [0.0, 1.0])
    end
end
