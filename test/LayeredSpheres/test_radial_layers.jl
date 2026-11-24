# =============================================================================
#  test_radial_layers.jl — layered sphere whose layers are transversely
#  isotropic about the radial direction (`RadialTI`), and the parameters of a
#  layer (`layer_param`).
#
#  References, independent of the solver under test:
#   1. isotropic layers written as `RadialTI`: the amplitude recurrences of the
#      isotropic sphere, with every kind of interface;
#   2. Kreher's theorem in a layer: the derivative of 𝐄:ℕ:𝐄 when the modulus of
#      a layer moves along a tensor 𝔹 that turns with the point is the integral
#      of ε:𝔹(x):ε over the layer, computed by a quadrature of the pointwise
#      field of the isotropic sphere (`LayeredSphereFields`), for each of the
#      five moduli and of the two conductivities;
#   3. the homogeneous sphere of radial and tangential conductivities k_r, k_t,
#      neutral in a medium of conductivity (k_r/2)(−1 + √(1 + 8k_t/k_r))
#      (Schulgasser 1983; Milton 2002, eq. 7.30), and its elastic counterpart
#      under a hydrostatic strain, u = r^s with n s(s + 1) = 2(2k − l) (Kushch &
#      Mogilevskaya 2022, eqs. 5-8), neutral for 3k₀ = n s + 2l;
#   4. a layer cut into two layers of the same moduli is the same sphere.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import QuadGK
import SymPyPythonCall

const RL = MeanFieldHomogenization.LayeredSpheres

rl_C₀ = iso_stiffness(10.0, 6.0)
rl_C = (iso_stiffness(40.0, 20.0), iso_stiffness(4.0, 3.0))
rl_K = (TensISO{3}(5.0), TensISO{3}(0.3))
rl_radii = (0.6, 1.0)
rl_E = Tens(Symmetric([1.0 0.2 0.1; 0.2 -0.5 0.3; 0.1 0.3 0.7]))
rl_shell = RadialTI{4}(; Crrrr = 12.0, Cθθθθ = 30.0, Cθθφφ = 8.0, Crrθθ = 5.0, Crθrθ = 4.0)
# A positive definite modulus whose deviatoric modes oscillate (complex exponents).
rl_oscillating = RadialTI{4}(; Crrrr = 0.768, Cθθθθ = 2.519, Cθθφφ = 0.052, Crrθθ = 0.622, Crθrθ = 3.6)
rl_data(t) = collect(TensND.get_data(t))
rl_unit(i, n) = ntuple(j -> float(i == j), n)

# ∫ g(x) dV over ra < |x| < rb, divided by the volume of the unit ball.
function rl_shell_integral(g, ra, rb; n = 12, nφ = 24)
    xs, ws = QuadGK.gauss(n)
    total = 0.0
    for (xr, wr) in zip(xs, ws), (c, wc) in zip(xs, ws), j in 1:nφ
        r = (ra + rb) / 2 + (rb - ra) / 2 * xr
        φ = 2π * (j - 0.5) / nφ
        x = r .* [sqrt(1 - c^2) * cos(φ), sqrt(1 - c^2) * sin(φ), c]
        total += wr * wc * (2π / nφ) * (rb - ra) / 2 * r^2 * g(x)
    end
    return total / (4π / 3)
end

function rl_cell(sphere; key = :C, P₀ = rl_C₀, f = 0.3)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(key => P₀); fraction = :rest)
    add_phase!(rve, :I, sphere, Dict(key => P₀); fraction = f)
    return rve
end

@testset "layers transversely isotropic about the radius" begin
    @testset "the modulus of a layer" begin
        c = RadialTI(rl_C[1])
        k, μ = k_mu(rl_C[1])
        @test get_coefficient(c, :Crrrr) ≈ k + 4μ / 3
        @test get_coefficient(c, :Cθθφφ) ≈ k - 2μ / 3
        @test get_coefficient(c, :Crθrθ) ≈ μ
        # At a point, the tensor of fixed axes it stands for.
        x = [0.3, -0.5, 0.8]
        @test canonical_components(tensor_at(c, x)) ≈ canonical_components(rl_C[1])
        T = tensor_at(rl_shell, x)
        n = Tens(x / norm(x))
        @test n ⋅ (n ⋅ T ⋅ n) ⋅ n ≈ 12.0
        @test RadialTI(TensTI{4}(rl_shell.data..., [0.0, 0.0, 1.0])) == rl_shell
        @test RadialTI(TensTI{2}(2.0, 5.0, [0.0, 0.0, 1.0])) == RadialTI{2}(2.0, 5.0)
        # Names: the components of the local frame, or the Walpole coefficients.
        for (sel, v) in ((:Crrrr, 13.0), (:Cθθθθ, 31.0), (:Cθθφφ, 7.0), (:Crrθθ, 4.5), (:Crθrθ, 3.5), (:ℓ₃, 2.0))
            d = set_coefficient(rl_shell, sel, v)
            @test get_coefficient(d, sel) ≈ v
            others = filter(!=(sel), sel === :ℓ₃ ? (:ℓ₁, :ℓ₂, :ℓ₅, :ℓ₆) : (:Crrrr, :Cθθθθ, :Cθθφφ, :Crrθθ, :Crθrθ))
            @test all(get_coefficient(d, o) ≈ get_coefficient(rl_shell, o) for o in others)
        end
        κ = RadialTI{2}(2.0, 5.0)                           # tangential, radial
        @test get_coefficient(κ, :radial) == 5.0
        @test get_coefficient(set_coefficient(κ, :tangential, 3.0), :tangential) == 3.0
        @test_throws ArgumentError get_coefficient(rl_shell, :k)
        @test_throws ArgumentError set_coefficient(rl_shell, :k, 1.0)
        @test_throws ArgumentError get_coefficient(κ, :axial)
        # One radial layer makes every layer one; a tensor of fixed axes is refused.
        @test LayeredSphere(rl_radii, (rl_C[1], rl_shell)) isa RL.RadialLayeredSphere
        @test !(LayeredSphere(rl_radii, rl_C) isa RL.RadialLayeredSphere)
        @test_throws ArgumentError LayeredSphere(rl_radii, (TensTI{4}(rl_shell.data..., [0.0, 0.0, 1.0]), rl_shell))
        @test eltype(2 * rl_shell) == Float64 && (rl_shell * 2).data == (2 * rl_shell).data
    end

    @testset "isotropic layers, written as radial ones" begin
        interfaces = (
            (PerfectInterface(), PerfectInterface()),
            (SpringInterface(; sn = 0.02, st = 0.05), MembraneInterface(3.0, 2.0)),
        )
        for itf in interfaces, external in (true, false)
            iso = LayeredSphere(rl_radii, rl_C; interfaces = itf)
            rad = LayeredSphere(rl_radii, (RadialTI(rl_C[1]), rl_C[2]); interfaces = itf)
            for f in (gradient_gradient_concentration, flux_gradient_concentration, property_contribution)
                @test rl_data(f(rad, rl_C₀, rl_C₀; external)) ≈ rl_data(f(iso, rl_C₀, rl_C₀; external)) rtol = 1.0e-11
            end
            A, N = MeanFieldHomogenization.Core.concentration_and_contribution(rad, rl_C₀, rl_C₀; external)
            @test rl_data(N) ≈ rl_data(property_contribution(iso, rl_C₀, rl_C₀; external)) rtol = 1.0e-11
            A, B = MeanFieldHomogenization.Core.concentration_and_flux_average(rad, rl_C₀, rl_C₀; external)
            @test rl_data(A) ≈ rl_data(gradient_gradient_concentration(iso, rl_C₀, rl_C₀; external)) rtol = 1.0e-11
            for layer in 1:2, internal in (true, false)
                @test rl_data(gradient_gradient_concentration(rad, rl_C₀; layer, external, internal)) ≈
                    rl_data(gradient_gradient_concentration(iso, rl_C₀; layer, external, internal)) rtol = 1.0e-11
            end
        end
        # The mean stress of a layer, which for an isotropic one is ℂₖ : ⟨ε⟩ₖ.
        rad = LayeredSphere(rl_radii, (RadialTI(rl_C[1]), rl_C[2]))
        for layer in 1:2
            A = gradient_gradient_concentration(rad, rl_C₀; layer)
            @test rl_data(flux_gradient_concentration(rad, rl_C₀; layer)) ≈ rl_data(rl_C[layer]) .* rl_data(A) rtol = 1.0e-11
        end
        @test_throws BoundsError gradient_gradient_concentration(rad, rl_C₀; layer = 3)
        @test_throws BoundsError flux_gradient_concentration(rad, rl_C₀; layer = 0)

        K₀ = TensISO{3}(1.0)
        for itf in ((PerfectInterface(), PerfectInterface()), (KapitzaInterface(0.1), SurfaceConductiveInterface(0.4))), external in (true, false)
            iso = LayeredSphere(rl_radii, rl_K; interfaces = itf)
            rad = LayeredSphere(rl_radii, (RadialTI(rl_K[1]), rl_K[2]); interfaces = itf)
            for f in (gradient_gradient_concentration, flux_gradient_concentration)
                @test rl_data(f(rad, K₀, K₀; external)) ≈ rl_data(f(iso, K₀, K₀; external)) rtol = 1.0e-11
            end
            @test rl_data(property_contribution(rad, K₀; external)) ≈ rl_data(property_contribution(iso, K₀; external)) rtol = 1.0e-10
            A, N = MeanFieldHomogenization.Core.concentration_and_contribution(rad, K₀, K₀; external)
            @test rl_data(N) ≈ rl_data(property_contribution(iso, K₀; external)) rtol = 1.0e-10
            A, B = MeanFieldHomogenization.Core.concentration_and_flux_average(rad, K₀, K₀; external)
            @test rl_data(B) ≈ rl_data(flux_gradient_concentration(iso, K₀, K₀; external)) rtol = 1.0e-11
            for layer in 1:2, internal in (true, false)
                @test rl_data(gradient_gradient_concentration(rad, K₀; layer, external, internal)) ≈
                    rl_data(gradient_gradient_concentration(iso, K₀; layer, external, internal)) rtol = 1.0e-11
            end
        end
        @test_throws BoundsError gradient_gradient_concentration(LayeredSphere(rl_radii, (RadialTI(rl_K[1]), rl_K[2])), K₀; layer = 3)
    end

    @testset "the derivative along a tensor that turns with the point" begin
        iso = LayeredSphere(rl_radii, rl_C)
        fields = LayeredSphereFields(iso, rl_C₀)
        for (layer, (ra, rb)) in enumerate(((0.0, 0.6), (0.6, 1.0))), i in 1:5
            𝔹 = RadialTI{4}(rl_unit(i, 5)...)
            moved(t) = LayeredSphere(rl_radii, Base.setindex(iso.moduli, RadialTI(iso.moduli[layer]) + t * 𝔹, layer))
            d = ForwardDiff.derivative(t -> rl_E ⊡ (property_contribution(moved(t), rl_C₀, rl_C₀) ⊡ rl_E), 0.0)
            form(x) = (ε = local_gradient_gradient_concentration(fields, x) ⊡ rl_E; ε ⊡ (tensor_at(𝔹, x) ⊡ ε))
            @test d ≈ rl_shell_integral(form, ra, rb) rtol = 1.0e-10
        end
        K₀ = TensISO{3}(1.0)
        isoK = LayeredSphere(rl_radii, rl_K)
        fieldsK = LayeredSphereTransportFields(isoK, K₀)
        G = Tens([0.3, -0.5, 0.8])
        for (layer, (ra, rb)) in enumerate(((0.0, 0.6), (0.6, 1.0))), i in 1:2
            𝑩 = RadialTI{2}(rl_unit(i, 2)...)
            moved(t) = LayeredSphere(rl_radii, Base.setindex(isoK.moduli, RadialTI(isoK.moduli[layer]) + t * 𝑩, layer))
            d = ForwardDiff.derivative(t -> G ⋅ (property_contribution(moved(t), K₀) ⋅ G), 0.0)
            form(x) = (g = local_gradient_gradient_concentration(fieldsK, x) ⋅ G; g ⋅ (tensor_at(𝑩, x) ⋅ g))
            @test d ≈ rl_shell_integral(form, ra, rb) rtol = 1.0e-10
        end
    end

    @testset "neutral spheres" begin
        kᵣ, kₜ = 2.0, 5.0
        ball = LayeredSphere((1.0,), (RadialTI{2}(kₜ, kᵣ),))
        k✶ = kᵣ / 2 * (-1 + sqrt(1 + 8kₜ / kᵣ))
        @test abs(only(rl_data(property_contribution(ball, TensISO{3}(k✶))))) < 1.0e-13
        @test only(rl_data(property_contribution(ball, TensISO{3}(0.9k✶)))) > 0
        # The same with symbols.
        sr, st = SymPyPythonCall.symbols("k_r k_t"; positive = true)
        s✶ = sr / 2 * (-1 + sqrt(1 + 8st / sr))
        N = only(TensND.get_data(property_contribution(LayeredSphere((1,), (RadialTI{2}(st, sr),)), TensISO{3}(s✶))))
        @test iszero(SymPyPythonCall.simplify(N))
        # Elasticity, spherical part: u = r^s, neutral when 3k₀ = n s + 2l.
        h = RL._hill_moduli(rl_shell)
        s = -1 / 2 + sqrt(1 / 4 + 2 * (2 * h.k - h.l) / h.n)
        N = property_contribution(LayeredSphere((1.0,), (rl_shell,)), rl_C₀, iso_stiffness((h.n * s + 2 * h.l) / 3, 6.0))
        @test abs(TensND.get_data(N)[1]) < 1.0e-12
    end

    @testset "oscillating modes" begin
        @test RL._oscillates(rl_oscillating) && !RL._oscillates(rl_shell)
        whole = LayeredSphere(rl_radii, (rl_C[1], rl_oscillating))
        cut = LayeredSphere((0.6, 0.8, 1.0), (rl_C[1], rl_oscillating, rl_oscillating))
        N = property_contribution(whole, rl_C₀, rl_C₀)
        @test eltype(N) == Float64
        @test rl_data(property_contribution(cut, rl_C₀, rl_C₀)) ≈ rl_data(N) rtol = 1.0e-10
        # The same cut on real exponents.
        @test rl_data(property_contribution(LayeredSphere((0.6, 0.8, 1.0), (rl_C[1], rl_shell, rl_shell)), rl_C₀, rl_C₀)) ≈
            rl_data(property_contribution(LayeredSphere(rl_radii, (rl_C[1], rl_shell)), rl_C₀, rl_C₀)) rtol = 1.0e-10
        # A derivative goes through the complex modes: N is homogeneous of degree one in the moduli.
        scaled(t) = TensND.get_data(property_contribution(LayeredSphere(rl_radii, (t * rl_C[1], t * rl_oscillating)), t * rl_C₀, t * rl_C₀))[2]
        @test ForwardDiff.derivative(scaled, 1.0) ≈ TensND.get_data(N)[2] rtol = 1.0e-10
    end

    @testset "in the schemes, and the parameters of a layer" begin
        sphere = LayeredSphere(rl_radii, (rl_C[1], rl_shell))
        rve = rl_cell(sphere)
        kV, kMT, kR = (k_mu(homogenize(rve, s, :C))[1] for s in (Voigt(), MoriTanaka(), Reuss()))
        @test kR < kMT < kV
        @test k_mu(homogenize(rve, SelfConsistent(), :C))[1] ≈ kMT rtol = 1.0e-2
        bulk(C) = k_mu(C)[1]
        # A coefficient in the frame that turns with the point, against the sphere rebuilt.
        rebuilt(x) = bulk(homogenize(rl_cell(LayeredSphere(rl_radii, (rl_C[1], set_coefficient(rl_shell, :Crrrr, x)))), MoriTanaka(), :C))
        p = layer_param(:I, 2, :Crrrr)
        @test get_param(rve, p) == 12.0
        @test derivative(rve, MoriTanaka(), p; indexer = bulk) ≈ ForwardDiff.derivative(rebuilt, 12.0) rtol = 1.0e-12
        # An isotropic sphere keeps the names of its isotropic layers.
        iso_rve = rl_cell(LayeredSphere(rl_radii, rl_C))
        @test get_param(set_param(iso_rve, layer_param(:I, 1, :μ), 25.0), layer_param(:I, 1, :μ)) == 25.0
        @test get_param(set_param(rve, layer_param(:I, 1, :Crθrθ), 25.0), layer_param(:I, 1, :Crθrθ)) == 25.0
        # The radius of a layer, by the parameter of the geometry.
        radius(x) = bulk(homogenize(rl_cell(LayeredSphere((x, 1.0), (rl_C[1], rl_shell))), MoriTanaka(), :C))
        @test derivative(rve, MoriTanaka(), geometry(:I, :radii, 1); indexer = bulk) ≈ ForwardDiff.derivative(radius, 0.6) rtol = 1.0e-12
        # Conduction.
        rveK = rl_cell(LayeredSphere(rl_radii, (rl_K[1], RadialTI{2}(0.2, 0.6))); key = :K, P₀ = TensISO{3}(1.0))
        cond(K) = only(TensND.get_data(K))
        @test cond(homogenize(rveK, Reuss(), :K)) < cond(homogenize(rveK, MoriTanaka(), :K)) < cond(homogenize(rveK, Voigt(), :K))
        @test derivative(rveK, MoriTanaka(), layer_param(:I, 2, :radial); output = :K, indexer = cond) > 0

        @testset "second moments in a layer" begin
            iso = LayeredSphere(rl_radii, rl_C)
            cell = rl_cell(iso)
            fields = LayeredSphereFields(iso, rl_C₀)
            f₂ = 1 - 0.6^3
            𝔹 = RadialTI{4}(1.0, 0.0, 0.0, 0.0, 0.0)                    # ε_rr²
            form(x) = (ε = local_gradient_gradient_concentration(fields, x) ⊡ rl_E; ε ⊡ (tensor_at(𝔹, x) ⊡ ε))
            @test second_moment(cell, Dilute(), (:I, 2), rl_E, 𝔹) ≈ rl_shell_integral(form, 0.6, 1.0) / f₂ rtol = 1.0e-10
            @test get_param(cell, layer_param(:I, 2, 𝔹)) == 0
            # An isotropic form: ⟨εᵈ:εᵈ⟩ of the layer, by an isotropic direction or by its radial writing.
            𝕂 = TensISO{3}(0.0, 1.0)
            @test second_moment(cell, MoriTanaka(), (:I, 2), rl_E, 𝕂) ≈ second_moment(cell, MoriTanaka(), (:I, 2), rl_E, RadialTI(𝕂)) rtol = 1.0e-11
            # The moments of the layers add up to that of the phase.
            whole = sum(layer_volume_fraction(iso, k) * second_moment(cell, MoriTanaka(), (:I, k), rl_E, 𝕂) for k in 1:2)
            ε̄ = concentration_tensors(cell, MoriTanaka(), :C).strain[:I] ⊡ rl_E
            @test whole > ε̄ ⊡ (𝕂 ⊡ ε̄)
        end

        @testset "refusals" begin
            @test_throws ArgumentError get_param(rve, layer_param(:M, 1, :k))
            @test_throws ArgumentError get_param(rve, layer_param(:Z, 1, :k))
            @test_throws ArgumentError get_param(rve, layer_param(:I, 3, :k))
            @test_throws ArgumentError LayeredSphereFields(sphere, rl_C₀)
            @test_throws ArgumentError LayeredSphereTransportFields(LayeredSphere((1.0,), (RadialTI{2}(1.0, 2.0),)), TensISO{3}(1.0))
            @test_throws ArgumentError MeanFieldHomogenization.Core.harmonic_response(sphere, rl_C₀, 2)
            @test_throws ArgumentError MeanFieldHomogenization.Core.harmonic_response(LayeredSphere((1.0,), (RadialTI{2}(1.0, 2.0),)), TensISO{3}(1.0), 1)
            @test_throws ArgumentError layer_gradient_average(sphere, rl_C₀, rl_E, 1)
            @test_throws ArgumentError RL._cond_layer_moduli(sphere)
            @test_throws ArgumentError RL._layer_iso_pairs(sphere)
        end
    end
end
