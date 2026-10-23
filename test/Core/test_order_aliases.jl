# =============================================================================
#  test_order_aliases.jl — one function per notion whatever the tensor order,
#  and the names of elasticity and of conduction that forward to it.
#
#  Each generic serves a fourth-order (elastic) and a second-order (conduction)
#  call. Each explicit name returns exactly what the generic returns at its own
#  order, and refuses the other order with an error that names the generic.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const OA_C₁, OA_C₀ = TensISO{3}(90.0, 40.0), TensISO{3}(30.0, 10.0)
const OA_K₁, OA_K₀ = TensISO{3}(8.0), TensISO{3}(2.0)
const OA_ELL = Ellipsoid(1.0, 0.8, 0.5)
const OA_TIMES = [0.0, 0.4, 1.5]

# The arguments of an order: the property pair, then the remote load.
_oa_pair(::Val{4}) = (OA_C₁, OA_C₀)
_oa_pair(::Val{2}) = (OA_K₁, OA_K₀)
_oa_other(::Val{4}) = Val(2)
_oa_other(::Val{2}) = Val(4)

# An alias called at the wrong order throws an `ArgumentError` naming the generic.
function _oa_refuses(f, generic::Symbol)
    err = try
        f()
        nothing
    catch e
        e
    end
    return err isa ArgumentError && occursin("`$(generic)`", sprint(showerror, err))
end

_oa_same(a, b) = get_array(a) == get_array(b)

@testset "Order aliases — every generic serves both orders" begin
    for N in (4, 2)
        P₁, P₀ = _oa_pair(Val(N))
        for f in (
                gradient_gradient_concentration, flux_gradient_concentration,
                gradient_flux_concentration, flux_flux_concentration,
                property_contribution, inverse_property_contribution,
            )
            @test f(OA_ELL, P₁, P₀) isa TensND.AbstractTens{N, 3}
        end
        Nc = property_contribution(OA_ELL, P₁, P₀)
        @test delta_property(Nc, 0.1) isa TensND.AbstractTens{N, 3}
        @test delta_inverse_property(Nc, 0.1) isa TensND.AbstractTens{N, 3}
    end
end

@testset "Order aliases — dilute tensors, contributions and deltas" begin
    for (alias, generic, N) in (
            (strain_strain_concentration, gradient_gradient_concentration, 4),
            (stress_strain_concentration, flux_gradient_concentration, 4),
            (strain_stress_concentration, gradient_flux_concentration, 4),
            (stress_stress_concentration, flux_flux_concentration, 4),
            (stiffness_contribution, property_contribution, 4),
            (compliance_contribution, inverse_property_contribution, 4),
            (conductivity_contribution, property_contribution, 2),
            (resistivity_contribution, inverse_property_contribution, 2),
        )
        good, bad = _oa_pair(Val(N)), _oa_pair(_oa_other(Val(N)))
        @test _oa_same(alias(OA_ELL, good...), generic(OA_ELL, good...))
        @test _oa_refuses(() -> alias(OA_ELL, bad...), nameof(generic))
    end
    for (alias, generic, N) in (
            (delta_stiffness, delta_property, 4),
            (delta_compliance, delta_inverse_property, 4),
            (delta_conductivity, delta_property, 2),
            (delta_resistivity, delta_inverse_property, 2),
        )
        good = property_contribution(OA_ELL, _oa_pair(Val(N))...)
        bad = property_contribution(OA_ELL, _oa_pair(_oa_other(Val(N)))...)
        @test _oa_same(alias(good, 0.1), generic(good, 0.1))
        @test _oa_refuses(() -> alias(bad, 0.1), nameof(generic))
    end
end

@testset "Order aliases — concentration tensors of a scheme and Levin" begin
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => OA_C₀, :K => OA_K₀); fraction = :rest)
    add_phase!(rve, :I, OA_ELL, Dict(:C => OA_C₁, :K => OA_K₁); fraction = 0.2)
    ct4 = concentration_tensors(rve, MoriTanaka(:M), :C)
    ct2 = concentration_tensors(rve, MoriTanaka(:M), :K)
    for (alias, generic) in (
            (strain_strain_concentration, gradient_gradient_concentration),
            (stress_strain_concentration, flux_gradient_concentration),
            (strain_stress_concentration, gradient_flux_concentration),
            (stress_stress_concentration, flux_flux_concentration),
            (stiffness_contribution, property_contribution),
            (compliance_contribution, inverse_property_contribution),
        )
        @test _oa_same(alias(ct4, :I), generic(ct4, :I))
        @test _oa_refuses(() -> alias(ct2, :I), nameof(generic))
    end
    @test _oa_same(conductivity_contribution(ct2, :I), property_contribution(ct2, :I))
    @test _oa_refuses(() -> conductivity_contribution(ct4, :I), :property_contribution)

    # Levin: an eigenstrain is a second-order tensor, an eigengradient a vector.
    ε̂ = Dict(:M => TensISO{3}(1.0e-3), :I => TensISO{3}(2.0e-3))
    ĝ = Dict(:M => Tens([1.0, 0.0, 0.5]), :I => Tens([0.0, 2.0, -1.0]))
    @test _oa_same(levin_free_strain(ct4, ε̂), levin_free_gradient(ct4, ε̂))
    @test _oa_refuses(() -> levin_free_strain(ct2, ĝ), :levin_free_gradient)
    σ̂ = Dict(n => -(phase_property(rve, n, :C) ⊡ ε̂[n]) for n in (:M, :I))
    q̂ = Dict(n => phase_property(rve, n, :K) ⋅ ĝ[n] for n in (:M, :I))
    @test _oa_same(levin_prestress(ct4, σ̂), levin_preflux(ct4, σ̂))
    @test _oa_refuses(() -> levin_prestress(ct2, q̂), :levin_preflux)
end

@testset "Order aliases — laminate layers" begin
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => OA_C₁, :K => OA_K₁); thickness = 0.3)
    add_layer!(lam, :B, Dict(:C => OA_C₀, :K => OA_K₀); thickness = 0.7)
    for (alias, generic) in (
            (layer_strain_concentration, layer_gradient_concentration),
            (layer_stress_concentration, layer_flux_concentration),
        )
        @test _oa_same(alias(lam, :A), generic(lam, :A))
        @test generic(lam, :A; property = :K) isa TensND.AbstractTens{2, 3}
        @test _oa_refuses(() -> alias(lam, :A; property = :K), nameof(generic))
    end
    # The kernels on Kelvin–Mandel blocks: 6 × 6 in elasticity, 3 × 3 in conduction.
    MC = MeanFieldHomogenization.Core
    C6, Ch6 = MeanFieldHomogenization.Laminates._concentration_setup(lam, :A, :C)
    K3, Kh3 = MeanFieldHomogenization.Laminates._concentration_setup(lam, :A, :K)
    for (alias, generic) in (
            (MC.laminate_strain_concentration, MC.laminate_gradient_concentration),
            (MC.laminate_stress_concentration, MC.laminate_flux_concentration),
            (MC.laminate_stress_strain_concentration, MC.laminate_flux_gradient_concentration),
            (MC.laminate_strain_stress_concentration, MC.laminate_gradient_flux_concentration),
        )
        @test alias(C6, Ch6) == generic(C6, Ch6)
        @test size(generic(K3, Kh3)) == (3, 3)
        @test_throws MethodError alias(K3, Kh3)
    end
end

@testset "Order aliases — layered sphere fields and averages" begin
    sph4 = LayeredSphere((0.6, 1.0), (OA_C₁, TensISO{3}(50.0, 20.0)))
    sph2 = LayeredSphere((0.6, 1.0), (OA_K₁, TensISO{3}(4.0)))
    f4 = LayeredSphereFields(sph4, OA_C₀)
    f2 = LayeredSphereTransportFields(sph2, OA_K₀)
    x = [0.3, 0.2, 0.1]
    ε∞ = TensISO{3}(1.0e-3)
    ∇T∞ = Tens([1.0, 0.0, 0.5])
    for (alias, generic) in (
            (local_strain_strain_concentration, local_gradient_gradient_concentration),
            (local_stress_strain_concentration, local_flux_gradient_concentration),
            (local_strain_stress_concentration, local_gradient_flux_concentration),
            (local_stress_stress_concentration, local_flux_flux_concentration),
        )
        @test _oa_same(alias(f4, x), generic(f4, x))
        @test generic(f2, x) isa TensND.AbstractTens{2, 3}
        @test _oa_refuses(() -> alias(f2, x), nameof(generic))
    end
    for (alias, generic) in ((local_strain, local_gradient), (local_stress, local_flux))
        @test _oa_same(alias(f4, x, ε∞), generic(f4, x, ε∞))
        @test _oa_refuses(() -> alias(f2, x, ∇T∞), nameof(generic))
    end
    for (alias, generic) in (
            (sphere_strain_average, sphere_gradient_average),
            (sphere_stress_average, sphere_flux_average),
        )
        @test _oa_same(alias(sph4, OA_C₀, ε∞), generic(sph4, OA_C₀, ε∞))
        @test _oa_refuses(() -> alias(sph2, OA_K₀, ∇T∞), nameof(generic))
    end
    for (alias, generic) in (
            (layer_strain_average, layer_gradient_average),
            (layer_stress_average, layer_flux_average),
        )
        @test _oa_same(alias(sph4, OA_C₀, ε∞, 1), generic(sph4, OA_C₀, ε∞, 1))
        @test _oa_refuses(() -> alias(sph2, OA_K₀, ∇T∞, 1), nameof(generic))
    end
    LS = MeanFieldHomogenization.LayeredSpheres
    @test _oa_same(LS.layer_stiffness_average(sph4), LS.layer_property_average(sph4, OA_C₀))
    @test _oa_same(LS.layer_compliance_average(sph4), LS.layer_inverse_property_average(sph4, OA_C₀))
    @test _oa_same(LS.layer_conductivity_average(sph2), LS.layer_property_average(sph2, OA_K₀))
    @test _oa_same(LS.layer_resistivity_average(sph2), LS.layer_inverse_property_average(sph2, OA_K₀))
end

@testset "Order aliases — aging viscoelasticity" begin
    sph4 = LayeredSphere((0.6, 1.0), (OA_C₁, TensISO{3}(50.0, 20.0)))
    sph2 = LayeredSphere((0.6, 1.0), (OA_K₁, TensISO{3}(4.0)))
    C₀ = heaviside_law(OA_C₀)
    K₀ = heaviside_law(OA_K₀)
    n = length(OA_TIMES)
    for (s, ref, rows) in ((sph4, C₀, 6n), (sph2, K₀, 3n))
        @test size(gradient_gradient_concentration_alv(s, ref, OA_TIMES)) == (rows, rows)
        @test size(property_contribution_alv(s, ref, OA_TIMES)) == (rows, rows)
    end
    @test strain_strain_concentration_alv(sph4, C₀, OA_TIMES) ==
        gradient_gradient_concentration_alv(sph4, C₀, OA_TIMES)
    @test stiffness_contribution_alv(sph4, C₀, OA_TIMES) == property_contribution_alv(sph4, C₀, OA_TIMES)
    @test conductivity_contribution_alv(sph2, K₀, OA_TIMES) == property_contribution_alv(sph2, K₀, OA_TIMES)
    @test _oa_refuses(
        () -> strain_strain_concentration_alv(sph2, K₀, OA_TIMES), :gradient_gradient_concentration_alv
    )
    @test _oa_refuses(() -> stiffness_contribution_alv(sph2, K₀, OA_TIMES), :property_contribution_alv)
    @test _oa_refuses(() -> conductivity_contribution_alv(sph4, C₀, OA_TIMES), :property_contribution_alv)
    # A second-order tensor stands for its Heaviside law.
    @test property_contribution_alv(sph2, OA_K₀, OA_TIMES) == property_contribution_alv(sph2, K₀, OA_TIMES)
    # The crack functions are elastic only, and their names in elasticity synonyms.
    crack = PennyCrack(1.0)
    H̃ = inverse_property_contribution_alv(crack, C₀, OA_TIMES)
    Ñ = property_contribution_alv(crack, C₀, OA_TIMES)
    @test compliance_contribution_alv(crack, C₀, OA_TIMES) == H̃
    @test stiffness_contribution_alv(crack, C₀, OA_TIMES) == Ñ
    @test delta_compliance_alv(crack, H̃, 0.1) == delta_inverse_property_alv(crack, H̃, 0.1)
    @test delta_stiffness_alv(crack, Ñ, 0.1) == delta_property_alv(crack, Ñ, 0.1)
    C_ref = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(C₀, OA_TIMES, 6)
    @test stiffness_contribution_alv_at(crack, C_ref) == property_contribution_alv_at(crack, C_ref)
    # The Volterra matrix of a running estimate carries its order in its size.
    K_run = property_contribution_alv(sph2, K₀, OA_TIMES) + I
    @test size(property_contribution_alv(sph2, K_run, OA_TIMES)) == (3n, 3n)
    @test_throws ArgumentError property_contribution_alv(sph2, ones(4n, 4n), OA_TIMES)
end

@testset "deprecated moduli extractors still answer, with a warning" begin
    # They forward to TensND; each is checked against the call it now makes.
    MC = MeanFieldHomogenization.Core
    C = TensISO{3}(3 * 5.0, 2 * 2.0)
    @test (@test_deprecated MC.extract_iso_moduli(C)) == TensND.E_nu(C)
    @test (@test_deprecated MC.extract_iso_conductivity(TensISO{3}(2.5))) == 2.5
    Cti = TensND.tens_TI(10.0, 3.0, 2.5, 12.0, 2.0, [0.0, 0.0, 1.0])
    E, ν₁, ν₂, H, Γ = TensND.arg_TI_Hoenig(inv(Cti))
    @test (@test_deprecated MC.extract_ti_moduli(Cti, (0.0, 0.0, 1.0))) == (E = E, H = H, ν₁ = ν₁, ν₂ = ν₂, Γ = Γ)
    # A tensor that is not a `TensTI` is first projected on the TI class.
    g = @test_deprecated MC.extract_ti_moduli(TensND.generic_tens(Cti), (0.0, 0.0, 1.0))
    @test g.E ≈ E && g.H ≈ H && g.Γ ≈ Γ
end
