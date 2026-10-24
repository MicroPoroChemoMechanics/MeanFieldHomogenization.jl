# =============================================================================
#  test_concentrations.jl — concentration and contribution tensors of every
#  phase under every scheme, and Levin's theorem.
#
#  Three kinds of check:
#    * identities that hold by construction for any consistent estimate —
#      Σ w 𝔸 = 𝕀, Σ w 𝔸_σε = ℂ^hom, σ = ℂ_r : ε in every homogeneous phase,
#      ℂ_ref + Σ w ℕ = ℂ^hom for any reference;
#    * closed forms written here from scalars (dilute and Mori–Tanaka spheres,
#      the exact two-phase thermal expansion, the Biot coefficient), also under
#      ForwardDiff and on symbolic numbers;
#    * the error paths of the readers.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff as FD

const _ct_requires_matrix = MeanFieldHomogenization.Schemes.requires_matrix

_ct_rel(a, b) = norm(get_array(a) - get_array(b)) / max(norm(get_array(b)), floatmin())
_ct_id(prop) = prop === :C ? tens_Id4(Val(3), Val(Float64)) : tens_Id2(Val(3), Val(Float64))
_ct_apply(P::TensND.AbstractTens{4, 3}, A) = P ⊡ A
_ct_apply(P::TensND.AbstractTens{2, 3}, A) = P ⋅ A

const _CT_1 = TensISO{3}(1.0)
# Spherical and deviatoric coefficients of an isotropic fourth-order tensor
# a 𝕁 + b 𝕂, read from its components: 𝟏 : 𝔸 : 𝟏 = 3a and 𝔸₁₂₁₂ = b / 2.
_ct_bulk(A) = (_CT_1 ⊡ A ⊡ _CT_1) / 3
_ct_shear(A) = 2 * get_array(A)[1, 2, 1, 2]

function _ct_cell(; cracks = true)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(
        rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0), :K => TensISO{3}(2.0));
        fraction = :rest
    )
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 0.4),
        Dict(:C => TensISO{3}(90.0, 40.0), :K => TensISO{3}(8.0)); fraction = 0.2
    )
    add_phase!(
        rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 1.0), :K => TensISO{3}(0.1));
        fraction = 0.1
    )
    cracks && add_phase!(
        rve, :c, PennyCrack(1.0), Dict(:C => TensISO{3}(30.0, 10.0), :K => TensISO{3}(2.0));
        density = 0.05
    )
    return rve
end

const _CT_SCHEMES = (
    Voigt(), Reuss(), Dilute(:M), DiluteDual(:M), MoriTanaka(:M), Maxwell(:M),
    PonteCastanedaWillis(:M), SelfConsistent(), AsymmetricSelfConsistent(:M),
    DifferentialScheme(:M),
)

# Iterative schemes close their averages at their own convergence tolerance;
# the differential scheme integrates its concentration tensors along with the
# medium, so its matrix is consistent to the ODE tolerance only.
_ct_tol(::Union{SelfConsistent, AsymmetricSelfConsistent}) = 1.0e-7
_ct_tol(::DifferentialScheme) = 1.0e-5
_ct_tol(_) = 1.0e-12

@testset "Concentration tensors — averages close under every scheme" begin
    for prop in (:C, :K), sch in _CT_SCHEMES
        rve = _ct_cell()
        ct = concentration_tensors(rve, sch, prop)
        P_hom = ct.effective
        tol = _ct_tol(sch)
        bound = sch isa Union{Voigt, Reuss}
        @test keys(ct) == (bound ? [:M, :I, :P] : [:M, :I, :P, :c])
        @test ct.cracks == (bound ? Symbol[] : [:c])
        @test ct.reference === (_ct_requires_matrix(sch) ? :M : nothing)

        # The two averages.
        sA = sum(ct.weights[n] * ct.strain[n] for n in keys(ct))
        sS = sum(ct.weights[n] * ct.stress[n] for n in keys(ct))
        @test _ct_rel(sA, _ct_id(prop)) < 1.0e-12
        @test _ct_rel(sS, P_hom) < tol
        # The effective tensor is the one `homogenize` returns.
        h = homogenize(rve, sch, prop)
        @test _ct_rel(P_hom, h) < (sch isa DifferentialScheme ? 1.0e-6 : 1.0e-14)
        @test _ct_rel(ct.compliance, inv(P_hom)) < 1.0e-12
        # Every homogeneous phase obeys its own law, the matrix included —
        # the real test of the matrix closures and of the differential ODE.
        for n in keys(ct)
            n in ct.cracks && continue
            P_n = ct.properties[n]
            @test _ct_rel(ct.stress[n], _ct_apply(P_n, ct.strain[n])) < tol
        end
        # A crack family carries no stress.
        for c in ct.cracks
            @test iszero(get_array(ct.stress[c]))
        end

        # Contributions, for the default reference and for an arbitrary one.
        P_M = ct.properties[:M]
        P_any = prop === :C ? TensISO{3}(50.0, 20.0) : TensISO{3}(5.0)
        contribution = prop === :C ? property_contribution : property_contribution
        dual_contribution = prop === :C ? inverse_property_contribution : inverse_property_contribution
        for (ref, P_ref) in ((:M, P_M), (P_any, P_any))
            N = sum(ct.weights[n] * contribution(ct, n; reference = ref) for n in keys(ct))
            @test _ct_rel(P_ref + N, P_hom) < tol
            H = sum(ct.weights[n] * dual_contribution(ct, n; reference = ref) for n in keys(ct))
            @test _ct_rel(inv(P_ref) + H, ct.compliance) < tol
        end
        if _ct_requires_matrix(sch)
            @test _ct_rel(contribution(ct, :I), contribution(ct, :I; reference = :M)) == 0
        end
    end
end

@testset "Concentration tensors — a phase of zero amount" begin
    # A phase declared with a zero fraction keeps a finite concentration tensor:
    # the dilute one in the matrix for the schemes built on it, the differential
    # scheme included, whose medium never leaves the matrix.
    C₀, C₁ = TensISO{3}(30.0, 10.0), TensISO{3}(90.0, 40.0)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.4), Dict(:C => C₁); fraction = 0.0)
    A_dil = gradient_gradient_concentration(concentration_tensors(rve, Dilute(:M), :C), :I)
    for sch in _CT_SCHEMES
        ct = concentration_tensors(rve, sch, :C)
        @test all(n -> all(isfinite, get_array(ct.strain[n])) && all(isfinite, get_array(ct.stress[n])), keys(ct))
        if sch isa Union{MoriTanaka, Maxwell, PonteCastanedaWillis, DifferentialScheme, SelfConsistent, AsymmetricSelfConsistent}
            @test _ct_rel(gradient_gradient_concentration(ct, :I), A_dil) < (sch isa DifferentialScheme ? 1.0e-6 : 1.0e-10)
        end
    end
end

@testset "Concentration tensors — stress-driven forms" begin
    rve = _ct_cell()
    ct = concentration_tensors(rve, MoriTanaka(:M), :C)
    S_hom = ct.compliance
    sB = sum(ct.weights[n] * flux_flux_concentration(ct, n) for n in keys(ct))
    sA = sum(ct.weights[n] * gradient_flux_concentration(ct, n) for n in keys(ct))
    @test _ct_rel(sB, _ct_id(:C)) < 1.0e-12
    @test _ct_rel(sA, S_hom) < 1.0e-12
    @test _ct_rel(gradient_gradient_concentration(ct, :I), ct.strain[:I]) == 0
    @test _ct_rel(flux_gradient_concentration(ct, :I), ct.stress[:I]) == 0

    ctK = concentration_tensors(rve, MoriTanaka(:M); property = :K)
    sB = sum(ctK.weights[n] * flux_flux_concentration(ctK, n) for n in keys(ctK))
    sA = sum(ctK.weights[n] * gradient_flux_concentration(ctK, n) for n in keys(ctK))
    @test _ct_rel(sB, _ct_id(:K)) < 1.0e-12
    @test _ct_rel(sA, ctK.compliance) < 1.0e-12
    @test _ct_rel(gradient_gradient_concentration(ctK, :I), ctK.strain[:I]) == 0
    @test _ct_rel(flux_gradient_concentration(ctK, :I), ctK.stress[:I]) == 0
    @test sprint(show, ct) == "ConcentrationTensors(M, I, P, c)"
end

# Closed forms of a sphere in an isotropic matrix, written from scalars. The
# dilute strain concentration tensor is a_k 𝕁 + a_μ 𝕂 with
#     a_k = 1 / (1 + α₀ (k₁/k₀ - 1)),   a_μ = 1 / (1 + β₀ (μ₁/μ₀ - 1)),
#     α₀ = 3k₀ / (3k₀ + 4μ₀),           β₀ = 6(k₀ + 2μ₀) / (5(3k₀ + 4μ₀)),
# and Mori–Tanaka loads it by the matrix strain, 1 / (1 - f + f a) per mode.
_ct_ak(k₀, μ₀, k₁) = 1 / (1 + 3k₀ / (3k₀ + 4μ₀) * (k₁ / k₀ - 1))
_ct_aμ(k₀, μ₀, μ₁) = 1 / (1 + 6(k₀ + 2μ₀) / (5(3k₀ + 4μ₀)) * (μ₁ / μ₀ - 1))
_ct_mt(a, f) = a / (1 - f + f * a)
_ct_aK(K₀, K₁) = 3K₀ / (2K₀ + K₁)

function _ct_spheres(k₀, μ₀, k₁, μ₁, f; K₀ = 2.0, K₁ = 7.0, T = Float64)
    rve = RVE(; T = T, distribution_shape = Ellipsoid(1.0))
    add_phase!(
        rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k₀, 2μ₀), :K => TensISO{3}(K₀));
        fraction = :rest
    )
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k₁, 2μ₁), :K => TensISO{3}(K₁));
        fraction = f
    )
    return rve
end

@testset "Concentration tensors — dilute and Mori–Tanaka spheres in closed form" begin
    k₀, μ₀, k₁, μ₁, f = 10.0, 4.0, 40.0, 25.0, 0.3
    ak, aμ = _ct_ak(k₀, μ₀, k₁), _ct_aμ(k₀, μ₀, μ₁)
    rve = _ct_spheres(k₀, μ₀, k₁, μ₁, f)

    ct = concentration_tensors(rve, Dilute(:M), :C)
    A = gradient_gradient_concentration(ct, :I)
    @test _ct_bulk(A) ≈ ak rtol = 1.0e-13
    @test _ct_shear(A) ≈ aμ rtol = 1.0e-13
    @test _ct_bulk(gradient_gradient_concentration(ct, :M)) ≈ (1 - f * ak) / (1 - f) rtol = 1.0e-13

    for sch in (MoriTanaka(:M), Maxwell(:M), PonteCastanedaWillis(:M))
        ct = concentration_tensors(rve, sch, :C)
        A = gradient_gradient_concentration(ct, :I)
        @test _ct_bulk(A) ≈ _ct_mt(ak, f) rtol = 1.0e-13
        @test _ct_shear(A) ≈ _ct_mt(aμ, f) rtol = 1.0e-13
        A₀ = gradient_gradient_concentration(ct, :M)
        @test _ct_bulk(A₀) ≈ 1 / (1 - f + f * ak) rtol = 1.0e-13
        @test _ct_shear(A₀) ≈ 1 / (1 - f + f * aμ) rtol = 1.0e-13
        # The stress tensor of the inclusion is its stiffness on its strain.
        @test _ct_bulk(flux_gradient_concentration(ct, :I)) ≈ 3k₁ * _ct_mt(ak, f) rtol = 1.0e-13
    end

    aK = _ct_aK(2.0, 7.0)
    ctK = concentration_tensors(rve, MoriTanaka(:M), :K)
    @test get_array(gradient_gradient_concentration(ctK, :I))[1, 1] ≈ _ct_mt(aK, f) rtol = 1.0e-13
    @test get_array(gradient_gradient_concentration(ctK, :M))[1, 1] ≈ 1 / (1 - f + f * aK) rtol = 1.0e-13
end

# Exact thermal expansion of a two-phase composite, from its compliance alone:
#     α^hom = α₁ + (α₂ - α₁) : (𝕊₂ - 𝕊₁)⁻¹ : (𝕊^hom - 𝕊₁).
_ct_levin_exact(α₁, α₂, S₁, S₂, S) = α₁ + (α₂ - α₁) ⊡ inv(S₂ - S₁) ⊡ (S - S₁)

@testset "Levin — exact for two phases, under every scheme" begin
    C₁, C₂ = TensISO{3}(30.0, 10.0), TensISO{3}(150.0, 60.0)
    α₁, α₂ = 1.0e-5 * _CT_1, 4.0e-5 * _CT_1
    for shape in (Ellipsoid(1.0), Ellipsoid(1.0, 1.0, 0.3))
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₁); fraction = :rest)
        add_phase!(rve, :I, shape, Dict(:C => C₂); fraction = 0.3)
        for sch in _CT_SCHEMES
            ct = concentration_tensors(rve, sch, :C)
            exact = _ct_levin_exact(α₁, α₂, inv(C₁), inv(C₂), ct.compliance)
            E = levin_free_gradient(ct, Dict(:M => α₁, :I => α₂))
            @test _ct_rel(E, exact) < max(_ct_tol(sch), 1.0e-11)
            # A free strain is a prestress -ℂ_r : ε*_r: the two forms agree.
            Σ = levin_preflux(ct, Dict(:M => -(C₁ ⊡ α₁), :I => -(C₂ ⊡ α₂)))
            @test _ct_rel(Σ, -(ct.effective ⊡ E)) < max(_ct_tol(sch), 1.0e-11)
        end
    end
    # A phase left out carries nothing; a uniform free strain is the macroscopic one.
    rve = _ct_cell(; cracks = false)
    ct = concentration_tensors(rve, MoriTanaka(:M), :C)
    @test _ct_rel(levin_free_gradient(ct, Dict(:M => α₁, :I => α₁, :P => α₁)), α₁) < 1.0e-12
    @test iszero(get_array(levin_free_gradient(ct, Dict{Symbol, typeof(α₁)}())))
end

@testset "Levin — Biot tensor of a porous solid" begin
    C_s = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
    add_phase!(rve, :P, Ellipsoid(1.0, 1.0, 0.2), Dict(:C => zero(C_s)); fraction = 0.2)
    for sch in (MoriTanaka(:S), SelfConsistent(), DifferentialScheme(:S))
        ct = concentration_tensors(rve, sch, :C)
        # A pore pressure p is the prestress -p 𝟏 of the pores: Σ* = -p b.
        b = -levin_preflux(ct, Dict(:P => -_CT_1))
        @test _ct_rel(b, biot_tensor(ct.effective, C_s)) < _ct_tol(sch)
    end
    # The pressure also fills a family of aligned cracks, which enters with its
    # density and its tensor per unit density; the single-solid formula
    # 𝟏 : (𝕀 - 𝕊_s : ℂ^hom) holds all the same.
    add_phase!(rve, :c, PennyCrack(1.0), Dict(:C => C_s); density = 0.1)
    for sch in (MoriTanaka(:S), SelfConsistent(), DifferentialScheme(:S))
        ct = concentration_tensors(rve, sch, :C)
        b = -levin_preflux(ct, Dict(:P => -_CT_1, :c => -_CT_1))
        @test _ct_rel(b, biot_tensor(ct.effective, C_s)) < _ct_tol(sch)
        b_pores = -levin_preflux(ct, Dict(:P => -_CT_1))
        @test !(_ct_rel(b_pores, b) < 1.0e-3)
    end
end

# The conduction twin: with q = -K_r · (∇T - g*_r), the free gradient of two
# phases follows from the effective resistivity alone,
#     G* = g*₁ + (g*₂ - g*₁) · (R₂ - R₁)⁻¹ · (R^hom - R₁),   R = K⁻¹.
_ct_levin_exact_K(g₁, g₂, R₁, R₂, R) = g₁ + (g₂ - g₁) ⋅ inv(R₂ - R₁) ⋅ (R - R₁)

@testset "Levin — exact for two phases in conduction, under every scheme" begin
    K₁, K₂ = TensISO{3}(2.0), TensISO{3}(9.0)
    g₁, g₂ = Tens([1.0, 0.0, 0.5]), Tens([0.0, 2.0, -1.0])
    for shape in (Ellipsoid(1.0), Ellipsoid(1.0, 1.0, 0.3))
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => K₁); fraction = :rest)
        add_phase!(rve, :I, shape, Dict(:K => K₂); fraction = 0.3)
        for sch in _CT_SCHEMES
            ct = concentration_tensors(rve, sch, :K)
            G = levin_free_gradient(ct, Dict(:M => g₁, :I => g₂))
            @test G isa TensND.AbstractTens{1, 3}
            exact = _ct_levin_exact_K(g₁, g₂, inv(K₁), inv(K₂), ct.compliance)
            @test _ct_rel(G, exact) < max(_ct_tol(sch), 1.0e-11)
            # A free gradient is the pre-flux K_r · g*_r: the two forms agree.
            Q = levin_preflux(ct, Dict(:M => K₁ ⋅ g₁, :I => K₂ ⋅ g₂))
            @test _ct_rel(Q, ct.effective ⋅ G) < max(_ct_tol(sch), 1.0e-11)
        end
    end
    # A uniform eigengradient is the macroscopic one, cracks or not.
    rve = _ct_cell()
    ct = concentration_tensors(rve, MoriTanaka(:M), :K)
    @test _ct_rel(levin_free_gradient(ct, Dict(:M => g₁, :I => g₁, :P => g₁)), g₁) < 1.0e-12
    @test iszero(get_array(levin_free_gradient(ct, Dict{Symbol, typeof(g₁)}())))
    # A pre-flux on the faces of a crack family contributes, weighted by its
    # density, exactly as the strain it carries does in elasticity.
    q = Tens([0.0, 0.0, 1.0])
    @test _ct_rel(levin_preflux(ct, Dict(:c => q)), ct.weights[:c] * (q ⋅ gradient_gradient_concentration(ct, :c))) < 1.0e-14
end

@testset "Concentration tensors — ForwardDiff against closed forms" begin
    k₀, μ₀, k₁, μ₁, f₀ = 10.0, 4.0, 40.0, 25.0, 0.3
    A_I(f) = _ct_bulk(gradient_gradient_concentration(concentration_tensors(_ct_spheres(k₀, μ₀, k₁, μ₁, f), MoriTanaka(:M), :C), :I))
    ref_I(f) = _ct_mt(_ct_ak(k₀, μ₀, k₁), f)
    @test FD.derivative(A_I, f₀) ≈ FD.derivative(ref_I, f₀) rtol = 1.0e-12
    # Nested, under distinct tags.
    @test FD.derivative(f -> FD.derivative(A_I, f), f₀) ≈ FD.derivative(f -> FD.derivative(ref_I, f), f₀) rtol = 1.0e-11

    # Through a property: the matrix closure carries the derivative too.
    A_M(k) = _ct_bulk(gradient_gradient_concentration(concentration_tensors(_ct_spheres(k₀, μ₀, k, μ₁, f₀), MoriTanaka(:M), :C), :M))
    ref_M(k) = 1 / (1 - f₀ + f₀ * _ct_ak(k₀, μ₀, k))
    @test FD.derivative(A_M, k₁) ≈ FD.derivative(ref_M, k₁) rtol = 1.0e-12

    # Levin's thermal expansion through the Mori–Tanaka bulk modulus,
    #     α = α₁ + (α₂ - α₁) (1/k - 1/k₀) / (1/k₁ - 1/k₀).
    α₀, α₁ = 1.0e-5, 4.0e-5
    function α_levin(f)
        ct = concentration_tensors(_ct_spheres(k₀, μ₀, k₁, μ₁, f), MoriTanaka(:M), :C)
        return tr(get_array(levin_free_gradient(ct, Dict(:M => α₀ * _CT_1, :I => α₁ * _CT_1)))) / 3
    end
    function α_ref(f)
        k = k₀ + f * (k₁ - k₀) * _ct_mt(_ct_ak(k₀, μ₀, k₁), f)
        return α₀ + (α₁ - α₀) * (1 / k - 1 / k₀) / (1 / k₁ - 1 / k₀)
    end
    @test α_levin(f₀) ≈ α_ref(f₀) rtol = 1.0e-12
    @test FD.derivative(α_levin, f₀) ≈ FD.derivative(α_ref, f₀) rtol = 1.0e-11

    # Conduction: the gradient concentration of the spheres, a / (1 - f + f a)
    # with a = 3K₀ / (2K₀ + K₁), and the free gradient through the Mori–Tanaka
    # conductivity, nested derivatives included.
    K₀, K₁, g₀, g₁ = 2.0, 7.0, 1.0, 3.0
    e₁ = Tens([1.0, 0.0, 0.0])
    A_K(f) = get_array(gradient_gradient_concentration(concentration_tensors(_ct_spheres(k₀, μ₀, k₁, μ₁, f), MoriTanaka(:M), :K), :I))[1, 1]
    ref_K(f) = _ct_mt(_ct_aK(K₀, K₁), f)
    @test FD.derivative(A_K, f₀) ≈ FD.derivative(ref_K, f₀) rtol = 1.0e-12
    @test FD.derivative(f -> FD.derivative(A_K, f), f₀) ≈ FD.derivative(f -> FD.derivative(ref_K, f), f₀) rtol = 1.0e-11
    function G_levin(f)
        ct = concentration_tensors(_ct_spheres(k₀, μ₀, k₁, μ₁, f), MoriTanaka(:M), :K)
        return get_array(levin_free_gradient(ct, Dict(:M => g₀ * e₁, :I => g₁ * e₁)))[1]
    end
    function G_ref(f)
        K = K₀ + f * (K₁ - K₀) * _ct_mt(_ct_aK(K₀, K₁), f)
        return g₀ + (g₁ - g₀) * (1 / K - 1 / K₀) / (1 / K₁ - 1 / K₀)
    end
    @test G_levin(f₀) ≈ G_ref(f₀) rtol = 1.0e-12
    @test FD.derivative(G_levin, f₀) ≈ FD.derivative(G_ref, f₀) rtol = 1.0e-11
end

const _CT_HAS_SYMPY = try
    @eval import SymPyPythonCall
    true
catch
    false
end

if _CT_HAS_SYMPY
    # `@eval` defers the expansion of `SymPyPythonCall.@syms` until the conditional
    # import above has run.
    @eval @testset "Concentration tensors — SymPy closed forms" begin
        SymPyPythonCall.@syms k0::positive μ0::positive k1::positive μ1::positive f::positive
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k0, 2μ0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k1, 2μ1)); fraction = f)
        ak, aμ = _ct_ak(k0, μ0, k1), _ct_aμ(k0, μ0, μ1)
        for sch in (MoriTanaka(:M), Maxwell(:M))
            ct = concentration_tensors(rve, sch, :C)
            a, b = get_data(gradient_gradient_concentration(ct, :I))
            @test SymPyPythonCall.simplify(a - _ct_mt(ak, f)) == 0
            @test SymPyPythonCall.simplify(b - _ct_mt(aμ, f)) == 0
        end
        ct = concentration_tensors(rve, Dilute(:M), :C)
        a, _ = get_data(gradient_gradient_concentration(ct, :I))
        @test SymPyPythonCall.simplify(a - ak) == 0

        # Levin on symbols: the two-phase thermal expansion through the
        # Mori–Tanaka bulk modulus.
        SymPyPythonCall.@syms α0::positive α1::positive
        ct = concentration_tensors(rve, MoriTanaka(:M), :C)
        one2 = TensISO{3}(SymPyPythonCall.Sym(1))
        E = levin_free_gradient(ct, Dict(:M => α0 * one2, :I => α1 * one2))
        k_hom = k0 + f * (k1 - k0) * _ct_mt(ak, f)
        α_ref = α0 + (α1 - α0) * (1 / k_hom - 1 / k0) / (1 / k1 - 1 / k0)
        @test SymPyPythonCall.simplify(get_array(E)[1, 1] - α_ref) == 0
        @test SymPyPythonCall.simplify(get_array(E)[1, 2]) == 0
        # Biot coefficient of the porous limit, b = 1 - k^hom / k₀.
        rve_p = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_p, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k0, 2μ0)); fraction = :rest)
        add_phase!(rve_p, :P, Ellipsoid(1.0), Dict(:C => zero(TensISO{3}(3k0, 2μ0))); fraction = f)
        ct = concentration_tensors(rve_p, MoriTanaka(:M), :C)
        b = -levin_preflux(ct, Dict(:P => -one2))
        k_p = k0 + f * (0 - k0) * _ct_mt(_ct_ak(k0, μ0, 0), f)
        @test SymPyPythonCall.simplify(get_array(b)[1, 1] - (1 - k_p / k0)) == 0

        # Conduction: the gradient concentration and the free gradient of the
        # Mori–Tanaka estimate on spheres, in closed form.
        SymPyPythonCall.@syms K0::positive K1::positive g0::real g1::real
        rve_K = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_K, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(K0)); fraction = :rest)
        add_phase!(rve_K, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(K1)); fraction = f)
        ct = concentration_tensors(rve_K, MoriTanaka(:M), :K)
        aK = _ct_aK(K0, K1)
        @test SymPyPythonCall.simplify(get_array(gradient_gradient_concentration(ct, :I))[1, 1] - _ct_mt(aK, f)) == 0
        e = Tens(SymPyPythonCall.Sym[1, 0, 0])
        G = levin_free_gradient(ct, Dict(:M => g0 * e, :I => g1 * e))
        K_hom = K0 + f * (K1 - K0) * _ct_mt(aK, f)
        @test SymPyPythonCall.simplify(get_array(G)[1] - (g0 + (g1 - g0) * (1 / K_hom - 1 / K0) / (1 / K1 - 1 / K0))) == 0
        @test SymPyPythonCall.simplify(get_array(G)[2]) == 0
    end
end

const _CT_HAS_SYMBOLICS = try
    @eval import Symbolics
    true
catch
    false
end

if _CT_HAS_SYMBOLICS
    # `@eval` defers the expansion of `Symbolics.@variables` until the
    # conditional import above has run.
    @eval @testset "Concentration tensors — Symbolics.jl closed forms" begin
        Symbolics.@variables k0 μ0 k1 μ1 f
        rve = RVE(; T = Symbolics.Num, distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k0, 2μ0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k1, 2μ1)); fraction = f)
        ak, aμ = _ct_ak(k0, μ0, k1), _ct_aμ(k0, μ0, μ1)
        point = Dict(k0 => 10.0, μ0 => 4.0, k1 => 40.0, μ1 => 25.0, f => 0.3)
        _num(x) = Symbolics.value(Symbolics.substitute(x, point))
        for sch in (Dilute(:M), MoriTanaka(:M), Maxwell(:M))
            ct = concentration_tensors(rve, sch, :C)
            a, b = get_data(gradient_gradient_concentration(ct, :I))
            @test a isa Symbolics.Num
            ref_a, ref_b = sch isa Dilute ? (ak, aμ) : (_ct_mt(ak, f), _ct_mt(aμ, f))
            @test _num(a) ≈ _num(ref_a) rtol = 1.0e-13
            @test _num(b) ≈ _num(ref_b) rtol = 1.0e-13
            N = property_contribution(ct, :I)
            @test _num(get_data(N)[1]) ≈ _num(3 * (k1 - k0) * ref_a) rtol = 1.0e-13
        end
        # Conduction, through the same schemes.
        Symbolics.@variables K0 K1
        rve_K = RVE(; T = Symbolics.Num, distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_K, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(K0)); fraction = :rest)
        add_phase!(rve_K, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(K1)); fraction = f)
        point_K = Dict(K0 => 2.0, K1 => 7.0, f => 0.3)
        _numK(x) = Symbolics.value(Symbolics.substitute(x, point_K))
        aK = _ct_aK(K0, K1)
        for sch in (Dilute(:M), MoriTanaka(:M), Maxwell(:M))
            ct = concentration_tensors(rve_K, sch, :K)
            a = get_array(gradient_gradient_concentration(ct, :I))[1, 1]
            ref = sch isa Dilute ? aK : _ct_mt(aK, f)
            @test _numK(a) ≈ _numK(ref) rtol = 1.0e-13
            N = get_array(property_contribution(ct, :I))[1, 1]
            @test _numK(N) ≈ _numK((K1 - K0) * ref) rtol = 1.0e-13
        end
    end
end

@testset "Concentration tensors — laminates and particle assemblies" begin
    C₁, C₂ = TensISO{3}(30.0, 10.0), TensISO{3}(150.0, 60.0)
    for itf in (PerfectInterface(), SpringInterface(; sn = 1.0e-3, st = 2.0e-3), MembraneInterface(0.5, 0.3))
        lam = Laminate(; normal = (0, 0, 1))
        add_layer!(lam, :A, Dict(:C => C₁, :K => TensISO{3}(1.0)); thickness = 0.3, interface = itf)
        add_layer!(lam, :B, Dict(:C => C₂, :K => TensISO{3}(5.0)); thickness = 0.7)
        ct = concentration_tensors(lam, Laminated(), :C)
        @test keys(ct) == [:A, :B]
        @test ct.reference === nothing
        @test _ct_rel(ct.effective, homogenize(lam, Laminated(), :C)) < 1.0e-14
        for n in keys(ct)
            @test _ct_rel(ct.stress[n], ct.properties[n] ⊡ ct.strain[n]) < 1.0e-12
        end
        # A spring interface carries part of E in its opening, so the strain
        # average over the layers no longer closes; a membrane carries part of
        # Σ, so the stress average no longer does.
        sA = sum(ct.weights[n] * ct.strain[n] for n in keys(ct))
        sS = sum(ct.weights[n] * ct.stress[n] for n in keys(ct))
        @test (_ct_rel(sA, _ct_id(:C)) < 1.0e-12) == !(itf isa SpringInterface)
        @test (_ct_rel(sS, ct.effective) < 1.0e-12) == !(itf isa MembraneInterface)
        if itf isa PerfectInterface
            ctK = concentration_tensors(lam, Laminated(), :K)
            @test _ct_rel(sum(ctK.weights[n] * ctK.stress[n] for n in keys(ctK)), ctK.effective) < 1.0e-12
            @test _ct_rel(sum(ctK.weights[n] * ctK.strain[n] for n in keys(ctK)), _ct_id(:K)) < 1.0e-12
        end
    end
    # The same in conduction: a Kapitza resistance carries part of the
    # macroscopic gradient in its temperature jump, a conducting surface part of
    # the macroscopic flux.
    for itf in (PerfectInterface(), KapitzaInterface(0.1), SurfaceConductiveInterface(0.2))
        lam = Laminate(; normal = (0, 0, 1))
        add_layer!(lam, :A, Dict(:K => TensISO{3}(1.0)); thickness = 0.3, interface = itf)
        add_layer!(lam, :B, Dict(:K => TensISO{3}(5.0)); thickness = 0.7)
        ct = concentration_tensors(lam, Laminated(), :K)
        @test _ct_rel(ct.effective, homogenize(lam, Laminated(), :K)) < 1.0e-14
        for n in keys(ct)
            @test _ct_rel(ct.stress[n], ct.properties[n] ⋅ ct.strain[n]) < 1.0e-12
        end
        sA = sum(ct.weights[n] * ct.strain[n] for n in keys(ct))
        sS = sum(ct.weights[n] * ct.stress[n] for n in keys(ct))
        @test (_ct_rel(sA, _ct_id(:K)) < 1.0e-12) == !(itf isa KapitzaInterface)
        @test (_ct_rel(sS, ct.effective) < 1.0e-12) == !(itf isa SurfaceConductiveInterface)
    end
    # Levin on a laminate: exact for two phases, interfaces aside.
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => C₁); thickness = 0.3)
    add_layer!(lam, :B, Dict(:C => C₂); thickness = 0.7)
    ct = concentration_tensors(lam, Laminated(), :C)
    α₁, α₂ = 1.0e-5 * _CT_1, 4.0e-5 * _CT_1
    exact = _ct_levin_exact(α₁, α₂, inv(C₁), inv(C₂), ct.compliance)
    @test _ct_rel(levin_free_gradient(ct, Dict(:A => α₁, :B => α₂)), exact) < 1.0e-11

    Cm, Ci = TensISO{3}(3.0, 0.8), TensISO{3}(30.0, 12.0)
    for kind in (:sc, :bcc), sch in (ClusterModel(), EquivalentInclusion())
        asm = cubic_lattice(kind, Dict(:C => Cm), Dict(:C => Ci); fraction = 0.25, cutoff = 1.5)
        ct = concentration_tensors(asm, sch, :C)
        @test first(keys(ct)) === :matrix
        @test _ct_rel(ct.effective, homogenize(asm, sch, :C)) < 1.0e-14
        @test _ct_rel(sum(ct.weights[n] * ct.strain[n] for n in keys(ct)), _ct_id(:C)) < 1.0e-12
        @test _ct_rel(sum(ct.weights[n] * ct.stress[n] for n in keys(ct)), ct.effective) < 1.0e-12
        @test _ct_rel(Cm + sum(ct.weights[n] * property_contribution(ct, n) for n in keys(ct)), ct.effective) < 1.0e-12
    end
    for kind in (:sc, :bcc), sch in (ClusterModel(), EquivalentInclusion())
        asm = cubic_lattice(kind, Dict(:K => TensISO{3}(1.0)), Dict(:K => TensISO{3}(20.0)); fraction = 0.25, cutoff = 2.0)
        ct = concentration_tensors(asm, sch, :K)
        @test _ct_rel(ct.effective, homogenize(asm, sch, :K)) < 1.0e-14
        @test _ct_rel(sum(ct.weights[n] * ct.strain[n] for n in keys(ct)), _ct_id(:K)) < 1.0e-12
        @test _ct_rel(sum(ct.weights[n] * ct.stress[n] for n in keys(ct)), ct.effective) < 1.0e-12
        @test _ct_rel(TensISO{3}(1.0) + sum(ct.weights[n] * property_contribution(ct, n) for n in keys(ct)), ct.effective) < 1.0e-12
    end
    @test keys(concentration_tensors(cubic_lattice(:sc, Dict(:C => Cm), Dict(:C => Ci); fraction = 0.25), ClusterModel(), :C)) ==
        [:matrix, :family_1]
    # One sphere in a spherical SVE: the equivalent inclusion method is
    # Mori–Tanaka exactly, concentration tensors included.
    asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(3.0)))
    add_matrix!(asm, Dict(:C => Cm))
    add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(1.0), Dict(:C => Ci))
    f = particle_volume_fraction(asm, :p1)
    ct = concentration_tensors(asm, EquivalentInclusion(), :C)
    km = get_data(Cm)[1] / 3, get_data(Cm)[2] / 2
    ki = get_data(Ci)[1] / 3, get_data(Ci)[2] / 2
    @test _ct_bulk(gradient_gradient_concentration(ct, :p1)) ≈ _ct_mt(_ct_ak(km[1], km[2], ki[1]), f) rtol = 1.0e-12
    @test _ct_shear(gradient_gradient_concentration(ct, :p1)) ≈ _ct_mt(_ct_aμ(km[1], km[2], ki[2]), f) rtol = 1.0e-12
end

@testset "Concentration tensors — error paths" begin
    rve = _ct_cell()
    ct = concentration_tensors(rve, SelfConsistent(), :C)
    @test_throws ArgumentError property_contribution(ct, :I)          # no reference
    @test_throws ArgumentError inverse_property_contribution(ct, :I)
    @test_throws ArgumentError property_contribution(ct, :I; reference = :c)  # a crack
    @test_throws ArgumentError gradient_gradient_concentration(ct, :nope)
    @test_throws ArgumentError levin_free_gradient(ct, Dict(:c => _CT_1))   # no material
    @test levin_preflux(ct, Dict(:c => -_CT_1)) isa TensND.AbstractTens{2, 3}
    @test_throws ArgumentError levin_preflux(ct, Dict(:nope => _CT_1))
    # One reader serves both orders; the names of one physics refuse the other.
    ctK = concentration_tensors(rve, SelfConsistent(), :K)
    @test gradient_gradient_concentration(ctK, :I) isa TensND.AbstractTens{2, 3}
    @test_throws ArgumentError strain_strain_concentration(ctK, :I)
    @test_throws ArgumentError conductivity_contribution(ct, :I; reference = :M)
    @test_throws MethodError levin_free_gradient(ctK, Dict(:I => _CT_1))
    @test_throws MethodError levin_free_gradient(ct, Dict(:I => Tens([1.0, 0.0, 0.0])))
    @test_throws ArgumentError levin_free_gradient(ctK, Dict(:c => Tens([1.0, 0.0, 0.0])))
    @test_throws "levin_preflux" levin_free_gradient(ctK, Dict(:c => Tens([1.0, 0.0, 0.0])))
    @test_throws "levin_preflux" levin_free_gradient(ct, Dict(:c => _CT_1))
    # A scheme the cell does not support.
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => TensISO{3}(30.0, 10.0)); thickness = 0.3)
    add_layer!(lam, :B, Dict(:C => TensISO{3}(90.0, 40.0)); thickness = 0.7)
    @test_throws ArgumentError concentration_tensors(lam, MoriTanaka(:A), :C)
end

@testset "Differential scheme — fully anisotropic conduction state" begin
    # Regression: the conductivity state of the differential scheme was laid
    # out on nine components and rebuilt by `reshape`, so as soon as the
    # estimate lost every symmetry — two tilted families of spheroids — the
    # ODE state and its reconstruction disagreed and `homogenize` threw a
    # `DimensionMismatch`. Continuity against the aligned case, which never
    # leaves the transversely isotropic state, checks the full one.
    function cell(θ)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0, 1.0, 0.4; euler_angles = (0.3, θ, 0.0)),
            Dict(:K => TensISO{3}(8.0)); fraction = 0.2
        )
        add_phase!(
            rve, :J, Ellipsoid(1.0, 1.0, 0.4; euler_angles = (1.0, 2θ, 0.0)),
            Dict(:K => TensISO{3}(8.0)); fraction = 0.1
        )
        return rve
    end
    K_tilted = homogenize(cell(0.7), DifferentialScheme(:M), :K)
    k = get_array(K_tilted)
    @test k ≈ k' rtol = 1.0e-12
    @test isposdef(Symmetric(Matrix(k)))
    @test _ct_rel(homogenize(cell(1.0e-7), DifferentialScheme(:M), :K), homogenize(cell(0.0), DifferentialScheme(:M), :K)) < 1.0e-6
    ct = concentration_tensors(cell(0.7), DifferentialScheme(:M), :K)
    @test _ct_rel(ct.effective, K_tilted) < 1.0e-6
    # With a crack family as well: the scheme probes the crack against the
    # starting medium, isotropic in value but stored dense, where the thermal
    # COD of a circular crack took the root of a discriminant rounded below zero.
    rve_c = cell(0.7)
    add_phase!(rve_c, :c, PennyCrack(1.0), Dict(:K => TensISO{3}(2.0)); density = 0.05)
    K_c = homogenize(rve_c, DifferentialScheme(:M), :K)
    @test isposdef(Symmetric(Matrix(get_array(K_c))))
    ct_c = concentration_tensors(rve_c, DifferentialScheme(:M), :K)
    @test _ct_rel(ct_c.effective, K_c) < 1.0e-6
    @test _ct_rel(sum(ct_c.weights[n] * ct_c.strain[n] for n in keys(ct_c)), _ct_id(:K)) < 1.0e-12

    # The rates shared by the effective ODE and the concentration one are
    # inferred, on plain and on dual numbers.
    Sch = MeanFieldHomogenization.Schemes
    rve = cell(0.7)
    paths = Sch._resolve_paths(DifferentialScheme(:M).trajectory, rve, 100, :M)
    params, _ = Sch._diff_setup(rve, paths, :K, :M, TensISO{3}(2.0), :stiffness)
    @test @inferred(Sch._diff_rates(params, 0.5, Float64)) isa NTuple{2, Vector{Float64}}
    D = FD.Dual{Nothing, Float64, 1}
    @test @inferred(Sch._diff_rates(params, 0.5, D)) isa NTuple{2, Vector{D}}
    @test @inferred(Sch._diff_unit_rates(params, 0.5, D)) isa Vector{D}
    @test Sch._diff_unit_rates(params, 0.5, Float64) .* [params.targets[n] for n in vcat(params.solid_names, params.crack_names)] ≈
        vcat(Sch._diff_rates(params, 0.5, Float64)...)
end

@testset "Differential scheme — the compliance form gives the same tensors" begin
    # The two forms integrate the same process on 𝕊 or on ℂ; their concentration
    # tensors agree to the tolerance of the integration, cracks included.
    rve = _ct_cell()
    for prop in (:C, :K)
        dual = DifferentialScheme(:M; formulation = :compliance)
        ct_s = concentration_tensors(rve, DifferentialScheme(:M), prop)
        ct_c = concentration_tensors(rve, dual, prop)
        @test _ct_rel(ct_c.effective, homogenize(rve, dual, prop)) < 1.0e-6
        @test _ct_rel(ct_c.effective, ct_s.effective) < 1.0e-5
        for n in keys(ct_s)
            @test _ct_rel(ct_c.strain[n], ct_s.strain[n]) < 1.0e-5
        end
        @test _ct_rel(sum(ct_c.weights[n] * ct_c.stress[n] for n in keys(ct_c)), ct_c.effective) < 1.0e-12
    end
end

@testset "No trace of the former terminology" begin
    # The package speaks of concentration tensors; no name, exported or not,
    # may carry the former term.
    function walk(m::Module, seen = Set{Module}())
        m in seen && return Symbol[]
        push!(seen, m)
        out = Symbol[]
        for n in names(m; all = true)
            occursin(r"locali[sz]"i, string(n)) && push!(out, n)
            isdefined(m, n) || continue
            v = getfield(m, n)
            v isa Module && parentmodule(v) === m && v !== m && append!(out, walk(v, seen))
        end
        return out
    end
    @test isempty(walk(MeanFieldHomogenization))
end
