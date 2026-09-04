using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_scheme_integration.jl — `LayeredSphere` as a first-class RVE phase.
#
#  A composite sphere has NO Hill tensor, but it does have a concentration
#  tensor, which is how it enters the mean-field schemes (as in echoes).  These
#  tests pin the three properties that make the integration correct:
#
#    1. degeneracy — a single-layer sphere must behave exactly like the
#       equivalent spherical `Ellipsoid`, in every scheme;
#    2. independence — the result must NOT depend on the (meaningless) phase
#       property declared for a heterogeneous inclusion;
#    3. consistency — `N = ⟨C:A⟩ - C₀:A` between the contribution and the
#       localization tensors.
# =============================================================================

const C_M = TensISO{3}(3 * 20.0, 2 * 8.0)
const C_I = TensISO{3}(3 * 50.0, 2 * 20.0)
const C_core = TensISO{3}(3 * 80.0, 2 * 35.0)
const C_shell = TensISO{3}(3 * 5.0, 2 * 2.0)

@testset "LayeredSphere — trait is_homogeneous_inclusion" begin
    @test is_homogeneous_inclusion(Ellipsoid(1.0, 1.0, 1.0))
    @test is_homogeneous_inclusion(Ellipsoid(2.0, 1.0, 0.5))
    @test !is_homogeneous_inclusion(LayeredSphere((1.0,), (C_I,)))
    @test !is_homogeneous_inclusion(LayeredSphere((0.5, 1.0), (C_core, C_shell)))
end

@testset "LayeredSphere — single layer reduces to the Eshelby sphere" begin
    s = LayeredSphere((1.0,), (C_I,))
    ell = Ellipsoid(1.0, 1.0, 1.0)

    A_s = strain_strain_loc(s, C_I, C_M)
    A_e = strain_strain_loc(ell, C_I, C_M)
    @test collect(TensND.get_data(A_s)) ≈ collect(TensND.get_data(A_e)) rtol = 1.0e-12

    N_s = stiffness_contribution(s, C_I, C_M)
    N_e = stiffness_contribution(ell, C_I, C_M)
    @test collect(TensND.get_data(N_s)) ≈ collect(TensND.get_data(N_e)) rtol = 1.0e-12

    CA_s = stress_strain_loc(s, C_I, C_M)
    CA_e = C_I ⊡ A_e
    @test Array(CA_s) ≈ Array(CA_e) rtol = 1.0e-12
end

@testset "LayeredSphere — N = ⟨C:A⟩ - C₀:A" begin
    for radii in ((1.0,), (0.8, 1.0), (0.4, 0.7, 1.0))
        moduli = ntuple(k -> TensISO{3}(3 * (10.0 + 20k), 2 * (4.0 + 8k)), length(radii))
        s = LayeredSphere(radii, moduli)
        A = strain_strain_loc(s, C_I, C_M)
        CA = stress_strain_loc(s, C_I, C_M)
        N = stiffness_contribution(s, C_I, C_M)
        @test Array(N) ≈ Array(CA - C_M ⊡ A) rtol = 1.0e-12
    end
end

@testset "LayeredSphere — identical layers reduce to the homogeneous sphere" begin
    s2 = LayeredSphere((0.6, 1.0), (C_I, C_I))
    ell = Ellipsoid(1.0, 1.0, 1.0)
    @test collect(TensND.get_data(strain_strain_loc(s2, C_I, C_M))) ≈
        collect(TensND.get_data(strain_strain_loc(ell, C_I, C_M))) rtol = 1.0e-10
    @test collect(TensND.get_data(stiffness_contribution(s2, C_I, C_M))) ≈
        collect(TensND.get_data(stiffness_contribution(ell, C_I, C_M))) rtol = 1.0e-10
end

@testset "Schemes — single-layer phase ≡ spherical Ellipsoid phase" begin
    r_lay = RVE()
    add_phase!(r_lay, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M); fraction = :rest)
    add_phase!(
        r_lay, :I, LayeredSphere((1.0,), (C_I,)), Dict(:C => C_I);
        fraction = 0.25
    )

    r_ell = RVE()
    add_phase!(r_ell, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M); fraction = :rest)
    add_phase!(r_ell, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_I); fraction = 0.25)

    for scheme in (MoriTanaka(), SelfConsistent(), Dilute(), DifferentialScheme())
        a = k_mu(homogenize(r_lay, scheme, :C))
        b = k_mu(homogenize(r_ell, scheme, :C))
        @test a[1] ≈ b[1] rtol = 1.0e-10
        @test a[2] ≈ b[2] rtol = 1.0e-10
    end
end

@testset "Schemes — the declared phase property is irrelevant" begin
    # Regression: the kernels used to compute `(P_i - C₀) : A` with the phase
    # property, which for a heterogeneous inclusion is meaningless — the answer
    # depended on whether one declared the core, the shell or anything else,
    # and could even come out with the wrong sign.
    s = LayeredSphere((0.8, 1.0), (C_core, C_shell))

    function eff(prop, scheme)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M); fraction = :rest)
        add_phase!(r, :I, s, Dict(:C => prop); fraction = 0.3)
        return k_mu(homogenize(r, scheme, :C))
    end

    for scheme in (MoriTanaka(), SelfConsistent(), Dilute(), DifferentialScheme())
        ref = eff(C_core, scheme)
        for prop in (C_shell, C_M, TensISO{3}(1.0, 1.0))
            got = eff(prop, scheme)
            @test got[1] ≈ ref[1] rtol = 1.0e-10
            @test got[2] ≈ ref[2] rtol = 1.0e-10
        end
    end
end

@testset "Conductivity — layered sphere in a scheme" begin
    K_M = TensISO{3}(1.0)
    s = LayeredSphere((0.9, 1.0), (TensISO{3}(1.0e-9), TensISO{3}(50.0)))

    function eff(prop)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
        add_phase!(r, :I, s, Dict(:K => prop); fraction = 0.3)
        return tr(Array(homogenize(r, MoriTanaka(), :K))) / 3
    end

    ref = eff(K_M)
    # A conductive shell around an insulating core must raise the effective
    # conductivity well above the matrix value — the old code returned exactly
    # the matrix value (zero contribution).
    @test ref > 1.5
    @test eff(TensISO{3}(50.0)) ≈ ref rtol = 1.0e-10
    @test eff(TensISO{3}(1.0e-9)) ≈ ref rtol = 1.0e-10

    # Single layer ≡ spherical Ellipsoid.
    r_lay = RVE()
    add_phase!(r_lay, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
    add_phase!(
        r_lay, :I, LayeredSphere((1.0,), (TensISO{3}(5.0),)), Dict(:K => TensISO{3}(5.0));
        fraction = 0.25
    )
    r_ell = RVE()
    add_phase!(r_ell, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
    add_phase!(
        r_ell, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => TensISO{3}(5.0));
        fraction = 0.25
    )
    @test tr(Array(homogenize(r_lay, MoriTanaka(), :K))) ≈
        tr(Array(homogenize(r_ell, MoriTanaka(), :K))) rtol = 1.0e-10
end

@testset "Schemes — core/shell bounds" begin
    # A stiff core in a compliant shell must sit between the two homogeneous
    # limits: all-shell and all-core inclusions.
    s = LayeredSphere((0.8, 1.0), (C_core, C_shell))

    function eff_layered()
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M); fraction = :rest)
        add_phase!(r, :I, s, Dict(:C => C_core); fraction = 0.3)
        return k_mu(homogenize(r, MoriTanaka(), :C))
    end
    function eff_homog(C)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M); fraction = :rest)
        add_phase!(r, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C); fraction = 0.3)
        return k_mu(homogenize(r, MoriTanaka(), :C))
    end

    k_lay, μ_lay = eff_layered()
    k_soft, μ_soft = eff_homog(C_shell)
    k_stiff, μ_stiff = eff_homog(C_core)

    @test k_soft < k_lay < k_stiff
    @test μ_soft < μ_lay < μ_stiff
end
