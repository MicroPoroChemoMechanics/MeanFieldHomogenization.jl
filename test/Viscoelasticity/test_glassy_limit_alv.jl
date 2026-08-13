using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_glassy_limit_alv.jl — the glassy (instantaneous) limit of the ALV
#  pipeline.
#
#  For ANY relaxation law, the trapezoidal block `(1, 1)` is `R(t_0, t_0)`,
#  i.e. the phase's *instantaneous* (glassy) stiffness, and every Volterra
#  operation (product, inverse, divide, layered recurrence, scheme
#  fixed point) preserves that block.  Hence
#
#      [R̃^hom]_{(1,1)} = C^hom_elastic( { R_r(t_0, t_0) } )
#
#  and, equivalently, the effective creep curve starts exactly on the
#  instantaneous elastic compliance : `J^E_eff(t_0, t_0) = 1 / E^hom(t_0)`.
#
#  Unlike `test_schemes_alv.jl` / `test_crack_schemes_alv.jl` — which feed
#  *constant* (Heaviside) laws and check every diagonal block — the laws
#  here genuinely relax and genuinely age, so only the first block is
#  constrained.  The elastic reference is computed by `homogenize`, a
#  disjoint code path from `homogenize_alv`.
# =============================================================================

const _to_mandel_glassy = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

# First diagonal block of an ALV relaxation matrix.
_first_block(R::AbstractMatrix) = R[1:6, 1:6]

# Uniaxial creep compliance J^E_eff(t_i, t_0) from a relaxation matrix.
function _uniaxial_creep_glassy(R::AbstractMatrix)
    J = volterra_inverse(R; block_size = 6)
    n = size(J, 1) ÷ 6
    return [sum(J[6 * (i - 1) + 1, 6 * (j - 1) + 1] for j in 1:n) for i in 1:n]
end

# Young's modulus of an isotropic 4-tensor.
function _young(C)
    K, μ = TensND.get_data(C)[1] / 3, TensND.get_data(C)[2] / 2
    return 9 * K * μ / (3 * K + μ)
end

# ── 1. Two-phase Maxwell composite, every ellipsoidal scheme ────────────────

@testset "glassy limit — Maxwell 2-phase, all schemes" begin
    law_M = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    law_I = maxwell_iso(20.0, 8.0, 0.3, 2.0)
    f_I = 0.25
    t_0 = 0.7
    times = [t_0, 1.4, 2.1]

    # Instantaneous (glassy) stiffnesses at the first grid point.
    C_M_0 = visco_eval(law_M, t_0, t_0)
    C_I_0 = visco_eval(law_I, t_0, t_0)

    build(prop_M, prop_I) = let
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => prop_M))
        add_phase!(
            rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => prop_I);
            fraction = f_I
        )
        rve
    end

    for sch in (
            Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(),
            Maxwell(), PonteCastanedaWillis(), SelfConsistent(),
            DifferentialScheme(),
        )
        ref = _to_mandel_glassy(homogenize(build(C_M_0, C_I_0), sch, :C))
        R = homogenize_alv(build(law_M, law_I), sch, :C; times = times)
        @test isapprox(_first_block(R), ref; rtol = 1.0e-9, atol = 1.0e-10)
    end
end

# ── 2. Oriented spheroid with isotropic orientation average ────────────────

@testset "glassy limit — spheroid, symmetrize = :iso" begin
    law_M = maxwell_iso(8.0, 3.0, 0.8, 1.5)
    law_I = maxwell_iso(25.0, 11.0, 2.0, 0.4)
    t_0 = 0.25
    times = [t_0, 0.9]

    build(prop_M, prop_I) = let
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => prop_M))
        add_phase!(
            rve, :I, Spheroid(0.2), Dict(:C => prop_I);
            fraction = 0.15, symmetrize = :iso
        )
        rve
    end

    for sch in (MoriTanaka(), Dilute())
        ref = _to_mandel_glassy(
            homogenize(
                build(visco_eval(law_M, t_0, t_0), visco_eval(law_I, t_0, t_0)),
                sch, :C
            )
        )
        R = homogenize_alv(build(law_M, law_I), sch, :C; times = times)
        @test isapprox(_first_block(R), ref; rtol = 1.0e-9, atol = 1.0e-10)
    end
end

# ── 3. Cracks in a Maxwell matrix ──────────────────────────────────────────

@testset "glassy limit — penny cracks in a Maxwell matrix" begin
    law_M = maxwell_iso(5.0, 2.0, 0.7, 1.1)
    ε = 0.08
    t_0 = 0.4
    times = [t_0, 1.0, 1.6]
    C_M_0 = visco_eval(law_M, t_0, t_0)

    build(prop) = let
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => prop))
        add_phase!(
            rve, :CRACK, PennyCrack(1.0), Dict(:C => prop);
            density = ε, symmetrize = :iso
        )
        rve
    end

    for sch in (Dilute(), MoriTanaka(), Maxwell(), PonteCastanedaWillis())
        ref = _to_mandel_glassy(homogenize(build(C_M_0), sch, :C))
        R = homogenize_alv(build(law_M), sch, :C; times = times)
        @test isapprox(_first_block(R), ref; rtol = 1.0e-9, atol = 1.0e-10)
    end
end

# ── 4. Order 2 (conductivity) ──────────────────────────────────────────────

@testset "glassy limit — order 2 (conductivity)" begin
    # Isotropic order-2 Maxwell relaxation `α(t, t') = α exp(-(t-t')/τ)`.
    relax2(α, τ) = ViscoLaw(
        function (t, tp)
            t < tp && return TensISO{3}(zero(α))
            TensISO{3}(α * exp(-(t - tp) / τ))
        end, :relaxation
    )
    law_M = relax2(2.0, 0.6)
    law_I = relax2(9.0, 1.4)
    t_0 = 0.3
    times = [t_0, 1.0, 1.7]

    build(prop_M, prop_I) = let
        rve = RVE(:M)
        add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => prop_M))
        add_phase!(
            rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => prop_I);
            fraction = 0.3
        )
        rve
    end

    K_M_0 = visco_eval(law_M, t_0, t_0)
    K_I_0 = visco_eval(law_I, t_0, t_0)

    for sch in (
            Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(),
            Maxwell(), DifferentialScheme(),
        )
        ref = TensND.get_array(homogenize(build(K_M_0, K_I_0), sch, :K))
        R = homogenize_alv(build(law_M, law_I), sch, :K; times = times)
        @test isapprox(R[1:3, 1:3], ref; rtol = 1.0e-8, atol = 1.0e-9)
    end
end

# ── 5. Ageing solidifying composite — the `scripts/53` invariant ───────────
#
# Reduced version of `scripts/53_ageing_creep_solid.jl` : a Maxwell matrix,
# a pore and `N` solidifying shells, each becoming solid at its own setting
# time.  Both topologies (composite sphere / separate inclusions) are
# checked, at a loading age before and after some of the setting times.

@testset "glassy limit — ageing solidifying composite" begin
    k0, μ0 = 1.0 / (3 * (1 - 2 * 0.2)), 1.0 / (2 * 1.2)
    k1, μ1 = 5.0 / (3 * (1 - 2 * 0.3)), 5.0 / (2 * 1.3)
    finf, fp = 0.3, 0.1
    C_p = TensISO{3}(3.0e-8, 2.0e-8)
    C_1 = TensISO{3}(3 * k1, 2 * μ1)
    N, α = 4, 4.0

    setting_times = [
        (f / (finf - f))^(1 / α)
            for f in [(i + 0.5) * finf / N for i in 0:(N - 1)]
    ]

    law_M = maxwell_iso(k0, μ0, 0.2, 0.133)
    law_1 = maxwell_iso(k1, μ1, 1.0, 1.67)

    # History-dependent layer law: solid only if it had set at loading time.
    layer_law(t_set) = ViscoLaw(
        function (t, tp)
            t < tp && return zero(C_p)
            tp ≥ t_set ? law_1.eval_fun(t, tp) : C_p
        end, :relaxation
    )
    # Its glassy value at (t, t).
    layer_glassy(t, t_set) = t ≥ t_set ? C_1 : C_p

    build_whole(t; elastic) = let
        rve = RVE(:M)
        add_matrix!(
            rve, Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => elastic ? visco_eval(law_M, t, t) : law_M)
        )
        add_phase!(
            rve, :PORE, Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => elastic ? C_p : heaviside_law(C_p)); fraction = fp
        )
        for i in 1:N
            prop = elastic ? layer_glassy(t, setting_times[i]) :
                layer_law(setting_times[i])
            add_phase!(
                rve, Symbol(:INC_, i), Ellipsoid(1.0, 1.0, 1.0),
                Dict(:C => prop); fraction = finf / N
            )
        end
        rve
    end

    build_layers(t; elastic) = let
        rve = RVE(:M)
        add_matrix!(
            rve, Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => elastic ? visco_eval(law_M, t, t) : law_M)
        )
        cumulative = cumsum(vcat([fp], fill(finf / N, N)))
        radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)
        moduli = ntuple(N + 1) do k
            if k == 1
                elastic ? C_p : heaviside_law(C_p)
            elseif elastic
                layer_glassy(t, setting_times[N - k + 2])
            else
                layer_law(setting_times[N - k + 2])
            end
        end
        add_phase!(
            rve, :INCLUSION, LayeredSphere(radii, moduli),
            Dict(:C => elastic ? C_p : heaviside_law(C_p));
            fraction = fp + finf
        )
        rve
    end

    # `t_0 = 0.5` : no layer has set yet.  `t_0 = 1.5`, `2.5` : some have.
    for t_0 in (0.5, 1.5, 2.5), builder in (build_whole, build_layers)
        times = collect(range(t_0, t_0 + 2.0; length = 4))
        R = homogenize_alv(
            builder(t_0; elastic = false), MoriTanaka(), :C;
            times = times
        )
        C_el = homogenize(builder(t_0; elastic = true), MoriTanaka(), :C)

        # Block form.
        @test isapprox(
            _first_block(R), _to_mandel_glassy(C_el);
            rtol = 1.0e-9, atol = 1.0e-10
        )
        # Engineering form: the creep curve starts on 1 / E^hom(t_0).
        @test _uniaxial_creep_glassy(R)[1] ≈ 1 / _young(C_el) rtol = 1.0e-9
    end
end
