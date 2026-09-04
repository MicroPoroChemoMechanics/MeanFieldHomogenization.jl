using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_rheology_iso.jl — lifting to tensors, the ViscoLaw bridge, and
#  `homogenize_lc`.
#
#  The point of this file is the *unification*: one model object must drive the
#  Laplace-Carson route and the ageing time-domain route to the same answer.
#  The last testset is the three-route agreement, which is the strongest check
#  in the suite because the two pipelines share no code at all.
# =============================================================================

@testset "IsoRheology — BulkShear pairing" begin
    m = iso_rheology(zener_maxwell(30.0, 10.0, 1.0), zener_maxwell(10.0, 5.0, 0.5))
    C = carson_relaxation(m, 0.7)
    @test C isa TensISO{4, 3}
    α, β = TensND.get_data(C)
    @test α ≈ 3 * carson_relaxation(m.a, 0.7)
    @test β ≈ 2 * carson_relaxation(m.b, 0.7)

    # The limits, as tensors.
    @test TensND.get_data(glassy_modulus(m)) == (3 * 40.0, 2 * 15.0)
    @test TensND.get_data(equilibrium_modulus(m)) == (3 * 30.0, 2 * 10.0)
    @test !is_fluid(m)

    # J* = (R*)^{-1}, exactly, because `inv` of a TensISO is closed form.
    @test all(
        isapprox.(TensND.get_data(carson_creep(m, 0.7) ⊡ carson_relaxation(m, 0.7)), 1.0)
    )

    # A bare number in either slot is an elastic channel.
    e = iso_rheology(2500.0, zener_maxwell(10.0, 5.0, 0.5))
    @test TensND.get_data(carson_relaxation(e, 1.0))[1] ≈ 3 * 2500.0
end

@testset "IsoRheology — YoungPoisson pairing" begin
    E = zener_maxwell(30000.0, 10000.0, 1.0)
    ν = 0.25
    m = iso_rheology_E_nu(E, ν)
    C = carson_relaxation(m, 0.7)
    Ep = carson_relaxation(E, 0.7)
    @test TensND.get_data(C)[1] ≈ Ep / (1 - 2ν)
    @test TensND.get_data(C)[2] ≈ Ep / (1 + ν)

    # With a constant ν the closed-form time value is available channel-wise.
    Ct = relaxation(m, 0.7)
    @test TensND.get_data(Ct)[1] ≈ relaxation(E, 0.7) / (1 - 2ν)

    # A relaxing ν is legitimate in the Carson domain; the channels then mix,
    # so the whole tensor transform has to be inverted.
    νm = zener_maxwell(0.2, 0.15, 3.0)
    mν = iso_rheology_E_nu(E, νm)
    Cν = carson_relaxation(mν, 0.7)
    @test TensND.get_data(Cν)[1] ≈
        carson_relaxation(E, 0.7) / (1 - 2 * carson_relaxation(νm, 0.7))
    @test relaxation(mν, 0.7) isa TensISO{4, 3}
end

@testset "ViscoLaw from a model — the bridge to the ageing pipeline" begin
    m = iso_rheology(zener_maxwell(30.0, 10.0, 1.0), zener_maxwell(10.0, 5.0, 0.5))
    law = ViscoLaw(m)
    @test visco_mode(law) == :relaxation
    # Non-ageing by construction: only `t - t'` matters.
    @test TensND.get_data(law(1.7, 0.7)) == TensND.get_data(law(3.0, 2.0))
    @test TensND.get_data(law(1.0, 0.0)) == TensND.get_data(relaxation(m, 1.0))
    # Causal, and the zero keeps the symmetry class.
    z = law(0.0, 1.0)
    @test z isa TensISO{4, 3}
    @test all(iszero, TensND.get_data(z))

    # The creep mode too.
    lawJ = ViscoLaw(m; mode = :creep)
    @test visco_mode(lawJ) == :creep
    @test all(isapprox.(TensND.get_data(lawJ(1.0, 0.0)), TensND.get_data(creep(m, 1.0))))

    # A scalar model works as well, for the block_size = 1 pipeline.
    ls = ViscoLaw(zener_maxwell(2.0, 3.0, 1.0))
    @test ls(1.0, 0.0) ≈ 2 + 3 * exp(-1.0)
    @test iszero(ls(0.0, 1.0))

    @test_throws ArgumentError ViscoLaw(m; mode = :nonsense)
end

@testset "the Prony bridge: Volterra inverse == the converted chain" begin
    # `trapezoidal_matrix ∘ volterra_inverse` on a Maxwell chain must give the
    # same discrete operator as `trapezoidal_matrix` on its exact Kelvin
    # counterpart.  This ties the new algebra to the shipped ALV pipeline.
    m = zener_maxwell(2.0, 5.0, 1.0)
    k = maxwell_to_kelvin(m)
    T = collect(range(0.0, 6.0; length = 60))
    R = trapezoidal_matrix(ViscoLaw(m), T)
    J_from_inverse = volterra_inverse(R; block_size = 1)
    J_from_conversion = trapezoidal_matrix(ViscoLaw(k; mode = :creep), T)
    @test norm(J_from_inverse - J_from_conversion) / norm(J_from_conversion) < 5.0e-3
end

@testset "homogenize_lc — single p reproduces the hand-written complex route" begin
    Zm = iso_rheology(zener_maxwell(1.0, 2.0, 1.0), zener_maxwell(0.6, 1.2, 0.7))
    Zi = iso_rheology(zener_maxwell(8.0, 4.0, 0.3), zener_maxwell(5.0, 2.5, 0.4))
    f = 0.25
    function cell(p)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zm, p)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zi, p)); fraction = f
        )
        return rve
    end
    ω = 2.0
    got = homogenize_lc(cell, MoriTanaka(), :C; p = im * ω)
    want = homogenize(cell(im * ω), MoriTanaka(), :C)
    @test TensND.get_data(got) == TensND.get_data(want)

    @test_throws ArgumentError homogenize_lc(cell, MoriTanaka(), :C)
    @test_throws ArgumentError homogenize_lc(
        cell, MoriTanaka(), :C; p = im, times = [1.0]
    )
end

@testset "three routes agree on a non-ageing composite" begin
    # Route 1: the complex modulus at p = iω (the correspondence principle).
    # Route 2: `homogenize_lc` — the same, inverted back to the time domain.
    # Route 3: `homogenize_alv` — the ageing Volterra machinery applied to a
    #          non-ageing material, which shares no code with the other two.
    Zm = iso_rheology(zener_maxwell(1.0, 2.0, 1.0), zener_maxwell(0.6, 1.2, 0.7))
    Zi = iso_rheology(zener_maxwell(8.0, 4.0, 0.3), zener_maxwell(5.0, 2.5, 0.4))
    f = 0.25
    function cell(p)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zm, p)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zi, p)); fraction = f
        )
        return rve
    end

    ts = [0.05, 0.2, 1.0, 3.0, 10.0]
    μ_talbot = [TensND.get_data(C)[2] / 2 for C in homogenize_lc(cell, MoriTanaka(), :C; times = ts)]
    μ_gs = [
        TensND.get_data(C)[2] / 2
            for C in homogenize_lc(
                cell, MoriTanaka(), :C; times = ts, method = GaverStehfest(16)
            )
    ]
    μ_dh = [
        TensND.get_data(C)[2] / 2
            for C in homogenize_lc(cell, MoriTanaka(), :C; times = ts, method = DeHoog())
    ]

    # The three inversion algorithms must agree far more closely with each
    # other than any of them does with the discretized time route.
    @test maximum(abs.(μ_talbot .- μ_dh) ./ μ_talbot) < 1.0e-8
    @test maximum(abs.(μ_talbot .- μ_gs) ./ μ_talbot) < 1.0e-4

    T = vcat(0.0, exp10.(range(-3, log10(12.0); length = 220)))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => ViscoLaw(Zm)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => ViscoLaw(Zi)); fraction = f)
    Rt = homogenize_alv(rve, MoriTanaka(), :C; times = T)
    _, β = iso_params_from_blocks(Rt)
    μ_alv = (β * ones(length(T))) ./ 2

    for (i, t) in enumerate(ts)
        j = argmin(abs.(T .- t))
        # 0.5 % is the trapezoidal discretization error of the time route on a
        # 220-point grid — the dominant term, and it shrinks with the grid.
        @test abs(μ_alv[j] - μ_talbot[i]) / μ_talbot[i] < 5.0e-3
    end
end

@testset "homogenize_lc is differentiable" begin
    m_k, m_μ = zener_maxwell(1.0, 2.0, 1.0), zener_maxwell(0.6, 1.2, 0.7)
    function μ_hom(f, method)
        Zi = iso_rheology(Spring(8.0), Spring(5.0))
        Zm = iso_rheology(m_k, m_μ)
        cell(p) = begin
            rve = RVE{typeof(f)}()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zm, p)); fraction = :rest)
            add_phase!(
                rve, :I, Ellipsoid(1.0), Dict(:C => carson_relaxation(Zi, p));
                fraction = f
            )
            rve
        end
        C = homogenize_lc(cell, MoriTanaka(), :C; times = [1.0], method)
        return TensND.get_data(C[1])[2] / 2
    end

    # Against finite differences, on the accurate inversion: `FixedTalbot`
    # resolves the function to ~1e-12, so the difference quotient is clean.
    g_talbot = ForwardDiff.derivative(f -> μ_hom(f, FixedTalbot(24)), 0.25)
    fd = (μ_hom(0.2505, FixedTalbot(24)) - μ_hom(0.2495, FixedTalbot(24))) / 1.0e-3
    @test isapprox(g_talbot, fd; rtol = 1.0e-6)

    # And the real-arithmetic route — the reason `GaverStehfest` exists —
    # gives the same derivative to its own accuracy budget.  Comparing it
    # against finite differences directly would be measuring the difference
    # quotient's noise, not the gradient.
    g_gs = ForwardDiff.derivative(f -> μ_hom(f, GaverStehfest(16)), 0.25)
    @test isapprox(g_gs, g_talbot; rtol = 1.0e-3)
end
