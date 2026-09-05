using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_crack_schemes_alv.jl — crack-aware ALV homogenization schemes:
#  Voigt / Reuss (cracks ignored), Dilute / DiluteDual / MT / Maxwell / PCW
#  (cracks add ΔC̃ = (4π/3) ε · stiffness_contribution_alv to the
#  numerator), SC / ASC (cracks iterated against the running estimate).
# =============================================================================

const _to_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

function _setup_crack_elastic(; k_M = 5.0, μ_M = 2.0, ε = 0.1, n_times = 4)
    times = collect(range(0.0, 1.0; length = n_times))
    C_M_t = TensISO{3}(3 * k_M, 2 * μ_M)
    return (;
        C_M_t, law_M = heaviside_law(C_M_t),
        crack = PennyCrack(1.0), ε, times,
    )
end

_build_alv(ctx) = let
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.law_M); fraction = :rest)
    add_phase!(
        rve, :CRACK, ctx.crack, Dict(:C => ctx.law_M);
        density = ctx.ε, symmetrize = :iso
    )
    rve
end

_build_el(ctx) = let
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.C_M_t); fraction = :rest)
    add_phase!(
        rve, :CRACK, ctx.crack, Dict(:C => ctx.C_M_t);
        density = ctx.ε, symmetrize = :iso
    )
    rve
end

@testset "Voigt / Reuss with cracks — match elastic (cracks ignored)" begin
    ctx = _setup_crack_elastic()
    n = length(ctx.times)
    for sch in (Voigt(), Reuss())
        ref = _to_mandel(homogenize(_build_el(ctx), sch, :C))
        R = homogenize_alv(_build_alv(ctx), sch, :C; times = ctx.times)
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(R[rows, rows], ref; atol = 1.0e-12)
        end
    end
end

@testset "Dilute / DiluteDual / MT / Maxwell / PCW with cracks — elastic limit" begin
    ctx = _setup_crack_elastic()
    n = length(ctx.times)
    for sch in (
            Dilute(), DiluteDual(), MoriTanaka(), Maxwell(),
            PonteCastanedaWillis(),
        )
        ref = _to_mandel(homogenize(_build_el(ctx), sch, :C))
        R = homogenize_alv(_build_alv(ctx), sch, :C; times = ctx.times)
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(R[rows, rows], ref; atol = 1.0e-10, rtol = 1.0e-10)
        end
    end
end

@testset "SC / ASC with cracks — Bristow-Budiansky-O'Connell consistency" begin
    # Three sanity checks for self-consistent ALV with cracks.
    ctx = _setup_crack_elastic()
    rve = _build_alv(ctx)

    # 1) At very low density, SC ≈ Dilute (perturbative).
    times = ctx.times
    rve_low = let
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => ctx.law_M); fraction = :rest)
        add_phase!(
            r, :CRACK, ctx.crack, Dict(:C => ctx.law_M);
            density = 0.001, symmetrize = :iso
        )
        r
    end
    R_dil_low = homogenize_alv(rve_low, Dilute(), :C; times = times)
    R_sc_low = homogenize_alv(
        rve_low, SelfConsistent(), :C; times = times,
        abstol = 1.0e-13, reltol = 1.0e-12, maxiters = 500
    )
    @test isapprox(R_dil_low[1:6, 1:6], R_sc_low[1:6, 1:6]; atol = 1.0e-3)

    # 2) SC stays stiffer than Dilute at moderate density (Dilute over-softens).
    R_dil = homogenize_alv(rve, Dilute(), :C; times = times)
    R_sc = homogenize_alv(
        rve, SelfConsistent(), :C; times = times,
        abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 500
    )
    α_dil = R_dil[1, 1] + 2 * R_dil[1, 2]
    α_sc = R_sc[1, 1] + 2 * R_sc[1, 2]
    @test α_sc > α_dil   # SC stiffer (less crack softening) than Dilute

    # 3) SC stays stiffer than ASC at moderate density.  The ECHOES
    # symmetric SC fixed point (with `strain_Stress_α = A_α·S_n` for
    # solids and `strain_Stress_c = H_c` for cracks) differs from the
    # ASC compliance-form fixed point — SC is on the matrix-stiff
    # branch and ASC on the matrix-distinguished compliance branch.
    R_asc = homogenize_alv(
        rve, AsymmetricSelfConsistent(), :C;
        times = times, abstol = 1.0e-12, maxiters = 500
    )
    α_sc_block = R_sc[1, 1] + 2 * R_sc[1, 2]
    α_asc_block = R_asc[1, 1] + 2 * R_asc[1, 2]
    @test α_sc_block ≥ α_asc_block
end

@testset "Crack stiffness contribution helper" begin
    ctx = _setup_crack_elastic()
    crack = ctx.crack
    times = ctx.times
    C_M = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(ctx.law_M, times, 6)
    Ñ = MeanFieldHomogenization.Viscoelasticity.stiffness_contribution_alv_at(crack, C_M)
    H̃ = compliance_contribution_alv(crack, ctx.law_M, times)
    # Ñ = -C̃·H̃·C̃ — round-trip identity at machine precision.
    @test isapprox(Ñ, -(C_M * H̃ * C_M); atol = 1.0e-12)
end
