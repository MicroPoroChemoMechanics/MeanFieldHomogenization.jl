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

@testset "ALV penny crack — the flat limit of a void spheroid, aging matrix" begin
    # A penny crack of density ε is the limit of void spheroids of aspect ω at
    # volume fraction (4π/3) ω ε. The Hill kernel of the spheroid combines the
    # inverses of k + 4μ/3 and μ alone, so the limit fixes the order of the
    # Volterra factors of the COD tensor, which an aging matrix makes matter.
    V = MeanFieldHomogenization.Viscoelasticity
    times = [0.0, 0.3, 0.7, 1.2, 2.0]
    matrix = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)) :
            TensISO{3}(0.0, 0.0)
    )
    void = ViscoLaw((t, tp) -> TensISO{3}(0.0, 0.0))
    ε, ω = 0.1, 1.0e-5
    cracked = RVE()
    add_phase!(cracked, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(cracked, :F, PennyCrack(1.0), Dict(:C => matrix); density = ε, symmetrize = :iso)
    porous = RVE()
    add_phase!(porous, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(porous, :V, Spheroid(ω), Dict(:C => void); fraction = 4π / 3 * ω * ε, symmetrize = :iso)
    for (a, b) in zip(
            V.iso_params_from_blocks(homogenize_alv(cracked, Dilute(:M), :C; times)),
            V.iso_params_from_blocks(homogenize_alv(porous, Dilute(:M), :C; times)),
        )
        @test a ≈ b rtol = 1.0e-4
    end
end

@testset "ALV penny cracks in an aging matrix, self-consistent — Echoes reference" begin
    # The running estimate takes the COD tensor of `_penny_cod_alv` too, in the
    # Budiansky–O'Connell branch of the self-consistent scheme. Reference: Echoes
    # `homogenize_visco(…, scheme = SC)` with a crack of aspect 1e-7, whence the
    # tolerance.
    V = MeanFieldHomogenization.Viscoelasticity
    times = [0.0, 0.4, 1.5]
    matrix = ViscoLaw(
        (t, tp) -> t >= tp ?
            TensISO{3}(6.0 * (1 + 0.5tp) * exp(-(t - tp) / 0.8), 2.0 * (1 + tp) * exp(-(t - tp) / 1.5)) :
            TensISO{3}(0.0, 0.0)
    )
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => matrix); fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => matrix); density = 0.1, symmetrize = :iso)
    α, β = V.iso_params_from_blocks(homogenize_alv(rve, SelfConsistent(), :C; times))
    lower(rows) = [i >= j ? rows[i][j] : 0.0 for i in 1:3, j in 1:3]
    @test α ≈ lower([[4.272689333671794], [-1.2841395313625679, 4.003516637323179], [-0.36990479296289147, -3.601174676535309, 4.755441449535003]]) rtol = 1.0e-6
    @test β ≈ lower([[1.7297859422284707], [-0.5474610920054662, 1.864948809282462], [-0.2589671145064161, -1.8299722030102268, 2.712448059696108]]) rtol = 1.0e-6
end
