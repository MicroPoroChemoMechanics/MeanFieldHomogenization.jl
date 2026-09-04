# =============================================================================
#  test_voigt_reuss.jl — Voigt and Reuss bounds on the effective property.
#
#  Coverage:
#   1. Single-phase RVE recovers the matrix property exactly.
#   2. Two-phase isotropic composite matches the closed-form Voigt /
#      harmonic-mean Reuss formulas.
#   3. Voigt ⪰ Reuss in the Loewner order on a random grid of moduli.
#   4. CrackDensity phases are ignored (volume contribution is zero in the
#      penny limit).
#   5. Conductivity (`property = :K`) follows the same rules as elasticity.
#   6. ForwardDiff.Dual gradient through the volume fraction and the moduli.
#   7. Complex moduli (frequency-domain) — eltype propagation + Im → 0
#      consistency.
#   8. Symbol shortcuts (`:Voigt`, `:V`, `:Reuss`, `:R`) match the type form.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using Random

const ATOL_VR = 1.0e-12
const RTOL_VR = 1.0e-10

@testset "Voigt / Reuss — sanity (single-phase RVE)" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    @test homogenize(rve, Voigt()) ≈ C_m
    @test homogenize(rve, Reuss()) ≈ C_m
end

@testset "Voigt / Reuss — closed form on iso 2-phase" begin
    C_m = TensISO{3}(30.0, 10.0)              # 3k=30, 2μ=10
    C_i = TensISO{3}(60.0, 20.0)
    f = 0.3
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)

    # Voigt: arithmetic mean
    Cv = homogenize(rve, Voigt())
    expected_v = TensISO{3}((1 - f) * 30.0 + f * 60.0, (1 - f) * 10.0 + f * 20.0)
    @test maximum(abs.(get_array(Cv) .- get_array(expected_v))) < ATOL_VR

    # Reuss: harmonic mean
    Cr = homogenize(rve, Reuss())
    expected_r = TensISO{3}(
        1 / ((1 - f) / 30.0 + f / 60.0),
        1 / ((1 - f) / 10.0 + f / 20.0),
    )
    @test maximum(abs.(get_array(Cr) .- get_array(expected_r))) < ATOL_VR
end

@testset "Voigt / Reuss — Voigt ⪰ Reuss on random grid" begin
    Random.seed!(2026)
    for _ in 1:8
        # Random iso phases with positive moduli
        km, μm = 1.0 + 5rand(), 0.5 + 3rand()
        ki, μi = 1.0 + 5rand(), 0.5 + 3rand()
        f = 0.05 + 0.85rand()
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3km, 2μm)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3ki, 2μi));
            fraction = f
        )

        Cv = homogenize(rve, Voigt())
        Cr = homogenize(rve, Reuss())
        eig_diff = eigvals(Symmetric(KM(Cv) .- KM(Cr)))
        @test all(e -> e > -RTOL_VR * max(1.0, maximum(abs, KM(Cv))), eig_diff)
    end
end

@testset "Voigt / Reuss — cracks are ignored" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :CRACK, PennyCrack(1.0), Dict(:C => C_m); density = 0.1)
    # No CrackDensity contribution → effective C = matrix C
    @test homogenize(rve, Voigt()) ≈ C_m
    @test homogenize(rve, Reuss()) ≈ C_m
end

@testset "Voigt / Reuss — conductivity (property=:K)" begin
    K_m = TensISO{3}(2.0)   # iso 2nd-order conductivity (eigenvalue 2)
    K_i = TensISO{3}(8.0)   # iso 2nd-order conductivity (eigenvalue 8)
    f = 0.25
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => K_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => K_i); fraction = f)

    Kv = homogenize(rve, Voigt(); property = :K)
    Kr = homogenize(rve, Reuss(); property = :K)
    # Iso K_voigt eigenvalue = (1-f)*2 + f*8 = 3.5
    # Iso K_reuss eigenvalue = 1/((1-f)/2 + f/8) = 1/(0.375 + 0.03125) ≈ 2.461
    @test get_array(Kv)[1, 1] ≈ (1 - f) * 2 + f * 8
    @test get_array(Kr)[1, 1] ≈ 1 / ((1 - f) / 2 + f / 8)
end

@testset "Voigt / Reuss — ForwardDiff sensitivity to f" begin
    C_m_arr = [30.0, 10.0]
    C_i_arr = [60.0, 20.0]

    f_voigt(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(C_m_arr...)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(C_i_arr...));
            fraction = f
        )
        get_array(homogenize(rve, Voigt()))[1, 1, 1, 1]
    end
    df = ForwardDiff.derivative(f_voigt, 0.3)
    # ∂Voigt[1111]/∂f at iso: Voigt[1111] = (1-f)·C_m[1111] + f·C_i[1111]
    expected = get_array(TensISO{3}(C_i_arr...))[1, 1, 1, 1] -
        get_array(TensISO{3}(C_m_arr...))[1, 1, 1, 1]
    @test df ≈ expected
end

@testset "Voigt / Reuss — Complex moduli (frequency-domain)" begin
    # Viscoelastic phases with small imaginary part
    δ = 0.05
    C_m = TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im)
    C_i = TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im)
    f = 0.3
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)

    Cv = homogenize(rve, Voigt())
    Cr = homogenize(rve, Reuss())
    @test eltype(Cv) <: Complex
    @test eltype(Cr) <: Complex
    @test all(isfinite, get_array(Cv))
    @test all(isfinite, get_array(Cr))

    # Im → 0 limit
    rve_re = RVE()
    add_phase!(rve_re, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve_re, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = f
    )

    rve_0 = RVE()
    add_phase!(rve_0, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0 + 0im, 10.0 + 0im)); fraction = :rest)
    add_phase!(
        rve_0, :I, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(60.0 + 0im, 20.0 + 0im)); fraction = f
    )

    Cv0 = homogenize(rve_0, Voigt())
    Cr0 = homogenize(rve_0, Reuss())
    Cv_re = homogenize(rve_re, Voigt())
    Cr_re = homogenize(rve_re, Reuss())
    @test maximum(abs.(real.(get_array(Cv0)) .- get_array(Cv_re))) < ATOL_VR
    @test maximum(abs.(imag.(get_array(Cv0)))) < ATOL_VR
    @test maximum(abs.(real.(get_array(Cr0)) .- get_array(Cr_re))) < ATOL_VR
    @test maximum(abs.(imag.(get_array(Cr0)))) < ATOL_VR
end

@testset "Voigt / Reuss — Symbol shortcuts" begin
    C_m = TensISO{3}(30.0, 10.0)
    C_i = TensISO{3}(60.0, 20.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.3)

    @test homogenize(rve, :Voigt) ≈ homogenize(rve, Voigt())
    @test homogenize(rve, :V) ≈ homogenize(rve, Voigt())
    @test homogenize(rve, :VOIGT) ≈ homogenize(rve, Voigt())
    @test homogenize(rve, :Reuss) ≈ homogenize(rve, Reuss())
    @test homogenize(rve, :R) ≈ homogenize(rve, Reuss())
    @test homogenize(rve, :REUSS) ≈ homogenize(rve, Reuss())
end
