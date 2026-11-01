# =============================================================================
#  test_one_shot.jl — one-shot schemes with a matrix:
#  Dilute, DiluteDual, MoriTanaka.
#
#  Coverage:
#   1. Single-phase RVE returns the matrix property.
#   2. Two-phase iso composite : all three schemes are between Voigt and Reuss.
#   3. MT and Dilute agree to first order in f → 0 (dilute limit consistency).
#   4. Crack RVE handled by Dilute/DiluteDual through the
#      compliance-contribution path.
#   5. Conductivity (`property = :K`) — same recipes via gradient_gradient_concentration.
#   6. ForwardDiff sensitivity through the volume fraction.
#   7. Complex moduli — eltype propagation + Im → 0 consistency.
#   8. Symbol shortcuts for all three schemes.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const ATOL_ONE = 1.0e-10
const RTOL_ONE = 1.0e-9

@testset "Dilute / DiluteDual / MoriTanaka — sanity" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    @test homogenize(rve, Dilute()) ≈ C_m
    @test homogenize(rve, DiluteDual()) ≈ C_m
    @test homogenize(rve, MoriTanaka()) ≈ C_m
end

@testset "Dilute / DiluteDual / MoriTanaka — bracketed by Voigt/Reuss" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    Cv = homogenize(rve, Voigt())
    Cr = homogenize(rve, Reuss())
    Cd = homogenize(rve, Dilute())
    Cdd = homogenize(rve, DiluteDual())
    Cmt = homogenize(rve, MoriTanaka())

    # Check the principal stiffness component is bracketed
    Vv = get_array(Cv)[1, 1, 1, 1]
    Vr = get_array(Cr)[1, 1, 1, 1]
    for sch in (Cd, Cdd, Cmt)
        Vs = get_array(sch)[1, 1, 1, 1]
        @test Vr - RTOL_ONE * abs(Vr) ≤ Vs ≤ Vv + RTOL_ONE * abs(Vv)
    end
end

@testset "Dilute / MT — agree to first order in dilute limit" begin
    C_m = TensISO{3}(30.0, 10.0)
    C_i = TensISO{3}(60.0, 20.0)
    f = 1.0e-4   # very dilute
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)

    Cd = get_array(homogenize(rve, Dilute()))
    Cmt = get_array(homogenize(rve, MoriTanaka()))
    @test maximum(abs, Cd .- Cmt) < 1.0e-6   # O(f²) error
end

@testset "Dilute / DiluteDual — crack RVE reduces stiffness" begin
    C_m = TensISO{3}(30.0, 10.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :CRACK, PennyCrack(1.0), Dict(:C => C_m); density = 0.1)

    Cd = homogenize(rve, Dilute())
    Cdd = homogenize(rve, DiluteDual())
    # Both must reduce the [3333] (normal-to-crack) component
    C33_m = get_array(C_m)[3, 3, 3, 3]
    @test get_array(Cdd)[3, 3, 3, 3] < C33_m
    @test get_array(Cd)[3, 3, 3, 3] < C33_m
    # Both finite
    @test all(isfinite, get_array(Cd))
    @test all(isfinite, get_array(Cdd))
end

@testset "Dilute / DiluteDual / MoriTanaka — conductivity" begin
    K_m = TensISO{3}(2.0)
    K_i = TensISO{3}(8.0)
    f = 0.25
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => K_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => K_i); fraction = f)

    Kv = get_array(homogenize(rve, Voigt(); property = :K))[1, 1]
    Kr = get_array(homogenize(rve, Reuss(); property = :K))[1, 1]

    for sch in (Dilute(), DiluteDual(), MoriTanaka())
        Ks = get_array(homogenize(rve, sch; property = :K))[1, 1]
        @test Kr - RTOL_ONE * abs(Kr) ≤ Ks ≤ Kv + RTOL_ONE * abs(Kv)
    end
end

@testset "Dilute / MT — ForwardDiff sensitivity to f" begin
    C_m_arr = (30.0, 10.0)
    C_i_arr = (60.0, 20.0)

    f_dilute(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(C_m_arr...)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0),
            Dict(:C => TensISO{3}(C_i_arr...)); fraction = f
        )
        get_array(homogenize(rve, Dilute()))[1, 1, 1, 1]
    end
    f_mt(f) = begin
        DT = typeof(f)
        rve = RVE(; T = DT)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(C_m_arr...)); fraction = :rest)
        add_phase!(
            rve, :I, Ellipsoid(1.0),
            Dict(:C => TensISO{3}(C_i_arr...)); fraction = f
        )
        get_array(homogenize(rve, MoriTanaka()))[1, 1, 1, 1]
    end

    df_dil = ForwardDiff.derivative(f_dilute, 0.3)
    df_mt = ForwardDiff.derivative(f_mt, 0.3)
    @test isfinite(df_dil)
    @test isfinite(df_mt)
    # Sensitivity must be positive (increasing fraction of stiffer inclusion)
    @test df_dil > 0
    @test df_mt > 0
end

@testset "Dilute / DiluteDual / MoriTanaka — Complex moduli" begin
    δ = 0.05
    C_m = TensISO{3}(30.0 + δ * im, 10.0 + 0.5δ * im)
    C_i = TensISO{3}(60.0 + δ * im, 20.0 + 0.5δ * im)
    f = 0.3
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = f)

    for sch in (Dilute(), DiluteDual(), MoriTanaka())
        Cs = homogenize(rve, sch)
        @test eltype(Cs) <: Complex
        @test all(isfinite, get_array(Cs))
    end

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
    for sch in (Dilute(), DiluteDual(), MoriTanaka())
        C_re = get_array(homogenize(rve_re, sch))
        C_0 = get_array(homogenize(rve_0, sch))
        @test maximum(abs.(real.(C_0) .- C_re)) < ATOL_ONE
        @test maximum(abs.(imag.(C_0))) < ATOL_ONE
    end
end

@testset "Dilute / DiluteDual / MoriTanaka — Symbol shortcuts" begin
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.3
    )

    @test homogenize(rve, :Dilute) ≈ homogenize(rve, Dilute())
    @test homogenize(rve, :DIL) ≈ homogenize(rve, Dilute())
    @test homogenize(rve, :DiluteDual) ≈ homogenize(rve, DiluteDual())
    @test homogenize(rve, :DILD) ≈ homogenize(rve, DiluteDual())
    @test homogenize(rve, :MoriTanaka) ≈ homogenize(rve, MoriTanaka())
    @test homogenize(rve, :MT) ≈ homogenize(rve, MoriTanaka())
end

# Exact voids (zero stiffness): DiluteDual adds their finite compliance
# contribution. It returned NaN before 0.16.3, through an infinite compliance
# times a zero concentration. Reference: a spherical void in an isotropic
# matrix, ℍ = (1/(3k₀) + 1/(4μ₀)) 𝕁 + 5(3k₀ + 4μ₀)/(2μ₀(9k₀ + 8μ₀)) 𝕂, and in
# conduction 3/(2k₀) 𝟏.
@testset "DiluteDual — exact voids" begin
    k₀, μ₀, κ₀ = 10.0, 5.0, 2.0
    C₀, K₀ = TensISO{3}(3k₀, 2μ₀), TensISO{3}(κ₀)
    H = TensISO{3}(1 / (3k₀) + 1 / (4μ₀), 5 * (3k₀ + 4μ₀) / (2μ₀ * (9k₀ + 8μ₀)))
    for f in (0.1, 0.2)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀, :K => K₀); fraction = :rest)
        add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => zero(C₀), :K => zero(K₀)); fraction = f)
        @test homogenize(rve, DiluteDual(:M), :C) ≈ inv(inv(C₀) + f * H)
        @test homogenize(rve, DiluteDual(:M), :K) ≈ inv(inv(K₀) + f * TensISO{3}(3 / (2κ₀)))
    end
end

@testset "MoriTanaka — aligned triaxial ellipsoids stay orthotropic" begin
    # The Hill tensor of a triaxial ellipsoid in an isotropic matrix is
    # orthotropic, and so is every product of the chain: the estimate, projected on
    # the major-symmetric tensors, is the nine-constant orthotropic tensor, equal to
    # the one computed on generic tensors with a numerical Hill tensor.
    C₀ = TensISO{3}(3 * 20.0, 2 * 12.0)
    C₁ = TensISO{3}(3 * 80.0, 2 * 50.0)
    mt(C; kw...) = let rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(3.0, 2.0, 1.0), Dict(:C => C₁); fraction = 0.3)
        homogenize(rve, MoriTanaka(), :C; kw...)
    end
    x = mt(C₀)
    @test x isa TensND.TensOrtho{4, Float64, 9}
    y = mt(TensND.generic_tens(C₀); method = :nestedquadgk, reltol = 1.0e-12, abstol = 1.0e-14)
    @test norm(mandel(x) - mandel(y)) / norm(mandel(y)) < 1.0e-12
end

@testset "MoriTanaka — families of different frames: the major-symmetric part, with a warning" begin
    # Two families that share no frame: 𝔑 : ⟨𝔸⟩⁻¹ is not major-symmetric
    # [benveniste1991]. The estimate is its major-symmetric part, rebuilt here from
    # the dilute tensors 𝔸ᵢ and contributions 𝔑ᵢ of each family, and so is its
    # derivative in the fraction f of the first, in closed form:
    #     d/df (𝔑 : ⟨𝔸⟩⁻¹) = 𝔑₁ : ⟨𝔸⟩⁻¹ − 𝔑 : ⟨𝔸⟩⁻¹ : (𝔸₁ − 𝕀) : ⟨𝔸⟩⁻¹,
    # with 𝔑 = Σ fᵢ 𝔑ᵢ and ⟨𝔸⟩ = (1 − Σ fᵢ) 𝕀 + Σ fᵢ 𝔸ᵢ. In Kelvin–Mandel form the
    # major-symmetric part of a tensor is the symmetric part of its matrix.
    C₀ = TensISO{3}(3 * 20.0, 2 * 12.0)
    C₁, C₂ = TensISO{3}(3 * 80.0, 2 * 50.0), TensISO{3}(3 * 5.0, 2 * 2.0)
    f₂ = 0.1
    function cell(f)
        rve = RVE(; T = typeof(f))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(3.0, 2.0, 1.0; euler_angles = (0.4, 0.9, 0.2)), Dict(:C => C₁); fraction = f)
        add_phase!(rve, :J, Ellipsoid(1.0, 1.0, 4.0; euler_angles = (1.2, 0.3, 0.8)), Dict(:C => C₂); fraction = f₂)
        return rve
    end
    km(X) = TensND.mandel(TensND.canonical_components(X))
    sympart(X) = (X + transpose(X)) / 2
    f = 0.2
    dil = concentration_tensors(cell(f), Dilute(:M), :C)
    A₁, A₂ = km(gradient_gradient_concentration(dil, :I)), km(gradient_gradient_concentration(dil, :J))
    N₁, N₂ = km(property_contribution(dil, :I)), km(property_contribution(dil, :J))
    I₆ = Matrix{Float64}(I, 6, 6)
    A_avg = (1 - f - f₂) * I₆ + f * A₁ + f₂ * A₂
    N = f * N₁ + f₂ * N₂
    raw = km(C₀) + N / A_avg
    @test norm(raw - transpose(raw)) / norm(raw) > 1.0e-4

    mt(φ) = km(homogenize(cell(φ), MoriTanaka(:M), :C))
    C = @test_logs (:warn, r"no major symmetry") mt(f)
    @test C ≈ sympart(raw) rtol = 1.0e-12
    dC = @test_logs (:warn, r"no major symmetry") ForwardDiff.derivative(mt, f)
    @test dC ≈ sympart(N₁ / A_avg - N / A_avg * (A₁ - I₆) / A_avg) rtol = 1.0e-10
end
