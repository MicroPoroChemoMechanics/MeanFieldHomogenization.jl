using Test
using MeanFieldHomogenization
import Elliptic
import ForwardDiff

@testset "Elliptic — Float64 fast path matches Elliptic.jl" begin
    for m in (0.0, 0.1, 0.5, 0.9, 0.99)
        @test ell_K(m) ≈ Elliptic.K(m)    rtol = 1.0e-12
        @test ell_E(m) ≈ Elliptic.E(m)    rtol = 1.0e-12
    end
    for (φ, m) in ((0.3, 0.5), (1.0, 0.2), (π / 4, 0.9))
        @test ell_F(φ, m) ≈ Elliptic.F(φ, m)  rtol = 1.0e-12
        @test ell_E(φ, m) ≈ Elliptic.E(φ, m)  rtol = 1.0e-12
    end
end

@testset "Elliptic — AGM path matches Float64 path (via BigFloat)" begin
    for m in (0.1, 0.5, 0.9)
        @test Float64(ell_K(BigFloat(m))) ≈ ell_K(m)  rtol = 1.0e-10
        @test Float64(ell_E(BigFloat(m))) ≈ ell_E(m)  rtol = 1.0e-10
    end
    for (φ, m) in ((0.3, 0.5), (1.0, 0.2))
        @test Float64(ell_F(BigFloat(φ), BigFloat(m))) ≈ ell_F(φ, m)  rtol = 1.0e-10
        @test Float64(ell_E(BigFloat(φ), BigFloat(m))) ≈ ell_E(φ, m)  rtol = 1.0e-10
    end
end

@testset "Elliptic — complex parameter, by the fixed AGM iteration" begin
    # A complex scalar does not compare: no stopping test, the 60 iterations.
    # Reference: the defining integrals over [0, π/2], by quadrature.
    quadgk = MeanFieldHomogenization.Core.QuadGK.quadgk
    E = MeanFieldHomogenization.Elliptic
    for m in (0.3 + 0.2im, -0.5 - 0.4im)
        K_ref = quadgk(θ -> 1 / sqrt(1 - m * sin(θ)^2), 0.0, π / 2; rtol = 1.0e-14)[1]
        E_ref = quadgk(θ -> sqrt(1 - m * sin(θ)^2), 0.0, π / 2; rtol = 1.0e-14)[1]
        @test E.ell_K(m) ≈ K_ref rtol = 1.0e-13
        @test E.ell_E(m) ≈ E_ref rtol = 1.0e-13
    end
end

@testset "Elliptic — Carlson identities (RF, RD)" begin
    # K(m) = RF(0, 1-m, 1)
    for m in (0.1, 0.5, 0.9)
        @test ell_RF(0.0, 1 - m, 1.0) ≈ ell_K(m)  rtol = 1.0e-10
        # E(m) = RF(0, 1-m, 1) - (m/3) RD(0, 1-m, 1)
        @test ell_RF(0.0, 1 - m, 1.0) - (m / 3) * ell_RD(0.0, 1 - m, 1.0) ≈ ell_E(m)  rtol = 1.0e-10
    end
    # Homogeneity of R_F: R_F(λx, λy, λz) = λ^{-1/2} R_F(x, y, z)
    for λ in (2.0, 3.0, 5.0)
        @test ell_RF(λ, 2λ, 3λ) ≈ λ^(-0.5) * ell_RF(1.0, 2.0, 3.0)  rtol = 1.0e-10
    end
end

@testset "Elliptic — dual numbers over BigFloat keep the precision of BigFloat" begin
    # The stopping tests of the AGM and of Carlson's duplication used the
    # tolerance of Float64 for any dual number, whatever its value type.
    m0, φ0 = big"0.3", big"0.6"
    K = ForwardDiff.derivative(ell_K, m0)
    @test K ≈ (ell_E(m0) - (1 - m0) * ell_K(m0)) / (2 * m0 * (1 - m0))  rtol = 1.0e-70
    @test ForwardDiff.value(ell_K(ForwardDiff.Dual(m0, one(m0)))) ≈ ell_K(m0)  rtol = 1.0e-70
    @test ForwardDiff.derivative(φ -> ell_F(φ, m0), φ0) ≈ 1 / sqrt(1 - m0 * sin(φ0)^2)  rtol = 1.0e-70
end

@testset "Elliptic — exact rational arguments take the precision they float to" begin
    # A `Rational{BigInt}` floats to `BigFloat`; the duplication stopped at the
    # tolerance of `Float64` and lost all but sixteen digits.
    x, y, z = big(1) // 1, big(2) // 1, big(3) // 1
    @test ell_RF(x, y, z) ≈ ell_RF(big"1.0", big"2.0", big"3.0")  rtol = 1.0e-70
    @test ell_RD(x, y, z) ≈ ell_RD(big"1.0", big"2.0", big"3.0")  rtol = 1.0e-70
    @test ell_RF(1, 2, 3) ≈ ell_RF(1.0, 2.0, 3.0)  rtol = 1.0e-15
end

@testset "Elliptic — ForwardDiff compatibility (Dual)" begin
    # dK/dm = (E(m) - (1-m)K(m)) / (2m(1-m))
    for m0 in (0.2, 0.5, 0.7)
        d_ad = ForwardDiff.derivative(ell_K, m0)
        d_th = (ell_E(m0) - (1 - m0) * ell_K(m0)) / (2 * m0 * (1 - m0))
        @test d_ad ≈ d_th  rtol = 1.0e-12
    end

    # dE/dm = (E(m) - K(m)) / (2m)
    for m0 in (0.2, 0.5, 0.7)
        d_ad = ForwardDiff.derivative(ell_E, m0)
        d_th = (ell_E(m0) - ell_K(m0)) / (2 * m0)
        @test d_ad ≈ d_th  rtol = 1.0e-12
    end

    # Incomplete: dF/dφ = 1/√(1 - m sin² φ)
    φ0, m0 = 0.6, 0.4
    d_ad = ForwardDiff.derivative(φ -> ell_F(φ, m0), φ0)
    d_th = 1 / sqrt(1 - m0 * sin(φ0)^2)
    @test d_ad ≈ d_th  rtol = 1.0e-6
    # dE/dφ = √(1 - m sin² φ)
    d_ad = ForwardDiff.derivative(φ -> ell_E(φ, m0), φ0)
    d_th = sqrt(1 - m0 * sin(φ0)^2)
    @test d_ad ≈ d_th  rtol = 1.0e-6
end

# ── SymPy weak extension (skipped when SymPy is not installed) ──────────────
sympy_loaded = try
    @eval using SymPyPythonCall
    true
catch
    false
end

if sympy_loaded
    @testset "Elliptic — SymPy weak extension" begin
        ext = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationSymPyPythonCallExt)
        @test ext !== nothing

        m = SymPyPythonCall.symbols("m")
        φ = SymPyPythonCall.symbols("φ")

        # Symbolic K/E dispatch to sympy.elliptic_{k,e} — the returned
        # expression must be a simple head, *not* the 60-level AGM nest
        # that would otherwise blow up the pretty-printer.
        Km = ell_K(m)
        Em = ell_E(m)
        @test Km isa SymPyPythonCall.Sym
        @test Em isa SymPyPythonCall.Sym
        @test occursin("elliptic_k", string(Km))
        @test occursin("elliptic_e", string(Em))

        Fφm = ell_F(φ, m)
        Eφm = ell_E(φ, m)
        @test occursin("elliptic_f", string(Fφm))
        @test occursin("elliptic_e", string(Eφm))

        # Mixed Sym / Number overloads
        @test occursin("elliptic_f", string(ell_F(φ, 0.5)))
        @test occursin("elliptic_f", string(ell_F(0.5, m)))
    end
else
    @info "SymPy not available in the test environment — skipping SymPy-extension tests."
end
