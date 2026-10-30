# =============================================================================
#  test_crack_near_circle.jl
#
#  An elliptical crack close to a circle:
#   0. the angular integrals 𝒞, 𝒮, ℰ in the convention of SifAniso;
#   1. the angular integrals 𝒞, 𝒮 of its COD tensor were formed as
#      (ℰ − η²𝒦)/k² and (𝒦 − ℰ)/k², which cancel as k² → 0; checked against the
#      same closed forms in 256-bit arithmetic, where the cancellation is
#      harmless;
#   2. its derivative with respect to a semi-axis was lost where the two meet
#      (the crack became a `PennyCrack`), and AD through the cubature path
#      stopped on dual integration bounds; checked against the cubature and
#      against finite differences.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

@testset "Crack near the circle — 𝒞, 𝒮, ℰ against 256-bit closed forms" begin
    CR = MeanFieldHomogenization.Cracks
    setprecision(BigFloat, 256) do
        for η in (1.0, 1 - 1.0e-12, 1 - 1.0e-8, 1 - 1.0e-4, 0.95, 0.9487, 0.5)
            v = CR._elliptic_CS(η)
            ηb = big(η)
            k² = 1 - ηb^2
            r = if iszero(k²)
                (big(π) / 4, big(π) / 4, big(π) / 2)
            else
                K, E = CR.ell_K(k²), CR.ell_E(k²)
                ((E - ηb^2 * K) / k², (K - E) / k², E)
            end
            @test maximum(abs.(big.(v) .- r) ./ abs.(r)) < 2.0e-15
        end
    end
end

@testset "Angular integrals of 𝐁 in the convention of SifAniso" begin
    # 𝒞, 𝒮, ℰ as defined by barthelemyIJSS2026 (eq-KE, eq-defCS), over ϕ with
    # the radical √(cos²ϕ + η² sin²ϕ), and as they arise on its contour
    # ξ*(φ) = η cos φ ℓ + sin φ m, ρ = √(η² cos²φ + sin²φ):
    #     ¼∫ρ dφ = ℰ,  ¼∫sin²φ/ρ dφ = 𝒞,  ¼∫η² cos²φ/ρ dφ = η²𝒮,
    # the two being exchanged by ϕ = π/2 − φ. The integrands are periodic and
    # analytic, so the trapezoidal rule over [0, 2π] converges geometrically.
    CR = MeanFieldHomogenization.Cracks
    trap(f; n = 4000) = 2π / n * sum(f(2π * k / n) for k in 0:(n - 1))
    for η in (0.3, 0.7, 0.95, 1.0)
        𝒞, 𝒮, ℰ = CR._elliptic_CS(η)
        D(ϕ) = cos(ϕ)^2 + η^2 * sin(ϕ)^2
        @test trap(ϕ -> sqrt(D(ϕ))) / 4 ≈ ℰ rtol = 1.0e-13
        @test trap(ϕ -> cos(ϕ)^2 / sqrt(D(ϕ))) / 4 ≈ 𝒞 rtol = 1.0e-13
        @test trap(ϕ -> sin(ϕ)^2 / sqrt(D(ϕ))) / 4 ≈ 𝒮 rtol = 1.0e-13
        ρ(φ) = sqrt(η^2 * cos(φ)^2 + sin(φ)^2)
        @test trap(φ -> ρ(φ)) / 4 ≈ ℰ rtol = 1.0e-13
        @test trap(φ -> sin(φ)^2 / ρ(φ)) / 4 ≈ 𝒞 rtol = 1.0e-13
        @test trap(φ -> η^2 * cos(φ)^2 / ρ(φ)) / 4 ≈ η^2 * 𝒮 rtol = 1.0e-13
    end
end

@testset "Crack near the circle — derivative with respect to a semi-axis" begin
    canon = P -> TensND.get_array(TensND.canonical(P))
    rel = (A, B) -> maximum(abs, A .- B) / maximum(abs, B)
    C_iso = TensISO{3}(3 * 30.0, 2 * 18.0)
    C_full = TensND.canonical(C_iso)
    f = b -> canon(cod_tensor(EllipticCrack(1.0, b), C_iso))
    for b0 in (1 - 1.0e-6, 0.9)
        d = ForwardDiff.derivative(f, b0)
        d_cub = ForwardDiff.derivative(
            b -> canon(cod_tensor(EllipticCrack(1.0, b), C_full; method = :nestedquadgk, abstol = 1.0e-12, reltol = 1.0e-10)), b0
        )
        h = 1.0e-8                                   # both sides stay below b = 1
        d_fd = (f(b0 + h) .- f(b0 - h)) ./ (2h)
        @test rel(d, d_cub) < 1.0e-12
        @test rel(d, d_fd) < 1.0e-6
    end
    # At the circle 𝐁 has a kink: it depends on η = b/a with a ≥ b, and the
    # roles of a and b swap there. ForwardDiff orders two duals of equal value
    # by their partials, so the semi-axis being increased is sorted as the
    # larger one: AD returns the one-sided derivative in the direction of the
    # increase, here the right-hand one, and not the left-hand one.
    d1 = ForwardDiff.derivative(f, 1.0)
    h = 1.0e-7
    @test rel(d1, (f(1 + h) .- f(1.0)) ./ h) < 1.0e-5
    @test rel(d1, (f(1.0) .- f(1 - h)) ./ h) > 1
    @test EllipticCrack(1.0, ForwardDiff.Dual(1.0, 1.0)) isa EllipticCrack{<:Any, EllipticShape}
    @test EllipticCrack(1.0, 1.0) isa EllipticCrack{<:Any, Penny}
end
