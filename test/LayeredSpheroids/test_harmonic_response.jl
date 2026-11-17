using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
import SpecialFunctions

# =============================================================================
#  Legendre functions of any order and response of a layered spheroid to the
#  spheroidal harmonics of any degree and order.
#
#  References: the closed-form tables of the orders 0 to 2; the perfect
#  interface in closed form; the axial and transverse ratios b/a of the
#  transfer recurrence; the sphere as the limit of a spheroid of aspect ratio
#  one (`LayeredSphere`), with both imperfect interfaces.
# =============================================================================

const _LSH = MeanFieldHomogenization.LayeredSpheroids

@testset "Legendre functions of any order" begin
    for x in (1.05, 2.0, 7.0, 0.4im, 3.0im)
        # the closed forms of the order 1: P₁¹ = √(x²−1), Q₁¹ = √(x²−1) (arccoth x − x/(x²−1))
        p, dp, q, dq = _LSH.legendre_order_q(x, 1, 3)
        @test p[2] ≈ sqrt(x^2 - 1) && q[2] ≈ sqrt(x^2 - 1) * (acoth(x) - x / (x^2 - 1))
        # the Wronskian of the order m: P Q′ − P′ Q = (−1)^m (n+m)!/(n−m)!/(1 − x²)
        for m in 0:5
            p, dp, q, dq = _LSH.legendre_order_q(x, m, 8)
            for n in m:8
                w = (-1)^m * factorial(n + m) / factorial(n - m) / (1 - x^2)
                @test p[n + 1] * dq[n + 1] - dp[n + 1] * q[n + 1] ≈ w rtol = 1.0e-10
            end
        end
    end
    for x in (-0.7, 0.2, 0.95)
        p, dp = _LSH.legendre_order_p(x, 2, 8)
        @test p[3] ≈ 3 * (1 - x^2) && dp[3] ≈ -6x       # P₂² = 3(1 − x²)
        # Ferrers' functions are orthogonal with the norm 2(n+m)!/((2n+1)(n−m)!)
        m = 3
        xs, ws = _LSH.QuadGK.gauss(20)
        G = sum(w * (_LSH.legendre_order_p(t, m, 7)[1] * transpose(_LSH.legendre_order_p(t, m, 7)[1])) for (t, w) in zip(xs, ws))
        @test G[(m + 1):end, (m + 1):end] ≈ Diagonal([2 * factorial(n + m) / ((2n + 1) * factorial(n - m)) for n in m:7]) atol = 1.0e-9
    end
    @test all(iszero, first(_LSH.legendre_order_q(2.0, 4, 2)))
    @test all(iszero, first(_LSH.legendre_order_p(0.3, 4, 2)))
    @test_throws DomainError _LSH.legendre_order_q(2.0, -1, 3)
    @test_throws DomainError _LSH.legendre_order_p(0.3, 1, -1)
end

@testset "spheroid response — perfect interface in closed form, transfer recurrence" begin
    K₀ = TensISO{3}(1.0)
    k₁ = 7.0
    for (axis, disk) in ((1.0, 0.4), (0.3, 0.9))
        s = LayeredSpheroid((axis,), (disk,), (TensISO{3}(k₁),))
        for m in 0:3
            Z = harmonic_response(s, K₀, m, 6)
            P, dP, Q, dQ = _LSH.legendre_order_q(s.q[1], m, 6)
            # continuity of T and of the normal flux, degree by degree
            ref = [(k₁ - 1) / (dQ[n + 1] / dP[n + 1] - k₁ * Q[n + 1] / P[n + 1]) for n in max(m, 1):6]
            @test diag(Z) ≈ ref rtol = 1.0e-13
            @test norm(Z - Diagonal(diag(Z))) == 0
        end
        ba = _LSH.spheroid_ba_ratios(s, K₀)
        @test harmonic_response(s, K₀, 0, 5)[1, 1] ≈ ba[1] rtol = 1.0e-13
        @test harmonic_response(s, K₀, -1, 5)[1, 1] ≈ ba[2] rtol = 1.0e-13
    end
    @test_throws ArgumentError harmonic_response(LayeredSpheroid((1.0,), (0.4,), (TensISO{3}(2.0),)), K₀, 4, 3)
end

@testset "spheroid response — the sphere as a limit, imperfect interfaces" begin
    # f_n^m → (2/c)^n Γ(n+½)/√π y_n^m and F_n^m → (−1)^m (c/2)^{n+1} √π/Γ(n+3/2) Y_n^m as c → 0
    K₀ = TensISO{3}(1.0)
    R = 1.0
    for intf in (KapitzaInterface(0.3), SurfaceConductiveInterface(0.7)), ε in (1.0e-3, 1.0e-5)
        sphere = LayeredSphere((R,), (TensISO{3}(5.0),); interfaces = (intf,))
        spheroid = LayeredSpheroid((R * (1 + ε),), (R,), (TensISO{3}(5.0),); interfaces = (intf,))
        c = spheroid.c
        for m in 0:3, n in max(m, 1):4
            Z = harmonic_response(spheroid, K₀, m, 6)[n - max(m, 1) + 1, n - max(m, 1) + 1]
            ζ = harmonic_response(sphere, K₀, n)
            limit = ζ * (-1)^m * (2 / c)^(2n + 1) * SpecialFunctions.gamma(n + 0.5) * SpecialFunctions.gamma(n + 1.5) /
                (π * factorial(n - m) * factorial(n + m))
            @test abs(Z - limit) ≤ 10 * ε * abs(limit)
        end
    end
end

@testset "spheroid response — degrees coupled by an interface, derivatives" begin
    K₀ = TensISO{3}(1.0)
    s = LayeredSpheroid((0.5,), (1.0,), (TensISO{3}(3.0),); interfaces = (KapitzaInterface(0.4),))
    Z = harmonic_response(s, K₀, 2, 7)
    # the degrees 2…7 couple within a parity, never across
    @test abs(Z[1, 3]) > 1.0e-6
    @test norm(Z[1:2:end, 2:2:end]) ≤ 1.0e-14 * norm(Z)
    @test norm(Z[2:2:end, 1:2:end]) ≤ 1.0e-14 * norm(Z)
    # derivative with respect to the Kapitza resistance, against a centered difference
    z(ρ) = harmonic_response(LayeredSpheroid((0.5,), (1.0,), (TensISO{3}(3.0),); interfaces = (KapitzaInterface(ρ),)), K₀, 2, 5)[1, 3]
    h = 1.0e-6
    @test ForwardDiff.derivative(ρ -> real(z(ρ)), 0.4) ≈ real(z(0.4 + h) - z(0.4 - h)) / 2h rtol = 1.0e-6
end
