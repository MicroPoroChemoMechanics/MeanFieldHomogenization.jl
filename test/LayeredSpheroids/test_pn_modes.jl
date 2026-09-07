using Test
using MeanFieldHomogenization
using ForwardDiff

const LSpn = MeanFieldHomogenization.LayeredSpheroids

# =============================================================================
#  test_pn_modes.jl — the generic Papkovich–Neuber mode evaluator.
#
#  Case I already has closed forms of its own, validated against Eshelby. The
#  generic evaluator has to reproduce them exactly — same modes, same numbers —
#  which is what licenses using it for cases II and III, where no independent
#  closed form exists yet. That cross-check is this file's reason to exist.
# =============================================================================

@testset "Papkovich–Neuber mode evaluator" begin
    c, μ, ν = 0.8, 3.0, 0.27

    @testset "Jet2 is exact second-order arithmetic" begin
        # A jet must differentiate a product the way ForwardDiff would, which is
        # the whole reason it replaces hand Leibniz.
        f(v) = (2 + v[1]^2 * v[2]) * (v[3] - v[1] * v[2]^2)
        v0 = [0.3, 1.7, 2.1]
        a = LSpn.Jet2{Float64}(
            2 + v0[1]^2 * v0[2],
            (2 * v0[1] * v0[2], v0[1]^2, 0.0),
            (2 * v0[2], 0.0, 0.0, 2 * v0[1], 0.0, 0.0)
        )
        b = LSpn.Jet2{Float64}(
            v0[3] - v0[1] * v0[2]^2,
            (-v0[2]^2, -2 * v0[1] * v0[2], 1.0),
            (0.0, -2 * v0[1], 0.0, -2 * v0[2], 0.0, 0.0)
        )
        ab = a * b
        @test ab.v ≈ f(v0)
        g = ForwardDiff.gradient(f, v0)
        H = ForwardDiff.hessian(f, v0)
        @test collect(ab.d) ≈ g rtol = 1.0e-12
        @test [ab.h[1], ab.h[2], ab.h[3]] ≈ [H[1, 1], H[2, 2], H[3, 3]] rtol = 1.0e-12
        @test [ab.h[4], ab.h[5], ab.h[6]] ≈ [H[1, 2], H[1, 3], H[2, 3]] rtol = 1.0e-12
    end

    @testset "it reproduces the case-I closed forms" begin
        worst = 0.0
        for pot in (0, 3), kind in (:P, :Q)
            degs = pot == 0 ? (0:2:6) : (1:2:7)
            for n in degs, (p, q) in ((0.37, 1.6), (-0.81, 1.15), (0.05, 4.2))
                pb, qb, w = sqrt(1 - p^2), sqrt(q^2 - 1), sqrt(q^2 - p^2)
                u, t = LSpn.mode_fields(
                    LSpn.PNMode(pot, kind, n, 0, :cos), 0.0, p, q, c, μ, ν, Float64
                )
                Pv, Pd = LSpn.legendre_degrees(:P0, p, n:n)
                Rv, Rd = LSpn.legendre_degrees(kind === :P ? :P0 : :Q0, q, n:n)
                ddR = (2q * Rd[1] - n * (n + 1) * Rv[1]) / (1 - q^2)
                Uq, Up, Tq, Tp = pot == 0 ?
                    LSpn._case1_phi0_terms(Pv[1], Pd[1], Rv[1], Rd[1], ddR, p, q) :
                    LSpn._case1_phi3_terms(Pv[1], Pd[1], Rv[1], Rd[1], ddR, p, q, c, ν)
                ref = (
                    0.0, pb * Up / (2μ * c * w), qb * Uq / (2μ * c * w),
                    0.0, pb * qb * Tp / (c^2 * w^4), Tq / (c^2 * w^4),
                )
                got = (u[1], u[2], u[3], t[1], t[2], t[3])
                for k in 1:6
                    worst = max(worst, abs(got[k] - ref[k]) / max(abs(ref[k]), 1.0e-12))
                end
            end
        end
        @test worst < 1.0e-11
    end

    @testset "case-I axisymmetry comes out of the evaluator, not of an assumption" begin
        for pot in (0, 3), kind in (:P, :Q), n in (1, 2, 3, 4)
            u, t = LSpn.mode_fields(
                LSpn.PNMode(pot, kind, n, 0, :cos), 1.1, 0.4, 1.7, c, μ, ν, Float64
            )
            @test iszero(u[1])
            @test iszero(t[1])
        end
    end

    @testset "orders 1 and 2 evaluate, and depend on the azimuth" begin
        # No independent reference yet — cases II and III are what will supply
        # one. What is checked here is that the machinery runs at those orders
        # and that the azimuthal factor actually reaches the fields.
        for m in (1, 2), pot in (0, 1, 2, 3), trig in (:cos, :sin)
            n = max(m, 2)
            u1, t1 = LSpn.mode_fields(
                LSpn.PNMode(pot, :Q, n, m, trig), 0.3, 0.4, 1.7, c, μ, ν, Float64
            )
            u2, t2 = LSpn.mode_fields(
                LSpn.PNMode(pot, :Q, n, m, trig), 0.3 + 0.7, 0.4, 1.7, c, μ, ν, Float64
            )
            @test all(isfinite, u1) && all(isfinite, t1)
            @test any(k -> !isapprox(u1[k], u2[k]; atol = 1.0e-14), 1:3) ||
                any(k -> !isapprox(t1[k], t2[k]; atol = 1.0e-14), 1:3)
        end
        @test_throws ArgumentError LSpn.mode_fields(
            LSpn.PNMode(0, :Q, 3, 3, :cos), 0.0, 0.4, 1.7, c, μ, ν, Float64
        )
    end

    @testset "generic over the element type" begin
        md = LSpn.PNMode(3, :Q, 3, 1, :cos)
        f(x) = LSpn.mode_fields(md, 0.3, 0.4, x, c, μ, ν, typeof(x))[2][3]
        d = ForwardDiff.derivative(f, 1.7)
        fd = (f(1.7 + 1.0e-7) - f(1.7 - 1.0e-7)) / 2.0e-7
        @test d ≈ fd rtol = 1.0e-5
        ub, tb = LSpn.mode_fields(md, big(0.3), big(0.4), big(1.7), big(c), big(μ), big(ν), BigFloat)
        uf, tf = LSpn.mode_fields(md, 0.3, 0.4, 1.7, c, μ, ν, Float64)
        @test Float64.(collect(ub)) ≈ collect(uf) rtol = 1.0e-11
        @test Float64.(collect(tb)) ≈ collect(tf) rtol = 1.0e-11
    end
end
