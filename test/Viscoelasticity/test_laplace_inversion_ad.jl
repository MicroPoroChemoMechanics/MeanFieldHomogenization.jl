using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

# =============================================================================
#  test_laplace_inversion_ad.jl — ForwardDiff through the inversion.
#
#  This is the requirement that ruled out `InverseLaplace.jl`, whose entry
#  points are annotated `t::AbstractFloat` (`Dual <: Real` holds, `Dual <:
#  AbstractFloat` does not) and whose Stehfest coefficients are `BigFloat`.
#  Everything here would fail against a hard-coded working type, which is why
#  the nested-Dual case is included even though nobody asks for second
#  derivatives of a relaxation function every day.
# =============================================================================

const AD_METHODS = (
    ("GaverStehfest", GaverStehfest(16), 1.0e-5),
    ("FixedTalbot", FixedTalbot(24), 1.0e-8),
    ("TalbotTrefethen", TalbotTrefethen(24), 1.0e-8),
    ("DeHoog", DeHoog(), 1.0e-6),
)

# R*(p) = θ₁ + θ₂ pτ/(1+pτ), τ = θ₃   ⇒   R(t) = θ₁ + θ₂ exp(-t/θ₃)
Rstar(θ, p) = θ[1] + θ[2] * p * θ[3] / (1 + p * θ[3])
Rtime(θ, t) = θ[1] + θ[2] * exp(-t / θ[3])

const θ0 = [2.0, 3.0, 1.5]
const t0 = 0.8

@testset "gradient wrt model parameters" begin
    want = ForwardDiff.gradient(θ -> Rtime(θ, t0), θ0)
    for (name, m, tol) in AD_METHODS
        got = ForwardDiff.gradient(θ -> inverse_carson(p -> Rstar(θ, p), t0, m), θ0)
        @test length(got) == 3
        @test maximum(abs.(got .- want) ./ abs.(want)) < tol
    end
end

@testset "derivative wrt time" begin
    want = -θ0[2] / θ0[3] * exp(-t0 / θ0[3])
    for (name, m, tol) in AD_METHODS
        got = ForwardDiff.derivative(t -> inverse_carson(p -> Rstar(θ0, p), t, m), t0)
        @test abs(got - want) / abs(want) < 10tol
    end
end

@testset "inverse_carson_rate agrees with the time derivative" begin
    want = -θ0[2] / θ0[3] * exp(-t0 / θ0[3])
    for (name, m, tol) in AD_METHODS
        got = inverse_carson_rate(
            p -> Rstar(θ0, p), t0, m; f_glassy = θ0[1] + θ0[2]
        )
        @test abs(got - want) / abs(want) < 10tol
    end
end

@testset "nested Duals — second derivative" begin
    # The test that catches a `Complex{Float64}` hard-coded anywhere in the
    # accumulation: a nested Dual cannot be converted into it.
    f(m) = ForwardDiff.derivative(
        x -> ForwardDiff.derivative(
            y -> inverse_carson(p -> Rstar([2.0, 3.0, y], p), t0, m), x
        ), θ0[3]
    )
    want = ForwardDiff.derivative(
        x -> ForwardDiff.derivative(y -> Rtime([2.0, 3.0, y], t0), x), θ0[3]
    )
    for (name, m, tol) in AD_METHODS
        @test abs(f(m) - want) / abs(want) < 100tol
    end
end

@testset "Complex{Dual} path through the contour methods" begin
    # FixedTalbot and TalbotTrefethen evaluate the transform at complex nodes,
    # so a parameter Dual arrives wrapped as `Complex{Dual}`.
    for m in (FixedTalbot(24), TalbotTrefethen(24))
        g = ForwardDiff.derivative(
            E1 -> inverse_carson(p -> Rstar([2.0, E1, 1.5], p), t0, m), 3.0
        )
        @test isapprox(g, exp(-t0 / 1.5); rtol = 1.0e-8)
    end
end

@testset "tensor-valued transform, differentiated" begin
    Cstar(θ, p) = TensISO{3}(3 * Rstar(θ, p), 2 * Rstar(θ, p) / 2)
    g = ForwardDiff.gradient(
        θ -> TensND.get_data(inverse_carson(p -> Cstar(θ, p), t0, FixedTalbot(24)))[1],
        θ0
    )
    want = 3 * ForwardDiff.gradient(θ -> Rtime(θ, t0), θ0)
    @test maximum(abs.(g .- want) ./ abs.(want)) < 1.0e-8
end

@testset "the returned type carries the seeded partials" begin
    # `@inferred` would not hold (the weight cache is a `Dict`), so the check
    # is on the value type actually produced.
    r = ForwardDiff.derivative(
        E1 -> inverse_carson(p -> Rstar([2.0, E1, 1.5], p), t0, GaverStehfest(16)), 3.0
    )
    @test r isa Float64
    dual_out = inverse_carson(
        p -> Rstar([2.0, ForwardDiff.Dual(3.0, 1.0), 1.5], p), t0, GaverStehfest(16)
    )
    @test dual_out isa ForwardDiff.Dual
    @test ForwardDiff.partials(dual_out, 1) ≈ exp(-t0 / 1.5) rtol = 1.0e-5
end
