using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_laplace_tensor.jl — inverting transforms that return tensors.
#
#  The linear methods are weighted sums, so tensor support is free *provided*
#  the accumulator is never seeded with `zero(...)`.  That is not a stylistic
#  preference: `Base.zero(::AbstractTens{4,dim,T})` returns a `TensISO`
#  whatever the input class, so a `zero`-seeded accumulator would silently
#  collapse a `TensTI` transform onto the isotropic class and drop its axis.
#  The `TensTI` test below is the regression guard for that — it passes
#  trivially in the `TensISO` case, which is why it has to be written out.
# =============================================================================

const TENSOR_METHODS = (GaverStehfest(16), FixedTalbot(24), TalbotTrefethen(24), DeHoog())

@testset "TensISO transform — matches componentwise scalar inversion" begin
    k∞, k_d, τk = 2.0, 3.0, 1.0
    μ∞, μ_d, τμ = 1.0, 4.0, 0.5
    lc(a, b, τ, p) = a + b * p * τ / (1 + p * τ)
    Cstar(p) = TensISO{3}(3 * lc(k∞, k_d, τk, p), 2 * lc(μ∞, μ_d, τμ, p))

    for m in TENSOR_METHODS, t in (0.1, 0.7, 4.0)
        C = inverse_carson(Cstar, t, m)
        @test C isa TensISO{4, 3, Float64}
        α, β = TensND.get_data(C)
        # Real, not `Complex` with a 1e-17 imaginary residue.
        @test eltype(TensND.get_data(C)) === Float64
        # Absolute, measured against the glassy scale of each channel — that
        # is what an inverse Laplace transform actually controls (see the tail
        # note in `test_laplace_inversion.jl`).
        # 5e-5 is the same fraction-of-scale gate the scalar tests use for
        # Gaver-Stehfest at N = 16; `t = 4` is eight shear relaxation times
        # out, which is where that floor actually shows.
        atol_k = (m isa GaverStehfest ? 5.0e-5 : 1.0e-8) * 3 * (k∞ + k_d)
        atol_μ = (m isa GaverStehfest ? 5.0e-5 : 1.0e-8) * 2 * (μ∞ + μ_d)
        @test isapprox(α, 3 * (k∞ + k_d * exp(-t / τk)); atol = atol_k)
        @test isapprox(β, 2 * (μ∞ + μ_d * exp(-t / τμ)); atol = atol_μ)
        # Same numbers as inverting each channel on its own.
        @test isapprox(α, inverse_carson(p -> 3 * lc(k∞, k_d, τk, p), t, m); rtol = 1.0e-12)
    end
end

@testset "TensTI transform — class and axis survive (the `zero` trap)" begin
    axis = (0.0, 0.0, 1.0)
    Cstar(p) = TensTI{4}(
        ntuple(i -> (i + 1.0) * (1 + p * 1.0 / (1 + p * 1.0)), 6)..., axis
    )
    for m in TENSOR_METHODS
        C = inverse_carson(Cstar, 0.7, m)
        @test C isa TensTI{4}
        @test TensND.axis(C) == axis
        @test eltype(TensND.get_data(C)) === Float64
        # Channel-by-channel value check against the scalar pair: each channel
        # is `E∞ + E₁ pτ/(1+pτ)` with `E∞ = E₁ = i+1` and `τ = 1`, whose
        # inverse is `(i+1)(1 + e^{-t})`.
        want(i) = (i + 1.0) * (1 + exp(-0.7))
        tol = m isa GaverStehfest ? 1.0e-5 : 1.0e-8
        for (i, d) in enumerate(TensND.get_data(C))
            @test isapprox(d, want(i); atol = tol * 2 * (i + 1.0))
        end
    end
end

@testset "6×6 Mandel matrix transform" begin
    Mstar(p) = [i == j ? (1.0 + 2.0 * p / (p + 1)) : 0.0 for i in 1:6, j in 1:6]
    for m in TENSOR_METHODS
        M = inverse_carson(Mstar, 1.0, m)
        @test M isa Matrix{Float64}
        @test size(M) == (6, 6)
        tol = m isa GaverStehfest ? 1.0e-5 : 1.0e-8
        @test isapprox(M[1, 1], 1 + 2 * exp(-1.0); atol = 3tol)
        @test iszero(M[1, 2])
    end
end

@testset "tensor grid inversion" begin
    Cstar(p) = TensISO{3}(3 * (2 + 3p / (p + 1)), 2 * (1 + 4p / (p + 2)))
    ts = [0.1, 0.5, 2.0]
    for m in TENSOR_METHODS
        Cs = inverse_carson(Cstar, ts, m)
        @test length(Cs) == 3
        @test all(C -> C isa TensISO{4, 3, Float64}, Cs)
    end
end
