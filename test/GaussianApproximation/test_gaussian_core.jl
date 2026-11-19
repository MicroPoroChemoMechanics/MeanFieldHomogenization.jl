# =============================================================================
#  test_gaussian_core.jl — the pieces of the Gaussian approximation that do not
#  depend on a physics.
#
#   1. The grid: nodes, sizes, refusals, the memory estimate.
#   2. Maz'ya's quasi-interpolation: the saturation level against the published
#      coefficients, second order above it.
#   3. The FFT product against the plain double sum, in 2D and 3D.
#   4. The solve: against a dense matrix; its derivatives (single and nested
#      duals) against ForwardDiff through the dense solve; its refusals.
# =============================================================================

using Test
using MeanFieldHomogenization
using LinearAlgebra
using StaticArrays
using ForwardDiff

const GC = MeanFieldHomogenization.GaussianApproximation

@testset "grid" begin
    g = GaussianGrid((0.0, -1.0), (2.0, 0.0), 8; H = 2.0)
    @test g.dims == (8, 4) && g.h == 0.25
    @test GC.node(g, CartesianIndex(1, 1)) == SVector(0.125, -0.875)
    @test length(collect(GC.nodes(g))) == 32
    @test GaussianGrid((0, 0), (1, 1), 4).h == 0.25             # integers are promoted
    @test occursin("8×4", sprint(show, g))
    @test_throws DimensionMismatch GaussianGrid((0.0, 0.0), (1.0,), 4)
    @test_throws ArgumentError GaussianGrid((0.0, 0.0), (1.0, 1.0), 0)
    @test_throws ArgumentError GaussianGrid((0.0, 0.0), (1.0, -1.0), 4)
    @test_throws ArgumentError GaussianGrid((0.0,), (1.0,), 4; H = -1.0)
    @test_throws ArgumentError GaussianGrid{1, Float64}(SVector(0.0), -0.1, (4,), 2.0)
    @test_throws ArgumentError GaussianGrid{1, Float64}(SVector(0.0), 0.1, (0,), 2.0)
    # padded size: at least 2n - 1, a product of 2, 3 and 5
    @test GC._padded_dims((8, 4)) == (15, 8)
    @test GC.convolution_bytes((8, 4), 3, 2) == 15 * 8 * (8 * 3 + 16 * 3)
    @test GC._check_memory((8, 4), 3, 2, 10^6) === nothing
    @test_throws ArgumentError GC._check_memory((512, 512), 6, 3, 10^6)
end

@testset "quasi-interpolation" begin
    # saturation level in R³: Maz'ya & Schmidt (1996), coefficients of order 0
    @test saturation_error(3.0, 3) ≈ 8.303e-13 rtol = 1.0e-3
    @test saturation_error(4.0, 3) ≈ 4.295e-17 rtol = 1.0e-3
    @test saturation_error(2.0) ≈ 2 * exp(-2π^2) rtol = 1.0e-6
    # the generating function integrates to h^d
    h, H = 0.1, 2.0
    @test sum(gaussian_generator(x^2 + y^2, h, H, Val(2)) for x in -3:0.01:3, y in -3:0.01:3) * 1.0e-4 ≈ h^2 rtol = 1.0e-10
    # a constant is reproduced down to the saturation level, far from the edges
    for H in (1.0, 2.0)
        uh = quasi_interpolate(x -> 1.0, GaussianGrid((0.0,), (1.0,), 80; H))
        dev = maximum(abs(uh((x,)) - 1) for x in 0.4:0.001:0.6)
        @test saturation_error(H) / 2 < dev ≤ saturation_error(H) * (1 + 1.0e-6)
    end
    # second order in the step for a smooth function: the error is H h² u''/4
    errs = map((40, 80)) do n
        g = GaussianGrid((0.0,), (1.0,), n; H = 3.0)
        abs(quasi_interpolate(x -> sinpi(x[1]), g)((0.5,)) - 1)
    end
    @test errs[1] / errs[2] ≈ 4 rtol = 0.02
    @test errs[2] ≈ 3.0 * (1 / 80)^2 * π^2 / 4 rtol = 0.02
    g = GaussianGrid((0.0, 0.0), (1.0, 1.0), 20)
    @test quasi_interpolate(ones(20, 20), g) isa QuasiInterpolant
    @test quasi_interpolate(ones(20, 20), g)((0.5, 0.5)) ≈ 1 atol = 1.01 * saturation_error(2.0, 2)
    @test_throws DimensionMismatch quasi_interpolate(ones(3, 3), g)
end

# A symmetric, even, positive kernel of m × m components on a grid of any dimension:
# a Gaussian, whose transform is positive, times a positive definite block.
function _gc_kernel(::Val{m}, p) where {m}
    S = SMatrix{m, m}([1 / (1 + i + j) for i in 1:m, j in 1:m])
    return Δ -> exp(-sum(abs2, Δ) / 3) * (p * S + I)
end

@testset "FFT product" begin
    for (dims, m) in (((7, 5), 1), ((6, 4), 2), ((4, 3, 5), 2), ((3, 4, 2), 3))
        samples = GC.sample_kernel(_gc_kernel(Val(m), 0.7), dims, Val(m))
        @test length(samples) == m * (m + 1) ÷ 2 && size(first(samples)) == 2 .* dims .- 1
        X = ntuple(_ -> randn(dims), m)
        Y = GC.convolve!(ntuple(_ -> zeros(dims), m), GC.GridConvolution(samples, Val(m)), X)
        ref = GC.convolve_direct(samples, X, Val(m))
        @test all(isapprox.(Y, ref; rtol = 1.0e-11))
    end
    @test_throws DimensionMismatch GC.GridConvolution(GC.sample_kernel(_gc_kernel(Val(2), 0.7), (4, 4), Val(2)), Val(3))
end

# The dense matrix of the masked system, generic in the number type.
function _gc_dense(K::GC.SampledKernel{m, T}, inside, spring) where {m, T}
    N = length(inside)
    self = GC._self_block(K)
    A = zeros(T, m * N, m * N)
    off = CartesianIndex(map(s -> (s + 1) ÷ 2, size(first(K.samples))))
    for (i, I) in enumerate(inside), (j, J) in enumerate(inside), p in 1:m, q in 1:m
        A[(p - 1) * N + i, (q - 1) * N + j] = K.samples[GC._pair(p, q)][I - J + off] +
            (i == j) * (spring[i] * self[p, q] + K.onsite[p, q])
    end
    return A
end

@testset "solve" begin
    dims, m = (6, 5), 2
    inside = [I for I in CartesianIndices(dims) if isodd(I[1] + 2 * I[2]) || I[1] > 2]
    N = length(inside)
    spring = [0.3 * mod(k, 3) for k in 1:N]
    onsite = SMatrix{2, 2}(0.4, 0.1, 0.1, 0.2)
    kernel(p) = GC.SampledKernel(GC.sample_kernel(_gc_kernel(Val(m), p), dims, Val(m)), p * onsite)
    R = randn(m * N, 2)
    K, A = kernel(0.7), _gc_dense(kernel(0.7), inside, spring)
    @test issymmetric(A)
    @test GC.masked_product(K, inside, spring, R) ≈ A * R rtol = 1.0e-11
    @test GC.gaussian_solve(K, inside, spring, R; rtol = 1.0e-12) ≈ A \ R rtol = 1.0e-9
    @test GC.gaussian_solve(K, inside, spring, R; rtol = 1.0e-12, symmetric = false) ≈ A \ R rtol = 1.0e-9
    # no on-site block by default
    @test iszero(GC.SampledKernel(K.samples, Val(m)).onsite)
    # a solve that does not converge is an error, not a field
    @test_throws ErrorException GC.gaussian_solve(K, inside, spring, R; itmax = 1)
    # numbers an iteration cannot compare are refused
    @test_throws ArgumentError GC.gaussian_solve(GC.SampledKernel(map(s -> big.(s), K.samples), Val(m)), inside, spring, R)

    # derivatives: the implicit-function theorem against ForwardDiff through the dense solve
    f(p) = sum(abs2, GC.gaussian_solve(kernel(p), inside, spring, (1 + p^2) * R; rtol = 1.0e-13))
    fdense(p) = (k = kernel(p); sum(abs2, _gc_dense(k, inside, spring) \ ((1 + p^2) * R)))
    @test ForwardDiff.derivative(f, 0.7) ≈ ForwardDiff.derivative(fdense, 0.7) rtol = 1.0e-8
    # nested, under two tags
    @test ForwardDiff.derivative(p -> ForwardDiff.derivative(f, p), 0.7) ≈
        ForwardDiff.derivative(p -> ForwardDiff.derivative(fdense, p), 0.7) rtol = 1.0e-7
    # a gradient: two partials, in the kernel and in the right-hand side
    g(v) = sum(GC.gaussian_solve(kernel(v[1]), inside, spring, v[2] * R; rtol = 1.0e-13))
    gdense(v) = sum(_gc_dense(kernel(v[1]), inside, spring) \ (v[2] * R))
    @test ForwardDiff.gradient(g, [0.7, 1.3]) ≈ ForwardDiff.gradient(gdense, [0.7, 1.3]) rtol = 1.0e-8
    # the product alone
    @test ForwardDiff.derivative(p -> sum(GC.masked_product(kernel(p), inside, spring, p * R)), 0.7) ≈
        ForwardDiff.derivative(p -> sum(_gc_dense(kernel(p), inside, spring) * (p * R)), 0.7) rtol = 1.0e-10

    @testset "inference" begin
        C = GC.GridConvolution(K.samples, Val(m))
        X = ntuple(_ -> randn(dims), m)
        @inferred GC.convolve!(ntuple(_ -> zeros(dims), m), C, X)
        @inferred GC.gaussian_solve(K, inside, spring, R)
    end
end
