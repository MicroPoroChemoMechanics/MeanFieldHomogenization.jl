# `_static_ldiv`: the solve of the local Newton iterations and of the tangents of
# the laws. Above 14 × 14 it makes `\`'s own LAPACK calls on buffers of the task,
# so that it must give the bits of `\`, raise its exceptions, and allocate
# nothing once warm; below that size, for a triangular matrix and for duals, it
# is `\`. A crystal of 12 slip systems, whose local equations are 24 × 24, runs
# through it.

using Test
using MeanFieldHomogenization
using LinearAlgebra
using Random
using StaticArrays
using ForwardDiff

const _SL = MeanFieldHomogenization.MaterialLaws._static_ldiv

# A random system whose matrix spans `spread` decades, as Jacobians of laws do.
_sl_system(rng, N, M, spread) = (
    SMatrix{N, N}(randn(rng, N, N) .* exp10.(spread .* rand(rng, N, N))),
    M == 0 ? SVector{N}(randn(rng, N)) : SMatrix{N, M}(randn(rng, N, M)),
)

@testset "_static_ldiv: the bits and the exceptions of `\\`" begin
    rng = Xoshiro(20261029)
    for (N, M) in ((24, 0), (24, 6), (15, 0), (6, 0), (6, 6)), spread in (0, 4)
        for _ in 1:200
            A, B = _sl_system(rng, N, M, spread)
            @test _SL(A, B) == A \ B
        end
    end
    # Triangular matrices take the triangular routes of `\`.
    A, b = _sl_system(rng, 24, 0, 0)
    for T in (LowerTriangular, UpperTriangular, Diagonal)
        L = SMatrix{24, 24}(T(Matrix(A)))
        @test _SL(L, b) == L \ b
    end
    # A singular matrix, a non-finite one: the exceptions of `\`.
    S = SMatrix{24, 24}(vcat(Matrix(A)[1:23, :], zeros(1, 24)))           # a zero row
    @test_throws SingularException S \ b
    @test_throws SingularException _SL(S, b)
    A_nan = setindex(A, NaN, 3, 5)
    @test_throws ArgumentError A_nan \ b
    @test_throws ArgumentError _SL(A_nan, b)
end

@testset "_static_ldiv: no allocation, inferred, duals, tasks" begin
    rng = Xoshiro(1)
    A, b = _sl_system(rng, 24, 0, 2)
    _, C = _sl_system(rng, 24, 6, 0)
    @test (@inferred _SL(A, b)) isa SVector{24, Float64}
    @test (@inferred _SL(A, C)) isa SMatrix{24, 6, Float64}
    solve(A, b) = _SL(A, b)
    solve(A, b); solve(A, C)
    @test (@allocated solve(A, b)) == 0
    @test (@allocated solve(A, C)) == 0
    # A dual right-hand side goes through `\`: d(A⁻¹ x b)/dx = A⁻¹ b.
    @test ForwardDiff.derivative(x -> _SL(A, x * b), 1.0) ≈ A \ b rtol = 1.0e-12
    # Each task its own buffers: concurrent solves give the serial bits.
    systems = [_sl_system(Xoshiro(k), 24, 0, 2) for k in 1:16]
    serial = [A \ b for (A, b) in systems]
    concurrent = fetch.([Threads.@spawn _SL(A, b) for (A, b) in systems])
    @test concurrent == serial
end
