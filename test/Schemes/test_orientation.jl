using Test
using MeanFieldHomogenization

# =============================================================================
#  test_orientation.jl — discretized orientation families
#  (`src/Schemes/orientation.jl`).
#
#  Pichler-Hellmich (2011) convention / `disc_theta` from echoes:
#    θ_i = (π/2)(i-1)/(N-1), mid-point bounds clamped to [0, π/2],
#    w_i = cos θ_i⁻ − cos θ_i⁺,  Σ w_i = 1.
# =============================================================================

@testset "polar_orientation_bins — convention and normalization" begin
    for N in (2, 3, 5, 12, 64)
        bins = polar_orientation_bins(N)
        @test length(bins) == N

        θ = [b.θ for b in bins]
        w = [b.weight for b in bins]

        # Angles: endpoints included, strictly increasing, uniformly spread.
        @test θ[1] ≈ 0.0
        @test θ[end] ≈ π / 2
        @test issorted(θ)
        @test all(θ .≥ 0.0)
        @test all(θ .≤ π / 2 + 1.0e-15)
        for i in 1:N
            @test θ[i] ≈ (π / 2) * (i - 1) / (N - 1)
        end

        # Weights: partition of the hemisphere, Σ w = 1, all positive.
        @test sum(w) ≈ 1.0
        @test all(w .> 0.0)
    end
end

@testset "polar_orientation_bins — N = 2, limiting case" begin
    bins = polar_orientation_bins(2)
    @test length(bins) == 2
    @test bins[1].θ ≈ 0.0
    @test bins[2].θ ≈ π / 2

    # Bounds: θ₁⁻ = 0 (clamped), θ₁⁺ = π/4 ; θ₂⁻ = π/4, θ₂⁺ = π/2 (clamped).
    @test bins[1].weight ≈ 1 - cos(π / 4)
    @test bins[2].weight ≈ cos(π / 4)
    @test bins[1].weight + bins[2].weight ≈ 1.0
end

@testset "polar_orientation_bins — return type" begin
    bins = polar_orientation_bins(4)
    @test bins isa Vector{@NamedTuple{θ::Float64, weight::Float64}}
    @test bins[1] isa NamedTuple
    @test propertynames(bins[1]) == (:θ, :weight)
end

@testset "polar_orientation_bins — N < 2 rejected" begin
    @test_throws ArgumentError polar_orientation_bins(1)
    @test_throws ArgumentError polar_orientation_bins(0)
    @test_throws ArgumentError polar_orientation_bins(-3)
end

@testset "polar_orientation_bins — convergence to the isotropic average" begin
    # Σ wᵢ f(θᵢ) must converge to ∫ f(θ) sin θ dθ over [0, π/2] in O(Δθ²).
    # Tested on f(θ) = cos²θ, whose exact average is 1/3.
    err(N) = abs(sum(b.weight * cos(b.θ)^2 for b in polar_orientation_bins(N)) - 1 / 3)

    e_small = err(16)
    e_big = err(64)
    @test e_big < e_small
    # Second order: dividing Δθ by 4 must divide the error by ~16.
    @test e_big < e_small / 8
    @test e_big < 1.0e-3
end
