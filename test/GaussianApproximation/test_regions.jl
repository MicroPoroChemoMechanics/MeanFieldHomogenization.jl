# =============================================================================
#  test_regions.jl — describing a shape: catalog, set operations, functions,
#  images. Every region is checked on membership, on its box, and on its
#  measure against the count of the cells of a fine grid.
# =============================================================================

using Test
using MeanFieldHomogenization
using LinearAlgebra
using StaticArrays

const GR = MeanFieldHomogenization.GaussianApproximation

# Area counted on a grid of n nodes.
_gr_area(region; n = 400) = (g = GaussianGrid(region, n); length(inside_nodes(region, g)) * g.h^2)

@testset "catalog" begin
    shapes = (
        Disk(0.8; center = (0.3, -0.2)), Ellipse(1.0, 0.4; angle = 0.6), Ellipse(1, 2),
        Superellipse(1.0, 0.6, 3.0; center = (1.0, 1.0), angle = -0.3), Superellipse(1.0, 1.0, 0.3),
        Rectangle(0.7, 0.3; angle = 1.1), Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
    )
    for r in shapes
        lo, hi = bounding_box(r)
        @test all(lo .< hi)
        @test _gr_area(r) ≈ measure(r) rtol = 5.0e-3
        # nothing of the region outside its box
        g = GaussianGrid(lo .- 0.2, hi .+ 0.2, 120)
        @test all(I -> all(lo .≤ GR.node(g, I) .≤ hi), inside_nodes(r, g))
    end
    @test measure(Disk(2.0)) ≈ 4π
    @test measure(Superellipse(1.0, 0.5, 1.0)) ≈ π / 2            # an ellipse
    @test (0.0, 0.0) ∈ Rectangle(1.0, 1.0) && !((1.2, 0.0) ∈ Rectangle(1.0, 1.0))
    @test !((0.9, 0.9) ∈ Rectangle(1.0, 1.0; angle = π / 4)) && (1.2, 0.0) ∈ Rectangle(1.0, 1.0; angle = π / 4)
    @test (0.5, 0.5) ∈ Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    @test !((1.5, 1.5) ∈ Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))
    for bad in (
            () -> Disk(-1.0), () -> Ellipse(1.0, 0.0), () -> Superellipse(1.0, 1.0, -1.0),
            () -> Superellipse(0.0, 1.0, 1.0), () -> Superellipse(1.0, -1.0, 1.0),
            () -> Rectangle(0.0, 1.0), () -> Rectangle(1.0, -1.0), () -> Polygon([(0, 0), (1, 0)]),
            () -> StarShape(φ -> 1.0, 0.0),
        )
        @test_throws ArgumentError bad()
    end
    @test_throws DimensionMismatch Disk(1.0; center = (0.0, 0.0, 0.0))
end

@testset "functions and images" begin
    star = StarShape(φ -> 1 + 0.3cos(5φ), 1.3; center = (0.5, 0.0))
    # area of r ≤ ρ(φ): ∫ ρ²/2 dφ = π (1 + 0.3²/2)
    @test measure(star) === nothing
    @test _gr_area(star) ≈ π * (1 + 0.045) rtol = 5.0e-3
    @test bounding_box(star) == (SVector(-0.8, -1.3), SVector(1.8, 1.3))

    quarter = Region(x -> x[1]^2 + x[2]^2 ≤ 1, (0, 0), (1, 1))
    @test _gr_area(quarter) ≈ π / 4 rtol = 5.0e-3
    @test bounding_box(quarter) == (SVector(0.0, 0.0), SVector(1.0, 1.0))
    @test_throws DimensionMismatch Region(x -> true, (0, 0), (1, 1, 1))
    ball = Region(x -> sum(abs2, x) ≤ 1, (-1, -1, -1), (1, 1, 1))     # dimension from the corners
    @test (0.0, 0.0, 0.5) ∈ ball && !((0.9, 0.9, 0.9) ∈ ball)

    mask = [i + j ≤ 5 for i in 1:6, j in 1:4]
    px = PixelRegion(mask, 0.5; lo = (1.0, 2.0))
    @test measure(px) == count(mask) * 0.25
    @test bounding_box(px) == (SVector(1.0, 2.0), SVector(4.0, 4.0))
    @test (1.1, 2.1) ∈ px && !((3.9, 3.9) ∈ px) && !((0.0, 0.0) ∈ px) && !((9.0, 9.0) ∈ px)
    @test _gr_area(px; n = 240) ≈ measure(px) rtol = 1.0e-2
    @test PixelRegion(trues(2, 2, 2), 1.0) isa AbstractRegion{3}
    @test_throws ArgumentError PixelRegion(falses(3, 3), 1.0)
    @test_throws ArgumentError PixelRegion(mask, 0.0)
end

@testset "set operations" begin
    a, b = Disk(1.0), Disk(1.0; center = (1.0, 0.0))
    lens = 2π / 3 - sqrt(3) / 2                # two unit disks at distance 1
    @test a ∪ b isa UnionRegion && a ∩ b isa IntersectionRegion && setdiff(a, b) isa DifferenceRegion
    @test _gr_area(a ∩ b) ≈ lens rtol = 5.0e-3
    @test _gr_area(a ∪ b) ≈ 2π - lens rtol = 5.0e-3
    @test _gr_area(setdiff(a, b)) ≈ π - lens rtol = 5.0e-3
    @test bounding_box(a ∪ b) == (SVector(-1.0, -1.0), SVector(2.0, 1.0))
    @test bounding_box(a ∩ b) == (SVector(0.0, -1.0), SVector(1.0, 1.0))
    @test bounding_box(setdiff(a, b)) == bounding_box(a)
    @test measure(a ∪ b) === nothing
    @test_throws ArgumentError bounding_box(a ∩ Disk(1.0; center = (5.0, 0.0)))
    ring = Annulus(1.0, 0.5)
    @test _gr_area(ring) ≈ 0.75π rtol = 5.0e-3
    @test !((0.0, 0.0) ∈ ring) && (0.7, 0.0) ∈ ring
    @test_throws ArgumentError Annulus(1.0, 1.5)
    @test_throws ArgumentError Annulus(1.0, 0.0)
end

@testset "a region on a grid" begin
    g = GaussianGrid(Disk(1.0), 10)
    @test g.dims == (10, 10) && g.H == 2.0
    @test_throws ArgumentError inside_nodes(Disk(0.01; center = (0.6, 0.6)), g)
    # covered fractions: their sum is the area, with far fewer cells than a count needs
    θ = coverage(Disk(1.0), GaussianGrid(Disk(1.0), 40))
    @test all(0 .≤ θ .≤ 1) && any(0 .< θ .< 1)
    @test sum(θ) * (2 / 40)^2 ≈ π rtol = 2.0e-3
    # eroded by δ, the disk has the radius 1 - δ
    θ = coverage(Disk(1.0), GaussianGrid(Disk(1.0), 40); erosion = 0.2)
    @test sum(θ) * (2 / 40)^2 ≈ π * 0.8^2 rtol = 5.0e-3
    # a contour that touches its box is still subdivided there
    θ = coverage(Rectangle(1.0, 1.0), GaussianGrid(Rectangle(1.0, 1.0), 10); erosion = 0.05)
    @test sum(θ) * 0.2^2 ≈ 1.9^2 rtol = 2.0e-2
end
