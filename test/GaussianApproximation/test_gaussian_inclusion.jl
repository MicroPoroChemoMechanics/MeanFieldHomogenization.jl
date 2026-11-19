# =============================================================================
#  test_gaussian_inclusion.jl — a heterogeneity of any shape by the Gaussian
#  approximation of its integral equation.
#
#   1. The smoothed potentials: their Laplacians, the Green operator far from
#      the node, the Hill tensor of the sphere at the node.
#   2. Regions of space: ellipsoid, supershapes, torus, voxels; covered
#      fractions.
#   3. Eshelby's solution: spheres and tilted ellipsoids, stiffer, softer,
#      rigid, void and anisotropic heterogeneities, conduction and elasticity.
#   4. The contract: contributions, schemes, derivatives, guard rails.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff

const GI = MeanFieldHomogenization.GaussianApproximation
const GI_K = TensISO{3}(2.0)
const GI_C = MeanFieldHomogenization.iso_stiffness_E_nu(10.0, 0.3)

_gi_err(A, Aref) = norm(TensND.canonical_components(A) - TensND.canonical_components(Aref)) /
    norm(TensND.canonical_components(Aref))
# Kelvin–Mandel matrix of an order-4 tensor of the package, in the canonical basis.
_gi_mandel(t) = SMatrix{6, 6}(TensND.mandel(t, CanonicalBasis{3, Float64}()))

@testset "smoothed potentials" begin
    H = 2.0
    φ(ρ) = exp(-ρ^2 / H) / (π * H)^1.5
    for ρ in (0.0, 0.6, 1.0, 3.7)
        α, β = GI.radial_hessian(ρ, H)
        A, B, C = GI.radial_fourth_gradient(ρ, H)
        # ΔΦ₀ = -φ, ΔΦ₂ = Φ₀ (trace of the fourth gradient) and Δ²Φ₂ = -φ
        @test 3α + β ≈ -φ(ρ) atol = 1.0e-14
        @test 5A + B ≈ α atol = 1.0e-13
        @test 7B + C ≈ β atol = 1.0e-13
        @test 15A + 10B + C ≈ -φ(ρ) atol = 1.0e-13
    end
    @inferred GI.radial_fourth_gradient(1.3, H)

    # far from the node, the Green operator itself: exactly in conduction (a
    # Gaussian is seen from outside as a point source), up to a quadrupole of
    # relative size H/ρ² in elasticity
    for Δ in ((12, 5, -7), (20, 0, 0), (9, 9, 9))
        x = collect(float.(Δ))
        @test GI.smoothed_green_operator(GI_K, Δ, H) ≈ TensND.get_array(green_operator_iso(GI_K, x)) rtol = 1.0e-12
        exact = _gi_mandel(green_operator_iso(GI_C, x))
        @test norm(GI.smoothed_green_operator(GI_C, Δ, H) - exact) / norm(exact) < 4 * H / sum(abs2, Δ)
    end
    # at the node, φ(0) times the Hill tensor of the sphere
    @test GI.smoothed_green_operator(GI_C, (0, 0, 0), H) ≈ φ(0.0) * _gi_mandel(hill_tensor(Ellipsoid(1.0), GI_C)) rtol = 1.0e-12
    @test GI.smoothed_green_operator(GI_K, (0, 0, 0), H) ≈ φ(0.0) / (3 * 2.0) * I rtol = 1.0e-12
    @test issymmetric(GI.smoothed_green_operator(GI_C, (1, -2, 3), H))
end

@testset "regions of space" begin
    volume(region; n = 60) = (g = GaussianGrid(region, n); sum(coverage(region, g; samples = 3)) * g.h^3)
    ell = Ellipsoid(1.0, 0.6, 0.3; euler_angles = (0.4, 0.9, 0.2))
    r = Region(ell)
    @test measure(r) ≈ 4π / 3 * 0.18
    @test volume(r) ≈ measure(r) rtol = 1.0e-2
    # the end of the long axis is inside, the same distance along the short one is not
    e₁, e₃ = (TensND.canonical_components(TensND.tens_basis(ell.basis, i)) for i in (1, 3))
    @test 0.99 * e₁ ∈ r && !(0.99 * e₃ ∈ r) && 0.29 * e₃ ∈ r
    lo, hi = bounding_box(r)
    @test all(lo .< -0.29) && all(hi .< 1.0 + 1.0e-12) && lo ≈ -hi

    for s in (Supersphere(1.0, 0.7), Superspheroid(1.0, 0.5, 2.0))
        rs = Region(s)
        @test measure(rs) ≈ shape_volume(s)
        @test volume(rs) ≈ measure(rs) rtol = 2.0e-2
    end
    @test bounding_box(Region(Superspheroid(1.0, 0.5, 2.0))) == (SVector(-1.0, -1.0, -0.5), SVector(1.0, 1.0, 0.5))

    torus = Torus(1.0, 0.3; center = (0.5, 0.0, 0.0))
    @test measure(torus) ≈ 2π^2 * 0.09
    @test volume(torus) ≈ measure(torus) rtol = 1.0e-2
    @test (1.5, 0.0, 0.0) ∈ torus && !((0.5, 0.0, 0.0) ∈ torus)
    @test_throws ArgumentError Torus(0.3, 1.0)

    # voxels, and set operations in space
    mask = [sum(abs2, (i, j, k) .- 8.5) ≤ 49 for i in 1:16, j in 1:16, k in 1:16]
    @test volume(PixelRegion(mask, 0.125; lo = (-1, -1, -1)); n = 16) ≈ count(mask) * 0.125^3 rtol = 1.0e-10
    shell = setdiff(Region(Ellipsoid(1.0)), Region(Ellipsoid(0.5)))
    @test volume(shell) ≈ 4π / 3 * (1 - 0.125) rtol = 1.0e-2
end

@testset "Eshelby's solution" begin
    sphere = GaussianInclusion(Ellipsoid(1.0); n = 16)
    @test occursin("GaussianInclusion", sprint(show, sphere))
    @test inclusion_volume(sphere) ≈ 4π / 3 rtol = 5.0e-3
    @test MeanFieldHomogenization.Core.dimension(sphere) == 3
    @test MeanFieldHomogenization.Core.inclusion_basis(sphere) === sphere.basis
    @test MeanFieldHomogenization.Core.shape_trait(sphere) isa GI.GaussianShape
    @test !MeanFieldHomogenization.Core.is_homogeneous_inclusion(sphere)

    # conduction: stiffer, softer, nearly insulating
    for k₁ in (20.0, 0.2, 2.0e-4)
        K₁ = TensISO{3}(k₁)
        @test _gi_err(gradient_gradient_concentration(sphere, K₁, GI_K), gradient_gradient_concentration(Ellipsoid(1.0), K₁, GI_K)) < 4.0e-2
    end
    # a thousand times more conducting: the gradient inside nearly vanishes, the
    # flux side stays finite, and the error of this coarse grid is larger
    K∞ = TensISO{3}(2.0e3)
    @test _gi_err(flux_gradient_concentration(sphere, K∞, GI_K), flux_gradient_concentration(Ellipsoid(1.0), K∞, GI_K)) < 8.0e-2
    # one solve gives both tensors, and is kept
    @test fe_assembly_count(sphere) == 4
    # no contrast: the heterogeneity is the matrix, exactly
    @test TensND.get_array(gradient_gradient_concentration(sphere, GI_K, GI_K)) == I
    @test TensND.get_array(flux_gradient_concentration(sphere, GI_K, GI_K)) == 2.0 * I

    # a tilted ellipsoid, given by the shape or by a region and its angles
    ang = (0.4, 0.9, 0.2)
    exact = Ellipsoid(1.0, 0.7, 0.5; euler_angles = ang)
    K₁ = TensISO{3}(40.0)
    Aex = gradient_gradient_concentration(exact, K₁, GI_K)
    for g in (GaussianInclusion(exact; n = 24), GaussianInclusion(Region(Ellipsoid(1.0, 0.7, 0.5)); n = 24, euler_angles = ang))
        @test _gi_err(gradient_gradient_concentration(g, K₁, GI_K), Aex) < 5.0e-2
    end
    # one grid: a larger error, which the second grid reduces
    one_grid = GaussianInclusion(Ellipsoid(1.0); n = 16, extrapolate = false)
    @test _gi_err(gradient_gradient_concentration(one_grid, TensISO{3}(20.0), GI_K), gradient_gradient_concentration(Ellipsoid(1.0), TensISO{3}(20.0), GI_K)) >
        _gi_err(gradient_gradient_concentration(sphere, TensISO{3}(20.0), GI_K), gradient_gradient_concentration(Ellipsoid(1.0), TensISO{3}(20.0), GI_K))

    # elasticity: stiffer, softer, and two contrasts of no definite sign, one of
    # them transversely isotropic
    small = GaussianInclusion(Ellipsoid(1.0); n = 12)
    C_ti = TensND.TensTI{4}(60.0, 40.0, 12.0, 30.0, 16.0, (0.0, 0.0, 1.0))
    mixed = TensISO{3}(3 * 40.0, 2 * 1.0)                   # stiffer in bulk, softer in shear
    for C₁ in (MeanFieldHomogenization.iso_stiffness_E_nu(100.0, 0.2), MeanFieldHomogenization.iso_stiffness_E_nu(1.0, 0.2), C_ti, mixed)
        @test _gi_err(gradient_gradient_concentration(small, C₁, GI_C), gradient_gradient_concentration(Ellipsoid(1.0), C₁, GI_C)) < 6.0e-2
        @test _gi_err(flux_gradient_concentration(small, C₁, GI_C), flux_gradient_concentration(Ellipsoid(1.0), C₁, GI_C)) < 6.0e-2
    end
    @test GI._contrast_sign(GI._local_matrix(mixed, small.basis) - GI._local_matrix(GI_C, small.basis)) == 0
end

@testset "contract, schemes, derivatives" begin
    g = GaussianInclusion(Ellipsoid(1.0); n = 12)
    K₁ = TensISO{3}(20.0)
    @test check_inclusion_interface(g; physics = :conduction, verbose = false)
    # the generic contribution tensors, from the two concentration tensors
    @test _gi_err(property_contribution(g, K₁, GI_K), property_contribution(Ellipsoid(1.0), K₁, GI_K)) < 8.0e-2
    for scheme in (MoriTanaka(), SelfConsistent())
        Khom = map((g, Ellipsoid(1.0))) do incl
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => GI_K); fraction = :rest)
            add_phase!(rve, :I, incl, Dict(:K => K₁); fraction = 0.2, symmetrize = IsoSymmetrize())
            homogenize(rve, scheme, :K)
        end
        @test _gi_err(Khom[1], Khom[2]) < 2.0e-2
    end

    # the concentration tensor depends on the ratio of the conductivities only:
    # k₁ ∂A/∂k₁ + k₀ ∂A/∂k₀ = 0, exactly for the discrete system
    A11(k₁, k₀) = TensND.get_array(gradient_gradient_concentration(GaussianInclusion(Ellipsoid(1.0, 1.0, 0.6); n = 10), TensISO{3}(k₁), TensISO{3}(k₀)))[1, 1]
    ∂ = ForwardDiff.gradient(v -> A11(v[1], v[2]), [20.0, 2.0])
    @test abs(20.0 * ∂[1] + 2.0 * ∂[2]) < 1.0e-8 * abs(20.0 * ∂[1])
    @test abs(∂[1]) > 1.0e-4
    # and its second derivative along a common factor vanishes: nested duals
    @test abs(ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> A11(20.0 * b, 2.0 * b), a), 1.0)) < 1.0e-7 * A11(20.0, 2.0)
    # softer side, solved with the opposite sign
    @test ForwardDiff.derivative(k -> A11(k, 2.0), 0.2) ≈ (A11(0.2 + 1.0e-4, 2.0) - A11(0.2 - 1.0e-4, 2.0)) / 2.0e-4 rtol = 1.0e-5
end

@testset "guard rails" begin
    @test_throws ArgumentError GaussianInclusion(Ellipsoid(1.0); n = 2)
    @test_throws ArgumentError GaussianInclusion(Ellipsoid(1.0); n = 64, memory_limit = 10^6)
    @test_throws ArgumentError GaussianInclusion(Region(x -> false, (-1, -1, -1), (1, 1, 1)); n = 8)
    g = GaussianInclusion(Ellipsoid(1.0); n = 4, extrapolate = false)
    # an anisotropic reference medium is refused, with the way out
    @test_throws ArgumentError gradient_gradient_concentration(g, GI_K, TensND.Tens([3.0 0.4 0.1; 0.4 2.0 0.3; 0.1 0.3 1.0]))
    # the shape is not differentiable
    @test_throws ArgumentError ForwardDiff.derivative(a -> inclusion_volume(GaussianInclusion(Region(x -> sum(abs2, x) ≤ a^2, (-a, -a, -a), (a, a, a)))), 1.0)
    @test_throws ArgumentError MeanFieldHomogenization.Schemes._replace_geom_field(g, Val(:region), nothing, 1.0)
    @test_throws ArgumentError MeanFieldHomogenization.Schemes._replace_geom_field(g, Val(:region), 1, 1.0)
end
