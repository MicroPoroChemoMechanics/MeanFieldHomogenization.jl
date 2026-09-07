# Superspherical and superspheroidal geometry.
#
# Every closed form here has a value it must reproduce at p = 1 and p = 1/2 --
# sphere and octahedron, spheroid and double cone -- so the whole file is
# oracles rather than golden numbers. The two things that are *not* obvious, and
# that carry most of the tests, are the interior extremum of a superspheroid's
# radius and the normal on a crease.

using Test
using MeanFieldHomogenization
using LinearAlgebra: norm
import ForwardDiff
import SymPy
import TensND
using SymPy: Sym

const _UNIT(v) = v ./ norm(v)

# A quasi-uniform set of directions, deterministic, avoiding the axes and the
# coordinate planes so that a smooth-point test really tests a smooth point.
const _DIRS = let
    ds = NTuple{3, Float64}[]
    n = 12
    for i in 1:n, j in 1:(2n)
        θ = π * (i - 0.5) / n
        φ = 2π * (j - 0.5) / (2n)
        push!(ds, (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)))
    end
    ds
end

@testset "Supersphere / Superspheroid — geometry" begin

    @testset "construction, promotion and refusals" begin
        @test eltype(Supersphere(1, 1)) === Float64          # Integer promoted
        @test eltype(Supersphere(1.0f0, 0.5f0)) === Float32  # Float32 preserved
        @test eltype(Supersphere(1, 0.3)) === Float64
        # `_floatlike` keeps a Rational rather than widening it, deliberately:
        # the same rule the rest of the package follows.
        @test eltype(Superspheroid(1, 2, 1 // 2)) === Rational{Int}

        # A single Dual argument must carry the whole shape: this is the
        # promotion defect that keeps reappearing elsewhere in the package.
        d = ForwardDiff.Dual{:t}(1.0, 1.0)
        @test eltype(Supersphere(d, 0.3)) <: ForwardDiff.Dual
        @test eltype(Supersphere(1.0, d)) <: ForwardDiff.Dual
        @test eltype(Superspheroid(1.0, d, 0.4)) <: ForwardDiff.Dual

        @test_throws ArgumentError Supersphere(-1.0, 0.5)
        @test_throws ArgumentError Supersphere(1.0, 0.0)
        @test_throws ArgumentError Superspheroid(1.0, -2.0, 0.5)
        @test_throws ArgumentError Superspheroid(1.0, 2.0, -0.5)

        @test shape_exponent(Supersphere(1.0, 0.3)) ≈ 0.6
        @test sprint(show, Supersphere(1.0, 0.3)) == "Supersphere(a = 1.0, p = 0.3)"
        @test occursin("c = 2.0", sprint(show, Superspheroid(1.0, 2.0, 0.3)))
    end

    @testset "classification" begin
        @test is_concave(Supersphere(1.0, 0.3))
        @test !is_convex(Supersphere(1.0, 0.3))
        @test is_convex(Supersphere(1.0, 0.5))       # the octahedron is convex
        @test !is_concave(Supersphere(1.0, 0.5))
        @test is_sphere(Supersphere(1.0, 1.0))
        @test !is_sphere(Supersphere(1.0, 1.0 + 1.0e-12))
        @test is_concave(Superspheroid(1.0, 2.0, 0.4))
    end

    @testset "the radial map inverts the level set" begin
        for p in (0.25, 0.5, 0.8, 1.0, 2.0, 6.0), a in (1.0, 2.5)
            s = Supersphere(a, p)
            for n in _DIRS
                x = surface_point(s, n)
                @test abs(level_set(s, x)) < 1.0e-12
                @test radial_distance(s, n) ≈ norm(x)
            end
        end
        for p in (0.3, 0.5, 1.0, 3.0), (a, c) in ((1.0, 2.0), (2.0, 0.5))
            s = Superspheroid(a, c, p)
            for n in _DIRS
                @test abs(level_set(s, surface_point(s, n))) < 1.0e-12
            end
        end
    end

    @testset "closed forms against the shapes they must reproduce" begin
        @test shape_volume(Supersphere(1.0, 1.0)) ≈ 4π / 3
        @test shape_volume(Supersphere(2.0, 1.0)) ≈ 4π * 8 / 3
        @test shape_volume(Supersphere(1.0, 0.5)) ≈ 4 / 3           # octahedron
        @test shape_volume(Superspheroid(1.0, 2.0, 1.0)) ≈ 4π * 2 / 3
        @test shape_volume(Superspheroid(1.0, 2.0, 0.5)) ≈ 2π * 2 / 3  # double cone
        @test projected_area(Supersphere(1.0, 1.0)) ≈ π
        @test projected_area(Supersphere(1.0, 0.5)) ≈ 2            # square, diagonal 2a
        @test projected_area(Supersphere(3.0, 1.0)) ≈ 9π

        # The two families coincide only at p = 1, and it is worth pinning that
        # they do *not* elsewhere: `|x|^m + |y|^m` is not `(x²+y²)^{m/2}` unless
        # m = 2, so a supersphere is not the superspheroid of equal semi-axes.
        # Reading one for the other would be a silent factor on every volume.
        @test shape_volume(Supersphere(1.7, 1.0)) ≈ shape_volume(Superspheroid(1.7, 1.7, 1.0))
        for p in (0.3, 0.5, 2.0)
            @test !isapprox(
                shape_volume(Supersphere(1.7, p)), shape_volume(Superspheroid(1.7, 1.7, p));
                rtol = 1.0e-3,
            )
        end

        # Large 1/m is exactly where a naive Γ product overflows; `loggamma`
        # is why this is finite. p = 0.05 means an exponent of 0.1.
        @test isfinite(shape_volume(Supersphere(1.0, 0.05)))
        @test shape_volume(Supersphere(1.0, 0.05)) > 0

        for s in (Supersphere(1.3, 0.35), Superspheroid(1.0, 2.5, 0.7))
            r = equivalent_sphere_radius(s)
            @test 4π * r^3 / 3 ≈ shape_volume(s)
        end
    end

    @testset "monotone cubic radii, and the bounds really bound" begin
        for p in (0.25, 0.5, 0.9, 1.0, 1.5, 5.0)
            s = Supersphere(1.0, p)
            r1, r2, r3 = s.a, edge_radius(s), diagonal_radius(s)
            @test (r1 - r2) * (r2 - r3) ≥ -1.0e-15      # r2 lies between r1 and r3
            @test r2 ≈ radial_distance(s, _UNIT((1.0, 1.0, 0.0)))
            @test r3 ≈ radial_distance(s, _UNIT((1.0, 1.0, 1.0)))
        end
        @test diagonal_radius(Supersphere(1.0, 1.0)) ≈ 1
        @test diagonal_radius(Supersphere(1.0, 0.5)) ≈ 1 / sqrt(3)

        # The bounds are asserted by sampling, not by trusting the formula --
        # which is the point: for a superspheroid the extremum is generally an
        # interior direction, not a pole and not the equator.
        shapes = vcat(
            [Supersphere(1.0, p) for p in (0.25, 0.5, 1.0, 3.0)],
            [
                Superspheroid(a, c, p) for p in (0.3, 0.5, 1.0, 1.5, 4.0)
                    for (a, c) in ((1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (3.0, 0.4))
            ],
        )
        for s in shapes
            R, r = bounding_radius(s), inner_radius(s)
            rs = [radial_distance(s, n) for n in _DIRS]
            @test maximum(rs) ≤ R + 1.0e-12
            @test minimum(rs) ≥ r - 1.0e-12
            @test r ≤ R
        end

        # And the interior extremum is genuinely reached away from both axes for
        # a shape that has one: m = 4 (p = 2) with a ≠ c.
        s = Superspheroid(1.0, 2.0, 2.0)
        @test bounding_radius(s) > max(s.a, s.c) + 1.0e-6
    end

    @testset "outward normal" begin
        for p in (0.3, 0.5, 1.0, 2.5)
            s = Supersphere(1.0, p)
            for n in _DIRS[1:7:end]
                x = surface_point(s, n)
                nu = outward_normal(s, x)
                @test norm(nu) ≈ 1
                # Against a finite difference of the level set, which is the
                # definition rather than a restatement of the formula.
                g = ForwardDiff.gradient(y -> level_set(s, y), collect(x))
                @test norm(nu .- g ./ norm(g)) < 1.0e-6
                @test sum(nu .* x) > 0        # points outward
            end
        end

        # The creases and the conical points. The raw formula forms `0 * Inf`
        # there for 2p < 1, so what is tested is that a finite unit vector comes
        # out at all, and that it is the documented one. Outwardness is *not*
        # asserted: on a re-entrant crease the one-sided limit normal is
        # tangential to the radius, which is the geometry, not a defect.
        for p in (0.3, 2.5), s in (Supersphere(1.0, p), Superspheroid(1.0, 2.0, p))
            for n in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
                nu = outward_normal(s, surface_point(s, n))
                @test all(isfinite, nu)
                @test norm(nu) ≈ 1
            end
        end

        # The documented conventions, spelled out so a future refactor cannot
        # quietly change them.
        sp = Supersphere(1.0, 0.3)                       # 2p = 0.6 < 1
        @test outward_normal(sp, surface_point(sp, (1.0, 0.0, 0.0))) == (1.0, 0.0, 0.0)
        @test outward_normal(sp, surface_point(sp, (-1.0, 0.0, 0.0))) == (-1.0, 0.0, 0.0)
        xc = surface_point(sp, _UNIT((1.0, 1.0, 0.0)))   # one vanishing coordinate
        @test outward_normal(sp, xc) == (0.0, 0.0, 1.0)
        sq = Superspheroid(1.0, 2.0, 0.3)
        @test outward_normal(sq, surface_point(sq, (1.0, 0.0, 0.0))) == (0.0, 0.0, 1.0)
        @test outward_normal(sq, surface_point(sq, (0.0, 0.0, -1.0))) == (0.0, 0.0, -1.0)

        # For 2p > 1 a vanishing coordinate contributes exactly nothing, so the
        # normal there is the smooth one and *is* outward.
        sf = Supersphere(1.0, 2.5)
        @test outward_normal(sf, surface_point(sf, (1.0, 0.0, 0.0))) == (1.0, 0.0, 0.0)
        @test sum(
            outward_normal(sf, surface_point(sf, _UNIT((1.0, 1.0, 0.0)))) .*
                surface_point(sf, _UNIT((1.0, 1.0, 0.0)))
        ) > 0

        @test_throws ArgumentError outward_normal(sp, (0.0, 0.0, 0.0))
    end

    @testset "automatic differentiation through the geometry" begin
        # Volume, against the analytic derivative of 8a³·D(3,2p) in `a`.
        for p in (0.35, 1.0, 2.0)
            f = a -> shape_volume(Supersphere(a, p))
            @test ForwardDiff.derivative(f, 1.7) ≈ 3 * f(1.7) / 1.7
        end
        # And in `p`, against a central difference -- there is no closed form to
        # compare with, so this is the check that the Γ path differentiates.
        for a in (1.0, 2.0)
            g = p -> shape_volume(Supersphere(a, p))
            h = 1.0e-6
            @test ForwardDiff.derivative(g, 0.7) ≈ (g(0.7 + h) - g(0.7 - h)) / 2h rtol = 1.0e-5
        end
        # Heterogeneous: one Dual argument among plain floats, the defect this
        # package has had to fix repeatedly.
        n = _UNIT((0.3, 0.5, 0.8))
        @test ForwardDiff.derivative(a -> radial_distance(Supersphere(a, 0.4), n), 1.0) ≈
            radial_distance(Supersphere(1.0, 0.4), n)          # homogeneous of degree 1 in a
        @test !iszero(ForwardDiff.derivative(p -> radial_distance(Supersphere(1.0, p), n), 0.4))
        @test !iszero(
            ForwardDiff.derivative(c -> bounding_radius(Superspheroid(1.0, c, 1.5)), 2.0)
        )
        # Through a surface point and a normal, componentwise.
        J = ForwardDiff.jacobian(
            q -> collect(surface_point(Supersphere(q[1], q[2]), n)), [1.0, 0.4]
        )
        @test size(J) == (3, 2)
        @test all(isfinite, J)
        @test !iszero(J)
        @test all(
            isfinite,
            ForwardDiff.jacobian(
                q -> collect(
                    outward_normal(
                        Supersphere(q[1], q[2]),
                        surface_point(Supersphere(q[1], q[2]), n)
                    )
                ),
                [1.0, 0.4],
            ),
        )
    end
end

# The two branches a shape takes when its own parameters stop being ordinary
# numbers. They are not defensive code: the first is how the geometry
# differentiates in its morphology, and the second is how it refuses when it
# cannot. Both were reachable and neither was exercised, which a coverage
# report is good at finding and a test suite is not.
@testset "Supersphere / Superspheroid — non-numeric element types" begin
    @testset "the spheroid normal on a symbolic element type" begin
        # A `Dual` *compares*, so an AD element type still takes the hard
        # branch with its case analysis. The soft branch is for the genuinely
        # symbolic type, where the comparisons are undecidable and the case
        # analysis has to be skipped. It must still return the same normal
        # wherever the hard branch has no case to make -- away from the axis
        # and away from the equator.
        n = _UNIT((0.6, 0.3, 0.7))
        for (a, c, p) in ((1.0, 1.5, 0.7), (1.0, 0.6, 1.4), (1.0, 2.0, 1.0))
            hard = Superspheroid(a, c, p)
            nu = outward_normal(hard, surface_point(hard, n))

            soft = Superspheroid(Sym(a), Sym(c), Sym(p))
            @test TensND.is_hard_numeric(eltype(hard))
            @test !TensND.is_hard_numeric(eltype(soft))
            nv = outward_normal(soft, surface_point(soft, Sym.(n)))
            @test all(Float64.(float.(nv)) .≈ nu)
        end
    end

    @testset "the spheroid normal differentiates in the shape parameters" begin
        # `Dual` takes the hard branch, and that branch has to carry the
        # derivative through its own case analysis. The supersphere had this
        # test; the spheroid did not.
        n = _UNIT((0.6, 0.3, 0.7))
        J = ForwardDiff.jacobian(
            q -> collect(
                outward_normal(
                    Superspheroid(q[1], q[2], 0.7),
                    surface_point(Superspheroid(q[1], q[2], 0.7), n)
                )
            ),
            [1.0, 1.5],
        )
        @test all(isfinite, J)
        @test !iszero(J)

        # Against a central difference, on the component the aspect ratio moves
        # most: the normal is a direction, so this is the honest check that the
        # case analysis did not drop a term.
        nz(c) = outward_normal(
            Superspheroid(1.0, c, 0.7), surface_point(Superspheroid(1.0, c, 0.7), n)
        )[3]
        h = 1.0e-6
        @test ForwardDiff.derivative(nz, 1.5) ≈ (nz(1.5 + h) - nz(1.5 - h)) / 2h rtol = 1.0e-5
    end

    @testset "a predicate refuses rather than answering wrongly" begin
        # `is_sphere` is `p == 1`. On a symbolic `p` SymPy cannot decide that,
        # and an undecidable comparison there returns `false` instead of
        # raising -- so the guard has to raise first, or a symbolic supersphere
        # would silently report itself as never spherical.
        p = SymPy.symbols("p", positive = true)
        s = Supersphere(Sym(1), p)
        @test !TensND.is_hard_numeric(eltype(s))
        @test_throws ArgumentError is_sphere(s)
        @test_throws ArgumentError is_concave(s)
        @test_throws ArgumentError is_convex(s)
    end
end
