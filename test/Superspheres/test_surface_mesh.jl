# The triangulated surface of a supershape.
#
# Two oracles carry most of this file, and neither is a golden number.
#
#   * `p = 1/2` **is** the octahedron, so the radial map is the identity, the
#     mesh is exact, and the relaxation sweep must be a fixed point of it. Any
#     of the three ways of getting relaxation wrong breaks that immediately.
#   * the volume enclosed by the triangulation converges to the closed form of
#     `shape_volume`, and one eighth of the full cell must equal the cone over a
#     single octant patch — exactly, not approximately, since the three flat
#     coordinate faces pass through the origin.

using Test
using MeanFieldHomogenization
using LinearAlgebra: norm, cross, dot

_unit3t(v) = v ./ sqrt(v[1]^2 + v[2]^2 + v[3]^2)

@testset "supershape surface mesh" begin

    @testset "the subdivision has the sizes it should" begin
        for L in 0:4
            N = 2^L
            p = octant_patch(L)
            @test node_count(p) == (N + 1) * (N + 2) ÷ 2
            @test triangle_count(p) == 4^L
            o = unit_octahedron(L)
            @test triangle_count(o) == 8 * 4^L
            # Every node of the patch lies on the unit sphere.
            @test all(n -> abs(norm(collect(n)) - 1) < 1.0e-14, p.nodes)
        end
        @test_throws ArgumentError octant_patch(-1)
        @test eltype(octant_patch(1; T = Float32)) === Float32
        @test occursin("TriSurface", sprint(show, octant_patch(1)))
    end

    @testset "the coordinate planes are exact" begin
        # This is the property octant symmetry rests on: a node on a plane has
        # that coordinate as a hard zero, not a small number.
        for L in 0:3
            p = octant_patch(L)
            for (i, c) in enumerate(p.constraint)
                x = p.nodes[i]
                c == 1 && @test x[1] == 0
                c == 2 && @test x[2] == 0
                c == 3 && @test x[3] == 0
                c == 4 && @test count(==(0), x) == 2   # an axis vertex
            end
            # And the reflection merges them rather than doubling them: without
            # normalizing the sign of zero, `-0.0` hashes apart from `0.0` and
            # the three planes carry two copies of every node.
            o = unit_octahedron(L)
            uniq = Set(o.nodes)
            @test length(uniq) == node_count(o)
        end
    end

    @testset "the surface is closed and outward-oriented" begin
        for L in 1:3, shape in (
                    Supersphere(1.0, 0.3), Supersphere(1.0, 1.0),
                    Superspheroid(1.0, 2.0, 0.8),
                )
            s = shape_surface(shape, L)
            # Every edge is shared by exactly two triangles.
            count_of = Dict{Tuple{Int, Int}, Int}()
            for (a, b, c) in s.tris, (i, j) in ((a, b), (b, c), (c, a))
                k = minmax(i, j)
                count_of[k] = get(count_of, k, 0) + 1
            end
            @test all(==(2), values(count_of))
            # Euler characteristic of a sphere.
            @test node_count(s) - length(count_of) + triangle_count(s) == 2
            # Outward, tested against the radial direction — legitimate because
            # the body is star-shaped about the origin.
            for t in s.tris
                a, b, c = collect.((s.nodes[t[1]], s.nodes[t[2]], s.nodes[t[3]]))
                @test dot(cross(b - a, c - a), (a + b + c) / 3) > 0
            end
        end
    end

    @testset "p = 1/2 is exact, and relaxation is a fixed point of it" begin
        s = Supersphere(1.0, 0.5)
        for L in 1:4
            m = shape_surface(s, L)
            # The octahedron is representable, so this is not convergence.
            @test mesh_volume(m) ≈ shape_volume(s) atol = 1.0e-14
            q = mesh_quality(m)
            @test q.min_angle ≈ 60 atol = 1.0e-9
            @test q.ratio ≈ 1 atol = 1.0e-12
        end
        # The fixed point. Getting the tangential projection or the boundary
        # chains wrong turns 60° angles into 30° ones here.
        before = shape_surface(s, 3)
        after = shape_surface(s, 3; relax = 20)
        @test maximum(norm(collect(a) .- collect(b)) for (a, b) in zip(before.nodes, after.nodes)) <
            1.0e-12
        @test mesh_quality(after).min_angle ≈ 60 atol = 1.0e-9
    end

    @testset "one eighth of the cell is the cone over one patch" begin
        for shape in (Supersphere(1.0, 0.3), Supersphere(2.0, 1.0), Supersphere(1.0, 3.0))
            for L in 0:3
                patch = shape_surface(shape, L; octant = true)
                full = shape_surface(shape, L)
                @test 8 * mesh_volume(patch) ≈ mesh_volume(full) rtol = 1.0e-12
            end
        end
    end

    @testset "the meshed volume converges to the closed form" begin
        for shape in (
                Supersphere(1.0, 1.0), Supersphere(1.0, 0.3),
                Superspheroid(1.0, 2.0, 0.7),
            )
            V = shape_volume(shape)
            errs = [abs(mesh_volume(shape_surface(shape, L)) - V) / V for L in 1:4]
            @test issorted(errs; rev = true)          # monotone improvement
            @test errs[end] < errs[1] / 8             # and at a useful rate
            # The concave range converges more slowly, and visibly so: a flat
            # triangle cuts the corner at every crease, and a concave shape is
            # nothing but creases. 5.7 % at level 4 for p = 0.3 against 0.6 %
            # for the sphere.
            @test errs[end] < 0.08
        end
        # Areas have no closed form to compare against, so only
        # self-consistency -- and the *direction* of convergence is worth
        # pinning, because it flips with convexity and the naive expectation is
        # wrong half the time. An inscribed triangulation under-resolves a
        # convex surface, so the area rises to its limit; on a concave one the
        # chords span the re-entrant creases and enclose more surface than there
        # is, so the area falls to it.
        for (shape, rising) in (
                (Supersphere(1.0, 1.0), true), (Supersphere(1.0, 3.0), true),
                (Supersphere(1.0, 0.7), true), (Supersphere(1.0, 0.4), false),
                (Supersphere(1.0, 0.3), false),
            )
            as = [mesh_area(shape_surface(shape, L)) for L in 1:5]
            d = diff(as)
            @test all(rising ? (>(0)) : (<(0)), d)
            @test abs(d[end]) < abs(d[1]) / 8      # and it is converging
        end
        # And p = 1/2 does not move: the octahedron is representable, so the
        # "sequence" is one number plus the round-off of summing thousands of
        # triangle areas (1e-13 relative at level 5, over 8192 of them). The
        # value itself is exact -- a regular octahedron with vertices at unit
        # distance has edge sqrt(2) and area 4 sqrt(3).
        oct = [mesh_area(shape_surface(Supersphere(1.0, 0.5), L)) for L in 1:5]
        @test (maximum(oct) - minimum(oct)) / oct[1] < 1.0e-12
        @test oct[1] ≈ 4 * sqrt(3)
    end

    @testset "relaxation equalizes edge lengths, at a price in angle" begin
        # The raw radial map compresses the mesh near the axis vertices of a
        # concave shape and stretches it near the body diagonal, which is what
        # the sweep is for -- and it is a trade, not a free improvement.
        for shape in (
                Supersphere(1.0, 0.3), Supersphere(1.0, 0.25),
                Superspheroid(1.0, 2.5, 0.35),
            )
            raw = mesh_quality(shape_surface(shape, 3))
            rel = mesh_quality(shape_surface(shape, 3; relax = 30))
            # What the sweep optimizes is edge-length uniformity...
            @test rel.ratio < raw.ratio
            # ...and it pays for it in the smallest angle. Pinned because the
            # trade-off is the documented behavior, not a defect: asserting the
            # angle improves would be asserting something false.
            @test rel.min_angle < raw.min_angle
            @test rel.min_angle > 0.7 * raw.min_angle  # but it must not collapse
            @test rel.min_angle > 5                    # usable as a mesher boundary
        end

        # And the plane constraints survive it exactly — the octant would stop
        # being native otherwise.
        s = Supersphere(1.0, 0.3)
        m = shape_surface(s, 3; relax = 30)
        for (i, c) in enumerate(m.constraint)
            x = m.nodes[i]
            c == 1 && @test x[1] == 0
            c == 2 && @test x[2] == 0
            c == 3 && @test x[3] == 0
        end
        # A pinned axis vertex has not moved at all.
        pinned = findall(==(Int8(4)), m.constraint)
        @test !isempty(pinned)
        for i in pinned
            @test norm(collect(m.nodes[i])) ≈ radial_distance(s, _unit3t(m.nodes[i]))
        end

        @test_throws ArgumentError shape_surface(s, 2; relax = 1, step = 0.0)
        @test_throws ArgumentError shape_surface(s, 2; relax = 1, step = 1.5)
    end

    @testset "a density field biases the node distribution" begin
        s = Supersphere(1.0, 0.6)
        # Cluster toward the +z pole: ρ large there.
        dens = x -> 1 + 20 * max(x[3], 0)^2
        plain = shape_surface(s, 3; relax = 25)
        biased = shape_surface(s, 3; relax = 25, density = dens)
        top(m) = count(n -> n[3] > 0.8 * maximum(p[3] for p in m.nodes), m.nodes)
        @test top(biased) ≥ top(plain)
        @test mesh_quality(biased).min_angle > 1
        # The bias must not open the surface.
        @test mesh_volume(biased) > 0
    end
end

@testset "relaxation leaves a node alone when it cannot be moved" begin
    # Two guards in `relax_surface!` that a well-formed mesh never reaches, and
    # which therefore need asking for. Both say the same thing: a node with
    # nothing usable to move toward stays exactly where it was.
    s = Supersphere(1.0, 0.8)

    # A density that vanishes everywhere makes the weighted centroid undefined.
    # The node must be left in place, not moved to `NaN`.
    a = shape_surface(s, 2; relax = 0)
    b = shape_surface(s, 2; relax = 0)
    relax_surface!(b, s; iterations = 3, density = _ -> 0.0)
    @test all(all(isfinite, p) for p in b.nodes)
    @test b.nodes == a.nodes

    # And the ordinary path does move things, so the test above is not vacuous.
    c = shape_surface(s, 2; relax = 0)
    relax_surface!(c, s; iterations = 3)
    @test c.nodes != a.nodes
    @test mesh_quality(c).hmax / mesh_quality(c).hmin <
        mesh_quality(a).hmax / mesh_quality(a).hmin
end
