using Test
using MeanFieldHomogenization
using TensND
using Random
import ForwardDiff

# After a warm-up call, the bytes `f(args...)` allocates.
_allocs(f, args...) = (f(args...); @allocated f(args...))

@testset "Hot paths: the default method is resolved at compile time" begin
    C = TensISO{3}(3 * 30.0, 2 * 18.0)
    sph, pen = Ellipsoid(1.0), PennyCrack(1.0)
    # `method = :auto`, the default, used to reach `Val(method)` at run time:
    # a dynamic dispatch and a dynamic keyword call on every Hill tensor and
    # every COD tensor, and a result of type `Any` for every caller.
    @test @inferred(hill_tensor(sph, C)) isa TensISO{4, 3}
    @test @inferred(cod_tensor(pen, C)) isa TensND.AbstractTens{2, 3}
    @test _allocs(hill_tensor, sph, C) == 0
    @test _allocs(cod_tensor, pen, C) == 0
    # A method chosen at run time takes the other branch of `_with_method`,
    # and returns the same tensor to the bit.
    for m in (:auto, :analytical)
        method = Ref(m)[]
        @test get_array(hill_tensor(sph, C; method)) == get_array(hill_tensor(sph, C))
        @test get_array(cod_tensor(pen, C; method)) == get_array(cod_tensor(pen, C))
    end
end

@testset "Hot paths: spheroids and cylinders infer a concrete tensor" begin
    # The symmetry axis of a spheroid or a cylinder is read off its basis. Read as
    # `Tuple(column)`, its length was known at run time only, and Julia 1.14 then
    # inferred the abstract `TensTI` for every tensor built on it, Hill tensors
    # and concentrations included.
    C = TensISO{3}(3 * 30.0, 2 * 18.0)
    shapes = (
        Ellipsoid(3.0, 1.0, 1.0; euler_angles = (0.4, 1.1, 0.2)),
        Ellipsoid(1.0, 1.0, 0.3; euler_angles = (0.9, 0.3, 2.0)),
        Cylinder(1.0, 1.0, TensND.RotatedBasis(0.5, 0.7, 0.1)),
    )
    for g in shapes
        @test isconcretetype(typeof(@inferred hill_tensor(g, C)))
        @test isconcretetype(typeof(@inferred tens_UA(g)))
        @test isconcretetype(typeof(@inferred tens_VA(g)))
        @test @inferred(MeanFieldHomogenization.Core._basis_axis(g.basis, 1)) isa NTuple{3, Float64}
    end
    @test @inferred(laminate_normal(Laminate(; normal = (0.0, 0.6, 0.8)))) isa NTuple{3, Float64}
end

@testset "Hot paths: shapes are built without allocating a sort" begin
    MFH_Core = MeanFieldHomogenization.Core
    # The permutation sorting the semi-axes is `sortperm(collect(t); rev = true)`,
    # equal entries kept in order, on every triple with ties and on random ones.
    vals = (0.0, -0.0, 1.0, 2.0, 1.0e300)
    for a in vals, b in vals, c in vals
        t = (a, b, c)
        @test MFH_Core._descending_order(t) == Tuple(sortperm(collect(t); rev = true))
    end
    rng = Random.Xoshiro(20261025)
    for _ in 1:1000
        t = Tuple(rand(rng, 1:3, 3) .* 1.0)
        @test MFH_Core._descending_order(t) == Tuple(sortperm(collect(t); rev = true))
    end
    # A dual sorts on its value, as `isless` does.
    d = ForwardDiff.Dual{:sort}.((1.0, 1.0, 5.0), (3.0, 2.0, 1.0))
    @test MFH_Core._descending_order(d) == (3, 1, 2)
end
