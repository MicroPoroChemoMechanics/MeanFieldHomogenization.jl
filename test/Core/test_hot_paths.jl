using Test
using MeanFieldHomogenization
using TensND

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
