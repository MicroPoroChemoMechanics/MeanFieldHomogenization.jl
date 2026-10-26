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
    # The basis of a sorted shape has its dimension in its type: the permuted
    # basis used to be built from a matrix whose size was read at run time.
    B = TensND.CanonicalBasis{3, Float64}()
    U = Union{TensND.CanonicalBasis{3, Float64}, TensND.RotatedBasis{3, Float64}}
    @test @inferred(
        Tuple{NTuple{3, Float64}, U},
        MFH_Core._sort_axes_and_basis((1.0, 1.0, 5.0), B, Val(:ellipsoid_3d))
    )[1] == (5.0, 1.0, 1.0)
end

@testset "Hot paths: a self-consistent step reads the RVE once" begin
    S = MeanFieldHomogenization.Schemes
    C0 = iso_stiffness(30.0, 10.0)
    C1 = iso_stiffness(150.0, 80.0)
    Cn = iso_stiffness(40.0, 15.0)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C1); fraction = 0.3)
    views = S._phase_views(rve, S._volume_phase_names(rve, :M), :C)
    @test views isa Tuple
    # The step on the views is the step read phase by phase from the RVE, to the bit.
    A, CA = S._phase_dilute_and_stress_average(rve, :I, :C, Cn)
    ref = C0 + (zero(Cn) + 0.3 * (CA - C0 ⊡ A))
    step = @inferred S._asc_step_stiffness(views, C0, Cn, (;))
    @test get_array(step) == get_array(ref)
    # What used to be looked up at every step (phase names, fractions, properties,
    # then a dynamic call per phase) is gone from it.
    @test _allocs(S._asc_step_stiffness, views, C0, Cn, (;)) == 0

    # The asymmetric scheme starts from the matrix unless `init` names another
    # start: starting from the matrix explicitly is the default, to the bit.
    seeded = AsymmetricSelfConsistent(; init = C0, reltol = 1.0e-12, abstol = 0.0)
    default = AsymmetricSelfConsistent(; reltol = 1.0e-12, abstol = 0.0)
    @test get_array(homogenize(rve, seeded, :C)) == get_array(homogenize(rve, default, :C))

    # Beyond `_MAX_TUPLE_VIEWS` phases the views are a vector. Nine equal phases
    # are one phase of the total fraction: the same estimate, up to the rounding
    # of the nine-term sums.
    many = RVE()
    add_phase!(many, :M, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    for i in 1:9
        add_phase!(many, Symbol(:I, i), Ellipsoid(1.0), Dict(:C => C1); fraction = 0.3 / 9)
    end
    @test S._phase_views(many, S._volume_phase_names(many, :M), :C) isa Vector
    for sch in (
            AsymmetricSelfConsistent(; reltol = 1.0e-12, abstol = 0.0),
            SelfConsistent(; reltol = 1.0e-12, abstol = 0.0),
        )
        @test get_array(homogenize(many, sch, :C)) ≈ get_array(homogenize(rve, sch, :C)) rtol = 1.0e-12
    end
end
