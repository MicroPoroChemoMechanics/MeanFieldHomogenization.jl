# =============================================================================
#  test_rve.jl
#
#  Validates the RVE / Phase / Amount data model:
#   1. Construction and progressive registration of matrix + inclusion phases.
#   2. Matrix volume fraction is implicit and ignores crack densities.
#   3. Distribution-shape coercion (nothing → UniformDistribution sphere ;
#      AbstractInclusion auto-wrapped ; passthrough for explicit
#      AbstractDistributionShape).
#   4. Argument validation: duplicate names, missing matrix, mutually
#      exclusive fraction/density kwargs, negative amounts.
#   5. Element-type propagation : Float64 (default), ForwardDiff.Dual,
#      Complex{Float64}.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

@testset "RVE — basic construction & accessors" begin
    rve = RVE()
    @test rve isa RVE{Float64}
    @test isempty(rve.phase_names)
    @test rve.distribution_shape isa UniformDistribution

    # Matrix
    C₀ = TensISO{3}(30.0, 10.0)
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    @test rve.phases[:M] isa Phase
    @test phase_property(rve, :M, :C) === C₀

    # Inclusion (volume fraction)
    C₁ = TensISO{3}(60.0, 20.0)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.3), Dict(:C => C₁); fraction = 0.2)
    @test rve.amounts[:I] isa VolumeFraction
    @test volume_fraction(rve, :I) ≈ 0.2
    @test crack_density(rve, :I) == 0.0

    # Crack (density)
    add_phase!(rve, :CRACK, PennyCrack(1.0), Dict(:C => C₀); density = 0.05)
    @test rve.amounts[:CRACK] isa CrackDensity
    @test crack_density(rve, :CRACK) ≈ 0.05
    @test volume_fraction(rve, :CRACK) == 0.0

    # Implicit matrix fraction = 1 - 0.2 (cracks excluded from sum)
    @test remainder_volume_fraction(rve) ≈ 0.8
    @test volume_fraction(rve, :M) ≈ 0.8

    # Insertion order respected
    @test phase_names(rve) == [:M, :I, :CRACK]
    @test inclusion_phase_names(rve, :M) == [:I, :CRACK]
    @test rve.phase_names == [:M, :I, :CRACK]

    # No-op validation succeeds
    @test validate_rve(rve) === rve
end

@testset "RVE — argument errors" begin
    rve = RVE()
    C₀ = TensISO{3}(30.0, 10.0)

    # Cannot add a phase before the matrix...
    @test_throws ArgumentError validate_rve(rve)

    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    # Cannot add the matrix twice
    @test_throws ArgumentError add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    # Cannot use the matrix name for a phase
    @test_throws ArgumentError add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.1)

    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.2)
    # Duplicate phase name
    @test_throws ArgumentError add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.1)
    # Must specify exactly one of fraction / density
    @test_throws ArgumentError add_phase!(rve, :J, Ellipsoid(1.0), Dict(:C => C₀))
    @test_throws ArgumentError add_phase!(
        rve, :J, Ellipsoid(1.0), Dict(:C => C₀);
        fraction = 0.1, density = 0.1
    )
    # Negative amount caught by validate_rve
    add_phase!(rve, :J, Ellipsoid(1.0), Dict(:C => C₀); fraction = -0.1)
    @test_throws ArgumentError validate_rve(rve)
end

@testset "RVE — distribution_shape coercion" begin
    # Default → UniformDistribution(Ellipsoid(1.0))
    rve_default = RVE()
    @test rve_default.distribution_shape isa UniformDistribution
    @test rve_default.distribution_shape.shape isa Ellipsoid

    # AbstractInclusion auto-wrapped
    ell = Ellipsoid(2.0, 1.0, 1.0)
    rve_wrap = RVE(; distribution_shape = ell)
    @test rve_wrap.distribution_shape isa UniformDistribution
    @test rve_wrap.distribution_shape.shape === ell

    # Explicit AbstractDistributionShape passthrough
    ds = UniformDistribution(Ellipsoid(1.0, 1.0, 0.5))
    rve_explicit = RVE(; distribution_shape = ds)
    @test rve_explicit.distribution_shape === ds
end

@testset "RVE — element-type propagation" begin
    # ForwardDiff.Dual amounts (sensitivity analysis on volume fraction)
    DT = ForwardDiff.Dual{Nothing, Float64, 1}
    rve_dual = RVE(; T = DT)
    @test rve_dual isa RVE{DT}
    C₀ = TensISO{3}(30.0, 10.0)
    add_phase!(rve_dual, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(
        rve_dual, :I, Ellipsoid(1.0), Dict(:C => C₀);
        fraction = ForwardDiff.Dual{Nothing}(0.2, 1.0)
    )
    @test rve_dual.amounts[:I] isa VolumeFraction{DT}
    @test ForwardDiff.value(remainder_volume_fraction(rve_dual)) ≈ 0.8
    @test ForwardDiff.partials(remainder_volume_fraction(rve_dual))[1] ≈ -1.0

    # Complex{Float64} amounts (frequency-domain volume-fraction sweep —
    # rarely useful, but compatibility must hold).
    rve_c = RVE(; T = Complex{Float64})
    add_phase!(rve_c, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(
        rve_c, :I, Ellipsoid(1.0), Dict(:C => C₀);
        fraction = 0.2 + 0.0im
    )
    @test rve_c.amounts[:I] isa VolumeFraction{Complex{Float64}}
    @test remainder_volume_fraction(rve_c) ≈ 0.8 + 0.0im

    # Parametric constructor: strictly equivalent to the `T = …` keyword.
    @test RVE{Complex{Float64}}() isa RVE{Complex{Float64}}
    @test typeof(RVE{DT}()) === typeof(RVE(; T = DT))
    ds = UniformDistribution(Ellipsoid(1.0, 1.0, 0.5))
    @test RVE{Float64}(; distribution_shape = ds).distribution_shape === ds
end

@testset "RVE — heterogeneous amounts (declared T is a floor, not a cast)" begin
    C₀ = TensISO{3}(30.0, 10.0)
    DT = ForwardDiff.Dual{Nothing, Float64, 1}
    d = ForwardDiff.Dual{Nothing}(0.2, 1.0)

    # An amount WIDER than the declared floor is stored as such, not narrowed:
    # no `T = …` is needed to differentiate w.r.t. a fraction, or to sweep a
    # complex one.
    rve = RVE()                                # floor = Float64
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = d)
    @test rve.amounts[:I] isa VolumeFraction{DT}
    @test eltype(rve) === DT                     # effective eltype
    @test eltype(typeof(rve)) === Float64        # declared floor
    @test ForwardDiff.partials(remainder_volume_fraction(rve))[1] ≈ -1.0

    rve_c = RVE()
    add_phase!(rve_c, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve_c, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.2 + 0.01im)
    @test rve_c.amounts[:I] isa VolumeFraction{Complex{Float64}}
    @test remainder_volume_fraction(rve_c) ≈ 0.8 - 0.01im

    # Amounts of different element types coexist in one RVE and promote only
    # where the values meet.
    rve_mix = RVE()
    add_phase!(rve_mix, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve_mix, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = d)
    add_phase!(rve_mix, :J, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.1)
    add_phase!(rve_mix, :CRACK, PennyCrack(1.0), Dict(:C => C₀); density = 0.05)
    @test rve_mix.amounts[:I] isa VolumeFraction{DT}
    @test rve_mix.amounts[:J] isa VolumeFraction{Float64}
    @test rve_mix.amounts[:CRACK] isa CrackDensity{Float64}
    @test eltype(rve_mix) === DT
    fm = remainder_volume_fraction(rve_mix)         # cracks excluded from the sum
    @test ForwardDiff.value(fm) ≈ 0.7
    @test ForwardDiff.partials(fm)[1] ≈ -1.0
    @test volume_fraction(rve_mix, :J) ≈ 0.1
    @test crack_density(rve_mix, :CRACK) ≈ 0.05

    # A NARROWER amount is still widened to the declared floor (no silent
    # narrowing of the RVE, and `Int` fractions keep behaving as before).
    rve_floor = RVE{Complex{Float64}}()
    add_phase!(rve_floor, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve_floor, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.2)
    @test rve_floor.amounts[:I] isa VolumeFraction{Complex{Float64}}

    rve_int = RVE()
    add_phase!(rve_int, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve_int, :I, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0)
    @test remainder_volume_fraction(rve_int) === 1.0

    # promote_rve / convert force a floor after the fact.
    rve_p = promote_rve(rve_int, Complex{Float64})
    @test rve_p isa RVE{Complex{Float64}}
    @test rve_p.amounts[:I] isa VolumeFraction{Complex{Float64}}
    @test remainder_volume_fraction(rve_p) === Complex{Float64}(1.0)
    @test convert(RVE{Complex{Float64}}, rve_int) isa RVE{Complex{Float64}}
    @test convert(RVE{Float64}, rve_int) === rve_int
end

@testset "RVE — Amounts unit tests" begin
    # Construction
    @test VolumeFraction(0.3) isa VolumeFraction{Float64}
    @test CrackDensity(0.05) isa CrackDensity{Float64}

    # _sums_to_unit dispatch
    @test MeanFieldHomogenization.Schemes._sums_to_unit(VolumeFraction(0.3))
    @test !MeanFieldHomogenization.Schemes._sums_to_unit(CrackDensity(0.05))

    # eltype
    @test eltype(VolumeFraction(0.3)) === Float64
    @test eltype(CrackDensity{Complex{Float64}}(0.0 + 0.0im)) === Complex{Float64}

    # amount_value
    @test MeanFieldHomogenization.Schemes.amount_value(VolumeFraction(0.3)) ≈ 0.3
    @test MeanFieldHomogenization.Schemes.amount_value(CrackDensity(0.05)) ≈ 0.05
end
