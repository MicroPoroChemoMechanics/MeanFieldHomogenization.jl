# =============================================================================
#  test_laminate_nesting.jl — declarative multiscale chaining (`Homogenized`).
#
#  The contract under test: writing a multiscale model as ONE object must give
#  bit-for-bit what the explicit chaining gives, at any depth, in either
#  direction, and under every scheme — including the iterative ones.
#
#  Coverage:
#   1. Laminate inside an RVE and RVE inside a Laminate, against the explicit
#      chaining, for Mori-Tanaka, self-consistent and differential.
#   2. A three-level chain.
#   3. `property = nothing` inherits the key: one inner cell answers `:C`
#      *and* `:K`.
#   4. Memoization: exactly ONE inner evaluation per (cell, key) and per
#      `homogenize` call, even across ~100 self-consistent iterations.
#   5. `NestedParameter`: get/set round trip, non-mutation, and a ForwardDiff
#      sensitivity crossing every scale in one pass, against finite differences.
#   6. The zero-cost guarantee when nothing is nested.
#   7. The cycle guard.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const MFHC_N = MeanFieldHomogenization.Core
_ison(k, μ) = TensISO{3}(3k, 2μ)

function _inner_laminate(μA = 0.8)
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => _ison(2.0, μA), :K => TensISO{3}(2.0)); fraction = 0.3)
    add_layer!(lam, :B, Dict(:C => _ison(0.5, 0.2), :K => TensISO{3}(0.3)); fraction = 0.7)
    return lam
end

function _outer(prop_value; kw...)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _ison(1.0, 0.4), :K => TensISO{3}(1.0)); fraction = :rest)
    add_phase!(
        rve, :agg, Ellipsoid(1.0), prop_value;
        fraction = 0.3, symmetrize = :iso, kw...
    )
    return rve
end

@testset "Nesting — laminate in RVE == explicit chaining" begin
    lam = _inner_laminate()
    rve_x = _outer(Dict(:C => homogenize(lam, Laminated(), :C)))
    rve_d = _outer(Dict(:C => Homogenized(_inner_laminate(), Laminated())))

    for scheme in (MoriTanaka(), Dilute(), SelfConsistent(), DifferentialScheme())
        @test homogenize(rve_x, scheme, :C) ≈ homogenize(rve_d, scheme, :C)
    end
end

@testset "Nesting — RVE in laminate (the other direction)" begin
    inner = RVE()
    add_phase!(inner, :M, Ellipsoid(1.0), Dict(:C => _ison(3.0, 1.2)); fraction = :rest)
    add_phase!(inner, :pore, Ellipsoid(1.0), Dict(:C => _ison(1.0e-9, 1.0e-9)); fraction = 0.2)

    lam_d = Laminate(; normal = (0, 0, 1))
    add_layer!(lam_d, :A, Dict(:C => Homogenized(inner, MoriTanaka())); fraction = 0.4)
    add_layer!(lam_d, :B, Dict(:C => _ison(0.5, 0.2)); fraction = 0.6)

    lam_x = Laminate(; normal = (0, 0, 1))
    add_layer!(lam_x, :A, Dict(:C => homogenize(inner, MoriTanaka(), :C)); fraction = 0.4)
    add_layer!(lam_x, :B, Dict(:C => _ison(0.5, 0.2)); fraction = 0.6)

    @test homogenize(lam_d, Laminated(), :C) ≈ homogenize(lam_x, Laminated(), :C)
    @test homogenize(lam_d, Voigt(), :C) ≈ homogenize(lam_x, Voigt(), :C)
end

@testset "Nesting — a three-level chain" begin
    # scale 0: an RVE ; scale 1: a laminate containing it ; scale 2: an RVE
    # containing that laminate.
    inner = RVE()
    add_phase!(inner, :M, Ellipsoid(1.0), Dict(:C => _ison(3.0, 1.2)); fraction = :rest)
    add_phase!(inner, :pore, Ellipsoid(1.0), Dict(:C => _ison(1.0e-9, 1.0e-9)); fraction = 0.2)

    mid_d = Laminate(; normal = (0, 0, 1))
    add_layer!(mid_d, :A, Dict(:C => Homogenized(inner, MoriTanaka())); fraction = 0.4)
    add_layer!(mid_d, :B, Dict(:C => _ison(0.5, 0.2)); fraction = 0.6)
    top_d = _outer(Dict(:C => Homogenized(mid_d, Laminated())))

    mid_x = Laminate(; normal = (0, 0, 1))
    add_layer!(mid_x, :A, Dict(:C => homogenize(inner, MoriTanaka(), :C)); fraction = 0.4)
    add_layer!(mid_x, :B, Dict(:C => _ison(0.5, 0.2)); fraction = 0.6)
    top_x = _outer(Dict(:C => homogenize(mid_x, Laminated(), :C)))

    @test homogenize(top_d, MoriTanaka(), :C) ≈ homogenize(top_x, MoriTanaka(), :C)
end

@testset "Nesting — one inner cell answers :C and :K" begin
    lam = _inner_laminate()
    h = Homogenized(lam, Laminated())           # property = nothing → inherits
    rve = _outer(Dict(:C => h, :K => h))
    rve_x = _outer(
        Dict(
            :C => homogenize(lam, Laminated(), :C),
            :K => homogenize(lam, Laminated(), :K),
        )
    )
    @test homogenize(rve, MoriTanaka(), :C) ≈ homogenize(rve_x, MoriTanaka(), :C)
    @test homogenize(rve, MoriTanaka(), :K) ≈ homogenize(rve_x, MoriTanaka(), :K)

    # An explicit `property` reads a different key from the inner cell.
    h2 = Homogenized(lam, Laminated(); property = :K)
    rve2 = RVE()
    add_phase!(rve2, :M, Ellipsoid(1.0), Dict(:X => TensISO{3}(1.0)); fraction = :rest)
    add_phase!(rve2, :agg, Ellipsoid(1.0), Dict(:X => h2); fraction = 0.3, symmetrize = :iso)
    @test homogenize(rve2, MoriTanaka(), :X) ≈
        homogenize(_outer(Dict(:K => homogenize(lam, Laminated(), :K))), MoriTanaka(), :K)
end

# A cell that counts how many times it is homogenized, to observe the
# memoization from the outside.
mutable struct CountingRVE <: MeanFieldHomogenization.Core.AbstractHomogenizationCell
    inner::Any
    count::Int
end
MFHC_N.validate_cell(c::CountingRVE) = (MFHC_N.validate_cell(c.inner); c)
MFHC_N.cell_member_names(c::CountingRVE) = MFHC_N.cell_member_names(c.inner)
MFHC_N.cell_container_property(c::CountingRVE, n::Symbol, k::Symbol) =
    MFHC_N.cell_container_property(c.inner, n, k)
# The scheme argument must be typed `HomogenizationScheme`: leaving it
# untyped would be ambiguous with the error-raising fallback
# `_evaluate(::AbstractHomogenizationCell, ::HomogenizationScheme, ::Val)`,
# which is more specific on the second argument.
function MFHC_N._evaluate(
        c::CountingRVE, scheme::HomogenizationScheme, ::Val{p}; kw...
    ) where {p}
    c.count += 1
    return MFHC_N._evaluate(c.inner, scheme, Val(p); kw...)
end

@testset "Nesting — memoization across an iterative solve" begin
    lam = _inner_laminate()
    counter = CountingRVE(lam, 0)
    rve = _outer(Dict(:C => Homogenized(counter, Laminated())))

    homogenize(rve, MoriTanaka(), :C)
    @test counter.count == 1                     # one solve, whatever the reads

    # The self-consistent fixed point reads the phase properties once per
    # iteration; without the call-scoped cache this would be ~100 solves.
    counter.count = 0
    homogenize(rve, SelfConsistent(), :C)
    @test counter.count == 1

    # A second `homogenize` call opens a FRESH scope: nothing leaks between
    # evaluations (this is what keeps two autodiff tags from colliding).
    homogenize(rve, MoriTanaka(), :C)
    @test counter.count == 2
end

@testset "Nesting — NestedParameter" begin
    rve = _outer(Dict(:C => Homogenized(_inner_laminate(), Laminated())))
    p = nested(:agg, :C, property(:A, :C, :shear))

    @test get_param(rve, p) ≈ 2 * 0.8            # TensISO stores 2μ

    rve2 = set_param(rve, p, 2 * 1.5)
    @test get_param(rve2, p) ≈ 2 * 1.5
    @test get_param(rve, p) ≈ 2 * 0.8            # the original is untouched

    # One ForwardDiff pass across BOTH scales, against central differences.
    function μ_eff(x)
        r = _outer(Dict(:C => homogenize(_inner_laminate(x / 2), Laminated(), :C)))
        return k_mu(homogenize(r, MoriTanaka(), :C))[2]
    end
    ad = derivative(rve, MoriTanaka(), p; indexer = C -> k_mu(C)[2])
    h = 1.0e-6
    @test ad ≈ (μ_eff(1.6 + h) - μ_eff(1.6 - h)) / (2h) rtol = 1.0e-5

    # A nested thickness works the same way.
    pt = nested(:agg, :C, thickness(:A))
    @test get_param(rve, pt) ≈ 0.3
    function μ_eff_h(x)
        lam = Laminate(; normal = (0, 0, 1))
        add_layer!(lam, :A, Dict(:C => _ison(2.0, 0.8)); thickness = x)
        add_layer!(lam, :B, Dict(:C => _ison(0.5, 0.2)); thickness = 0.7)
        r = _outer(Dict(:C => homogenize(lam, Laminated(), :C)))
        return k_mu(homogenize(r, MoriTanaka(), :C))[2]
    end
    ad_h = derivative(rve, MoriTanaka(), pt; indexer = C -> k_mu(C)[2])
    @test ad_h ≈ (μ_eff_h(0.3 + h) - μ_eff_h(0.3 - h)) / (2h) rtol = 1.0e-5

    # The lens refuses a property that is not a nested cell.
    plain = _outer(Dict(:C => _ison(2.0, 0.8)))
    @test_throws ArgumentError get_param(plain, p)
end

@testset "Nesting — zero-cost when nothing is nested" begin
    t = _ison(1.0, 0.4)
    @test MFHC_N.resolve_property(t, :C) === t
    @test MFHC_N.resolve_property(3.5, :anything) === 3.5

    plain = _outer(Dict(:C => t))
    nestedrve = _outer(Dict(:C => Homogenized(_inner_laminate(), Laminated())))
    @test MFHC_N.has_nested_property(plain, :C) == false
    @test MFHC_N.has_nested_property(nestedrve, :C) == true

    # The raw accessor never resolves; the resolving one does.
    @test MFHC_N.cell_container_property(nestedrve, :agg, :C) isa MFHC_N.Homogenized
    @test phase_property_raw(nestedrve, :agg, :C) isa MFHC_N.Homogenized
    @test phase_property(nestedrve, :agg, :C) isa TensND.AbstractTens{4, 3}
end

@testset "Nesting — the cycle guard" begin
    deep = _ison(1.0, 0.4)
    for _ in 1:(MFHC_N.MAX_NESTING + 3)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => deep); fraction = :rest)
        deep = Homogenized(r, Voigt())
    end
    top = RVE()
    add_phase!(top, :M, Ellipsoid(1.0), Dict(:C => deep); fraction = :rest)
    @test_throws ErrorException homogenize(top, Voigt(), :C)
end
