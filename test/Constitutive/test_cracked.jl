using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const C0_CR = TensISO{3}(3 * 30.0, 2 * 18.0)      # matrix: k = 30, μ = 18
const CANON_CR = TensND.CanonicalBasis{3, Float64}()
_g(t) = get_array(TensND.change_tens(t, CANON_CR))

# Uniaxial strain along e₃ — the direction normal to the first family, so
# compression closes it.
_ε33(e) = from_tensors(
    TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> (i == j == 3) ? e : 0.0)
)

function _cracked_rve(θs, ds)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0_CR); fraction = :rest)
    for (i, (θ, d)) in enumerate(zip(θs, ds))
        add_phase!(
            rve, Symbol("F", i), PennyCrack(1.0; euler_angles = (θ, 0.0)),
            Dict(:C => C0_CR); density = d
        )
    end
    return rve
end

# Drive a monotone strain path to `ε_end` in `n` equal steps.
function _drive(mat, ε_end, n; cache = MaterialCache())
    st = initial_state(mat)
    r = nothing
    for k in 1:n
        r = material_response(mat, _ε33(ε_end * k / n), st, 0.0; cache = cache)
        st = state(r)
    end
    return (r = r, st = st, cache = cache)
end

@testset "all families open reproduces the scheme" begin
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))
    st = initial_state(mat)
    @test open_set(st) == (true,)
    @test apertures(st) == (1.0e-3,)

    # A tiny increment cannot close anything, so the tangent is the fully
    # cracked stiffness.
    r = material_response(mat, _ε33(-1.0e-8), st, 0.0)
    @test _g(tangent(r)) ≈ _g(homogenize(rve, MoriTanaka())) rtol = 1.0e-12
end

@testset "compression closes a family and stiffens the medium" begin
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))

    open_C = _g(homogenize(rve, MoriTanaka()))[3, 3, 3, 3]
    closed_C = _g(C0_CR)[3, 3, 3, 3]              # a closed crack is invisible
    @test open_C < closed_C                        # premise

    out = _drive(mat, -5.0e-4, 1)
    @test open_set(out.st) == (true,)
    @test 0 < apertures(out.st)[1] < 1.0e-3        # closing, not yet closed
    @test _g(tangent(out.r))[3, 3, 3, 3] ≈ open_C rtol = 1.0e-12

    out = _drive(mat, -2.0e-3, 1)
    @test open_set(out.st) == (false,)
    @test apertures(out.st)[1] == 0.0
    @test _g(tangent(out.r))[3, 3, 3, 3] ≈ closed_C rtol = 1.0e-12
end

@testset "the stress is continuous through closure" begin
    # The signature of a correctly INTEGRATED piecewise-linear law. Computing
    # `C_hom(final) : ε` instead would jump at closure, because the medium only
    # stiffens from the closure point onwards.
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))

    # Bracket the closure strain by bisection on the open/closed flag.
    lo, hi = -2.0e-3, 0.0
    for _ in 1:60
        mid = (lo + hi) / 2
        open_set(_drive(mat, mid, 1).st)[1] ? (hi = mid) : (lo = mid)
    end
    δ = 1.0e-9
    σ_before = _g(stress(_drive(mat, hi + δ, 1).r))[3, 3]
    σ_after = _g(stress(_drive(mat, lo - δ, 1).r))[3, 3]
    @test σ_before ≈ σ_after rtol = 1.0e-4         # continuous across the event

    # …and the closed branch really is stiffer than the open one.
    slope_open = (
        _g(stress(_drive(mat, hi + δ, 1).r))[3, 3] -
            _g(stress(_drive(mat, hi + 1.0e-4, 1).r))[3, 3]
    ) / (-1.0e-4 + δ)
    slope_closed = (
        _g(stress(_drive(mat, lo - 1.0e-4, 1).r))[3, 3] -
            _g(stress(_drive(mat, lo - δ, 1).r))[3, 3]
    ) / (-1.0e-4 + δ)
    @test slope_closed > slope_open
end

@testset "the answer does not depend on the step size" begin
    # THE test for sub-stepping: events are located exactly, so subdividing the
    # same monotone path must change nothing.
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))
    ref = _drive(mat, -2.0e-3, 1)
    σref = _g(stress(ref.r))
    for n in (2, 5, 20, 200)
        out = _drive(mat, -2.0e-3, n)
        @test _g(stress(out.r)) ≈ σref rtol = 1.0e-10
        @test open_set(out.st) == open_set(ref.st)
        @test apertures(out.st)[1] ≈ apertures(ref.st)[1] atol = 1.0e-14
    end

    # Same, with three non-coaxial families and a self-consistent scheme — the
    # configuration that used to make the scheme diverge.
    rve3 = _cracked_rve((0.0, π / 4, π / 3), (0.06, 0.05, 0.04))
    mat3 = MicrocrackedMaterial(rve3, SelfConsistent(); ω₀ = (1.0e-3, 1.2e-3, 8.0e-4))
    ref3 = _drive(mat3, -3.0e-3, 1)
    for n in (4, 25)
        out = _drive(mat3, -3.0e-3, n)
        @test _g(stress(out.r)) ≈ _g(stress(ref3.r)) rtol = 1.0e-8
        @test open_set(out.st) == open_set(ref3.st)
    end
end

@testset "a closed family reopens under tension" begin
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))
    cache = MaterialCache()

    st = state(material_response(mat, _ε33(-3.0e-3), initial_state(mat), 0.0; cache = cache))
    @test open_set(st) == (false,)

    r = material_response(mat, _ε33(1.0e-3), st, 0.0; cache = cache)
    @test open_set(state(r)) == (true,)
    @test apertures(state(r))[1] > 0
    # Reopening is sub-stepped, so the stress reflects both branches: the closed
    # one up to σ_nn = 0, the open one beyond.
    @test _g(stress(r))[3, 3] > 0
end

@testset "the cache is keyed on the open set alone" begin
    rve = _cracked_rve((0.0, π / 4), (0.08, 0.06))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3, 1.0e-3))
    out = _drive(mat, -3.0e-3, 50)
    st = cache_stats(out.cache)
    @test st.entries <= 4                  # 2² configurations at most…
    @test st.hits > st.entries             # …and they are reused, not recomputed
end

@testset "check_material_interface on a smooth branch" begin
    rve = _cracked_rve((0.0,), (0.1,))
    mat = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-2,))
    # Large ω₀ keeps the probe far from closure, where the law is smooth and a
    # finite-difference tangent is meaningful.
    @test check_material_interface(mat; probe = _ε33(-1.0e-6), verbose = false)
end

@testset "guards" begin
    rve_nocrack = RVE()
    add_phase!(rve_nocrack, :M, Ellipsoid(1.0), Dict(:C => C0_CR); fraction = :rest)
    add_phase!(rve_nocrack, :I, Ellipsoid(1.0), Dict(:C => C0_CR); fraction = 0.1)
    @test_throws ArgumentError MicrocrackedMaterial(rve_nocrack, MoriTanaka())

    rve = _cracked_rve((0.0,), (0.1,))
    @test_throws ArgumentError MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3, 2.0e-3))
    @test_throws ArgumentError MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (-1.0e-3,))
end
