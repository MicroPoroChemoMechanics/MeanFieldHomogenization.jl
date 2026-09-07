using Test
using MeanFieldHomogenization.LayeredSpheroids: legendre_odd

# =============================================================================
#  test_legendre.jl — associated Legendre recurrences (`legendre.jl`).
#
#  Checked against closed-form low-degree polynomials (P₁, P₃, P₅, P₁¹,
#  Q₁, Q₁¹) and, for the higher-degree / Q¹ entries with no simple
#  closed form at hand, against a central finite-difference derivative
#  — cheap and independent of the recurrence itself.
# =============================================================================

@testset "legendre_odd — P0 (plain Legendre) against closed forms" begin
    x = 1.7
    P0, dP0 = legendre_odd(:P0, x, 3)   # degrees 1, 3, 5
    @test P0[1] ≈ x                         # P₁(x) = x
    @test P0[2] ≈ (5x^3 - 3x) / 2            # P₃(x)
    @test P0[3] ≈ (63x^5 - 70x^3 + 15x) / 8  # P₅(x)
    @test dP0[1] ≈ 1.0
    @test dP0[2] ≈ (15x^2 - 3) / 2
end

@testset "legendre_odd — Q0 against Q₁(x) = x·arccoth(x) - 1" begin
    x = 1.7
    ax = atanh(1 / x)
    Q0, dQ0 = legendre_odd(:Q0, x, 1)
    @test Q0[1] ≈ x * ax - 1
    @test dQ0[1] ≈ ax - x / (x^2 - 1)
end

@testset "legendre_odd — P1 (q branch) against P₁¹(q) = √(q²-1)" begin
    x = 1.7
    P1, dP1 = legendre_odd(:P1, x, 1)
    @test P1[1] ≈ sqrt(x^2 - 1)
    @test dP1[1] ≈ x / sqrt(x^2 - 1)
end

@testset "legendre_odd — P1p (p branch) against P₁¹(p) = -√(1-p²)" begin
    p = 0.3
    P1p, dP1p = legendre_odd(:P1p, p, 1)
    @test P1p[1] ≈ -sqrt(1 - p^2)
    @test dP1p[1] ≈ -p / (-sqrt(1 - p^2))
end

@testset "legendre_odd — derivative tables match finite differences" begin
    # Coarse sanity check only (higher degrees grow fast, so a fixed step
    # trades truncation vs. cancellation error): the precise validation is
    # the closed-form checks above and the coupling-matrix cross-check
    # against the independent BigFloat series in `test_coupling.jl`.
    h = 1.0e-5
    for x in (1.3, 1.7, 3.3), Nseries in (1, 2, 4)
        for kind in (:P0, :Q0, :P1, :Q1)
            vals_p, ders = legendre_odd(kind, x, Nseries)
            vals_hi, _ = legendre_odd(kind, x + h, Nseries)
            vals_lo, _ = legendre_odd(kind, x - h, Nseries)
            fd = (vals_hi .- vals_lo) ./ (2h)
            @test all(isapprox.(fd, ders; rtol = 1.0e-4, atol = 1.0e-6))
        end
        p = 0.3
        P1p, dP1p = legendre_odd(:P1p, p, Nseries)
        P1p_hi, _ = legendre_odd(:P1p, p + h, Nseries)
        P1p_lo, _ = legendre_odd(:P1p, p - h, Nseries)
        fd = (P1p_hi .- P1p_lo) ./ (2h)
        @test all(isapprox.(fd, dP1p; rtol = 1.0e-4, atol = 1.0e-6))
    end
end

@testset "legendre_odd — generic over Complex and BigFloat" begin
    q = im * 2.3   # oblate confocal parameter q = iτ
    P0, _ = legendre_odd(:P0, q, 3)
    @test eltype(P0) <: Complex
    @test P0[1] ≈ q

    xb = big"1.7"
    P0b, _ = legendre_odd(:P0, xb, 3)
    @test eltype(P0b) == BigFloat
    @test P0b[1] ≈ xb
end

@testset "legendre_degrees — arbitrary degree sets" begin
    LSpd = MeanFieldHomogenization.LayeredSpheroids
    x = 1.7

    # Slicing is exact: the reference table is built at the SAME `Nmax` the
    # slicer uses, so any difference would be a bookkeeping error, not roundoff.
    for kind in (:P0, :Q0, :P1, :P1p, :Q1)
        xx = kind === :P1p ? 0.4 : x
        for degs in ([0, 2, 4, 6], [1, 3, 5, 7], [3], [5, 2, 0])
            tab, dtab = LSpd.legendre_table(kind, xx, maximum(degs))
            v, d = LSpd.legendre_degrees(kind, xx, degs)
            @test v == [tab[n + 1] for n in degs]
            @test d == [dtab[n + 1] for n in degs]
        end
    end

    # `legendre_odd` is now a thin wrapper; it must not have moved.
    for kind in (:P0, :Q0, :P1, :Q1), 𝒩 in (1, 4, 9)
        vo, dvo = LSpd.legendre_odd(kind, x, 𝒩)
        vd, dvd = LSpd.legendre_degrees(kind, x, 1:2:(2𝒩 - 1))
        @test vo == vd
        @test dvo == dvd
    end

    @testset "the Q tables depend on Nmax, at roundoff" begin
        # Not a defect, and worth pinning so nobody "fixes" it: `Pₙ` grows with
        # the upward recurrence and is bit-identical whatever `Nmax` is asked
        # for, but `Qₙ` is the MINIMAL solution and is built by Miller's
        # downward recurrence, which starts above the highest degree requested
        # and normalizes on a closed form. Change `Nmax` and the arithmetic path
        # changes with it. The two agree to a few ulp, never bit-for-bit.
        for kind in (:Q0, :Q1)
            t6, _ = LSpd.legendre_table(kind, x, 6)
            t7, _ = LSpd.legendre_table(kind, x, 7)
            @test t6 != t7[1:7]
            @test t6 ≈ t7[1:7] rtol = 1.0e-14
        end
        p6, _ = LSpd.legendre_table(:P0, x, 6)
        p7, _ = LSpd.legendre_table(:P0, x, 7)
        @test p6 == p7[1:7]
    end

    @test_throws ArgumentError LSpd.legendre_degrees(:P0, x, Int[])
    @test_throws ArgumentError LSpd.legendre_degrees(:P0, x, [-1, 2])
    @test_throws ArgumentError LSpd.legendre_table(:nope, x, 3)
end

@testset "order m = 2 — against a 600-bit outside witness" begin
    LSpd = MeanFieldHomogenization.LayeredSpheroids

    # NOT checked against SymPy. Building `Qₙ` by the upward recurrence
    # symbolically and then evaluating it in floating point reproduces exactly
    # the cancellation this module exists to avoid, and such a "reference"
    # reports errors growing with the degree that the implementation does not
    # have. The witness here is the original upward recurrence at 600 bits,
    # which is outside the code under test and numerically sound.
    function witness(kind::Symbol, x::Float64, Nmax::Int)
        return setprecision(BigFloat, 600) do
            b = BigFloat(x)
            q = Vector{BigFloat}(undef, Nmax + 1)
            if kind === :Q2
                x2m1 = b^2 - 1
                ax = atanh(1 / b)
                q[1] = 2b / x2m1
                q[2] = 2 / x2m1
                q[3] = 3 * x2m1 * ax - 3b + 2b / x2m1
            elseif kind === :P2
                q[1] = zero(b); q[2] = zero(b); q[3] = 3 * (b^2 - 1)
            else  # :P2p
                q[1] = zero(b); q[2] = zero(b); q[3] = 3 * (1 - b^2)
            end
            for n in 2:(Nmax - 1)
                q[n + 2] = ((2n + 1) * b * q[n + 1] - (n + 2) * q[n]) / (n - 2 + 1)
            end
            return Float64.(q)
        end
    end

    @testset "Pₙ² is exact on both branches" begin
        for (kind, xs) in ((:P2, (1.05, 1.7, 5.0)), (:P2p, (-0.9, 0.0, 0.37, 0.95)))
            for x in xs
                v, _ = LSpd.legendre_table(kind, x, 8)
                w = witness(kind, x, 8)
                for n in 2:8
                    @test v[n + 1] ≈ w[n + 1] rtol = 1.0e-13 atol = 1.0e-300
                end
            end
        end
    end

    @testset "Qₙ² tracks the witness at every degree" begin
        # The error is uniform in `n`, which is what a normalization error looks
        # like; an instability would grow with the degree.
        for (x, tol) in ((1.02, 1.0e-10), (1.3, 1.0e-11), (2.5, 1.0e-13), (12.0, 1.0e-10))
            v, _ = LSpd.legendre_table(:Q2, x, 8)
            w = witness(:Q2, x, 8)
            errs = [abs(v[n + 1] - w[n + 1]) / abs(w[n + 1]) for n in 0:8]
            @test maximum(errs) < tol
            # uniform: the spread across degrees stays within an order of
            # magnitude of the worst case
            @test maximum(errs) < 30 * max(minimum(errs), 1.0e-16)
        end
    end

    @testset "the m = 2 recurrence starts at n = m, not below" begin
        # Its leading coefficient `n - m + 1` vanishes at `n = m - 1`, so the
        # seed has to reach degree 2. `Q₀²` and `Q₁²` satisfy the degenerate
        # relation `Q₀² = x Q₁²` instead of stepping through it.
        #
        # Note the tolerance: when Miller runs it renormalizes the WHOLE
        # sequence, so the exact closed forms seeded at degrees 0 and 1 come
        # back carrying the `Q₂²` seed's cancellation. That is the right
        # trade — what matters downstream is that the recurrence relation holds
        # between neighboring degrees, not that one degree is exact in
        # isolation.
        x = 3.0
        v, _ = LSpd.legendre_table(:Q2, x, 4)
        @test v[1] ≈ x * v[2] rtol = 1.0e-13
        @test v[2] ≈ 2 / (x^2 - 1) rtol = 1.0e-12
        for kind in (:P2, :P2p)
            vv, _ = LSpd.legendre_table(kind, kind === :P2p ? 0.4 : x, 4)
            @test iszero(vv[1])
            @test iszero(vv[2])
        end
    end

    @testset "generic over the element type" begin
        for kind in (:P2, :Q2)
            vb, db = LSpd.legendre_table(kind, big(2.5), 6)
            vf, df = LSpd.legendre_table(kind, 2.5, 6)
            @test eltype(vb) === BigFloat
            @test Float64.(vb) ≈ vf rtol = 1.0e-11
            @test Float64.(db) ≈ df rtol = 1.0e-11
        end
        vc, _ = LSpd.legendre_table(:Q2, Complex(0.0, 2.5), 5)   # oblate substitution
        @test eltype(vc) <: Complex
    end
end
