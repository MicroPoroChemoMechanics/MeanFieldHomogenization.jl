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
