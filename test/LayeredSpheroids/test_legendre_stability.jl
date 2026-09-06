# =============================================================================
#  test_legendre_stability.jl — the two numerical defects that made an
#  n-layer spheroid diverge away from the particle, and made raising
#  `Nseries` to gain accuracy lose it instead.
#
#   1. `Qₙ` is the MINIMAL solution of the Legendre three-term recurrence, so
#      running it upward amplifies the seed's rounding error by `ρ^{2n}` with
#      `ρ = |x + √(x²−1)|`.  Measured against the same recurrence at 600 bits,
#      `Q₁₅(5)` was wrong by a relative `1.3e13` and `Q₁₅(50)` by `7.9e26`.
#      It is now run downward (Miller) whenever `ρ` makes upward unsafe — and
#      kept upward when `ρ ≈ 1`, where Miller would need thousands of steps
#      and upward loses nothing.
#
#   2. In the matrix the growing amplitudes above degree 1 must vanish
#      identically (regularity at infinity).  Recomputing them from the layer
#      transfer left the linear solve's `O(1e-17)` residue, which `P_{2r-1}(q)`
#      then amplified without bound.
#
#  The reference below is deliberately the ORIGINAL upward recurrence at 600
#  bits: an outside check, not the shipped algorithm evaluated in wider
#  precision, which would only confirm itself.
# =============================================================================

using Test
using MeanFieldHomogenization
using MeanFieldHomogenization.LayeredSpheroids: legendre_odd, spheroid_state_sequence
using TensND
using LinearAlgebra

_arccoth_ref(x) = atanh(one(x) / x)

"`Qₙ(x)`, `n = 0..Nmax`, by the plain upward recurrence — accurate at 600 bits."
function _ref_Q0(x, Nmax)
    ax = _arccoth_ref(x)
    tab = [ax, x * ax - one(x)]
    for n in 1:(Nmax - 1)
        push!(tab, ((2n + 1) * x * tab[n + 1] - n * tab[n]) / (n + 1))
    end
    return tab
end

"`Qₙ¹(x)`, `n = 0..Nmax`, by the plain upward recurrence — accurate at 600 bits."
function _ref_Q1(x, Nmax)
    ax = _arccoth_ref(x)
    xb = sqrt(x^2 - one(x))
    x2 = x^2
    x2m1 = x2 - one(x)
    tab = [zero(x), xb * ax - x / xb, x * xb * (3 * ax - (3 * x2 - 2) / (x * x2m1))]
    for n in 2:(Nmax - 1)
        push!(tab, ((2n + 1) * x * tab[n + 1] - (n + 1) * tab[n]) / n)
    end
    return tab
end

@testset "Legendre Q — stable to high degree, against a 600-bit reference" begin
    setprecision(BigFloat, 600) do
        N = 12
        cases = (
            ("prolate, ρ≈1", 1.02, BigFloat(102) / 100),
            ("prolate", 1.5, BigFloat(3) / 2),
            ("prolate", 5.0, BigFloat(5)),
            ("prolate", 50.0, BigFloat(50)),
            ("oblate, ρ≈1", im * 0.02, Complex{BigFloat}(0, BigFloat(2) / 100)),
            ("oblate", im * 1.5, Complex{BigFloat}(0, BigFloat(3) / 2)),
            ("oblate", im * 5.0, Complex{BigFloat}(0, BigFloat(5))),
            ("oblate", im * 50.0, Complex{BigFloat}(0, BigFloat(50))),
        )
        for (_, x, xb) in cases
            v0, _ = legendre_odd(:Q0, x, N)
            v1, _ = legendre_odd(:Q1, x, N)
            r0 = _ref_Q0(xb, 2N)
            r1 = _ref_Q1(xb, 2N)
            for r in 1:N
                deg = 2r - 1
                @test abs(v0[r] - r0[deg + 1]) / abs(r0[deg + 1]) < 1.0e-13
                @test abs(v1[r] - r1[deg + 1]) / abs(r1[deg + 1]) < 1.0e-10
            end
        end
    end
end

@testset "Legendre Q — the derivative identity" begin
    # The derivatives are built from
    #     (x² − 1) dQₙᵐ/dx = n x Qₙᵐ − (n + m) Qₙ₋₁ᵐ,
    # which needs consecutive degrees; check it on the full tables rather than
    # on the odd-degree view.  For `m = 1` the identity starts at `n = 2`: the
    # `m = 1` recurrence is singular at `n = 0`, so `Q₀¹` is carried as a
    # placeholder zero that the identity would otherwise pick up — a trap that
    # made an earlier version of this check report a factor-3 discrepancy.
    Q0t = MeanFieldHomogenization.LayeredSpheroids._Q0_table
    Q1t = MeanFieldHomogenization.LayeredSpheroids._Q1_table
    for x in (1.02, 1.5, 5.0, 50.0, im * 0.02, im * 1.5, im * 5.0, im * 50.0)
        for (tabf, m, n0) in ((Q0t, 0, 1), (Q1t, 1, 2))
            tab, dtab = tabf(x, 16)
            x2m1 = x^2 - 1
            for n in n0:16
                lhs = x2m1 * dtab[n + 1]
                rhs = n * x * tab[n + 1] - (n + m) * tab[n]
                @test abs(lhs - rhs) / max(abs(rhs), 1.0e-300) < 1.0e-9
            end
        end
    end
end

@testset "spheroid — the far field is regular, and Nseries-independent" begin
    # The symptom both fixes remove: away from the particle the axial gradient
    # must return to the remote one, and the answer must not depend on where
    # the series is truncated.  Before, `Nseries = 12` gave `-3.4e24` at 100
    # particle radii, and raising `Nseries` made it worse.
    a, t = 1.0, 2.0
    cbar = sqrt(t^2 - a^2)
    k₀ = TensISO{3}(1.0)
    ref = nothing
    for ns in (3, 5, 8, 12, 20)
        s = LayeredSpheroid(
            (a,), (t,), (TensISO{3}(1.0e-6),);
            interfaces = (SurfaceConductiveInterface(3.0),), Nseries = ns,
        )
        for z in (10.0, 100.0, 2000.0)
            g = collect(local_gradient(s, k₀, im * (z / cbar), 0.7, 0.0; H_axial = 1.0, H_trans = 0.0))
            @test all(isfinite, real.(g))
            @test real(g[3]) ≈ 1.0 atol = 1.0e-3
        end
        # Nseries-independence, the property that used to fail catastrophically.
        g10 = real(
            collect(
                local_gradient(s, k₀, im * (10.0 / cbar), 0.7, 0.0; H_axial = 1.0, H_trans = 0.0)
            )[3]
        )
        ref === nothing && (ref = g10)
        @test g10 ≈ ref rtol = 1.0e-6
    end
end

@testset "spheroid — regularity at infinity is imposed, not recovered" begin
    # Every GROWING amplitude in the matrix above degree 1 is exactly zero.
    for ns in (3, 8, 12)
        s = LayeredSpheroid(
            (1.0,), (2.0,), (TensISO{3}(1.0e-6),);
            interfaces = (SurfaceConductiveInterface(3.0),), Nseries = ns,
        )
        X = spheroid_state_sequence(s, TensISO{3}(1.0), false)
        A_matrix = X[end][1:ns]
        @test A_matrix[1] == 1                        # the imposed remote field
        @test all(iszero, A_matrix[2:end])            # exact zeros, not O(eps)
    end
end

@testset "spheroid — a single layer still matches Eshelby exactly" begin
    # The guard against over-correcting: for a single confocal spheroid with a
    # perfect interface the series is exact at degree 1, so the result must
    # equal the closed form at ANY truncation — including the 1:60 flat disc,
    # where an under-converged Miller silently returned only 7 digits.
    K0 = TensISO{3}(2.0)
    for (a, b, k1) in ((3.0, 1.0, 5.0), (0.5, 2.0, 5.0), (10.0, 0.3, 8.0), (0.05, 3.0, 8.0))
        K1 = TensISO{3}(k1)
        A_classic = inv(TensISO{3}(1.0) + hill_tensor(Spheroid(a / b), K0) ⋅ (K1 - K0))
        for ns in (2, 6, 16)
            s = LayeredSpheroid((a,), (b,), (K1,); Nseries = ns, axis = (0.0, 0.0, 1.0))
            @test get_array(gradient_gradient_loc(s, K1, K0)) ≈
                get_array(A_classic) rtol = 1.0e-12
        end
    end
end
