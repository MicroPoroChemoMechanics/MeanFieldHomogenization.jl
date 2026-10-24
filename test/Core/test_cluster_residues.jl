using Test
using MeanFieldHomogenization
using TensND

# =============================================================================
#  Residue sums over groups of poles (`Core._cluster_weights`), on which the
#  residue algorithms for the Hill tensor and for cracks rest. Each sum is held
#  to partial fractions evaluated in extended precision, where the cancellation
#  between the residues of close poles costs nothing; and the inner integral of
#  the Hill tensor to a quadrature of the same rational function.
# =============================================================================

const _CR = MeanFieldHomogenization.Core
const _EL = MeanFieldHomogenization.Elasticity

# Σ_{j ∈ S} Res_{wⱼ} f N / D, D = lead Πⱼ (w - wⱼ) with distinct poles, f = log
# or 1: each residue is f(wⱼ) N(wⱼ) / (lead Π_{l ≠ j} (wⱼ - w_l)).
function _residue_sum_big(N, poles, S, lead, logfactor)
    return setprecision(BigFloat, 512) do
        w = Complex{BigFloat}.(poles)
        s = zero(Complex{BigFloat})
        for j in S
            d = Complex{BigFloat}(lead)
            for l in eachindex(w)
                l == j || (d *= w[j] - w[l])
            end
            f = logfactor ? log(w[j]) : one(w[j])
            s += f * evalpoly(w[j], Complex{BigFloat}.(N)) / d
        end
        ComplexF64(s)
    end
end

function _residue_sum_clusters(N, poles, nsum, lead, logfactor)
    s = zero(ComplexF64)
    for members in _CR._pole_clusters(poles, nsum; logfactor)
        c, H_one, H_log = _CR._cluster_weights(poles, members, lead; logfactor)
        H = logfactor ? H_log : H_one
        s += sum(_CR._taylor_at(N, c, length(H)) .* H)
    end
    return s
end

@testset "Taylor coefficients of a polynomial" begin
    a = [2.0, -1.0, 0.5, 3.0]
    c = 0.4 + 0.3im
    T = _CR._taylor_at(a, c, 6)
    @test T[5:6] == zeros(ComplexF64, 2)
    for t in (0.1, -0.2 + 0.05im)
        @test evalpoly(t, T) ≈ evalpoly(c + t, a) rtol = 1.0e-14
    end
end

@testset "residue sums over groups of poles" begin
    N = [0.7, -1.3, 0.4, 2.1, -0.6, 0.9]
    lead = 2.0 + 0.0im
    # Well separated poles, a pair 1e-6 apart, a triple spread over 1e-3: the
    # last two are summed as groups. The poles of the lower half plane bound the
    # groups without being summed.
    upper = ComplexF64[
        -0.5 + 0.4im,
        0.3 + 0.8im, 0.3 + 0.8im + 1.0e-6,
        1.2 + 0.7im, 1.2 + 0.701im, 1.2007 + 0.7004im,
    ]
    poles = vcat(upper, conj.(upper))
    for logfactor in (false, true)
        clusters = @inferred _CR._pole_clusters(poles, length(upper); logfactor)
        @test sort(length.(clusters)) == [1, 2, 3]
        @inferred _CR._cluster_weights(poles, clusters[1], lead; logfactor)
        ref = _residue_sum_big(N, poles, eachindex(upper), lead, logfactor)
        @test _residue_sum_clusters(N, poles, length(upper), lead, logfactor) ≈ ref rtol = 1.0e-13
    end

    # Coincident poles, a triple and a double: the reference spreads each over
    # a regular polygon of radius 1e-20, which leaves the sum unchanged to
    # within 1e-40.
    c3, c2 = 0.2 + 0.5im, 1.0 + 1.0im
    upper = ComplexF64[c3, c3, c3, c2, c2, -0.7 + 0.9im]
    poles = vcat(upper, conj.(upper))
    big_upper = setprecision(BigFloat, 512) do
        δ = big"1e-20"
        vcat(
            [Complex{BigFloat}(c3) + δ * cispi(2 * big(j) / 3) for j in 0:2],
            [Complex{BigFloat}(c2) + δ * cispi(big(j)) for j in 0:1],
            [Complex{BigFloat}(-0.7 + 0.9im)],
        )
    end
    big_poles = vcat(big_upper, conj.(big_upper))
    for logfactor in (false, true)
        ref = setprecision(BigFloat, 512) do
            w = big_poles
            s = zero(Complex{BigFloat})
            for j in eachindex(big_upper)
                d = Complex{BigFloat}(lead)
                for l in eachindex(w)
                    l == j || (d *= w[j] - w[l])
                end
                s += (logfactor ? log(w[j]) : one(w[j])) * evalpoly(w[j], Complex{BigFloat}.(N)) / d
            end
            ComplexF64(s)
        end
        @test _residue_sum_clusters(N, poles, length(upper), lead, logfactor) ≈ ref rtol = 1.0e-13
    end
end

@testset "inner integral of the Hill tensor by residues" begin
    # Wₙ = 2 ∫₀^∞ Eₙ(t) / ((1 + t²)² t⁶ Q(ξ(t))) dt, ξ(t) = (t² - 1)/(2t), with
    # the polynomial Q of a triclinic matrix, and of an isotropic one entered as
    # a general tensor: around a sphere Q ∝ (1 + z²)³, and the eight poles of the
    # upper half plane meet at i.
    quadgk = MeanFieldHomogenization.Core.QuadGK.quadgk
    basis = TensND.CanonicalBasis{3, Float64}()
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    λ, μ = 18.0, 18.0
    KM_iso = [
        λ + 2μ λ λ 0 0 0;
        λ λ + 2μ λ 0 0 0;
        λ λ λ + 2μ 0 0 0;
        0 0 0 2μ 0 0;
        0 0 0 0 2μ 0;
        0 0 0 0 0 2μ
    ]
    arr(C) = [Float64(C[i, j, k, l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
    for (KM, η, ω, φ) in ((KM_tri, 0.7, 0.4, 0.3), (KM_tri, 1.0, 20.0, 1.1), (KM_iso, 1.0, 1.0, 0.8))
        C = arr(TensND.from_mandel(Float64.(KM), basis))
        sys = _CR._build_poly_system(C, [cos(φ), sin(φ) / η, 0.0], [0.0, 0.0, 1 / ω])
        W = @inferred _EL._hill_line_weights(sys.Q, sys.roots_uhp)
        Wq = map(0:6) do n
            f(t) = t^(7 - n) * ((t^2 - 1) / 2)^n / ((1 + t^2)^2 * t^6 * real(sys.Q((t^2 - 1) / (2t))))
            2 * quadgk(f, 0.0, 1.0, Inf; rtol = 1.0e-14)[1]
        end
        @test W ≈ Wq rtol = 1.0e-11
    end
end
