using Test
using LinearAlgebra
using MeanFieldHomogenization
using TensND
using ForwardDiff
import Symbolics
import DECUHR, Integrals   # the `:decuhr` route lives in an extension

# =============================================================================
#  Derivatives of the crack opening tensor 𝐁 through the residue algorithm
#  (`method = :residues`): the reduced kernel Q̂*ₙₙ is a residue sum of N/Q over
#  the roots of the acoustic polynomial, found on values; its derivatives are
#  residue sums of N (-δQ)ʲ / Q₀ʲ⁺¹ over the same roots. Each is held to an exact
#  identity, to the closed form of the isotropic matrix (a triple root), or to
#  the cubature of the same kernel, an independent evaluation.
# =============================================================================

const _CD_CRK = MeanFieldHomogenization.Cracks
const _CD_KM = [
    210.0 80.0 75.0 5.0 4.0 3.0;
    80.0 195.0 90.0 -2.0 3.0 -1.0;
    75.0 90.0 220.0 1.0 -2.0 2.0;
    5.0 -2.0 1.0 60.0 2.5 1.5;
    4.0 3.0 -2.0 2.5 65.0 -1.0;
    3.0 -1.0 2.0 1.5 -1.0 55.0
]
const _CD_TIGHT = (abstol = 1.0e-13, reltol = 1.0e-11)
_cd_rel(a, b) = norm(a - b) / norm(b)
_cd_stiffness(KM) = TensND.from_mandel(KM, TensND.CanonicalBasis{3, eltype(KM)}())
_cd_cod(crack, C, method) = canonical_components(cod_tensor(crack, C; method = method, _CD_TIGHT...))
# An isotropic stiffness entered as a general tensor.
_cd_general(C) = TensND.Tens(Array(get_array(C)), TensND.CanonicalBasis{3, eltype(C)}())
# Centered difference with the step that balances truncation and rounding,
# h = ε^(1/3) × the scale of the variable: a reference only.
function _cd_centered(f, x, scale = max(abs(x), 1.0))
    h = cbrt(eps()) * scale
    return (f(x + h) - f(x - h)) / (2h)
end
# First derivatives under one tag: `ForwardDiff.derivative` tags its dual numbers
# with the function it differentiates, so that every closure of this file would
# compile the three algorithms again. The nested derivative below goes through
# `ForwardDiff.derivative` itself.
_cd_seed(x) = x
const _CD_TAG = typeof(ForwardDiff.Tag(_cd_seed, Float64))
_cd_derivative(f, x::Float64) = map(d -> ForwardDiff.partials(d, 1), f(ForwardDiff.Dual{_CD_TAG}(x, 1.0)))

@testset "reduced crack kernel by residues with dual arguments" begin
    # The kernel at one direction of the crack plane, with a derivative in the
    # stiffness and one in the direction (two partials): the value is that of the
    # Float64 sums, and the derivatives are those of the quadrature of the same
    # kernel.
    E = zeros(6, 6); E[3, 3] = 1.0; E[2, 5] = E[5, 2] = 0.4
    n̂ = [0.0, 0.0, 1.0]
    components(KM) = Array(TensND.components(_cd_stiffness(KM), TensND.CanonicalBasis{3, eltype(KM)}()))
    direction(y) = [0.7 * cos(0.4 + y), sin(0.4 + y), zero(y)]
    x, y = ForwardDiff.Dual{_CD_TAG}(0.0, 1.0, 0.0), ForwardDiff.Dual{_CD_TAG}(0.0, 0.0, 1.0)
    R = @inferred _CD_CRK._Qnn_star_residue(components(_CD_KM + x * E), direction(y), n̂)
    R₀ = @inferred _CD_CRK._Qnn_star_residue(components(_CD_KM), direction(0.0), n̂)
    @test ForwardDiff.value.(R) ≈ R₀ rtol = 1.0e-14
    Q = _CD_CRK._Qnn_star_nestedquadgk(components(_CD_KM + x * E), direction(y), n̂; abstol = 1.0e-14, reltol = 1.0e-12)
    for k in 1:2
        @test _cd_rel(ForwardDiff.partials.(R, k), ForwardDiff.partials.(Q, k)) < 1.0e-9
    end
    # A tolerance below what the sums reach sends the kernel to its quadrature.
    # The kernel is homogeneous of degree 1 in ℂ: its derivative along ℂ is itself.
    C_dual, ξ = components(_CD_KM .* ForwardDiff.Dual{_CD_TAG}(1.0, 1.0)), direction(0.0)
    F = _CD_CRK._Qnn_star_residue(C_dual, ξ, n̂; rtol = 5.0e-16)
    @test F != _CD_CRK._Qnn_star_residue(C_dual, ξ, n̂)
    @test ForwardDiff.value.(F) ≈ R₀ rtol = 1.0e-12
    @test ForwardDiff.partials.(F, 1) ≈ ForwardDiff.value.(F) rtol = 1.0e-9
end

@testset "𝐁 by residues: derivatives on exact identities" begin
    # 𝐁(λℂ) = 𝐁(ℂ)/λ: d𝐁/dλ = -𝐁 at λ = 1, with the crack along the axes or not.
    C₀ = _cd_stiffness(_CD_KM)
    ℓ, m, n = [0.36, 0.48, 0.8], [0.8, -0.6, 0.0], [0.48, 0.64, -0.6]
    for crack in (EllipticCrack(1.0, 0.45), EllipticCrack(1.0, 0.45, hcat(ℓ, m, n)), RibbonCrack(1.0), RibbonCrack(0.7, hcat(ℓ, m, n)))
        B = _cd_cod(crack, C₀, :residues)
        dB = cod_tensor(crack, ForwardDiff.Dual{_CD_TAG}(1.0, 1.0) * C₀; method = :residues, _CD_TIGHT...)
        @test ForwardDiff.partials.(canonical_components(dB), 1) ≈ -B rtol = 1.0e-9
        # The value under dual numbers is that of the Float64 sums.
        @test ForwardDiff.value.(canonical_components(dB)) ≈ B rtol = 1.0e-13
    end
    # Two nested levels, through `ForwardDiff.derivative`: d²𝐁/dλ² = 2𝐁.
    for crack in (EllipticCrack(1.0, 0.45), RibbonCrack(1.0))
        f(λ) = canonical_components(cod_tensor(crack, λ * C₀; method = :residues))
        @test ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), 1.0) ≈ 2 * f(1.0) rtol = 1.0e-6
    end
end

@testset "𝐁 by residues: derivatives against the cubatures, a centered difference and the closed form" begin
    E = zeros(6, 6); E[1, 1] = 1.0; E[1, 4] = E[4, 1] = 0.5; E[5, 6] = E[6, 5] = -0.3
    stiffness(x) = _cd_stiffness(_CD_KM + x * E)
    frame = hcat([0.36, 0.48, 0.8], [0.8, -0.6, 0.0], [0.48, 0.64, -0.6])
    # The stiffness and the aspect ratio of an elliptic crack, the stiffness for a
    # ribbon, which has no `:decuhr` algorithm.
    paths = (
        (x -> (EllipticCrack(one(x), 0.45 * one(x), frame), stiffness(x)), 100.0, true),
        (x -> (EllipticCrack(one(x), 0.45 + x, frame), stiffness(zero(x))), 1.0, true),
        (x -> (RibbonCrack(one(x), frame), stiffness(x)), 100.0, false),
    )
    # At a tolerance where every direction is summed by residues; tight ones for
    # the references.
    residues(crack, C) = canonical_components(cod_tensor(crack, C; method = :residues, abstol = 1.0e-12, reltol = 1.0e-9))
    decuhr(crack, C) = canonical_components(cod_tensor(crack, C; method = :decuhr, abstol = 1.0e-10, reltol = 1.0e-8))
    for (path, scale, has_decuhr) in paths
        d_res = _cd_derivative(x -> residues(path(x)...), 0.0)
        @test _cd_rel(d_res, _cd_derivative(x -> _cd_cod(path(x)..., :nestedquadgk), 0.0)) < 1.0e-8
        has_decuhr && @test _cd_rel(d_res, _cd_derivative(x -> decuhr(path(x)...), 0.0)) < 1.0e-6
        @test _cd_rel(d_res, _cd_centered(x -> _cd_cod(path(x)..., :residues), 0.0, scale)) < 1.0e-7
    end
    # Isotropic values in a general tensor: a triple root of Q.
    iso(μ) = TensISO{3}(3 * 30.0 * one(μ), 2μ)
    for crack in (EllipticCrack(1.0, 0.45), RibbonCrack(1.0))
        f(μ) = residues(crack, _cd_general(iso(μ)))
        d_res = _cd_derivative(f, 18.0)
        @test _cd_rel(d_res, _cd_derivative(μ -> canonical_components(cod_tensor(crack, iso(μ))), 18.0)) < 1.0e-8
        @test _cd_rel(d_res, _cd_centered(μ -> _cd_cod(crack, _cd_general(iso(μ)), :residues), 18.0)) < 1.0e-7
    end
end

@testset "compliance differentiated through the residues" begin
    # ℍ = k n̲ ⊗ˢ 𝐁 ⊗ˢ n̲ is linear in 𝐁: homogeneous of degree -1 in ℂ.
    C₀ = _cd_stiffness(_CD_KM)
    crack = EllipticCrack(1.0, 0.45)
    H(λ) = canonical_components(inverse_property_contribution(crack, λ * C₀; method = :residues, _CD_TIGHT...))
    @test _cd_derivative(H, 1.0) ≈ -H(1.0) rtol = 1.0e-9
end

@testset "𝐁 by residues: values that are not Float64" begin
    crack = EllipticCrack(1.0, 0.45)
    Symbolics.@variables x
    @test_throws r"method = :nestedquadgk" cod_tensor(crack, _cd_stiffness(Symbolics.Num.(_CD_KM) .* x); method = :residues)
    @test_throws r"method = :nestedquadgk" cod_tensor(crack, _cd_stiffness(big.(_CD_KM)); method = :residues)
    @test_throws r"method = :nestedquadgk" cod_tensor(RibbonCrack(big(1.0)), _cd_stiffness(_CD_KM); method = :residues)
end
