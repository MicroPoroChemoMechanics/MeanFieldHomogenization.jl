using Test
using LinearAlgebra
using MeanFieldHomogenization
using TensND
using ForwardDiff
import Symbolics
import DECUHR, Integrals   # the `:decuhr` route lives in an extension

# =============================================================================
#  Derivatives through the residue algorithm (`method = :residues`).
#
#  The roots of the acoustic polynomial Q are found on the values of its
#  coefficients and never differentiated: with Q = Q₀ + δQ, the derivatives come
#  from residue sums of N (-δQ)ʲ / Q₀ʲ⁺¹ over the roots of Q₀. Each derivative is
#  held to an exact identity, to the closed form where the matrix is isotropic
#  (a triple root), to the two cubatures, which are independent algorithms, and
#  to a centered difference of the residue algorithm itself in Float64.
# =============================================================================

const _RD_EL = MeanFieldHomogenization.Elasticity
const _RD_CR = MeanFieldHomogenization.Core
const _RD_KM = [
    210.0 80.0 75.0 5.0 4.0 3.0;
    80.0 195.0 90.0 -2.0 3.0 -1.0;
    75.0 90.0 220.0 1.0 -2.0 2.0;
    5.0 -2.0 1.0 60.0 2.5 1.5;
    4.0 3.0 -2.0 2.5 65.0 -1.0;
    3.0 -1.0 2.0 1.5 -1.0 55.0
]
const _RD_TIGHT = (abstol = 1.0e-13, reltol = 1.0e-11)
_rd_canon(P) = TensND.get_array(TensND.canonical(P))
_rd_rel(a, b) = norm(a - b) / norm(b)
_rd_stiffness(KM) = TensND.from_mandel(KM, TensND.CanonicalBasis{3, eltype(KM)}())
_rd_hill(ell, C, method) = _rd_canon(hill_tensor(ell, C; method = method, _RD_TIGHT...))
# The adaptive cubature at the tolerances it reaches in a reasonable time.
_rd_hill_decuhr(ell, C) = _rd_canon(hill_tensor(ell, C; method = :decuhr, abstol = 1.0e-10, reltol = 1.0e-8))
# An isotropic stiffness entered as a general tensor, which the anisotropic
# algorithms take as it is.
_rd_general(C) = TensND.Tens(Array(get_array(C)), TensND.CanonicalBasis{3, eltype(C)}())
# Centered difference with the step that balances truncation (h² f‴/6) and
# rounding (ε f/h), h = ε^(1/3) × the scale of the variable: a reference only.
function _rd_centered(f, x, scale = max(abs(x), 1.0))
    h = cbrt(eps()) * scale
    return (f(x + h) - f(x - h)) / (2h)
end
# First derivatives under one tag. `ForwardDiff.derivative` tags its dual numbers
# with the function it differentiates, so that every closure of this file would
# compile the three algorithms again; the nested derivatives below go through
# `ForwardDiff.derivative` itself.
_rd_seed(x) = x
_rd_seed2(x) = x
const _RD_TAG = typeof(ForwardDiff.Tag(_rd_seed, Float64))
_rd_derivative(f, x::Float64) = map(d -> ForwardDiff.partials(d, 1), f(ForwardDiff.Dual{_RD_TAG}(x, 1.0)))
# Second derivatives under two fixed tags as well, the outer one ordered after
# the inner one as `ForwardDiff` orders its nested tags.
const _RD_TAG2 = typeof(ForwardDiff.Tag(_rd_seed2, ForwardDiff.Dual{_RD_TAG, Float64, 1}))
function _rd_second(f, x::Float64)
    inner(v, dv) = ForwardDiff.Dual{_RD_TAG}(v, dv)
    y = f(ForwardDiff.Dual{_RD_TAG2}(inner(x, 1.0), inner(1.0, 0.0)))
    return map(d -> ForwardDiff.partials(ForwardDiff.partials(d, 1), 1), y)
end

@testset "moments of the derivatives of the inner integral" begin
    # W⁽ʲ⁾ₘ = 2 ∫₀^∞ t^(7+6j-m) vᵐ / ((1 + t²)² (t⁶ Q₀(ξ(t)))ʲ⁺¹) dt, v = (t² - 1)/2,
    # ξ = v/t: the residue sums over the poles of multiplicity j + 1 against a
    # quadrature of the definition, for separated roots and for a triple one. Around
    # a sphere in an isotropic matrix the triple root is i, the double pole of the
    # integrand: the sums are ill-conditioned there, as those of the weights
    # themselves, and their bound says by how much.
    quadgk = _RD_CR.QuadGK.quadgk
    λ, μ = 18.0, 18.0
    KM_iso = [
        λ + 2μ λ λ 0 0 0;
        λ λ + 2μ λ 0 0 0;
        λ λ λ + 2μ 0 0 0;
        0 0 0 2μ 0 0;
        0 0 0 0 2μ 0;
        0 0 0 0 0 2μ
    ]
    cases = (
        (_RD_KM, 0.7, 0.4, 0.3, true), (_RD_KM, 1.0, 20.0, 1.1, true),
        (KM_iso, 0.6, 0.3, 0.8, true), (KM_iso, 1.0, 1.0, 0.8, false),
    )
    for (KM, η, ω, φ, well_conditioned) in cases, j in (1, 2)
        C = Array(TensND.components(_rd_stiffness(Float64.(KM)), TensND.CanonicalBasis{3, Float64}()))
        sys = _RD_CR._build_poly_system(C, [cos(φ), sin(φ) / η, 0.0], [0.0, 0.0, 1 / ω])
        N = 6 * (j + 1)
        p = _RD_EL._hill_poles(sys.roots_uhp, j + 1)
        clusters = _RD_CR._pole_clusters(p.poles, p.nupper; logfactor = true)
        lead = (_RD_CR.Polynomials.coeffs(sys.Q)[end] / 64)^(j + 1)
        W, err = _RD_EL._hill_line_sums(p, clusters, lead, N)
        Wq = map(0:N) do m
            f(t) = t * ((t^2 - 1) / (2t))^m / ((1 + t^2)^2 * sys.Q((t^2 - 1) / (2t))^(j + 1))
            2 * quadgk(f, 0.0, 1.0, Inf; rtol = 1.0e-14)[1]
        end
        @test length(W) == N + 1
        @test maximum(abs, W - Wq) <= err + 1.0e-13 * maximum(abs, Wq)
        if well_conditioned
            @test W ≈ Wq rtol = 1.0e-10
            @test err <= 1.0e-8 * maximum(abs, Wq)
        else
            @test err > 1.0e-8 * maximum(abs, Wq)
        end
    end
end

@testset "weights of the inner integral with dual coefficients" begin
    # dWₙ/dx by the residue sums, against the derivative of the quadrature of the
    # definition taken under the integral sign, -∫ ξⁿ ∂ₓQ / Q² (…): the same
    # derivative by another evaluation.
    quadgk = _RD_CR.QuadGK.quadgk
    E = zeros(6, 6); E[1, 1] = 1.0; E[2, 4] = E[4, 2] = 0.5
    η, ω, φ = 0.7, 0.4, 0.3
    function weights(x)
        C = Array(TensND.components(_rd_stiffness(_RD_KM + x * E), TensND.CanonicalBasis{3, typeof(x)}()))
        sys = _RD_CR._build_poly_system(C, [cos(φ), sin(φ) / η, 0.0], [0.0, 0.0, 1 / ω])
        return _RD_EL._hill_line_weights(sys.Q, sys.roots_uhp), sys.Q
    end
    W, Q = weights(ForwardDiff.Dual(0.0, 1.0))
    @test @inferred(first(weights(ForwardDiff.Dual(0.0, 1.0)))) isa Vector{<:ForwardDiff.Dual}
    @test ForwardDiff.value.(W) == first(weights(0.0))           # the same sums, bit for bit
    q₀ = ForwardDiff.value.(_RD_CR.Polynomials.coeffs(Q))
    q₁ = first.(ForwardDiff.partials.(_RD_CR.Polynomials.coeffs(Q)))
    dW = map(0:6) do n
        f(t) = (ξ = (t^2 - 1) / (2t); -t * ξ^n * evalpoly(ξ, q₁) / ((1 + t^2)^2 * evalpoly(ξ, q₀)^2))
        2 * quadgk(f, 0.0, 1.0, Inf; rtol = 1.0e-14)[1]
    end
    @test first.(ForwardDiff.partials.(W)) ≈ dW rtol = 1.0e-10
end

@testset "Hill tensor by residues: derivatives on exact identities" begin
    # ℙ(λℂ) = ℙ(ℂ)/λ: dℙ/dλ = -ℙ at λ = 1, and a uniform scaling of the ellipsoid
    # leaves ℙ unchanged, for an ellipsoid along the axes and a rotated one.
    C₀ = _rd_stiffness(_RD_KM)
    for angles in ((0.0, 0.0, 0.0), (0.4, 1.1, -0.7))
        ell(s) = Ellipsoid(s, 0.6s, 0.3s; euler_angles = angles .* one(s))
        P = _rd_hill(ell(1.0), C₀, :residues)
        @test _rd_derivative(λ -> _rd_hill(ell(one(λ)), λ * C₀, :residues), 1.0) ≈ -P rtol = 1.0e-9
        @test norm(_rd_derivative(s -> _rd_hill(ell(s), one(s) * C₀, :residues), 1.0)) <= 1.0e-9 * norm(P)
    end
    # Two nested levels: d²ℙ/dλ² = 2ℙ.
    ell = Ellipsoid(1.0, 0.6, 0.3)
    f(λ) = _rd_hill(ell, λ * C₀, :residues)
    @test _rd_second(f, 1.0) ≈ 2 * f(1.0) rtol = 1.0e-8
end

@testset "Hill tensor by residues: the value under dual numbers is the Float64 one" begin
    C₀ = _rd_stiffness(_RD_KM)
    ell = Ellipsoid(1.0, 0.6, 0.3)
    P_dual = hill_tensor(ell, _rd_stiffness(_RD_KM .* ForwardDiff.Dual{_RD_TAG}(1.0, 1.0)); method = :residues)
    @test ForwardDiff.value.(_rd_canon(P_dual)) == _rd_canon(hill_tensor(ell, C₀; method = :residues))
    @test eltype(hill_tensor(ell, C₀; method = :residues)) === Float64
    @test eltype(P_dual) <: ForwardDiff.Dual
end

@testset "Hill tensor by residues: derivatives against the cubatures and a centered difference" begin
    # The stiffness, the semi-axes and the orientation in turn. `:nestedquadgk` and
    # `:decuhr` differentiate their integrands, algorithms that share nothing with
    # the residue sums; the centered difference is taken on the residue algorithm
    # in Float64, whose roots move with the variable.
    E = zeros(6, 6); E[1, 1] = 1.0; E[1, 4] = E[4, 1] = 0.5; E[5, 6] = E[6, 5] = -0.3
    stiffness(x) = _rd_stiffness(_RD_KM + x * E)
    angles = (0.4, 1.1, -0.7)
    rotated(b, c, θ) = Ellipsoid(one(θ), b, c; euler_angles = (θ, angles[2] * one(θ), angles[3] * one(θ)))
    paths = (
        (x -> (rotated(0.6 * one(x), 0.3 * one(x), angles[1] * one(x)), stiffness(x)), 100.0),
        (x -> (rotated(0.6 + x, 0.3 * one(x), angles[1] * one(x)), stiffness(zero(x))), 1.0),
        (x -> (rotated(0.6 * one(x), 0.3 + x, angles[1] * one(x)), stiffness(zero(x))), 1.0),
        (x -> (rotated(0.6 * one(x), 0.3 * one(x), angles[1] + x), stiffness(zero(x))), 1.0),
    )
    for (path, scale) in paths
        d_res = _rd_derivative(x -> _rd_hill(path(x)..., :residues), 0.0)
        @test _rd_rel(d_res, _rd_derivative(x -> _rd_hill(path(x)..., :nestedquadgk), 0.0)) < 1.0e-9
        @test _rd_rel(d_res, _rd_centered(x -> _rd_hill(path(x)..., :residues), 0.0, scale)) < 1.0e-7
    end
    # The adaptive cubature, slow on dual numbers, on the stiffness path only.
    path = first(first(paths))
    @test _rd_rel(
        _rd_derivative(x -> _rd_hill(path(x)..., :residues), 0.0),
        _rd_derivative(x -> _rd_hill_decuhr(path(x)...), 0.0)
    ) < 1.0e-6
    # Several partials at once: the columns of a Jacobian are the derivatives
    # taken one at a time.
    f(v) = vec(_rd_hill(rotated(v[1], v[2], v[3]), stiffness(v[4]), :residues))
    v₀ = [0.6, 0.3, angles[1], 0.0]
    J = ForwardDiff.jacobian(f, v₀)
    for (k, (path, _)) in enumerate(paths[[2, 3, 4, 1]])
        @test J[:, k] ≈ vec(_rd_derivative(x -> _rd_hill(path(x)..., :residues), 0.0)) rtol = 1.0e-9
    end
    # A second derivative (two nested levels) against the centered difference of
    # the first.
    ell = Ellipsoid(1.0, 0.6, 0.3)
    f1(x) = _rd_hill(ell, stiffness(x), :residues)
    @test _rd_rel(_rd_second(f1, 0.0), _rd_centered(x -> _rd_derivative(f1, x), 0.0, 100.0)) < 1.0e-6
end

@testset "Hill tensor by residues: derivatives at multiple roots" begin
    # Isotropic values in a general tensor: Q has a triple root, whose derivative
    # with respect to a modulus does not exist; that of ℙ does, and the closed
    # form of the isotropic matrix gives it. Around the sphere the triple root is
    # i, where the weights and their derivatives are taken by quadrature of their
    # definition.
    iso(k, μ) = TensISO{3}(3k, 2μ)
    for axes in ((1.0, 1.0, 1.0), (1.0, 0.6, 0.3), (1.0, 1.0, 0.2))
        ell(x) = Ellipsoid((axes .* one(x))...)
        for (modulus, x₀) in ((μ -> iso(30.0 * one(μ), μ), 18.0), (k -> iso(k, 18.0 * one(k)), 30.0))
            f(x) = _rd_hill(ell(x), _rd_general(modulus(x)), :residues)
            d_res = _rd_derivative(f, x₀)
            @test _rd_rel(d_res, _rd_derivative(x -> _rd_canon(hill_tensor(ell(x), modulus(x))), x₀)) < 1.0e-8
            @test _rd_rel(d_res, _rd_centered(f, x₀)) < 1.0e-7
        end
    end
    # A transversely isotropic matrix along the axis of a spheroid: the roots meet
    # at i in the plane of isotropy.
    ti(x) = TensND.ti_from_constants(2.179 + x, 0.579 * one(x), 0.689 * one(x), 10.345 * one(x), one(x), (0.0, 0.0, 1.0))
    for axes in ((1.0, 1.0, 2.0), (1.0, 0.7, 0.4))
        ell(x) = Ellipsoid((axes .* one(x))...)
        d(method) = _rd_derivative(x -> _rd_hill(ell(x), ti(x), method), 0.0)
        @test _rd_rel(d(:residues), d(:nestedquadgk)) < 1.0e-8
        @test _rd_rel(d(:residues), _rd_centered(x -> _rd_hill(ell(x), ti(x), :residues), 0.0)) < 1.0e-7
    end
end

@testset "Hill tensor by residues: values that are not Float64" begin
    C₀ = _rd_stiffness(_RD_KM)
    ell = Ellipsoid(1.0, 0.6, 0.3)
    # Integers are floats in disguise, and a single precision is promoted by the
    # Float64 of the geometry.
    C_int = zeros(Int, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C_int[i, j, k, l] = 100 * (i == j) * (k == l) + 80 * ((i == k) * (j == l) + (i == l) * (j == k)) +
            70 * (i == j == k == l)
    end
    P_int = _rd_canon(hill_tensor(ell, TensND.Tens(C_int); method = :residues))
    @test P_int ≈ _rd_canon(hill_tensor(ell, TensND.Tens(Float64.(C_int)); method = :residues)) rtol = 1.0e-14
    @test _rd_canon(hill_tensor(ell, _rd_stiffness(Float32.(_RD_KM)); method = :residues)) ≈
        _rd_canon(hill_tensor(ell, C₀; method = :residues)) rtol = 1.0e-6
    # No route for a symbol, a complex modulus, or another precision: refused with
    # the way out, in the stiffness, in the semi-axes or in the orientation.
    Symbolics.@variables x
    @test_throws r"method = :nestedquadgk" hill_tensor(ell, _rd_stiffness(Symbolics.Num.(_RD_KM) .* x); method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(ell, _rd_stiffness(big.(_RD_KM)); method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(Ellipsoid(1.0f0, 0.6f0, 0.3f0; euler_angles = (0.4f0, 1.1f0, -0.7f0)), _rd_stiffness(Float32.(_RD_KM)); method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(ell, _rd_stiffness(complex.(_RD_KM)); method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(Ellipsoid(big(1.0), big(0.6), big(0.3)), C₀; method = :residues)
    @test_throws r"method = :nestedquadgk" hill_tensor(Ellipsoid(1.0, 0.6, 0.3; euler_angles = (big(0.4), big(1.1), big(-0.7))), C₀; method = :residues)
end
