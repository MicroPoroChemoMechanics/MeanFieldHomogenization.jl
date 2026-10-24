using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
import MeanFieldHomogenization.Elasticity: _hill_2d_iso, _hill_2d_aniso

# =============================================================================
#  test_hill_2d.jl — 2-D Hill tensor: isotropic closed form and anisotropic
#  quadrature.
#
#  Both paths are reached from the SAME `hill_tensor(ell, C₀)` call:
#  `_hill_2d_iso` when `C₀` is a `TensISO{4,2}`, `_hill_2d_aniso` otherwise.
#  They must therefore agree when `C₀` is isotropic but stored in the generic
#  form — that agreement is what this file is mainly here to lock down.
#
#  External reference: Mura (1987) eq. 11.22, Eshelby tensor of the elliptic
#  cylinder, with P = S : C₀⁻¹.  Package convention:
#  C₀ = TensISO{2}(α, β), α = 3k, β = 2μ, plane strain ⇒ ν = (α-β)/(2α).
# =============================================================================

# The same tensor as a generic `Tens`: forces the anisotropic routing.
_as_generic(C) = TensND.Tens(Array(C))

# Mura (1987) eq. 11.22 — elliptic cylinder, semi-axes (a, b), plane strain,
# Poisson's ratio ν.
function _mura_S(a, b, nu)
    s = 1 / (2 * (1 - nu))
    ab = (a + b)^2
    S = zeros(2, 2, 2, 2)
    S[1, 1, 1, 1] = s * ((b^2 + 2a * b) / ab + (1 - 2nu) * b / (a + b))
    S[2, 2, 2, 2] = s * ((a^2 + 2a * b) / ab + (1 - 2nu) * a / (a + b))
    S[1, 1, 2, 2] = s * (b^2 / ab - (1 - 2nu) * b / (a + b))
    S[2, 2, 1, 1] = s * (a^2 / ab - (1 - 2nu) * a / (a + b))
    v = s * ((a^2 + b^2) / (2ab) + (1 - 2nu) / 2)
    for (i, j, p, q) in ((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1))
        S[i, j, p, q] = v
    end
    return S
end

function _mura_P(a, b, k, mu)
    α, β = 3k, 2mu
    nu = (α - β) / (2α)
    S = _mura_S(a, b, nu)
    Cinv = TensISO{2}(1 / α, 1 / β)
    return [
        sum(S[i, j, m, n] * Cinv[m, n, p, q] for m in 1:2, n in 1:2)
            for i in 1:2, j in 1:2, p in 1:2, q in 1:2
    ]
end

_maxdiff(P, Q) = maximum(abs, Array(P) - Array(Q))

@testset "hill 2D — isotropic closed form vs Mura (1987)" begin
    for (k, mu) in ((5.0, 2.0), (1.0, 1.0), (10.0, 0.5), (0.5, 4.0))
        for rho in (1.0, 0.8, 0.5, 0.2)
            ell = Ellipsoid(1.0, rho)
            P = hill_tensor(ell, TensISO{2}(3k, 2mu))
            @test _maxdiff(P, _mura_P(1.0, rho, k, mu)) < 1.0e-12
        end
    end
end

@testset "hill 2D — 𝕌 and 𝕍 rebuild Mura's P, through the circle" begin
    # P = 𝕌/(λ + 2μ) + (𝕍 - 𝕌)/μ, an identity of the isotropic Green operator,
    # checked against Mura's closed form; λ + 2μ = C₁₁₁₁ = (α + β)/2 for
    # TensISO{2}(α, β).
    # 𝕌 jumped where the semi-axes came within 1e-6 of each other, and its
    # diagonal carried a 3/2 factor that belongs to 3-D.
    for (k, mu) in ((5.0, 2.0), (1.0, 1.0)), rho in (1.0, 1 - 1.0e-7, 1 - 1.0e-5, 0.7, 0.2)
        ell = Ellipsoid(1.0, rho)
        U, V = tens_UA(ell), tens_VA(ell)
        @test _maxdiff(U / ((3k + 2mu) / 2) + (V - U) / mu, _mura_P(1.0, rho, k, mu)) < 1.0e-12
    end
    # Circle: (1/2π)∮ ξ⊗ξ⊗ξ⊗ξ dθ, i.e. 3/8 and 1/8.
    U = components_canon(tens_UA(Ellipsoid(1.0, 1.0)))
    @test U[1, 1, 1, 1] ≈ 3 / 8
    @test U[1, 1, 2, 2] ≈ 1 / 8
    @test U[1, 2, 1, 2] ≈ 1 / 8
    @test iszero(U[1, 1, 1, 2])
    # Derivative through a = b: d/db [b(a + 2b) / 2(a + b)²] = 1/4 at a = b = 1.
    @test ForwardDiff.derivative(b -> components_canon(tens_UA(Ellipsoid(1.0, b)))[1, 1, 1, 1], 1.0) ≈ 1 / 4
    @test abs(ForwardDiff.derivative(b -> components_canon(tens_UA(Ellipsoid(1.0, b)))[1, 1, 2, 2], 1.0)) < 1.0e-15
end

@testset "hill 2D — quadrature anisotrope vs Mura (1987)" begin
    for (k, mu) in ((5.0, 2.0), (1.0, 1.0)), rho in (1.0, 0.5, 0.3)
        ell = Ellipsoid(1.0, rho)
        P = _hill_2d_aniso(ell, _as_generic(TensISO{2}(3k, 2mu)))
        @test _maxdiff(P, _mura_P(1.0, rho, k, mu)) < 1.0e-9
    end
end

@testset "hill 2D — both paths agree on the same C₀" begin
    # Regression: the `_hill_2d_iso` closed form and the general
    # `_hill_2d_aniso` quadrature start from the same `hill_tensor` call and
    # must return the same tensor.  They differed by ~1e-2 before the closed
    # form was fixed.
    for (k, mu) in ((5.0, 2.0), (2.0, 3.0), (10.0, 0.5)), rho in (1.0, 0.7, 0.4)
        ell = Ellipsoid(1.0, rho)
        C = TensISO{2}(3k, 2mu)
        @test _maxdiff(hill_tensor(ell, C), _hill_2d_aniso(ell, _as_generic(C))) < 1.0e-8
    end
end

@testset "hill 2D — symmetries and sign" begin
    for rho in (1.0, 0.6)
        P = hill_tensor(Ellipsoid(1.0, rho), TensISO{2}(15.0, 4.0))
        # Major and minor symmetries.
        p = Array(P)
        for perm in ((3, 4, 1, 2), (2, 1, 3, 4), (1, 2, 4, 3))
            @test all(isapprox.(p, permutedims(p, perm); atol = 1.0e-12))
        end
        @test P[1, 1, 1, 1] > 0
        @test P[2, 2, 2, 2] > 0
        @test P[1, 2, 1, 2] > 0
    end
end

@testset "hill 2D — the anisotropic quadrature is exactly symmetric" begin
    # The minor and major symmetries were imposed by averaging in place, so an
    # entry already overwritten entered the next average of its orbit: equal
    # only to rounding. Each orbit is now averaged once, and its members are
    # equal exactly — what decides the symmetric storage of the result.
    C = TensND.Tens(TensND.from_mandel([10.0 3.0 1.0; 3.0 8.0 0.5; 1.0 0.5 4.0]))
    P = get_array(_hill_2d_aniso(Ellipsoid(1.0, 0.6; angle = 0.3), C))
    @test P == permutedims(P, (3, 4, 1, 2)) == permutedims(P, (2, 1, 3, 4)) == permutedims(P, (1, 2, 4, 3))
end

@testset "hill 2D — limite incompressible (k = Inf)" begin
    mu = 2.0
    for rho in (1.0, 0.5, 0.25)
        ell = Ellipsoid(1.0, rho)
        P_inf = hill_tensor(ell, TensISO{2}(Inf, 2mu))
        P_big = hill_tensor(ell, TensISO{2}(3.0e12, 2mu))
        @test all(isfinite, P_inf)
        # The `isinf` branch must be the continuous limit of the general
        # branch.
        @test _maxdiff(P_inf, P_big) < 1.0e-9
        # Incompressibility: the spherical part of P vanishes.
        @test abs(P_inf[1, 1, 1, 1] + P_inf[1, 1, 2, 2]) < 1.0e-12
    end
end

@testset "hill 2D — cercle : P est isotrope 2D" begin
    k, mu = 5.0, 2.0
    P = hill_tensor(Ellipsoid(1.0, 1.0), TensISO{2}(3k, 2mu))
    @test P isa TensND.TensISO{4, 2}
    # Closed-form eigenvalues: P_J = 1/(3k+2μ), P_K = (3k+4μ)/(4μ(3k+2μ)).
    PJ, PK = TensND.get_data(P)
    @test PJ ≈ 1 / (3k + 2mu)
    @test PK ≈ (3k + 4mu) / (4mu * (3k + 2mu))
end
