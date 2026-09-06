using Test
using TensND
using SymPy
using LinearAlgebra

# =============================================================================
#  test_pn_symbolic.jl — the Papkovich–Neuber displacement and stress operators
#  in confocal prolate spheroidal coordinates, derived and checked here rather
#  than quoted.
#
#  WHY THIS FILE EXISTS. The elastic n-layer confocal spheroid needs `u` and
#  `σ·e_q` expressed on the Papkovich–Neuber potentials, in the spheroidal
#  frame, because those are what the interface conditions match. Duan, Yi,
#  Huang & Wang (Proc. R. Soc. A 461, 2005, doi:10.1098/rspa.2004.1396) give the
#  representation — their (2.2) — and then refer to Love (1927) for the
#  operators themselves; Barthélémy & Bignonnet (IJES 2020) build the confocal
#  multilayer transfer matrices but only for conduction. **The six expressions
#  below are in neither.** They are derived from the chart, and this file is
#  what keeps them honest.
#
#  COST AND MEMORY. `simplify` on a `div σ` component in a chart with nested
#  radicals does not terminate and exhausts memory. Every check here avoids it:
#  either an exact RATIONAL point (no rounding, so a zero is a proof at that
#  point) or elimination of the derivative atoms followed by `expand` (a
#  polynomial identity in free symbols, which is a proof everywhere). Budget is
#  about 80 s.
# =============================================================================

const _S = coorsys_spheroidal()
const _ϕ, _p, _q = getcoords(_S)
const _c = symbols("c", positive = true)
const _μ = symbols("mu_pn", positive = true)
const _ν = symbols("nu_pn", real = true)
const _λ = 2 * _μ * _ν / (1 - 2 * _ν)

const _e₃ = Tens([Sym(0), Sym(0), Sym(1)])   # Cartesian ê₃
const _z = _c * _p * _q                      # the chart's z

const _φ₀ = SymFunction("phi0")(_p, _q)
const _φ₃ = SymFunction("phi3")(_p, _q)

"`∂^{np+nq} φ / ∂p^{np} ∂q^{nq}`, with the zeroth derivative being `φ` itself."
_D(φ, np, nq) = (np == 0 && nq == 0) ? φ : diff(φ, _p, np, _q, nq)

"""
Rewrite rules eliminating every `p`-derivative of order ≥ 2 using `Δφ = 0`.

At order `m = 0` the chart's Laplacian vanishes iff

    (1 - p²) φ_pp - 2p φ_p + (q² - 1) φ_qq + 2q φ_q = 0,

and differentiating that relation once in `p` and once in `q` supplies the
third-order eliminations, which `div σ` needs.
"""
_harmonic_rules(φ) = [
    _D(φ, 3, 0) => (
        4 * _p * _D(φ, 2, 0) + 2 * _D(φ, 1, 0) -
            (_q^2 - 1) * _D(φ, 1, 2) - 2 * _q * _D(φ, 1, 1)
    ) / (1 - _p^2),
    _D(φ, 2, 1) => (
        2 * _p * _D(φ, 1, 1) - (_q^2 - 1) * _D(φ, 0, 3) -
            4 * _q * _D(φ, 0, 2) - 2 * _D(φ, 0, 1)
    ) / (1 - _p^2),
    _D(φ, 2, 0) => (
        2 * _p * _D(φ, 1, 0) - (_q^2 - 1) * _D(φ, 0, 2) - 2 * _q * _D(φ, 0, 1)
    ) / (1 - _p^2),
]

"Impose harmonicity of both potentials. Twice: the third-order rules put `φ_pp`
back, and the second-order rule then removes it again."
function _impose_harmonic(e)
    for φ in (_φ₀, _φ₃), _ in 1:2, r in _harmonic_rules(φ)
        e = subs(e, r)
    end
    return e
end

"Replace each surviving derivative by a free symbol — highest order first, so
`subs` never matches inside a higher derivative. What remains is a polynomial
identity, which `expand` settles without `simplify`."
function _atomize(e)
    for (k, φ) in ((0, _φ₀), (3, _φ₃))
        for (np, nq) in ((1, 2), (0, 3), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0))
            e = subs(e, _D(φ, np, nq) => symbols("a$(k)_$(np)$(nq)"))
        end
    end
    return e
end

"An exact rational point: no rounding, so a zero here is a proof at this point."
const _PT = Dict(
    _p => Sym(2) // 7, _q => Sym(11) // 5, _c => Sym(3) // 2,
    _μ => Sym(7) // 3, _ν => Sym(1) // 4, _ϕ => Sym(2) // 9,
)
_at_point(e) = expand(subs(e, _PT...))

"Papkovich–Neuber, Duan (2.2), in the case-I gauge `φ₁ = φ₂ = 0`."
_pn_displacement(φ₀, φ₃) =
    (GRAD(φ₀ + _z * φ₃, _S) - 4 * (1 - _ν) * φ₃ * _e₃) / (2 * _μ)

function _pn_stress(φ₀, φ₃)
    ε = SYMGRAD(_pn_displacement(φ₀, φ₃), _S)
    return _λ * tr(ε) * one(ε) + 2 * _μ * ε
end

@testset "Papkovich–Neuber in spheroidal coordinates — case I (axisymmetric)" begin

    @testset "the chart is the one the derivation assumes" begin
        OM = components_canon(getOM(_S))
        @test iszero(simplify(OM[3] - _c * _p * _q))
        # `getOM` is written with the chart's auxiliary symbols `p̄`, `q̄`. They
        # are opaque to `subs` — not functions of `p`, `q` — and rebuilding them
        # with `symbols("p̄")` yields a DIFFERENT symbol, since the chart's carry
        # assumptions. Take the objects the expression itself contains.
        bars = Dict(
            string(σ) => σ for σ in free_symbols(OM[1]) if string(σ) in ("p̄", "q̄")
        )
        @test length(bars) == 2
        unbar(e) = subs(
            subs(e, bars["p̄"] => sqrt(1 - _p^2)), bars["q̄"] => sqrt(_q^2 - 1)
        )
        @test iszero(simplify(unbar(OM[1]^2 + OM[2]^2) - _c^2 * (1 - _p^2) * (_q^2 - 1)))
    end

    @testset "the potentials used below are harmonic" begin
        @test iszero(simplify(LAPLACE(_p * _q, _S)))                          # P₁(p)P₁(q)
        @test iszero(simplify(LAPLACE((3 * _p^2 - 1) * (3 * _q^2 - 1) / 4, _S)))  # P₂P₂
    end

    @testset "axisymmetry is structural, not imposed" begin
        # `φ₁ = φ₂ = 0` and `m = 0` must kill the azimuthal displacement and the
        # azimuthal traction outright. If either survives, the gauge is wrong.
        𝐞 = normalized_basis(_S)
        u = _pn_displacement(_φ₀, _φ₃)
        σ = _pn_stress(_φ₀, _φ₃)
        @test iszero(simplify(components(u, 𝐞, (:cont,))[1]))
        @test iszero(simplify(components(σ, 𝐞, (:cont, :cont))[1, 3]))
    end

    @testset "closed form of the displacement in the spheroidal frame" begin
        # The two expressions this whole development rests on, and they are not
        # in the literature:
        #
        #   u_p = p̄ / (2 μ c w) [ ∂_p φ₀ + c p q ∂_p φ₃ + c q (4ν - 3) φ₃ ]
        #   u_q = q̄ / (2 μ c w) [ ∂_q φ₀ + c p q ∂_q φ₃ + c p (4ν - 3) φ₃ ]
        #
        # with p̄ = √(1-p²), q̄ = √(q²-1), w = √(q²-p²). The `c` in the
        # denominator is the Lamé coefficient's: `χ_p = c w / p̄`, so
        # `(∇f)_p = (p̄ / c w) ∂_p f`. Dropping it costs a factor `c`, which no
        # dimensional check would catch — `c`, `∂_p φ₀` and `φ₃` all carry
        # different dimensions, and the two terms stay consistent either way.
        𝐞 = normalized_basis(_S)
        uc = components(_pn_displacement(_φ₀, _φ₃), 𝐞, (:cont,))
        w = sqrt(_q^2 - _p^2)
        u_p_ref = sqrt(1 - _p^2) / (2 * _μ * _c * w) *
            (_D(_φ₀, 1, 0) + _c * _p * _q * _D(_φ₃, 1, 0) + _c * _q * (4 * _ν - 3) * _φ₃)
        u_q_ref = sqrt(_q^2 - 1) / (2 * _μ * _c * w) *
            (_D(_φ₀, 0, 1) + _c * _p * _q * _D(_φ₃, 0, 1) + _c * _p * (4 * _ν - 3) * _φ₃)
        # At the rational point, for the reason above: the difference involves
        # √(1-p²)/√(q²-p²) against √((1-p²)/(q²-p²)), which `simplify` leaves
        # alone. The derivatives are atomized first, so what is compared is a
        # linear form in them with exact surd coefficients.
        @test iszero(simplify(_at_point(_atomize(uc[2] - u_p_ref))))
        @test iszero(simplify(_at_point(_atomize(uc[3] - u_q_ref))))
    end

    @testset "equilibrium at an exact rational point, for explicit potentials" begin
        for (φ₀, φ₃) in (
                (_p * _q, Sym(0)),                                   # potential part alone
                (Sym(0), _p * _q),                                   # the z φ₃ ê₃ term alone
                ((3 * _p^2 - 1) * (3 * _q^2 - 1) / 4, _p * _q),      # both, different degrees
            )
            d = components_canon(DIV(_pn_stress(φ₀, φ₃), _S))
            for i in 1:3
                @test iszero(simplify(_at_point(d[i])))
            end
        end
    end

    @testset "equilibrium holds identically, for ANY harmonic pair" begin
        # The strong statement, and the reason the representation is admissible:
        # `div σ = 0` follows from the harmonicity of `φ₀` and `φ₃` alone. This
        # is Papkovich–Neuber re-derived in the spheroidal chart, not quoted
        # from Love (1927).
        d = components_canon(DIV(_pn_stress(_φ₀, _φ₃), _S))
        for i in 1:3
            @test iszero(_at_point(_atomize(_impose_harmonic(d[i]))))
        end
    end
end
