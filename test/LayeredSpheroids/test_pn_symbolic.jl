using Test
using TensND
using SymPy
using LinearAlgebra

# =============================================================================
#  test_pn_symbolic.jl — the Papkovich–Neuber displacement and stress operators
#  in confocal prolate spheroidal coordinates, derived and checked here rather
#  than quoted.
#
#  WHAT IS AND IS NOT A RESULT HERE. `div σ = 0` is neither: Papkovich–Neuber
#  with harmonic potentials satisfies Navier identically, classically. That
#  check is a self-test of the chart and of the operator wiring. The results are
#  the closed-form operators, and the banding argument that decides how the
#  transfer matrices are written.
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

"Impose harmonicity, atomize the derivatives, land on the rational point. The
composition is the cheap stand-in for `simplify`, which does not terminate on
these expressions."
_reduce(e) = simplify(_at_point(_atomize(_impose_harmonic(e))))

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

    @testset "the stress has a compact coordinate-free form" begin
        # Writing σ out in the chart gives a ten-line expression that is exact
        # and unusable. The structure behind it is one line. With
        # `Φ = φ₀ + z φ₃` and `φ₃` harmonic, `∇²Φ = 2 ∂_z φ₃`, hence
        #
        #     tr ε = -(1 - 2ν) ∂_z φ₃ / μ,
        #     σ = -2ν (∂_z φ₃) 𝟏 + ∇∇Φ - 4(1 - ν) sym(ê₃ ⊗ ∇φ₃).
        #
        # Every spheroidal component is then a projection of this, which is how
        # the theory page presents it — an equation a reader can hold, and the
        # chart doing the bookkeeping.
        Φ = _φ₀ + _z * _φ₃
        u = (GRAD(Φ, _S) - 4 * (1 - _ν) * _φ₃ * _e₃) / (2 * _μ)
        ε = SYMGRAD(u, _S)
        ∇φ₃ = GRAD(_φ₃, _S)
        ∂zφ₃ = ∇φ₃ ⋅ _e₃

        @test iszero(_reduce(tr(ε) + (1 - 2 * _ν) * ∂zφ₃ / _μ))

        σ_def = _λ * tr(ε) * one(ε) + 2 * _μ * ε
        σ_cf = -2 * _ν * ∂zφ₃ * one(ε) + HESS(Φ, _S) -
            2 * (1 - _ν) * (_e₃ ⊗ ∇φ₃ + ∇φ₃ ⊗ _e₃)
        cd, cc = components_canon(σ_def), components_canon(σ_cf)
        for i in 1:3, j in i:3
            @test iszero(_reduce(cd[i, j] - cc[i, j]))
        end
    end

    @testset "Cartesian derivatives in (p, q)" begin
        # `∂_z` and `∂_ρ` on the chart. These are what the interface conditions
        # are written on — see the next testset for why not `∂_p`, `∂_q`.
        f = SymFunction("f_probe")(_p, _q)
        ∇f = GRAD(f, _S)
        eρ = Tens([cos(_ϕ), sin(_ϕ), Sym(0)])
        w2 = _q^2 - _p^2

        ∂z_ref = (_q * (1 - _p^2) * diff(f, _p) + _p * (_q^2 - 1) * diff(f, _q)) / (_c * w2)
        ∂ρ_ref = sqrt(1 - _p^2) * sqrt(_q^2 - 1) *
            (-_p * diff(f, _p) + _q * diff(f, _q)) / (_c * w2)

        atom_f(e) = foldl(
            (a, d) -> subs(a, d[1] => d[2]), (
                (diff(f, _p, 1, _q, 1), symbols("f11")), (diff(f, _p), symbols("f10")),
                (diff(f, _q), symbols("f01")), (f, symbols("f00")),
            ); init = e
        )
        red_f(e) = simplify(expand(subs(atom_f(e), _PT...)))
        @test iszero(red_f(∇f ⋅ _e₃ - ∂z_ref))
        @test iszero(red_f(∇f ⋅ eρ - ∂ρ_ref))
    end

    @testset "which multipliers keep the coupling banded" begin
        # A confocal interface is a surface `q = const`, so the conditions must
        # hold for every `p`, and each side's potentials are Legendre series in
        # `p`. Whether the transfer matrix comes out BANDED or TRIANGULAR is
        # decided by which multiplier acts on `P_n(p)`:
        #
        #   (1 - p²) P_n′ = n(n+1)/(2n+1) (P_{n-1} - P_{n+1})   — reach 1
        #   p P_n         = [(n+1) P_{n+1} + n P_{n-1}]/(2n+1)  — reach 1
        #   p² P_n, (1-p²) P_n, p(1-p²) P_n′                    — reach 2
        #   P_n′,  p P_n′                                       — reach GROWS with n
        #
        # `U_q` contains no `p`-derivative at all, so it is banded as it stands.
        # `U_p` carries a bare `∂_p Φ` and is triangular; weighting it by
        # `(1 - p²)` — shared geometry across a confocal interface, and the
        # weight for which the `P_n′` are orthogonal — makes it banded too.
        #
        # The Cartesian frame is not a way round this: `∂_z` happens to package
        # `∂_p` as `(1 - p²) ∂_p`, but `∂_ρ` carries `p ∂_p`, and
        # `p P_n′ = n P_n + P_{n-1}′` puts the bare derivative straight back.
        P(n) = sympy.legendre(n, _p)
        for n in 1:5
            @test iszero(
                simplify(
                    expand(
                        (1 - _p^2) * diff(P(n), _p) -
                            Sym(n) * (n + 1) // (2n + 1) * (P(n - 1) - P(n + 1))
                    )
                )
            )
            @test iszero(
                simplify(
                    expand(
                        _p * P(n) - (Sym(n + 1) * P(n + 1) + Sym(n) * P(n - 1)) // (2n + 1)
                    )
                )
            )
        end
        # Reach, measured rather than asserted: how far from `n` a multiplier
        # actually spreads the degrees.
        proj(e, k) = simplify(integrate(expand(e) * P(k) * Sym(2k + 1) // 2, (_p, -1, 1)))
        function reach(mult, n)
            e = expand(mult(n))
            d = Int(sympy.degree(e, gen = _p))
            hit = [k for k in 0:d if !iszero(proj(e, k))]
            return isempty(hit) ? 0 : maximum(abs.(hit .- n))
        end
        for n in 3:5
            @test reach(m -> (1 - _p^2) * diff(P(m), _p), n) == 1
            @test reach(m -> _p * P(m), n) == 1
            @test reach(m -> _p^2 * P(m), n) == 2
            @test reach(m -> (1 - _p^2) * P(m), n) == 2
            @test reach(m -> _p * (1 - _p^2) * diff(P(m), _p), n) == 2
        end
        # The two that are NOT banded: their reach grows with the degree, which
        # is exactly what makes the matrix triangular instead.
        @test reach(m -> diff(P(m), _p), 5) == 5
        @test reach(m -> _p * diff(P(m), _p), 6) == 6
    end

    @testset "the Hessian in the spheroidal frame" begin
        # Extracted from the chart, not quoted. With `w² = q² - p²`,
        # `p̄² = 1 - p²`, `q̄² = q² - 1`:
        #
        #   (∇F)_p     = p̄ F_p / (c w),            (∇F)_q = q̄ F_q / (c w)
        #   (∇∇F)_qq   = q̄² F_qq /(c²w²) + p̄²(q F_q - p F_p)/(c²w⁴)
        #   (∇∇F)_pq   = p̄ q̄ [ F_pq/w² + (p F_q - q F_p)/w⁴ ] / c²
        #   (∇∇F)_φφ   = (q F_q - p F_p)/(c²w²)
        #
        # These are what turn the coordinate-free stress into the traction
        # components the interface conditions match.
        F = SymFunction("F_hess")(_p, _q)
        𝐞 = normalized_basis(_S)
        H = components(HESS(F, _S), 𝐞, (:cont, :cont))
        G = components(GRAD(F, _S), 𝐞, (:cont,))
        pb, qb, w2 = sqrt(1 - _p^2), sqrt(_q^2 - 1), _q^2 - _p^2

        atomF(e) = foldl(
            (a, d) -> subs(a, d[1] => d[2]), [
                (diff(F, _p, 2), symbols("h20")), (diff(F, _p, 1, _q, 1), symbols("h11")),
                (diff(F, _q, 2), symbols("h02")), (diff(F, _p), symbols("h10")),
                (diff(F, _q), symbols("h01")),
            ]; init = e
        )
        redF(e) = simplify(expand(subs(atomF(e), _PT...)))

        @test iszero(redF(G[2] - pb * diff(F, _p) / (_c * sqrt(w2))))
        @test iszero(redF(G[3] - qb * diff(F, _q) / (_c * sqrt(w2))))
        @test iszero(
            redF(
                H[3, 3] - (
                    qb^2 * diff(F, _q, 2) / (_c^2 * w2) +
                        pb^2 * (_q * diff(F, _q) - _p * diff(F, _p)) / (_c^2 * w2^2)
                )
            )
        )
        @test iszero(
            redF(
                H[2, 3] - pb * qb * (
                    diff(F, _p, 1, _q, 1) / w2 +
                        (_p * diff(F, _q) - _q * diff(F, _p)) / w2^2
                ) / _c^2
            )
        )
        @test iszero(
            redF(H[1, 1] - (_q * diff(F, _q) - _p * diff(F, _p)) / (_c^2 * w2))
        )
    end

    @testset "the tractions on a confocal surface" begin
        # The four quantities a perfect interface at `q = qℓ` matches. Every
        # purely geometric factor is SHARED across a confocal interface, so it
        # cancels; what is left is what must be continuous:
        #
        #   u:  U_q/μ  and  (1-p²) U_p/μ      (the `(1-p²)` makes it banded)
        #   t:  T_q    and  (1-p²) T_p        (traction carries no 1/μ: with
        #                                      Papkovich–Neuber, σ has no μ)
        #
        # with, writing Φ = φ₀ + c p q φ₃,
        #
        #   U_p = ∂_p Φ - 4 c q (1-ν) φ₃
        #   U_q = ∂_q Φ - 4 c p (1-ν) φ₃
        #   T_q = q̄²w² Φ_qq + p̄²(q Φ_q - p Φ_p)
        #         - 2ν c w²(q p̄² φ₃_p + p q̄² φ₃_q) - 4(1-ν) c p q̄² w² φ₃_q
        #   T_p = w² Φ_pq + p Φ_q - q Φ_p - 2(1-ν) c w²(q φ₃_q + p φ₃_p)
        #
        # `T_q = c²w⁴ σ_qq` and `T_p = c²w⁴ σ_pq/(p̄q̄)`. `T_q` holds for HARMONIC
        # potentials — the compact form of σ rests on `∇²Φ = 2∂_z φ₃`.
        𝐞 = normalized_basis(_S)
        pb, qb, w2 = sqrt(1 - _p^2), sqrt(_q^2 - 1), _q^2 - _p^2
        Φ = _φ₀ + _c * _p * _q * _φ₃
        σ = _pn_stress(_φ₀, _φ₃)
        sc = components(σ, 𝐞, (:cont, :cont))
        uc = components(_pn_displacement(_φ₀, _φ₃), 𝐞, (:cont,))

        U_p = diff(Φ, _p) - 4 * _c * _q * (1 - _ν) * _φ₃
        U_q = diff(Φ, _q) - 4 * _c * _p * (1 - _ν) * _φ₃
        @test iszero(_reduce(uc[2] - pb * U_p / (2 * _μ * _c * sqrt(w2))))
        @test iszero(_reduce(uc[3] - qb * U_q / (2 * _μ * _c * sqrt(w2))))

        T_q = qb^2 * w2 * diff(Φ, _q, 2) + pb^2 * (_q * diff(Φ, _q) - _p * diff(Φ, _p)) -
            2 * _ν * _c * w2 * (_q * pb^2 * diff(_φ₃, _p) + _p * qb^2 * diff(_φ₃, _q)) -
            4 * (1 - _ν) * _c * _p * qb^2 * w2 * diff(_φ₃, _q)
        T_p = w2 * diff(Φ, _p, 1, _q, 1) + _p * diff(Φ, _q) - _q * diff(Φ, _p) -
            2 * (1 - _ν) * _c * w2 * (_q * diff(_φ₃, _q) + _p * diff(_φ₃, _p))
        @test iszero(_reduce(_c^2 * w2^2 * sc[3, 3] - T_q))
        @test iszero(_reduce(_c^2 * w2^2 * sc[2, 3] / (pb * qb) - T_p))
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
        # NOT a result: Papkovich–Neuber with harmonic potentials satisfies
        # Navier identically, classically and without reference to coordinates.
        # This is a self-test of the toolchain that produces every expression
        # above — the chart, the `GRAD`/`SYMGRAD`/`DIV` wiring, the gauge as
        # transcribed from Duan (2.2), the λ ↔ ν conversion. A non-zero residual
        # here would mean the operators are wrong, which is why it earns its
        # place despite costing the bulk of this file's runtime.
        d = components_canon(DIV(_pn_stress(_φ₀, _φ₃), _S))
        for i in 1:3
            @test iszero(_at_point(_atomize(_impose_harmonic(d[i]))))
        end
    end
end
