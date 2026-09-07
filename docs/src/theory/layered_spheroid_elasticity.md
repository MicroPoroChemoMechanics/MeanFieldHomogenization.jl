# [The elastic confocal spheroid — Papkovich–Neuber in `(ϕ, p, q)`](@id th-spheroid-elasticity)

[The confocal spheroid](@ref th-layered-spheroid) solves conduction. This page
solves the **axisymmetric elastic** problem in the same chart: the
Papkovich–Neuber representation and its gauge, the displacement and stress
operators it produces, the structural fact that decides how the conditions are
matched, and the solver those add up to.

!!! note "What this covers, and what it does not"
    **All three elementary problems**, so the full transversely isotropic strain
    concentration tensor exists and the mean-field schemes consume it — see
    [the solver](@ref th-spheroid-elastic-solver) and
    [what a scheme needs](@ref th-spheroid-elastic-scheme). **Prolate and
    oblate** alike, the latter through the complex substitution
    ``q = i\tau``, ``c = -i\bar c``, exactly as on the conduction side.

    Case I is the one **derived in full** below, because its operators are the
    ones written out here; cases II and III reuse them unchanged and differ
    only in their mode lists.

    **Perfect interfaces only**, and the default axis ``\underline e_3``. The
    interface restriction is not a gap to be filled:
    [it is structural](@ref th-spheroid-imperfect).

## Why this is a derivation and not a transcription

The pieces exist separately, and none of them is the piece needed here.

- A **uniform** spheroid with an interphase or an imperfect interface is
  published: Duan, Yi, Huang & Wang [duanRSPA2005](@cite) give the
  Papkovich–Neuber representation and the three elementary problems.
- The **confocal multilayer** transfer-matrix formalism is published, but for
  the scalar Laplace equation: Barthélémy & Bignonnet
  [barthelemyBignonnetIJES2020](@cite), itself imported *from* elasticity
  after Hervé & Zaoui [herve1993](@cite) and Hervé-Luanco
  [herveLuanco2014](@cite).
- The **operators** — ``\underline u`` and ``\boldsymbol\sigma`` expressed on
  the potentials, in the spheroidal frame — are in neither. Duan's (2.2) states
  the representation and refers to Love (1927) for the rest.

So they are derived below, from the chart, and every claim is checked by a
`@example` block rather than asserted. The heavier checks live in
`test/LayeredSpheroids/test_pn_symbolic.jl`, which runs in CI.

## The representation, and one set of operators for all three problems

With ``\mu`` the shear modulus and ``\nu`` Poisson's ratio, Papkovich–Neuber
writes the displacement on four harmonic potentials. Collecting the last three
into ``\underline\varphi = (\varphi_1,\varphi_2,\varphi_3)`` and writing
``\Phi = \varphi_0 + \underline x\cdot\underline\varphi``:

```math
2\mu\,\underline u = \nabla\Phi - 4(1-\nu)\,\underline\varphi .
```

**The stress follows in closed form, and this is what makes the three elementary
problems share one implementation.** Each ``\varphi_i`` being harmonic,
``\nabla^2\Phi = 2\operatorname{div}\underline\varphi``, so the trace collapses
and

```math
\operatorname{tr}\boldsymbol\varepsilon
   = -\frac{1-2\nu}{\mu}\operatorname{div}\underline\varphi,
\qquad
\boxed{\;\boldsymbol\sigma = -2\nu\,(\operatorname{div}\underline\varphi)\,\mathbf 1
 + \nabla\nabla\Phi - 4(1-\nu)\,\operatorname{sym}(\nabla\underline\varphi).\;}
```

Nothing in either statement refers to a particular case. Checked for four
*arbitrary* harmonic potentials — in the Cartesian chart, where no metric factor
can hide anything:

```@example spheroid_elastic
using TensND, SymPy, LinearAlgebra
S = coorsys_spheroidal()
ϕ, p, q = getcoords(S)
c = symbols("c", positive = true)
μ = symbols("mu", positive = true)
ν = symbols("nu", real = true)
λ = 2μ * ν / (1 - 2ν)
nothing # hide
```

```@example spheroid_elastic
Cart = coorsys_cartesian()
X, Y, Z = getcoords(Cart)
f = ntuple(i -> SymFunction("f$(i-1)")(X, Y, Z), 4)
φv = Tens([f[2], f[3], f[4]])
Φc = f[1] + X * f[2] + Y * f[3] + Z * f[4]
uc = (GRAD(Φc, Cart) - 4 * (1 - ν) * φv) / (2μ)
εc = SYMGRAD(uc, Cart)
PTc = Dict(X => Sym(2)//7, Y => Sym(3)//5, Z => Sym(11)//9, μ => Sym(7)//3, ν => Sym(1)//4)
atomc(e) = foldl((a, g) -> subs(a, g[1] => g[2]),
    [(diff(f[i], v...), symbols("w$(i)_$(join(string.(v)))"))
     for i in 1:4 for v in ((X, 2), (Y, 2), (X, 1, Y, 1), (X, 1, Z, 1), (Y, 1, Z, 1),
                            (X,), (Y,), (Z,))] ∪ [(f[i], symbols("w$(i)_0")) for i in 1:4];
    init = e)
harmc(e) = foldl((a, g) -> subs(a, diff(g, Z, 2) => -diff(g, X, 2) - diff(g, Y, 2)), f; init = e)
simplify(expand(subs(atomc(harmc(tr(εc) + (1 - 2ν) * DIV(φv, Cart) / μ)), PTc...)))
```

Zero — and `test/LayeredSpheroids/test_pn_symbolic.jl` carries the same check on
all six components of ``\boldsymbol\sigma``.

### The series, in the notation of Barthélémy & Bignonnet

The conduction page expands the temperature as
[barthelemyBignonnetIJES2020](@cite) does, and elasticity keeps that convention:
in layer ``\ell``,

```math
\varphi_i^{(\ell)} = \sum_{n,m} P_n^m(p)\Big\{
   \big[a^{i,m}_{\ell,n}P_n^m(q) + b^{i,m}_{\ell,n}Q_n^m(q)\big]\cos m\varphi
 + \big[c^{i,m}_{\ell,n}P_n^m(q) + d^{i,m}_{\ell,n}Q_n^m(q)\big]\sin m\varphi\Big\}.
```

``P_n^m(p)P_n^m(q)`` has a finite limit as ``q\to1`` and is a **regular**
harmonic; ``P_n^m(p)Q_n^m(q)`` blows up on the focal segment and is an
**irregular** one. A core carries only regular harmonics, the matrix only
irregular ones plus the remote field. So a letter encodes
*(regularity, azimuthal parity)* — ``a`` regular-cos, ``b`` irregular-cos, ``c``
regular-sin, ``d`` irregular-sin — and the superscript ``i`` is the extra index
elasticity needs, conduction having a single field where this has four
potentials.

### The gauge

The representation is redundant by one function, and the gauge is fixed problem
by problem, after [duanRSPA2005](@cite):

| | remote loading | active potentials | order ``m`` |
|:--|:--|:--|:--:|
| **I** | axisymmetric (``\varepsilon_{33}``, ``\varepsilon_{11}=\varepsilon_{22}``) | ``\varphi_0, \varphi_3`` — ``\varphi_1=\varphi_2=0`` | ``0`` |
| **II** | transverse shear (``\varepsilon_{12}``) | ``\varphi_0, \varphi_3`` at ``m=2``; ``(\varphi_1,\varphi_2)`` from one ``\Psi`` at ``m=1`` | ``2, 1`` |
| **III** | longitudinal shear (``\varepsilon_{13}``) | ``\varphi_0, \varphi_3`` at ``m=1``; ``\varphi_1`` at ``m=0``; **plus two rigid-body rotations** | ``1, 0`` |

The rotations in case III are not decoration: Duan attributes to their omission
the error in Riccardi & Montheillet (1999).

**Case I is what follows.** The gauge is the simplest of the three, and the one
whose fields are axisymmetric — which turns out to give two structural checks
for free.

## Case I — the operators

Write ``\Phi = \varphi_0 + z\,\varphi_3`` for the combined potential, and use the
chart's own abbreviations
``\bar p = \sqrt{1-p^2}``, ``\bar q = \sqrt{q^2-1}``, ``w = \sqrt{q^2-p^2}``,
with ``z = c\,p\,q``.

```@example spheroid_elastic
e₃ = Tens([Sym(0), Sym(0), Sym(1)])
φ₀ = SymFunction("phi0")(p, q)
φ₃ = SymFunction("phi3")(p, q)
Φ  = φ₀ + c * p * q * φ₃

u = (GRAD(Φ, S) - 4 * (1 - ν) * φ₃ * e₃) / (2μ)
ε = SYMGRAD(u, S)
σ = λ * tr(ε) * one(ε) + 2μ * ε
nothing # hide
```

### Axisymmetry is structural

Nothing above imposes ``\partial_\varphi = 0`` on the *fields*; it is imposed on
the potentials. The azimuthal displacement and the azimuthal traction come out
zero on their own — which is the check that the gauge is the right one:

```@example spheroid_elastic
𝐞 = normalized_basis(S)
uc = components(u, 𝐞, (:cont,))
sc = components(σ, 𝐞, (:cont, :cont))
simplify.([uc[1], sc[1, 3]])          # u_ϕ , σ_ϕq
```

### The displacement

```math
u_p = \frac{\bar p}{2\mu\,c\,w}\Big[\partial_p\varphi_0
      + c\,p\,q\,\partial_p\varphi_3 + c\,q\,(4\nu-3)\,\varphi_3\Big],
\qquad
u_q = \frac{\bar q}{2\mu\,c\,w}\Big[\partial_q\varphi_0
      + c\,p\,q\,\partial_q\varphi_3 + c\,p\,(4\nu-3)\,\varphi_3\Big].
```

The ``c`` in the denominator is the Lamé coefficient's — ``\chi_p = c\,w/\bar p``,
so ``(\nabla f)_p = (\bar p / c\,w)\,\partial_p f``. Dropping it costs a factor
``c`` that no dimensional argument would catch, since the two terms in the
bracket stay mutually consistent either way.

The check compares at an exact rational point rather than by `simplify`: the
chart writes ``\bar p/w`` where the formula above writes
``\sqrt{1-p^2}/\sqrt{q^2-p^2}``, and SymPy will not recombine radicals without
being told their signs — the same caveat as on
[the conduction page](@ref th-spheroid-chart).

```@example spheroid_elastic
PT = Dict(p => Sym(2)//7, q => Sym(11)//5, c => Sym(3)//2,
          μ => Sym(7)//3, ν => Sym(1)//4, ϕ => Sym(2)//9)
atomize(e) = foldl((a, d) -> subs(a, d[1] => d[2]), [
        (diff(φ, p, np, q, nq), symbols("a$(k)_$(np)$(nq)"))
        for (k, φ) in ((0, φ₀), (3, φ₃)) for (np, nq) in ((1, 1), (1, 0), (0, 1))
    ] ∪ [(φ₀, symbols("a0_00")), (φ₃, symbols("a3_00"))]; init = e)
at_point(e) = simplify(expand(subs(atomize(e), PT...)))

# The compact form of σ below rests on `∇²Φ = 2 ∂_z φ₃`, so checking it means
# imposing harmonicity first. In this chart and at `m = 0`, `Δφ = 0` reads
# `(1-p²)φ_pp - 2p φ_p + (q²-1)φ_qq + 2q φ_q = 0`, which eliminates every
# p-derivative of order ≥ 2; differentiating it once in p and once in q gives
# the third-order eliminations that `div σ` needs.
D(φ, np, nq) = (np == 0 && nq == 0) ? φ : diff(φ, p, np, q, nq)
rules(φ) = [
    D(φ, 3, 0) => (4p * D(φ, 2, 0) + 2D(φ, 1, 0) -
                   (q^2 - 1) * D(φ, 1, 2) - 2q * D(φ, 1, 1)) / (1 - p^2),
    D(φ, 2, 1) => (2p * D(φ, 1, 1) - (q^2 - 1) * D(φ, 0, 3) -
                   4q * D(φ, 0, 2) - 2D(φ, 0, 1)) / (1 - p^2),
    D(φ, 2, 0) => (2p * D(φ, 1, 0) - (q^2 - 1) * D(φ, 0, 2) - 2q * D(φ, 0, 1)) / (1 - p^2),
]
harmonic(e) = foldl((a, r) -> subs(a, r), vcat(rules(φ₀), rules(φ₃),
                                               rules(φ₀), rules(φ₃)); init = e)
atomize3(e) = foldl((a, d) -> subs(a, d[1] => d[2]), [
        (D(φ, np, nq), symbols("b$(k)_$(np)$(nq)"))
        for (k, φ) in ((0, φ₀), (3, φ₃))
        for (np, nq) in ((1, 2), (0, 3), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0))
    ]; init = e)
reduce_h(e) = expand(subs(atomize3(harmonic(e)), PT...))

w = sqrt(q^2 - p^2)
u_p_ref = sqrt(1 - p^2) / (2μ * c * w) *
    (diff(φ₀, p) + c * p * q * diff(φ₃, p) + c * q * (4ν - 3) * φ₃)
u_q_ref = sqrt(q^2 - 1) / (2μ * c * w) *
    (diff(φ₀, q) + c * p * q * diff(φ₃, q) + c * p * (4ν - 3) * φ₃)
at_point.([uc[2] - u_p_ref, uc[3] - u_q_ref])
```

### The stress, without coordinates

Written out in the chart, ``\sigma_{qq}`` runs to ten lines. The structure
behind it is one. Because ``\varphi_3`` is harmonic,
``\nabla^2\Phi = 2\,\partial_z\varphi_3``, so the trace collapses:

```math
\operatorname{tr}\boldsymbol\varepsilon = -\frac{1-2\nu}{\mu}\,\partial_z\varphi_3,
\qquad
\boxed{\;\boldsymbol\sigma = -2\nu\,(\partial_z\varphi_3)\,\mathbf 1
 + \nabla\nabla\Phi - 4(1-\nu)\,\operatorname{sym}\!\left(\underline e_3\otimes\nabla\varphi_3\right).\;}
```

Every spheroidal component is a projection of that single equation:

```math
\sigma_{qq} = -2\nu\,\partial_z\varphi_3 + (\nabla\nabla\Phi)_{qq}
              - 4(1-\nu)\,(\underline e_3\!\cdot\!\underline e_q)(\nabla\varphi_3)_q,
```
```math
\sigma_{pq} = (\nabla\nabla\Phi)_{pq}
   - 2(1-\nu)\Big[(\underline e_3\!\cdot\!\underline e_p)(\nabla\varphi_3)_q
                + (\underline e_3\!\cdot\!\underline e_q)(\nabla\varphi_3)_p\Big],
```

with the three geometric factors

```math
\underline e_3\!\cdot\!\underline e_p = \frac{q\,\bar p}{w},\qquad
\underline e_3\!\cdot\!\underline e_q = \frac{p\,\bar q}{w},\qquad
\partial_z\varphi_3 = \frac{q(1-p^2)\,\partial_p\varphi_3 + p(q^2-1)\,\partial_q\varphi_3}
                           {c\,(q^2-p^2)}.
```

Both boxed statements, checked:

```@example spheroid_elastic
∇φ₃  = GRAD(φ₃, S)
∂zφ₃ = ∇φ₃ ⋅ e₃
σ_cf = -2ν * ∂zφ₃ * one(ε) + HESS(Φ, S) - 2 * (1 - ν) * (e₃ ⊗ ∇φ₃ + ∇φ₃ ⊗ e₃)
cd, cc = components_canon(σ), components_canon(σ_cf)
[reduce_h(tr(ε) + (1 - 2ν) * ∂zφ₃ / μ);
 [reduce_h(cd[i, j] - cc[i, j]) for i in 1:3 for j in i:3]]
```

## [Why the tangential condition carries a factor ``1-p^2``](@id th-spheroid-banding)

This is the structural point, and it is what separates the elastic problem from
the conduction one.

A confocal interface is a surface ``q = \text{const}``, so a matching condition
must hold for **every** ``p``, and each side carries a Legendre series in ``p``.
Whether the transfer matrix comes out *banded* or *triangular* is decided by
which multiplier acts on ``P_n(p)``:

```math
(1-p^2)\,P_n' = \frac{n(n+1)}{2n+1}\big(P_{n-1} - P_{n+1}\big),
\qquad
p\,P_n = \frac{(n+1)P_{n+1} + n\,P_{n-1}}{2n+1},
```

each reaching one degree either way, while a **bare** derivative reaches all the
way down,

```math
P_n' = (2n-1)P_{n-1} + (2n-5)P_{n-3} + \cdots
```

Now look at the two displacement conditions. A confocal interface leaves the
*geometry* unchanged across it — ``\bar p``, ``\bar q``, ``w`` and ``c`` are
shared by both sides — so every purely geometric prefactor cancels from a
matching condition, and what must be continuous is

```math
\frac{U_q}{\mu} = \frac{1}{\mu}\Big[\partial_q\varphi_0 + c\,p\,q\,\partial_q\varphi_3
                  + c\,p\,(4\nu-3)\,\varphi_3\Big],
\qquad
\frac{U_p}{\mu} = \frac{1}{\mu}\Big[\partial_p\Phi - 4c\,q\,(1-\nu)\,\varphi_3\Big].
```

``U_q`` carries **no** ``p``-derivative at all: only ``p\,P_n`` appears, so it is
banded to ``\pm 1`` as it stands. ``U_p`` carries ``\partial_p\Phi`` bare, and is
therefore triangular — every lower degree at once.

The cure is to match ``(1-p^2)\,U_p`` instead. That is legitimate twice over:
``1-p^2`` is shared geometry, and ``f \equiv 0`` on ``(-1,1)`` exactly when
``(1-p^2)f \equiv 0`` there. It is also the *natural* weight, being the one for
which the derivatives are orthogonal,

```math
\int_{-1}^{1}(1-p^2)\,P_n'(p)\,P_m'(p)\,\mathrm dp = \frac{2n(n+1)}{2n+1}\,\delta_{nm},
```

and it turns every bare ``P_n'`` into the banded combination above, leaving
``(1-p^2)U_p`` banded to ``\pm 2``.

So the pair actually matched is ``\big(u_q,\;(1-p^2)\,u_p\big)``, and the
transfer matrices are **banded — even across a perfect interface**, unlike
conduction, whose perfect-interface blocks are diagonal.

!!! note "The Cartesian components are not the answer either"
    It is tempting to sidestep the weighting by matching ``u_z`` and ``u_\rho``.
    Only half of that works. Since

    ```math
    \partial_z = \frac{q(1-p^2)\,\partial_p + p(q^2-1)\,\partial_q}{c\,(q^2-p^2)},
    \qquad
    \partial_\rho = \frac{\bar p\,\bar q\,\big(q\,\partial_q - p\,\partial_p\big)}{c\,(q^2-p^2)},
    ```

    ``\partial_z`` already packages ``\partial_p`` as ``(1-p^2)\partial_p`` and
    ``u_z`` is banded for free — but ``\partial_\rho`` carries ``p\,\partial_p``,
    and ``p\,P_n' = n\,P_n + P_{n-1}'`` puts the bare derivative straight back.
    The weighting is what matters, not the frame.

The identities, and the counter-examples so that none of this is taken on faith:

```@example spheroid_elastic
P(n) = n < 0 ? Sym(0) : sympy.legendre(n, p)
proj(e, k) = simplify(integrate(expand(e) * P(k) * Sym(2k + 1) // 2, (p, -1, 1)))
reach(f, n) = (d = Int(sympy.degree(expand(f(n)), gen = p));
               [k for k in 0:d if !iszero(proj(f(n), k))])

banded = [(n, reach(m -> (1 - p^2) * diff(P(m), p), n), reach(m -> p * P(m), n),
              reach(m -> p * (1 - p^2) * diff(P(m), p), n)) for n in 3:4]
```

```@example spheroid_elastic
loose = [(n, reach(m -> diff(P(m), p), n), reach(m -> p * diff(P(m), p), n)) for n in 3:5]
```

The first table never strays more than two degrees from ``n``; the second
reaches degree ``0`` from ``n = 5``.

## [The four quantities a perfect interface matches](@id th-spheroid-interface)

Two of the four are displacements and two are tractions. Writing the traction
out needs the Hessian in the chart, and that too is extracted rather than
quoted — the expression is *linear* in the derivatives of ``F``, so each
coefficient is unambiguous:

```math
(\nabla F)_p = \frac{\bar p\,F_{,p}}{c\,w},\qquad
(\nabla F)_q = \frac{\bar q\,F_{,q}}{c\,w},\qquad
(\nabla\nabla F)_{\varphi\varphi} = \frac{q F_{,q} - p F_{,p}}{c^2 w^2},
```
```math
(\nabla\nabla F)_{qq} = \frac{\bar q^2 F_{,qq}}{c^2 w^2}
   + \frac{\bar p^2\,(q F_{,q} - p F_{,p})}{c^2 w^4},
\qquad
(\nabla\nabla F)_{pq} = \frac{\bar p\,\bar q}{c^2}
   \left[\frac{F_{,pq}}{w^2} + \frac{p F_{,q} - q F_{,p}}{w^4}\right].
```

Projecting the boxed stress on ``\underline e_q`` and clearing the shared
factors gives

```math
c^2 w^4\,\sigma_{qq} = T_q,\qquad
c^2 w^4\,\sigma_{pq} = \bar p\,\bar q\;T_p,
```
```math
T_q = \bar q^2 w^2 \Phi_{,qq} + \bar p^2\big(q\Phi_{,q} - p\Phi_{,p}\big)
  - 2\nu c\,w^2\big(q\bar p^2\varphi_{3,p} + p\bar q^2\varphi_{3,q}\big)
  - 4(1-\nu)\,c\,p\,\bar q^2 w^2\,\varphi_{3,q},
```
```math
T_p = w^2 \Phi_{,pq} + p\Phi_{,q} - q\Phi_{,p}
      - 2(1-\nu)\,c\,w^2\big(q\varphi_{3,q} + p\varphi_{3,p}\big).
```

Both hold for harmonic potentials; ``T_q`` uses ``\nabla^2\Phi = 2\partial_z\varphi_3``
and would be false without it.

**The conditions themselves.** A confocal interface shares its geometry, so
``\bar p``, ``\bar q``, ``w`` and ``c`` cancel, and what must be continuous
across ``q = q_\ell`` is

```math
\left[\frac{U_q}{\mu}\right] = 0,\quad
\left[\frac{(1-p^2)\,U_p}{\mu}\right] = 0,\quad
\big[\,T_q\,\big] = 0,\quad
\big[\,(1-p^2)\,T_p\,\big] = 0,
```
```math
U_p = \partial_p\Phi - 4cq(1-\nu)\varphi_3,\qquad
U_q = \partial_q\Phi - 4cp(1-\nu)\varphi_3.
```

The displacements carry ``1/\mu`` and the tractions do not: under
Papkovich–Neuber the stress comes out with no factor ``\mu`` at all, as the
boxed form shows. The ``(1-p^2)`` on the two tangential lines is the weight of
the previous section.

Everything above, checked at once:

```@example spheroid_elastic
pb, qb, w2 = sqrt(1 - p^2), sqrt(q^2 - 1), q^2 - p^2
U_p = diff(Φ, p) - 4c * q * (1 - ν) * φ₃
U_q = diff(Φ, q) - 4c * p * (1 - ν) * φ₃
T_q = qb^2 * w2 * diff(Φ, q, 2) + pb^2 * (q * diff(Φ, q) - p * diff(Φ, p)) -
    2ν * c * w2 * (q * pb^2 * diff(φ₃, p) + p * qb^2 * diff(φ₃, q)) -
    4 * (1 - ν) * c * p * qb^2 * w2 * diff(φ₃, q)
T_p = w2 * diff(Φ, p, 1, q, 1) + p * diff(Φ, q) - q * diff(Φ, p) -
    2 * (1 - ν) * c * w2 * (q * diff(φ₃, q) + p * diff(φ₃, p))

[at_point(uc[2] - pb * U_p / (2μ * c * sqrt(w2))),
 at_point(uc[3] - qb * U_q / (2μ * c * sqrt(w2))),
 reduce_h(c^2 * w2^2 * sc[3, 3] - T_q),
 at_point(c^2 * w2^2 * sc[2, 3] / (pb * qb) - T_p)]
```

## [A self-test of the chart, not a result](@id th-spheroid-selftest)

That ``\operatorname{div}\boldsymbol\sigma = 0`` is **not** something this page
establishes. Papkovich–Neuber with harmonic potentials satisfies Navier's
equation identically — that is the whole content of the representation, it is
classical, and it is coordinate-free, so re-deriving it in a particular chart
proves nothing about elasticity.

What the check below *does* establish is that the machinery producing the
operators above is sound: TensND's spheroidal chart, the wiring of `GRAD`,
`SYMGRAD` and `DIV`, the transcription of the gauge ``\varphi_1=\varphi_2=0``
with its ``\underline e_3`` term, and the ``\lambda \leftrightarrow \nu``
conversion. Every expression on this page comes out of that same machinery, so
a non-zero residual here would have invalidated all of them. It is a plumbing
test, and it is worth running for exactly that reason.

In this chart and at ``m = 0``, harmonicity reads

```math
(1-p^2)\,\varphi_{,pp} - 2p\,\varphi_{,p} + (q^2-1)\,\varphi_{,qq} + 2q\,\varphi_{,q} = 0,
```

which eliminates every ``p``-derivative of order ``\ge 2``; differentiating it
once in ``p`` and once in ``q`` supplies the third-order eliminations that
``\operatorname{div}\boldsymbol\sigma`` needs. Imposing only that:

```@example spheroid_elastic
dz = components_canon(DIV(σ, S))[3]           # the axial component
reduce_h(dz)
```

Zero, as the theorem requires — and only the axial component is shown, to keep
the build short. `test/LayeredSpheroids/test_pn_symbolic.jl` checks the other
two, together with the chart itself and equilibrium for explicit harmonics.

## [One evaluator, and the cases as data](@id th-spheroid-one-evaluator)

The boxed stress refers to no particular case, so neither does the code. A
single routine turns one harmonic mode into ``\underline u`` and the traction
``\boldsymbol\sigma\cdot\underline e_q``; what distinguishes the three problems
is which modes are in the list, at which order ``m``, and what the remote field
is. Cases II and III therefore need no new derivation.

Two things make that practical.

**A mode is a product, so its derivatives are products.** A term
``P_n^m(p)\,R_n^m(q)\,T(m\varphi)`` has nothing in it to differentiate: the
``p`` and ``q`` factors come from the Legendre tables, their second derivatives
from the associated Legendre equation, and the azimuthal factor is a sine or a
cosine. What *does* need differentiating is ``\Phi``, because
``\underline x\cdot\underline\varphi`` multiplies a mode by a coordinate. That
one product is carried by a **second-order jet** — value, gradient and Hessian
traveling together, with `*` implementing Leibniz — rather than by
`ForwardDiff`, which would have to nest inside a solve that is itself
differentiated when one asks for a sensitivity.

**The check is a cross-check, not a self-test.** Case I has closed-form
operators of its own, and those were validated against Eshelby. The generic path
reproduces them to ``7\cdot10^{-15}`` over both potentials, both regularities,
degrees ``0`` to ``7`` and three points, and returns ``u_\varphi`` and
``\sigma_{\varphi q}`` as exact zeros. Two independent routes agreeing is worth
considerably more than either one agreeing with itself.

### [No rigid-body rotations are needed](@id th-spheroid-no-rotation)

Duan adds two to case III and attributes to their omission the error in
Riccardi & Montheillet (1999). They are not needed here, and the reason is that
the **full** four-potential set already spans them. A rotation about
``\underline e_2`` is

```math
\varphi_1 = A\,z,\qquad \varphi_3 = -A\,x
\quad\Longrightarrow\quad
\underline u \propto (z,\,0,\,-x),
```

the **antisymmetric** combination of the very two potentials whose *symmetric*
combination, ``\varphi_1 = \alpha z`` with ``\varphi_3 = +\alpha x``, is the
remote shear itself. Duan's extra unknowns are an artifact of a gauge that drops
one of the two, not a physical requirement. A rotation carries no strain, so its
traction must vanish identically — which is how the test states it — and the
Eshelby oracle confirms the whole thing: case III lands on the analytic answer
with no rotation in the unknown list.

### [Two things that are not optional](@id th-spheroid-not-optional)

Both were found by the oracle disagreeing, and both are easy to get wrong.

**Degree ``0`` of ``\varphi_0``.** Only its *regular* part is a constant and
therefore inert; the irregular one is ``P_0(p)Q_0(q) = \operatorname{arccoth} q``,
an essential mode. Dropping it does not merely lose accuracy — it makes the
case-I system **genuinely inconsistent**, least-squares residual ``10^{-1}``
instead of ``10^{-16}``, and the answer out by 10%. Its regular counterpart is
an exactly zero column, which the solver filters rather than tolerates:
``Float64`` QR survived it, a `ForwardDiff.Dual` element type did not.

**Parity.** Each case's remote field fixes the admissible parity of ``\Phi``
under ``p \to -p``, and the potentials inherit one parity each — ``\varphi_0``
even and ``\varphi_3`` odd in case I, and so on, a mode picking up
``(-1)^{n+m}``. Admitting both parities does not just waste columns: it splices
in the *other* problem of the same order, the one whose remote field is odd
where this one's is even, and the truncated system then leaks between them.

## [The solver, and what it is checked against](@id th-spheroid-elastic-solver)

`spheroid_strain_concentration` runs six solves and assembles the tensor;
`spheroid_elastic_coefficients` is the case-I-only entry, kept because its
independent closed forms are what the generic path is checked against.

Each solve assembles the conditions at **every** interface at once and solves
globally. There is no per-interface transfer matrix to chain: the degrees
couple, so the natural object is one system rather than a product of
``2\mathcal N\times2\mathcal N`` blocks.

The projections onto the Legendre degrees are computed by Gauss–Legendre
quadrature in ``p``. That is **exact**, not approximate — each condition is a
polynomial in ``p`` once the shared radicals are cleared, so a rule with enough
nodes integrates it to the last bit. Banding is what bounds the truncation
error, not how the matrix is built.

The azimuth needs no quadrature at all: every field component of a given case
carries **one** azimuthal harmonic, the same on both sides of an interface, so
it divides out of a matching condition and a single generic ``\varphi`` suffices.

The system is over-determined, and the extra rows are **redundant rather than
conflicting**: for a single inclusion the least-squares residual comes out at
``10^{-16}``. That residual is returned as a diagnostic — if it stops being
small for a single inclusion, something upstream is wrong, which is exactly how
the two defects in
[the previous section](@ref th-spheroid-not-optional) announced themselves.

### The oracle

A single homogeneous spheroid must be Eshelby — on **all 81 components** of the
concentration tensor, not on one scalar — and its interior series must collapse
to the handful of coefficients a uniform strain can carry. Both hold, prolate
and oblate:

| ``\omega`` | | max discrepancy vs Eshelby | residual |
|:--|:--|:--|:--|
| 1.5, 2, 4, 10 | prolate | ``8\cdot10^{-15}`` | ``\le 2\cdot10^{-14}`` |
| 0.7, 0.4, 0.2 | oblate | ``6\cdot10^{-15}`` | ``\le 2\cdot10^{-15}`` |

For an oblate spheroid every intermediate is complex and the answer real; the
imaginary part comes out at ``8\cdot10^{-17}`` relative, and the code refuses
rather than truncates if it does not.

Then, on the assembly rather than on one solve: a shell given the core's own
moduli changes the answer by ``10^{-15}``, which certifies the chaining; the two
members of each shear pair agree, which nothing enforces; and through
`homogenize`, a single-layer spheroid degenerates onto the equivalent
`Ellipsoid` to ``2\cdot10^{-15}`` under both Dilute and Mori–Tanaka.

### Where `Float64` gives out

A layered spheroid converges geometrically, the residual falling by about a
factor ``3`` per unit of ``\mathcal N``. In `Float64` that stops paying at
``\mathcal N \approx 12``:

| ``\mathcal N`` | residual, `Float64` | residual, 256-bit | ``\lvert\Delta\rvert`` between them |
|:--:|:--|:--|:--|
| 8 | ``2.1\!\cdot\!10^{-5}`` | ``2.1\!\cdot\!10^{-5}`` | ``1\!\cdot\!10^{-16}`` |
| 12 | ``2.3\!\cdot\!10^{-7}`` | ``2.3\!\cdot\!10^{-7}`` | ``6\!\cdot\!10^{-16}`` |
| 16 | ``5.9\!\cdot\!10^{-8}`` *(stalls)* | ``2.8\!\cdot\!10^{-9}`` | ``3\!\cdot\!10^{-9}`` |

Raise the element type, not the truncation.

**This is not an empirical accident: it is the published criterion.**
[barthelemyBignonnetIJES2020](@cite) appendix C asks for
``\max\!\left(0.8\,(2\mathcal N - 1),\, 16\right)`` significant digits, so
double precision stops sufficing once ``0.8(2\mathcal N-1) > 16``, that is at
``\mathcal N = 11``. The measurement above puts the departure between
``\mathcal N = 12`` and ``14`` — the rule is a step or two conservative, as a
criterion should be, and it is the same mechanism: what runs out is the accuracy
of the coupling between degrees, not anything about elasticity.

## [What a homogenization scheme gets](@id th-spheroid-elastic-scheme)

A scheme consumes an inclusion's **volume-averaged strain concentration
tensor** ``\mathbb A``, defined by
``\langle\boldsymbol\varepsilon\rangle = \mathbb A : \boldsymbol E`` over the
whole composite inclusion. For a spheroid ``\mathbb A`` is transversely
isotropic about the axis, and the six-dimensional space of symmetric
second-order tensors splits into three subspaces it does not mix:

| subspace | dim | loading | fixes |
|:--|:--:|:--|:--|
| axisymmetric — ``\underline e_3\otimes\underline e_3``, ``\mathbf 1 - \underline e_3\otimes\underline e_3`` | 2 | **case I** | a ``2\times2`` block |
| transverse shear — ``\varepsilon_{11}-\varepsilon_{22}``, ``2\varepsilon_{12}`` | 2 | **case II** | one scalar |
| longitudinal shear — ``2\varepsilon_{13}``, ``2\varepsilon_{23}`` | 2 | **case III** | one scalar |

Six solves — the two members of each subspace, which are the same problem
rotated about the axis — fill all six coefficients. That the two members of a
pair agree is a check the assembly gets for free, and nothing in it enforces
that agreement.

### Averaging without per-layer bookkeeping

The average is taken by the **divergence theorem**, which reduces it to a
surface integral on a confocal boundary:

```math
\langle\boldsymbol\varepsilon\rangle\,V
   = \oint_{q=q_N}\operatorname{sym}(\underline u\otimes\underline e_q)\,\mathrm dS,
\qquad \mathrm dS = c^2\,\bar q\,w\,\mathrm dp\,\mathrm d\varphi .
```

With perfect interfaces ``\underline u`` is continuous, so the *interior*
boundaries cancel in pairs and only the outer surface survives — whatever
``N`` is, and with no volume integral anywhere. A **single layer's** average is
the difference of the two moments on its own boundaries, which is what a
stiffness contribution tensor needs:

```math
\mathbb N_C = \sum_k f_k\,(\mathbb C_k - \mathbb C_0):\mathbb A_k ,
```

irrecoverable from the total alone, the layers having different moduli. The
inner term drops out on its own for the core: the confocal surface degenerates
to the focal segment there and ``\mathrm dS \propto \bar q`` vanishes with it.

The two routes — layers from differences of their own boundary moments, the
total from one moment on the outer boundary — are independent, so
``\sum_k f_k\mathbb A_k = \mathbb A`` is a check rather than a restatement. It
holds to the truncation accuracy.

## [Imperfect interfaces, and why they do not fit](@id th-spheroid-imperfect)

The conduction solver takes a Kapitza resistance or a surface conductance
directly. The elastic one does not, and the reason is structural rather than a
missing feature.

Take a spring law, `[u_q] = s_n\,\sigma_{qq}`. Substituting the two verified
forms `u_q = \bar q\,U_q/(2\mu c w)` and `\sigma_{qq} = T_q/(c^2w^4)`:

```math
\frac{U_q^{+}}{\mu^{+}} - \frac{U_q^{-}}{\mu^{-}}
   \;=\; \frac{2\,s_n\,T_q}{c\,\bar q\,w^{3}} .
```

``w^{3} = (q^2-p^2)^{3/2}`` is an **odd** power of the metric factor. No
rearrangement removes it: clearing it from one side plants it on the other. The
condition therefore stops being a polynomial identity in ``p``, and with it go
both the exactness of the Gauss projection and
[the banding](@ref th-spheroid-banding) that makes truncation legitimate. A
perfect interface escapes this because every geometric factor there is *shared*
and cancels; a compliance is a new length scale that does not.

**The route that does work is a thin interphase** — an extra confocal layer,
which the solver already handles. But the two are not interchangeable, and the
difference is worth seeing. The normal thickness of a confocal shell is
``\chi_q\,\mathrm dq`` with ``\chi_q = c\,w/\bar q``, so

```math
\frac{\text{thickness at the equator}}{\text{thickness at the pole}}
  = \frac{q}{\sqrt{q^2-1}} = \omega ,
```

exactly the aspect ratio. A confocal coating on a 1:5 spheroid is five times
thicker around its waist than at its tips. That is "confocal surfaces are not
homothetic" stated in millimeters, and it means a confocal interphase models a
compliance that **varies along the interface** rather than a uniform spring.

```@example spheroid_elastic
χ = Lame(S)
[simplify(subs(χ[3], p => 1)),                       # pole
 simplify(subs(χ[3], p => 0)),                       # equator
 simplify(subs(χ[3], p => 0) / subs(χ[3], p => 1) - q / sqrt(q^2 - 1))]
```

## [Appendix — the two formulas from BB2020 this page leans on](@id th-spheroid-elastic-appendix)

Recalled because the argument above uses them, not for completeness; the chart
itself is on [the conduction page](@ref th-spheroid-chart).

**Orthogonality (appendix B).** The projections that turn a matching condition
into equations are Legendre projections, and two orthogonality relations do the
work — the plain one for a condition already free of ``p``-derivatives, and the
weighted one that
[the tangential condition](@ref th-spheroid-banding) is built around:

```math
\int_{-1}^{1} P_n(p)\,P_m(p)\,\mathrm dp = \frac{2}{2n+1}\,\delta_{nm},
\qquad
\int_{-1}^{1} (1-p^2)\,P_n'(p)\,P_m'(p)\,\mathrm dp = \frac{2n(n+1)}{2n+1}\,\delta_{nm}.
```

The second is why ``1-p^2`` is the *natural* weight there rather than a
convenient one.

**Precision (appendix C).** Keeping ``\mathcal N`` terms means a highest degree
of ``2\mathcal N - 1``, and the coupling between degrees has to be computed to

```math
\max\!\left(0.8\,(2\mathcal N - 1),\; 16\right)\ \text{significant digits}
```

which is what [the ceiling measured above](@ref th-spheroid-elastic-solver)
runs into. Both relations, checked:

```@example spheroid_elastic
[simplify(integrate(P(3) * P(3), (p, -1, 1)) - Sym(2) // 7),
 simplify(integrate(P(2) * P(4), (p, -1, 1))),
 simplify(integrate((1 - p^2) * diff(P(3), p)^2, (p, -1, 1)) - Sym(2 * 3 * 4) // 7),
 simplify(integrate((1 - p^2) * diff(P(2), p) * diff(P(4), p), (p, -1, 1)))]
```

## What comes next

Not cases II and III — they are in. What is left is smaller and none of it is
obstructed the way an imperfect interface is:

- **Pointwise fields.** The solver returns the harmonic amplitudes, so
  ``\underline u``, ``\boldsymbol\varepsilon`` and ``\boldsymbol\sigma`` at a
  point are a matter of summing modes — the conduction side already offers
  that through `local_temperature` and its siblings.
- **An arbitrary axis.** The elementary problems are written about
  ``\underline e_3``; a spheroid tilted in the global frame needs the result
  rotated, which the solver does not yet do for you.
- **A stiffness rather than a strain concentration.** `stiffness_contribution`
  is assembled layer by layer; the compliance-side twin is not written.

[The roadmap](@ref dev-elastic-spheroid) carries the numerical traps that apply
throughout.