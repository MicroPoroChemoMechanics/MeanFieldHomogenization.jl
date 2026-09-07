# [The elastic confocal spheroid — Papkovich–Neuber in `(ϕ, p, q)`](@id th-spheroid-elasticity)

[The confocal spheroid](@ref th-layered-spheroid) is solved here for
**conduction** only. This page builds the elastic groundwork: the
Papkovich–Neuber representation in the same chart, the displacement and stress
operators it produces, and the structural fact that decides what the transfer
matrices look like.

!!! warning "Work in progress"
    What is on this page is derived and checked. The transfer matrices
    themselves are not written yet — see
    [the roadmap](@ref dev-elastic-spheroid) for the sequence.

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

## The representation

With ``\mu`` the shear modulus and ``\nu`` Poisson's ratio, Papkovich–Neuber
writes the displacement on four harmonic potentials
``\varphi_0, \varphi_1, \varphi_2, \varphi_3``:

```math
2\mu\,\underline u \;=\; \nabla\!\left(\varphi_0 + x\varphi_1 + y\varphi_2 + z\varphi_3\right)
\;-\; 4(1-\nu)\,(\varphi_1, \varphi_2, \varphi_3),
```

which is Duan's (2.2). The representation is redundant by one function, and the
gauge is fixed problem by problem:

| | remote loading | active potentials | order ``m`` |
|:--|:--|:--|:--:|
| **I** | axisymmetric (``\varepsilon_{33}``, ``\varepsilon_{11}=\varepsilon_{22}``) | ``\varphi_0, \varphi_3`` — ``\varphi_1=\varphi_2=0`` | ``0`` |
| **II** | transverse shear (``\varepsilon_{12}``) | ``\varphi_0, \varphi_3`` at ``m=2``; ``(\varphi_1,\varphi_2)`` from one ``\Psi`` at ``m=1`` | ``2, 1`` |
| **III** | longitudinal shear (``\varepsilon_{13}``) | ``\varphi_0, \varphi_3`` at ``m=1``; ``\varphi_1`` at ``m=0``; **plus two rigid-body rotations** | ``1, 0`` |

The rotations in case III are not decoration: Duan attributes to their omission
the error in Riccardi & Montheillet (1999).

**This page covers case I.**

## Case I — the operators

Write ``\Phi = \varphi_0 + z\,\varphi_3`` for the combined potential, and use the
chart's own abbreviations
``\bar p = \sqrt{1-p^2}``, ``\bar q = \sqrt{q^2-1}``, ``w = \sqrt{q^2-p^2}``,
with ``z = c\,p\,q``.

```@example spheroid_elastic
using TensND, SymPy, LinearAlgebra
S = coorsys_spheroidal()
ϕ, p, q = getcoords(S)
c = symbols("c", positive = true)
μ = symbols("mu", positive = true)
ν = symbols("nu", real = true)
λ = 2μ * ν / (1 - 2ν)

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

## [The solver, and what it is checked against](@id th-spheroid-elastic-solver)

`spheroid_elastic_coefficients` assembles the four conditions at every
interface and solves once, globally. There is no per-interface transfer matrix
to chain: the degrees couple, so the natural object is one system rather than a
product of ``2\mathcal N \times 2\mathcal N`` blocks.

The projections onto the Legendre degrees are computed by Gauss–Legendre
quadrature in ``p``. That is **exact**, not approximate — each condition is a
polynomial in ``p`` once the shared radicals are cleared, so a rule with enough
nodes integrates it to the last bit. Banding is what bounds the truncation
error, not how the matrix is built.

Each region contributes ``4\mathcal N - 1`` amplitudes and each interface
``4\mathcal N`` conditions, degree ``0`` of ``\varphi_0`` being a constant
potential that moves nothing. The system is therefore over-determined by one row
per interface, and those rows are **redundant rather than conflicting**: for a
single inclusion the least-squares residual comes out at ``10^{-16}``. That
residual is returned, and it is a diagnostic — if it stops being small for a
single inclusion, something upstream is wrong.

### The oracle

A single homogeneous spheroid must be Eshelby, and its interior series must
collapse to the two coefficients a uniform strain can carry — degree ``2`` of
``\varphi_0`` and degree ``1`` of ``\varphi_3``. Both hold:

| ``\omega`` | interior coefficients ``\ne 0`` | ``\varepsilon_a`` vs Eshelby | ``\varepsilon_t`` vs Eshelby |
|:--|:--:|:--|:--|
| 1.5 | 2 | ``8\!\cdot\!10^{-16}`` | ``9\!\cdot\!10^{-16}`` |
| 2 | 2 | ``1\!\cdot\!10^{-15}`` | ``1\!\cdot\!10^{-15}`` |
| 5 | 2 | ``7\!\cdot\!10^{-15}`` | ``8\!\cdot\!10^{-14}`` |
| 20 | 2 | ``5\!\cdot\!10^{-15}`` | ``2\!\cdot\!10^{-14}`` |

A shell given the core's own moduli changes the answer by ``10^{-15}``, which is
what certifies the chaining.

### Where `Float64` gives out

A layered spheroid converges geometrically, the residual falling by about a
factor ``3`` per unit of ``\mathcal N``. In `Float64` that stops paying at
``\mathcal N \approx 12``:

| ``\mathcal N`` | residual, `Float64` | residual, 256-bit | ``\lvert\Delta\rvert`` between them |
|:--:|:--|:--|:--|
| 8 | ``2.1\!\cdot\!10^{-5}`` | ``2.1\!\cdot\!10^{-5}`` | ``1\!\cdot\!10^{-16}`` |
| 12 | ``2.3\!\cdot\!10^{-7}`` | ``2.3\!\cdot\!10^{-7}`` | ``6\!\cdot\!10^{-16}`` |
| 16 | ``5.9\!\cdot\!10^{-8}`` *(stalls)* | ``2.8\!\cdot\!10^{-9}`` | ``3\!\cdot\!10^{-9}`` |

Raise the element type, not the truncation. The conduction solver records the
same limit after [barthelemyBignonnetIJES2020](@cite) appendix C; the elastic
blocks are wider, so it arrives sooner.

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

## What comes next

The remaining sequence is in [the roadmap](@ref dev-elastic-spheroid): order
``m = 2`` in `legendre.jl`, then cases II and III, then the banded transfer
matrices for perfect interfaces — validated against
`hill_tensor(Spheroid(ω), C₀)` and against the collapse of the series to a
handful of coefficients when no interphase is present — and finally imperfect
interfaces.
