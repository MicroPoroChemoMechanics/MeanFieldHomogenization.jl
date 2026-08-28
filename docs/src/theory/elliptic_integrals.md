# [Elliptic integrals](@id th-elliptic-integrals)

The [`MeanFieldHomogenization.Elliptic`](@ref MeanFieldHomogenization.Elliptic) submodule provides
type-generic Legendre and Carlson integrals that flow end-to-end through
automatic differentiation (`ForwardDiff.Dual`), arbitrary precision
(`BigFloat`) and symbolic scalars (`SymPy.Sym`, `Symbolics.Num`).

## Legendre elliptic integrals

Throughout the submodule, `m` denotes the **parameter** (not the modulus),
with the convention ``m = k^2`` and ``0 \le m < 1``.

### Complete integrals

```math
K(m) = \int_0^{\pi/2}
       \frac{\mathrm{d}\theta}{\sqrt{1 - m \sin^2\theta}},
\qquad
E(m) = \int_0^{\pi/2}
       \sqrt{1 - m \sin^2\theta}\,\mathrm{d}\theta .
```

### Incomplete integrals

```math
F(\varphi, m) = \int_0^{\varphi}
       \frac{\mathrm{d}\theta}{\sqrt{1 - m \sin^2\theta}},
\qquad
E(\varphi, m) = \int_0^{\varphi}
       \sqrt{1 - m \sin^2\theta}\,\mathrm{d}\theta .
```

Both reduce to the complete integrals at ``\varphi = \pi/2``.

## Arithmetic–geometric mean (AGM)

For any `Number` subtype other than `Float64`, the complete integrals
are evaluated using the classical AGM recursion (Abramowitz & Stegun
17.6, NIST DLMF 19.8). Starting from

```math
a_0 = 1, \quad b_0 = \sqrt{1 - m}, \quad c_0 = \sqrt{m},
```

the sequence

```math
a_{n+1} = \frac{a_n + b_n}{2}, \qquad
b_{n+1} = \sqrt{a_n b_n},      \qquad
c_{n+1} = \frac{a_n - b_n}{2}
```

converges quadratically, and

```math
K(m) = \frac{\pi}{2\, \mathrm{agm}(1, \sqrt{1 - m})},
\qquad
E(m) = K(m)\;\bigl(1 - \tfrac{1}{2}\textstyle\sum_{n \ge 0} 2^n c_n^2\bigr).
```

Eight to twelve iterations are typically enough to reach `Float64`
precision; `BigFloat` needs only a few more.

## Carlson symmetric forms

Incomplete integrals are delegated to Carlson's symmetric integrals
(Carlson 1995, *Numerical computation of real or complex elliptic
integrals*):

```math
R_F(x, y, z) = \frac{1}{2}\int_0^{+\infty}
      \frac{\mathrm{d}t}{\sqrt{(t+x)(t+y)(t+z)}},
```

```math
R_D(x, y, z) = \frac{3}{2}\int_0^{+\infty}
      \frac{\mathrm{d}t}{(t+z)\sqrt{(t+x)(t+y)(t+z)}}.
```

With these two primitives one recovers the Legendre integrals via

```math
F(\varphi, m) = \sin\varphi \; R_F(\cos^2\varphi,\;
                                    1 - m\sin^2\varphi,\; 1),
```

```math
E(\varphi, m) = F(\varphi, m)
              - \tfrac{1}{3} m \sin^3\varphi\;
                R_D(\cos^2\varphi,\; 1 - m\sin^2\varphi,\; 1).
```

`R_F` and `R_D` are implemented by the duplication theorem (Carlson
1995, §2): each iteration halves the relative spread of
``x, y, z`` until their ratio approaches unity, at which point a
fifth-order Taylor series in the Carlson invariants ``E_2, E_3`` (and
``E_4, E_5`` for ``R_D``) is used. The recursion is arithmetic-only —
no branch cuts, no transcendentals — so it extends to any `Number`
subtype.

## Dispatch table

| Scalar type          | Backend                                          |
| :------------------- | :----------------------------------------------- |
| `Float64`            | `Elliptic.jl` (GSL C binding, fastest)           |
| `ForwardDiff.Dual`   | pure-Julia AGM / Carlson (derivatives work)      |
| `BigFloat`, generic  | AGM / Carlson                                    |
| `SymPy.Sym`          | `sympy.elliptic_{k,e,f}` via the SymPy weak ext  |
| `Symbolics.Num`      | AGM / Carlson (verbose; use `simplify` if needed)|

!!! note "Why a SymPy weak extension?"
    The AGM unrolls ~60 nested `sqrt(a*b)` operations. On a `SymPy.Sym`
    input this builds a deeply nested symbolic tree that overflows
    SymPy's pretty-printer. The weak extension `MeanFieldHomogenizationSymPyExt`
    (loaded automatically whenever `SymPy` is loaded alongside
    `MeanFieldHomogenization`) routes `ell_K`, `ell_E`, `ell_F` on `Sym` arguments
    directly to `sympy.elliptic_{k,e,f}`, returning the native closed
    form instead.

## Special cases

- **``m = 0``**: ``K(0) = E(0) = \pi/2``;
  ``F(\varphi, 0) = E(\varphi, 0) = \varphi``.
- **``m \to 1``**: ``K(m) \sim -\tfrac{1}{2}\log(1 - m)`` (logarithmic
  divergence); ``E(1) = 1``. The `Float64` fast path through
  `Elliptic.jl` throws at ``m = 1``.
- **``\varphi = \pi/2``**: incomplete integrals coincide with complete
  ones.

## References

- M. Abramowitz and I.A. Stegun, *Handbook of Mathematical Functions*,
  §17.6, Dover 1972.
- NIST Digital Library of Mathematical Functions,
  [§19](https://dlmf.nist.gov/19).
- B.C. Carlson, *Numerical computation of real or complex elliptic
  integrals*, Numerical Algorithms **10** (1995) 13–26.
- B.C. Carlson and E.M. Notis, *Algorithms for incomplete elliptic
  integrals*, ACM Transactions on Mathematical Software **7** (1981)
  398–403 — public-domain SLATEC routines `DRF` / `DRD`.
