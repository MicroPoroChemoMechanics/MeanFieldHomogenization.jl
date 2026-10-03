# [Cross-validation against Echoes](@id dev-validation)

!!! info "Before this page"
    [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes), for the
    PyCall setup the cross-checks rely on, and
    [Testing conventions](@ref dev-testing-conventions), for the oracles of the
    test suite that these cross-checks complement.

`MeanFieldHomogenization` is a port of the C++ [`Echoes`](@cite echoes) code, so nearly
every quantity has an independent reference. The cross-checks live in
`scripts/bench_echoes/` and call Echoes through `PyCall`
(setup: [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes)); the ALV
ones also ship `*_python.json` dumps so they run without an Echoes install.

## Cross-checks

| Script | Validated | Tolerance |
| :--- | :--- | :--- |
| `benchmark.jl` | ``\mathbb P``, ``\Delta\mathbb S`` (residues, DECUHR), ``\partial\mathbb P/\partial\mathbb C`` | machine (aniso), ~1e-6 (iso) |
| `benchmark_nlayers.jl` | n-layer averages ``\alpha_k, \beta_k``, local fields | 1e-6 |
| `benchmark_porous.jl` | porous moduli, 10 schemes × 2 shapes × 7 porosities | 1e-6 |
| `benchmark_strength.jl` | mortar ``k, \mu, E`` + strength ``f_c`` [pichler2011](@cite) | 1 % / 2 % |
| `benchmark_hill_derivative.jl` | ``\partial\mathbb P/\partial\mathbb C`` (ISO, ORTHO) | 1e-5 / 1e-4 |

## Agreement measured

| Quantity | Agreement |
| :--- | :--- |
| Hill ``\mathbb P``, anisotropic (same residue algorithm both sides) | machine precision |
| Hill ``\mathbb P``, isotropic (analytic vs quadrature) | 1e-6 … 1e-8 |
| ``\partial\mathbb P/\partial\mathbb C`` | ISO 1e-15, ORTHO 1e-6 |
| Crack ``\mathbb H``, penny | machine precision |
| Porous moduli — Voigt, Reuss, Dilute, DiluteDual, Maxwell, PCW, MT | 2.5e-14 |
| Porous moduli — SC, ``\varphi \le 0.3`` | 5.7e-7 |
| n-layer sphere ``\alpha_k`` / ``\beta_k`` (30 random 2–8-layer stacks) | 4.6e-13 / 4.4e-14 |
| n-layer sphere, local ``\sigma_{rr}, \sigma_{\theta\theta}`` (80 points) | 5.8e-16 |
| Rotational averages TI / ISO (C++ transcribed verbatim) | 1e-12 |
| ALV per-layer ``\alpha(t,t')``, ``\beta(t,t')`` | 1e-16 diag, 1e-6 off-diag |
| Dual-interface concentrations / effective diffusivity | 1e-8 / 1e-5 |
| Cracked ALV creep, PCW / SC + interface | 1e-4 / 1.4e-4 |
| Three-scale mortar ``k, \mu, E`` | 1e-3 |

## Deliberate divergences

Conventions and formulation choices, not numerical error — do not "fix" them.

| Divergence | Nature | Explained in |
| :--- | :--- | :--- |
| Elliptic crack ``\mathbb H`` | normalization, see below | [COD tensors](@ref th-cod-tensors) |
| MT on a cracked RVE | ``\mathbb B : \mathbb A^{-1}`` vs additive closure; both agree with PCW at low density | [Viscoelasticity](@ref man-viscoelasticity) |
| ASC, porous oblate | Echoes' compliance-form ASC converges to another branch | `benchmark_porous.jl` |
| DifferentialScheme, porous ``\varphi \ge 0.5`` | pre-existing gap, 2.6e-3 → 6.2e-2, reproducible | `benchmark_porous.jl` |
| Strength ``f_c`` (2 %) | water/air regularized to a small positive stiffness | [Strength](@ref app-strength) |

`MeanFieldHomogenization` normalizes the crack compliance by the **minor** semi-axis
``b``, Echoes by the **major** semi-axis ``a``, so for an in-plane aspect
ratio ``\eta = b/a``:

```math
\mathbb H_{\mathrm{MFH}} = \tfrac{3}{4}\,\underline{n} \otimes^s \boldsymbol{B} \otimes^s \underline{n},
\qquad
\mathbb H_{\mathrm{Echoes}} = \eta\,\mathbb H_{\mathrm{MFH}} .
```

Verified at ``\eta = 0.7, 0.5, 0.3, 0.1``: the ratio is ``\eta`` to four
decimals, and the ``3/4`` is ``\eta``-independent to machine precision.

!!! warning "Degenerate cases hide conventions"
    At ``\eta = 1`` both conventions coincide, and a localization term that
    vanishes in every symmetric configuration is invisible to a test suite
    built from those configurations. See
    [Testing conventions](@ref dev-testing-conventions).

## Running

```julia
import DECUHR, Integrals      # the :decuhr back-end lives in an extension
include("scripts/bench_echoes/benchmark.jl")
```

The other scripts of the first table are run by the same `include`.

## Where to go next

[Performance notes](@ref dev-performance-notes) turns from accuracy to cost:
what the hot paths do, and where the time goes.
