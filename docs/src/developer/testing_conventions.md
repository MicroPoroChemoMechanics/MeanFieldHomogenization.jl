# [Testing conventions](@id dev-testing-conventions)

## Layout

Tests mirror the source tree, one directory per sub-module, aggregated by
`test/runtests.jl`:

```
test/
  Elliptic/  Core/  Elasticity/  Cracks/  Conductivity/
  LayeredSpheres/  LayeredSpheroids/  Schemes/  Viscoelasticity/
  regression/          # cross-cutting cases spanning the dispatch surface
  runtests.jl
```

`runtests.jl` does three things before any `include`, and all three matter:

- it `import`s `DECUHR`, `Integrals` and `NonlinearSolve` so the package
  **extensions** activate — several tests cross-validate `method = :decuhr`
  against the residue and nested-QuadGK backends, and exercise the SciML
  self-consistent solvers. Without those imports the corresponding methods do
  not exist and the tests silently cover less than they appear to;
- it calls `Random.seed!` once, so a CI failure is reproducible locally instead
  of depending on the draw;
- it wraps everything in a single top-level `@testset "MeanFieldHomogenization"`.

## What a good test looks like here

Every new public function ships with at least a smoke test — but a smoke test is
a floor, not a target. The tests that actually catch bugs in this package are of
three kinds:

1. **Closed-form oracles.** Where an analytical result exists, assert against it
   at tight tolerance; `rtol = 1e-13` is routinely achievable. Example: the
   penny-crack conduction COD ``b = 2/(\pi^{2}K)`` in
   `test/Cracks/test_thermal.jl`.
2. **Cross-backend agreement.** The same quantity computed two independent ways
   must agree — `:residues` vs `:decuhr`; quadrature vs the BigFloat monomial
   series (`test/LayeredSpheroids/test_coupling.jl`); the general series
   solution vs the perfect-interface closed form
   (`test/LayeredSpheroids/test_conductivity.jl`).
3. **Invariants.** Identities that hold whatever the numbers:
   ``\sum_i I_i^{\boldsymbol{A}} = 1``, the Eshelby identities, or the
   contribution invariant ``\boldsymbol{N}_K = \boldsymbol{B}_\Omega -
   \boldsymbol{K}_0\cdot\boldsymbol{A}_\Omega`` for a layered particle.

!!! warning "Test an asymmetric case, not only the symmetric one"
    Degenerate cases hide convention errors, because competing conventions tend
    to *agree* there. A penny crack has ``\eta = 1``, and at ``\eta = 1`` the
    normalizations of the crack compliance ``\mathbb{H}`` used by
    `MeanFieldHomogenization`, by Echoes and by the literature all coincide (see
    [Crack opening displacement](../theory/cod_tensors.md), section
    *Conventions*). A suite covering only the penny therefore cannot detect a
    wrong ``\eta``-dependence.

    This is not hypothetical. As of 2026-07 the elastic ``\mathbb{H}`` is
    covered by a smoke test plus penny and ribbon cases only, so the elliptic
    ``\eta \ne 1`` behavior of the ``\boldsymbol{B} \to \mathbb{H}`` factor is
    pinned by no test at all. The same trap applies to spheres among spheroids
    and to isotropy among symmetry classes: **always add one asymmetric case**.

## Number types are part of the contract

The package is generic over the scalar type, and that genericity is advertised
in the dispatch tables of the theory pages — so it is tested explicitly:

| type | role |
| :--- | :--- |
| `Float64` | the default path |
| `ForwardDiff.Dual` | every algorithm advertised as AD-safe must be exercised *through* [`derivative`](@ref), not merely called with a `Dual` |
| `BigFloat` | precision oracle, notably for the spheroid coupling matrices |
| symbolic (`SymPy`, `Symbolics`) | closed-form paths only |

When adding an algorithm, state its AD compatibility in the docstring **and**
back that claim with a test.

## Coverage

Run coverage through `.github/scripts/coverage.jl` rather than the stock
`julia-processcoverage` action.

The stock processor misattributes lines inside generated functions and
multi-line expressions — observed reporting ~59 % where the true figure is
~95 %. Trust the script's numbers over the action's.

## Checking the documentation without a full build

`docs/make.jl` re-executes every page and takes tens of minutes, most of it
spent on pages nobody just edited. For the edit/check loop use the partial
check instead — it runs only the `@setup` / `@example` / `@repl` blocks of the
pages you name, one fresh module per block group, exactly as Documenter
sandboxes them:

```
julia --project=docs docs/check_blocks.jl theory/hill_tensors.md manual/
julia --project=docs docs/check_blocks.jl $(git diff --name-only -- 'docs/src/*.md')
```

It catches what breaks a build in practice: undefined names, a binding that
shadows an import (`strip = ...` over `Base.strip`), state a block silently
inherited from elsewhere, method errors. Exit status is non-zero on any failure.

It does **not** replace the full build, which is still what validates
cross-references, the `pages` tree, citations and the page-size thresholds.

And neither of them can tell you that an *interactive* figure came out blank:
Documenter reports nothing, and the HTML is well-formed either way. That needs a
browser, with software WebGL enabled — the exact invocation is recorded at the
end of `docs/check_blocks.jl`. Without those flags plotly.js reports "WebGL is
not supported by your browser" and every 3-D scene renders as a gray box, which
looks like a broken figure and is not one.
