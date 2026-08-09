# Contributing to MeanFieldHomogenization.jl

Contributions are welcome, from a typo in the documentation to a new inclusion
type. This file says where to go, what is expected of a change, and how to get
help.

## Getting help and reporting problems

Use the [issue
tracker](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/issues) for
everything — bug reports, questions about how to model something, and feature
requests alike. There is no separate mailing list or forum.

A bug report is most useful when it contains:

- the version of `MeanFieldHomogenization.jl` and of Julia (`versioninfo()`), plus the
  output of `] status` if an extension is involved;
- a **runnable** snippet, as short as you can make it, that produces the wrong
  answer or the error;
- what you expected instead, and where that expectation comes from — a
  published formula, a limiting case, another code. Micromechanics has many
  conventions, and about half of the reports that look like bugs turn out to be
  a normalization or a frame difference. Saying which convention you are using
  resolves those immediately.

If you are unsure whether the behavior is wrong or merely surprising, open the
issue anyway and say so.

## Before you write code

For anything beyond a small fix, **open an issue first**. The package is built
around a small number of contracts — the three inclusion entry gates, the
material-symmetry dispatch, the shared elasticity/transport algebra — and a
change that fights those contracts is much cheaper to redirect at the design
stage than at review.

The developer documentation is the place to start. It is written for exactly
this purpose:

| You want to add | Read |
| --- | --- |
| a new inclusion morphology | [Adding a new inclusion](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/dev/developer/adding_inclusion/) |
| a new algorithm for `ℙ` | [Adding a new algorithm](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/dev/developer/adding_algorithm/) |
| a new homogenization scheme | [Adding a scheme](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/dev/developer/adding_scheme/) |
| anything else | [Architecture](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/dev/developer/architecture/) |

A user-defined morphology often does **not** need a change to the package at
all: the `CustomInclusion` contract lets external code join every scheme by
answering one of three entry gates. See *Custom inclusions* in the manual
before proposing a new built-in type.

## Development workflow

```shell
git clone https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl
cd MeanFieldHomogenization.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the documentation locally with:

```shell
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## What a pull request should carry

- **Tests.** Every behavioral change comes with a test. Tests mirror the
  source tree, one directory per sub-module; read [Testing
  conventions](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/dev/developer/testing_conventions/)
  first — it explains, among other things, why a test built only from symmetric
  configurations can be blind to a whole class of errors.
- **A source for the formula.** The documentation rule is that no formula
  appears without a traceable source: a published citation added to
  `docs/src/references.bib`, or an explicit derivation on the page. Please do
  not add a reference you have not checked resolves; DOIs in that file are
  verified against Crossref.
- **Formatting.** The code is formatted with
  [Runic](https://github.com/fredrikekre/Runic.jl) and CI checks it:

  ```shell
  julia -e 'using Pkg; Pkg.activate("runic", shared=true); Pkg.add("Runic")'
  julia --project=@runic -e 'using Runic; Runic.main(["--inplace", "src/", "test/"])'
  ```

- **US English, systematically.** All prose — comments, docstrings, error
  messages, testset names, commit messages, docs — is US English (`localization`,
  `parameterized`, `center`, `color`, `behavior`). A pre-commit hook catches
  UK spellings before they reach CI:

  ```shell
  git config core.hooksPath .githooks      # once per clone
  python3 .github/scripts/spelling_tool.py check .          # whole tree
  python3 .github/scripts/spelling_tool.py convert . --to us --dry-run  # preview fixes
  ```

  The exceptions are listed in `.spelling-ignore` (the historical
  `CHANGELOG.md`, the French audit note `scripts/bench/DIAGNOSTIC.md`, and the
  `echoes2mfh` translation test, which deliberately asserts UK detection).

- **Documentation.** A new public function needs a docstring and an entry in
  the relevant `docs/src/api/` page. A new capability usually deserves a
  paragraph in the manual too.
- **Green CI.** Tests, formatting and the documentation build all run on pull
  requests.

## Type genericity, the one easy trap

Routines are generic in the number type on purpose: the same code path has to
run on `Float64`, on `ForwardDiff` dual numbers, on `Complex` moduli and on
symbolic numbers. Concretely, that means avoiding `Float64` annotations and
`zeros(...)` without an element type in any code that a scheme can reach. A
contribution that quietly breaks differentiability is the failure mode this
package is most exposed to.

## Code of conduct

Be civil and assume good faith. Technical disagreement is welcome; personal
remarks are not. Conduct concerns can be raised privately with the maintainer
at the address in `Project.toml`.

## License

By contributing you agree that your contribution is licensed under the MIT
License, like the rest of the package.
