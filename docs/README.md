# Building these docs

From the repository root, the full build is

```
julia --project=docs docs/make.jl
```

It first generates the Literate pages from `scripts/` (`docs/literate.jl`), then
runs three pre-flights that need no `@example` (the bibliography, the `@ref`
anchors, a draft build that checks docstring coverage and references), then the
real build.

## Shorter runs

- **Checks only.** `MFH_DOCS_PREFLIGHT_ONLY=1` stops after the pre-flights, in
  a few minutes: missing docstrings, unresolvable `@ref`, the page tree, the
  bibliography and the heading rules.
- **Some pages only.** `MFH_DOCS_ONLY=theory/hill_tensors,manual/` builds the
  pages whose path contains one of the comma-separated patterns, plus the home
  page and the references (see `docs/partial.jl`). It checks neither links nor
  docstring coverage, and it never deploys.
- **The code blocks of one page.** `julia --project=docs docs/check_blocks.jl
  theory/hill_tensors.md` runs the `@setup`/`@example`/`@repl` blocks of the
  named pages, without Documenter.
- **Notebooks.** `MFH_DOCS_NOTEBOOKS=1` also writes the Jupyter notebooks of the
  Literate pages.

## Pull requests

A pull request does not run the full build. Two jobs of
`.github/workflows/Documentation.yml` replace it: `preflight`, the draft build
of the whole tree (`MFH_DOCS_PREFLIGHT_ONLY=1`), which checks every reference,
and `pages`, a matrix of partial builds that run side by side, one per group of
`docs/shards.jl`. A page belongs to the first group that names it, and to
`rest` if none does, so a new page is built without being assigned. To build
one shard locally:

```
julia -e 'include("docs/pages.jl"); include("docs/shards.jl");
          println(join(shard_pages("cement", doc_page_leaves(pages)), ","))'
MFH_DOCS_ONLY=<that list> julia --project=docs docs/make.jl
```

The full build runs on `main`, on tags and on demand, and is the only one that
deploys.

## Previewing the site

The site lands in `docs/build/1`, and its links have no `.html` suffix
(`cleanUrls`), so a plain static server answers 404 to all of them. Serve it
with VitePress instead, from `docs/`:

```
npm run docs:preview
```

`npm` comes from the `NodeJS_20_jll` artifact that DocumenterVitepress pulls in,
so nothing has to be installed system-wide; if it is not on `PATH`:

```
PATH="$(dirname $(ls ~/.julia/artifacts/*/bin/npm | head -1)):$PATH" npm run docs:preview
```

`npm run docs:dev` does the same with hot reload while editing.
