# Where the static figures come from

Maintenance note, not a license notice: every figure below is the author's own
work, reused freely. It lives outside `docs/src/` on purpose — a stray markdown
file under `src/` that is not in `make.jl`'s `pages` makes Documenter complain.

Figures that are **drawn at build time** (`@example` blocks with `Plots`/`GR`, or
the interactive Plotly scenes built by `scripts/common/docviz.jl`) are not listed
here: their source is the page or the script, which is the point of drawing them
that way.

## `src/assets/geometry/`

| File | Origin | Notes |
| :--- | :--- | :--- |
| `eshelby_inclusion.png` | Echoes book, `img/eshelbypbincl.png` | Eshelby *inclusion* problem: eigenstrain τ, `u = 0` at ∞ |
| `eshelby_inhomogeneity.png` | Echoes book, `img/eshelbypb.png` | Eshelby *inhomogeneity* problem: ℂᴵ in ℂ, remote `u = E·x` |
| `layered_sphere.svg` | hand-authored, this repository | Concentric `n`-layer sphere in an unbounded matrix ℂ₀, remote `u = E·x`; replaces `eshelby_generalized.png`, which was a low-resolution raster |
| `crack_frame.svg` | Echoes book, `img/crack.svg` (rendered from `img/crack.asy`) | Elliptical crack with its frame (ℓ, m, n) and `a`, `b = ηa`, `c = ωa` |
| `layered_ellipsoid.png` | earlier talk of the author's | N-layer confocal ellipsoid, domains Ω₁…Ω_{N+1}, interfaces I₁…I_N |

`eshelby_generalized.png` and `three_phase_model.png` are no longer used. The
first was replaced by `layered_sphere.svg` — vector, legible at any size, and
self-contained: it carries its own neutral panel rather than a
`prefers-color-scheme` variant, because an SVG served as an image resolves
that query against the operating system, not against the site's theme toggle. The second was dropped rather than redrawn: the three-phase model
is a *scheme* built on the composite-sphere solution, not a statement of that
problem, so a figure of it did not belong on the theory page for the pattern.

To re-render `crack_frame.svg` from Asymptote: in `img/crack.asy`, keep the
`settings.outformat = "svg"; settings.render = 0;` lines active, then `asy crack.asy`.

## `src/assets/schemes/`

| File | Origin | Notes |
| :--- | :--- | :--- |
| `rve_decomposition.png` | Echoes book, `img/ver_ell_spn_dec.png` | RVE at `u = E·x` ≡ a sum of single-inclusion problems at `u = E⁰·x` in ℂ_m |
| `rve_mori_tanaka.png` | Echoes book, `img/ver_ell_spn.png` | Matrix + ellipsoids + coated spheres: the Mori-Tanaka morphology |
| `rve_self_consistent.png` | Echoes book, `img/verSC.png` | Polycrystal tessellation — no phase plays the role of a matrix |
| `rve_pcw.png` | earlier talk of the author's | Ponte Castañeda–Willis: each inclusion inside its distribution ellipsoid |
| `rve_maxwell.png` | earlier talk of the author's | Cluster of inclusions ≡ one equivalent inclusion Ω, both at remote ε |
