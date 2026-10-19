# Where the static figures come from

Maintenance note, not a license notice: every figure below is the author's own
work, reused freely. It lives outside `docs/src/` on purpose — a stray markdown
file under `src/` that is not in `make.jl`'s `pages` makes Documenter complain.

Figures that are **drawn at build time** (`@example` blocks with `Plots`/`GR`, or
the interactive Plotly scenes built by `scripts/common/docviz.jl`) are not listed
here: their source is the page or the script, which is the point of drawing them
that way.

## Drawn figures: `docs/assets_src`

Every schematic of the documentation is produced by `docs/assets_src` (Luxor for
the drawing, Typst for the labels, one palette by role) and committed as SVG; see
`docs/assets_src/README.md` to regenerate them. Several are redrawn after
earlier figures, whose origin is kept below.

| File under `src/assets/` | Generator | Redrawn after |
| :--- | :--- | :--- |
| `geometry/eshelby_inclusion.svg` | `geometry.jl` | Echoes book, `img/eshelbypbincl.png` |
| `geometry/eshelby_inhomogeneity.svg` | `geometry.jl` | Echoes book, `img/eshelbypb.png` |
| `geometry/layered_sphere.svg` | `geometry.jl` | a hand-written SVG of this repository |
| `geometry/layered_ellipsoid.svg` | `geometry.jl` | an earlier talk of the author's; the layers now drawn confocal, with their common foci |
| `schemes/rve_decomposition.svg` | `schemes.jl` | Echoes book, `img/ver_ell_spn_dec.png` |
| `schemes/rve_mori_tanaka.svg` | `schemes.jl` | Echoes book, `img/ver_ell_spn.png` |
| `schemes/rve_self_consistent.svg` | `schemes.jl` | Echoes book, `img/verSC.png` |
| `schemes/rve_maxwell.svg` | `schemes.jl` | an earlier talk of the author's |
| `schemes/rve_pcw.svg` | `schemes.jl` | an earlier talk of the author's; the safety domains now pave the space |
| `schemes/scheme_*.svg` | `schemes.jl` | new: one card per scheme, on the theory page |
| `schemes/particle_assembly.svg` | `vignettes.jl` | new |
| `veritz.svg` | `applications.jl` | Echoes book |
| `ver_multi_mix.svg` | `applications.jl` | Echoes book, after someCBM2022 |
| `lamellar/lamellar_scales.svg`, `lamellar/interfoliar_layer.svg` | `applications.jl` | hand-written SVGs of this repository |
| `fe/axi_schematic.svg` | `applications.jl` | a `Plots` drawing of `scripts/fe/make_axi_figures.jl` |
| `applications/cement_paste_scales.svg` | `applications.jl` | new, after Fig. 2 of Lavergne et al. (2018) |
| `applications/solidifying_sphere.svg` | `applications.jl` | new, after Sanahuja (2013) |
| `rheology/*.svg` | `rheology.jl` | the former `make_rheology_diagrams.jl` of this directory |
| `nn/network_sketch.svg`, `fe/crack_mesh_sketch.svg` | `vignettes.jl` | new |

`eshelby_generalized.png` and `three_phase_model.png` are no longer used. The
first was replaced by `layered_sphere.svg`. The second was dropped rather than
redrawn: the three-phase model is a *scheme* built on the composite-sphere
solution, not a statement of that problem, so a figure of it did not belong on
the theory page for the pattern.

## Plotted figures: the scripts

The plots of finite-element and neural-network results under `src/assets/fe/`,
`src/assets/fe_coupling/` and `src/assets/nn/` are written by the scripts that
compute them (`scripts/fe/make_*_figures.jl`, `scripts/fe/arma2011_welltest.jl`,
`scripts/nn/`). The screenshots under `src/assets/mfhstudio/` are taken from
a running MFH Studio, in the light and the dark theme, by
`src/studio/tools/screenshots.mjs` (repository root), which says how to run it.
