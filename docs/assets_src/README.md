# `docs/assets_src` — the generators of the drawn figures

Every figure that is *drawn* rather than plotted from data lives here as the
program that produces it, and its output is committed under `docs/src/assets/`.
The documentation build never runs these scripts, so `Luxor` and `Typstry` stay
out of `docs/Project.toml`.

- Drawing: [Luxor](https://github.com/JuliaGraphics/Luxor.jl).
- Labels: [Typst](https://typst.app), through
  [Typstry](https://github.com/jakobjpeters/Typstry.jl) — mathematics in New
  Computer Modern Math with the notation of the documentation (underlined
  vectors, bold second-order tensors, blackboard fourth-order tensors), words in
  DejaVu Sans. Typst's own SVG is written into each sheet, every glyph defined
  once, so the files are self-contained and light.
- Randomness (inclusion positions, grains, packings): seeded `StableRNG`s, so a
  regeneration gives the same drawing.

| file | output under `docs/src/assets/` |
| :--- | :--- |
| `src/style.jl` | — the palette by role, the sheet and its white panel, `label`, the primitives, the inclusion families, packings, Voronoi cells |
| `src/geometry.jl` | `geometry/`: Eshelby's inclusion and inhomogeneity, the layered sphere, the confocal layered spheroid, the decompositions of the proof of Levin's theorem, with prestresses and with free strains |
| `src/schemes.jl` | `schemes/rve_*.svg`, the morphologies the schemes read, and `schemes/scheme_*.svg`, one card per scheme |
| `src/applications.jl` | `veritz.svg`, `ver_multi_mix.svg`, `lamellar/`, `fe/axi_schematic.svg`, `applications/cement_paste_scales.svg`, `applications/solidifying_sphere.svg` |
| `src/rheology.jl` | `rheology/`: the spring-and-dashpot networks of the model catalog |
| `src/vignettes.jl` | `schemes/particle_assembly.svg`, `nn/network_sketch.svg`, `fe/crack_mesh_sketch.svg` |

## Regenerating

```bash
julia --project=docs/assets_src -e 'using Pkg; Pkg.instantiate()'
julia --project=docs/assets_src docs/assets_src/make.jl            # every family
julia --project=docs/assets_src docs/assets_src/make.jl schemes    # one family
```

Families: `geometry`, `schemes`, `applications`, `rheology`, `vignettes`.

Then **look at the result** — `rsvg-convert -z 2 -o out.png in.svg` renders a
sheet at readable size. Overlapping text is the failure mode of drawings like
these, and nothing in the build reports it.

## Rules the sheets follow

1. **One palette, by role.** Colors are named in `style.jl` by what they stand
   for — matrix, effective medium, coating, aggregate, crack, highlight — and
   never written as hex codes in the figure files, so that one change there
   recolors every figure. The matrix is a light blue, the effective medium a
   light purple; the three inclusion families of the scheme drawings
   (ellipsoids, needles, coated spheres) look the same everywhere.
2. **Every sheet carries its own white panel.** An SVG served as an image
   follows the operating system's color scheme, not the site's theme toggle, so
   it must not rely on the page background.
3. **The symbols are the ones on the page.** A drawing carries the names of the
   constructor arguments and the notation of the formulas it illustrates, so
   that picture, equation and code read side by side.
4. **Layout is computed where it can be.** The rheology sheets place branches
   `PITCH` apart and labels `LABEL_GAP` from their element; zoom lines are the
   common tangents of the spot and of the panel it magnifies.

The MPCM talk draws its own versions of a few of these (French words, no panel)
by including these same files.
