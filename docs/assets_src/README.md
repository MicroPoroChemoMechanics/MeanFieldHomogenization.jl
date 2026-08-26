# `docs/assets_src` — the generators of the drawn figures

Vector figures that are *drawn* rather than plotted from data live here as the
program that produces them, and their output is committed under
`docs/src/assets/`. The documentation build never runs these scripts, so
`Luxor` and `MathTeXEngine` stay out of `docs/Project.toml`.

| generator | output |
|:---|:---|
| `make_rheology_diagrams.jl` | `docs/src/assets/rheology/*.svg` — the spring-and-dashpot networks of the model catalog |

`rheology_symbols.jl` holds the drawing primitives: `spring`, `dashpot`,
`springpot` (the parabolic element), `parallel_frame`, `terminal`, and the
label/caption helpers.

## Regenerating

```bash
julia --project=docs/assets_src docs/assets_src/make_rheology_diagrams.jl
```

Then **look at the result** — `rsvg-convert -z 2 -o out.png in.svg` renders a
sheet at readable size. Overlapping text is the failure mode these sheets are
prone to, and nothing in the build will report it.

## Two rules the sheets follow

1. **The symbols are the ones on the page.** A diagram carries the names of the
   constructor arguments and of the formulas it illustrates, so that picture,
   equation and code snippet can be read side by side. Labels are
   `LaTeXString`s typeset through `MathTeXEngine`, because several of those
   names cannot be written in Unicode at all (there is no subscript `b`).
2. **Layout is computed, not eyeballed.** Branches sit `PITCH` apart, labels
   `LABEL_UP` from their own element, captions `CAPTION_DROP` below the *lowest*
   element of the diagram — not below its axis. Hand-tuned offsets are what
   produce overlapping text when a sheet is later edited.
