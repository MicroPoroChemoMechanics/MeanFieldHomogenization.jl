# Worked examples

Ten models to open with **Open…** and take apart. Each one is a small, running
script *and* a studio model: change a number in the form, watch the Julia on the
right change with it, press **Run**.

Start with `01`, then jump to whichever line interests you.

| file | what it shows | after |
| :--- | :--- | :--- |
| `01_porous_schemes.jl` | all ten schemes on one figure; where they agree, and where they fail | `scripts/28`, `scripts/20` |
| `02_cracked_solid.jl` | cracks enter with a **density**, not a fraction; orientation averaging | `scripts/15`, `scripts/86` |
| `03_coated_inclusion.jl` | one inclusion with concentric shells | `scripts/30` |
| `04_conductivity_fibres.jl` | the same schemes answering a **conduction** problem | `scripts/32` |
| `05_two_scales.jl` | the multiscale seam, and a sweep that reaches through it | `scripts/42` |
| `06_laminate_basics.jl` | the exact laminate; Backus, and both bounds saturating | `scripts/33` |
| `07_laminate_interfaces.jl` | an imperfect interface and the size effect it brings | `scripts/34` |
| `08_laminate_multiscale.jl` | a layer that is itself a homogenized cell | `scripts/36` |
| `09_ageing_creep.jl` | a viscoelastic phase, and the creep curve | `scripts/53`, `scripts/62` |
| `10_sensitivities.jl` | derivatives of the effective property, by autodiff | `scripts/26` |

## How these differ from `scripts/`

The scripts under `scripts/` are the package's own demonstrations: longer,
narrated, and often checking a closed-form result. They build their cells at
top level, the way one writes Julia by hand — which the studio *preserves* but
cannot offer for editing.

These examples are the same models written the way the studio writes them: one
builder function per scale, and a trailing comment block carrying the model
itself. That block is what lets a file reopen exactly as it was left, with every
form filled in.

So: read `scripts/` for the physics, open `examples/` to turn the knobs.

## Regenerating

    python3 examples/build_examples.py

They are generated from `build_examples.py`, not maintained by hand, so an
example cannot drift from what the interface would produce today. Cell ids are
numbered rather than random, which makes regenerating a no-op when nothing has
changed — and `test_every_example_validates_and_is_up_to_date` fails when it is
not.
