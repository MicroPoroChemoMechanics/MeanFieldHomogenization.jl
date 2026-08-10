# MFH Studio

A local web interface for building MeanFieldHomogenization scripts: draw the shape of each
phase, run the model, and read existing scripts back without damaging them.

From Julia, once `using MeanFieldHomogenization` (the studio ships with the checkout):

```julia
mfhstudio()                  # start and open a browser
mfhstudio(port = 9000)       # pick the port
mfhstudio(no_browser = true) # stay in the terminal
mfhstudio(check = true)      # verify the Julia side and exit
```

`mfhstudio` blocks like `Pluto.run()`; Ctrl-C stops the server and its sidecar
cleanly. `wait = false` returns the process handle to keep the REPL free.

The same tool starts from a shell:

```bash
cd tools/mfhstudio
python3 -m mfhstudio                 # start and open a browser
python3 -m mfhstudio --port 9000     # pick the port
python3 -m mfhstudio --no-browser    # stay in the terminal
python3 -m mfhstudio --check         # verify the Julia side and exit
python3 -m mfhstudio --project @env  # use another Julia environment
```

On Windows the command is `python -m mfhstudio`, or `py -3 -m mfhstudio` — a
`python3.exe` there is often a Microsoft Store stub or an MSYS2 build that
cannot import the studio, and the `py` launcher finds the real installation.
The Julia launcher tries all three in that spirit and skips any that fails to
import `mfhstudio`, so it does not settle on an interpreter that would only
fail once the server started.

Over SSH or in a VS Code remote session the studio serves on the *remote*
machine; open the printed URL locally, which VS Code and `ssh -L` both forward.
The console says so when it detects one. It still asks the local browser to
open — VS Code's helper forwards the request — but runs that helper with its
streams redirected, because it is a Node script that otherwise prints a
`url.parse()` deprecation warning into our output, right where it reads like a
failure of ours.

Requires Python 3.10+ (standard library only — no `pip install`) and a `julia`
on `PATH` able to load MeanFieldHomogenization. Set `JULIA` to point at a specific
executable.

### Getting the Julia side to start

`--check` diagnoses this without paying the ten-second load, and is the right
first command on a new machine.

Two things go wrong in practice.

**The environment has never been instantiated.** The sidecar's `using JSON3`
then dies with a stack trace whose only advice is to run `Pkg.instantiate()`.
The sidecar now runs it itself on first start, which takes a few minutes once.
By hand:

    julia --project=<MeanFieldHomogenization.jl> -e 'using Pkg; Pkg.instantiate()'

**The clone's local `Manifest.toml` pins dependencies to sibling checkouts.**
That file is untracked (`.gitignore`) and specific to each development machine;
on one of ours it records, for instance:

    [[deps.TensND]]
    path = "../TensND.jl"

which only resolves when `TensND.jl` sits next to `MeanFieldHomogenization.jl`. That is a
development override, and the way out is a separate environment resolved from
the General registry, where MeanFieldHomogenization itself now lives:

    julia -e 'using Pkg; Pkg.activate("mfhstudio", shared=true); \
              Pkg.add("MeanFieldHomogenization")'

then `python -m mfhstudio --project @mfhstudio`. To point that environment at a
clone you are editing instead of the released version, add the dependencies
first and develop the clone on top —

    julia -e 'using Pkg; Pkg.activate("mfhstudio", shared=true); \
              Pkg.add("TensND"); Pkg.develop(path=raw"<MeanFieldHomogenization.jl>")'

— the `Pkg.add` before the develop being what stops the resolver from turning
those dependencies back into path entries. `--check` reads the manifest and
names the missing checkout itself, so it will tell you which packages to add.

If Julia is unavailable for any reason the interface still comes up: you can
build and save a script, and a banner says what is off. Only the 3-D view,
reading a script back, and Run need the sidecar.

The user-facing guide, with screenshots, is
`docs/src/tools/mfhstudio.md`. This file covers how the thing is put together.

## Shape

```
browser (HTML/JS, Plotly, draggable scale graph)
    │  REST
    ▼
Python  mfhstudio/            model state, Julia generation, script read-back
    │  JSON lines over stdio
    ▼
Julia   sidecar               3-D traces, Meta.parse, execution
```

The model lives in Python. That is what makes the sidecar disposable: it can be
restarted after a wedge without losing work.

| file | role |
| :--- | :--- |
| `mfhstudio/model.py` | the authoring model — a *graph of cells*, not one RVE |
| `mfhstudio/codegen.py` | model → Julia |
| `mfhstudio/readback.py` | script → model, with verbatim preservation |
| `mfhstudio/juliabridge.py` | the sidecar process: start, call, restart |
| `mfhstudio/catalog.py` | the form definitions — available without Julia |
| `mfhstudio/convert.py` | Open on a `.py` runs echoes2mfh in-process |
| `mfhstudio/server.py` | HTTP endpoints, session state, the file browser |
| `julia/sidecar.jl` | the JSON-lines loop |
| `julia/introspect.jl` | the feature catalog, read from the live package |
| `julia/geometry.jl` | 3-D traces |
| `julia/parse_script.jl` | `Meta.parse` → nodes that tile the file exactly |
| `web/graph.js` | the draggable scale graph |
| `web/split.js` | the panel separators |
| `web/picker.js` | the file dialog |
| `examples/build_examples.py` | regenerates `examples/*.jl` |

## Two kinds of cell

`Cell.kind` is `"rve"` or `"laminate"`, and the difference runs the whole depth
of the app: an RVE has a matrix and inclusion phases, a laminate has an ordered
stack of layers and a normal. What they share is that both carry *members* with
a name and a property list, which is all the multiscale seam needs — so
`Cell.members()` (and `cellMembers` in `graph.js`, its one JavaScript twin) is
what the dependency walk, the validation, the ports and the connectors are
written against. Anything that reaches for `.phases` directly is a bug waiting
for a laminate.

Only `Laminated`, `Voigt` and `Reuss` apply to a laminate; the constraint is
published in the catalog as a name list and *intersected* with the introspected
schemes rather than substituted for them, so the "never hard-code the scheme
list" rule survives.

## Three design decisions worth knowing

**The catalog has two halves.** Form definitions — which fields a spheroid
needs, which lenses exist — are interface concerns and live in Python, so they
are there immediately. Only the scheme list and each scheme's solver options
come from the sidecar, and they replace the fallback wholesale rather than
merging, so a scheme MeanFieldHomogenization drops disappears. Putting the forms behind
the sidecar is what once made every control dead on a machine where Julia
failed to start.

**The scheme list is introspected, never hard-coded.** Schemes come from
`subtypes(HomogenizationScheme)`, and each scheme's solver options are read
from the constant it declares for the purpose (`_SC_SOLVER_KWARGS`,
`_DIFF_RESERVED_OPTIONS`). Probing the constructor would not do: those schemes
take a `kwargs...` bag that accepts anything, so `SelfConsistent(; nsteps = 3)`
succeeds while `nsteps` is meaningless there. Reading the declared keys is what
keeps the interface exactly in step with the schemes.

**3-D reuses the parametrizations, not the trace builders.**
`scripts/common/docviz.jl` — the code that draws the documentation figures —
returns JavaScript object literals from its `*_trace` functions, which are not
JSON and would have to be injected into the page as executable script. The
numeric layer under them (`ellipsoid_surface`, `cylinder_surface`,
`disc_surface`, `param_surface`) returns plain arrays, so real JSON is built
from those. Shapes therefore look the same in the interface as in the manual,
without the injection.

**A property's frame is not its phase's frame.** Anisotropic forms carry their
own Euler angles, separate from the shape's: a tilted fiber made of an untilted
material and an untilted fiber made of a tilted material are different
materials. What the angles become depends on the symmetry class — a
transversely isotropic tensor takes an *axis* (`vecbasis(RotatedBasis(...))[:, 3]`,
ψ being irrelevant), an orthotropic one takes the basis itself. The angle fields
accept `π/4` and `2pi/3` and the expression reaches the script as written,
which is both exact and readable where the decimal is neither.

**Read-back only claims what it can prove.** Beyond the structural checks,
before accepting a construct the reader renders it exactly as it would be saved
and compares against the source it came from. If the two differ, the original
text is kept and the reason is reported. Parsing something is not the same as
understanding it, and the difference is exactly where silent damage would come
from.

## The protocol

One JSON object per line, both directions:

```json
{"id": 7, "op": "traces", "payload": {"expr": "Spheroid(0.4)", "cutaway": true}}
{"id": 7, "ok": true, "result": {"data": [...], "layout": {...}}}
```

Ops: `ping`, `catalog`, `traces`, `parse`, `run`. The first line the sidecar
ever writes is `{"event": "ready", ...}`.

`run` executes in a fresh anonymous module with stdout redirected to a
temporary file. The redirect has to be undone however the script ends —
otherwise the protocol's own replies would vanish into the capture — so a
timeout *interrupts* the task rather than abandoning it, and the reply says
`wedged` if the task refused to unwind, which makes the client restart.

## Tests

```bash
python3 tests/test_studio.py          # no Julia needed
python3 tests/test_studio.py --julia  # adds the sidecar-backed tests
```

The load-bearing one is preservation: every script under `scripts/` is opened
and written back, and no line may be lost.

`examples/*.jl` are generated, and two tests keep them honest — every example
must validate with no problems, and regenerating must be a no-op. Cell ids are
numbered rather than random for exactly that reason. After changing the emitter:

```bash
python3 examples/build_examples.py
```

## Known limits

- Custom, FE and neural inclusions are not modeled. Scripts using them open and
  are preserved, but those parts are not editable.

- Read-back claims the vocabulary the emitter *writes* — one builder function
  per scale. A cell assembled at top level, which is how every demo under
  `scripts/` is written, is preserved verbatim and not offered as a form. That
  is what `examples/` is for.

- An inner `Homogenized` cannot sit inside an ALV chain (MeanFieldHomogenization cannot
  re-express a homogenized result as a `ViscoLaw`); the interface blocks the
  combination.
- `web/vendor/plotly.min.js` is the official *plotly-gl3d* partial bundle,
  vendored so the interface works with no outbound network.
