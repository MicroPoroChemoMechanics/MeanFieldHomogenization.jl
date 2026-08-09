# Welcome — MFH Studio is starting

You are running a full Linux machine in your browser. Nothing has been
installed on your own computer, and nothing will be. Close the tab when you are
done and it all disappears.

## What is happening right now

A browser tab should open by itself within a few seconds.

If it does not: look for the **PORTS** tab, next to the TERMINAL panel at the
bottom. Find the line labeled *MFH Studio*, hover it, and click the **globe**
icon. That opens the interface.

In the interface, a badge in the top-right corner says `starting…` and turns
green — `Julia ready` — after about ten seconds. Until then the 3-D view and
the **Run** button are asleep. Everything else already works.

## What to do first

1. Click **Open…** — it starts in the `examples` folder.
2. Choose **`01_porous_schemes.jl`**.
3. Press **Run** (top right).

You are looking at a porous material: spherical holes in a solid. The figure
compares every homogenization scheme the package knows, as the porosity goes
from 0 to 1.

Now change something and watch:

- In the left panel, under **PHASES → SOLID**, change `k (bulk)` from `72` to
  `30`. The Julia on the right updates as you type. Press **Run** again.
- Open the **Sweep** tab and change the range, or remove some schemes from the
  list.
- Open **`06_laminate_basics.jl`** for a completely different microstructure —
  a stack of layers, whose answer is exact rather than estimated.

The ten examples are listed in
[`tools/mfhstudio/examples/README.md`](../tools/mfhstudio/examples/README.md),
each one naming the demonstration script it was drawn from.

## Things worth knowing

**The script is the point.** The panel on the right is not a preview — it is
the file. **Save as…** writes it, and it runs on its own with plain Julia,
outside this tool, forever. The studio is a way of writing a script, not a
format you are locked into.

**Nothing you do here is saved to the project.** This is your own throwaway
copy. Edit anything.

**To keep your work**, use **Save as…** inside the studio, then download the
file: right-click it in the file tree on the left of VS Code → *Download…*.

**Free hours.** GitHub gives every account a monthly allowance of Codespaces
time. This machine is the smallest size, which stretches it furthest. When you
are finished, stop the codespace: open <https://github.com/codespaces> and
click *Stop*. It also stops itself after 30 minutes of inactivity.

## If something looks wrong

The studio's own log:

```bash
tail -f /tmp/mfhstudio.log
```

To restart it:

```bash
bash .devcontainer/start-studio.sh
```

If the Julia badge stays red, this tells you why:

```bash
cd tools/mfhstudio && python3 -m mfhstudio --check
```

## Going further

- [The studio's manual](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/stable/tools/mfhstudio/)
  — every panel, and the conventions the interface removes for you.
- [The package documentation](https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl/stable/)
  — the theory, and the library underneath.
- `scripts/` in the file tree — sixty worked demonstrations, in plain Julia.
