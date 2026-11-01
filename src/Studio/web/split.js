/* Draggable separators between the panels.
 *
 * Two mechanisms, because the two axes are laid out differently and pretending
 * otherwise would fight the layout rather than drive it.
 *
 * *Columns* are a CSS grid. Only the two outer tracks are stored — the middle
 * one is `minmax(0, 1fr)` and absorbs whatever is left. That is one number per
 * side instead of three that must be kept summing to the window width, and it
 * survives a window resize with no arithmetic at all.
 *
 * *Rows* are a flex column. A separator redistributes between the pane above it
 * and the pane below, writing the two measured pixel heights straight into
 * `flex-grow`. Grow factors are relative, so pixels are a perfectly good unit
 * for them, and keeping their sum fixed is what makes the move local: the other
 * panes do not shift.
 *
 * Sizes are kept in `localStorage`. Double-clicking a separator drops the
 * stored value for that one and lets the stylesheet answer again.
 */

(function () {
  const KEY = "mfhstudio.layout";

  //: A column never shrinks past this, nor does it squeeze the middle one
  //: below `MIN_MID`: a two-pixel panel is not a smaller panel, it is a lost
  //: one, and getting it back means guessing where the separator went.
  const MIN_COL = 240;
  const MIN_MID = 280;
  const MIN_ROW = 60;

  let layout = { cols: {}, rows: {} };

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const d = JSON.parse(raw);
        layout = { cols: d.cols || {}, rows: d.rows || {} };
      }
    } catch (e) {
      // A corrupt or unavailable store is not worth a broken interface.
      layout = { cols: {}, rows: {} };
    }
  }

  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(layout));
    } catch (e) {
      /* private mode, quota — the layout simply does not persist */
    }
  }

  /* ── applying ───────────────────────────────────────────────────── */

  function applyCols() {
    const root = document.documentElement;
    for (const side of ["l", "r"]) {
      const v = layout.cols[side];
      if (v) root.style.setProperty(`--col-${side}`, v + "px");
      else root.style.removeProperty(`--col-${side}`);
    }
  }

  function applyRows() {
    for (const [id, v] of Object.entries(layout.rows)) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (typeof v === "string" && v.endsWith("px")) {
        // The "fixed-below" panes carry a height, not a share.
        el.style.flex = `0 0 ${v}`;
        el.style.maxHeight = "none";
      } else {
        el.style.flexGrow = v;
      }
    }
  }

  function apply() {
    applyCols();
    applyRows();
  }

  /** Tell the panels their box changed.
   *
   * Plotly's `responsive: true` listens to the *window*, not to the container,
   * so a pane that changes size while the window does not stays drawn at its
   * old dimensions. The connectors are recomputed from `getBoundingClientRect`
   * and need the same nudge.
   */
  function reflow() {
    for (const id of ["view3d", "plot"]) {
      const el = document.getElementById(id);
      if (el && el.children.length && window.Plotly && Plotly.Plots) {
        try {
          Plotly.Plots.resize(el);
        } catch (e) {
          /* nothing plotted yet */
        }
      }
    }
    if (window.drawGraph && !document.querySelector("#graph[hidden]")) window.drawGraph();
  }

  /* ── dragging ───────────────────────────────────────────────────── */

  function startColumn(gutter, e) {
    const main = document.querySelector("main");
    const side = gutter.dataset.col;
    const pane = document.getElementById(side === "l" ? "left" : "right");
    const mid = document.getElementById("middle");
    const start = e.clientX;
    const w0 = pane.getBoundingClientRect().width;
    const mid0 = mid.getBoundingClientRect().width;

    return (ev) => {
      // The right column grows when the pointer moves *left*.
      const delta = (ev.clientX - start) * (side === "l" ? 1 : -1);
      const room = mid0 + w0 - MIN_MID;
      const w = Math.max(MIN_COL, Math.min(room, w0 + delta));
      layout.cols[side] = Math.round(w);
      applyCols();
      void main.offsetWidth;
    };
  }

  /** Put every pane of this column on the same scale before moving one pair.
   *
   * Grow factors are relative, which is what lets pixels be used as factors —
   * but only if *all* the siblings are expressed in them. Writing 362 and 74
   * onto two panes while a third still carries its stylesheet's 2 does not
   * move a boundary, it collapses the third: 2 against 436 is nothing. So the
   * whole column is converted first, and the drag stays the local thing it
   * looks like.
   *
   * A hidden pane (the scale graph, with a single scale) has no height to
   * read, so it is scaled from its stylesheet share instead — otherwise it
   * would come back as a sliver the first time a second scale is added.
   */
  function normalizeRows(gutter) {
    const panes = [...gutter.parentElement.querySelectorAll(".pane")];
    const shown = panes.filter((el) => !el.hidden);
    const px = shown.reduce((s, el) => s + el.getBoundingClientRect().height, 0);
    const grow = shown.reduce(
      (s, el) => s + (parseFloat(getComputedStyle(el).flexGrow) || 0), 0
    );
    const scale = grow > 0 && px > 0 ? px / grow : 1;
    for (const el of panes) {
      layout.rows[el.id] = el.hidden
        ? Math.round((parseFloat(getComputedStyle(el).flexGrow) || 1) * scale)
        : Math.round(el.getBoundingClientRect().height);
    }
    applyRows();
  }

  function startRow(gutter, e) {
    const above = document.getElementById(gutter.dataset.above);
    const below = document.getElementById(gutter.dataset.below);
    if (!above || !below) return null;
    if (gutter.dataset.mode !== "fixed-below") normalizeRows(gutter);
    const start = e.clientY;
    const a0 = above.getBoundingClientRect().height;
    const b0 = below.getBoundingClientRect().height;
    const total = a0 + b0;

    if (gutter.dataset.mode === "fixed-below") {
      // The output log is the one pane with a height rather than a share, so
      // growing it takes from every pane above in proportion — the way a
      // footer does. The bound is therefore the column, not the pane directly
      // above: clamping against that one cut the drag short at whatever height
      // it happened to have.
      const room = gutter.parentElement.getBoundingClientRect().height - 3 * MIN_ROW;
      return (ev) => {
        const b = Math.max(0, Math.min(Math.max(0, room), b0 - (ev.clientY - start)));
        layout.rows[below.id] = Math.round(b) + "px";
        applyRows();
      };
    }

    return (ev) => {
      const delta = ev.clientY - start;
      const a = Math.max(MIN_ROW, Math.min(total - MIN_ROW, a0 + delta));
      // Pixels as grow factors: the ratio is all flexbox reads, and holding
      // the sum fixed keeps every other pane where it was.
      layout.rows[above.id] = Math.round(a);
      layout.rows[below.id] = Math.round(total - a);
      applyRows();
    };
  }

  function wire(gutter) {
    const vertical = gutter.classList.contains("gutter-v");

    gutter.addEventListener("pointerdown", (e) => {
      const move = vertical ? startColumn(gutter, e) : startRow(gutter, e);
      if (!move) return;
      gutter.classList.add("dragging");
      gutter.setPointerCapture(e.pointerId);
      e.preventDefault();

      const onMove = (ev) => move(ev);
      const onUp = () => {
        gutter.classList.remove("dragging");
        gutter.removeEventListener("pointermove", onMove);
        gutter.removeEventListener("pointerup", onUp);
        gutter.removeEventListener("pointercancel", onUp);
        save();
        reflow();
      };
      gutter.addEventListener("pointermove", onMove);
      gutter.addEventListener("pointerup", onUp);
      gutter.addEventListener("pointercancel", onUp);
    });

    gutter.addEventListener("dblclick", () => {
      if (vertical) {
        delete layout.cols[gutter.dataset.col];
      } else {
        // The panes of a column share one scale, so resetting one boundary
        // and leaving the others in pixels would leave the column lopsided in
        // a way no handle put it. Reset the column.
        const ids = [...gutter.parentElement.querySelectorAll(".pane")].map((el) => el.id);
        ids.push(gutter.dataset.below);
        for (const id of ids) {
          delete layout.rows[id];
          const el = document.getElementById(id);
          if (!el) continue;
          el.style.removeProperty("flex-grow");
          el.style.removeProperty("flex");
          el.style.removeProperty("max-height");
        }
      }
      apply();
      save();
      reflow();
    });
  }

  function init() {
    load();
    apply();
    document.querySelectorAll(".gutter").forEach(wire);
  }

  window.Split = { init, reflow };
})();
