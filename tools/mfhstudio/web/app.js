/* MFH Studio — the browser side.
 *
 * The model lives on the server; this file edits a local copy, POSTs it back
 * and re-renders from the answer. That keeps one source of truth and means the
 * Julia script shown on the right is always the one that would be saved.
 */

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
};

const S = {
  model: null,
  catalog: null,
  cellId: null,
  phaseIdx: 0,
  problems: [],
  keptReport: null,
  path: null,
  catalogIntrospected: false,
};

/* ── server ─────────────────────────────────────────────────────── */

async function api(path, body) {
  const opt = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
  if (j && j.error) throw new Error(j.error);
  return j;
}

function toast(msg, bad = false) {
  const t = el("div", { class: bad ? "bad" : "" }, msg);
  $("#toast").append(t);
  setTimeout(() => t.remove(), bad ? 9000 : 3500);
}

/** Push the edited model and re-render from the server's answer. */
async function push() {
  try {
    const st = await api("/api/model", { model: S.model });
    apply(st);
  } catch (e) {
    toast(e.message, true);
  }
}

function apply(st) {
  S.model = st.model;
  S.problems = st.problems || [];
  if (!S.cellId || !S.model.cells.some((c) => c.id === S.cellId)) {
    S.cellId = S.model.cells.length ? S.model.cells[0].id : null;
  }
  $("#code").textContent = st.script;
  $("#problems").textContent = S.problems.length
    ? `${S.problems.length} problem(s)`
    : "";
  $("#problems").className = S.problems.length ? "problem" : "muted";
  S.path = st.path || S.path;
  setPathLabel(S.path);
  render();
  draw3d();
}

/* ── helpers over the model ─────────────────────────────────────── */

const cell = () => S.model.cells.find((c) => c.id === S.cellId) || null;
const phases = () => (cell() ? cell().phases : []);
const phase = () => phases()[S.phaseIdx] || null;
const isLaminate = (c = cell()) => !!c && c.kind === "laminate";
/** The cell carrying the final result, which is what a panel targets by default. */
const rootCell = () =>
  S.model.cells.find((c) => c.id === S.model.root_cell)
  || S.model.cells[S.model.cells.length - 1] || null;
/** Phases or layers, whichever this cell has — `graph.js` owns the definition. */
const members = (c = cell()) => cellMembers(c);

/** The schemes a cell can be homogenized with.
 *
 * A laminate takes only the three that need no matrix phase. The list is still
 * the introspected one, intersected rather than substituted, so a scheme
 * MeanFieldHomogenization drops disappears here too.
 */
function schemesFor(c) {
  const all = S.catalog.schemes || [];
  if (!isLaminate(c)) return all;
  const allowed = (S.catalog.constraints || {}).laminate_schemes || [];
  return all.filter((s) => allowed.includes(s.name));
}

const geomForm = (kind) =>
  (S.catalog.geometries || []).find((g) => g.kind === kind) || null;
const propForm = (name) =>
  (S.catalog.properties || []).find((p) => p.name === name) || null;

/** Render a value as a Julia *float* literal.
 *
 * Geometry constructors take an NTuple of one element type, so a stray
 * integer makes `LayeredSphere((0.6, 1), …)` a `Tuple{Float64, Int64}` and
 * the call fails to dispatch. JavaScript has no int/float distinction, so
 * the decimal point has to be put back explicitly.
 */
function jnum(v) {
  if (typeof v === "string") return v.trim() === "" ? "0.0" : v;
  if (!isFinite(v)) return String(v);
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
}

/** A Julia tuple; only a 1-tuple needs the trailing comma. */
function jtuple(xs) {
  return xs.length === 1 ? `(${xs[0]},)` : `(${xs.join(", ")})`;
}

/** `; euler_angles = (…)` for the preview, empty when the shape is unrotated. */
function anglesExpr(g) {
  const a = (g.euler_angles || []).filter((x) => x !== "" && x != null);
  if (!a.length) return "";
  const vals = a.map(jnum).join(", ");
  return a.length === 1 ? `; euler_angles = (${vals},)` : `; euler_angles = (${vals})`;
}

/** The Julia expression for a phase's geometry, for the 3-D preview.
 *
 * The orientation belongs here: a preview that drew every inclusion in the
 * canonical frame would quietly disagree with the script it is previewing.
 */
function geomExpr(ph) {
  if (!ph) return "";
  const g = ph.geometry, a = g.args || {};
  const n = (k, d) => {
    const v = a[k];
    return jnum(v === undefined || v === "" ? d : v);
  };
  const rot = anglesExpr(g);
  switch (g.kind) {
    case "spheroid": return `Spheroid(${n("omega", 1)}${rot})`;
    case "ellipsoid": return `Ellipsoid(${n("a", 1)}, ${n("b", 1)}, ${n("c", 1)}${rot})`;
    case "cylinder": return `Cylinder(${n("b", 1)}, ${n("c", 1)}${rot})`;
    case "penny_crack": return `PennyCrack(${n("a", 1)}${rot})`;
    case "elliptic_crack": return `EllipticCrack(${n("a", 1)}, ${n("b", 0.5)}${rot})`;
    case "ribbon_crack": return `RibbonCrack(${n("b", 1)}${rot})`;
    case "layered_sphere": {
      const r = (g.layers || []).map((l) => jnum(l.radius));
      if (!r.length) return "";
      const mods = (g.layers || []).map(() => "iso_stiffness(1.0, 1.0)");
      return `LayeredSphere(${jtuple(r)}, ${jtuple(mods)})`;
    }
    case "layered_spheroid": {
      // The moduli do not change the picture, so the preview uses placeholders
      // and only the geometry has to be right.
      const f = (g.layers || []).map((l) => jnum(l.fraction ?? 1));
      if (!f.length) return "";
      const mods = f.map(() => "TensISO{3}(1.0)");
      return `layered_spheroid_from_fractions(${n("omega", 0.5)}, `
        + `${n("radius", 1)}, ${jtuple(f)}, ${jtuple(mods)}; `
        + `Nseries = ${Math.max(1, parseInt(a.Nseries, 10) || 5)})`;
    }
    default: return "";
  }
}

/* ── rendering ──────────────────────────────────────────────────── */

function render() {
  const focus = captureFocus();
  _keySeq = 0;
  renderScales();
  renderSweep();
  renderSens();
  renderAlv();
  renderParams();
  renderKept();
  restoreFocus(focus);
}

function renderScales() {
  const sel = $("#cell-select");
  sel.replaceChildren(
    ...S.model.cells.map((c) =>
      el("option", { value: c.id, selected: c.id === S.cellId }, c.name)
    )
  );
  const c = cell();
  $("#cell-name").value = c ? c.name : "";
  $("#cell-matrix").value = c ? c.matrix_name : "";
  // A laminate has no matrix — that is the point of it — so the field goes
  // away rather than sitting there inert.
  $("#cell-matrix-field").hidden = isLaminate(c);
  $("#cell-kind").value = c ? (c.kind || "rve") : "rve";
  $("#phase-head").hidden = isLaminate(c);

  // The graph only earns its place once there is more than one scale: with a
  // single RVE there is nothing to connect and it would be noise.
  const multi = S.model.cells.length > 1;
  $("#graph").hidden = !multi;
  $("#graph-head").hidden = !multi;
  // The separator below the graph goes with it: a handle that resizes nothing
  // is worse than no handle.
  $("#gutter-graph").hidden = !multi;
  if (multi) drawGraph();

  $("#phases").replaceChildren(
    ...(isLaminate(c) ? laminateEditor(c) : phases().map(phaseCard))
  );
}

/* ── laminate ───────────────────────────────────────────────────── */

/** The stack: its frame, then one card per layer in stacking order.
 *
 * A layer is not a phase. It has no shape of its own — the geometry of a
 * laminate is the stacking direction, which belongs to the cell — and no
 * orientation average, the cell being deterministic rather than random. What
 * it does have is the same property list a phase has, so `propertyEditor` is
 * reused whole and a layer gets the nested seam, the viscoelastic laws and the
 * anisotropic forms without any of them being written twice.
 */
function laminateEditor(c) {
  c.layers = c.layers || [];
  const kind = c.layers.length ? c.layers[0].amount_kind : "fraction";

  const out = [
    el("div", { class: "note" },
      "A periodic stack of parallel layers of common normal — no matrix and no "
      + "reference medium. `Laminated` is exact for it; `Voigt` and `Reuss` "
      + "bracket it, and two of the bracketings are equalities."),
    el("h3", {}, "Stacking direction"),
    field("Given as", select(
      [["normal", "normal vector"], ["euler", "ZYZ Euler angles"]],
      c.frame_mode || "normal",
      (v) => { c.frame_mode = v; push(); }
    )),
  ];

  if ((c.frame_mode || "normal") === "euler") {
    out.push(anglesEditor(c, 3));
  } else {
    c.normal = c.normal && c.normal.length === 3 ? c.normal : [0, 0, 1];
    out.push(
      el("div", { class: "note" },
        "(0, 0, 1) is the canonical frame, where the kernel skips the rotation "
        + "altogether."),
      el("div", { class: "grid3" },
        ...["nₓ", "n_y", "n_z"].map((lbl, i) =>
          field(lbl, input(c.normal[i] ?? 0, (v) => {
            c.normal[i] = isFinite(+v) && v.trim() !== "" ? +v : v;
            push();
          }))
        ))
    );
  }

  out.push(
    el("h3", {}, "Layers",
      el("button", {
        class: "small",
        title: "Add a layer on top of the stack",
        onclick: () => {
          c.layers.push({
            name: String.fromCharCode(65 + c.layers.length),
            amount_kind: kind, amount: kind === "fraction" ? 0.5 : 1.0,
            interface: { kind: "PerfectInterface", args: {} },
            properties: [{
              key: ":C", source: "builder", builder: "iso_stiffness",
              form: "iso_kmu", args: { k: 10, mu: 5 }, scheme_options: {},
            }],
          });
          push();
        },
      }, "+")),
    // The choice belongs to the cell, not the layer: MeanFieldHomogenization
    // refuses a stack that mixes the two forms, because with an imperfect
    // interface the absolute period is physically meaningful.
    field("Layers given by", select(
      [["fraction", "volume fraction (Σ = 1)"], ["thickness", "absolute thickness"]],
      kind,
      (v) => { for (const l of c.layers) l.amount_kind = v; push(); }
    )),
    el("div", { class: "note" }, kind === "fraction"
      ? "Fractions of the period; they must sum to 1. The period is then "
        + "irrelevant — unless an interface is imperfect, which is what "
        + "absolute thicknesses are for."
      : "Absolute heights. Their sum is the period L, and an imperfect "
        + "interface enters with weight 1/L — an interface density.")
  );

  c.layers.forEach((l, i) => out.push(layerCard(c, l, i)));
  return out;
}

function layerCard(c, l, i) {
  const seam = (l.properties || []).some((p) => p.source === "cell");
  const move = (d) => {
    const j = i + d;
    if (j < 0 || j >= c.layers.length) return;
    c.layers.splice(j, 0, c.layers.splice(i, 1)[0]);
    push();
  };
  const itf = l.interface || (l.interface = { kind: "PerfectInterface", args: {} });
  const itfForm = (S.catalog.interfaces || []).find((x) => x.name === itf.kind);

  return el("div", { class: "card" },
    el("header", {},
      el("b", {}, `${i + 1}. ${l.name || "(unnamed)"}`),
      seam ? el("span", { class: "tag seam" }, "nested scale") : null,
      el("button", { class: "small", title: "Move down the stack", onclick: () => move(-1) }, "↑"),
      el("button", { class: "small", title: "Move up the stack", onclick: () => move(+1) }, "↓"),
      el("button", { class: "small", onclick: () => { c.layers.splice(i, 1); push(); } }, "−")),
    el("div", { class: "grid2" },
      field("Name", input(l.name, (v) => { l.name = v; push(); })),
      field(l.amount_kind === "thickness" ? "thickness h" : "fraction f",
        input(l.amount, (v) => {
          l.amount = isFinite(+v) && v.trim() !== "" ? +v : v;
          push();
        }))
    ),
    el("h3", {}, "Properties",
      el("button", {
        class: "small",
        onclick: () => {
          l.properties.push({
            key: ":C", source: "builder", builder: "iso_stiffness",
            form: "iso_kmu", args: { k: 10, mu: 5 }, scheme_options: {},
          });
          push();
        },
      }, "+")),
    ...(l.properties || []).map((pr, j) => propertyEditor(l, pr, j)),
    el("h3", {}, "Interface on top"),
    el("div", { class: "note" },
      i === c.layers.length - 1
        ? "The last one closes the cell back onto layer 1 by periodicity."
        : `Between this layer and ${(c.layers[i + 1] || {}).name || "the next"}.`),
    select(
      (S.catalog.interfaces || []).map((x) => [x.name, x.label]),
      itf.kind,
      (v) => {
        const f = (S.catalog.interfaces || []).find((x) => x.name === v);
        const args = {};
        for (const fl of (f && f.fields) || []) args[fl.name] = fl.default;
        l.interface = { kind: v, args };
        push();
      }
    ),
    itfForm && itfForm.doc ? el("div", { class: "note" }, itfForm.doc) : null,
    ...((itfForm && itfForm.fields) || []).map((fl) =>
      field(fl.label, input(itf.args[fl.name] ?? fl.default, (v) => {
        itf.args[fl.name] = fl.type === "code"
          ? v
          : (isFinite(+v) && v.trim() !== "" ? +v : v);
        push();
      })))
  );
}

function phaseCard(ph, i) {
  const seam = (ph.properties || []).some((p) => p.source === "cell");
  const head = el(
    "header", {},
    el("b", {}, ph.name || "(unnamed)"),
    ph.is_matrix ? el("span", { class: "tag matrix" }, "matrix") : null,
    seam ? el("span", { class: "tag seam" }, "nested scale") : null,
    el("button", {
      class: "small",
      title: "Remove this phase",
      onclick: (e) => { e.stopPropagation(); cell().phases.splice(i, 1); push(); },
    }, "−")
  );

  const body = el("div", {});
  if (i === S.phaseIdx) {
    body.append(
      field("Name", input(ph.name, (v) => { ph.name = v; push(); })),
      field("Role", select(
        [["inclusion", "inclusion"], ["matrix", "matrix"]],
        ph.is_matrix ? "matrix" : "inclusion",
        (v) => {
          ph.is_matrix = v === "matrix";
          if (ph.is_matrix) {
            // MFH derives the matrix amount as 1 − Σ f and refuses to be told
            // otherwise, so the field simply goes away.
            for (const o of cell().phases) if (o !== ph) o.is_matrix = false;
            cell().matrix_name = ph.name;
          }
          push();
        }
      )),
      ph.is_matrix
        ? el("div", { class: "note" },
            "The matrix amount is derived as 1 − Σ f of the inclusions; MeanFieldHomogenization raises if it is set.")
        : el("div", { class: "grid2" },
            field("Amount", select(
              [["fraction", "volume fraction"], ["density", "crack density"]],
              ph.amount_kind, (v) => { ph.amount_kind = v; push(); }
            )),
            field("Value", input(ph.amount, (v) => {
              ph.amount = isFinite(+v) && v.trim() !== "" ? +v : v;
              push();
            }))
          ),
      geometryEditor(ph),
      field("Orientation average", select(
        (S.catalog.symmetrize || []).map((s) => [s.name, s.label]),
        ph.symmetrize, (v) => { ph.symmetrize = v; push(); }
      )),
      el("h3", {}, "Properties",
        el("button", {
          class: "small",
          onclick: () => {
            ph.properties.push({
              key: ":C", source: "builder", builder: "iso_stiffness",
              form: "iso_kmu", args: { k: 10, mu: 5 }, scheme_options: {},
            });
            push();
          },
        }, "+")),
      ...(ph.properties || []).map((pr, j) => propertyEditor(ph, pr, j))
    );
  }

  return el("div", {
    class: "card" + (i === S.phaseIdx ? " selected" : ""),
    // Selecting a phase must not steal clicks meant for the controls inside
    // it. Re-rendering here would destroy the very input or select the user
    // just pressed, so the field never takes focus and a dropdown closes the
    // instant it opens — which looks exactly like "the mouse does nothing"
    // while the keyboard still works, since Tab fires no click.
    onclick: (e) => {
      if (e.target.closest("input, select, textarea, button, label, option")) return;
      if (i === S.phaseIdx) return;
      S.phaseIdx = i;
      render();
      draw3d();
    },
  }, head, body);
}

function geometryEditor(ph) {
  const g = ph.geometry;
  const form = geomForm(g.kind);
  const box = el("div", {},
    field("Shape", select(
      (S.catalog.geometries || []).map((x) => [x.kind, x.name]),
      g.kind,
      (v) => {
        g.kind = v;
        const f = geomForm(v);
        g.args = {};
        for (const fl of (f && f.fields) || []) g.args[fl.name] = fl.default;
        // Layers belong to the shape: switching kind rebuilds them so a
        // sphere's radii never masquerade as a spheroid's fractions.
        if (f && f.layered) g.layers = defaultLayers(f);
        else g.layers = [];
        push();
      }
    ))
  );
  if (form && form.doc) box.append(el("div", { class: "note" }, form.doc));
  const fields = (form && form.fields) || [];
  if (fields.length) {
    box.append(el("div", { class: fields.length > 2 ? "grid3" : "grid2" },
      ...fields.map((f) =>
        field(f.label, input(g.args[f.name] ?? f.default, (v) => {
          g.args[f.name] = isFinite(+v) && v.trim() !== "" ? +v : v;
          push();
        }))
      )));
  }
  if (form && form.angles) box.append(anglesEditor(g, form.angles));
  if (form && form.layered) box.append(layersEditor(g, form));
  return box;
}

/** Euler angles, as many as the shape has.
 *
 * Radians, because that is what MeanFieldHomogenization takes and what the generated
 * script must contain: converting here would either put `deg2rad(...)` in the
 * script or leave an opaque decimal in it, and neither reads back cleanly.
 * The degree equivalent is shown beside the field instead.
 */
const ANGLE_NAMES = ["θ", "φ", "ψ"];

/** Evaluate an angle written as a small arithmetic expression, or NaN.
 *
 * Angles are the one place where the exact value has a name: `π/4` is worth
 * writing, and rounding it to seventeen digits in the script loses both the
 * intent and the exactness. So the field accepts `pi`/`π` arithmetic, keeps
 * the text for the script, and evaluates it only to show the degrees and to
 * orient the 3-D preview.
 *
 * The grammar is restricted to numbers, π and `+ - * / ^ ( )` — anything else
 * is refused rather than handed to the evaluator.
 */
function angleValue(v) {
  if (typeof v === "number") return v;
  if (typeof v !== "string" || v.trim() === "") return NaN;
  const src = v.replace(/π/g, "pi").trim();
  if (!/^[-+*/^().\s\d]*(pi[-+*/^().\s\d]*)*$/.test(src)) return NaN;
  // `2pi` is Julia's implicit multiplication, not JavaScript's syntax error.
  const js = src.replace(/(\d)\s*pi/g, "$1*pi").replace(/\bpi\b/g, "Math.PI")
                .replace(/\^/g, "**");
  try {
    const x = Function(`"use strict"; return (${js});`)();
    return typeof x === "number" && isFinite(x) ? x : NaN;
  } catch (e) {
    return NaN;
  }
}

function anglesEditor(g, count) {
  g.euler_angles = g.euler_angles || [];
  const deg = (v) => {
    const x = angleValue(v);
    return isFinite(x) ? `${(x * 180 / Math.PI).toFixed(1)}°` : "";
  };
  const box = el("div", {},
    el("h3", {}, "Orientation",
      el("button", {
        class: "small",
        title: "Back to the canonical frame",
        onclick: () => { g.euler_angles = []; push(); },
      }, "reset")),
    el("div", { class: "note" },
      "ZYZ Euler angles in radians — `π/4`, `2pi/3` and plain decimals all "
      + "work, and an expression is kept as written in the script. "
      + (count === 2
        ? "θ and φ point the symmetry axis."
        : "θ, φ, ψ orient the principal axes."))
  );
  const grid = el("div", { class: count > 2 ? "grid3" : "grid2" });
  for (let i = 0; i < count; i++) {
    const cur = g.euler_angles[i];
    grid.append(field(
      `${ANGLE_NAMES[i]} (rad) ${deg(cur)}`,
      input(cur ?? "", (v) => {
        const t = v.trim();
        // Keep the array dense up to the last angle the user set: MFH takes a
        // tuple of 0-3 and reads it positionally.
        g.euler_angles[i] = t === "" ? 0.0 : (isFinite(+t) && t !== "" ? +t : t);
        while (g.euler_angles.length && isTrailingZero(g.euler_angles)) {
          g.euler_angles.pop();
        }
        push();
      })
    ));
  }
  box.append(grid);
  return box;
}

/** A trailing exact zero carries no information: dropping it keeps the
 *  generated call short and the canonical frame free of `euler_angles`. */
function isTrailingZero(arr) {
  const last = arr[arr.length - 1];
  return last === 0 || last === 0.0;
}

/** The layer table.
 *
 * A layered sphere is described by the outer radius of each shell; a confocal
 * spheroid cannot be, because the constructor requires one shared focal
 * distance and hand-entered radii never give it. That one is described by
 * volume fraction instead, and MFH solves for the confocal radii. The form
 * follows whichever the geometry declares.
 */
function defaultLayers(form) {
  const byFraction = form.layer_by === "fraction";
  const propName = form.layer_property || "iso_kmu";
  const pf = propForm(propName) || { builder: "iso_stiffness", fields: [] };
  const mkArgs = (scale) => {
    const a = {};
    for (const fl of pf.fields || []) a[fl.name] = fl.default * scale;
    return a;
  };
  const layer = (radius, fraction, scale) => ({
    radius, fraction,
    property: {
      key: byFraction ? ":K" : ":C", source: "builder",
      builder: pf.builder, form: propName, args: mkArgs(scale), scheme_options: {},
    },
    interface: { kind: "PerfectInterface", args: {} },
  });
  return [layer(0.6, 0.3, 1), layer(1.0, 0.7, 3)];
}

function layersEditor(g, form) {
  const byFraction = (form && form.layer_by) === "fraction";
  const propName = (form && form.layer_property) || "iso_kmu";
  const mk = () => {
    const f = propForm(propName) || { builder: "iso_stiffness", fields: [] };
    const args = {};
    for (const fl of f.fields || []) args[fl.name] = fl.default;
    return {
      radius: 1.0, fraction: 0.5,
      property: {
        key: byFraction ? ":K" : ":C", source: "builder",
        builder: f.builder, form: propName, args, scheme_options: {},
      },
      interface: { kind: "PerfectInterface", args: {} },
    };
  };

  const box = el("div", {},
    el("h3", {}, "Layers",
      el("button", {
        class: "small",
        onclick: () => {
          g.layers = g.layers || [];
          const l = mk();
          if (!byFraction) {
            const last = g.layers.length ? +g.layers[g.layers.length - 1].radius : 0.5;
            l.radius = (isFinite(last) ? last : 0.5) + 0.5;
          }
          g.layers.push(l);
          push();
        },
      }, "+")),
    el("div", { class: "note" }, byFraction
      ? "Core first. Fractions are of the total volume and are normalized; "
        + "the confocal radii are solved for."
      : "Ascending radii, r = 0 implicit at the center; layer 1 is the core.")
  );

  (g.layers || []).forEach((l, i) => {
    const pf = propForm(l.property && l.property.form) || propForm(propName);
    const fields = (pf && pf.fields) || [];
    const row = el("div", { class: fields.length > 1 ? "grid3" : "grid2" },
      byFraction
        ? field("fraction", input(l.fraction ?? 0.5, (v) => {
            l.fraction = isFinite(+v) && v.trim() !== "" ? +v : v; push();
          }))
        : field("outer r", input(l.radius, (v) => {
            l.radius = isFinite(+v) && v.trim() !== "" ? +v : v; push();
          })),
      ...fields.map((fl) => field(fl.label, input(
        (l.property.args || {})[fl.name] ?? fl.default,
        (v) => {
          l.property.args[fl.name] = isFinite(+v) && v.trim() !== "" ? +v : v;
          push();
        }
      )))
    );
    box.append(el("div", { class: "card" },
      el("header", {}, el("b", {}, `layer ${i + 1}`),
        el("button", { class: "small", onclick: () => { g.layers.splice(i, 1); push(); } }, "\u2212")),
      row,
      // The anisotropic interfaces are laminate-only: they are written in the
      // layer frame (ℓ, m, n), and a concentric shell has no such frame.
      field("Interface with the next layer", select(
        (S.catalog.interfaces || [])
          .filter((x) => !x.laminate_only)
          .map((x) => [x.name, x.label]),
        (l.interface && l.interface.kind) || "PerfectInterface",
        (v) => {
          const f = (S.catalog.interfaces || []).find((x) => x.name === v);
          const args = {};
          for (const fl of (f && f.fields) || []) args[fl.name] = fl.default;
          l.interface = { kind: v, args };
          push();
        }
      ))
    ));
  });
  return box;
}

function propertyEditor(ph, pr, j) {
  const others = S.model.cells.filter((c) => c.id !== S.cellId);
  const sourceOptions = [
    ["builder", "from moduli"],
    ["expr", "Julia expression"],
  ];
  if (others.length) sourceOptions.push(["cell", "another scale (nested)"]);

  const box = el("div", { class: "card" },
    el("header", {},
      el("b", {}, pr.key),
      el("button", { class: "small", onclick: () => { ph.properties.splice(j, 1); push(); } }, "−")),
    el("div", { class: "grid2" },
      field("Key", input(pr.key, (v) => { pr.key = v.startsWith(":") ? v : ":" + v; push(); })),
      field("Source", select(sourceOptions, pr.source, (v) => { pr.source = v; push(); }))
    )
  );

  if (pr.source === "builder") {
    box.append(field("Parametrization", select(
      (S.catalog.properties || []).map((p) => [p.name, p.label]),
      pr.form || "iso_kmu",
      (v) => {
        const f = propForm(v);
        pr.form = v;
        pr.builder = (f && f.builder) || "iso_stiffness";
        pr.args = {};
        for (const fl of (f && f.fields) || []) pr.args[fl.name] = fl.default;
        push();
      }
    )));
    const f = propForm(pr.form || "iso_kmu");
    if (f && f.doc) box.append(el("div", { class: "note" }, f.doc));
    const fields = (f && f.fields) || [];
    box.append(el("div", { class: fields.length > 2 ? "grid3" : "grid2" },
      ...fields.map((fl) =>
        field(fl.label, input(pr.args[fl.name] ?? fl.default, (v) => {
          pr.args[fl.name] = isFinite(+v) && v.trim() !== "" ? +v : v;
          push();
        }))
      )));
    // An anisotropic tensor means nothing without the frame its constants are
    // written in. This is *not* the shape's orientation above: the two are
    // independent, and a tilted fiber made of an untilted material is a
    // different material from an untilted fiber made of a tilted one.
    if (f && f.orientation) {
      pr.euler_angles = pr.euler_angles || [];
      box.append(anglesEditor(pr, f.orientation));
    }
  } else if (pr.source === "expr") {
    box.append(field("Julia", input(pr.expr || "", (v) => { pr.expr = v; push(); })));
  } else if (pr.source === "cell") {
    box.append(
      el("div", { class: "note" },
        "The seam: this property is the effective value of another scale, "
        + "resolved by the outer scheme when it reads the key."),
      el("div", { class: "grid2" },
        field("Inner scale", select(
          others.map((c) => [c.id, c.name]),
          pr.cell || (others[0] && others[0].id),
          (v) => { pr.cell = v; push(); }
        )),
        field("Homogenized with", select(
          (S.catalog.schemes || []).map((s) => [s.name, s.name]),
          pr.scheme || "MoriTanaka",
          (v) => { pr.scheme = v; push(); }
        ))
      ),
      schemeOptions(pr.scheme || "MoriTanaka", pr.scheme_options || (pr.scheme_options = {}))
    );
  }
  return box;
}

/** Editable solver options for a scheme, straight from the catalog. */
function schemeOptions(name, target) {
  const s = (S.catalog.schemes || []).find((x) => x.name === name);
  const opts = ((s && s.options) || []).filter((o) => o.editable);
  if (!opts.length) return el("div", { class: "note" }, "No solver options.");
  return el("div", { class: "grid2" },
    ...opts.map((o) =>
      field(o.name, typeof o.default === "boolean"
        ? checkbox(target[o.name] ?? o.default, (v) => { target[o.name] = v; push(); })
        : input(target[o.name] ?? o.default, (v) => {
            target[o.name] = v.trim() === "" ? null : +v;
            push();
          }))
    ));
}

/* ── sweep ──────────────────────────────────────────────────────── */

const OUTPUT_KINDS = [
  ["k", "k (bulk)"], ["mu", "μ (shear)"], ["E", "E"], ["nu", "ν"],
  ["km", "Kelvin-Mandel component"], ["comp", "tensor component (2nd order)"],
  ["trace3", "tr/3 (mean conductivity)"],
];
const ISOTROPIC_ONLY = ["k", "mu", "E", "nu"];

function renderSweep() {
  const sw = S.model.sweep;
  sw.schemes = sw.schemes && sw.schemes.length
    ? sw.schemes : [{ name: "MoriTanaka", options: {} }];
  sw.outputs = sw.outputs && sw.outputs.length ? sw.outputs : [{ kind: "k" }];
  const t = $("#tab-sweep");
  const inner = sw.lens.inner || (sw.lens.inner = { kind: "amount", phase: "", property: ":C", field_name: "semi_axes", index: 1, member: "", inner: null });
  const sweeping = sw.enabled && sw.mode !== "single";
  // The sweep runs on one cell, and it need not be the one being edited.
  const target = S.model.cells.find((c) => c.id === sw.cell) || rootCell();

  const kids = [
    field("What to compute", select(
      [["single", "one point, with the amounts above"], ["sweep", "sweep a parameter"]],
      sw.mode || "sweep",
      (v) => {
        sw.mode = v;
        sw.enabled = true;
        // Same exclusivity as the other two panels.
        S.model.sens.enabled = false;
        S.model.alv.enabled = false;
        push();
      }
    )),
    el("div", { class: "note" }, sw.mode === "single"
      ? "Homogenizes once with the fractions entered in Scales."
      : "Varies one lens over a range and draws the curve."),
  ];

  if (sweeping) {
    kids.push(
      el("div", { class: "grid3" },
        field("Variable", input(sw.variable, (v) => { sw.variable = v || "x"; push(); })),
        field("From", input(sw.start, (v) => { sw.start = +v; push(); })),
        field("To", input(sw.stop, (v) => { sw.stop = +v; push(); }))
      ),
      field("Points", input(sw.length, (v) => { sw.length = Math.max(2, +v | 0); push(); })),
      el("h3", {}, "What varies"),
      field("Lens", select(
        lensesFor(target).map((l) => [l.name, l.label]),
        sw.lens.kind, (v) => { sw.lens.kind = v; push(); }
      )),
      lensDoc(sw.lens.kind),
      ...lensFields(sw.lens, inner, target)
    );
  }

  // ── schemes: a list, so several land on one figure
  kids.push(el("h3", {}, "Schemes",
    el("button", {
      class: "small",
      onclick: () => { sw.schemes.push({ name: "MoriTanaka", options: {} }); push(); },
    }, "+")));
  sw.schemes.forEach((sc, i) => {
    kids.push(el("div", { class: "card" },
      el("header", {}, el("b", {}, sc.name),
        sw.schemes.length > 1
          ? el("button", { class: "small", onclick: () => { sw.schemes.splice(i, 1); push(); } }, "\u2212")
          : null),
      field("Scheme", select(
        schemesFor(target).map((x) => [x.name, x.name]),
        sc.name,
        (v) => {
          sc.name = v;
          // Options belong to the scheme that reads them; the server prunes
          // the rest, and clearing here keeps the form from flashing stale
          // inputs in between.
          sc.options = {};
          push();
        }
      )),
      schemeOptions(sc.name, sc.options || (sc.options = {}))
    ));
  });

  // ── outputs
  kids.push(el("h3", {}, "Outputs",
    el("button", {
      class: "small",
      onclick: () => { sw.outputs.push({ kind: "km", i: 1, j: 1 }); push(); },
    }, "+")));
  const needsIso = sw.outputs.some((o) => ISOTROPIC_ONLY.includes(o.kind));
  // A laminate of more than one layer is transversely isotropic about its
  // normal even with isotropic layers — that is what Backus says — so it falls
  // into the same trap as an unaveraged oriented inclusion.
  const anyFree = S.model.cells.some(
    (c) => c.phases.some((p) => p.symmetrize === "none")
      || (c.kind === "laminate" && (c.layers || []).length > 1)
  );
  if (needsIso && sw.projection === "none" && anyFree) {
    kids.push(el("div", { class: "note problem" },
      "k, μ, E and ν exist only for an isotropic tensor. This model has phases "
      + "with no orientation average, so the result need not be isotropic — "
      + "pick a reporting projection below, or plot Kelvin-Mandel components."));
  }
  sw.outputs.forEach((o, i) => {
    const row = [field("Quantity", select(OUTPUT_KINDS, o.kind, (v) => { o.kind = v; push(); }))];
    if (o.kind === "km" || o.kind === "comp") {
      row.push(
        field("i", input(o.i ?? 1, (v) => { o.i = Math.max(1, +v | 0); push(); })),
        field("j", input(o.j ?? 1, (v) => { o.j = Math.max(1, +v | 0); push(); }))
      );
    }
    kids.push(el("div", { class: "card" },
      el("header", {}, el("b", {}, outputLabel(o)),
        sw.outputs.length > 1
          ? el("button", { class: "small", onclick: () => { sw.outputs.splice(i, 1); push(); } }, "\u2212")
          : null),
      el("div", { class: row.length > 1 ? "grid3" : "grid2" }, ...row)
    ));
  });

  kids.push(
    el("div", { class: "grid2" },
      field("Property", input(sw.property, (v) => { sw.property = v.startsWith(":") ? v : ":" + v; push(); })),
      field("Report as", select(
        (S.catalog.projections || []).map((p) => [p.name, p.label]),
        sw.projection, (v) => { sw.projection = v; push(); }
      ))
    )
  );
  kids.push(
    field("", checkboxLabel(
      "clip negative values to zero", !!sw.clamp_zero,
      (v) => { sw.clamp_zero = v; push(); }
    )),
    el("div", { class: "note" },
      "A scheme pushed outside its range gives negative moduli — Dilute does, "
      + "well before f = 1 — and on a shared figure that one curve sets the "
      + "scale for every other. Clipping is a display choice; on a single "
      + "scheme, a negative modulus is the useful signal and is better left "
      + "showing.")
  );
  if (sweeping) {
    kids.push(field("Plot", checkboxLabel("draw a figure", sw.plot, (v) => { sw.plot = v; push(); })));
  }
  t.replaceChildren(...kids);
}

function outputLabel(o) {
  const i = o.i ?? 1, j = o.j ?? 1;
  return { k: "k", mu: "μ", E: "E", nu: "ν", trace3: "tr/3",
           km: `KM[${i},${j}]`, comp: `C[${i},${j}]` }[o.kind] || o.kind;
}

function lensDoc(kind) {
  const l = (S.catalog.lenses || []).find((x) => x.name === kind);
  return l && l.doc ? el("div", { class: "note" }, l.doc) : el("span");
}

/** The lenses that exist on this kind of cell.
 *
 * Not a nicety: `AmountParameter` *raises* on a laminate, pointing at
 * `ThicknessParameter`, and the two laminate lenses have no meaning on an RVE.
 * Offering them regardless would move the error from the form to the run.
 */
function lensesFor(c) {
  const k = isLaminate(c) ? "laminate" : "rve";
  return (S.catalog.lenses || []).filter(
    (l) => !l.cells || l.cells.includes(k)
  );
}

function lensFields(lens, inner, c = cell()) {
  const names = members(c).map((p) => [p.name, p.name]);
  const out = [];
  if (lens.kind === "nested") {
    out.push(
      el("div", { class: "grid2" },
        field(isLaminate(c) ? "Through layer" : "Through phase",
          select(names, lens.member || (names[0] && names[0][0]), (v) => { lens.member = v; push(); })),
        field("Key", input(lens.property, (v) => { lens.property = v.startsWith(":") ? v : ":" + v; push(); }))
      ),
      el("h3", {}, "Inside that scale"),
      field("Lens", select(
        lensesFor(c).filter((l) => l.name !== "nested").map((l) => [l.name, l.label]),
        inner.kind, (v) => { inner.kind = v; push(); }
      )),
      ...lensFields(inner, {}, c)
    );
    return out;
  }
  if (lens.kind === "thickness") {
    out.push(field("Layer", select(
      names, lens.phase || (names[0] && names[0][0]),
      (v) => { lens.phase = v; push(); }
    )));
  } else if (lens.kind === "interface_param") {
    const fields = ((S.catalog.lenses || []).find((l) => l.name === "interface_param")
      || {}).fields || [];
    out.push(el("div", { class: "grid2" },
      field("Interface (on top of layer)", input(lens.index, (v) => {
        lens.index = Math.max(1, +v | 0); push();
      })),
      field("Field", select(
        fields.map((f) => [f, f]), lens.field_name,
        (v) => { lens.field_name = v; push(); }
      ))
    ));
  } else if (lens.kind === "amount") {
    out.push(field("Phase", select(names, lens.phase || (names[0] && names[0][0]), (v) => { lens.phase = v; push(); })));
  } else if (lens.kind === "property") {
    out.push(el("div", { class: "grid3" },
      field("Phase", select(names, lens.phase, (v) => { lens.phase = v; push(); })),
      field("Key", input(lens.property, (v) => { lens.property = v; push(); })),
      field("Index", input(lens.index, (v) => { lens.index = +v | 0 || 1; push(); }))
    ));
  } else if (lens.kind === "geometry") {
    out.push(el("div", { class: "grid3" },
      field("Phase", select(names, lens.phase, (v) => { lens.phase = v; push(); })),
      field("Field", input(lens.field_name, (v) => { lens.field_name = v; push(); })),
      field("Index", input(lens.index, (v) => { lens.index = +v | 0 || 1; push(); }))
    ));
  } else if (lens.kind === "shape_param") {
    out.push(el("div", { class: "grid2" },
      field("Field", input(lens.field_name, (v) => { lens.field_name = v; push(); })),
      field("Index", input(lens.index, (v) => { lens.index = +v | 0 || 1; push(); }))
    ));
  }
  return out;
}

/* ── sensitivities ──────────────────────────────────────────────── */

/** Derivatives of the effective property, through the lenses.
 *
 * MeanFieldHomogenization's wrappers take the cell and the lens directly, so
 * this panel is the sweep's lens editor and the sweep's output picker put side
 * by side — there is no closure to write and no new vocabulary to learn.
 *
 * The point where the derivative is taken is the model itself: `get_param`
 * reads x₀ off the cell, so the amounts and moduli entered in Scales are the
 * point, and nothing is typed twice.
 */
function renderSens() {
  const s = S.model.sens;
  s.lenses = s.lenses && s.lenses.length ? s.lenses : [{ kind: "amount", phase: "", property: ":C", field_name: "semi_axes", index: 1, member: "", inner: null }];
  s.output = s.output || { kind: "k" };
  const t = $("#tab-sens");
  const target = S.model.cells.find((c) => c.id === s.cell) || rootCell();
  const many = s.kind !== "derivative";

  const kids = [
    field("Compute", checkboxLabel(
      "differentiate the effective property", s.enabled,
      (v) => {
        s.enabled = v;
        // The three result modes are exclusive: one `Result` section is
        // emitted, and two of them would fight over the same `cell`.
        if (v) { S.model.alv.enabled = false; S.model.sweep.enabled = false; }
        push();
      }
    )),
    el("div", { class: "note" },
      "ForwardDiff through the whole scheme — and through a nested scale too, "
      + "the `nested` lens composing across the seam in one pass."),
  ];

  if (!s.enabled) { t.replaceChildren(...kids); return; }

  kids.push(
    field("Kind", select(
      (S.catalog.sensitivities || []).map((x) => [x.name, x.label]),
      s.kind,
      (v) => {
        s.kind = v;
        if (v === "derivative") s.lenses = s.lenses.slice(0, 1);
        push();
      }
    )),
    sensDoc(s.kind),
    el("div", { class: "grid2" },
      field("Scale", select(
        S.model.cells.map((c) => [c.id, c.name]),
        s.cell || (target && target.id),
        (v) => { s.cell = v; push(); }
      )),
      field("Property", input(s.property, (v) => {
        s.property = v.startsWith(":") ? v : ":" + v; push();
      }))
    ),
    field("Scheme", select(
      schemesFor(target).map((x) => [x.name, x.name]),
      s.scheme,
      (v) => { s.scheme = v; s.scheme_options = {}; push(); }
    )),
    schemeOptions(s.scheme, s.scheme_options || (s.scheme_options = {}))
  );

  kids.push(el("h3", {}, many ? "Parameters" : "Parameter",
    many
      ? el("button", {
          class: "small",
          onclick: () => {
            s.lenses.push({ kind: "amount", phase: "", property: ":C", field_name: "semi_axes", index: 1, member: "", inner: null });
            push();
          },
        }, "+")
      : null));

  s.lenses.forEach((l, i) => {
    const inner = l.inner || (l.inner = { kind: "amount", phase: "", property: ":C", field_name: "semi_axes", index: 1, member: "", inner: null });
    kids.push(el("div", { class: "card" },
      el("header", {}, el("b", {}, `p${i + 1}`),
        s.lenses.length > 1
          ? el("button", { class: "small", onclick: () => { s.lenses.splice(i, 1); push(); } }, "−")
          : null),
      field("Lens", select(
        lensesFor(target).map((x) => [x.name, x.label]),
        l.kind, (v) => { l.kind = v; push(); }
      )),
      lensDoc(l.kind),
      ...lensFields(l, inner, target)
    ));
  });

  // A jacobian differentiates the whole tensor, so there is nothing to extract
  // — and that is also the way out when the result is not isotropic.
  if (s.kind !== "jacobian") {
    const row = [field("Quantity", select(OUTPUT_KINDS, s.output.kind, (v) => {
      s.output.kind = v; push();
    }))];
    if (s.output.kind === "km" || s.output.kind === "comp") {
      row.push(
        field("i", input(s.output.i ?? 1, (v) => { s.output.i = Math.max(1, +v | 0); push(); })),
        field("j", input(s.output.j ?? 1, (v) => { s.output.j = Math.max(1, +v | 0); push(); }))
      );
    }
    row.push(field("Report as", select(
      (S.catalog.projections || []).map((p) => [p.name, p.label]),
      s.projection || "iso", (v) => { s.projection = v; push(); }
    )));
    kids.push(el("h3", {}, "Differentiated quantity"),
      el("div", { class: row.length > 2 ? "grid3" : "grid2" }, ...row),
      el("div", { class: "note" },
        "The projection sits inside the differentiated function, so what comes "
        + "out is the derivative of the reported quantity."));
  } else {
    kids.push(el("div", { class: "note" },
      "The whole effective tensor, flattened — no scalar is extracted, so no "
      + "isotropy is required."));
  }

  kids.push(el("div", { class: "note" },
    "The answer is a table, in the output panel: a gradient is not a curve."));
  t.replaceChildren(...kids);
}

function sensDoc(kind) {
  const x = (S.catalog.sensitivities || []).find((s) => s.name === kind);
  return x && x.doc ? el("div", { class: "note" }, x.doc) : el("span");
}

/* ── viscoelasticity ────────────────────────────────────────────── */

function renderAlv() {
  const a = S.model.alv;
  a.component = a.component && a.component.length === 2 ? a.component : [1, 1];
  const blocked = S.model.cells.some((c) =>
    c.phases.some((p) => (p.properties || []).some((x) => x.source === "cell")));
  const viscoPhases = [];
  for (const c of S.model.cells) {
    for (const ph of c.phases) {
      for (const pr of ph.properties || []) {
        const f = propForm(pr.form);
        if (f && f.visco) viscoPhases.push(`${c.name}.${ph.name}${pr.key}`);
      }
    }
  }

  $("#tab-alv").replaceChildren(
    blocked
      ? el("div", { class: "note problem" },
          "Ageing viscoelasticity cannot be combined with a nested scale: "
          + "MeanFieldHomogenization cannot re-express a homogenized inner result as a "
          + "ViscoLaw. Remove the seam first.")
      : el("span"),
    el("div", { class: "note" },
      "Give a phase a viscoelastic law in Scales → Properties → Parametrization "
      + "(Maxwell, Kelvin chain, elastic, or a custom J(t, t′)). "
      + (viscoPhases.length
          ? "Found on: " + viscoPhases.join(", ") + "."
          : "No phase carries one yet, so this run has nothing to age.")),
    field("", checkboxLabel("Ageing linear viscoelastic run", a.enabled, (v) => {
      a.enabled = v;
      // One `Result` section is emitted; two run modes would fight over it.
      if (v) S.model.sens.enabled = false;
      push();
    })),
    el("div", { class: "grid3" },
      field(a.log_time ? "log₁₀ t from" : "t from", input(a.t_start, (v) => { a.t_start = +v; push(); })),
      field(a.log_time ? "log₁₀ t to" : "t to", input(a.t_stop, (v) => { a.t_stop = +v; push(); })),
      field("Steps", input(a.length, (v) => { a.length = Math.max(2, +v | 0); push(); }))
    ),
    field("", checkboxLabel("logarithmic time", a.log_time, (v) => { a.log_time = v; push(); })),
    el("div", { class: "grid2" },
      field("Property", input(a.property, (v) => { a.property = v.startsWith(":") ? v : ":" + v; push(); })),
      field("Scale", select(
        S.model.cells.map((c) => [c.id, c.name]),
        a.cell || (S.model.cells[0] && S.model.cells[0].id),
        (v) => { a.cell = v; push(); }
      ))
    ),
    el("h3", {}, "Curve"),
    el("div", { class: "note" },
      "The effective relaxation operator is inverted to a creep operator; the "
      + "curve is the response to a unit step, read on one Kelvin-Mandel "
      + "component. (1, 1) is uniaxial."),
    el("div", { class: "grid2" },
      field("component i", input(a.component[0], (v) => { a.component[0] = Math.max(1, +v | 0); push(); })),
      field("component j", input(a.component[1], (v) => { a.component[1] = Math.max(1, +v | 0); push(); }))
    ),
    field("Plot", checkboxLabel("draw a figure", a.plot !== false, (v) => { a.plot = v; push(); })),
    el("div", { class: "note" },
      "The schemes come from the Sweep tab, so several can share the figure.")
  );
}

/* ── constants and kept blocks ──────────────────────────────────── */

function renderParams() {
  const t = $("#tab-params");
  t.replaceChildren(
    el("h3", {}, "Constants",
      el("button", {
        class: "small",
        onclick: () => { S.model.params.push({ name: "x", value: "1.0", comment: "", origin: null, edited: true }); push(); },
      }, "+")),
    ...S.model.params.map((p, i) =>
      el("div", { class: "card" },
        el("header", {}, el("b", {}, p.name),
          p.origin && !p.edited ? el("span", { class: "tag" }, "as written") : null,
          el("button", { class: "small", onclick: () => { S.model.params.splice(i, 1); push(); } }, "−")),
        el("div", { class: "grid2" },
          field("Name", input(p.name, (v) => { p.name = v; p.edited = true; push(); })),
          field("Value", input(p.value, (v) => { p.value = v; p.edited = true; push(); }))
        ))
    ),
    S.model.params.length ? el("div", { class: "note" },
      "A constant read from a file keeps its original text until you edit it, "
      + "so multi-line layout is not collapsed.") : el("span")
  );
}

function renderKept() {
  const t = $("#tab-kept");
  const kept = S.model.opaque || [];
  const rep = S.keptReport;
  t.replaceChildren(
    el("div", { class: "note" },
      "Code MFH Studio did not recognize. It is written back unchanged — "
      + "editing the model never rewrites it."),
    rep && rep.exact
      ? el("div", { class: "note" }, "This file was written by the studio and reopened exactly.")
      : el("span"),
    ...kept.map((o) =>
      el("div", {},
        o.note ? el("div", { class: "muted" }, o.note) : null,
        el("div", { class: "kept" }, o.source))
    ),
    kept.length ? null : el("div", { class: "muted" }, "Nothing kept.")
  );
}

/* ── small widgets ──────────────────────────────────────────────── */

/* Every panel is rebuilt from scratch on each change, which is simple and
 * keeps one source of truth — but it also destroys whatever the user was
 * typing in. Each control therefore carries a stable key derived from where
 * it sits in the model, and `render()` puts the caret back afterwards. */
let _keySeq = 0;
function nextKey(label) {
  return `${label || "f"}#${_keySeq++}`;
}

function field(label, node) {
  if (node && node.dataset && !node.dataset.k) node.dataset.k = nextKey(label);
  return el("div", { class: "field" }, label ? el("label", {}, label) : null, node);
}
function input(value, on) {
  const n = el("input", { type: "text", spellcheck: "false" });
  n.value = value ?? "";
  n.addEventListener("change", () => on(n.value));
  return n;
}
function select(pairs, value, on) {
  const n = el("select", {}, ...pairs.map(([v, t]) =>
    el("option", { value: v, selected: String(v) === String(value) }, t)));
  n.addEventListener("change", () => on(n.value));
  return n;
}
function checkbox(value, on) {
  const n = el("input", { type: "checkbox" });
  n.checked = !!value;
  n.addEventListener("change", () => on(n.checked));
  return n;
}

/** Where the caret was, so a rebuild does not throw the user out of a field. */
function captureFocus() {
  const a = document.activeElement;
  if (!a || !a.dataset || !a.dataset.k) return null;
  const f = { k: a.dataset.k };
  if (a.selectionStart != null) {
    f.start = a.selectionStart;
    f.end = a.selectionEnd;
  }
  return f;
}

function restoreFocus(f) {
  if (!f) return;
  const n = document.querySelector(`[data-k="${CSS.escape(f.k)}"]`);
  if (!n) return;
  n.focus();
  if (f.start != null && n.setSelectionRange) {
    try { n.setSelectionRange(f.start, f.end); } catch { /* not a text input */ }
  }
}
function checkboxLabel(text, value, on) {
  return el("label", { class: "inline" }, checkbox(value, on), " " + text);
}

/* ── 3-D ────────────────────────────────────────────────────────── */

let lastExpr = null;
async function draw3d() {
  const c = cell();
  // A laminate's layers have no shape of their own: the geometry of the cell
  // *is* the stack, so what gets drawn is the cell. The expression for it is
  // built by the server from the same code generator that writes the script —
  // a second spelling here would let the picture drift from the model.
  const lam = isLaminate(c);
  const body = lam
    ? { cell: c.id, cutaway: $("#cutaway").checked }
    : { expr: geomExpr(phase()), cutaway: $("#cutaway").checked };
  const shown = lam
    ? `Laminate — ${(c.layers || []).length} layer(s)`
    : body.expr;
  $("#shape-label").textContent = shown || "";
  if (lam && (c.params || []).length) {
    // A builder taking parameters has no single stack to draw.
    Plotly.purge("view3d");
    $("#shape-label").textContent = shown + " — parametrized, nothing to draw";
    lastExpr = null;
    return;
  }
  if (!lam && !body.expr) { Plotly.purge("view3d"); lastExpr = null; return; }
  if (!S.catalogIntrospected) {
    // The sidecar draws the shapes; without it, say so once rather than
    // firing a request per edit that can only fail.
    Plotly.purge("view3d");
    $("#shape-label").textContent = shown + " — 3-D needs Julia";
    return;
  }
  const key = JSON.stringify([body, lam ? JSON.stringify(c) : null]);
  if (key === lastExpr) return;
  lastExpr = key;
  try {
    const sc = await api("/api/traces", body);
    Plotly.react("view3d", sc.data, sc.layout, { displayModeBar: false, responsive: true });
  } catch (e) {
    // A shape the sidecar cannot build is a modeling error worth showing.
    $("#shape-label").textContent = shown + " — " + e.message.split("\n")[0];
  }
}

/* ── run ────────────────────────────────────────────────────────── */

async function run() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "Running…";
  $("#stdout").textContent = "";
  try {
    const r = await api("/api/run", {});
    $("#stdout").textContent = (r.stdout || "") + (r.error ? "\n" + r.error : "");
    if (r.results) plotResults(r.results);
    if (r.ok) toast("Ran cleanly.");
    else toast(r.timeout ? "Timed out." : "The script raised — see the output.", true);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run";
  }
}

function plotResults(res) {
  const x = res.x || [];
  const traces = Object.entries(res)
    .filter(([k]) => k !== "x" && k !== "xlabel")
    .map(([k, y]) => ({ x, y, name: k, type: "scatter", mode: "lines+markers" }));
  if (!traces.length) return;
  Plotly.react("plot", traces, {
    margin: { l: 48, r: 10, t: 10, b: 38 },
    xaxis: { title: res.xlabel || "x" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    legend: { orientation: "h" },
  }, { displayModeBar: false, responsive: true });
}

/* ── files ──────────────────────────────────────────────────────── */

async function openFile() {
  const path = await Picker.pick("open", S.path || "");
  if (!path) return;

  // An Echoes script is opened by translating it. The extension already says
  // which of the two is happening, so there is no separate convert button —
  // only the one extra question of where the Julia should go.
  let output = null;
  if (/\.py$/i.test(path)) {
    const suggested = path.replace(/\.py$/i, ".jl");
    output = await Picker.pick(
      "save",
      suggested.replace(/[^\\/]*$/, ""),
      suggested.replace(/^.*[\\/]/, "")
    );
    if (!output) return;
  }

  try {
    const st = await api("/api/open", output ? { path, output } : { path });
    S.keptReport = st.read_report || null;
    apply(st);
    if (st.conversion) {
      showConversion(st.conversion);
    } else {
      const r = st.read_report || {};
      toast(r.exact
        ? "Reopened from the model embedded in the file."
        : `Read: ${r.recognized || 0} construct(s) understood, ${r.opaque || 0} kept as written.`);
    }
  } catch (e) {
    toast(e.message, true);
  }
}

/** What the translation did, and what it refused. */
function showConversion(c) {
  toast(`${c.source.replace(/^.*[\\/]/, "")} → ${c.output.replace(/^.*[\\/]/, "")}: ${c.summary}`,
        c.blocking > 0);
  const banner = $("#banner");
  if (!c.findings.length) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.replaceChildren(
    el("b", {}, `Translated from ${c.source.replace(/^.*[\\/]/, "")} — `),
    el("span", {}, c.summary),
    el("pre", {}, c.findings
      .slice(0, 10)
      .map((f) => `line ${f.line}  [${f.severity}]  ${f.reason}`)
      .join("\n"))
  );
}

async function saveFile(askPath) {
  let path = S.path;
  if (askPath || !path) {
    const suggested = (S.model && S.model.title ? S.model.title : "model") + ".jl";
    path = await Picker.pick("save", S.path || "", suggested);
    if (!path) return;
  }
  try {
    const r = await api("/api/save", { path });
    S.path = r.path;
    setPathLabel(r.path);
    toast(`Saved ${r.bytes} bytes to ${r.path}`);
  } catch (e) {
    toast(e.message, true);
  }
}

function setPathLabel(p) {
  const el = $("#path-label");
  el.textContent = p || "untitled";
  el.title = p || "No file yet";
}

/* ── sidecar status ─────────────────────────────────────────────── */

async function pollSidecar() {
  let s;
  try {
    s = await api("/api/sidecar");
  } catch {
    return; // the server will answer eventually
  }
  const b = $("#sidecar");
  const err = s.error || s.catalog_error;
  if (s.ready && s.introspected) { b.textContent = "Julia ready"; b.className = "badge ok"; }
  else if (s.running) { b.textContent = "loading MeanFieldHomogenization…"; b.className = "badge"; }
  else if (err) { b.textContent = "Julia unavailable"; b.className = "badge bad"; b.title = err; }
  else { b.textContent = "starting…"; b.className = "badge"; }

  // Say plainly what is off and what still works. A dead-looking interface
  // with a stack trace only in the console is the worst of both.
  const banner = $("#banner");
  if (err && !s.ready) {
    banner.hidden = false;
    banner.replaceChildren(
      el("b", {}, "Julia is not available — "),
      el("span", {}, "you can still build and save a script; the 3-D view, "
        + "reading a script back, and Run are off."),
      el("pre", {}, String(err).split("\n").slice(0, 8).join("\n"))
    );
  } else {
    banner.hidden = true;
  }

  // Upgrade the catalog as soon as the sidecar can answer.
  if (s.ready && S.catalog && !S.catalogIntrospected) await loadCatalog();
}

/* ── boot ───────────────────────────────────────────────────────── */

async function loadCatalog() {
  const cat = await api("/api/catalog");
  const was = S.catalogIntrospected;
  S.catalog = cat;
  S.catalogIntrospected = !!cat.introspected;
  // Re-render once the live scheme options arrive, so their inputs appear.
  if (S.model && S.catalogIntrospected && !was) render();
}

async function boot() {
  try {
    // The catalog answers with or without Julia; the model always does. The
    // interface must come up either way — otherwise every control throws on
    // a null model and the whole thing looks broken.
    await loadCatalog();
    apply(await api("/api/state"));
  } catch (e) {
    toast("Could not reach the studio server: " + e.message, true);
  }
}

function wire() {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.remove("on"));
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      $("#tab-" + b.dataset.tab).classList.add("on");
    }));

  $("#cell-select").addEventListener("change", (e) => {
    S.cellId = e.target.value; S.phaseIdx = 0; render(); draw3d();
  });
  $("#cell-add").addEventListener("click", () => {
    const n = S.model.cells.length + 1;
    // Stagger new boxes so they never land on top of each other; the user can
    // drag them wherever afterwards and the position is remembered.
    S.model.cells.push({
      id: "", name: `scale${n}`, matrix_name: "MATRIX", params: [],
      builder_name: null, rve_options: {},
      ui: { x: 40 + ((n - 1) % 3) * 210, y: 30 + Math.floor((n - 1) / 3) * 150 },
      phases: [{
        name: "MATRIX", is_matrix: true, amount_kind: "fraction", amount: 0, symmetrize: "none",
        geometry: { kind: "spheroid", args: { omega: 1.0 }, euler_angles: [], layers: [] },
        properties: [{ key: ":C", source: "builder", builder: "iso_stiffness", form: "iso_kmu", args: { k: 10, mu: 5 }, scheme_options: {} }],
      }],
    });
    push();
  });
  $("#cell-del").addEventListener("click", () => {
    const i = S.model.cells.findIndex((c) => c.id === S.cellId);
    if (i >= 0 && S.model.cells.length > 1) { S.model.cells.splice(i, 1); S.cellId = null; push(); }
  });
  $("#cell-name").addEventListener("change", (e) => { cell().name = e.target.value; push(); });
  $("#cell-kind").addEventListener("change", (e) => {
    const c = cell();
    c.kind = e.target.value;
    // Switching kind seeds the other shape rather than leaving it empty: an
    // empty laminate is a validation error, and starting on one reads as a
    // fault rather than as a choice. Whatever was there is kept, so switching
    // back and forth loses nothing.
    if (c.kind === "laminate" && !(c.layers || []).length) {
      c.layers = [
        { name: "A", amount_kind: "fraction", amount: 0.3,
          interface: { kind: "PerfectInterface", args: {} },
          properties: [{ key: ":C", source: "builder", builder: "iso_stiffness",
                         form: "iso_kmu", args: { k: 2.0, mu: 0.8 }, scheme_options: {} }] },
        { name: "B", amount_kind: "fraction", amount: 0.7,
          interface: { kind: "PerfectInterface", args: {} },
          properties: [{ key: ":C", source: "builder", builder: "iso_stiffness",
                         form: "iso_kmu", args: { k: 0.5, mu: 0.2 }, scheme_options: {} }] },
      ];
    }
    S.phaseIdx = 0;
    push();
  });
  $("#cell-matrix").addEventListener("change", (e) => { cell().matrix_name = e.target.value; push(); });
  $("#phase-add").addEventListener("click", () => {
    cell().phases.push({
      name: "PHASE" + cell().phases.length, is_matrix: false,
      amount_kind: "fraction", amount: 0.1, symmetrize: "none",
      geometry: { kind: "spheroid", args: { omega: 1.0 }, euler_angles: [], layers: [] },
      properties: [{ key: ":C", source: "builder", builder: "iso_stiffness", form: "iso_kmu", args: { k: 10, mu: 5 }, scheme_options: {} }],
    });
    S.phaseIdx = cell().phases.length - 1;
    push();
  });

  $("#cutaway").addEventListener("change", () => { lastExpr = null; draw3d(); });
  $("#run").addEventListener("click", run);
  $("#open").addEventListener("click", openFile);
  $("#save").addEventListener("click", () => saveFile(false));
  $("#saveas").addEventListener("click", () => saveFile(true));
}

wire();
Split.init();
boot();
pollSidecar();
setInterval(pollSidecar, 2000);
