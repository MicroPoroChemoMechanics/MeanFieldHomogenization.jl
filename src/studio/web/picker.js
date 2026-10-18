/* The file picker.
 *
 * `<input type="file">` hands back the file's *content* but never its path,
 * and a path is exactly what saving back to the same file needs. So the
 * picker lists the server's filesystem instead — which also means it keeps
 * working when the studio runs on a remote machine, where a native dialog
 * would show the wrong computer's disks.
 */

const Picker = (() => {
  let state = { path: null, selected: null, mode: "open", resolve: null };

  const q = (s) => document.querySelector(s);

  function humanSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " kB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  async function list(path) {
    // a save with nothing bound starts in the session's folder, not the library
    const args = [path ? "path=" + encodeURIComponent(path) : "", state.mode === "save" ? "for=save" : ""];
    const url = "/api/browse" + (args.some(Boolean) ? "?" + args.filter(Boolean).join("&") : "");
    const r = await studioFetch(url);
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    return j;
  }

  function row(icon, label, onActivate, extra) {
    const d = document.createElement("div");
    d.className = "row";
    const i = document.createElement("span");
    i.className = "ico";
    i.textContent = icon;
    const t = document.createElement("span");
    t.textContent = label;
    d.append(i, t);
    if (extra) {
      const s = document.createElement("span");
      s.className = "size";
      s.textContent = extra;
      d.append(s);
    }
    d.addEventListener("click", () => onActivate(d));
    d.addEventListener("dblclick", () => onActivate(d, true));
    return d;
  }

  async function show(dir) {
    let data;
    try {
      data = await list(dir);
    } catch (e) {
      toast(e.message, true);
      return;
    }
    state.path = data.path;
    state.selected = null;

    q("#picker-crumb").textContent = data.path;

    q("#picker-places").replaceChildren(
      ...data.places.map((p) => {
        const b = document.createElement("button");
        b.className = "small";
        b.textContent = p.label;
        b.title = p.path;
        b.addEventListener("click", () => show(p.path));
        return b;
      })
    );

    const list_ = q("#picker-list");
    const rows = [];
    if (data.parent) {
      rows.push(row("↑", "..", () => show(data.parent)));
    }
    for (const d of data.dirs) rows.push(row("📁", d.name, () => show(d.path)));
    for (const f of data.files) {
      rows.push(
        row(f.echoes ? "🐍" : "📄", f.name, (el, dbl) => {
          list_.querySelectorAll(".row.on").forEach((n) => n.classList.remove("on"));
          el.classList.add("on");
          state.selected = f.path;
          q("#picker-name").value = f.name;
          if (dbl) accept();
        }, f.echoes ? "Echoes · " + humanSize(f.size) : humanSize(f.size))
      );
    }
    if (!rows.length) {
      const e = document.createElement("div");
      e.className = "empty";
      e.textContent = state.mode === "save"
        ? "Nothing here yet — type a name below."
        : "No Julia or Echoes scripts here.";
      rows.push(e);
    }
    list_.replaceChildren(...rows);
  }

  function accept() {
    const name = q("#picker-name").value.trim();
    let path = state.selected;
    if (state.mode === "save" || !path) {
      if (!name) {
        toast("Give a file name.", true);
        return;
      }
      const sep = state.path.includes("\\") ? "\\" : "/";
      path = name.includes("/") || name.includes("\\")
        ? name
        : state.path.replace(/[\\/]+$/, "") + sep + name;
    }
    close();
    state.resolve && state.resolve(path);
  }

  function close() {
    q("#picker").hidden = true;
    document.removeEventListener("keydown", onKey);
  }

  function onKey(e) {
    if (e.key === "Escape") { close(); state.resolve && state.resolve(null); }
    else if (e.key === "Enter") accept();
  }

  /** Open the dialog; resolves with a path, or null if dismissed. */
  function pick(mode, startDir, suggestedName) {
    state.mode = mode;
    q("#picker-title").textContent = mode === "save"
      ? "Save the script as"
      : "Open a script  (.jl to edit, .py to translate from Echoes)";
    q("#picker-ok").textContent = mode === "save" ? "Save" : "Open";
    q("#picker-name").value = suggestedName || "";
    q("#picker").hidden = false;
    document.addEventListener("keydown", onKey);
    show(startDir || "");
    return new Promise((resolve) => { state.resolve = resolve; });
  }

  document.addEventListener("DOMContentLoaded", () => {
    q("#picker-close").addEventListener("click", () => { close(); state.resolve && state.resolve(null); });
    q("#picker-ok").addEventListener("click", accept);
    q("#picker").addEventListener("click", (e) => {
      if (e.target.id === "picker") { close(); state.resolve && state.resolve(null); }
    });
  });

  return { pick };
})();
