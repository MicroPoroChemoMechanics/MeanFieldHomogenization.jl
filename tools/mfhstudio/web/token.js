/* The session token, and the calls that carry it.
 *
 * The token arrives in the address the terminal prints. It is kept for the tab
 * and removed from the address bar, so that a screenshot or a shared link does
 * not carry it; every /api call sends it back in a header, which the server
 * requires. The theme the user chose is applied here too, before the page
 * draws, so that it never flashes in the other one.
 */

const STUDIO_TOKEN = (() => {
  const params = new URLSearchParams(location.search);
  let token = params.get("token");
  try {
    if (token) sessionStorage.setItem("mfhstudio.token", token);
    else token = sessionStorage.getItem("mfhstudio.token");
  } catch (e) {
    /* storage unavailable: the token lives as long as the page */
  }
  if (params.has("token")) history.replaceState(null, "", location.pathname);
  return token || "";
})();

/** `fetch` with the session token. */
function studioFetch(path, opt = {}) {
  const headers = Object.assign({}, opt.headers || {}, { "X-Studio-Token": STUDIO_TOKEN });
  return fetch(path, Object.assign({}, opt, { headers }));
}

/* ── theme: the system's, light or dark ─────────────────────────── */

const THEMES = ["auto", "light", "dark"];

function savedTheme() {
  try { return localStorage.getItem("mfhstudio.theme") || "auto"; } catch (e) { return "auto"; }
}

function applyTheme(theme) {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("mfhstudio.theme", theme); } catch (e) { /* not remembered */ }
  const b = document.getElementById("theme");
  if (b) b.textContent = theme === "dark" ? "☾" : theme === "light" ? "☀" : "◐";
}

applyTheme(savedTheme());
