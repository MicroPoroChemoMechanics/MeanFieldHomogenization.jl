/* The page's words, in English or in French.
 *
 * The English text is the key: `t("Add a layer")` returns its French when the
 * page is in French and `i18n/fr.json` holds it, the English otherwise. The
 * code keeps its sentences readable where they are used, and a sentence
 * without a translation shows in English rather than as a key. `{name}` in a
 * text is replaced from `vars`. The script, its comments and the messages from
 * Julia stay in English: they are what a run prints and a file keeps.
 *
 * Static elements of index.html carry `data-i18n` (their text),
 * `data-i18n-title` (their tooltip) or `data-i18n-placeholder`; the English is
 * read from the element the first time and kept on it.
 *
 * test/Studio/test_i18n.jl extracts every `t("…")` of the page, every marked
 * element and every label and note of the catalog, and checks that fr.json
 * translates each and holds nothing else.
 */

const I18n = (() => {
  let lang = "en";
  let french = null;

  /** The text of `english` in the page's language, `{name}` replaced from `vars`. */
  function t(english, vars) {
    let s = lang === "fr" && french && french[english] !== undefined ? french[english] : english;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
    return s;
  }

  const squash = (s) => s.replace(/\s+/g, " ").trim();

  function translatePage() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((e) => {
      if (e.dataset.en === undefined) e.dataset.en = squash(e.textContent);
      e.textContent = t(e.dataset.en);
    });
    document.querySelectorAll("[data-i18n-title]").forEach((e) => {
      if (e.dataset.enTitle === undefined) e.dataset.enTitle = squash(e.title);
      e.title = t(e.dataset.enTitle);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((e) => {
      if (e.dataset.enPlaceholder === undefined) e.dataset.enPlaceholder = squash(e.placeholder);
      e.placeholder = t(e.dataset.enPlaceholder);
    });
  }

  async function setLanguage(l) {
    if (l === "fr" && french === null) {
      try {
        const r = await studioFetch("/i18n/fr.json");
        french = r.ok ? await r.json() : {};
      } catch (e) {
        french = {};
      }
    }
    lang = l === "fr" ? "fr" : "en";
    try { localStorage.setItem("mfhstudio.lang", lang); } catch (e) { /* not remembered */ }
    translatePage();
  }

  function savedLanguage() {
    try { return localStorage.getItem("mfhstudio.lang") || "en"; } catch (e) { return "en"; }
  }

  return { t, setLanguage, savedLanguage, language: () => lang, translatePage };
})();

const t = I18n.t;
