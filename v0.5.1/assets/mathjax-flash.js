// Documenter pulls MathJax from a CDN through require.js, so the raw LaTeX of
// every formula stays on screen until that ~1.3 MB bundle has been fetched and
// run. On a theory page carrying thirty equations, behind a slow link, that
// intermediate state is what the reader sees first. Two things happen here.
//
//   1. A preload hint starts the MathJax download from the <head>, in parallel
//      with require.js, instead of waiting for require.js to resolve.
//   2. The document is marked as pending so `custom.css` can hide the untypeset
//      math. The mark is dropped as soon as MathJax reports it is done -- or
//      after a timeout, so an unreachable CDN degrades to visible LaTeX rather
//      than to a blank page.
//
// The URL below must stay in step with the one emitted by `assets/documenter.js`
// (Documenter hard-codes it); a mismatch makes the preload a wasted request
// instead of a warm cache hit. Recheck it after a Documenter upgrade.

(function () {
  "use strict";

  var MATHJAX_URL =
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.js";
  var FALLBACK_DELAY_MS = 5000;

  var link = document.createElement("link");
  link.rel = "preload";
  link.as = "script";
  link.crossOrigin = "anonymous";
  link.href = MATHJAX_URL;
  document.head.appendChild(link);

  var root = document.documentElement;
  root.classList.add("mathjax-pending");

  var poll = null;

  function reveal() {
    if (poll !== null) {
      clearInterval(poll);
      poll = null;
    }
    root.classList.remove("mathjax-pending");
  }

  // Display math already carries `.math-container`. Inline math is emitted as a
  // bare `<span>$...$</span>`, with no class of its own, so it has to be tagged
  // before it can be hidden.
  function markInlineMath() {
    var spans = document.querySelectorAll(".content span");
    for (var i = 0; i < spans.length; i++) {
      var span = spans[i];
      var text = span.textContent;
      if (
        span.children.length === 0 &&
        text.length > 1 &&
        text.charAt(0) === "$" &&
        text.charAt(text.length - 1) === "$"
      ) {
        span.classList.add("raw-math");
      }
    }
  }

  function attachToMathJax() {
    var startup = window.MathJax && window.MathJax.startup;
    if (!startup || !startup.promise) {
      return false;
    }
    startup.promise.then(reveal, reveal);
    return true;
  }

  document.addEventListener("DOMContentLoaded", function () {
    markInlineMath();
    if (!attachToMathJax()) {
      // `window.MathJax` is only defined once require.js has run
      // `documenter.js`, which may well be after this listener fires.
      poll = setInterval(function () {
        if (attachToMathJax()) {
          clearInterval(poll);
          poll = null;
        }
      }, 50);
    }
  });

  setTimeout(reveal, FALLBACK_DELAY_MS);
})();
