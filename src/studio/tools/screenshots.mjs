#!/usr/bin/env node
// The screenshots of MFH Studio in the documentation and the README, taken from
// a running studio so that they can be taken again whenever the page changes.
//
//   julia -e 'using MeanFieldHomogenization; mfhstudio(no_browser = true)'   # in another terminal
//   node screenshots.mjs --url 'http://127.0.0.1:8765/?token=…' [--out ../../../docs/src/assets/mfhstudio]
//
// Each picture is taken in the light and in the dark theme, from a studio just
// started (the first one is its default model):
//   overview_{light,dark}.png      the porous benchmark, run;
//   layered_{light,dark}.png       a coated inclusion, its shells cut away;
//   multiscale_{light,dark}.png    two scales and their seam, run;
//   viscoelastic_{light,dark}.png  a creeping mortar under two schemes;
//   readme_{light,dark}.png        the two scales again, in a smaller window.
// The pictures are taken at twice the pixel density and then reduced to 256
// colors (a third of the size, no visible loss), for instance with Python's PIL:
//   Image.open(f).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(f, optimize=True)
//
// Needs Node and playwright-core with a Chromium build. playwright-core is found
// through $PLAYWRIGHT_CORE, then in the global @playwright/mcp install, then by
// plain resolution.

import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
  (a.startsWith("--") ? acc.concat([[a.slice(2), all[i + 1]]]) : acc), []));
if (!args.url) throw new Error("give the address of a running studio: --url 'http://127.0.0.1:8765/?token=…'");
const out = resolve(args.out || join(here, "..", "..", "..", "docs", "src", "assets", "mfhstudio"));
const VIEW = { width: 1680, height: 980 };
const SMALL = { width: 1440, height: 880 };

function playwright() {
  const tries = [
    process.env.PLAYWRIGHT_CORE,
    (() => { try { return join(execSync("npm root -g").toString().trim(), "@playwright/mcp/node_modules/playwright-core"); } catch { return null; } })(),
    "playwright-core",
  ].filter(Boolean);
  const require = createRequire(import.meta.url);
  for (const t of tries) {
    try { return require(t); } catch { /* next */ }
  }
  throw new Error("playwright-core not found; set PLAYWRIGHT_CORE to its directory");
}

// The page's own call to the server, with the token it keeps.
const api = (page, path, body) => page.evaluate(([p, b]) => api(p, b), [path, body]);

async function ready(page, minutes) {
  await page.waitForFunction(() => document.querySelector("#status").classList.contains("ok"), null, { timeout: minutes * 60000 });
}

// Open an example of the library: the server lists the library first.
async function openExample(page, name) {
  const listing = await api(page, "/api/browse");
  const file = listing.files.find((f) => f.name === name);
  if (!file) throw new Error(`no example ${name} in ${listing.path}`);
  await api(page, "/api/open", { path: file.path });
}

async function run(page, minutes) {
  await page.click("#run");
  await page.waitForFunction(() => !document.querySelector("#run").disabled, null, { timeout: minutes * 60000 });
  await page.waitForTimeout(1500);   // the plot settles
}

async function shoot(page, id, prepare, view = VIEW) {
  await page.setViewportSize(view);
  for (const theme of ["light", "dark"]) {
    // the page reads its theme when it loads, the plot when it is drawn: a
    // reload in the theme, then the calculation again
    await page.evaluate((t) => { try { localStorage.setItem("mfhstudio.theme", t); } catch (e) { /* none */ } }, theme);
    await page.reload();
    await ready(page, 2);
    await page.waitForTimeout(300);
    await prepare(page);
    await page.waitForTimeout(800);
    // no toast of the run over the picture
    await page.evaluate(() => document.querySelector("#toast").replaceChildren());
    const file = join(out, `${id}_${theme}.png`);
    await page.screenshot({ path: file });
    console.log("wrote", file);
  }
}

const tab = (page, t) => page.click(`.tabs button[data-tab=${t}]`);

const { chromium } = playwright();
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: VIEW, deviceScaleFactor: 2 });
try {
  await page.goto(args.url);
  await page.evaluate(() => { try { localStorage.setItem("mfhstudio.lang", "en"); } catch (e) { /* none */ } });
  await page.reload();
  await ready(page, 15);
  // the first picture is the default model of a studio just started, which
  // nothing can bring back once an example is opened
  const start = await api(page, "/api/state");
  if (start.model.title !== "porous_benchmark" || start.path !== null) {
    throw new Error("the studio has been used: start a new one for the screenshots");
  }

  await shoot(page, "overview", async (p) => { await run(p, 5); });

  await openExample(page, "03_coated_inclusion.jl");
  await shoot(page, "layered", async (p) => {
    // the coated phase, whose shells the view cuts open
    await p.locator("#phases > .card").nth(1).click();
    await run(p, 5);
  });

  await openExample(page, "05_two_scales.jl");
  await shoot(page, "multiscale", async (p) => {
    // the outer scale, whose matrix is fed by the inner one
    await p.selectOption("#cell-select", { label: "paste" });
    await run(p, 5);
  });

  await openExample(page, "09_aging_creep.jl");
  // a second scheme from the Sweep tab, so that two curves share the figure
  const alv = await api(page, "/api/state");
  alv.model.sweep.schemes.push({ name: "SelfConsistent", options: {} });
  await api(page, "/api/model", { model: alv.model });
  await shoot(page, "viscoelastic", async (p) => { await tab(p, "alv"); await run(p, 10); });

  await openExample(page, "05_two_scales.jl");
  await shoot(page, "readme", async (p) => { await run(p, 5); }, SMALL);
} finally {
  await browser.close();
}
