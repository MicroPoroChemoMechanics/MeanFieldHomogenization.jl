"""Corpus classifier.

Answers "which of these scripts can be translated, and what blocks the rest"
without needing Julia, Echoes, or a complete mapping. Run it over a directory
to get a work queue rather than a guess.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field

from . import mapping

# Features that decide which MFH subsystem a script needs.
FEATURES = {
    "rve": "rve",
    "homogenize": "homogenize",
    "homogenize_visco": "homogenize_visco",
    "homogenize_derivative": "homogenize_derivative",
    "sphere_nlayers": "sphere_nlayers",
    "spheroid_nlayers": "spheroid_nlayers",
    "user_inclusion": "user_inclusion",
    "crack": "crack",
    "hill": "hill",
    "hill_dual": "hill",
    "hill_taylor_0": "hill_taylor",
    "hill_taylor_1": "hill_taylor",
    "eshelby": "eshelby",
    "visco_law": "visco_law",
    "crack_compliance": "crack_compliance",
}

# Which MFH feature answers each Echoes feature. Empty means "no counterpart".
MFH_TARGET = {
    "rve": "RVE / add_phase!",
    "homogenize": "homogenize",
    "homogenize_visco": "homogenize_alv",
    "homogenize_derivative": "derivative / gradient / jacobian (ForwardDiff)",
    "sphere_nlayers": "LayeredSphere",
    "spheroid_nlayers": "LayeredSpheroid",
    "user_inclusion": "CustomInclusion",
    "crack": "PennyCrack / EllipticCrack / RibbonCrack",
    "hill": "hill_tensor",
    "hill_taylor": "",
    "eshelby": "eshelby_tensor",
    "visco_law": "ViscoLaw",
    "crack_compliance": "cod_tensor",
}


@dataclass
class Entry:
    path: str
    family: str
    loc: int = 0
    parse_ok: bool = True
    parse_error: str = ""
    features: set[str] = field(default_factory=set)
    third_party: set[str] = field(default_factory=set)
    out_of_scope: set[str] = field(default_factory=set)
    echoes_symbols: set[str] = field(default_factory=set)
    unmapped: set[str] = field(default_factory=set)

    @property
    def verdict(self) -> str:
        if not self.parse_ok:
            return "PYTHON2"
        if self.out_of_scope:
            return "OUT_OF_SCOPE"
        if any(not MFH_TARGET.get(f, "x") for f in self.features):
            return "NO_MFH_TARGET"
        if self.unmapped:
            return "NEEDS_MAPPING"
        return "PORTABLE"

    @property
    def blocker(self) -> str:
        if not self.parse_ok:
            return f"Python 2 syntax: {self.parse_error}"
        if self.out_of_scope:
            m = sorted(self.out_of_scope)[0]
            return f"{m}: {mapping.OUT_OF_SCOPE_IMPORTS.get(m, '')}"
        for f in sorted(self.features):
            if not MFH_TARGET.get(f, "x"):
                return f"`{f}` has no verified MFH counterpart"
        if self.unmapped:
            return "unmapped Echoes symbols: " + ", ".join(
                sorted(self.unmapped)[:6]
            )
        return ""


STDLIB = {
    "math", "os", "sys", "time", "json", "csv", "re", "copy", "itertools",
    "functools", "random", "datetime", "warnings", "collections", "cmath",
    "subprocess", "pickle", "glob", "operator", "string", "io", "pathlib",
    "typing", "abc", "traceback", "argparse", "tempfile", "shutil", "textwrap",
    "__future__",
}


def scan_file(path: str, family: str, known: set[str]) -> Entry:
    src = open(path, encoding="utf-8", errors="replace").read()
    e = Entry(path=path, family=family, loc=len(src.splitlines()))
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        e.parse_ok = False
        e.parse_error = f"{exc.msg} (line {exc.lineno})"
        return e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                _note_import(e, a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            _note_import(e, node.module.split(".")[0])
        elif isinstance(node, ast.Name):
            _note_symbol(e, node.id, known)
        elif isinstance(node, ast.Attribute):
            _note_symbol(e, node.attr, known)
        elif isinstance(node, ast.ClassDef):
            for b in node.bases:
                if isinstance(b, ast.Name):
                    _note_symbol(e, b.id, known)
    return e


def _note_import(e: Entry, mod: str) -> None:
    if mod in ("echoes",) or mod in STDLIB:
        return
    if mod in mapping.OUT_OF_SCOPE_IMPORTS:
        e.out_of_scope.add(mod)
    e.third_party.add(mod)


def _note_symbol(e: Entry, name: str, known: set[str]) -> None:
    if name in FEATURES:
        e.features.add(FEATURES[name])
    if name in known:
        e.echoes_symbols.add(name)
        if name not in mapping.all_mapped_names():
            e.unmapped.add(name)


def echoes_symbol_set(pybind_root: str | None) -> set[str]:
    """Scrape the Echoes public names from the pybind11 sources."""
    import glob
    import re

    if not pybind_root or not os.path.isdir(pybind_root):
        return set(mapping.all_mapped_names())
    syms: set[str] = set()
    pat_def = re.compile(r'\.def[a-z_]*\(\s*"([^"]+)"')
    pat_attr = re.compile(r'attr\(\s*"([^"]+)"\s*\)')
    pat_val = re.compile(r'\.value\(\s*"([^"]+)"')
    for p in glob.glob(os.path.join(pybind_root, "**", "*.h"), recursive=True) + glob.glob(
        os.path.join(pybind_root, "**", "*.cpp"), recursive=True
    ):
        s = open(p, encoding="utf-8", errors="replace").read()
        syms |= set(pat_def.findall(s))
        syms |= set(pat_attr.findall(s))
        syms |= set(pat_val.findall(s))
    return syms


def scan_tree(root: str, pybind_root: str | None = None) -> list[Entry]:
    known = echoes_symbol_set(pybind_root)
    out: list[Entry] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            family = os.path.relpath(dirpath, root).split(os.sep)[0]
            out.append(scan_file(path, family, known))
    return sorted(out, key=lambda e: e.path)


def render_table(entries: list[Entry], root: str) -> str:
    rows = []
    width = max((len(os.path.relpath(e.path, root)) for e in entries), default=20)
    width = min(width, 52)
    rows.append(f"{'script'.ljust(width)}  {'LOC':>5}  {'verdict':<14}  blocker")
    rows.append("-" * (width + 30))
    for e in entries:
        rel = os.path.relpath(e.path, root)
        if len(rel) > width:
            rel = "…" + rel[-(width - 1):]
        rows.append(
            f"{rel.ljust(width)}  {e.loc:>5}  {e.verdict:<14}  {e.blocker[:60]}"
        )
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.verdict] = counts.get(e.verdict, 0) + 1
    rows.append("")
    rows.append(f"{len(entries)} scripts:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        rows.append(f"  {counts[k]:>4}  {k}")
    return "\n".join(rows)


def unmapped_histogram(entries: list[Entry]) -> list[tuple[str, int]]:
    """Which unmapped Echoes symbols block the most scripts -- the work queue."""
    hist: dict[str, int] = {}
    for e in entries:
        for s in e.unmapped:
            hist[s] = hist.get(s, 0) + 1
    return sorted(hist.items(), key=lambda kv: -kv[1])
