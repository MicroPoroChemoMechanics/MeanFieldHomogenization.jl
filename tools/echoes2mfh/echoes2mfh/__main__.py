"""echoes2mfh command line interface.

    python -m echoes2mfh survey    <dir>            classify a corpus
    python -m echoes2mfh translate <script.py>      emit Julia
    python -m echoes2mfh check-drift                validate the mapping tables
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

from . import mapping, survey
from .emit import emit
from .extract import extract

# The pybind11 sources of an Echoes checkout (`echoes_cpp/interface/pybind11/py_echoes`),
# from which the Echoes side of the survey and of the drift check reads the public
# names: given with `--pybind` or the `ECHOES_PYBIND` environment variable; without
# either, that side is skipped.
DEFAULT_PYBIND = os.environ.get("ECHOES_PYBIND")
DEFAULT_MFH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

EXIT_OK = 0
EXIT_FINDINGS = 2
EXIT_REFUSED = 3


# ---------------------------------------------------------------------------


def cmd_survey(args: argparse.Namespace) -> int:
    entries = survey.scan_tree(args.root, args.pybind)
    print(survey.render_table(entries, args.root))

    hist = survey.unmapped_histogram(entries)
    if hist:
        print("\nUnmapped Echoes symbols, by scripts blocked (the work queue):")
        for sym, n in hist[: args.top]:
            print(f"  {n:>4}  {sym}")

    if args.json:
        payload = [
            {
                "path": e.path,
                "family": e.family,
                "loc": e.loc,
                "verdict": e.verdict,
                "blocker": e.blocker,
                "features": sorted(e.features),
                "unmapped": sorted(e.unmapped),
                "third_party": sorted(e.third_party),
            }
            for e in entries
        ]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json}")
    return EXIT_OK


# ---------------------------------------------------------------------------


def cmd_translate(args: argparse.Namespace) -> int:
    src = open(args.script, encoding="utf-8", errors="replace").read()
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(
            f"echoes2mfh: cannot parse {args.script}: {e.msg} (line {e.lineno})",
            file=sys.stderr,
        )
        print(
            "  This is Python 2 syntax. Run `2to3 -w` on it first.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    script = extract(src, args.script)
    name = args.name or os.path.splitext(os.path.basename(args.script))[0]
    julia = emit(script, name, literate=args.literate)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(julia)
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(julia)

    findings = script.findings
    if args.report:
        payload = {
            "source": args.script,
            "findings": [
                {
                    "severity": f.severity,
                    "lineno": f.lineno,
                    "symbol": f.symbol,
                    "reason": f.reason,
                    "suggestion": f.suggestion,
                    "python": f.py_src,
                }
                for f in findings
            ],
        }
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    if findings:
        n_block = script.finding_count("blocking")
        print(
            f"\n{len(findings)} construct(s) not translated "
            f"({n_block} blocking). Each is marked UNTRANSLATED in the output.",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  line {f.lineno}: {f.reason}", file=sys.stderr)
        return EXIT_FINDINGS
    return EXIT_OK


# ---------------------------------------------------------------------------


def cmd_check_drift(args: argparse.Namespace) -> int:
    """Detect divergence between the mapping tables and the two live APIs."""
    problems = 0

    known = survey.echoes_symbol_set(args.pybind)
    if args.pybind and os.path.isdir(args.pybind):
        mapped = mapping.all_mapped_names()
        # Only symbols the corpus actually uses matter; the rest is noise.
        print(f"Echoes symbols found in bindings: {len(known)}")
        print(f"Symbols mapped or explicitly refused: {len(mapped & known)}")
        missing = known - mapped
        if args.verbose and missing:
            print(f"\nNot yet in a mapping table ({len(missing)}):")
            for s in sorted(missing):
                print(f"  {s}")
    else:
        print(f"(skipping Echoes side: {args.pybind} not found)")

    mfh_src = os.path.join(args.mfh, "src", "MeanFieldHomogenization.jl")
    if os.path.isfile(mfh_src):
        exported: set[str] = set()
        for line in open(mfh_src, encoding="utf-8"):
            s = line.strip()
            if s.startswith("export "):
                for tok in s[len("export "):].split(","):
                    tok = tok.strip()
                    if tok and tok.isidentifier():
                        exported.add(tok)
        print(f"\nMFH exported names: {len(exported)}")
        referenced = _referenced_mfh_names()
        unknown = {n for n in referenced if n not in exported}
        # TensND and Base names legitimately appear in the tables
        allowed = {
            "TensISO", "Array", "inv", "tr", "one", "zero", "eigen", "I",
            "Tuple", "Dict", "Float64", "Int", "string", "length", "sum",
            "maximum", "minimum", "abs", "max", "min", "round", "transpose",
            "vcat", "det", "norm", "eigvals", "pinv", "real", "imag", "conj",
            "sqrt", "exp", "log", "log10", "log2", "cos", "sin", "tan",
            "acos", "asin", "atan", "cosh", "sinh", "tanh", "floor", "ceil",
            "sign", "zeros", "ones", "plot", "scatter", "hline", "vline",
            "range", "Inf", "NaN", "pi",
            # module qualifiers and keyword names, not call targets
            "TensND", "Tens", "SymmetricTensor", "get_data", "tomandel",
            "angles", "e", "step", "length", "count", "im",
        }
        unknown -= allowed
        if unknown:
            problems += len(unknown)
            print(f"\nMapping targets NOT exported by MFH ({len(unknown)}):")
            for n in sorted(unknown):
                print(f"  {n}")
        else:
            print("All mapping targets resolve to MFH exports or Base/TensND.")
    else:
        print(f"(skipping MFH side: {mfh_src} not found)")

    # scheme aliases must still be accepted by MFH
    hom = os.path.join(args.mfh, "src", "Schemes", "homogenize.jl")
    if os.path.isfile(hom):
        text = open(hom, encoding="utf-8").read()
        for name in mapping.SCHEMES:
            if f":{name}" not in text:
                print(f"  note: scheme alias :{name} not found in homogenize.jl")

    return EXIT_OK if problems == 0 else EXIT_FINDINGS


def _referenced_mfh_names() -> set[str]:
    """Julia identifiers appearing as targets in the mapping tables."""
    import re

    out: set[str] = set()
    tables = [
        {k: v.ctor for k, v in mapping.SCHEMES.items()},
        mapping.SYMMETRIZE,
        mapping.PARAMSYM,
        mapping.ROTATIONAL_AVERAGE,
        mapping.INTERFACE_TYPE,
        {k: v.emit for k, v in mapping.GEOMETRY.items()},
        mapping.TENSOR_BUILDERS,
        mapping.SCALAR_CONVERSIONS,
        mapping.CONSTANTS,
        mapping.PHASE_ACCESSORS,
        mapping.PHASE_ACCESSORS_TRANSPORT,
        mapping.TENSOR_ATTRS,
        mapping.FUNCTIONS,
        mapping.NUMPY,
        mapping.NUMPY_ATTR,
    ]
    ident = re.compile(r"[A-Za-z_][A-Za-z0-9_!]*")
    for t in tables:
        for v in t.values():
            for m in ident.findall(str(v)):
                out.add(m)
    return out


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="echoes2mfh", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("survey", help="classify a corpus of Echoes scripts")
    s.add_argument("root")
    s.add_argument("--pybind", default=DEFAULT_PYBIND, help="pybind11 sources of Echoes (default: $ECHOES_PYBIND)")
    s.add_argument("--json", help="also write the classification as JSON")
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(func=cmd_survey)

    t = sub.add_parser("translate", help="translate one script to Julia")
    t.add_argument("script")
    t.add_argument("-o", "--output")
    t.add_argument("--name", help="name for the generated script")
    t.add_argument("--report", help="write findings as JSON")
    t.add_argument(
        "--literate",
        action="store_true",
        help="emit the Literate.jl dual-usage contract",
    )
    t.set_defaults(func=cmd_translate)

    d = sub.add_parser("check-drift", help="validate the mapping against both APIs")
    d.add_argument("--pybind", default=DEFAULT_PYBIND, help="pybind11 sources of Echoes (default: $ECHOES_PYBIND)")
    d.add_argument("--mfh", default=DEFAULT_MFH)
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_check_drift)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
