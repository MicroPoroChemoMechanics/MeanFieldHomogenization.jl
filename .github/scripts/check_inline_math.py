#!/usr/bin/env python3
"""Catch an inline formula that a line wrap turns into a Markdown block.

    python3 .github/scripts/check_inline_math.py [path ...]

## The rule this models

Documenter writes inline mathematics between double backticks, and a long
formula is wrapped like the prose around it. Markdown reads the blocks of a page
before its inline spans: a wrapped line that begins with a list marker, a quote
marker or a heading marker starts a new block, and the formula is cut in two.

    ``\\boldsymbol{Q} = \\boldsymbol{K}_0
    - \\boldsymbol{K}_0\\cdot\\underline{\\xi}``

is rendered as a paragraph ending with raw LaTeX, followed by a list item. No
build fails and no warning is printed: only the rendered page shows it.

The script flags every line, inside a double-backtick span, that begins with

    "- "  "+ "  "* "  "> "  "#"  "1. "  "1) "

or is blank. The fix is to end the previous line with the operator:

    ``\\boldsymbol{Q} = \\boldsymbol{K}_0 -
    \\boldsymbol{K}_0\\cdot\\underline{\\xi}``

## What is scanned

Markdown pages, Julia sources (their docstrings) and Literate scripts (their
`# ` comment lines), fenced blocks left out. With no argument: `docs/src`,
`src`, `ext` and `scripts`.
"""
import pathlib
import re
import sys

BREAKER = re.compile(r"(?:[-+*>]\s|#|\d+[.)]\s)")
SPAN = re.compile(r"``(.+?)``", re.S)


def prose(path, text):
    """The text of `path` with code replaced by empty lines, line numbers kept."""
    literate = path.suffix == ".jl" and "scripts" in path.parts
    lines, fenced = [], False
    for line in text.split("\n"):
        if literate:
            line = line[2:] if line.startswith("# ") else ""
        if line.lstrip().startswith("```"):
            fenced = not fenced
            lines.append("")
        else:
            lines.append("" if fenced else line)
    return "\n".join(lines)


def flags(path):
    text = prose(path, path.read_text(encoding="utf-8"))
    for span in SPAN.finditer(text):
        first = text.count("\n", 0, span.start()) + 1
        for offset, line in enumerate(span.group(1).split("\n")[1:], 1):
            if not line.strip() or BREAKER.match(line.lstrip()):
                yield first + offset, line.strip()


def files(arguments):
    roots = [pathlib.Path(a) for a in arguments] or [
        pathlib.Path(p) for p in ("docs/src", "src", "ext", "scripts")
    ]
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(p for p in root.rglob("*") if p.suffix in (".md", ".jl") and p.is_file())


def main(arguments):
    found = 0
    for path in files(arguments):
        for number, line in flags(path):
            found += 1
            print(f"{path}:{number}: an inline formula is cut by a line starting with {line[:40]!r}")
    if found:
        print(f"{found} inline formula(s) cut by a line wrap: end the previous line with the operator.")
        return 1
    print("OK: no inline formula is cut by a line wrap")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
