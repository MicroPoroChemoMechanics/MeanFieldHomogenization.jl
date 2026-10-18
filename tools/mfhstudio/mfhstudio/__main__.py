"""MFH Studio — a graphical builder for MeanFieldHomogenization scripts.

    python3 -m mfhstudio                 start and open a browser
    python3 -m mfhstudio --port 9000     pick the port
    python3 -m mfhstudio --no-browser    stay in the terminal
    python3 -m mfhstudio --check         verify the Julia side and exit
    python3 -m mfhstudio --project @env  use another Julia environment

On Windows, `python -m mfhstudio`.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(prog="mfhstudio", description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="port to listen on (default: 8765, or the next free one)")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--project",
        default=os.environ.get("MFHSTUDIO_PROJECT"),
        help=(
            "Julia environment to run in: a directory, or `@name` for a shared "
            "one. Defaults to the MeanFieldHomogenization checkout this tool sits in. Use "
            "it when that checkout's Manifest.toml pins packages to local paths "
            "you do not have."
        ),
    )
    p.add_argument(
        "--julia",
        default=os.environ.get("JULIA"),
        help="path to the Julia executable, if it is not on PATH",
    )
    p.add_argument(
        "--check", action="store_true",
        help="report on the Julia environment and exit",
    )
    args = p.parse_args(argv)

    if args.check:
        return _check(args)

    from .server import serve

    serve(
        args.host, args.port,
        open_browser=not args.no_browser,
        project=args.project, julia=args.julia,
    )
    return 0


def _check(args) -> int:
    from .juliabridge import PROJECT_ROOT, Bridge, SidecarUnavailable, find_julia
    from .preflight import check_project

    exe = args.julia or find_julia()
    print(f"julia      : {exe or 'NOT FOUND on PATH'}")
    if exe is None:
        print(
            "\nInstall Julia, or point --julia (or the JULIA environment "
            "variable) at the executable.\n"
            "On Windows a juliaup install usually puts julia.exe in\n"
            "  %USERPROFILE%\\.juliaup\\bin",
            file=sys.stderr,
        )
        return 1

    project = args.project or PROJECT_ROOT
    print(f"project    : {project}")

    # A shared environment (`@name`) is Julia's to resolve; only a directory
    # can be inspected from here.
    if not str(project).startswith("@"):
        rep = check_project(project)
        print()
        print(rep.text())
        if not rep.ok:
            return 1

    b = Bridge(julia=args.julia, project=project)
    try:
        b.start()
    except SidecarUnavailable as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    try:
        cat = b.catalog()
        print(f"\nMeanFieldHomogenization {cat['mfh_version']} on Julia {cat['julia_version']}")
        print(f"  schemes: {len(cat['schemes'])}")
        return 0
    finally:
        b.stop()


if __name__ == "__main__":
    sys.exit(main())
