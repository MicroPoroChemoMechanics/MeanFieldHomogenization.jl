#!/usr/bin/env bash
# Everything the container needs before the studio can be useful — run once,
# at creation, and baked into a Codespaces prebuild.
#
# It is deliberately slow and thorough here so that the first click is fast.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
echo "▸ MFH Studio setup in ${ROOT}"

# `Manifest.toml` is untracked (it pins one developer's sibling checkouts), so
# a fresh clone resolves everything from the General registry, where both
# MeanFieldHomogenization and TensND live.
echo "▸ resolving and precompiling the package environment (a few minutes, once)"
julia --project="${ROOT}" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
'

# Plots is NOT a dependency of MeanFieldHomogenization, and deliberately so —
# it is a heavy dependency that the library itself never needs. But every
# script the studio generates ends with a figure, so `using Plots` has to
# resolve from somewhere. On a developer's machine it comes from their default
# environment, through `LOAD_PATH`; here that environment has to be filled in,
# or the first press of Run fails with "Plots not found" on a model that is
# perfectly correct.
echo "▸ installing Plots into the default environment (what generated scripts use)"
julia -e '
    using Pkg
    Pkg.add("Plots")
    Pkg.precompile()
'

# Pay the package load once now, so the sidecar the studio starts finds
# everything already compiled.
echo "▸ warming the package image"
julia --project="${ROOT}" -e 'using MeanFieldHomogenization, TensND; using Plots; println("  ready: MFH ", pkgversion(MeanFieldHomogenization))'

# The studio's own check, which diagnoses the Julia side without paying the
# load a second time. Non-fatal: a container that cannot run Julia should still
# open, with the interface saying so.
echo "▸ verifying the studio can reach Julia"
( cd tools/mfhstudio && python3 -m mfhstudio --check ) || echo "  (check reported a problem — the interface will explain it)"

echo "▸ setup complete"
