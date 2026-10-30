#!/usr/bin/env bash
# Run every example and copy its figure and results into the documentation
# (docs/src/assets/fe_coupling/fenicsx), which does not run Python itself.
# The environment of the README must be set (PYTHON_JULIAPKG_EXE,
# PYTHON_JULIAPKG_PROJECT, PYTHON_JULIACALL_THREADS). Example 4 takes about
# 25 minutes on 4 threads; example 6 must run alone on the machine.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/results}
assets=$here/../../../docs/src/assets/fe_coupling/fenicsx
mkdir -p "$assets"
python=${PYTHON:-python}
for script in 01_plate_with_hole 02_notched_plate_crack_growth 03_borehole_closing_cracks 04_polycrystal_plate 05_composite_beam_creep 06_thread_scaling; do
    echo "== $script"
    "$python" "$here/$script.py" --out "$out/$script"
    cp "$out/$script"/*.svg "$assets/"
    cp "$out/$script/results.json" "$assets/$script.json"
done
