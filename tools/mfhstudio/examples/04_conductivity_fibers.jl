# =============================================================================
#  conductivity_fibers.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  Aligned conductive fibers (ω = 10) in a poor matrix. The result is transversely isotropic about e₃: K₃₃ follows the fibers, K₁₁ does not.
# =============================================================================
#
# After `scripts/32_spheroid_effective_conductivity.jl`.
#
# Everything above is stiffness; nothing about the schemes is. Store a
# second-order tensor under `:K`, ask for `:K`, and the same machinery
# answers a conduction problem. The inclusions are prolate spheroids —
# fibers — with no orientation average, so the effective conductivity is
# transversely isotropic and the plot follows components, not a scalar.
#

import Pkg
let d = @__DIR__
    while true
        pt = joinpath(d, "Project.toml")
        if isfile(pt) && occursin("MeanFieldHomogenization", read(pt, String))
            Pkg.activate(d; io = devnull)
            break
        end
        parent = dirname(d)
        parent == d && break
        d = parent
    end
end

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()

function build_rve()
    rve = RVE(:MATRIX)
    add_matrix!(rve, Spheroid(1.0), Dict(:K => TensISO{3}(1.0)))
    add_phase!(rve, :FIBERS, Spheroid(10.0), Dict(:K => TensISO{3}(50.0)); fraction = 0.2)
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const fs = range(0.0, 0.4; length = 21)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_rve()

function evaluate(scheme, f)
    cell = set_param(base_cell, amount(:FIBERS), f)
    C = homogenize(cell, scheme, :K)
    arr = Array(C)
    return (arr[1, 1], arr[3, 3])
end

const SCHEMES = [
    ("MoriTanaka", MoriTanaka()),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, f) for f in fs]
    RESULTS["$(name) C11"] = [r[1] for r in rows]
    RESULTS["$(name) C33"] = [r[2] for r in rows]
end

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(fs), "xlabel" => "f"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-28s  first = %.6f   last = %.6f\n" label first(ys) last(ys)
end

p = plot(; xlabel = "f", ylabel = "effective property", legend = :best)
for (label, ys) in CURVES
    plot!(p, fs, ys; label = label, lw = 2)
end
display(p)

#= mfhstudio-model v1
The studio reads this block to reopen the model exactly as it was.
Deleting it costs nothing but a best-effort re-reading of the code.
{
 "alv": {
  "cell": null,
  "component": [
   1,
   1
  ],
  "enabled": false,
  "length": 41,
  "log_time": false,
  "plot": true,
  "property": ":C",
  "scheme": "MoriTanaka",
  "t_start": 0.0,
  "t_stop": 10.0
 },
 "cells": [
  {
   "builder_name": null,
   "euler_angles": [],
   "frame_mode": "normal",
   "id": "cell1",
   "kind": "rve",
   "layers": [],
   "matrix_name": "MATRIX",
   "name": "rve",
   "normal": [
    0.0,
    0.0,
    1.0
   ],
   "params": [],
   "phases": [
    {
     "amount": 0.1,
     "amount_kind": "fraction",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "is_matrix": true,
     "name": "MATRIX",
     "properties": [
      {
       "args": {
        "k": 1.0
       },
       "builder": "TensISO{3}",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "iso_conduction",
       "key": ":K",
       "scheme": null,
       "scheme_options": {},
       "source": "builder",
       "visco": null
      }
     ],
     "symmetrize": "none"
    },
    {
     "amount": 0.2,
     "amount_kind": "fraction",
     "geometry": {
      "args": {
       "omega": 10.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "is_matrix": false,
     "name": "FIBERS",
     "properties": [
      {
       "args": {
        "k": 50.0
       },
       "builder": "TensISO{3}",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "iso_conduction",
       "key": ":K",
       "scheme": null,
       "scheme_options": {},
       "source": "builder",
       "visco": null
      }
     ],
     "symmetrize": "none"
    }
   ],
   "rve_options": {},
   "ui": {
    "x": 40,
    "y": 40
   }
  }
 ],
 "description": "Aligned conductive fibers (\u03c9 = 10) in a poor matrix. The result is transversely isotropic about e\u2083: K\u2083\u2083 follows the fibers, K\u2081\u2081 does not.",
 "opaque": [],
 "params": [],
 "root_cell": "cell1",
 "sens": {
  "cell": null,
  "enabled": false,
  "kind": "derivative",
  "lenses": [
   {
    "field_name": "semi_axes",
    "index": 1,
    "inner": null,
    "kind": "amount",
    "member": "",
    "phase": "",
    "property": ":C"
   }
  ],
  "output": {
   "kind": "k"
  },
  "projection": "iso",
  "property": ":C",
  "scheme": "MoriTanaka",
  "scheme_options": {}
 },
 "sweep": {
  "cell": "cell1",
  "clamp_zero": false,
  "enabled": true,
  "length": 21,
  "lens": {
   "field_name": "semi_axes",
   "index": 1,
   "inner": null,
   "kind": "amount",
   "member": "",
   "phase": "FIBERS",
   "property": ":C"
  },
  "mode": "sweep",
  "outputs": [
   {
    "i": 1,
    "j": 1,
    "kind": "comp"
   },
   {
    "i": 3,
    "j": 3,
    "kind": "comp"
   }
  ],
  "plot": true,
  "projection": "none",
  "property": ":K",
  "schemes": [
   {
    "name": "MoriTanaka",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 0.4,
  "variable": "f"
 },
 "title": "conductivity_fibers"
}
=#
