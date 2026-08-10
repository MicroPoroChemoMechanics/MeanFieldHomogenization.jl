# =============================================================================
#  coated_inclusion.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  A stiff core in a compliant shell, dispersed in a matrix. The shells are given by outer radius, ascending, with r = 0 at the center.
# =============================================================================
#
# After `scripts/30_average_nlayers.jl`.
#
# A `LayeredSphere` is one inclusion with concentric shells, given by their
# outer radii with r = 0 implicit at the center — not a nested cell. The
# interface between two shells can be imperfect; here they are perfect and
# the interest is in the coating alone.
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
    add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(20.0, 8.0)))
    add_phase!(
        rve, :COATED, LayeredSphere((0.7, 1.0), (iso_stiffness(60.0, 30.0), iso_stiffness(5.0, 2.0))), Dict(:C => iso_stiffness(60.0, 30.0));
        fraction = 0.2
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const fs = range(0.0, 0.5; length = 21)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_rve()

function evaluate(scheme, f)
    cell = set_param(base_cell, amount(:COATED), f)
    C = homogenize(cell, scheme, :C)
    C = best_fit_iso(C)
    km = k_mu(C)
    return (km[1], km[2])
end

const SCHEMES = [
    ("MoriTanaka", MoriTanaka()),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, f) for f in fs]
    RESULTS["$(name) k"] = [r[1] for r in rows]
    RESULTS["$(name) mu"] = [r[2] for r in rows]
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
   "frame_mode": "euler",
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
        "k": 20.0,
        "mu": 8.0
       },
       "builder": "iso_stiffness",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "iso_kmu",
       "key": ":C",
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
      "args": {},
      "euler_angles": [],
      "kind": "layered_sphere",
      "layers": [
       {
        "interface": {
         "args": {},
         "kind": "PerfectInterface"
        },
        "property": {
         "args": {
          "k": 60.0,
          "mu": 30.0
         },
         "builder": "iso_stiffness",
         "cell": null,
         "euler_angles": [],
         "expr": "",
         "form": "iso_kmu",
         "key": ":C",
         "scheme": null,
         "scheme_options": {},
         "source": "builder",
         "visco": null
        },
        "radius": 0.7
       },
       {
        "interface": {
         "args": {},
         "kind": "PerfectInterface"
        },
        "property": {
         "args": {
          "k": 5.0,
          "mu": 2.0
         },
         "builder": "iso_stiffness",
         "cell": null,
         "euler_angles": [],
         "expr": "",
         "form": "iso_kmu",
         "key": ":C",
         "scheme": null,
         "scheme_options": {},
         "source": "builder",
         "visco": null
        },
        "radius": 1.0
       }
      ]
     },
     "is_matrix": false,
     "name": "COATED",
     "properties": [
      {
       "args": {
        "k": 60.0,
        "mu": 30.0
       },
       "builder": "iso_stiffness",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "iso_kmu",
       "key": ":C",
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
 "description": "A stiff core in a compliant shell, dispersed in a matrix. The shells are given by outer radius, ascending, with r = 0 at the center.",
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
   "phase": "COATED",
   "property": ":C"
  },
  "mode": "sweep",
  "outputs": [
   {
    "kind": "k"
   },
   {
    "kind": "mu"
   }
  ],
  "plot": true,
  "projection": "iso",
  "property": ":C",
  "schemes": [
   {
    "name": "MoriTanaka",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 0.5,
  "variable": "f"
 },
 "title": "coated_inclusion"
}
=#
