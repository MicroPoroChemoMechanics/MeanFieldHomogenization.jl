# =============================================================================
#  laminate_multiscale.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  A three-scale chain: a porous RVE becomes one layer of a stack. The sweep varies the inner porosity through the seam.
# =============================================================================
#
# After `scripts/36_laminate_multiscale.jl`.
#
# The seam works on a laminate exactly as on an RVE: a layer property may be
# a `Homogenized`. Drag a connector onto a layer's `:C` slot in the graph and
# this is what you get.
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

# ── Scales ──────────────────────────────────────────────────────────────────
#
# Inner scales are defined first. A phase property holding a
# `Homogenized(cell, scheme)` is the seam: the outer scheme
# resolves the inner scale when it reads that property, and
# memoizes it for the duration of the call.

function build_porous()
    rve = RVE(:SOLID)
    add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(2.0, 0.8)))
    add_phase!(
        rve, :PORE, Spheroid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
        fraction = 0.25
    )
    return rve
end

function build_lam()
    lam = Laminate()
    add_layer!(
        lam, :POROUS, Dict(:C => Homogenized(build_porous(), MoriTanaka()));
        fraction = 0.4
    )
    add_layer!(lam, :DENSE, Dict(:C => iso_stiffness(0.5, 0.2)); fraction = 0.6)
    return lam
end

# ── Result ──────────────────────────────────────────────────────────────────
const φs = range(0.0, 0.5; length = 21)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_lam()

function evaluate(scheme, φ)
    cell = set_param(base_cell, nested(:POROUS, :C, amount(:PORE)), φ)
    C = homogenize(cell, scheme, :C)
    return (KM(C)[3, 3],)
end

const SCHEMES = [
    ("Laminated", Laminated()),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, φ) for φ in φs]
    RESULTS["$(name) KM33"] = [r[1] for r in rows]
end

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(φs), "xlabel" => "φ"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-28s  first = %.6f   last = %.6f\n" label first(ys) last(ys)
end

p = plot(; xlabel = "φ", ylabel = "effective property", legend = :best)
for (label, ys) in CURVES
    plot!(p, φs, ys; label = label, lw = 2)
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
   "matrix_name": "SOLID",
   "name": "porous",
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
     "name": "SOLID",
     "properties": [
      {
       "args": {
        "k": 2.0,
        "mu": 0.8
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
     "amount": 0.25,
     "amount_kind": "fraction",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "is_matrix": false,
     "name": "PORE",
     "properties": [
      {
       "args": {
        "k": 1e-06,
        "mu": 1e-06
       },
       "builder": "iso_stiffness",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "void",
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
  },
  {
   "builder_name": null,
   "euler_angles": [],
   "frame_mode": "normal",
   "id": "cell2",
   "kind": "laminate",
   "layers": [
    {
     "amount": 0.4,
     "amount_kind": "fraction",
     "interface": {
      "args": {},
      "kind": "PerfectInterface"
     },
     "name": "POROUS",
     "properties": [
      {
       "args": {
        "k": 10.0,
        "mu": 5.0
       },
       "builder": "iso_stiffness",
       "cell": "cell1",
       "euler_angles": [],
       "expr": "",
       "form": "iso_kmu",
       "key": ":C",
       "scheme": "MoriTanaka",
       "scheme_options": {},
       "source": "cell",
       "visco": null
      }
     ]
    },
    {
     "amount": 0.6,
     "amount_kind": "fraction",
     "interface": {
      "args": {},
      "kind": "PerfectInterface"
     },
     "name": "DENSE",
     "properties": [
      {
       "args": {
        "k": 0.5,
        "mu": 0.2
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
     ]
    }
   ],
   "matrix_name": "MATRIX",
   "name": "lam",
   "normal": [
    0.0,
    0.0,
    1.0
   ],
   "params": [],
   "phases": [],
   "rve_options": {},
   "ui": {
    "x": 320,
    "y": 100
   }
  }
 ],
 "description": "A three-scale chain: a porous RVE becomes one layer of a stack. The sweep varies the inner porosity through the seam.",
 "opaque": [],
 "params": [],
 "root_cell": "cell2",
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
  "cell": "cell2",
  "clamp_zero": false,
  "enabled": true,
  "length": 21,
  "lens": {
   "field_name": "semi_axes",
   "index": 1,
   "inner": {
    "field_name": "semi_axes",
    "index": 1,
    "inner": null,
    "kind": "amount",
    "member": "",
    "phase": "PORE",
    "property": ":C"
   },
   "kind": "nested",
   "member": "POROUS",
   "phase": "",
   "property": ":C"
  },
  "mode": "sweep",
  "outputs": [
   {
    "i": 3,
    "j": 3,
    "kind": "km"
   }
  ],
  "plot": true,
  "projection": "none",
  "property": ":C",
  "schemes": [
   {
    "name": "Laminated",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 0.5,
  "variable": "\u03c6"
 },
 "title": "laminate_multiscale"
}
=#
