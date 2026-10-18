# =============================================================================
#  laminate_interfaces.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  A spring interface on top of layer A. Because it enters with weight 1/L, thickening the stack stiffens it — a size effect a perfect interface does not have.
# =============================================================================
#
# After `scripts/34_laminate_interfaces.jl`.
#
# With a spring interface the answer stops depending on the volume fractions
# alone: the interface enters with weight 1/L, an interface *density*, so the
# absolute period matters. That is why the layers are given by thickness here
# and not by fraction — and why the sweep varies a thickness, which moves the
# period as well as the fraction.
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

function build_lam()
    lam = Laminate()
    add_layer!(
        lam, :A, Dict(:C => iso_stiffness(2.0, 0.8));
        thickness = 0.3, interface = SpringInterface(0.01, 0.02)
    )
    add_layer!(lam, :B, Dict(:C => iso_stiffness(0.5, 0.2)); thickness = 0.7)
    return lam
end

# ── Result ──────────────────────────────────────────────────────────────────
const h_As = range(0.1, 2.0; length = 21)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_lam()

function evaluate(scheme, h_A)
    cell = set_param(base_cell, thickness(:A), h_A)
    C = homogenize(cell, scheme, :C)
    KMC = KM(C)
    return (KMC[3, 3], KMC[4, 4])
end

const SCHEMES = [
    ("Laminated", Laminated()),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, h_A) for h_A in h_As]
    RESULTS["$(name) KM33"] = [r[1] for r in rows]
    RESULTS["$(name) KM44"] = [r[2] for r in rows]
end

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(h_As), "xlabel" => "h_A"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-28s  first = %.6f   last = %.6f\n" label first(ys) last(ys)
end

p = plot(; xlabel = "h_A", ylabel = "effective property", legend = :best)
for (label, ys) in CURVES
    plot!(p, h_As, ys; label = label, lw = 2)
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
   "kind": "laminate",
   "layers": [
    {
     "amount": 0.3,
     "amount_kind": "thickness",
     "interface": {
      "args": {
       "kn": 0.01,
       "kt": 0.02
      },
      "kind": "SpringInterface"
     },
     "name": "A",
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
     ]
    },
    {
     "amount": 0.7,
     "amount_kind": "thickness",
     "interface": {
      "args": {},
      "kind": "PerfectInterface"
     },
     "name": "B",
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
    "x": 40,
    "y": 40
   }
  }
 ],
 "description": "A spring interface on top of layer A. Because it enters with weight 1/L, thickening the stack stiffens it \u2014 a size effect a perfect interface does not have.",
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
   "kind": "thickness",
   "member": "",
   "phase": "A",
   "property": ":C"
  },
  "mode": "sweep",
  "outputs": [
   {
    "i": 3,
    "j": 3,
    "kind": "km"
   },
   {
    "i": 4,
    "j": 4,
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
  "start": 0.1,
  "stop": 2.0,
  "variable": "h_A"
 },
 "title": "laminate_interfaces"
}
=#
