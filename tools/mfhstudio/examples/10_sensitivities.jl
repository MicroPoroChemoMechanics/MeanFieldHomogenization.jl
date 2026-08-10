# =============================================================================
#  sensitivities.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  How the effective bulk modulus moves with the porosity and with the solid's own stiffness, at the fractions entered above.
# =============================================================================
#
# After `scripts/26_sensitivities.jl`.
#
# ForwardDiff straight through the scheme. The point x₀ is the model itself —
# `get_param` reads it off the cell — so the amounts and moduli entered in
# Scales are where the derivative is taken, and nothing is typed twice.
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

function build_rve()
    rve = RVE(:SOLID)
    add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(72.0, 32.0)))
    add_phase!(
        rve, :PORE, Spheroid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
        fraction = 0.2
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const cell = build_rve()
const scheme = MoriTanaka()

const params = [
    amount(:PORE),
    property(:SOLID, :C, 1),
]

# `get_param` reads the point the derivative is taken at, so the
# values in the model are the x₀ — nothing is entered twice.
const D = gradient(cell, scheme, params; output = :C, indexer = C -> k_mu(best_fit_iso(C))[1])

const LABELS = [
    "amount(PORE)",
    "SOLID:C[1]",
]
@printf "%-34s  d(k) / d(p)\n" "parameter"
for (name, g) in zip(LABELS, D)
    @printf "%-34s  %.8g\n" name g
end

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
   "matrix_name": "SOLID",
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
     "name": "SOLID",
     "properties": [
      {
       "args": {
        "k": 72.0,
        "mu": 32.0
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
  }
 ],
 "description": "How the effective bulk modulus moves with the porosity and with the solid's own stiffness, at the fractions entered above.",
 "opaque": [],
 "params": [],
 "root_cell": "cell1",
 "sens": {
  "cell": "cell1",
  "enabled": true,
  "kind": "gradient",
  "lenses": [
   {
    "field_name": "semi_axes",
    "index": 1,
    "inner": null,
    "kind": "amount",
    "member": "",
    "phase": "PORE",
    "property": ":C"
   },
   {
    "field_name": "semi_axes",
    "index": 1,
    "inner": null,
    "kind": "property",
    "member": "",
    "phase": "SOLID",
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
  "enabled": false,
  "length": 21,
  "lens": {
   "field_name": "semi_axes",
   "index": 1,
   "inner": null,
   "kind": "amount",
   "member": "",
   "phase": "",
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
  "projection": "none",
  "property": ":C",
  "schemes": [
   {
    "name": "MoriTanaka",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 1.0,
  "variable": "\u03c6"
 },
 "title": "sensitivities"
}
=#
