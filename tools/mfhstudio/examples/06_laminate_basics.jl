# =============================================================================
#  laminate_basics.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  A 30/70 bilayer normal to e₃. Compare the three numbers: Laminated equals Reuss on KM[3,3] and Voigt on KM[6,6], exactly.
# =============================================================================
#
# After `scripts/33_laminate_basics.jl`.
#
# A laminate is a *cell*, not an inclusion: a periodic stack with no matrix
# and no reference medium, and an exact effective behaviour rather than an
# estimate. Two isotropic layers give a transversely isotropic result — that
# is Backus (1962) — and the bounds saturate: exactly Reuss across the layers
# (KM 3,3) and exactly Voigt in their plane (KM 6,6), at once.
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

function build_lam()
    lam = Laminate()
    add_layer!(lam, :A, Dict(:C => iso_stiffness(2.0, 0.8)); fraction = 0.3)
    add_layer!(lam, :B, Dict(:C => iso_stiffness(0.5, 0.2)); fraction = 0.7)
    return lam
end

# ── Result ──────────────────────────────────────────────────────────────────
# One homogenization with the amounts entered in the model.
const cell = build_lam()

C_Laminated = homogenize(cell, Laminated(), :C)
KMC_Laminated = KM(C_Laminated)
@printf "Laminated                 KM33 = %.6f  KM66 = %.6f\n" KMC_Laminated[3, 3] KMC_Laminated[6, 6]
C_Voigt = homogenize(cell, Voigt(), :C)
KMC_Voigt = KM(C_Voigt)
@printf "Voigt                     KM33 = %.6f  KM66 = %.6f\n" KMC_Voigt[3, 3] KMC_Voigt[6, 6]
C_Reuss = homogenize(cell, Reuss(), :C)
KMC_Reuss = KM(C_Reuss)
@printf "Reuss                     KM33 = %.6f  KM66 = %.6f\n" KMC_Reuss[3, 3] KMC_Reuss[6, 6]

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
   "kind": "laminate",
   "layers": [
    {
     "amount": 0.3,
     "amount_kind": "fraction",
     "interface": {
      "args": {},
      "kind": "PerfectInterface"
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
     "amount_kind": "fraction",
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
 "description": "A 30/70 bilayer normal to e\u2083. Compare the three numbers: Laminated equals Reuss on KM[3,3] and Voigt on KM[6,6], exactly.",
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
   "phase": "",
   "property": ":C"
  },
  "mode": "single",
  "outputs": [
   {
    "i": 3,
    "j": 3,
    "kind": "km"
   },
   {
    "i": 6,
    "j": 6,
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
   },
   {
    "name": "Voigt",
    "options": {}
   },
   {
    "name": "Reuss",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 1.0,
  "variable": "\u03c6"
 },
 "title": "laminate_basics"
}
=#
