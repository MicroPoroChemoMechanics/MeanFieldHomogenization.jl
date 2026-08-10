# =============================================================================
#  ageing_creep.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  Elastic sand grains in a creeping paste. The curve is the uniaxial creep response, read off the Volterra inverse of the effective relaxation operator.
# =============================================================================
#
# After `scripts/53_ageing_creep_solid.jl` and `scripts/62_alv_schemes.jl`.
#
# A phase becomes viscoelastic through its *property*, not through a separate
# panel: pick a Kelvin chain or a Maxwell law in Properties, and the
# Viscoelastic tab then only chooses the time grid and the component to
# follow. `(1, 1)` is the uniaxial creep response.
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

function build_mortar()
    rve = RVE(:PASTE)
    add_matrix!(rve, Spheroid(1.0), Dict(:C => kelvin_iso(12.0, 8.0, [20.0], [12.0], [2.0], [2.0])))
    add_phase!(
        rve, :SAND, Spheroid(1.0), Dict(:C => heaviside_law(iso_stiffness(40.0, 30.0)));
        fraction = 0.35
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const times = range(0.0, 20.0; length = 41)

const cell = build_mortar()

#
# `homogenize_alv` returns the effective relaxation operator as a
# 6n x 6n block matrix on the time grid. Its Volterra inverse is
# the creep operator, and the response to a unit stress step is
# the row sum of its (11) blocks.
function uniaxial(R)
    J = volterra_inverse(R; block_size = 6)
    n = size(J, 1) ÷ 6
    return [sum(J[6 * (a - 1) + 1, 6 * (b - 1) + 1] for b in 1:n) for a in 1:n]
end

const RESULTS = Dict{String, Any}()
RESULTS["MoriTanaka"] = uniaxial(homogenize_alv(cell, MoriTanaka(), :C; times = times))

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(times), "xlabel" => "t"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-24s  J(t₁) = %.6g   J(t_end) = %.6g\n" label first(ys) last(ys)
end

p = plot(; xlabel = "t", ylabel = "uniaxial creep", legend = :best)
for (label, ys) in CURVES
    plot!(p, times, ys; label = label, lw = 2)
end
display(p)

#= mfhstudio-model v1
The studio reads this block to reopen the model exactly as it was.
Deleting it costs nothing but a best-effort re-reading of the code.
{
 "alv": {
  "cell": "cell1",
  "component": [
   1,
   1
  ],
  "enabled": true,
  "length": 41,
  "log_time": false,
  "plot": true,
  "property": ":C",
  "scheme": "MoriTanaka",
  "t_start": 0.0,
  "t_stop": 20.0
 },
 "cells": [
  {
   "builder_name": null,
   "euler_angles": [],
   "frame_mode": "euler",
   "id": "cell1",
   "kind": "rve",
   "layers": [],
   "matrix_name": "PASTE",
   "name": "mortar",
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
     "name": "PASTE",
     "properties": [
      {
       "args": {
        "k0": 12.0,
        "k1": 20.0,
        "mu0": 8.0,
        "mu1": 12.0,
        "tau_k": 2.0,
        "tau_mu": 2.0
       },
       "builder": "kelvin_iso",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "kelvin_iso",
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
     "amount": 0.35,
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
     "name": "SAND",
     "properties": [
      {
       "args": {
        "k": 40.0,
        "mu": 30.0
       },
       "builder": "heaviside_law",
       "cell": null,
       "euler_angles": [],
       "expr": "",
       "form": "visco_elastic",
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
 "description": "Elastic sand grains in a creeping paste. The curve is the uniaxial creep response, read off the Volterra inverse of the effective relaxation operator.",
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
 "title": "ageing_creep"
}
=#
