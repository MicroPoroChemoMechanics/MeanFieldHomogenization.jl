# =============================================================================
#  two_scales.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  A gel foam homogenized first, then used as the matrix of a paste. The sweep varies the inner porosity through the seam.
# =============================================================================
#
# After `scripts/42_cementpaste_iso.jl`.
#
# The multiscale seam: a phase property of the outer cell holds the effective
# property of the inner one, written `Homogenized(inner, scheme)`. The outer
# scheme resolves it when it reads the key, so the two scales are declared
# rather than chained by hand — and the sweep below reaches through the seam
# with a `nested` lens, in one pass.
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

function build_foam()
    rve = RVE()
    add_phase!(
        rve, :GEL, Spheroid(1.0), Dict(:C => iso_stiffness(20.0, 9.0));
        fraction = :rest
    )
    add_phase!(
        rve, :GELPORE, Spheroid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
        fraction = 0.28
    )
    return rve
end

function build_paste()
    rve = RVE()
    add_phase!(
        rve, :FOAM, Spheroid(1.0), Dict(:C => Homogenized(build_foam(), SelfConsistent()));
        fraction = :rest
    )
    add_phase!(
        rve, :CLINKER, Spheroid(1.0), Dict(:C => iso_stiffness(112.0, 50.0));
        fraction = 0.15
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const φ_gels = range(0.0, 0.5; length = 21)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_paste()

function evaluate(scheme, φ_gel)
    cell = set_param(base_cell, nested(:FOAM, :C, amount(:GELPORE)), φ_gel)
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
    rows = [evaluate(scheme, φ_gel) for φ_gel in φ_gels]
    RESULTS["$(name) k"] = [r[1] for r in rows]
    RESULTS["$(name) mu"] = [r[2] for r in rows]
end

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(φ_gels), "xlabel" => "φ_gel"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-28s  first = %.6f   last = %.6f\n" label first(ys) last(ys)
end

p = plot(; xlabel = "φ_gel", ylabel = "effective property", legend = :best)
for (label, ys) in CURVES
    plot!(p, φ_gels, ys; label = label, lw = 2)
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
   "matrix_name": "GEL",
   "name": "foam",
   "normal": [
    0.0,
    0.0,
    1.0
   ],
   "params": [],
   "phases": [
    {
     "amount": 0.1,
     "amount_kind": "rest",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "name": "GEL",
     "properties": [
      {
       "args": {
        "k": 20.0,
        "mu": 9.0
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
     "amount": 0.28,
     "amount_kind": "fraction",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "name": "GELPORE",
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
   "frame_mode": "euler",
   "id": "cell2",
   "kind": "rve",
   "layers": [],
   "matrix_name": "FOAM",
   "name": "paste",
   "normal": [
    0.0,
    0.0,
    1.0
   ],
   "params": [],
   "phases": [
    {
     "amount": 0.1,
     "amount_kind": "rest",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "name": "FOAM",
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
       "scheme": "SelfConsistent",
       "scheme_options": {},
       "source": "cell",
       "visco": null
      }
     ],
     "symmetrize": "none"
    },
    {
     "amount": 0.15,
     "amount_kind": "fraction",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
     "name": "CLINKER",
     "properties": [
      {
       "args": {
        "k": 112.0,
        "mu": 50.0
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
    "x": 320,
    "y": 110
   }
  }
 ],
 "description": "A gel foam homogenized first, then used as the matrix of a paste. The sweep varies the inner porosity through the seam.",
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
    "phase": "GELPORE",
    "property": ":C"
   },
   "kind": "nested",
   "member": "FOAM",
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
  "variable": "\u03c6_gel"
 },
 "title": "two_scales"
}
=#
