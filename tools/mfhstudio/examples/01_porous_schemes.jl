# =============================================================================
#  porous_schemes.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  Spherical pores in an isotropic solid, under every scheme at once. Voigt and Reuss bracket; the self-consistent schemes vanish at f = 1/2; Mori-Tanaka, Maxwell and PCW never do.
# =============================================================================
#
# After `scripts/28_porous_schemes.jl` and `scripts/20_voigt_reuss_bounds.jl`.
#
# Spherical pores in an isotropic solid, swept over the porosity, with every
# scheme the package offers on one figure — which is the usual reason to draw
# one. Voigt and Reuss bracket everything; Dilute and DiluteDual are only
# honest at small ``f`` and leave the bounds beyond it; the self-consistent
# scheme loses all rigidity at ``f = 1/2`` while Mori-Tanaka, Maxwell and PCW
# never do; the differential scheme percolates only at ``f = 1``.
#
# Solver options ride on the scheme that reads them, not on `homogenize`.
# They are the script's own: `select_best` matters for the self-consistent
# schemes near percolation, where several fixed points exist.
#
# Negative values are clipped, as the script clips them. `Dilute` goes below
# zero well before ``f = 1``, and on a figure shared by ten schemes that one
# curve would set the scale for all of them. It is a display choice: run a
# single scheme with the box unticked and the negative modulus shows, which
# is the useful signal that the estimate has left its range.
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
    rve = RVE(:SOLID)
    add_matrix!(rve, Spheroid(1.0), Dict(:C => iso_stiffness(72.0, 32.0)))
    add_phase!(
        rve, :PORE, Spheroid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
        fraction = 0.1
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const φs = range(0.0, 1.0; length = 101)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_rve()

function evaluate(scheme, φ)
    cell = set_param(base_cell, amount(:PORE), φ)
    C = homogenize(cell, scheme, :C)
    C = best_fit_iso(C)
    km = k_mu(C)
    return (max(km[1], 0.0), max(km[2], 0.0))
end

const SCHEMES = [
    ("Voigt", Voigt()),
    ("Reuss", Reuss()),
    ("Dilute", Dilute()),
    ("DiluteDual", DiluteDual()),
    ("MoriTanaka", MoriTanaka()),
    ("Maxwell", Maxwell()),
    ("PonteCastanedaWillis", PonteCastanedaWillis()),
    ("SelfConsistent", SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)),
    ("AsymmetricSelfConsistent", AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)),
    ("DifferentialScheme", DifferentialScheme(; nsteps = 300)),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, φ) for φ in φs]
    RESULTS["$(name) k"] = [r[1] for r in rows]
    RESULTS["$(name) mu"] = [r[2] for r in rows]
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
 "description": "Spherical pores in an isotropic solid, under every scheme at once. Voigt and Reuss bracket; the self-consistent schemes vanish at f = 1/2; Mori-Tanaka, Maxwell and PCW never do.",
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
  "clamp_zero": true,
  "enabled": true,
  "length": 101,
  "lens": {
   "field_name": "semi_axes",
   "index": 1,
   "inner": null,
   "kind": "amount",
   "member": "",
   "phase": "PORE",
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
    "name": "Voigt",
    "options": {}
   },
   {
    "name": "Reuss",
    "options": {}
   },
   {
    "name": "Dilute",
    "options": {}
   },
   {
    "name": "DiluteDual",
    "options": {}
   },
   {
    "name": "MoriTanaka",
    "options": {}
   },
   {
    "name": "Maxwell",
    "options": {}
   },
   {
    "name": "PonteCastanedaWillis",
    "options": {}
   },
   {
    "name": "SelfConsistent",
    "options": {
     "abstol": 1e-10,
     "maxiters": 300,
     "select_best": true
    }
   },
   {
    "name": "AsymmetricSelfConsistent",
    "options": {
     "abstol": 1e-10,
     "maxiters": 300,
     "select_best": true
    }
   },
   {
    "name": "DifferentialScheme",
    "options": {
     "nsteps": 300
    }
   }
  ],
  "start": 0.0,
  "stop": 1.0,
  "variable": "\u03c6"
 },
 "title": "porous_schemes"
}
=#
