# =============================================================================
#  cracked_solid.jl
#
#  Built with MFH Studio. Editing this file by hand is fine: the
#  studio reads it back and preserves anything it does not
#  recognize.
#
#  Penny cracks with a random orientation distribution, averaged exactly in the kernel. The abscissa is the crack density ε = N a³ / V, not a volume fraction.
# =============================================================================
#
# After `scripts/15_cracks_iso_interface.jl` and `scripts/86_crack_distributions.jl`.
#
# Penny cracks enter with a *density* rather than a volume fraction — a flat
# crack has none — and an isotropic orientation average turns a family of
# parallel cracks into an isotropic damaged solid. Remove the average and the
# result is transversely isotropic, which is why `k` is not asked for here
# without a reporting projection.
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
    rve = RVE()
    add_phase!(
        rve, :SOLID, Spheroid(1.0), Dict(:C => iso_stiffness(72.0, 32.0));
        fraction = :rest
    )
    add_phase!(
        rve, :CRACKS, PennyCrack(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6));
        density = 0.1, symmetrize = IsoSymmetrize()
    )
    return rve
end

# ── Result ──────────────────────────────────────────────────────────────────
const εs = range(0.0, 0.6; length = 25)

#
# `set_param` returns a *new* cell, leaving the original intact,
# so the sweep is a pure map rather than a mutation.
const base_cell = build_rve()

function evaluate(scheme, ε)
    cell = set_param(base_cell, amount(:CRACKS), ε)
    C = homogenize(cell, scheme, :C)
    C = best_fit_iso(C)
    km = k_mu(C)
    return (km[1], km[2])
end

const SCHEMES = [
    ("MoriTanaka", MoriTanaka()),
    ("SelfConsistent", SelfConsistent()),
]

const RESULTS = Dict{String, Any}()
for (name, scheme) in SCHEMES
    rows = [evaluate(scheme, ε) for ε in εs]
    RESULTS["$(name) k"] = [r[1] for r in rows]
    RESULTS["$(name) mu"] = [r[2] for r in rows]
end

# Published for MFH Studio; harmless when the script runs alone.
const MFHSTUDIO_RESULTS = merge(Dict("x" => collect(εs), "xlabel" => "ε"), RESULTS)

const CURVES = sort!(collect(RESULTS); by = first)

for (label, ys) in CURVES
    @printf "%-28s  first = %.6f   last = %.6f\n" label first(ys) last(ys)
end

p = plot(; xlabel = "ε", ylabel = "effective property", legend = :best)
for (label, ys) in CURVES
    plot!(p, εs, ys; label = label, lw = 2)
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
     "amount_kind": "rest",
     "geometry": {
      "args": {
       "omega": 1.0
      },
      "euler_angles": [],
      "kind": "spheroid",
      "layers": []
     },
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
     "amount_kind": "density",
     "geometry": {
      "args": {
       "a": 1.0
      },
      "euler_angles": [],
      "kind": "penny_crack",
      "layers": []
     },
     "name": "CRACKS",
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
     "symmetrize": "iso"
    }
   ],
   "rve_options": {},
   "ui": {
    "x": 40,
    "y": 40
   }
  }
 ],
 "description": "Penny cracks with a random orientation distribution, averaged exactly in the kernel. The abscissa is the crack density \u03b5 = N a\u00b3 / V, not a volume fraction.",
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
  "length": 25,
  "lens": {
   "field_name": "semi_axes",
   "index": 1,
   "inner": null,
   "kind": "amount",
   "member": "",
   "phase": "CRACKS",
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
   },
   {
    "name": "SelfConsistent",
    "options": {}
   }
  ],
  "start": 0.0,
  "stop": 0.6,
  "variable": "\u03b5"
 },
 "title": "cracked_solid"
}
=#
