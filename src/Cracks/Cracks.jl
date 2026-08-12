"""
    MeanFieldHomogenization.Cracks

COD tensors, compliance contributions, SIF and DIF for flat cracks
embedded in an elastic matrix of arbitrary anisotropy.  Public entry
points: [`cod_tensor`](@ref), [`compliance_contribution`](@ref),
[`sif`](@ref), [`dif`](@ref).
"""
module Cracks

using LinearAlgebra
using StaticArrays
using TensND
using Tensors
using QuadGK
using ..Elliptic
using Polynomials
using PolynomialRoots

import ..Core
using ..Core
# `ConductiveCrack` takes the ω → 0 limit of the spheroid contribution for an
# anisotropic reference medium, which needs the ellipsoid type.
import ..Elasticity
const MFH_Core = Core

# `compliance_contribution` / `delta_compliance` / `delta_resistivity` are
# Core-level generics (like `stiffness_contribution` & co.) so that *every*
# inclusion family — including user-defined ones living outside this
# sub-module — extends one single canonical function.  The crack methods
# below therefore *extend*, they do not declare.
import ..Core: compliance_contribution, delta_compliance, delta_resistivity,
    compliance_and_stiffness_contribution

include("geometry.jl")
include("cod_H_bridge.jl")
include("cod_analytical.jl")
include("cod_analytical_thermal.jl")
include("green_residue.jl")
include("green_nestedquadgk.jl")
include("green_decuhr.jl")
include("cod_numerical.jl")
include("compliance.jl")
include("conductive.jl")
include("sif.jl")
include("api.jl")

# ── Geometry ─────────────────────────────────────────────────────────────────
export CrackShape, Penny, EllipticShape, Ribbon
export ConductiveCrack, fracture_conductivity, with_conductivity
export EllipticCrack, RibbonCrack
export PennyCrack
export crack_basis, aspect_ratio, semi_major, semi_minor, crack_normal

# ── COD / compliance ─────────────────────────────────────────────────────────
export cod_tensor, B_tensor
export cod_from_compliance, compliance_from_cod
export crack_density_factor
# Internal perf seam consumed by `Schemes` (not re-exported by MeanFieldHomogenization).
export compliance_and_stiffness_contribution
# (`compliance_contribution`, `delta_compliance`, `delta_resistivity`,
# `stiffness_contribution`, `conductivity_contribution`, `delta_stiffness`
# and `delta_conductivity` are Core-level generics; the methods defined in
# `compliance.jl` / `api.jl` attach to them and are visible through Core's
# exports — nothing to re-export here.)

# ── SIF / DIF ────────────────────────────────────────────────────────────────
export sif, dif

end # module
