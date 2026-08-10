"""
    MeanFieldHomogenization.Elasticity

Hill polarization tensors for ellipsoidal inclusions (2D / 3D,
isotropic / anisotropic matrix).  Public entry point:
[`hill_tensor`](@ref).
"""
module Elasticity

using LinearAlgebra
using TensND
using QuadGK
using ..Elliptic
using Polynomials
using PolynomialRoots

import ..Core
using ..Core
const MFH_Core = Core

# Re-export the abstract inclusion supertype so that users can write
# `AbstractEllipsoidalInclusion` directly through `MeanFieldHomogenization`.
import ..Core: AbstractEllipsoidalInclusion

include("ellipsoid.jl")
include("cylinder.jl")
include("auxiliary_tensors.jl")
include("hill_3d_iso.jl")
include("hill_3d_cylinder_iso.jl")
include("hill_3d_ti_coaxial.jl")
include("hill_3d_aniso_residue.jl")
include("hill_3d_aniso_nestedquadgk.jl")
include("hill_3d_aniso_decuhr.jl")
include("hill_3d_cylinder_aniso.jl")
include("hill_2d_iso.jl")
include("hill_2d_aniso.jl")
include("surface_stiffness.jl")
include("api.jl")
include("param_conversions.jl")

# ── TI-coaxial dispatch refinement ──────────────────────────────────────────
# Inject specialized resolution for TI matrix + coaxial spheroid; falls back
# to the generic residue/DECUHR branch otherwise.  Defined here (after the
# inclusion types and `_ti_coaxial` are visible) to avoid Core→Elasticity
# circular dependency.

if isdefined(TensND, :TensTI)
    @eval function _ti_dispatch(method::Symbol, ell::Ellipsoid{3}, C₀::TensND.TensTI{4})
        if (method === :auto || method === :analytical) && _ti_coaxial(C₀, ell)
            return MFH_Core.Analytical()
        end
        method === :decuhr && return MFH_Core.DECUHR()
        method === :nestedquadgk && return MFH_Core.NestedQuadGK()
        method === :residues && return MFH_Core.Residue()
        # `:auto`, non-coaxial : hand over to the shared anisotropic default,
        # which picks a cubature (DECUHR when its extension is loaded, else the
        # type-generic NestedQuadGK) and never the residue path — see
        # `Core/dispatch.jl` `_aniso_default_algo` for why.
        return MFH_Core._aniso_default_algo(C₀)
    end
    @eval MFH_Core._resolve_algo(::Val{:auto}, ell::Ellipsoid{3}, C₀::TensND.TensTI{4}) =
        _ti_dispatch(:auto, ell, C₀)
    @eval MFH_Core._resolve_algo(::Val{:analytical}, ell::Ellipsoid{3}, C₀::TensND.TensTI{4}) =
        _ti_dispatch(:analytical, ell, C₀)
    @eval MFH_Core._resolve_algo(::Val{:residues}, ell::Ellipsoid{3}, C₀::TensND.TensTI{4}) =
        _ti_dispatch(:residues, ell, C₀)
    @eval MFH_Core._resolve_algo(::Val{:decuhr}, ell::Ellipsoid{3}, C₀::TensND.TensTI{4}) =
        _ti_dispatch(:decuhr, ell, C₀)
    @eval MFH_Core._resolve_algo(::Val{:nestedquadgk}, ell::Ellipsoid{3}, C₀::TensND.TensTI{4}) =
        _ti_dispatch(:nestedquadgk, ell, C₀)
end

export Ellipsoid, Spheroid
export EllipsoidShape, Spherical, Prolate, Oblate, Triaxial, Circular, Elliptic
export Cylinder, CylindricalShape, CircularCylindrical, EllipticCylindrical
export tens_IA, tens_UA, tens_VA
export hill_tensor
export surface_stiffness, equivalent_particle
export k_mu, iso_stiffness, E_nu, iso_stiffness_E_nu
export hoenig_params, hoenig_stiffness

end # module
