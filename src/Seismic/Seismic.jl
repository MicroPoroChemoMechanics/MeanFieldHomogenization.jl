"""
    MeanFieldHomogenization.Seismic

What the waves see of a homogenized cell: its mass density, the compliances of
its fracture sets, and, through the elastic waves of TensND, its velocities,
anisotropy parameters and splitting.

The sub-module is a *post-processor*: it reads a cell and a scheme, or a
homogenized stiffness, and returns the quantities of seismology. The waves
themselves (Christoffel problem, Thomsen's parameters, interfaces) are
TensND's; nothing of them is re-implemented here.

# Entry points

| Function | Returns |
|---|---|
| [`average_density`](@ref) | ``\\bar\\rho = \\sum_i f_i\\,\\rho_i`` over the phases carrying volume |
| [`fracture_compliance`](@ref) | the compliance of linear slip ``\\boldsymbol{Z}``, ``Z_N``, ``Z_T`` of a crack family, scheme by scheme |
| [`fractured_stiffness`](@ref) | a background crossed by sets of fractures declared by ``(\\underline{n}, \\boldsymbol{Z})`` |
| [`linear_slip_change`](@ref) | ``\\Delta\\mathbb{S} = \\operatorname{sym}(\\underline{n}\\otimes\\boldsymbol{Z}\\otimes\\underline{n})`` |
| [`slip_compliance`](@ref) | ``\\boldsymbol{Z}`` from a change of compliance of the form ``\\operatorname{sym}(\\underline{n}\\otimes\\boldsymbol{Z}\\otimes\\underline{n})`` |
| [`orthogonal_fracture_compliances`](@ref) | ``Z_N^i``, ``Z_T^i`` of three orthogonal sets read on a change of compliance |
"""
module Seismic

using TensND
using TensND: Tensors
using TensND.Tensors: Vec, Tensor, SymmetricTensor, otimes, otimesl, dotdot, dcontract, minorsymmetric
using StaticArrays: SVector

import ..Schemes
using LinearAlgebra: tr
using ..Schemes: RVE, Voigt, HomogenizationScheme, homogenize, crack_family_compliances, crack_density
using ..Core: delta_inverse_property
using ..Cracks: crack_normal

include("density.jl")
include("fracture_compliance.jl")
include("fracture_sets.jl")

export average_density, slip_compliance, fracture_compliance, linear_slip_change, fractured_stiffness,
    orthogonal_fracture_compliances

end
