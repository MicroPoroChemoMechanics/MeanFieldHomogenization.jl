"""
    MeanFieldHomogenization.Poromechanics

Poroelastic upscaling: the **Biot tensor** ``\\boldsymbol{B}``, the **Biot modulus**
``M``, the drained ↔ undrained conversion and the **Skempton tensor** of a
saturated porous or fractured medium whose solid phase is homogeneous.

The sub-module is a *post-processor*, not a scheme. It takes a drained
homogenized stiffness `C_hom` — from any cell and any scheme, or from
elsewhere entirely — together with the solid stiffness `C_s` and the Lagrangian
porosity `φ`, and closes the poroelastic constitutive law

```math
\\dot{\\boldsymbol{\\Sigma}} = \\mathbb{C}^{\\rm hom} : \\dot{\\boldsymbol{E}}
- \\dot p\\,\\boldsymbol{B}, \\qquad
\\dot\\varphi = \\boldsymbol{B} : \\dot{\\boldsymbol{E}} + \\frac{\\dot p}{M},
```

using the classical poroelastic relations ([coussy2004](@cite)) in the form
quoted as eq. (2) of [barthelemyARMA2011](@cite). No additional Eshelby problem
is solved: everything follows from `C_hom`, `C_s` and `φ`.

# Entry points

| Function | Returns |
|---|---|
| [`biot_tensor`](@ref) | ``\\boldsymbol{B} = \\boldsymbol{1} : (\\mathbb{I} - \\mathbb{S}_{\\rm s} : \\mathbb{C}^{\\rm hom})`` |
| [`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus) | ``1/M = \\boldsymbol{1} : \\mathbb{S}_{\\rm s} : (\\boldsymbol{B} - \\varphi\\boldsymbol{1})`` |
| [`biot_modulus`](@ref) | ``M`` (`Inf` for an incompressible solid) |
| [`poroelastic_parameters`](@ref) | all three at once, inverting ``\\mathbb{C}_s`` once |
| [`undrained_stiffness`](@ref) / [`drained_stiffness`](@ref) | ``\\mathbb{C}^{\\rm u} = \\mathbb{C}^{\\rm hom} + M\\,\\boldsymbol{B}\\otimes\\boldsymbol{B}`` and back |
| [`skempton_tensor`](@ref) | ``\\boldsymbol{B}^{\\rm sk}`` such that ``p = -\\boldsymbol{B}^{\\rm sk}:\\boldsymbol{\\Sigma}`` undrained |
| [`terzaghi_stress`](@ref) / [`biot_effective_stress`](@ref) | the two effective-stress measures |
| [`pore_volume_fraction`](@ref) | ``\\varphi`` summed over declared pore phases |

Everything is plain TensND algebra, hence type-generic (`Float64`, `BigFloat`,
`ForwardDiff.Dual`, `ComplexF64`, `SymPy.Sym`) at no cost.

!!! note "Homogeneous solid phase"
    The relations above hold when the solid phase has *uniform* elastic
    properties — the case of a rock matrix with pores or fractures. A medium
    built from two distinct solid constituents needs the general
    Levin/eigenstrain route, and `C_s` is then not defined.
"""
module Poromechanics

using LinearAlgebra
using TensND

import ..Core
const MFH_Core = Core

import ..Schemes

include("effective_stress.jl")
include("biot.jl")
include("rve_interface.jl")

export terzaghi_stress, biot_effective_stress
export biot_tensor, inverse_biot_modulus, biot_modulus, poroelastic_parameters
export undrained_stiffness, drained_stiffness, skempton_tensor
export pore_volume_fraction

end # module
