"""
    MeanFieldHomogenization.Brinkman

**Brinkman inhomogeneities** for the Stokes-to-Darcy homogenization of the
permeability [bignonnetIJES2022](@cite).

At the scale of the pores, a slow incompressible flow obeys the Stokes
equations in the fluid and vanishes in the solid; at the scale of the material,
it obeys Darcy's law. The permeability is estimated with the schemes of
conduction (dilute, Mori–Tanaka, self-consistent, differential, Maxwell), whose
auxiliary problem is no longer Eshelby's: one inclusion of size ``R``, of
permeability ``k_1`` (``0`` for a solid, ``\\infty`` for a clear fluid), embedded
in an unbounded medium of permeability ``k_0`` in which the flow obeys
Brinkman's equation

```math
-\\nabla p + \\mu\\,\\Delta\\underline{v} - \\mu\\,k^{-1}\\,\\underline{v} = \\underline{0},
\\qquad \\nabla\\cdot\\underline{v} = 0 .
```

Its solution depends on ``R`` through ``\\alpha_0 = R/\\sqrt{k_0}``: there is no Hill
tensor, and the inclusions below supply their concentration tensors directly
(level B of the [custom-inclusion interface](@ref man-custom-inclusions)).

Exports:

- the inclusions [`BrinkmanSphere`](@ref), [`BrinkmanCylinder`](@ref),
  [`BrinkmanLayer`](@ref) and [`BrinkmanCoatedSphere`](@ref);
- the media [`BrinkmanPorous`](@ref), [`BrinkmanFluid`](@ref) and
  [`BrinkmanSolid`](@ref);
- the formulations [`ResistanceForm`](@ref) and [`PermeabilityForm`](@ref),
  which say whether the phase property is the resistance ``r = k^{-1}`` or the
  permeability ``k``.
"""
module Brinkman

using SpecialFunctions: besselkx, besselix, besselk, besseli
using TensND

import ..Core
const MFH_Core = Core
import ..Schemes
using ..Elliptic: is_hard_numeric

export BrinkmanMedium, BrinkmanPorous, BrinkmanFluid, BrinkmanSolid
export BrinkmanForm, ResistanceForm, PermeabilityForm
export AbstractBrinkmanInclusion, BrinkmanSphere, BrinkmanCylinder, BrinkmanLayer, BrinkmanCoatedSphere

include("media.jl")
include("kernels.jl")
include("inclusions.jl")

end # module
