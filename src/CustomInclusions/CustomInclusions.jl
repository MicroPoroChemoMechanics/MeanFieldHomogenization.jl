"""
    MeanFieldHomogenization.CustomInclusions

The **user-defined inclusion contract**: everything needed to plug a
morphology `MeanFieldHomogenization` knows nothing about into every homogenization
scheme, in elasticity as in transport.

Two exports:

- [`CustomInclusion`](@ref) — a concrete, callback-driven inclusion, the
  value-level analog of subtyping `AbstractCustomInclusion`;
- [`check_inclusion_interface`](@ref) — a conformance checker that works on
  *any* `AbstractInclusion`, reporting which entry gate it satisfies and what
  is missing.

The contract itself (four levels, three entry gates, the *amount × contribution*
seam) is specified in `docs/src/developer/adding_inclusion.md`; the user-facing
tutorial is `docs/src/manual/custom_inclusions.md`.

Loaded after `concentration.jl` and `contribution.jl`, because the fallbacks
here `invoke` the generic methods defined there.
"""
module CustomInclusions

using TensND

import ..Core
import ..Elasticity

export CustomShape, CustomInclusion, check_inclusion_interface

include("custom_inclusion.jl")

end # module
