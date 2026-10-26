"""
    MeanFieldHomogenization.MaterialPoints

**Constitutive laws at a material point**: writing a nonlinear law in a few
lines, integrating it over a step with its consistent tangent, driving it along
a loading path, and plugging it into any caller of the Gauss-point contract, a
finite-element code included.

# Four independent pieces

| piece | what it says | how |
|:--|:--|:--|
| the **law** | *what* the material is | [`ResidualLaw`](@ref) (local equations) or [`PotentialLaw`](@ref) (free energy and dissipation), parameters in `p` |
| the **local algorithm** | *how* a step is integrated | [`LocalNewton`](@ref), with the tangent by the implicit-function theorem |
| the **loading** | *along which path* | [`LoadingPath`](@ref), each component driven in strain or in stress |
| the **coupling** | *where* the law is evaluated | [`point_response`](@ref) at one point, [`material_response`](@ref) in an element loop |

Any law runs with any algorithm, along any path, in any caller. A law is a
[`AbstractMFHMaterial`](@ref) like a homogenized material, so the coupling code
of the package serves both.

```julia
using MeanFieldHomogenization

law = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 2e3)
path = LoadingPath(range(0, 1; length = 101); ε11 = t -> 5e-3 * t)   # uniaxial tension
r = point_response(law, path)
r[:σ11], r[:p]                        # stress and cumulated plastic strain
```

# Numbers

Strains and stresses are [`Mandel`](@ref) vectors (`SVector{6}`), tangents
`SMatrix{6, 6}`: a step allocates nothing. Every number may be a
`ForwardDiff.Dual`, nested or not, in the strain, the state or the parameters:
the response then carries its derivatives, exact (no finite differences). The
small-strain kinematics ([`SmallStrain`](@ref)) is the only one so far.
"""
module MaterialPoints

using LinearAlgebra
using StaticArrays
using ForwardDiff
import TensND
import Tensors

import ..Constitutive
import ..Constitutive: AbstractMFHMaterial, AbstractMaterialState, MaterialResponse,
    initial_state, material_response, to_tensors

include("kinematics.jl")
include("laws.jl")
include("algorithms.jl")
include("contract.jl")
include("library.jl")
include("crystal.jl")
include("loading.jl")
include("driver.jl")

# ── Writing a law ────────────────────────────────────────────────────────────
export AbstractPointLaw, ResidualLaw, PotentialLaw, PointState
export kinematics, SmallStrain, state_layout, gradient_layout, flux_layout, law_parameters
export Mandel, identity_mandel, trace_mandel, deviator, von_mises, equivalent_strain

# ── Integrating it ───────────────────────────────────────────────────────────
export AbstractLocalAlgorithm, LocalNewton, LocalIntegrationError, point_update

# ── The laws shipped ─────────────────────────────────────────────────────────
export LinearElasticity, PowerLawElasticity, MaxwellViscoelasticity, BiotPoroelasticity
export NortonViscoplasticity, J2Plasticity, CrystalViscoplasticity, slip_systems

# ── Driving a point ──────────────────────────────────────────────────────────
export LoadingPath, point_response, PointResponse, PointDriverError

end # module
