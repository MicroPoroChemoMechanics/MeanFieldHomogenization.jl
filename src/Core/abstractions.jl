# =============================================================================
#  abstractions.jl
#
#  Unified inclusion-type hierarchy used throughout `MeanFieldHomogenization`.
#
#  Every inclusion geometry (ellipsoid, crack, multi-layer, user-defined, …)
#  subtypes `AbstractInclusion{T}`, where `T` is the element type of the
#  geometric scalars (semi-axes, half-widths, radii, …).  Sub-hierarchies
#  organize the dispatch tables at the next level:
#
#     AbstractInclusion{T}
#       ├── AbstractEllipsoidalInclusion{dim,T}   — ellipsoids (2D / 3D)
#       ├── AbstractCrack{T}                      — flat cracks (3D)
#       ├── AbstractLayeredInclusion{dim,T}       — multi-layer (scaffold)
#       └── AbstractCustomInclusion{T}            — user-supplied morphologies
#
#  Of the accessors below, only `shape_trait` is actually *read* by a kernel
#  (it keys the crack algebra in `Cracks/compliance.jl`); `dimension` and
#  `inclusion_basis` are introspection accessors expected by convention, and
#  `shape_tensor` is optional — it describes an equivalent ellipsoidal
#  envelope, which not every morphology has.  See
#  `docs/src/developer/adding_inclusion.md` for the full leveled contract.
#
#  They are deliberately declared here as *stub* `function` definitions (no
#  methods) so that sub-modules can add their own methods without
#  ambiguities — the sub-module always does
#
#      import ..Core: dimension, inclusion_basis, shape_trait
#      Core.dimension(::MyInclusion) = …
#
#  (`element_type` has a single generic fallback — see below.)
# =============================================================================

"""
    AbstractInclusion{T<:Number}

Root abstract supertype for every inclusion geometry recognized by
`MeanFieldHomogenization`.  The type parameter `T` is the element type of the
geometric scalars (semi-axes, half-widths, …) and propagates through
every tensor produced by the package, supporting `Float64`,
`ForwardDiff.Dual`, `SymPyPythonCall.Sym`, `Symbolics.Num`, …
"""
abstract type AbstractInclusion{T <: Number} end

"""
    AbstractEllipsoidalInclusion{dim,T} <: AbstractInclusion{T}

Supertype for ellipsoidal inclusions — solid ellipsoids (and their
degenerate limits: spheres, cylinders, discs …).  The first type
parameter `dim` encodes the spatial dimension (2 or 3).
"""
abstract type AbstractEllipsoidalInclusion{dim, T} <: AbstractInclusion{T} end

"""
    AbstractCrack{T} <: AbstractInclusion{T}

Supertype for flat-crack geometries (elliptic / ribbon / penny).  Cracks
always live in 3D physical space — the spatial dimension of the geometry
(= 2 in the crack plane) is not exposed in the type, as downstream
algorithms uniformly operate on the 3D stiffness tensor.
"""
abstract type AbstractCrack{T} <: AbstractInclusion{T} end

"""
    AbstractLayeredInclusion{dim,T} <: AbstractInclusion{T}

Supertype for multi-layer inclusions.  The concrete `LayeredSphere`
(concentric isotropic shells, Hervé-Zaoui recurrences for bulk, shear and
conductivity, with five interface types) is shipped in the `LayeredSpheres`
sub-module and extended to the aging-viscoelastic setting in
`Viscoelasticity/layered_alv.jl`.  Open extensions (coated cylinders,
anisotropic layers, excentered spheres) are tracked in
`docs/src/developer/roadmap.md`.
"""
abstract type AbstractLayeredInclusion{dim, T} <: AbstractInclusion{T} end

"""
    AbstractCustomInclusion{T} <: AbstractInclusion{T}

Supertype for **user-supplied inclusion morphologies** that do not fit any of
the built-in families — non-ellipsoidal shapes, patterns whose response is
obtained from an external solver (finite elements, series expansion, …), or
hand-crafted approximate formulas.  It is the `MeanFieldHomogenization` counterpart of
the `user_inclusion` extension point of the C++/Python *echoes* codebase.

Subtyping this abstract type is *not* mandatory — an inclusion may equally
well subtype [`AbstractEllipsoidalInclusion`](@ref), [`AbstractCrack`](@ref)
or [`AbstractLayeredInclusion`](@ref) when it genuinely belongs to that
family, and will then inherit that family's dispatch rules.  What this branch
buys is a neutral home for morphologies that belong to none of them: it is
disjoint from the other three, so adding methods for it can never create a
dispatch ambiguity, and [`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor)
accepts it directly (the ellipsoid-typed entry point does not).

See the developer page *Adding a new inclusion* for the full interface
contract, and [`CustomInclusion`](@ref man-custom-inclusions) for a
ready-made concrete type driven by callbacks.
"""
abstract type AbstractCustomInclusion{T} <: AbstractInclusion{T} end

# ─── Minimal interface ───────────────────────────────────────────────────────

"""
    dimension(incl::AbstractInclusion) -> Int

Spatial dimension of the inclusion's ambient space (2 or 3 for the
concrete inclusions shipped with the package).
"""
function dimension end

"""
    element_type(incl::AbstractInclusion{T}) -> Type{T}

Element type of the geometric scalars stored in the inclusion
(`Float64`, `ForwardDiff.Dual`, `SymPyPythonCall.Sym`, …).
"""
element_type(::AbstractInclusion{T}) where {T} = T

"""
    inclusion_basis(incl::AbstractInclusion) -> TensND.AbstractBasis

Local principal basis of the inclusion (principal frame for an
ellipsoid, ``(\\underline{\\ell}, \\underline{m}, \\underline{n})`` for a crack, …).  Used by
downstream algorithms to rotate the matrix stiffness / conductivity
into the inclusion frame.
"""
function inclusion_basis end

"""
    shape_trait(incl::AbstractInclusion) -> Type

Concrete shape classification of the inclusion, used as a type
parameter for Holy-style dispatch in the downstream kernels.  Typical
values: `Spherical`, `Prolate`, `Oblate`, `Triaxial`, `Circular`,
`Elliptic` (ellipsoids), `Penny`, `EllipticShape`, `Ribbon` (cracks).
"""
function shape_trait end

"""
    shape_tensor(incl::AbstractInclusion) -> AbstractTens{2}

Symmetric 2nd-order tensor encoding both the semi-axes and the
orientation of an **equivalent ellipsoidal envelope** of the inclusion, in the
global (canonical) frame.

!!! note "Optional"
    No kernel in the package reads it: an inclusion that supplies its own
    response tensors owes nothing about its outer shape, and a morphology with
    no ellipsoidal envelope simply has no `shape_tensor`. Implement it when
    the notion is meaningful.

```math
\\boldsymbol{A} = \\boldsymbol{R}\\cdot\\mathrm{diag}(a_1, a_2, \\dots)\\cdot\\boldsymbol{R}^{\\!T}
```

where ``\\boldsymbol{R}`` is the rotation mapping the canonical frame
onto the inclusion's local basis and the diagonal entries are the
semi-axes in the order dictated by the local basis.

Conventions for degenerate cases:

| Inclusion               | Diagonal (principal frame)      |
| ----------------------- | ------------------------------- |
| `Ellipsoid{3}`          | ``(a_1, a_2, a_3)``             |
| `Ellipsoid{2}`          | ``(a_1, a_2)``                  |
| `Cylinder`              | ``(\\infty, b, c)`` — axis ``\\underline{e}_1`` |
| `EllipticCrack`         | ``(a, b, 0)`` — normal ``\\underline{e}_3``     |
| `RibbonCrack`           | ``(\\infty, b, 0)``             |
"""
function shape_tensor end

"""
    eshelby_tensor(incl, C₀; method=:auto, abstol, reltol, maxiters) -> AbstractTens

Eshelby tensor of the inclusion `incl` embedded in a matrix of
stiffness / conductivity `C₀`, derived from the Hill polarization
tensor ``\\mathbb{P}`` (or ``\\boldsymbol{P}``) by the relations

```math
\\mathbb{S}^{\\mathrm{E}} = \\mathbb{P}:\\mathbb{C}_0
```

in elasticity (order 4), and

```math
\\boldsymbol{S}^{\\mathrm{E}} = \\boldsymbol{P}\\cdot\\boldsymbol{K}_0
```

in conductivity / diffusion (order 2).

The appropriate method is selected by dispatch on the order of `C₀`:
an `AbstractTens{4, 3}` (elasticity) triggers the double contraction
``\\mathbb{P}:\\mathbb{C}_0``, while an
`AbstractTens{2, 3}` (conductivity) triggers the simple contraction
``\\boldsymbol{P}\\cdot\\boldsymbol{K}_0``.

All keyword arguments (`method`, `abstol`, `reltol`, `maxiters`) are
forwarded verbatim to [`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor); see its docstring for the
set of admissible algorithm traits.

See also [`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor).
"""
function eshelby_tensor end

# =============================================================================
#  Concentration & contribution public API stubs
#
#  Declared here so that every sub-module (Elasticity, Cracks,
#  Conductivity, LayeredSphere, user extensions) can add methods via
#  `import ..Core: property_contribution` + method definition, all
#  attaching to a single generic function.  Definitions live in
#  `src/concentration.jl` and `src/contribution.jl` (loaded at
#  MeanFieldHomogenization top level after all sub-modules).
# =============================================================================

"""
    is_homogeneous_inclusion(incl) -> Bool

Whether the inclusion carries a **single uniform property**, so that the
mean-field identities of a homogeneous inhomogeneity apply:

```math
\\langle\\mathbb{C}:\\boldsymbol{\\varepsilon}\\rangle_i = \\mathbb{C}_i:\\mathbb{A}_i ,
\\qquad
\\mathbb{N}_i = (\\mathbb{C}_i - \\mathbb{C}_0):\\mathbb{A}_i .
```

`true` for every ellipsoid, cylinder and crack. `false` for internally
heterogeneous inclusions such as `LayeredSphere`, whose average stress must be
assembled layer by layer — no single ``\\mathbb{C}_i`` represents them, and feeding the
phase property into the formulas above gives a wrong answer (it can even come
out with the opposite sign).

Scheme kernels must branch on this trait rather than inlining the homogeneous
formula.
"""
is_homogeneous_inclusion(::AbstractInclusion) = true

# ── Concentration and contribution: one function per notion ─────────────────
#
# Each generic serves both tensor orders: a fourth-order stiffness in elasticity,
# where the gradient is the strain and the flux the stress, and a second-order
# conductivity in conduction, where they are the gradient and the opposite of
# the flux of the potential. The methods follow the order of the property
# tensors they receive. The names of elasticity and conduction are aliases,
# each restricted to its order (`order_aliases.jl`).

"""
    gradient_gradient_concentration(incl, P₁, P₀; kw...) -> AbstractTens

Dilute gradient concentration tensor of an inclusion of property `P₁` in an
infinite matrix of property `P₀`: the mean gradient in the inclusion per unit
remote gradient. With fourth-order stiffnesses it is the strain concentration
tensor ``\\mathbb{A}_{\\varepsilon\\varepsilon}`` of Eshelby's problem; with
second-order conductivities, the gradient concentration tensor
``\\boldsymbol{A}``. Explicit name in elasticity:
[`strain_strain_concentration`](@ref).
"""
function gradient_gradient_concentration end

"""
    flux_gradient_concentration(incl, P₁, P₀; kw...) -> AbstractTens

Dilute flux concentration tensor per unit remote gradient: the mean stress in
the inclusion per unit remote strain, ``\\mathbb{A}_{\\sigma\\varepsilon}``, in
elasticity; the mean opposite of the flux per unit remote gradient in
conduction. Explicit name in elasticity: [`stress_strain_concentration`](@ref).
"""
function flux_gradient_concentration end

"""
    gradient_flux_concentration(incl, P₁, P₀; kw...) -> AbstractTens

Dilute gradient concentration tensor per unit remote flux: the mean strain per
unit remote stress, ``\\mathbb{A}_{\\varepsilon\\sigma}``, in elasticity. Explicit
name in elasticity: [`strain_stress_concentration`](@ref).
"""
function gradient_flux_concentration end

"""
    flux_flux_concentration(incl, P₁, P₀; kw...) -> AbstractTens

Dilute flux concentration tensor per unit remote flux: the stress concentration
tensor ``\\mathbb{A}_{\\sigma\\sigma}`` in elasticity, its analog for the flux in
conduction. Explicit name in elasticity: [`stress_stress_concentration`](@ref).
"""
function flux_flux_concentration end

"""
    property_contribution(incl, P₁, P₀; kw...) -> AbstractTens
    property_contribution(crack, P₀; kw...)     -> AbstractTens

Size-independent contribution of an inclusion to the property tensor of a
matrix `P₀`: the stiffness contribution ``\\mathbb{N}`` in elasticity, the
conductivity contribution ``\\boldsymbol{N}`` in conduction. For a dilute family
of volume fraction ``f``, the effective property is corrected by ``f\\,\\mathbb{N}``
(see [`delta_property`](@ref)). Explicit names: [`stiffness_contribution`](@ref)
and [`conductivity_contribution`](@ref).
"""
function property_contribution end

"""
    inverse_property_contribution(incl, P₁, P₀; kw...) -> AbstractTens
    inverse_property_contribution(incl, P₀; kw...)     -> AbstractTens

Size-independent contribution of an inclusion to the inverse of the property
tensor of a matrix `P₀`: the compliance contribution ``\\mathbb{H}`` in elasticity,
the resistivity contribution ``\\boldsymbol{R}`` in conduction. For a dilute family
of volume fraction ``f``, the effective inverse property is corrected by
``f\\,\\mathbb{H}`` (see [`delta_inverse_property`](@ref)).

The **two-argument** form is the *flat-object* flavor used by cracks and, more
generally, by any inclusion registered in an
[`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) with a
[`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) amount: the
inclusion carries no property of its own, and the amount is applied afterwards
by the three-argument [`delta_inverse_property`](@ref).

Methods for the built-in cracks live in `Cracks/compliance.jl`; the generic
three-argument methods for solid inclusions live in `src/contribution.jl`.
Explicit names: [`compliance_contribution`](@ref) and
[`resistivity_contribution`](@ref).
"""
function inverse_property_contribution end

"""
    delta_property(N, f)        -> AbstractTens
    delta_property(incl, N, ε)  -> AbstractTens

Dilute correction of the effective property, ``f\\,\\mathbb{N}``, from the
size-independent contribution ``\\mathbb{N}`` (stiffness or conductivity) and the
volume fraction ``f``; the three-argument form applies a density-like amount
``\\varepsilon`` with the geometric prefactor of the inclusion, as for
[`delta_inverse_property`](@ref). Explicit names: [`delta_stiffness`](@ref) and
[`delta_conductivity`](@ref).
"""
function delta_property end

"""
    delta_inverse_property(H, f)        -> AbstractTens
    delta_inverse_property(incl, H, ε)  -> AbstractTens

Dilute correction of the effective inverse property from the size-independent
contribution ``\\mathbb{H}`` (compliance or resistivity).

The **two-argument** form is the volume-fraction one, ``f\\,\\mathbb{H}``. The
**three-argument** form is the *amount × contribution* seam of flat objects: it
carries the geometric prefactor relating a density-like amount to the effective
correction (``4\\pi/3`` for an elliptical crack of Budiansky density
``\\varepsilon^{3\\mathrm{d}} = N\\,a\\,b^{2}``, ``\\pi`` for a ribbon crack of
``\\varepsilon^{2\\mathrm{d}} = N\\,b^{2}``). Every inclusion meant to be registered
with a [`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) amount
must provide the three-argument methods of `delta_inverse_property` and
[`delta_property`](@ref). Explicit names: [`delta_compliance`](@ref) and
[`delta_resistivity`](@ref).
"""
function delta_inverse_property end

"""
    concentration_and_contribution(incl, P₁, P₀; kw...) -> (A, N)

Bundled evaluation of the dilute concentration tensor `A`
([`gradient_gradient_concentration`](@ref)) **and** the size-independent
contribution `N` ([`property_contribution`](@ref)) of the *same* inclusion in the
*same* reference medium, sharing the single expensive Hill / recurrence solve.

Both quantities are needed per phase by Mori-Tanaka and by the
self-consistent kernels; computing them separately evaluates
[`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor) — the dominant cost — twice with byte-identical
arguments.

!!! note "Contract"
    The returned pair must be **bitwise identical** to
    `(gradient_gradient_concentration(incl, P₁, P₀; kw...),
      property_contribution(incl, P₁, P₀; kw...))`.
    The generic fallback *is* that pair; specializations are a pure
    performance concern and must not reassociate the arithmetic.

Internal seam — not exported.
"""
function concentration_and_contribution end

"""
    concentration_and_flux_average(incl, P₁, P₀; kw...) -> (A, B)

Bundled `(gradient_gradient_concentration, flux_gradient_concentration)`, sharing
one concentration solve. Same bitwise contract as
[`concentration_and_contribution`](@ref).

Internal seam — not exported.
"""
function concentration_and_flux_average end

"""
    inverse_and_property_contribution(incl, P₀; kw...) -> (H, N)

Bundled two-argument contribution pair of a **flat** inclusion — the
density-amount counterpart of [`concentration_and_contribution`](@ref), sharing
whatever single expensive solve produces both (for a crack, one
[`cod_tensor`](@ref MeanFieldHomogenization.Cracks.cod_tensor)): the pair
`(inverse_property_contribution, property_contribution)`. Same bitwise
contract as [`concentration_and_contribution`](@ref): the generic fallback in
`src/contribution.jl` *is* that pair, and any specialization is a pure
performance concern.

Internal seam — not exported.
"""
function inverse_and_property_contribution end
