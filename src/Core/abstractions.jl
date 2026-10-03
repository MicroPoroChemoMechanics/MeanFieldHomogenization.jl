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
`ForwardDiff.Dual`, `SymPy.Sym`, `Symbolics.Num`, …
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
sub-module and extended to the ageing-viscoelastic setting in
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
(`Float64`, `ForwardDiff.Dual`, `SymPy.Sym`, …).
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
#  Localization & contribution public API stubs
#
#  Declared here so that every sub-module (Elasticity, Cracks,
#  Conductivity, LayeredSphere, user extensions) can add methods via
#  `import ..Core: stiffness_contribution` + method definition, all
#  attaching to a single generic function.  Definitions live in
#  `src/localization.jl` and `src/contribution.jl` (loaded at
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

"""
    strain_strain_loc(incl, C₁, C₀; kw...)  -> Tens{4,3}

Dilute strain-strain localization tensor ``\\mathbb{A}_{\\varepsilon\\varepsilon}`` (Eshelby).
"""
function strain_strain_loc end

"""
    stress_strain_loc(incl, C₁, C₀; kw...)  -> Tens{4,3}
"""
function stress_strain_loc end

"""
    strain_stress_loc(incl, C₁, C₀; kw...)  -> Tens{4,3}
"""
function strain_stress_loc end

"""
    stress_stress_loc(incl, C₁, C₀; kw...)  -> Tens{4,3}
"""
function stress_stress_loc end

"""
    gradient_gradient_loc(incl, K₁, K₀; kw...) -> Tens{2,3}
"""
function gradient_gradient_loc end

"""
    flux_gradient_loc(incl, K₁, K₀; kw...)    -> Tens{2,3}
"""
function flux_gradient_loc end

"""
    gradient_flux_loc(incl, K₁, K₀; kw...)    -> Tens{2,3}
"""
function gradient_flux_loc end

"""
    flux_flux_loc(incl, K₁, K₀; kw...)        -> Tens{2,3}
"""
function flux_flux_loc end

"""
    stiffness_contribution(incl, C₁, C₀; kw...) -> Tens{4,3}
    stiffness_contribution(crack, C₀; kw...)   -> Tens{4,3}

Size-independent stiffness contribution tensor ``\\mathbb{N}`` of an inclusion in
a matrix `C₀`.  For a dilute family of volume fraction ``f``:
``\\Delta\\mathbb{C}^{\\mathrm{hom}} = f\\,\\mathbb{N}`` (see [`delta_stiffness`](@ref)).
"""
function stiffness_contribution end

"""
    conductivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}
    conductivity_contribution(crack, K₀; kw...)     -> Tens{2,3}

Size-independent conductivity contribution tensor for the 2nd-order
transport problem.  Analog of [`stiffness_contribution`](@ref).
"""
function conductivity_contribution end

"""
    resistivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}

Size-independent resistivity contribution tensor of an inclusion
(2nd-order analog of [`compliance_contribution`](@ref) for solid
ellipsoids).
"""
function resistivity_contribution end

"""
    compliance_contribution(incl, P₁, P₀; kw...) -> Tens
    compliance_contribution(incl, P₀; kw...)     -> Tens

Size-independent compliance contribution tensor ``\\mathbb{H}`` of an inclusion in a
matrix `P₀`.  For a dilute family of volume fraction ``f``:
``\\Delta\\mathbb{S}^{\\mathrm{hom}} = f\\,\\mathbb{H}`` (see [`delta_compliance`](@ref)).

The **two-argument** form is the *flat-object* flavor used by cracks and,
more generally, by any inclusion registered in an
[`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) with a
[`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) amount: the
inclusion carries no property of its own, and the amount is applied
afterwards by the three-argument [`delta_compliance`](@ref).

Methods for the built-in cracks live in `Cracks/compliance.jl`; the generic
three-argument method for solid inclusions lives in `src/contribution.jl`.
"""
function compliance_contribution end

"""
    delta_stiffness(N, f) -> Tens{4,3}

Dilute effective-stiffness correction ``\\Delta\\mathbb{C} = f\\,\\mathbb{N}`` from the size-
independent contribution tensor ``\\mathbb{N}`` and the volume fraction ``f``.
"""
function delta_stiffness end

"""
    delta_conductivity(N_K, f) -> Tens{2,3}

Dilute effective-conductivity correction ``\\Delta\\boldsymbol{K} = f\\,\\boldsymbol{N}``,
with ``\\boldsymbol{N}`` the conductivity contribution `N_K`.
"""
function delta_conductivity end

"""
    delta_compliance(H, f)        -> Tens
    delta_compliance(incl, H, ε)  -> Tens

Dilute effective-compliance correction from the size-independent
contribution tensor ``\\mathbb{H}``.

The **two-argument** form is the volume-fraction one, ``\\Delta\\mathbb{S} = f\\,\\mathbb{H}``.  The
**three-argument** form is the *amount × contribution* seam of flat objects:
it carries the geometric prefactor relating a density-like amount to the
effective correction (``4\\pi/3`` for an elliptical crack of Budiansky density
``\\varepsilon^{3\\mathrm{d}} = N\\,a\\,b^{2}``, ``\\pi`` for a ribbon crack of
``\\varepsilon^{2\\mathrm{d}} = N\\,b^{2}``).  Every inclusion
meant to be registered with a
[`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) amount must provide
the three-argument methods of `delta_compliance`, [`delta_stiffness`](@ref),
[`delta_conductivity`](@ref) and [`delta_resistivity`](@ref).
"""
function delta_compliance end

"""
    delta_resistivity(R, f)        -> Tens{2,3}
    delta_resistivity(incl, R, ε)  -> Tens{2,3}

Dilute effective-resistivity correction — 2nd-order analog of
[`delta_compliance`](@ref), with the same two calling conventions.
"""
function delta_resistivity end

"""
    loc_and_stiffness(incl, P₁, P₀; kw...) -> (A, N)

Bundled evaluation of the dilute concentration tensor `A`
([`strain_strain_loc`](@ref) / [`gradient_gradient_loc`](@ref)) **and** the
size-independent contribution tensor `N` ([`stiffness_contribution`](@ref) /
[`conductivity_contribution`](@ref)) of the *same* inclusion in the *same*
reference medium, sharing the single expensive Hill / recurrence solve.

Both quantities are needed per phase by Mori-Tanaka and by the
self-consistent kernels; computing them separately evaluates
[`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor) — the dominant cost — twice with byte-identical
arguments.

!!! note "Contract"
    The returned pair must be **bitwise identical** to
    `(strain_strain_loc(incl, P₁, P₀; kw...),
      stiffness_contribution(incl, P₁, P₀; kw...))`.
    The generic fallback *is* that pair; specializations are a pure
    performance concern and must not reassociate the arithmetic.

Internal seam — not exported.
"""
function loc_and_stiffness end

"""
    loc_and_stress_average(incl, P₁, P₀; kw...) -> (A, B)

Bundled `(strain_strain_loc, stress_strain_loc)` — resp.
`(gradient_gradient_loc, flux_gradient_loc)` — sharing one localization
solve.  Same bitwise contract as [`loc_and_stiffness`](@ref).

Internal seam — not exported.
"""
function loc_and_stress_average end

"""
    compliance_and_stiffness_contribution(incl, P₀; kw...) -> (H, N)

Bundled two-argument contribution pair of a **flat** inclusion — the
density-amount counterpart of [`loc_and_stiffness`](@ref), sharing whatever
single expensive solve produces both (for a crack, one
[`cod_tensor`](@ref MeanFieldHomogenization.Cracks.cod_tensor)).

Returns `(compliance_contribution, stiffness_contribution)` for a 4th-order
`P₀` and `(compliance_contribution, conductivity_contribution)` for a
2nd-order one.  Same bitwise contract as [`loc_and_stiffness`](@ref): the
generic fallback in `src/contribution.jl` *is* that pair, and any
specialization is a pure performance concern.

Internal seam — not exported.
"""
function compliance_and_stiffness_contribution end
