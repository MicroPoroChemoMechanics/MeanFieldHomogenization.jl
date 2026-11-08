# =============================================================================
#  media.jl — what fills a Brinkman inclusion, and which property the phase
#  carries. Both are types, so that the limits k = 0 and k = ∞ are selected by
#  dispatch rather than by testing a stored infinity.
# =============================================================================

"""
    BrinkmanMedium

What fills a Brinkman inclusion: [`BrinkmanPorous`](@ref) (a permeability
``0 < k_1 < \\infty`` read from the phase property), [`BrinkmanFluid`](@ref)
(``k_1 = \\infty``, Stokes flow) or [`BrinkmanSolid`](@ref) (``k_1 = 0``).
"""
abstract type BrinkmanMedium end

"""
    BrinkmanPorous()

A porous inclusion, of the permeability read from the phase property. The
inclusion is homogeneous: the schemes form its average flux from that property.
"""
struct BrinkmanPorous <: BrinkmanMedium end

"""
    BrinkmanFluid()

An inclusion of clear fluid (``k_1 = \\infty``), in which the flow obeys the
Stokes equations. The phase property is not read.
"""
struct BrinkmanFluid <: BrinkmanMedium end

"""
    BrinkmanSolid()

An impervious inclusion (``k_1 = 0``): the velocity vanishes in it and the
fluid exerts a drag on it. The phase property is not read.
"""
struct BrinkmanSolid <: BrinkmanMedium end

"""
    BrinkmanForm

Which property the phases of the cell carry:

- [`ResistanceForm`](@ref) — the resistance ``\\boldsymbol{r} = \\boldsymbol{k}^{-1}``.
  The velocity plays the role of the gradient and the drag density ``-\\underline{f}/\\mu``
  that of the flux: ``-\\underline{f}/\\mu = \\boldsymbol{r}\\cdot\\underline{v}``.
  The reference must be finite; a clear-fluid matrix is ``\\boldsymbol{r}_0 = \\boldsymbol{0}``.
- [`PermeabilityForm`](@ref) — the permeability ``\\boldsymbol{k}``. The drag density plays
  the role of the gradient and the velocity that of the flux. An impervious
  matrix is ``\\boldsymbol{k}_0 = \\boldsymbol{0}``.

Both describe the same flow: the concentration tensors of one follow from those
of the other by ``\\boldsymbol{k}\\cdot\\boldsymbol{B} = \\boldsymbol{A}\\cdot\\boldsymbol{k}_0`` [bignonnetIJES2022](@cite).
"""
abstract type BrinkmanForm end

"""
    ResistanceForm()

The phase property is the resistance ``\\boldsymbol{r} = \\boldsymbol{k}^{-1}``; see [`BrinkmanForm`](@ref).
"""
struct ResistanceForm <: BrinkmanForm end

"""
    PermeabilityForm()

The phase property is the permeability ``\\boldsymbol{k}``; see [`BrinkmanForm`](@ref).
"""
struct PermeabilityForm <: BrinkmanForm end

"""
    AbstractBrinkmanInclusion{T} <: AbstractCustomInclusion{T}

Supertype of the Brinkman inclusions. Each concrete type stores its size, its
[`BrinkmanMedium`](@ref) and its [`BrinkmanForm`](@ref), and supplies the
concentration tensors of the Brinkman inhomogeneity problem.
"""
abstract type AbstractBrinkmanInclusion{T} <: MFH_Core.AbstractCustomInclusion{T} end

"Neutral shape tag of the Brinkman inclusions: no kernel dispatches on it."
struct BrinkmanShape end

MFH_Core.dimension(::AbstractBrinkmanInclusion) = 3
MFH_Core.shape_trait(::AbstractBrinkmanInclusion) = BrinkmanShape

# A porous inclusion has a single property, which the schemes read; a solid or a
# fluid one has none that can be multiplied (0·∞), so it supplies its average
# flux itself.
MFH_Core.is_homogeneous_inclusion(i::AbstractBrinkmanInclusion) = _homogeneous(_medium(i))
_homogeneous(::BrinkmanPorous) = true
_homogeneous(::BrinkmanMedium) = false
