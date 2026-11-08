# =============================================================================
#  inclusions.jl — the four Brinkman inclusions and their concentration tensors.
#
#  In the resistance form the gradient-side tensor is the velocity
#  concentration A (⟨v⟩ = A·V₀) and the flux-side one is ⟨r·A⟩ (the average
#  drag density per unit μV₀). In the permeability form the gradient-side
#  tensor is the force concentration B (⟨f⟩ = B·F₀) and the flux-side one is
#  ⟨k·B⟩ = A·k₀. Both are averages over the inclusion.
# =============================================================================

# Validation of a length; a symbolic one is taken as given.
_check_length(x, what) = _check_length(x, what, Val(is_hard_numeric(typeof(x))))
_check_length(x, what, ::Val{true}) = (0 < x < Inf) ||
    throw(ArgumentError("the $what of a Brinkman inclusion must be positive and finite, got $x"))
_check_length(x, what, ::Val{false}) = nothing

_float(x) = TensND.float_like(typeof(x))(x)

# Basis whose third vector is the axis (cylinder) or the normal (layer).
_brinkman_basis(::Nothing, euler_angles, T) = TensND.euler_basis(euler_angles; T = TensND.canonical_eltype(T))
_brinkman_basis(basis::TensND.AbstractBasis, _, _) = basis

"""
    BrinkmanSphere(R, medium = BrinkmanPorous(); form = ResistanceForm())

A sphere of radius `R` filled with `medium` ([`BrinkmanPorous`](@ref),
[`BrinkmanFluid`](@ref) or [`BrinkmanSolid`](@ref)), embedded in an unbounded
isotropic Brinkman medium. Its concentration tensors are isotropic and depend on
``\\alpha_0 = R/\\sqrt{k_0}`` and, for a porous sphere, on ``\\alpha_1 = R/\\sqrt{k_1}``
[bignonnetIJES2022](@cite):

- Stokes sphere: ``\\boldsymbol{A} = 3\\,\\dfrac{15 + 15\\alpha_0 + 6\\alpha_0^2 + \\alpha_0^3}{45 + 45\\alpha_0 + 6\\alpha_0^2 + \\alpha_0^3}\\,\\boldsymbol{1}``;
- impervious sphere: ``\\langle\\boldsymbol{r}\\cdot\\boldsymbol{A}\\rangle = \\dfrac{3\\,(3 + 3\\alpha_0 + \\alpha_0^2)}{2R^2}\\,\\boldsymbol{1}``, the drag of [brinkmanASR1949](@citet);
- porous sphere: the closed form derived in [Permeability from Stokes flow](@ref th-brinkman), which tends to
  ``3k_1/(k_1 + 2k_0)`` as ``R \\to \\infty`` (Darcy).

The phase property is the resistance or the permeability, according to `form`
([`BrinkmanForm`](@ref)); it is read only for a porous sphere.
"""
struct BrinkmanSphere{T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm} <: AbstractBrinkmanInclusion{T}
    radius::T
    medium::M
    form::F
    function BrinkmanSphere{T, M, F}(radius, medium, form) where {T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm}
        _check_length(radius, "radius")
        return new{T, M, F}(radius, medium, form)
    end
end

BrinkmanSphere(R::Number, medium::M, form::F) where {M <: BrinkmanMedium, F <: BrinkmanForm} =
    BrinkmanSphere{typeof(_float(R)), M, F}(_float(R), medium, form)
BrinkmanSphere(R::Number, medium::BrinkmanMedium = BrinkmanPorous(); form::BrinkmanForm = ResistanceForm()) =
    BrinkmanSphere(R, medium, form)

"""
    BrinkmanCylinder(R, medium = BrinkmanPorous(); form = ResistanceForm(),
                     euler_angles = (), basis = nothing)

An infinite circular cylinder of radius `R`, whose axis ``\\underline{n}`` is the third
vector of `basis` (built from the ZYZ `euler_angles` when no basis is given),
filled with `medium`. Its concentration tensors are transversely isotropic,
``A_n\\,\\underline{n}\\otimes\\underline{n} + A_t\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n})``, and involve the modified
Bessel functions of ``\\alpha_0 = R/\\sqrt{k_0}`` [bignonnetIJES2022](@cite).

In the resistance form the reference must have ``\\boldsymbol{r}_0 \\neq \\boldsymbol{0}``: in a clear fluid an
unbounded cylinder has no solution (Stokes' paradox). An orientation
distribution is applied by the scheme (`symmetrize = :iso`).
"""
struct BrinkmanCylinder{T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis} <:
    AbstractBrinkmanInclusion{T}
    radius::T
    medium::M
    form::F
    basis::B
    function BrinkmanCylinder{T, M, F, B}(radius, medium, form, basis) where {T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis}
        _check_length(radius, "radius")
        return new{T, M, F, B}(radius, medium, form, basis)
    end
end

BrinkmanCylinder(R::Number, medium::M, form::F, basis::B) where {M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis} =
    BrinkmanCylinder{typeof(_float(R)), M, F, B}(_float(R), medium, form, basis)
function BrinkmanCylinder(
        R::Number, medium::BrinkmanMedium = BrinkmanPorous();
        form::BrinkmanForm = ResistanceForm(), euler_angles::Tuple{Vararg{Real}} = (),
        basis::Union{Nothing, TensND.AbstractBasis} = nothing
    )
    return BrinkmanCylinder(R, medium, form, _brinkman_basis(basis, euler_angles, typeof(_float(R))))
end

"""
    BrinkmanLayer(h, medium = BrinkmanPorous(); form = ResistanceForm(),
                  euler_angles = (), basis = nothing)

The space between two parallel planes at distance ``2h``, of normal ``\\underline{n}`` (the
third vector of `basis`), filled with a porous medium or a clear fluid. The flow
normal to the planes is uniform, so ``A_n = 1``; the in-plane component follows
from a one-dimensional Brinkman problem [bignonnetIJES2022](@cite).

An impervious layer is refused (it would block the normal flow), and so is a
clear-fluid reference in the resistance form.
"""
struct BrinkmanLayer{T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis} <:
    AbstractBrinkmanInclusion{T}
    half_thickness::T
    medium::M
    form::F
    basis::B
    function BrinkmanLayer{T, M, F, B}(h, medium, form, basis) where {T <: Number, M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis}
        _check_length(h, "half-thickness")
        medium isa BrinkmanSolid && throw(
            ArgumentError(
                "an impervious BrinkmanLayer would block the flow normal to its planes " *
                    "(infinite drag); represent the solid by BrinkmanSphere or BrinkmanCylinder"
            )
        )
        return new{T, M, F, B}(h, medium, form, basis)
    end
end

BrinkmanLayer(h::Number, medium::M, form::F, basis::B) where {M <: BrinkmanMedium, F <: BrinkmanForm, B <: TensND.AbstractBasis} =
    BrinkmanLayer{typeof(_float(h)), M, F, B}(_float(h), medium, form, basis)
function BrinkmanLayer(
        h::Number, medium::BrinkmanMedium = BrinkmanPorous();
        form::BrinkmanForm = ResistanceForm(), euler_angles::Tuple{Vararg{Real}} = (),
        basis::Union{Nothing, TensND.AbstractBasis} = nothing
    )
    return BrinkmanLayer(h, medium, form, _brinkman_basis(basis, euler_angles, typeof(_float(h))))
end

"""
    BrinkmanCoatedSphere(R, R_out; form = ResistanceForm())

An impervious sphere of radius `R` wrapped in a shell of clear fluid up to the
radius `R_out > R`, embedded in an unbounded Brinkman medium. The averages run
over the whole coated sphere, so its volume fraction is that of the ball of
radius `R_out`. Used alone at volume fraction 1 in a self-consistent scheme, it
is the effective-medium counterpart of the cell model of [happelAIChE1958](@citet)
[bignonnetIJES2022](@cite). The phase property is not read.
"""
struct BrinkmanCoatedSphere{T <: Number, F <: BrinkmanForm} <: AbstractBrinkmanInclusion{T}
    radius::T
    outer_radius::T
    form::F
    function BrinkmanCoatedSphere{T, F}(R, R_out, form) where {T <: Number, F <: BrinkmanForm}
        _check_length(R, "core radius")
        _check_length(R_out, "outer radius")
        _check_shell(R, R_out, Val(is_hard_numeric(T)))
        return new{T, F}(R, R_out, form)
    end
end

_check_shell(R, R_out, ::Val{true}) = R_out > R || throw(
    ArgumentError(
        "the outer radius $R_out of a BrinkmanCoatedSphere must exceed its core radius $R; " *
            "without a fluid shell, use BrinkmanSphere(R, BrinkmanSolid())"
    )
)
_check_shell(R, R_out, ::Val{false}) = nothing

function BrinkmanCoatedSphere(R::Number, R_out::Number, form::F) where {F <: BrinkmanForm}
    Rp, Ro = promote(_float(R), _float(R_out))
    return BrinkmanCoatedSphere{typeof(Rp), F}(Rp, Ro, form)
end
BrinkmanCoatedSphere(R::Number, R_out::Number; form::BrinkmanForm = ResistanceForm()) =
    BrinkmanCoatedSphere(R, R_out, form)

# The medium of an inclusion; that of the coated sphere is composite, never
# homogeneous.
struct _CoatedMedium <: BrinkmanMedium end
_medium(i::AbstractBrinkmanInclusion) = i.medium
_medium(::BrinkmanCoatedSphere) = _CoatedMedium()

# ── Accessors ────────────────────────────────────────────────────────────────

MFH_Core.inclusion_basis(::Union{BrinkmanSphere{T}, BrinkmanCoatedSphere{T}}) where {T} =
    TensND.CanonicalBasis{3, TensND.canonical_eltype(T)}()
MFH_Core.inclusion_basis(i::Union{BrinkmanCylinder, BrinkmanLayer}) = i.basis

function _axis(i::Union{BrinkmanCylinder, BrinkmanLayer})
    e = TensND.vecbasis(i.basis)
    return (e[1, 3], e[2, 3], e[3, 3])
end

# ── Reference and phase properties ──────────────────────────────────────────

_iso_value(P::TensND.TensISO{2, 3}, _) = first(TensND.get_data(P))
_iso_value(P::TensND.AbstractTens{2, 3}, what) = throw(
    ArgumentError(
        "a Brinkman inclusion needs an isotropic $what (a TensISO{2,3}), got a " *
            "$(nameof(typeof(P))). Its closed forms solve the problem in an isotropic " *
            "medium only; average the orientations with `symmetrize = :iso`."
    )
)

# The variable of each formulation: α₀ = R√r₀ (resistance), s = √k₀/R (permeability).
_reference_variable(::ResistanceForm, R, P₀) = R * sqrt(_iso_value(P₀, "reference"))
_reference_variable(::PermeabilityForm, R, P₀) = sqrt(_iso_value(P₀, "reference")) / R

# An unbounded inclusion in a clear fluid has no Brinkman solution.
_require_bounded_reference(i::Union{BrinkmanCylinder, BrinkmanLayer}, x) = _require_bounded_reference(i.form, i, x)
_require_bounded_reference(::AbstractBrinkmanInclusion, _) = nothing
_require_bounded_reference(::PermeabilityForm, _, _) = nothing
_require_bounded_reference(::ResistanceForm, i, a) = iszero(a) && throw(
    ArgumentError(
        "$(nameof(typeof(i))) in a clear-fluid reference (r₀ = 0): an unbounded inclusion " *
            "has no Stokes solution (Stokes' paradox), so the dilute, Mori–Tanaka and " *
            "differential estimates do not exist with a fluid matrix. Use a porous reference " *
            "(a self-consistent scheme, or a microporous matrix)."
    )
)

_iso(x) = TensND.TensISO{3}(x)
function _ti(t, n, axis)
    T = promote_type(typeof(t), typeof(n), eltype(axis))
    return TensND.TensTI{2}(T(t), T(n), map(T, axis))
end

# ── Scalar values: (gradient-side, flux-side) per medium and form ───────────
#
#  `_values(i, x, P₁)` returns, for the inclusion `i` at the reference variable
#  `x`, the pair (gradient-side, flux-side) of scalar values (sphere) or of
#  (axial, transverse) pairs (cylinder, layer).

# Sphere
_values(i::BrinkmanSphere, a, P₁) = _sphere_values(i.medium, i.form, i.radius, a, P₁)
_sphere_values(::BrinkmanFluid, ::ResistanceForm, R, a, _) = (_sphere_fluid_r(a), zero(a))
_sphere_values(::BrinkmanSolid, ::ResistanceForm, R, a, _) = (zero(a), _sphere_solid_r(a) / R^2)
_sphere_values(::BrinkmanFluid, ::PermeabilityForm, R, s, _) = (zero(s), R^2 * _sphere_fluid_k(s))
_sphere_values(::BrinkmanSolid, ::PermeabilityForm, R, s, _) = (_sphere_solid_k(s), zero(s))
function _sphere_values(::BrinkmanPorous, f::ResistanceForm, R, a, P₁)
    r₁ = _iso_value(P₁, "phase resistance")
    # r₁ = 0 is the Stokes sphere exactly (the porous closed form reads 0/0 there).
    iszero(r₁) && return _sphere_values(BrinkmanFluid(), f, R, a, P₁)
    A = _sphere_porous_r(a, R * sqrt(r₁))
    return (A, r₁ * A)
end
function _sphere_values(::BrinkmanPorous, f::PermeabilityForm, R, s, P₁)
    k₁ = _iso_value(P₁, "phase permeability")
    # k₁ = 0 is the impervious sphere exactly.
    iszero(k₁) && return _sphere_values(BrinkmanSolid(), f, R, s, P₁)
    b = R / sqrt(k₁)
    B = _sphere_porous_k(s, b) * b^2 * s^2      # B = A k₀/k₁
    return (B, k₁ * B)
end

# Cylinder: pairs (axial, transverse)
_values(i::BrinkmanCylinder, x, P₁) = _cylinder_values(i.medium, i.form, i.radius, x, P₁)
_cylinder_values(::BrinkmanFluid, ::ResistanceForm, R, a, _) = (_cylinder_fluid_r(a), (zero(a), zero(a)))
function _cylinder_values(::BrinkmanSolid, ::ResistanceForm, R, a, _)
    g = _cylinder_solid_r(a) / R^2
    return ((zero(g), zero(g)), (g, 2g))
end
_cylinder_values(::BrinkmanFluid, ::PermeabilityForm, R, s, _) =
    ((zero(s), zero(s)), R^2 .* _cylinder_fluid_k(s))
function _cylinder_values(::BrinkmanSolid, ::PermeabilityForm, R, s, _)
    g = _cylinder_solid_k(s)
    return ((g, 2g), (zero(g), zero(g)))
end
function _cylinder_values(::BrinkmanPorous, f::ResistanceForm, R, a, P₁)
    r₁ = _iso_value(P₁, "phase resistance")
    iszero(r₁) && return _cylinder_values(BrinkmanFluid(), f, R, a, P₁)
    A = _cylinder_porous_r(a, R * sqrt(r₁))
    return (A, r₁ .* A)
end
function _cylinder_values(::BrinkmanPorous, f::PermeabilityForm, R, s, P₁)
    k₁ = _iso_value(P₁, "phase permeability")
    iszero(k₁) && return _cylinder_values(BrinkmanSolid(), f, R, s, P₁)
    b = R / sqrt(k₁)
    kA = _cylinder_porous_k(s, b)               # k₀A/R²
    return (b^2 .* kA, R^2 .* kA)               # B = A k₀/k₁, k₁B = k₀A
end

# Layer: pairs (normal, in-plane); the normal flow is uniform.
_values(i::BrinkmanLayer, x, P₁) = _layer_values(i.medium, i.form, i.half_thickness, x, P₁)
_layer_values(::BrinkmanFluid, ::ResistanceForm, R, a, _) = ((one(a), _layer_fluid_r(a)), (zero(a), zero(a)))
_layer_values(::BrinkmanFluid, ::PermeabilityForm, R, s, _) =
    ((zero(s), zero(s)), (R^2 * s^2, R^2 * _layer_fluid_k(s)))
function _layer_values(::BrinkmanPorous, f::ResistanceForm, R, a, P₁)
    r₁ = _iso_value(P₁, "phase resistance")
    iszero(r₁) && return _layer_values(BrinkmanFluid(), f, R, a, P₁)
    At = _layer_porous_r(a, R * sqrt(r₁))
    return ((one(At), At), (r₁ * one(At), r₁ * At))
end
function _layer_values(::BrinkmanPorous, ::PermeabilityForm, R, s, P₁)
    k₁ = _iso_value(P₁, "phase permeability")
    iszero(k₁) && throw(
        ArgumentError("a BrinkmanLayer of zero permeability would block the flow normal to its planes")
    )
    b = R / sqrt(k₁)
    kAt = _layer_porous_k(s, b)                 # k₀Aₜ/R²
    return ((b^2 * s^2, b^2 * kAt), (R^2 * s^2, R^2 * kAt))
end

# Coated sphere: R is the core radius, as in α₀.
function _values(c::BrinkmanCoatedSphere, x, _)
    η = c.radius / c.outer_radius
    return _coated_values(c.form, c.radius, x, η)
end
function _coated_values(::ResistanceForm, R, a, η)
    A, R²rA = _coated_r(a, η)
    return (A, R²rA / R^2)
end
function _coated_values(::PermeabilityForm, R, s, η)
    A, R²rA = _coated_k(s, η)
    return (R²rA * s^2, R^2 * s^2 * A)          # B = ⟨r·A⟩k₀, ⟨k·B⟩ = A k₀
end

# ── Tensors ──────────────────────────────────────────────────────────────────

_tensor(::Union{BrinkmanSphere, BrinkmanCoatedSphere}, v) = _iso(v)
_tensor(i::Union{BrinkmanCylinder, BrinkmanLayer}, (n, t)) = _ti(t, n, _axis(i))

_size(i::Union{BrinkmanSphere, BrinkmanCylinder, BrinkmanCoatedSphere}) = i.radius
_size(i::BrinkmanLayer) = i.half_thickness

function _concentrations(i::AbstractBrinkmanInclusion, P₁, P₀)
    x = _reference_variable(i.form, _size(i), P₀)
    _require_bounded_reference(i, x)
    return _values(i, x, P₁)
end

"""
    gradient_gradient_concentration(i::AbstractBrinkmanInclusion, P₁, P₀; kw...)

Average over the inclusion of the gradient-side concentration tensor: the
velocity concentration ``\\boldsymbol{A}`` in the resistance form, the force concentration
``\\boldsymbol{B}`` in the permeability form. `P₀` is the isotropic reference property,
`P₁` the phase property (read only for a porous inclusion).
"""
function MFH_Core.gradient_gradient_concentration(
        i::AbstractBrinkmanInclusion, P₁::TensND.AbstractTens{2, 3}, P₀::TensND.AbstractTens{2, 3}; kw...
    )
    return _tensor(i, first(_concentrations(i, P₁, P₀)))
end

"""
    flux_gradient_concentration(i::AbstractBrinkmanInclusion, P₁, P₀; kw...)

Average over the inclusion of the flux per unit remote gradient: the drag density
``\\langle\\boldsymbol{r}\\cdot\\boldsymbol{A}\\rangle`` (per unit ``\\mu V_0``) in the resistance form, the velocity
``\\langle\\boldsymbol{k}\\cdot\\boldsymbol{B}\\rangle = \\boldsymbol{A}\\cdot\\boldsymbol{k}_0`` in the permeability form. For a solid or fluid
inclusion it is the limit that no product of a property by a concentration
could give.
"""
function MFH_Core.flux_gradient_concentration(
        i::AbstractBrinkmanInclusion, P₁::TensND.AbstractTens{2, 3}, P₀::TensND.AbstractTens{2, 3}; kw...
    )
    return _tensor(i, last(_concentrations(i, P₁, P₀)))
end

# Flow only: Brinkman inclusions have no elastic counterpart.
for gen in (:(MFH_Core.gradient_gradient_concentration), :(MFH_Core.flux_gradient_concentration))
    @eval $gen(i::AbstractBrinkmanInclusion, ::TensND.AbstractTens{4, 3}, ::TensND.AbstractTens{4, 3}; kw...) =
        throw(
        ArgumentError(
            "$(nameof(typeof(i))) models the flow of a fluid (second-order permeability or " *
                "resistance); it has no fourth-order (elastic) concentration tensor"
        )
    )
end

# ── Bounds ───────────────────────────────────────────────────────────────────
#
#  A solid, fluid or coated inclusion is not homogeneous, so the Voigt and Reuss
#  bounds read its constituent properties here: 0 or ∞ in the formulation's
#  property (Voigt) and in its inverse (Reuss). An infinite Voigt average is the
#  correct bound and makes the self-consistent seed ask for `init`.

Schemes.has_layer_average(::AbstractBrinkmanInclusion) = true
Schemes._layer_voigt(i::AbstractBrinkmanInclusion, ref::TensND.AbstractTens) =
    _iso(_bound_value(_medium(i), i.form, Val(:voigt), eltype(ref)))
Schemes._layer_reuss(i::AbstractBrinkmanInclusion, ref::TensND.AbstractTens) =
    _iso(_bound_value(_medium(i), i.form, Val(:reuss), eltype(ref)))

# Average of the property (Voigt) or of its inverse (Reuss) over the inclusion.
_bound_value(::BrinkmanSolid, ::ResistanceForm, ::Val{:voigt}, T) = T(Inf)
_bound_value(::BrinkmanSolid, ::ResistanceForm, ::Val{:reuss}, T) = zero(T)
_bound_value(::BrinkmanFluid, ::ResistanceForm, ::Val{:voigt}, T) = zero(T)
_bound_value(::BrinkmanFluid, ::ResistanceForm, ::Val{:reuss}, T) = T(Inf)
_bound_value(::BrinkmanSolid, ::PermeabilityForm, ::Val{:voigt}, T) = zero(T)
_bound_value(::BrinkmanSolid, ::PermeabilityForm, ::Val{:reuss}, T) = T(Inf)
_bound_value(::BrinkmanFluid, ::PermeabilityForm, ::Val{:voigt}, T) = T(Inf)
_bound_value(::BrinkmanFluid, ::PermeabilityForm, ::Val{:reuss}, T) = zero(T)
# A core of r = ∞ and a shell of k = ∞: both averages are infinite.
_bound_value(::_CoatedMedium, ::BrinkmanForm, ::Val, T) = T(Inf)
