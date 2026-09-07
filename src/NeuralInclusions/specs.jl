# =============================================================================
#  specs.jl — the physics of the surrogate: which tensor class the network
#  predicts, and how its raw outputs become that tensor.
#
#  The guiding principle is that a network should only ever learn what is
#  genuinely unknown.  Three properties of the Hill tensor are *exact*, and
#  every one of them is enforced by construction here rather than fitted:
#
#  * **Material symmetry and major symmetry.**  The decoder emits a structured
#    TensND type — `TensISO`, `TensTI{4,·,5}`, `TensOrtho` — from the right
#    number of components, so `ℙ` cannot come out of the wrong class, and it
#    cannot come out non-major-symmetric.  This is why the surrogate predicts
#    `ℙ` and not `𝔸`.  `ℙ` *is* major-symmetric, so 5 transversely isotropic
#    components describe it exactly; `𝔸_εε` is not (`ℓ₃ ≠ ℓ₄`) and needs 6. On
#    top of that, a surrogate for `𝔸` would lose the exactness of `𝔸 = 𝕀` at
#    zero contrast.
#  * **Homogeneity in the reference moduli.**  `ℙ(λℂ₀) = ℙ(ℂ₀)/λ` exactly, so
#    the network never sees an absolute modulus: only the *shape* and, for
#    elasticity, `ν₀`.
#  * **Frame indifference.**  Components are predicted in the inclusion's own
#    frame; the axis or frame handed to the constructor puts the result in the
#    global frame. The network is never asked to learn a rotation.
#
#  On top of that, [`AffineHill`](@ref) removes `ν₀` from the inputs too, using
#  the shape/moduli factorization of the isotropic-matrix Hill tensor — the one
#  the theory page states and `Elasticity._hill_3d_iso` implements:
#
#      ℙ = 𝕌ᴬ/(λ₀+2μ₀) + (𝕍ᴬ − 𝕌ᴬ)/μ₀ = d · 𝕌ᴬ + 𝕍ᴬ/μ₀,   d = 1/(λ₀+2μ₀) − 1/μ₀,
#
#  which is nothing but a regrouping of the two terms.  `𝕌ᴬ` and `𝕍ᴬ` are the
#  auxiliary geometric tensors `tens_UA` and `tens_VA`, functions of the *shape
#  alone*.  Nothing shape-specific is
#  reimplemented to exploit this: both live in the same symmetry class as `ℙ`,
#  so the network simply predicts twice as many components and the decoder
#  contracts them with the two material coefficients.  In transport the
#  decomposition has a single term (`ℙ_K = 𝕍ᴬ/k₀`), so the material dependence
#  is exact with no material input at all.
# =============================================================================

# ─── Tensor classes ──────────────────────────────────────────────────────────

"""
    AbstractHillClass

Which structured TensND type a surrogate predicts, and therefore how many
components it emits. A class fixes the tensor order too: order 4 is elasticity,
order 2 is transport.
"""
abstract type AbstractHillClass end

"Isotropic 4th-order tensor `α𝕁 + β𝕂` — the Hill tensor of a sphere."
struct HillISO <: AbstractHillClass end

"""
Transversely isotropic, major-symmetric 4th-order tensor: the five Walpole
components `(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆)` of `TensTI{4,·,5}` — the Hill tensor of a
spheroid in an isotropic matrix.
"""
struct HillTI <: AbstractHillClass end

"""
Orthotropic 4th-order tensor: the nine components
`(C₁₁, C₂₂, C₃₃, C₁₂, C₁₃, C₂₃, C₄₄, C₅₅, C₆₆)` of `TensOrtho` — the Hill
tensor of a triaxial ellipsoid in an isotropic matrix.
"""
struct HillOrtho <: AbstractHillClass end

"Isotropic 2nd-order tensor — the transport Hill tensor of a sphere."
struct HillISO2 <: AbstractHillClass end

"""
Transversely isotropic 2nd-order tensor `(a, b)` of `TensTI{2,·,2}` — the
transport Hill tensor of a spheroid.
"""
struct HillTI2 <: AbstractHillClass end

"""
Transversely isotropic 4th-order tensor **without major symmetry**: the six
Walpole components `(ℓ₁, …, ℓ₆)` of `TensTI{4,·,6}`, for the *strain-side*
localization tensor `𝔸_εε` of a morphology axisymmetric about `n`.

Six and not five, because a localization tensor is genuinely not
major-symmetric: `𝔸_εε = [𝕀 + ℙ:(ℂ₁−ℂ₀)]⁻¹` is the inverse of a product of two
major-symmetric tensors, which does not commute. For an oblate spheroid at a
contrast of 2 the defect reaches 10 % and `ℓ₃ ≠ ℓ₄` outright, so projecting onto
the five-component form would lose a few percent.

Six and not eight, because `TensTI{4,·,8}` also carries the couplings `ℓ₇`, `ℓ₈`
that are *antisymmetric* in an index pair. A tensor mapping symmetric strains to
symmetric stresses has `ℓ₇ = ℓ₈ = 0` identically — measured, not assumed.

`𝔸_εε` is dimensionless, so no scale divides out: see
[`dimensionless_scale`](@ref).
"""
struct StrainLocTI <: AbstractHillClass end

"""
Same six-component transversely isotropic form as [`StrainLocTI`](@ref), for the
*stress-side* localization tensor `𝔸_σε`.

The two differ only in their physical dimension, and therefore in what makes
them dimensionless: `𝔸_εε` is of degree 0 in the moduli, `𝔸_σε` of degree +1.
"""
struct StressLocTI <: AbstractHillClass end

const _CLASS_NAMES = Dict{Symbol, AbstractHillClass}(
    :iso => HillISO(), :ti => HillTI(), :ortho => HillOrtho(),
    :iso2 => HillISO2(), :ti2 => HillTI2(),
    :loc_ti => StrainLocTI(), :stress_loc_ti => StressLocTI(),
)
const _NAMES_CLASS = Dict{Any, Symbol}(typeof(v) => k for (k, v) in _CLASS_NAMES)

"""
    hill_class(name::Symbol) -> AbstractHillClass

Class from its serialized name: `:iso`, `:ti`, `:ortho`, `:iso2`, `:ti2`.
"""
function hill_class(name::Symbol)
    haskey(_CLASS_NAMES, name) || throw(
        ArgumentError(
            "unknown Hill class :$name; known classes are " *
                "$(sort(collect(keys(_CLASS_NAMES))))"
        )
    )
    return _CLASS_NAMES[name]
end

class_name(c::AbstractHillClass) = _NAMES_CLASS[typeof(c)]

"""
    ncomponents(class) -> Int

Number of independent components the class carries — the width of one term of
the network output.
"""
ncomponents(::HillISO) = 2
ncomponents(::HillTI) = 5
ncomponents(::HillOrtho) = 9
ncomponents(::HillISO2) = 1
ncomponents(::HillTI2) = 2
ncomponents(::Union{StrainLocTI, StressLocTI}) = 6

"""
    tensor_order(class) -> Int

`4` for elasticity, `2` for transport. The localization and contribution
generics are declared per order, so this is what decides which physics a
surrogate serves.
"""
tensor_order(::Union{HillISO, HillTI, HillOrtho, StrainLocTI, StressLocTI}) = 4
tensor_order(::Union{HillISO2, HillTI2}) = 2

"""
    tensor_order(t::AbstractTens) -> Int

Order of a TensND tensor, read off its type — the same notion as for a class, so
the two can be compared directly.
"""
tensor_order(::TensND.AbstractTens{O, 3}) where {O} = O

# ─── Class ↔ tensor ──────────────────────────────────────────────────────────
#
#  `build` goes components → tensor, `components` goes back.  The two must be
#  exact inverses: `components` reads the training targets off the teacher and
#  `build` reconstructs them at evaluation time, so any mismatch would show up
#  as a bias no amount of training could remove.

"""
    build(class, c, frame) -> AbstractTens

Assemble the structured tensor of `class` from the component vector `c`.
`frame` is the symmetry axis (an `NTuple{3}`) for the TI classes, the material
frame (a TensND basis) for `HillOrtho`, and ignored for the isotropic ones.
"""
build(::HillISO, c, _frame) = TensND.TensISO{3}(c[1], c[2])
build(::HillTI, c, axis) = TensND.TensTI{4}(c[1], c[2], c[3], c[4], c[5], axis)
build(::HillISO2, c, _frame) = TensND.TensISO{3}(c[1])
build(::HillTI2, c, axis) = TensND.TensTI{2}(c[1], c[2], axis)

build(::Union{StrainLocTI, StressLocTI}, c, axis) =
    TensND.TensTI{4}(c[1], c[2], c[3], c[4], c[5], c[6], axis)

build(::HillOrtho, c, frame) = Core._make_ortho(
    eltype(c), c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], nothing, frame
)

"""
    components(class, P, frame; atol = 1.0e-8) -> NTuple

The independent components of `P`, in the order [`build`](@ref) expects — the
exact inverse of `build`.

Goes through `TensND.proj_tens` rather than reading `get_data` off the tensor,
for two reasons. First, robustness: a teacher is free to return whichever
concrete type it likes, and the conduction kernels do exactly that — the
analytic transport Hill tensor of a spheroid comes back as a *generic*
`Tens{2,3}`, not a `TensTI{2,·,2}`, so reading its data would yield nine numbers
where two were wanted. Second, and more valuable: `proj_tens` also returns the
relative residual of the projection, which for a tensor genuinely in the class is
zero to round-off. Checking it against `atol` turns a wrong axis, a wrong frame
or a wrong class — all of which would otherwise train happily on corrupted
labels — into a loud error at dataset-generation time.
"""
function components(
        class::AbstractHillClass, P::TensND.AbstractTens, frame; atol::Real = 1.0e-8
    )
    B, _d, drel = _project(class, P, frame)
    drel ≤ atol || throw(
        ArgumentError(
            "projecting this $(nameof(typeof(P))) onto class :$(class_name(class)) " *
                "leaves a relative residual of $drel, far above $atol: the tensor is " *
                "not in that class. Either the surrogate's class does not match the " *
                "geometry (a spheroid is :ti, a triaxial ellipsoid :ortho, a sphere " *
                ":iso) or the symmetry axis / frame is wrong."
        )
    )
    data = TensND.get_data(B)
    n = ncomponents(class)
    length(data) == n || error(
        "internal inconsistency: projecting onto :$(class_name(class)) produced " *
            "$(length(data)) components, expected $n"
    )
    return data
end

_project(::Union{HillISO, HillISO2}, P, _frame) = TensND.proj_tens(Val(:ISO), P)
_project(::Union{HillTI, HillTI2}, P, axis) = TensND.proj_tens(Val(:TI), P, axis)
_project(::HillOrtho, P, frame) = TensND.proj_tens(Val(:ORTHO), P, frame)

# `proj_tens(Val(:TI), …)` returns the *five*-component major-symmetric form, so
# it cannot serve a localization tensor. `Core.transverse_isotropify` is the exact
# rotation-group average about the axis and keeps the non-major-symmetric content
# (`TensTI{4,·,8}`); dropping `ℓ₇`, `ℓ₈` — which vanish identically for a tensor
# with the minor symmetries — leaves the six components wanted. The residual is
# measured explicitly, in canonical components so that two different bases cannot
# be compared by accident.
function _project(::Union{StrainLocTI, StressLocTI}, P, axis)
    l = TensND.get_ℓ8(Core.transverse_isotropify(P, axis))
    B = TensND.TensTI{4}(l[1], l[2], l[3], l[4], l[5], l[6], axis)
    ref = Array(TensND.components_canon(P))
    d = maximum(abs, Array(TensND.components_canon(B)) .- ref)
    nrm = maximum(abs, ref)
    return B, d, nrm > 0 ? d / nrm : d
end

# ─── Geometry → frame ────────────────────────────────────────────────────────
#
#  Which frame argument `build` and `components` need depends on the class, and
#  for the TI classes on *which* semi-axis is the distinct one.  This lives here,
#  next to the classes, because both ends of the pipeline go through it: the
#  labeler reads the teacher's components in this frame, and the inclusion
#  writes the network's prediction back into it.  One definition, so they cannot
#  disagree.

"""
    _axes(geom) -> NTuple

Semi-axes of anything that carries them — an `Ellipsoid`, a
[`NeuralHillInclusion`](@ref), a user geometry handed to
[`generate_dataset`](@ref).
"""
_axes(geom) = geom.semi_axes

_axes_equal(x, y) = isapprox(x, y; rtol = 1.0e-10)

"""
    _spheroid_axis_index(a) -> Int

Which of the three semi-axes is the distinct one — the symmetry axis of a
spheroid. Refuses a shape that is not a spheroid, because the alternative is to
silently pick an axis and rotate the tensor by 90°.
"""
function _spheroid_axis_index(a::NTuple{3})
    _axes_equal(a[2], a[3]) && return 1
    _axes_equal(a[1], a[3]) && return 2
    _axes_equal(a[1], a[2]) && return 3
    return throw(
        ArgumentError(
            "semi_axes = $a has no two equal entries, so it is not a spheroid and " *
                "has no single symmetry axis. A :ti/:ti2 surrogate describes a " *
                "spheroid; use an :ortho one for a general ellipsoid."
        )
    )
end

# distinct / equal.  Positive logarithm ⟹ prolate, negative ⟹ oblate.
function _spheroid_ratio(a::NTuple{3})
    i = _spheroid_axis_index(a)
    j = i == 1 ? 2 : 1
    return a[i] / a[j]
end

"""
    _class_frame(class, geom) -> frame

The frame argument [`build`](@ref) and [`components`](@ref) need: the symmetry
axis for a TI class, the material frame for an orthotropic one, `nothing` for an
isotropic one.

For the TI classes the *column* carrying the symmetry axis is derived from the
semi-axes — column 1 for a prolate spheroid, column 3 for an oblate one, which
is what the analytic kernels of `Elasticity._hill_3d_iso` use.
"""
_class_frame(::Union{HillISO, HillISO2}, _geom) = nothing
_class_frame(::HillOrtho, geom) = Core.inclusion_basis(geom)

_class_frame(::Union{HillTI, HillTI2}, geom) =
    Core._basis_col(Core.inclusion_basis(geom), _spheroid_axis_index(_axes(geom)))

# A localization pair does *not* get its axis from the semi-axes: the morphology
# it describes may well be a sphere (`FEExcenteredSphere` is one) whose response
# is transversely isotropic about a direction the outer shape says nothing about.
# The convention is therefore the package's usual one — **column 3 of the
# inclusion basis is the axis** — the same column `FEExcenteredSphere` solves
# about (`FiniteElements._fe_frame`) and the same one that carries a crack's
# normal. A wrong choice here does not pass silently: `components` measures the
# projection residual.
_class_frame(::Union{StrainLocTI, StressLocTI}, geom) =
    Core._basis_col(Core.inclusion_basis(geom), 3)

# ─── Material coefficients ───────────────────────────────────────────────────

"""
    material_coeffs(class, P₀) -> Tuple

Coefficients of the exact affine decomposition of the Hill tensor on the
shape-only tensors, for an **isotropic** reference medium:

- order 4: `(d, 1/μ₀)` with `d = 1/(λ₀+2μ₀) − 1/μ₀`, so that
  `ℙ = d·𝕌ᴬ + (1/μ₀)·𝕍ᴬ`;
- order 2: `(1/k₀,)`, so that `ℙ_K = 𝕍ᴬ/k₀`.

Used by [`AffineHill`](@ref). The number of entries is the number of terms the
network has to predict per component.
"""
function material_coeffs(::Union{HillISO, HillTI, HillOrtho}, C₀::TensND.TensISO{4, 3})
    α, β = TensND.get_data(C₀)
    inv_lm = 3 / (α + 2β)
    inv_m = 2 / β
    return (inv_lm - inv_m, inv_m)
end

material_coeffs(::Union{HillISO2, HillTI2}, K₀::TensND.TensISO{2, 3}) =
    (inv(only(TensND.get_data(K₀))),)

"""
    dimensionless_scale(class, P₀) -> Number

The modulus by which `ℙ` is multiplied to make it dimensionless — `2μ₀` in
elasticity, `k₀` in transport. Used by [`DimensionlessHill`](@ref): because
`ℙ` is homogeneous of degree `−1` in the reference moduli, `scale · ℙ` depends
on the shape and on `ν₀` alone, and on nothing at all in transport.
"""
dimensionless_scale(::Union{HillISO, HillTI, HillOrtho}, C₀::TensND.TensISO{4, 3}) =
    TensND.get_data(C₀)[2]

dimensionless_scale(::Union{HillISO2, HillTI2}, K₀::TensND.TensISO{2, 3}) =
    only(TensND.get_data(K₀))

# The localization pair. `𝔸_εε` is already dimensionless — degree 0 in the moduli
# — so nothing divides out; `𝔸_σε` is of degree +1, so `𝔸_σε/(2μ₀)` is the
# dimensionless quantity.
#
# The invariance at work here is *not* the one gate A uses. `ℙ(λℂ₀) = ℙ(ℂ₀)/λ`
# holds because `ℙ` sees only the reference medium. A heterogeneous morphology
# carries its constituents inside itself, so scaling `ℂ₀` alone changes the
# contrast and changes `𝔸`; what is exact is the **simultaneous** scaling of the
# reference medium and of every constituent (measured to 3·10⁻¹⁵ and 9·10⁻¹⁴ on
# `FEExcenteredSphere`). Which is exactly why the features of such a surrogate
# have to be contrast *ratios* rather than absolute moduli.
dimensionless_scale(::StrainLocTI, ::TensND.TensISO{4, 3}) = 1
dimensionless_scale(::StressLocTI, C₀::TensND.TensISO{4, 3}) =
    inv(TensND.get_data(C₀)[2])

_iso_only(P₀) = throw(
    ArgumentError(
        "the ellipsoid surrogates are trained for an **isotropic** reference " *
            "medium and got a $(nameof(typeof(P₀))). Both the feature set (ν₀ alone) " *
            "and the exact homogeneity used to decode assume isotropy; an " *
            "anisotropic reference needs a surrogate trained on the corresponding " *
            "feature set. Under `IsoSymmetrize`/`TISymmetrize` the scheme " *
            "pre-projects the reference medium, which is one way to get there."
    )
)

material_coeffs(::Union{StrainLocTI, StressLocTI}, ::TensND.AbstractTens) = throw(
    ArgumentError(
        "a localization tensor has no affine decomposition on shape-only tensors: " *
            "the structure `ℙ = d·𝕌ᴬ + 𝕍ᴬ/μ₀` belongs to the Hill tensor of an " *
            "ellipsoid in an isotropic matrix. Use `DimensionlessHill` for a " *
            ":loc_ti / :stress_loc_ti surrogate."
    )
)

material_coeffs(::AbstractHillClass, P₀::TensND.AbstractTens) = _iso_only(P₀)
dimensionless_scale(::AbstractHillClass, P₀::TensND.AbstractTens) = _iso_only(P₀)

# ─── Component transforms ────────────────────────────────────────────────────
#
#  A component whose magnitude spans decades over the sampled shapes (the
#  oblate `ℓ`s as ω → 0) is hopeless to fit on a linear scale: the fit would
#  spend all its capacity on the largest values.  A per-component `:log`
#  removes that, and it is stored with the model because getting it wrong is
#  not an approximation but a wrong tensor.

const TRANSFORMS = (:identity, :log)

_bad_transform(kind) = throw(
    ArgumentError("unknown output transform :$kind; expected one of $TRANSFORMS")
)

# Branch on the symbol rather than dispatching on `Val(kind)`: an iterative
# scheme evaluates a surrogate thousands of times, and `Val` of a runtime
# symbol is a fresh, uninferrable call every single time.

"""
    apply_transform(kind, z)

Map a physical component to the space the network is fitted in (`:identity` or
`:log`).
"""
function apply_transform(kind::Symbol, z)
    kind === :identity && return z
    kind === :log && return log(z)
    return _bad_transform(kind)
end

"""
    invert_transform(kind, y)

Inverse of [`apply_transform`](@ref) — network space back to a physical
component.
"""
function invert_transform(kind::Symbol, y)
    kind === :identity && return y
    kind === :log && return exp(y)
    return _bad_transform(kind)
end

# ─── Output specifications ───────────────────────────────────────────────────

"""
    AbstractOutputSpec

How the network's raw output vector becomes a Hill tensor. Two
implementations, differing only in how much of the material dependence is
exact rather than learned: [`DimensionlessHill`](@ref) and [`AffineHill`](@ref).
"""
abstract type AbstractOutputSpec end

"""
    DimensionlessHill(class)

The network predicts the `ncomponents(class)` components of the
**dimensionless** Hill tensor `scale · ℙ`. Exact in the scale of the reference
moduli and in the symmetry class; `ν₀` is an input feature and its dependence
is learned.

The general-purpose choice: it needs nothing of the reference medium beyond a
scale, so the same shape of surrogate transfers to an anisotropic matrix, and
to a localization pair (gate B) where no affine structure exists.
"""
struct DimensionlessHill{C <: AbstractHillClass} <: AbstractOutputSpec
    class::C
end

"""
    AffineHill(class)

The network predicts the components of the shape-only tensors `𝕌ᴬ` and `𝕍ᴬ` —
`nterms(class) · ncomponents(class)` numbers — and the decoder contracts them
with [`material_coeffs`](@ref). The whole material dependence is then exact:
`ν₀` is *not* an input, and the surrogate is a function of the shape alone.

Only available for an isotropic reference medium, which is where the affine
structure comes from.
"""
struct AffineHill{C <: AbstractHillClass} <: AbstractOutputSpec
    class::C
end

hill_class(spec::AbstractOutputSpec) = spec.class

"""
    nterms(spec) -> Int

How many shape tensors the network predicts per component: `1` for
[`DimensionlessHill`](@ref), and for [`AffineHill`](@ref) the length of
[`material_coeffs`](@ref) — 2 in elasticity, 1 in transport.
"""
nterms(::DimensionlessHill) = 1
nterms(spec::AffineHill) = tensor_order(spec.class) == 4 ? 2 : 1

"""
    noutputs(spec) -> Int

Width of the network's output layer.
"""
noutputs(spec::AbstractOutputSpec) = nterms(spec) * ncomponents(spec.class)

"""
    needs_nu(spec) -> Bool

Whether the feature set has to carry `ν₀`. False for [`AffineHill`](@ref),
whose material dependence is exact, and for any transport surrogate.
"""
needs_nu(spec::DimensionlessHill) = tensor_order(spec.class) == 4
needs_nu(::AffineHill) = false

spec_name(::DimensionlessHill) = :dimensionless
spec_name(::AffineHill) = :affine

"""
    output_spec(name::Symbol, class::Symbol) -> AbstractOutputSpec

Rebuild an output specification from its serialized names.
"""
function output_spec(name::Symbol, class::Symbol)
    c = hill_class(class)
    name === :dimensionless && return DimensionlessHill(c)
    name === :affine && return AffineHill(c)
    return throw(
        ArgumentError(
            "unknown output specification :$name; expected :dimensionless or :affine"
        )
    )
end

"""
    decode(spec, z, P₀, frame) -> AbstractTens

Turn the network's *untransformed* output `z` into the Hill tensor, in the
frame given.

`z` has length [`noutputs`](@ref); for [`AffineHill`](@ref) it is read as a
`ncomponents × nterms` column-major block, one column per shape tensor.
"""
function decode(spec::DimensionlessHill, z::AbstractVector, P₀, frame)
    return build(spec.class, z ./ dimensionless_scale(spec.class, P₀), frame)
end

function decode(spec::AffineHill, z::AbstractVector, P₀, frame)
    nc = ncomponents(spec.class)
    coeffs = material_coeffs(spec.class, P₀)
    c = ntuple(nc) do i
        s = z[i] * coeffs[1]
        for t in 2:length(coeffs)
            s += z[i + (t - 1) * nc] * coeffs[t]
        end
        s
    end
    return build(spec.class, c, frame)
end
