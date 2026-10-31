# =============================================================================
#  homogenize_alv.jl — public dispatcher routing ViscoLaw properties
#  through the ALV homogenization pipeline.
#
#  Usage : `homogenize_alv(rve, scheme, :C; times = T)`, with `T` a
#  `Vector{<:Real}` of monotonically increasing time points.
#
#  Note there is **no** auto-routing: `homogenize` does not inspect the phase
#  properties and dispatch here on finding a `ViscoLaw`.  The entry point is
#  explicit, deliberately — `has_visco_property` below is a helper for user
#  code and for the guards, not a dispatch hook.  The Laplace-Carson route is
#  equally explicit, through `homogenize_lc`.
# =============================================================================

"""
    has_visco_property(cell, prop::Symbol = :C) -> Bool

Return `true` if any member (phase or layer) of `cell` carries a
[`ViscoLaw`](@ref) under the key `prop`.

Reads through the **raw** accessor `cell_container_property`: this is a type
inspection, and going through the resolving accessor would run a whole inner
homogenization on every declaratively nested cell just to look at a type. A
nested cell is instead recursed into, so a `ViscoLaw` buried one scale down
is still reported.
"""
function has_visco_property(cell::MFH_Core.AbstractHomogenizationCell, prop::Symbol = :C)
    for name in MFH_Core.cell_member_names(cell)
        MFH_Core.cell_has_property(cell, name, prop) || continue
        _is_visco_value(MFH_Core.cell_container_property(cell, name, prop), prop) && return true
    end
    return false
end

_is_visco_value(::AbstractViscoLaw, ::Symbol) = true
_is_visco_value(::VolterraOperator, ::Symbol) = true
_is_visco_value(v, ::Symbol) = false
_is_visco_value(h::MFH_Core.Homogenized, key::Symbol) =
    has_visco_property(h.cell, h.property === nothing ? key : h.property)

# ── Iso projection of a 6n×6n block matrix (ECHOES `symmetrize=[ISO]`) ─────
#
# ECHOES applies an orientation-averaging projection to each phase
# 4-tensor whose RVE definition carries `symmetrize=[ISO]`.  For each
# 6×6 Mandel block, the iso projection is :
#   α = (1/3) Σᵢⱼ T_iijj  =  (M[1,1] + M[2,2] + M[3,3] + 2(M[1,2] + M[1,3] + M[2,3])) / 3
#   β = (Σᵢⱼ T_ijij - α) / 5  =  (trace(M) - α) / 5
#   M_iso = α 𝕁 + β 𝕂  (rebuilt via `iso_blocks_from_params` on a 1×1
#                       parameter pair).
#
# We apply this block-by-block to the (6n)×(6n) `Ñ` (and `Ã`) of the
# inclusions when their phase's `phase_symmetrize` is `IsoSymmetrize`.
# Iso averaging of a TI block over a uniform orientation distribution
# matches ECHOES `symmetrize=[ISO]` exactly.

"""
    _iso_project_blocks(M::AbstractMatrix) -> Matrix

Project every ``6\\times 6`` Mandel block of a ``6n\\times 6n`` ALV matrix to its iso
component (Reynolds average over the orthogonal group), returning a
new ``6n\\times 6n`` block matrix whose every block is iso.  Equivalent to
the ECHOES `symmetrize=[ISO]` orientation-averaging projection
applied to each ``(t_i, t_j)`` block independently.
"""
function _iso_project_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("_iso_project_blocks: matrix must be square"))
    sz % 6 == 0 ||
        throw(ArgumentError("_iso_project_blocks: size $(sz) not divisible by 6"))
    n = sz ÷ 6
    T = eltype(M)
    α = zeros(T, n, n)
    β = zeros(T, n, n)
    @inbounds for i in 1:n, j in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        a, b = TensND.iso_average_mandel(view(M, rows, cols))
        α[i, j] = a
        β[i, j] = b
    end
    return iso_blocks_from_params(α, β)
end

"""
    _ti_project_blocks(M::AbstractMatrix, axis) -> Matrix

Exact azimuthal average about `axis` of every ``6\\times 6`` Mandel block of a
``6n\\times 6n`` ALV matrix (echoes `visco_transverse_isotropify` counterpart).
Each block is averaged independently with
[`Core.ti_average_mandel`](@ref); the non-major-symmetric content of the
blocks (``\\ell_3 \\ne \\ell_4``, antisymmetric couplings) is preserved, which the
6-parameter TI closure of `conversions.jl` cannot represent — hence the
full-matrix return.
"""
function _ti_project_blocks(M::AbstractMatrix, axis)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("_ti_project_blocks: matrix must be square"))
    sz % 6 == 0 ||
        throw(ArgumentError("_ti_project_blocks: size $(sz) not divisible by 6"))
    n = sz ÷ 6
    out = similar(M, float(eltype(M)))
    @inbounds for i in 1:n, j in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        out[rows, cols] = MFH_Core.ti_average_mandel(view(M, rows, cols), axis)
    end
    return out
end

"""
    _maybe_symmetrize_alv(M, sym) -> Matrix

Apply the orientation-averaging projection corresponding to `sym` to a
``6n\\times 6n`` ALV block matrix : `NoSymmetrize` (passthrough),
`IsoSymmetrize` (block-by-block exact ``\\mathrm{SO}(3)`` average) and
`TISymmetrize(axis)` (block-by-block exact azimuthal average about the
axis).
"""
@inline _maybe_symmetrize_alv(M::AbstractMatrix, ::NoSymmetrize) = M
# An operator stored by blocks: the isotropic average read on its blocks; the
# transversely isotropic one on its full matrix, and kept full.
_maybe_symmetrize_alv(R::VolterraOperator{<:TensISO{4, 3}}, ::IsoSymmetrize) = R
_maybe_symmetrize_alv(R::VolterraOperator, ::IsoSymmetrize) = _iso_part(R)
_maybe_symmetrize_alv(R::VolterraOperator, s::TISymmetrize) = VolterraOperator(_maybe_symmetrize_alv(Matrix(R), s), R.grid)
@inline _maybe_symmetrize_alv(M::AbstractMatrix, ::IsoSymmetrize) =
    _iso_project_blocks(M)
@inline _maybe_symmetrize_alv(M::AbstractMatrix, sym::TISymmetrize) =
    _ti_project_blocks(M, sym.axis)

# ── Order-2 counterparts (3 × 3 time blocks) ───────────────────────────────
#
# The projectors above slice 6 × 6 MANDEL blocks, so they apply to the
# order-4 `(6n × 6n)` ALV matrices only.  The order-2 (conduction /
# diffusion) pipeline carries `(3n × 3n)` matrices whose blocks are plain
# 2-tensors, and feeding those to `_iso_project_blocks` either throws
# ("size not divisible by 6", odd `n`) or — worse, for even `n` — silently
# reinterprets two consecutive TIME blocks as one Mandel block and returns
# a matrix that is not an orientation average of anything.

"""
    _iso_project_blocks3(M::AbstractMatrix) -> Matrix

Exact ``\\mathrm{SO}(3)`` orientation average of every ``3\\times 3`` block of a ``3n\\times 3n``
order-2 ALV matrix.  The isotropic part of a 2-tensor ``\\boldsymbol{B}`` is
``\\tfrac{1}{3}(\\mathrm{tr}\\,\\boldsymbol{B})\\,\\boldsymbol{1}``, so each block collapses onto its spherical part.
"""
function _iso_project_blocks3(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("_iso_project_blocks3: matrix must be square"))
    sz % 3 == 0 ||
        throw(ArgumentError("_iso_project_blocks3: size $(sz) not divisible by 3"))
    n = sz ÷ 3
    out = zeros(float(eltype(M)), sz, sz)
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1)
        c = 3 * (j - 1)
        a = (M[r + 1, c + 1] + M[r + 2, c + 2] + M[r + 3, c + 3]) / 3
        for k in 1:3
            out[r + k, c + k] = a
        end
    end
    return out
end

"""
    _ti_project_blocks3(M::AbstractMatrix, axis) -> Matrix

Exact azimuthal average about `axis` of every ``3\\times 3`` block of a
``3n\\times 3n`` order-2 ALV matrix.  Averaging a 2-tensor ``\\boldsymbol{B}`` over the
rotations about the unit vector ``\\underline{n}`` keeps three invariants — the axial
component ``a_n = \\underline{n}\\cdot\\boldsymbol{B}\\cdot\\underline{n}``, the transverse mean
``a_t = (\\mathrm{tr}\\,\\boldsymbol{B} - a_n)/2``, and the axial antisymmetric part (the
component of ``\\boldsymbol{B}`` along ``[\\underline{n}]_{\\times}``, which commutes with those rotations) —
and averages everything else to zero:

```math
\\langle\\boldsymbol{B}\\rangle = a_t\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n})
  + a_n\\,\\underline{n}\\otimes\\underline{n} + c\\,[\\underline{n}]_{\\times},
\\qquad
c = \\tfrac{1}{2}\\,\\boldsymbol{B}^{\\mathrm{skew}} : [\\underline{n}]_{\\times} .
```

The antisymmetric term is preserved rather than dropped, mirroring
[`_ti_project_blocks`](@ref), which likewise keeps the block content a
symmetric closure cannot represent.
"""
function _ti_project_blocks3(M::AbstractMatrix, axis)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("_ti_project_blocks3: matrix must be square"))
    sz % 3 == 0 ||
        throw(ArgumentError("_ti_project_blocks3: size $(sz) not divisible by 3"))
    n = sz ÷ 3
    nv = _alv_unit_axis3(axis)
    out = zeros(float(eltype(M)), sz, sz)
    for i in 1:n, j in 1:n
        rows, cols = (3i - 2):(3i), (3j - 2):(3j)
        out[rows, cols] .= TensND.ti_average_matrix(view(M, rows, cols), nv)
    end
    return out
end

# The axis reaches us either as a plain 3-vector/tuple or as a TensND
# 1-tensor; its canonical components, whatever basis the vector is written in.
_alv_axis_components(axis::TensND.AbstractTens) = collect(TensND.canonical_components(axis))
_alv_axis_components(axis) = collect(axis)

function _alv_unit_axis3(axis)
    v = _alv_axis_components(axis)
    length(v) == 3 || throw(ArgumentError("ALV TI average: axis must have 3 components"))
    nrm = sqrt(sum(abs2, v))
    iszero(nrm) && throw(ArgumentError("ALV TI average: axis must be non-zero"))
    return v ./ nrm
end

"""
    _maybe_symmetrize_alv2(M, sym) -> Matrix

Order-2 counterpart of [`_maybe_symmetrize_alv`](@ref), for ``3n\\times 3n``
ALV matrices whose blocks are 2-tensors.
"""
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, ::NoSymmetrize) = M
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, ::IsoSymmetrize) =
    _iso_project_blocks3(M)
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, sym::TISymmetrize) =
    _ti_project_blocks3(M, sym.axis)

"""
    _trapezoidal_relaxation(law::AbstractViscoLaw, times, B) -> Matrix

Build the discrete relaxation block matrix from a `ViscoLaw`, regardless
of whether the law is in `:relaxation` or `:creep` mode.  When the law
is `:creep`, the trapezoidal compliance matrix is inverted (block forward
substitution at `block_size = B`) to obtain the corresponding relaxation
matrix — the convention every ALV scheme assumes internally.

`B` is the block size (`6` for order-4 4-tensor / Mandel, `3` for
order-2 vector-tensor).
"""
function _trapezoidal_relaxation(
        law::AbstractViscoLaw,
        times::AbstractVector{<:Real}, B::Int
    )
    M = trapezoidal_matrix(law, times)
    if visco_mode(law) === :creep
        return volterra_inverse(M; block_size = B)
    end
    return M
end

# An operator given directly — the result of the scale below — is already the relaxation
# operator on the grid: a plain matrix has its size checked, a `VolterraOperator` its grid,
# and it is materialized for the dense scheme formulas.
_trapezoidal_relaxation(R::AbstractMatrix, times::AbstractVector{<:Real}, B::Int) = _check_operator(R, times, B)
# An elastic tensor, written out on the grid: the tensor on the block diagonal.
_trapezoidal_relaxation(C::TensND.AbstractTens, times::AbstractVector{<:Real}, B::Int) =
    _check_operator(Matrix(VolterraOperator(C, times)), times, B)
_trapezoidal_relaxation(C::TensND.AbstractTens{2}, times::AbstractVector{<:Real}, B::Int) =
    _check_operator(Matrix(VolterraOperator(C, times)), times, B)
function _trapezoidal_relaxation(R::VolterraOperator, times::AbstractVector{<:Real}, B::Int)
    _same_grid(R.grid, TimeGrid(times), "homogenize_alv")
    return _check_operator(Matrix(R), times, B)
end

function _check_operator(R::AbstractMatrix, times::AbstractVector{<:Real}, B::Int)
    n = length(times)
    size(R) == (B * n, B * n) || throw(
        DimensionMismatch(
            "an ALV operator for $n instants and blocks of size $B is $(B * n)×$(B * n); " *
                "this one is $(size(R, 1))×$(size(R, 2)): was it computed on another time grid?"
        )
    )
    return R
end

# The order of a property, 4 or 2, from a law's sample or an operator's size.
_alv_order_int(x, times) = _val_int(_alv_order(x, times))
_val_int(::Val{N}) where {N} = N

"""
    _alv_property_order(law::AbstractViscoLaw, t) -> Int

Inspect the sample returned by `visco_eval(law, t, t)` and report the
tensor order (`2` for vector-tensor / ``3\\times 3``, `4` for 4-tensor / ``6\\times 6``
Mandel).  Used by [`homogenize_alv`](@ref) to dispatch between the
order-4 (stiffness / relaxation) and order-2 (conductivity / creep
admittance) pipelines.
"""
_alv_property_order(law::AbstractViscoLaw, t::Real) = _alv_sample_order(visco_eval(law, t, t))

_alv_sample_order(::TensND.AbstractTens{2, 3}) = 2
_alv_sample_order(::TensND.AbstractTens{4, 3}) = 4
# A plain array tells its order by its size only, a run-time value.
function _alv_sample_order(sample::AbstractMatrix)
    size(sample) == (3, 3) && return 2
    size(sample) == (6, 6) && return 4
    return _alv_order_error(sample)
end
_alv_sample_order(sample) = _alv_order_error(sample)
_alv_order_error(sample) = throw(
    ArgumentError("homogenize_alv: cannot infer ALV order from sample of type $(typeof(sample))")
)

"""
    homogenize_alv(rve, scheme, prop::Symbol; times = nothing) -> VolterraOperator

The effective relaxation operator of `rve` in aging linear viscoelasticity, on the time grid
`times` (a vector of instants or a [`TimeGrid`](@ref)): the operator that maps a strain
history on the grid to the stress history, or a temperature gradient to a flux in
conduction. `times` may be left out when a phase property is a [`VolterraOperator`](@ref):
its grid is then the grid of the computation.

**Phase properties.** The property `prop` of a phase may be

  * a law — a [`ViscoLaw`](@ref) or any [`AbstractViscoLaw`](@ref), in relaxation or creep form,
    or an elastic tensor, which is a law constant in time;
  * an operator on the same grid, as one scale hands it to the next (the result of another
    `homogenize_alv`): a [`VolterraOperator`](@ref), or the plain ``6n\\times 6n``
    (``3n\\times 3n`` in conduction) matrix of one;
  * a [`Homogenized`](@ref) cell, homogenized here on the same grid and memoized for the whole
    call, as in elasticity.

**Result.** A [`VolterraOperator`](@ref) on the grid, stored in the cheapest symmetry class
that holds it — isotropic, the class of a phase property (transversely isotropic about its
axis, orthotropic in its frame, …), transversely isotropic about ``\\underline{e}_3``,
orthotropic in the canonical frame — and dense otherwise. It is an `AbstractMatrix` that
multiplies and inverts like the full matrix, which `Matrix(R)` writes out, and computes as
the elastic tensors do (`⊡`, `inv`, `+`); see
[Aging viscoelastic operators and histories](@ref man-volterra-operators).

The ALV pipeline builds the discrete block matrices of every phase, computes the ALV Hill
kernel and dilute concentration tensors, and dispatches on `scheme` to the corresponding
`_alv` scheme function.

The function dispatches on the **order of the matrix property** (read
once from the matrix `ViscoLaw` sample type):
  * order-4 (4-tensor / ``6\\times 6`` Mandel kernel) → returns ``6n\\times 6n``
    relaxation matrix following the standard Hill-kernel +
    dilute-concentration pipeline.
  * order-2 (2-tensor / ``3\\times 3`` kernel; conductivity / diffusion /
    permittivity) → returns ``3n\\times 3n`` matrix via the order-2 ALV
    pipeline (`hill_kernel_order2`, time-space decoupling).

Supports two inclusion-geometry families, in both orders:
  * Single-shape ellipsoidal (`Ellipsoid`, `Spheroid`): the standard
    Hill-kernel + dilute-concentration pipeline.
  * `LayeredSphere`: the ALV recurrences feed its concentration and
    contribution directly — no Hill kernel is needed. At order 4 these are
    [`gradient_gradient_concentration_alv`](@ref) and [`property_contribution_alv`](@ref)
    (bulk and shear recurrences, [`bulk_concentration_alv`](@ref),
    [`shear_concentration_alv`](@ref)); at order 2,
    [`gradient_gradient_concentration_alv`](@ref) and [`property_contribution_alv`](@ref).
    The phase property is ignored (the layers carry their own kernels); pass
    any `ViscoLaw` (e.g. `heaviside_law(C_0)`) as a placeholder.

`symmetrize` is honored in both orders, with the projector of the matching
tensor order (``6\\times 6`` Mandel blocks for order 4, 2-tensor blocks for order 2 —
see [`_maybe_symmetrize_alv`](@ref) and [`_maybe_symmetrize_alv2`](@ref)).
`Voigt` and `Reuss` average the phase relaxation matrices (Voigt) or their
Volterra inverses (Reuss) directly, each over its orientation distribution
`symmetrize`; the geometry does not enter them.

!!! warning "Reference-updating schemes need an isotropic running medium"
    Every ALV Hill kernel, order 2 and order 4 alike, is built for an
    **isotropic** reference: that is what decouples the time and space
    parts.  `Voigt`, `Reuss`, `Dilute`, `DiluteDual`, `MoriTanaka`,
    `Maxwell` and `PCW` evaluate it against the fixed, isotropic matrix and
    accept any inclusion shape or orientation.  `SelfConsistent` and
    `DifferentialScheme` evaluate it against their *running* estimate, so
    every inclusion phase must keep that estimate isotropic — either a
    spherical inclusion with an isotropic phase law, or an isotropic
    orientation average `symmetrize = :iso`, which is also what randomly
    oriented inclusions and cracks mean physically. A `LayeredSphere`, whose
    contribution is isotropic by construction, is always accepted.

    An RVE satisfying neither raises an `ArgumentError` naming the phase,
    rather than reading iso parameters off a matrix that is not isotropic.
    A non-isotropic ALV matrix is refused for the same reason.  The elastic
    pipeline has no such restriction — its Hill tensor is available for
    anisotropic references.
"""
function homogenize_alv(
        rve::RVE, scheme::HomogenizationScheme,
        prop::Symbol; times = nothing, kw...
    )
    grid = _alv_grid(rve, prop, times)
    t = grid.times
    R = MFH_Core._with_nested_cache() do
        MFH_Core._with_alv_grid(() -> _homogenize_alv_rve(rve, scheme, prop; times = t, kw...), t)
    end
    return _compact(_result_operator(R, grid), _candidate_classes(rve, prop, t))
end

# A list of operators written out, for the formulas on full matrices; typed even
# when empty (a cell of cracks alone).
_dense_list(Xs) = Matrix[Matrix(X) for X in Xs]

_result_operator(R::VolterraOperator, grid) = R
_result_operator(R::AbstractMatrix, grid) = VolterraOperator(R, grid)

# The reference of the closed-form schemes: stored by blocks when the pipeline has it so.
_reference(C_0, kw) = get(kw, :C_0_op, C_0)

# A phase property as the relaxation operator on the grid, in the class of its values.
_relaxation_operator(law::AbstractViscoLaw, grid::TimeGrid) = _as_relaxation(VolterraOperator(law, grid), visco_mode(law))
_relaxation_operator(R::VolterraOperator, grid::TimeGrid) = (_same_grid(R.grid, grid, "homogenize_alv"); R)
# An elastic tensor (a conductivity at order 2): the constant kernel C H(t - t′), block
# diagonal, in the class of C. More specific than a matrix: a second-order tensor is one.
_relaxation_operator(C::TensND.AbstractTens, grid::TimeGrid) = VolterraOperator(C, grid)
_relaxation_operator(C::TensND.AbstractTens{2}, grid::TimeGrid) = VolterraOperator(C, grid)
_relaxation_operator(M::AbstractMatrix, grid::TimeGrid) =
    _compact(VolterraOperator(_check_operator(M, grid.times, 6), grid), (_ISO4, _TI4_E3, _ORTHO4_CANON))
# A creep law is turned into the relaxation operator the schemes assume (a run-time mode).
_as_relaxation(R, mode::Symbol) = mode === :creep ? inv(R) : R

# A nested cell met inside an aging viscoelastic homogenization, on its grid.
MFH_Core._resolve_alv(cell, scheme, prop::Symbol, times; kw...) = homogenize_alv(cell, scheme, prop; times, kw...)

# The grid of a computation: given, or that of the operators among the phase properties.
_alv_grid(rve, prop, times) = TimeGrid(times)
function _alv_grid(rve, prop, ::Nothing)
    grids = [g for g in map(_grid_of, _raw_properties(rve, prop)) if g !== nothing]
    isempty(grids) && throw(
        ArgumentError("homogenize_alv: no time grid; give `times`, or a phase property that is a VolterraOperator")
    )
    for g in grids
        _same_grid(first(grids), g, "homogenize_alv")
    end
    return first(grids)
end

_grid_of(R::VolterraOperator) = R.grid
_grid_of(v) = nothing

# The property values of the members, as stored (a nested cell is not resolved here).
_raw_properties(cell, prop) = [
    MFH_Core.cell_container_property(cell, name, prop)
        for name in MFH_Core.cell_member_names(cell) if MFH_Core.cell_has_property(cell, name, prop)
]

# The classes a result is tried in, cheapest first: isotropic, those of the phase
# properties (a law's value, an operator's class), transversely isotropic about e₃,
# orthotropic in the canonical frame.
function _candidate_classes(rve, prop, times)
    ℬ = TensND.CanonicalBasis{3, Float64}()
    phase = Any[c for c in (_property_class(v, times) for v in _raw_properties(rve, prop)) if c isa _StructuredClass]
    return Any[
        TensISO{3}(1.0, 1.0), TensISO{3}(1.0), phase...,
        TensTI{4}(1.0, 1.0, 0.0, 1.0, 1.0, (0.0, 0.0, 1.0)), TensTI{2}(1.0, 1.0, (0.0, 0.0, 1.0)),
        TensOrtho{4, Float64}(ntuple(_ -> 1.0, 9), ℬ), TensOrtho{2, Float64}((1.0, 1.0, 1.0), ℬ),
    ]
end
_property_class(law::AbstractViscoLaw, times) = _class_of(visco_eval(law, first(times), first(times)))
_property_class(R::VolterraOperator, times) = R.class
_property_class(C::TensND.AbstractTens, times) = _class_of(C)
_property_class(v, times) = nothing

function _homogenize_alv_rve(
        rve::RVE, scheme::HomogenizationScheme,
        prop::Symbol; times::AbstractVector{<:Real}, kw...
    )
    # 1. Matrix kernel.  `homogenize_alv` does not go through `homogenize`, so
    #    it resolves the scheme's reference medium itself.
    m = matrix_name(scheme, rve)
    C_M_law = phase_property(rve, m, prop)
    C_M_law isa ALVProperty ||
        throw(ArgumentError("homogenize_alv: matrix property $prop is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))

    # Dispatch on the property order (2 vs 4) inferred from the sample.
    order = _alv_order_int(C_M_law, times)
    if order == 2
        return _homogenize_alv_order2(
            rve, scheme, prop;
            times = times, kw...
        )
    end

    # The reference as an operator stored by the blocks of its class, and written
    # out for the formulas that compute on full matrices (cracks).
    C_0_op = _relaxation_operator(C_M_law, TimeGrid(times))
    C_0 = Matrix(C_0_op)
    f_M = volume_fraction(rve, m)

    # 2. Loop on inclusions, separating SOLIDS (`VolumeFraction`) from
    #    CRACKS (`CrackDensity`).  Cracks contribute ΔC̃_crack to the
    #    numerator of the schemes (no volume → no denominator effect).
    incl_names = inclusion_phase_names(rve, m)
    # Allow `fractions` to carry whatever element type the RVE amounts
    # store — typically `Float64` but also `ForwardDiff.Dual` for autodiff
    # sensitivities via `set_param(rve, AmountParameter(...), Dual(...))`.
    T_amount = isempty(incl_names) ? Float64 :
        promote_type((typeof(amount_value(rve.amounts[n])) for n in incl_names)...)
    fractions = T_amount[]
    # Eltype-generic containers : per-phase matrices may carry a wider
    # element type than the matrix kernel (e.g. `ForwardDiff.Dual` geometry
    # parameters flowing through `tens_UA`/`tens_VA`).
    # Operators stored by blocks (`VolterraOperator`) or full matrices.
    contribs = AbstractMatrix[]
    A_duts = AbstractMatrix[]
    C_phases = AbstractMatrix[C_0_op]
    # The orientation average of each entry of `C_phases`, for the bounds.
    phase_syms = AbstractSymmetrize[phase_symmetrize(rve, m)]
    H_phases = AbstractMatrix[]   # per-phase Hill kernels (for Maxwell distribution)
    # crack_data tuple: (geom, density, sym, H_iso_full).  `H_iso_full` is
    # the iso-projected size-independent compliance contribution H̃ scaled
    # by the Budiansky concentration factor (`4π/3` for 3D penny / elliptic,
    # `π` for 2D ribbon).  Used by the ECHOES-form MT / Maxwell / PCW
    # multiplicative formula `C_eff = C_M · (I + Σ ε_c·(4π/3)·H_iso_c·C_M)^{-1}`.
    crack_data = Tuple{Any, Any, AbstractSymmetrize, AbstractMatrix}[]
    # Cracks-against-C_M sums (stiffness, and compliance for Reuss/DiluteDual),
    # added rather than accumulated in place so that a dual crack density
    # promotes them.
    ΔC_cracks_M = zero(C_0)
    ΔJ_cracks_M = zero(C_0)

    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        sym = phase_symmetrize(rve, name)
        if _is_crack_family(a)
            geom = ph.geometry
            geom isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("homogenize_alv: phase $name has CrackDensity but geometry $(typeof(geom)) is not a crack"))
            ε = a.value
            # Optional interface-stiffness laws (`:Rn` normal, `:Rt` tangential).
            Rn_law = get(ph.properties, :Rn, nothing)
            Rt_law = get(ph.properties, :Rt, nothing)
            # Compliance + stiffness contributions, with optional
            # interface-stiffness Sevostianov correction
            # `B̃_eff = B̃ ∘ (𝟙 + b·K ∘ B̃)^{-vol}`.
            H̃ = inverse_property_contribution_alv(
                geom, C_M_law, times;
                Rn = Rn_law, Rt = Rt_law
            )
            # Iso-projected, Budiansky-scaled H̃ for the ECHOES-form MT
            # denominator (cf. ECHOES `compute_visco_strain_Stress`).  The
            # `delta_inverse_property_alv` factor (4π/3 elliptic, π ribbon)
            # absorbs the Budiansky-O'Connell density convention.
            H̃_full = _maybe_symmetrize_alv(
                delta_inverse_property_alv(geom, H̃, ε),
                sym
            )
            push!(crack_data, (geom, ε, sym, H̃_full))
            Ñ = -(C_0 * H̃ * C_0)
            ΔC = delta_property_alv(geom, Ñ, ε)
            ΔC = _maybe_symmetrize_alv(ΔC, sym)
            ΔC_cracks_M = ΔC_cracks_M + ΔC
            ΔJ = delta_inverse_property_alv(geom, H̃, ε)
            ΔJ = _maybe_symmetrize_alv(ΔJ, sym)
            ΔJ_cracks_M = ΔJ_cracks_M + ΔJ
        else
            C_r_law = phase_property(rve, name, prop)
            C_r, A_dut, N_dut, P_r = _inclusion_alv_quantities(
                ph.geometry, scheme, C_r_law, C_M_law, C_0_op, times
            )
            A_dut = _maybe_symmetrize_alv(A_dut, sym)
            N_dut = _maybe_symmetrize_alv(N_dut, sym)
            push!(C_phases, C_r)
            push!(phase_syms, sym)
            push!(A_duts, A_dut)
            push!(contribs, N_dut)
            push!(H_phases, P_r)
            push!(fractions, _amount_value(rve, name))
        end
    end

    return _homogenize_alv_dispatch(
        rve, scheme, prop, times,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, m;
        phase_syms = phase_syms,
        crack_data = crack_data,
        ΔC_cracks_M = ΔC_cracks_M,
        ΔJ_cracks_M = ΔJ_cracks_M,
        C_0_op = C_0_op,
        kw...
    )
end

# ── Per-geometry inclusion quantities ───────────────────────────────────────

"""
    _inclusion_alv_quantities(geom, scheme, C_r_law, C_M_law, C_0, times)
        -> (C_r, A_dut, N_dut, P_r)

Compute the four ``6n\\times 6n`` matrices needed by the ALV scheme
dispatch for a single inclusion of geometry `geom`.  Default method
covers ellipsoidal geometries (Hill kernel + dilute formulas);
specializations for `LayeredSphere` use the layered-sphere recurrences, and
the `scheme` only for the stiffness a bound assigns to the sphere.
"""
function _inclusion_alv_quantities(
        geom, scheme, C_r_law,
        C_M_law::ALVProperty,
        C_0::VolterraOperator,
        times::AbstractVector{<:Real}
    )
    C_r_law isa ALVProperty ||
        throw(ArgumentError("homogenize_alv: phase property is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))
    # In the class of each: the phase's law, the reference's, the geometry's (a
    # spheroid: transversely isotropic about its axis); each operation in the
    # cheapest class that holds its operands. One inverse for both quantities.
    C_r = _relaxation_operator(C_r_law, C_0.grid)
    P_r = _hill_kernel_op(geom, C_0)
    A_dut = _in_class_algebra(_alg_dilute_concentration, (C_r, C_0, P_r), ())
    N_dut = (C_r - C_0) * A_dut
    return (C_r, A_dut, N_dut, P_r)
end

function _inclusion_alv_quantities(
        sphere::LayeredSphere, scheme, _C_r_law,
        C_M_law::ALVProperty,
        C_0::VolterraOperator,
        times::AbstractVector{<:Real}
    )
    # Per-layer moduli are stored inside the LayeredSphere geometry; the
    # phase-level C_r_law is ignored (we accept any placeholder so that
    # the existing `add_phase!` API still works).
    A_dut = gradient_gradient_concentration_alv(sphere, C_M_law, times)
    N_dut = property_contribution_alv(sphere, C_M_law, times)
    # A layered sphere has no single stiffness. The only consumer of `C_r` is
    # a bound, which takes its own average of the layer kernels.
    C_r = _layer_bound_alv(scheme, sphere, times)
    # Placeholder Hill kernel (not used by Dilute / MT / Maxwell paths).
    P_r = zeros(eltype(C_0), size(C_0)...)
    return (C_r, A_dut, N_dut, P_r)
end

# The 6n × 6n relaxation matrices of the layers of a layered sphere, and their
# volume fractions. A layer stored as an elastic tensor is a Heaviside law.
function _layer_kernels_alv(sphere::LayeredSphere{T, N}, times) where {T, N}
    Cs = map(1:N) do k
        C = layer_modulus(sphere, k)
        return _trapezoidal_relaxation(_as_visco_law(C), times, 6)
    end
    return Cs, [layer_volume_fraction(sphere, k) for k in 1:N]
end

# Voigt and Reuss averages of the layer kernels: the stiffness each bound
# assigns to a layered sphere, as `layer_stiffness_average` and
# `layer_compliance_average` do in elasticity.
_layer_voigt_alv(sphere::LayeredSphere, times) = voigt_alv(_layer_kernels_alv(sphere, times)...)
_layer_reuss_alv(sphere::LayeredSphere, times) = reuss_alv(_layer_kernels_alv(sphere, times)...)

# The average a scheme reads as the stiffness of a layered sphere: the Reuss
# bound inverts the average of the layer compliances, every other scheme takes
# the Voigt average (only the Voigt bound reads it).
_layer_bound_alv(::Reuss, sphere::LayeredSphere, times) = _layer_reuss_alv(sphere, times)
_layer_bound_alv(::HomogenizationScheme, sphere::LayeredSphere, times) = _layer_voigt_alv(sphere, times)

# Convenience: extract the scalar amount value (volume fraction or crack
# density) from the RVE.  Preserves the element type of the wrapper —
# e.g. returns `ForwardDiff.Dual` when the RVE was constructed via
# `set_param(rve, AmountParameter(...), Dual(…))` — so autodiff flows
# through the entire ALV pipeline.
function _amount_value(rve::RVE, name::Symbol)
    amount = rve.amounts[name]
    return amount.value
end

# ── Iso-symmetry detection for the scheme fast path ─────────────────────────
#
# When every (6n × 6n) block matrix supplied to the scheme step is in
# iso form, the scheme algebra reduces to two independent scalar n × n
# Volterra problems on (α, β).  This is ~108× cheaper for matrix-matrix
# products and ~18× for inversion compared to the generic block-LU
# `(6n × 6n)` path.

# ── Dispatch table on scheme types ──────────────────────────────────────────
#
# Crack handling.  Every dispatcher honors the optional kwargs
#    crack_data    :: Vector{Tuple{geom, density, sym}}
#    ΔC_cracks_M   :: pre-aggregated stiffness contribution against C̃_M
#    ΔJ_cracks_M   :: pre-aggregated compliance contribution against C̃_M
# computed once in `homogenize_alv` (see the matrix-reference cracks
# loop above).  When `isempty(crack_data)`, the iso/TI fast paths are
# attempted; otherwise the crack-aware generic path is used.

@inline _has_cracks(kw) = haskey(kw, :crack_data) && !isempty(kw[:crack_data])

function _homogenize_alv_dispatch(
        ::RVE, ::Voigt, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; phase_syms, kw...
    )
    # Voigt ignores cracks (zero-volume convention, mirroring elastic
    # `Schemes/voigt.jl`).  Result depends only on solid volume fractions,
    # each phase stiffness averaged over its orientation distribution.
    fs = [f_M; fractions]
    Cs = map(_maybe_symmetrize_alv, C_phases, phase_syms)
    return _in_cheapest_algebra((), (Cs,)) do a, Cs
        _alg_voigt(a, Cs, fs)
    end
end

function _homogenize_alv_dispatch(
        ::RVE, ::Reuss, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; phase_syms, kw...
    )
    # Reuss ignores cracks (same convention as elastic `Schemes/reuss.jl`).
    fs = [f_M; fractions]
    all(s -> s isa NoSymmetrize, phase_syms) && return _in_cheapest_algebra((), (C_phases,)) do a, Cs
        _alg_reuss(a, Cs, fs)
    end
    # The compliance of each phase is averaged over its orientation
    # distribution, not its stiffness: the bound averages compliances.
    Ss = map((C, sym) -> _maybe_symmetrize_alv(_alv_inv(C), sym), C_phases, phase_syms)
    return _in_cheapest_algebra((), (Ss,)) do a, Ss
        _alg_inv(a, _alg_voigt(a, Ss, fs))
    end
end

function _homogenize_alv_dispatch(
        ::RVE, ::Dilute, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # Cracks: additive — `C̃_dilute + ΔC̃_cracks_M`.
    _has_cracks(kw) && return _alg_dilute(_DenseALV{6}(), C_0, _dense_list(contribs), fractions) .+ kw[:ΔC_cracks_M]
    return _in_class_algebra((_reference(C_0, kw),), (contribs,)) do a, C₀, Ns
        _alg_dilute(a, C₀, Ns, fractions)
    end
end

function _homogenize_alv_dispatch(
        ::RVE, ::DiluteDual, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # The dual scheme accumulates in COMPLIANCE space, but the per-phase
    # data collected upstream are the STIFFNESS contributions Ñ.  Map them
    # first, with the same exact identity the crack branch uses:
    #     H̃_r = −J̃_M ∘ Ñ_r ∘ J̃_M ,   J̃_M = C̃_M^{-vol} .
    # For a homogeneous inclusion this is algebraically the compliance
    # contribution `(J_r − J_M) ∘ B̃^dil` of the elastic `dilute_dual.jl`
    # (expand `B̃^dil = C̃_r ∘ Ã^dil ∘ J̃_M` and use `J̃_M ∘ C̃_M = 𝟙`); for a
    # heterogeneous one it is the definition the elastic pipeline uses.
    # Feeding the raw Ñ to the compliance sum would add a stiffness to a
    # compliance — see the `glassy limit` / `elastic limit` tests.
    J_M = volterra_inverse(C_0; block_size = 6)
    contribs_J = [-(J_M * N̄ * J_M) for N̄ in contribs]

    # Cracks: solid compliance contributions and the crack ΔJ̃, inverted back.
    _has_cracks(kw) &&
        return volterra_inverse(_alg_dilute(_DenseALV{6}(), J_M, contribs_J, fractions) .+ kw[:ΔJ_cracks_M]; block_size = 6)
    return _in_cheapest_algebra((C_0,), (contribs_J,)) do a, C₀, Hs
        _alg_dilute_dual(a, C₀, Hs, fractions)
    end
end

function _homogenize_alv_dispatch(
        ::RVE, ::MoriTanaka, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    if !_has_cracks(kw)
        return _in_class_algebra((_reference(C_0, kw),), (A_duts, contribs)) do a, C₀, As, Ns
            _alg_mori_tanaka(a, C₀, As, Ns, fractions, f_M)
        end
    end
    # Crack-aware MT — ECHOES B·A^{-vol} form.
    #
    # The ECHOES MT body for a phase α reads
    #   strain_Strain_α = sym(A_α(C_M))·C_M     (solid)
    #   stress_Strain_α = sym(C_α·A_α(C_M))·C_M (solid)
    #   strain_Strain_c = ε·sym(H_c(C_M))·C_M   (crack — traction-free → no
    #                                            stress contribution)
    # accumulated as
    #   A_E = f_M·I·C_M + Σ_α f_α·strain_Strain_α + Σ_c strain_Strain_c
    #   B_E = f_M·C_M    + Σ_α f_α·stress_Strain_α
    # and `C_eff = B_E · A_E^{-vol}`.  Factoring `C_M` out on the right
    # (allowed when symmetrize is iso/TI/ortho-compatible with C_M)
    # reduces to the MFH additive form **plus** a non-zero crack
    # `Ã_crack = ε·sym(H_c)·C_M` term in the denominator accumulator.
    # That is the ECHOES-equivalent MT formula at finite density.
    # `H_full = ε·sym(H_c)` already includes the Budiansky factor and the
    # iso/TI projection.  Right-multiplied by `C_0` (Volterra product) it is
    # `ε·sym(H_c)·C_M` — the ECHOES `strain_Strain` term; summed, not
    # accumulated into a buffer of `C_0`'s type, so that a dual crack density
    # carries.
    A_crack_total = sum(H_full * C_0 for (_, _, _, H_full) in kw[:crack_data])
    contribs_aug = vcat(_dense_list(contribs), [kw[:ΔC_cracks_M]])
    A_duts_aug = vcat(_dense_list(A_duts), [A_crack_total])
    fractions_aug = vcat(fractions, [one(eltype(C_0))])
    return mori_tanaka_alv(C_0, A_duts_aug, contribs_aug, fractions_aug, f_M)
end

function _homogenize_alv_dispatch(
        rve::RVE, scheme::Maxwell, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # The Hill kernel is built on the RVE's *distribution shape*, as in the
    # elastic `Schemes.maxwell` — not on a hard-coded sphere, which would make
    # the same scheme answer differently on the elastic and the ALV path.
    #
    # `homogenize_alv` does not go through `validate_cell`, so the requirement
    # is resolved here as well; otherwise an undeclared shape would surface as
    # a `nothing` field access rather than as the explanatory error.
    C_M_law = phase_property(rve, matrix, prop)
    H_0 = hill_kernel(Schemes.distribution_shape(rve, scheme).shape, C_M_law, times)
    return _alv_maxwell_form(C_0, contribs, fractions, H_0, kw)
end

# The Maxwell form `C̃₀ + Ñ ∘ (𝟙 - H̃ ∘ Ñ)⁻¹`, shared by Maxwell and PCW; cracks
# enter the sum of the contributions.
function _alv_maxwell_form(C_0, contribs, fractions, H, kw)
    if _has_cracks(kw)
        contribs_aug = vcat(_dense_list(contribs), [kw[:ΔC_cracks_M]])
        fractions_aug = vcat(fractions, [one(eltype(C_0))])
        return maxwell_alv(C_0, contribs_aug, fractions_aug; H_0 = H)
    end
    return _in_class_algebra((_reference(C_0, kw), H), (contribs,)) do a, C₀, H₀, Ns
        _alg_maxwell(a, C₀, Ns, fractions, H₀)
    end
end

# Self-Consistent ALV: re-routes to `self_consistent_alv` (different
# computation flow, since each iteration recomputes the per-phase Hill
# kernels against the running estimate).
function _homogenize_alv_dispatch(
        rve::RVE, sc::SelfConsistent, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # SC reads cracks directly from the RVE; strip the pre-aggregated
    # crack kwargs, and the phase averages, meant for the simpler scheme
    # dispatchers.
    kw_filt = Iterators.filter(
        p -> !(
            p[1] in
                (:crack_data, :ΔC_cracks_M, :ΔJ_cracks_M, :phase_syms, :C_0_op)
        ),
        kw
    )
    return _self_consistent_alv(
        rve, prop; times = times, matrix = matrix,
        sc.options..., kw_filt...
    )
end

# Asymmetric Self-Consistent ALV.  Same ingredients as `SelfConsistent`
# but the iteration update is anchored on the matrix property C_M
# rather than the running estimate (cf. `schemes_alv_extra.jl`).
function _homogenize_alv_dispatch(
        rve::RVE, asc::AsymmetricSelfConsistent, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    kw_filt = Iterators.filter(
        p -> !(
            p[1] in
                (:crack_data, :ΔC_cracks_M, :ΔJ_cracks_M, :phase_syms, :C_0_op)
        ),
        kw
    )
    return asymmetric_self_consistent_alv(
        rve, prop; times = times, matrix = matrix,
        asc.options..., kw_filt...
    )
end

# Ponte-Castañeda & Willis ALV.  Algebraically identical to Maxwell in
# the single-shape case, but uses the `rve.distribution_shape` for the
# Hill kernel instead of a fixed sphere.
function _homogenize_alv_dispatch(
        rve::RVE, scheme::PonteCastanedaWillis, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    C_M_law = phase_property(rve, matrix, prop)
    dist = Schemes.distribution_shape(rve, scheme)
    dist isa UniformDistribution ||
        throw(ArgumentError("PCW-ALV: only UniformDistribution is currently supported"))
    H_d = hill_kernel(dist.shape, C_M_law, times)
    # PCW is Maxwell with the Hill kernel of the distribution shape.
    return _alv_maxwell_form(C_0, contribs, fractions, H_d, kw)
end

# Differential ALV — SciML ODE on the fictitious incorporation time τ.
function _homogenize_alv_dispatch(
        rve::RVE, sch::DifferentialScheme, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    return differential_alv(
        rve, prop; times = times, matrix = matrix,
        _diff_alv_options(sch)...
    )
end

# Solver options shared by the order-4 and order-2 ALV differential
# drivers.  Mirrors the elastic `DifferentialScheme` contract: the
# recognized keywords are read out, everything else is forwarded to
# the SciML `solve`.
function _diff_alv_options(sch::DifferentialScheme)
    return (
        nsteps = get(sch.options, :nsteps, 100),
        trajectory = sch.trajectory,
        abstol = get(sch.options, :abstol, 1.0e-8),
        reltol = get(sch.options, :reltol, 1.0e-6),
        alg = get(sch.options, :alg, nothing),
        formulation = get(sch.options, :formulation, :stiffness),
        solver_kwargs = Base.structdiff(
            sch.options,
            NamedTuple{(:nsteps, :abstol, :reltol, :alg, :formulation)}
        ),
    )
end
