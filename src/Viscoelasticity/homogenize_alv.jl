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
        v = try
            MFH_Core.cell_container_property(cell, name, prop)
        catch
            continue
        end
        _is_visco_value(v, prop) && return true
    end
    return false
end

_is_visco_value(v, ::Symbol) = v isa ViscoLaw
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

@inline function _iso_project_mandel66(M::AbstractMatrix)
    @assert size(M) == (6, 6)
    # Full top-left 3×3 sum (echoes `isotropify`) — Volterra blocks are
    # generally NOT major-symmetric, so M[1,2] and M[2,1] must be summed
    # separately (a `2·M[1,2]` shortcut is only valid for symmetric blocks).
    return MFH_Core.iso_average_mandel66(M)
end

"""
    _iso_project_blocks(M::AbstractMatrix) -> Matrix

Project every 6×6 Mandel block of a `(6n × 6n)` ALV matrix to its iso
component (Reynolds average over the orthogonal group), returning a
new `(6n × 6n)` block matrix whose every block is iso.  Equivalent to
the ECHOES `symmetrize=[ISO]` orientation-averaging projection
applied to each `(t_i, t_j)` block independently.
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
        a, b = _iso_project_mandel66(view(M, rows, cols))
        α[i, j] = a
        β[i, j] = b
    end
    return iso_blocks_from_params(α, β)
end

"""
    _ti_project_blocks(M::AbstractMatrix, axis) -> Matrix

Exact azimuthal average about `axis` of every 6×6 Mandel block of a
`(6n × 6n)` ALV matrix (echoes `visco_transverse_isotropify` counterpart).
Each block is averaged independently with
[`Core.ti_average_mandel66`](@ref); the non-major-symmetric content of the
blocks (ℓ₃ ≠ ℓ₄, antisymmetric couplings) is preserved, which the
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
        out[rows, cols] = MFH_Core.ti_average_mandel66(view(M, rows, cols), axis)
    end
    return out
end

"""
    _maybe_symmetrize_alv(M, sym) -> Matrix

Apply the orientation-averaging projection corresponding to `sym` to a
`(6n × 6n)` ALV block matrix : `NoSymmetrize` (passthrough),
`IsoSymmetrize` (block-by-block exact SO(3) average) and
`TISymmetrize(axis)` (block-by-block exact azimuthal average about the
axis).
"""
@inline _maybe_symmetrize_alv(M::AbstractMatrix, ::NoSymmetrize) = M
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

Exact SO(3) orientation average of every 3×3 block of a `(3n × 3n)`
order-2 ALV matrix.  The isotropic part of a 2-tensor is
`(tr B / 3) 𝟙`, so each block collapses onto its spherical part.
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

Exact azimuthal average about `axis` of every 3×3 block of a
`(3n × 3n)` order-2 ALV matrix.  Averaging a 2-tensor `B` over the
rotations about the unit vector `n̂` keeps three invariants — the axial
component `a_n = n̂·B·n̂`, the transverse mean
`a_t = (tr B − a_n) / 2`, and the axial antisymmetric part (the
component of `B` along `[n̂]×`, which commutes with those rotations) —
and averages everything else to zero:

    ⟨B⟩ = a_t (𝟙 − n̂⊗n̂) + a_n n̂⊗n̂ + c [n̂]×  ,   c = (B_skew : [n̂]×) / 2 .

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
    T = float(eltype(M))
    out = zeros(T, sz, sz)
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1)
        c = 3 * (j - 1)
        B = @view M[(r + 1):(r + 3), (c + 1):(c + 3)]
        tr_B = B[1, 1] + B[2, 2] + B[3, 3]
        a_n = zero(T)
        for k in 1:3, l in 1:3
            a_n += nv[k] * B[k, l] * nv[l]
        end
        a_t = (tr_B - a_n) / 2
        # c = (B_skew : [n̂]×) / 2, with [n̂]×_{kl} = -ε_{klm} n̂_m.
        cc = (
            (B[3, 2] - B[2, 3]) * nv[1] +
                (B[1, 3] - B[3, 1]) * nv[2] +
                (B[2, 1] - B[1, 2]) * nv[3]
        ) / 2
        for k in 1:3, l in 1:3
            nn = nv[k] * nv[l]
            δ = (k == l) ? one(T) : zero(T)
            out[r + k, c + l] = a_t * (δ - nn) + a_n * nn
        end
        # + c [n̂]×
        out[r + 1, c + 2] -= cc * nv[3]; out[r + 2, c + 1] += cc * nv[3]
        out[r + 2, c + 3] -= cc * nv[1]; out[r + 3, c + 2] += cc * nv[1]
        out[r + 3, c + 1] -= cc * nv[2]; out[r + 1, c + 3] += cc * nv[2]
    end
    return out
end

# The axis reaches us either as a plain 3-vector/tuple or as a TensND
# 1-tensor, exactly as `TISymmetrize` stores it.
function _alv_unit_axis3(axis)
    v = axis isa TensND.AbstractTens ? collect(TensND.get_array(axis)) :
        collect(axis)
    length(v) == 3 || throw(ArgumentError("ALV TI average: axis must have 3 components"))
    nrm = sqrt(sum(abs2, v))
    iszero(nrm) && throw(ArgumentError("ALV TI average: axis must be non-zero"))
    return v ./ nrm
end

"""
    _maybe_symmetrize_alv2(M, sym) -> Matrix

Order-2 counterpart of [`_maybe_symmetrize_alv`](@ref), for `(3n × 3n)`
ALV matrices whose blocks are 2-tensors.
"""
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, ::NoSymmetrize) = M
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, ::IsoSymmetrize) =
    _iso_project_blocks3(M)
@inline _maybe_symmetrize_alv2(M::AbstractMatrix, sym::TISymmetrize) =
    _ti_project_blocks3(M, sym.axis)

"""
    _trapezoidal_relaxation(law::ViscoLaw, times, B) -> Matrix

Build the discrete relaxation block matrix from a `ViscoLaw`, regardless
of whether the law is in `:relaxation` or `:creep` mode.  When the law
is `:creep`, the trapezoidal compliance matrix is inverted (block forward
substitution at `block_size = B`) to obtain the corresponding relaxation
matrix — the convention every ALV scheme assumes internally.

`B` is the block size (`6` for order-4 4-tensor / Mandel, `3` for
order-2 vector-tensor).
"""
function _trapezoidal_relaxation(
        law::ViscoLaw,
        times::AbstractVector{<:Real}, B::Int
    )
    M = trapezoidal_matrix(law, times)
    if visco_mode(law) === :creep
        return volterra_inverse(M; block_size = B)
    end
    return M
end

"""
    _alv_property_order(law::ViscoLaw, t) -> Int

Inspect the sample returned by `visco_eval(law, t, t)` and report the
tensor order (`2` for vector-tensor / 3×3, `4` for 4-tensor / 6×6
Mandel).  Used by [`homogenize_alv`](@ref) to dispatch between the
order-4 (stiffness / relaxation) and order-2 (conductivity / creep
admittance) pipelines.
"""
function _alv_property_order(law::ViscoLaw, t::Real)
    sample = visco_eval(law, t, t)
    if sample isa TensND.AbstractTens{2, 3}
        return 2
    elseif sample isa TensND.AbstractTens{4, 3}
        return 4
    elseif sample isa AbstractMatrix
        if size(sample) == (3, 3)
            return 2
        elseif size(sample) == (6, 6)
            return 4
        end
    end
    throw(ArgumentError("homogenize_alv: cannot infer ALV order from sample of type $(typeof(sample))"))
end

"""
    homogenize_alv(rve, scheme, prop::Symbol; times) -> Matrix

ALV pipeline: build the discrete block matrices of every phase, compute
the ALV Hill kernel and dilute concentration tensors, and dispatch on
`scheme` to the corresponding `_alv` scheme function.

The function dispatches on the **order of the matrix property** (read
once from the matrix `ViscoLaw` sample type):
  * order-4 (4-tensor / 6×6 Mandel kernel) → returns `(6n × 6n)`
    relaxation matrix following the standard Hill-kernel +
    dilute-concentration pipeline.
  * order-2 (2-tensor / 3×3 kernel; conductivity / diffusion /
    permittivity) → returns `(3n × 3n)` matrix via the order-2 ALV
    pipeline (`hill_kernel_order2`, time-space decoupling).

Supports two inclusion-geometry families (order-4 only):
  * Single-shape ellipsoidal (`Ellipsoid`, `Spheroid`): the standard
    Hill-kernel + dilute-concentration pipeline.
  * `LayeredSphere`: bulk + shear ALV recurrences (see
    [`bulk_localization_alv`](@ref) and
    [`shear_localization_alv`](@ref)) feed
    [`stiffness_contribution_alv`](@ref) and
    [`strain_strain_loc_alv`](@ref) directly — no Hill kernel is
    needed.  In this case `phase_property(rve, name, :C)` is ignored
    (the per-layer moduli stored in the geometry are used instead);
    pass any `ViscoLaw` (e.g. `heaviside_law(C_0)`) as a placeholder.

`symmetrize` is honored in both orders, with the projector of the matching
tensor order (6×6 Mandel blocks for order 4, 2-tensor blocks for order 2 —
see [`_maybe_symmetrize_alv`](@ref) and [`_maybe_symmetrize_alv2`](@ref)).
`Voigt` and `Reuss` average the phase matrices directly, so the geometry —
and hence `symmetrize` — does not enter them.

!!! warning "Reference-updating schemes need an isotropic running medium"
    Every ALV Hill kernel, order 2 and order 4 alike, is built for an
    **isotropic** reference: that is what decouples the time and space
    parts.  `Voigt`, `Reuss`, `Dilute`, `DiluteDual`, `MoriTanaka`,
    `Maxwell` and `PCW` evaluate it against the fixed, isotropic matrix and
    accept any inclusion shape or orientation.  `SelfConsistent` and
    `DifferentialScheme` evaluate it against their *running* estimate, so
    every inclusion phase must keep that estimate isotropic — either a
    spherical inclusion with an isotropic phase law (`LayeredSphere`
    qualifies, its contribution being isotropic by construction), or an
    isotropic orientation average `symmetrize = :iso`, which is also what
    randomly oriented inclusions and cracks mean physically.

    An RVE satisfying neither raises an `ArgumentError` naming the phase,
    rather than reading iso parameters off a matrix that is not isotropic.
    A non-isotropic ALV matrix is refused for the same reason.  The elastic
    pipeline has no such restriction — its Hill tensor is available for
    anisotropic references.
"""
function homogenize_alv(
        rve::RVE, scheme::HomogenizationScheme,
        prop::Symbol; times::AbstractVector{<:Real}, kw...
    )
    # 1. Matrix kernel.  `homogenize_alv` does not go through `homogenize`, so
    #    it resolves the scheme's reference medium itself.
    m = matrix_name(scheme, rve)
    C_M_law = phase_property(rve, m, prop)
    C_M_law isa ViscoLaw ||
        throw(ArgumentError("homogenize_alv: matrix property $prop is not a ViscoLaw"))

    # Dispatch on the property order (2 vs 4) inferred from the sample.
    order = _alv_property_order(C_M_law, first(times))
    if order == 2
        return _homogenize_alv_order2(
            rve, scheme, prop;
            times = times, kw...
        )
    end

    C_0 = _trapezoidal_relaxation(C_M_law, times, 6)
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
    contribs = Matrix[]
    A_duts = Matrix[]
    C_phases = Matrix[C_0]
    H_phases = Matrix[]   # per-phase Hill kernels (for Maxwell distribution)
    # crack_data tuple: (geom, density, sym, H_iso_full).  `H_iso_full` is
    # the iso-projected size-independent compliance contribution H̃ scaled
    # by the Budiansky concentration factor (`4π/3` for 3D penny / elliptic,
    # `π` for 2D ribbon).  Used by the ECHOES-form MT / Maxwell / PCW
    # multiplicative formula `C_eff = C_M · (I + Σ ε_c·(4π/3)·H_iso_c·C_M)^{-1}`.
    crack_data = Tuple{Any, Any, AbstractSymmetrize, AbstractMatrix}[]
    ΔC_cracks_M = zeros(eltype(C_0), size(C_0)...)   # cracks-against-C_M sum
    ΔJ_cracks_M = zeros(eltype(C_0), size(C_0)...)   # for Reuss/DiluteDual

    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        sym = phase_symmetrize(rve, name)
        if a isa CrackDensity
            geom = ph.geometry
            geom isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("homogenize_alv: phase $name has CrackDensity but geometry $(typeof(geom)) is not a crack"))
            ε = a.value
            # Optional interface-stiffness laws (`:Rn` normal, `:Rt` tangential).
            Rn_law = haskey(ph.properties, :Rn) ? ph.properties[:Rn] : nothing
            Rt_law = haskey(ph.properties, :Rt) ? ph.properties[:Rt] : nothing
            # Compliance + stiffness contributions, with optional
            # interface-stiffness Sevostianov correction
            # `B̃_eff = B̃ ∘ (𝟙 + b·K ∘ B̃)^{-vol}`.
            H̃ = compliance_contribution_alv(
                geom, C_M_law, times;
                Rn = Rn_law, Rt = Rt_law
            )
            # Iso-projected, Budiansky-scaled H̃ for the ECHOES-form MT
            # denominator (cf. ECHOES `compute_visco_strain_Stress`).  The
            # `delta_compliance_alv` factor (4π/3 elliptic, π ribbon)
            # absorbs the Budiansky-O'Connell density convention.
            H̃_full = _maybe_symmetrize_alv(
                delta_compliance_alv(geom, H̃, ε),
                sym
            )
            push!(crack_data, (geom, ε, sym, H̃_full))
            Ñ = -(C_0 * H̃ * C_0)
            ΔC = delta_stiffness_alv(geom, Ñ, ε)
            ΔC = _maybe_symmetrize_alv(ΔC, sym)
            ΔC_cracks_M .+= ΔC
            ΔJ = delta_compliance_alv(geom, H̃, ε)
            ΔJ = _maybe_symmetrize_alv(ΔJ, sym)
            ΔJ_cracks_M .+= ΔJ
        else
            C_r_law = phase_property(rve, name, prop)
            C_r, A_dut, N_dut, P_r = _inclusion_alv_quantities(
                ph.geometry, C_r_law, C_M_law, C_0, times
            )
            A_dut = _maybe_symmetrize_alv(A_dut, sym)
            N_dut = _maybe_symmetrize_alv(N_dut, sym)
            push!(C_phases, C_r)
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
        crack_data = crack_data,
        ΔC_cracks_M = ΔC_cracks_M,
        ΔJ_cracks_M = ΔJ_cracks_M,
        kw...
    )
end

# ── Per-geometry inclusion quantities ───────────────────────────────────────

"""
    _inclusion_alv_quantities(geom, C_r_law, C_M_law, C_0, times)
        -> (C_r, A_dut, N_dut, P_r)

Compute the four `(6n × 6n)` matrices needed by the ALV scheme
dispatch for a single inclusion of geometry `geom`.  Default method
covers ellipsoidal geometries (Hill kernel + dilute formulas);
specializations for `LayeredSphere` use the layered-sphere recurrences.
"""
function _inclusion_alv_quantities(
        geom, C_r_law,
        C_M_law::ViscoLaw,
        C_0::AbstractMatrix,
        times::AbstractVector{<:Real}
    )
    C_r_law isa ViscoLaw ||
        throw(ArgumentError("homogenize_alv: phase property is not a ViscoLaw"))
    C_r = _trapezoidal_relaxation(C_r_law, times, 6)
    P_r = hill_kernel(geom, C_M_law, times)
    # Iso fast path : if every input matrix is iso (typical for
    # spherical inclusions in iso ALV matrix), compute the dilute
    # concentration and contribution as TWO scalar n×n Volterra
    # problems and lift back to (6n × 6n) at the end.  Avoids the
    # generic block-LU on the 6n×6n `(𝟙 + P̃ ∘ ΔC̃)` system.
    if _is_iso_block(C_r) && _is_iso_block(C_0) && _is_iso_block(P_r)
        αβ_E = _iso_pair(C_r)
        αβ_0 = _iso_pair(C_0)
        αβ_P = _iso_pair(P_r)
        αβ_A_dut = dilute_concentration_alv_iso(αβ_E, αβ_0, αβ_P)
        αβ_N_dut = dilute_contribution_alv_iso(αβ_E, αβ_0, αβ_P)
        A_dut = _iso_blocks(αβ_A_dut)
        N_dut = _iso_blocks(αβ_N_dut)
    elseif _is_ti_block(C_r) && _is_ti_block(C_0) && _is_ti_block(P_r)
        # TI fast path: shared canonical axis e_3 across phase, matrix
        # and Hill kernel.  Reduces the dilute concentration to a
        # (2n)×(2n) block-Volterra inverse + 2 scalar Volterra inverses.
        ℓ_E = _ti_pair(C_r)
        ℓ_0 = _ti_pair(C_0)
        ℓ_P = _ti_pair(P_r)
        ℓ_A_dut = dilute_concentration_alv_ti(ℓ_E, ℓ_0, ℓ_P)
        ℓ_N_dut = dilute_contribution_alv_ti(ℓ_E, ℓ_0, ℓ_P)
        A_dut = _ti_blocks(ℓ_A_dut)
        N_dut = _ti_blocks(ℓ_N_dut)
    elseif _is_ortho_block(C_r) && _is_ortho_block(C_0) && _is_ortho_block(P_r)
        # Ortho fast path: shared canonical material frame across phase,
        # matrix and Hill kernel.  Reduces the dilute concentration to a
        # (3n)×(3n) block-Volterra inverse + 3 scalar Volterra inverses.
        o_E = _ortho_pair(C_r)
        o_0 = _ortho_pair(C_0)
        o_P = _ortho_pair(P_r)
        o_A_dut = dilute_concentration_alv_ortho(o_E, o_0, o_P)
        o_N_dut = dilute_contribution_alv_ortho(o_E, o_0, o_P)
        A_dut = _ortho_blocks(o_A_dut)
        N_dut = _ortho_blocks(o_N_dut)
    else
        A_dut = dilute_concentration_alv(C_r, C_0, P_r)
        N_dut = dilute_contribution_alv(C_r, C_0, P_r)
    end
    return (C_r, A_dut, N_dut, P_r)
end

function _inclusion_alv_quantities(
        sphere::LayeredSphere, _C_r_law,
        C_M_law::ViscoLaw,
        C_0::AbstractMatrix,
        times::AbstractVector{<:Real}
    )
    # Per-layer moduli are stored inside the LayeredSphere geometry; the
    # phase-level C_r_law is ignored (we accept any placeholder so that
    # the existing `add_phase!` API still works).
    A_dut = strain_strain_loc_alv(sphere, C_M_law, times)
    N_dut = stiffness_contribution_alv(sphere, C_M_law, times)
    # No single C_r is well-defined for a layered sphere ; expose the
    # dilute-effective stiffness (C_0 + N_dut) as a representative
    # monolithic estimate so the Voigt / Reuss code paths still type-check
    # (they remain non-physical for a layered inclusion).
    C_r = C_0 .+ N_dut
    # Placeholder Hill kernel (not used by Dilute / MT / Maxwell paths).
    P_r = zeros(eltype(C_0), size(C_0)...)
    return (C_r, A_dut, N_dut, P_r)
end

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

"""
    _try_iso_pairs(matrices) -> Vector{Tuple} or nothing

If every matrix in `matrices` passes the iso-form check
(`_is_iso_block`), return a `Vector` of `(α, β)` `n×n` parameter
tuples extracted from each.  Otherwise return `nothing`.

Used by the scheme fast paths to opt into the iso pipeline only when
all phases really are iso 4-tensors.
"""
function _try_iso_pairs(matrices::AbstractVector{<:AbstractMatrix})
    T = isempty(matrices) ? Float64 : eltype(matrices[1])
    isempty(matrices) && return Tuple{Matrix{T}, Matrix{T}}[]
    out = Vector{Tuple{Matrix{T}, Matrix{T}}}()
    for M in matrices
        _is_iso_block(M) || return nothing
        push!(out, _iso_pair(M))
    end
    return out
end

"""
    _try_ti_tuples(matrices) -> Vector{NTuple{6, Matrix}} or nothing

If every matrix passes the TI-form check (`_is_ti_block`), return a
`Vector` of 6-tuples of `n×n` Walpole parameter matrices extracted
from each.  Otherwise return `nothing`.

Iso block matrices automatically satisfy the TI test (iso ⊂ TI), so a
mixed iso/TI phase setup with the canonical axis works.
"""
function _try_ti_tuples(matrices::AbstractVector{<:AbstractMatrix})
    T = isempty(matrices) ? Float64 : eltype(matrices[1])
    isempty(matrices) && return NTuple{6, Matrix{T}}[]
    out = Vector{NTuple{6, Matrix{T}}}()
    for M in matrices
        _is_ti_block(M) || return nothing
        push!(out, _ti_pair(M))
    end
    return out
end

"""
    _try_ortho_tuples(matrices) -> Vector{NTuple{12, Matrix}} or nothing

If every matrix passes the ortho-form check (`_is_ortho_block`), return
a `Vector` of 12-tuples of `n×n` ortho parameter matrices extracted
from each.  Otherwise return `nothing`.

Iso and TI (axis = e₃) block matrices automatically satisfy the ortho
test, so a mixed iso/TI/ortho phase setup with the canonical material
frame works.
"""
function _try_ortho_tuples(matrices::AbstractVector{<:AbstractMatrix})
    T = isempty(matrices) ? Float64 : eltype(matrices[1])
    isempty(matrices) && return NTuple{12, Matrix{T}}[]
    out = Vector{NTuple{12, Matrix{T}}}()
    for M in matrices
        _is_ortho_block(M) || return nothing
        push!(out, _ortho_pair(M))
    end
    return out
end

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
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # Voigt ignores cracks (zero-volume convention, mirroring elastic
    # `Schemes/voigt.jl`).  Result depends only on solid volume fractions.
    iso = _try_iso_pairs(C_phases)
    if iso !== nothing
        αβ_eff = voigt_alv_iso(iso, [f_M; fractions])
        return _iso_blocks(αβ_eff)
    end
    ti = _try_ti_tuples(C_phases)
    if ti !== nothing
        ℓ_eff = voigt_alv_ti(ti, [f_M; fractions])
        return _ti_blocks(ℓ_eff)
    end
    ortho = _try_ortho_tuples(C_phases)
    if ortho !== nothing
        o_eff = voigt_alv_ortho(ortho, [f_M; fractions])
        return _ortho_blocks(o_eff)
    end
    return voigt_alv(C_phases, [f_M; fractions])
end

function _homogenize_alv_dispatch(
        ::RVE, ::Reuss, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # Reuss ignores cracks (same convention as elastic `Schemes/reuss.jl`).
    iso = _try_iso_pairs(C_phases)
    if iso !== nothing
        αβ_eff = reuss_alv_iso(iso, [f_M; fractions])
        return _iso_blocks(αβ_eff)
    end
    ti = _try_ti_tuples(C_phases)
    if ti !== nothing
        ℓ_eff = reuss_alv_ti(ti, [f_M; fractions])
        return _ti_blocks(ℓ_eff)
    end
    ortho = _try_ortho_tuples(C_phases)
    if ortho !== nothing
        o_eff = reuss_alv_ortho(ortho, [f_M; fractions])
        return _ortho_blocks(o_eff)
    end
    return reuss_alv(C_phases, [f_M; fractions])
end

function _homogenize_alv_dispatch(
        ::RVE, ::Dilute, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    if !_has_cracks(kw)
        iso_contribs = _try_iso_pairs(contribs)
        if iso_contribs !== nothing && _is_iso_block(C_0)
            αβ_0 = _iso_pair(C_0)
            αβ_eff = dilute_alv_iso(αβ_0, iso_contribs, fractions)
            return _iso_blocks(αβ_eff)
        end
        ti_contribs = _try_ti_tuples(contribs)
        if ti_contribs !== nothing && _is_ti_block(C_0)
            ℓ_0 = _ti_pair(C_0)
            ℓ_eff = dilute_alv_ti(ℓ_0, ti_contribs, fractions)
            return _ti_blocks(ℓ_eff)
        end
        ortho_contribs = _try_ortho_tuples(contribs)
        if ortho_contribs !== nothing && _is_ortho_block(C_0)
            o_0 = _ortho_pair(C_0)
            o_eff = dilute_alv_ortho(o_0, ortho_contribs, fractions)
            return _ortho_blocks(o_eff)
        end
        return dilute_alv(C_0, contribs, fractions)
    end
    # Cracks: additive — `C̃_dilute + ΔC̃_cracks_M`.
    return dilute_alv(C_0, contribs, fractions) .+ kw[:ΔC_cracks_M]
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

    if _has_cracks(kw)
        # Solid compliance contribs + crack ΔJ̃, then invert back.
        J_eff = copy(J_M)
        @inbounds for (f, H̄) in zip(fractions, contribs_J)
            @. J_eff += f * H̄
        end
        J_eff .+= kw[:ΔJ_cracks_M]
        return volterra_inverse(J_eff; block_size = 6)
    end
    iso_contribs = _try_iso_pairs(contribs_J)
    if iso_contribs !== nothing && _is_iso_block(C_0)
        αβ_0 = _iso_pair(C_0)
        αβ_eff = dilute_dual_alv_iso(αβ_0, iso_contribs, fractions)
        return _iso_blocks(αβ_eff)
    end
    ti_contribs = _try_ti_tuples(contribs_J)
    if ti_contribs !== nothing && _is_ti_block(C_0)
        ℓ_0 = _ti_pair(C_0)
        ℓ_eff = dilute_dual_alv_ti(ℓ_0, ti_contribs, fractions)
        return _ti_blocks(ℓ_eff)
    end
    ortho_contribs = _try_ortho_tuples(contribs_J)
    if ortho_contribs !== nothing && _is_ortho_block(C_0)
        o_0 = _ortho_pair(C_0)
        o_eff = dilute_dual_alv_ortho(o_0, ortho_contribs, fractions)
        return _ortho_blocks(o_eff)
    end
    return dilute_dual_alv(C_0, contribs_J, fractions)
end

function _homogenize_alv_dispatch(
        ::RVE, ::MoriTanaka, ::Symbol, ::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    if !_has_cracks(kw)
        iso_contribs = _try_iso_pairs(contribs)
        iso_A = _try_iso_pairs(A_duts)
        if iso_contribs !== nothing && iso_A !== nothing && _is_iso_block(C_0)
            αβ_0 = _iso_pair(C_0)
            αβ_eff = mori_tanaka_alv_iso(αβ_0, iso_A, iso_contribs, fractions, f_M)
            return _iso_blocks(αβ_eff)
        end
        ti_contribs = _try_ti_tuples(contribs)
        ti_A = _try_ti_tuples(A_duts)
        if ti_contribs !== nothing && ti_A !== nothing && _is_ti_block(C_0)
            ℓ_0 = _ti_pair(C_0)
            ℓ_eff = mori_tanaka_alv_ti(ℓ_0, ti_A, ti_contribs, fractions, f_M)
            return _ti_blocks(ℓ_eff)
        end
        ortho_contribs = _try_ortho_tuples(contribs)
        ortho_A = _try_ortho_tuples(A_duts)
        if ortho_contribs !== nothing && ortho_A !== nothing && _is_ortho_block(C_0)
            o_0 = _ortho_pair(C_0)
            o_eff = mori_tanaka_alv_ortho(o_0, ortho_A, ortho_contribs, fractions, f_M)
            return _ortho_blocks(o_eff)
        end
        return mori_tanaka_alv(C_0, A_duts, contribs, fractions, f_M)
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
    sz = size(C_0, 1)
    T = eltype(C_0)
    A_crack_total = zeros(T, sz, sz)
    @inbounds for (geom, ε, sym, H_full) in kw[:crack_data]
        # `H_full = ε·sym(H_c)` already includes the Budiansky factor and
        # the iso/TI projection.  Right-multiply by `C_0` (Volterra-product)
        # to land on `ε·sym(H_c)·C_M` — the ECHOES `strain_Strain` term.
        A_crack_total .+= H_full * C_0
    end
    contribs_aug = vcat(contribs, [kw[:ΔC_cracks_M]])
    A_duts_aug = vcat(A_duts, [A_crack_total])
    fractions_aug = vcat(fractions, [one(T)])
    return mori_tanaka_alv(C_0, A_duts_aug, contribs_aug, fractions_aug, f_M)
end

function _homogenize_alv_dispatch(
        rve::RVE, ::Maxwell, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    # The Hill kernel is built on the RVE's *distribution shape*, as in the
    # elastic `Schemes.maxwell` — not on a hard-coded sphere, which would make
    # the same scheme answer differently on the elastic and the ALV path.
    C_M_law = phase_property(rve, matrix, prop)
    H_0 = hill_kernel(rve.distribution_shape.shape, C_M_law, times)
    if !_has_cracks(kw)
        iso_contribs = _try_iso_pairs(contribs)
        if iso_contribs !== nothing && _is_iso_block(C_0) && _is_iso_block(H_0)
            αβ_0 = _iso_pair(C_0)
            αβ_H_0 = _iso_pair(H_0)
            αβ_eff = maxwell_alv_iso(αβ_0, iso_contribs, fractions, αβ_H_0)
            return _iso_blocks(αβ_eff)
        end
        ti_contribs = _try_ti_tuples(contribs)
        if ti_contribs !== nothing && _is_ti_block(C_0) && _is_ti_block(H_0)
            ℓ_0 = _ti_pair(C_0)
            ℓ_H_0 = _ti_pair(H_0)
            ℓ_eff = maxwell_alv_ti(ℓ_0, ti_contribs, fractions, ℓ_H_0)
            return _ti_blocks(ℓ_eff)
        end
        ortho_contribs = _try_ortho_tuples(contribs)
        if ortho_contribs !== nothing && _is_ortho_block(C_0) && _is_ortho_block(H_0)
            o_0 = _ortho_pair(C_0)
            o_H_0 = _ortho_pair(H_0)
            o_eff = maxwell_alv_ortho(o_0, ortho_contribs, fractions, o_H_0)
            return _ortho_blocks(o_eff)
        end
        return maxwell_alv(C_0, contribs, fractions; H_0 = H_0)
    end
    # Crack-aware Maxwell : append cracks to the contribution sum (Σ).
    contribs_aug = vcat(contribs, [kw[:ΔC_cracks_M]])
    fractions_aug = vcat(fractions, [1.0])
    return maxwell_alv(C_0, contribs_aug, fractions_aug; H_0 = H_0)
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
    # crack kwargs that were meant for the simpler scheme dispatchers.
    kw_filt = Iterators.filter(
        p -> !(
            p[1] in
                (:crack_data, :ΔC_cracks_M, :ΔJ_cracks_M)
        ),
        kw
    )
    return self_consistent_alv(
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
                (:crack_data, :ΔC_cracks_M, :ΔJ_cracks_M)
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
        rve::RVE, ::PonteCastanedaWillis, prop::Symbol,
        times::AbstractVector,
        C_0, C_phases, A_duts, contribs,
        H_phases, fractions, f_M, matrix::Symbol; kw...
    )
    C_M_law = phase_property(rve, matrix, prop)
    dist = rve.distribution_shape
    dist isa UniformDistribution ||
        throw(ArgumentError("PCW-ALV: only UniformDistribution is currently supported"))
    H_d = hill_kernel(dist.shape, C_M_law, times)
    if !_has_cracks(kw)
        iso_contribs = _try_iso_pairs(contribs)
        if iso_contribs !== nothing && _is_iso_block(C_0) && _is_iso_block(H_d)
            αβ_0 = _iso_pair(C_0)
            αβ_H = _iso_pair(H_d)
            αβ_eff = maxwell_alv_iso(αβ_0, iso_contribs, fractions, αβ_H)
            return _iso_blocks(αβ_eff)
        end
        ti_contribs = _try_ti_tuples(contribs)
        if ti_contribs !== nothing && _is_ti_block(C_0) && _is_ti_block(H_d)
            ℓ_0 = _ti_pair(C_0)
            ℓ_H = _ti_pair(H_d)
            ℓ_eff = maxwell_alv_ti(ℓ_0, ti_contribs, fractions, ℓ_H)
            return _ti_blocks(ℓ_eff)
        end
        ortho_contribs = _try_ortho_tuples(contribs)
        if ortho_contribs !== nothing && _is_ortho_block(C_0) && _is_ortho_block(H_d)
            o_0 = _ortho_pair(C_0)
            o_H = _ortho_pair(H_d)
            o_eff = maxwell_alv_ortho(o_0, ortho_contribs, fractions, o_H)
            return _ortho_blocks(o_eff)
        end
        return pcw_alv(C_0, contribs, fractions; H_dist = H_d)
    end
    # Crack-aware PCW : same as Maxwell with rve distribution shape.
    contribs_aug = vcat(contribs, [kw[:ΔC_cracks_M]])
    fractions_aug = vcat(fractions, [1.0])
    return pcw_alv(C_0, contribs_aug, fractions_aug; H_dist = H_d)
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
# `OrdinaryDiffEq.solve`.
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
