# =============================================================================
#  api.jl — public entry point `cod_tensor` and `_kernel` method table
#  for flat cracks.  Dispatch via `Core._resolve_algo`.
# =============================================================================

"""
    _ti_aligned(C₀::TensTI{4}, ℬ_crack) -> Bool

Return `true` when the axis of transverse isotropy stored in `C₀` is
parallel to the third axis (the crack normal) of the crack-local basis
`ℬ_crack`.

NB: this is the actual TI symmetry axis (`TensND.axis(C₀)`), not the
third basis vector of `TensND.get_basis(C₀)` (which is always `e₃` for
a `TensTI{4}` since the underlying basis is canonical — the symmetry
axis is stored separately in the structured container).
"""
function _ti_aligned(C₀::TensND.TensTI{4}, ℬ_crack::TensND.AbstractBasis)
    axis_C = collect(TensND.axis(C₀))
    axis_n = TensND.components_canon(TensND.tens_basis(ℬ_crack, 3))
    # Both unit vectors, so parallel ⟺ (axis·n̂)² = 1. The **square**, not
    # `|axis·n̂|`: `Symbolics` does not fold `abs` of a literal, so on a
    # `TensTI{4, Num}` whose axis is exactly `e₃` the product came out as an
    # unevaluated `abs(1.0)` that nothing downstream could collapse — not even
    # `tsimplify`, which returns `-1 + abs(1.0)`. Squaring uses only `*`, which
    # does fold, and is mathematically identical here.
    return _is_unit_alignment(dot(axis_C, axis_n)^2)
end

# Both directions being unit vectors, `d = |axis · n̂|` equals 1 exactly when they
# are parallel.  A comparable scalar takes a tolerance; anything else is tested
# structurally, after simplification.
#
# The split is `Elliptic.is_hard_numeric`, **not** `d isa Real`: `Symbolics.Num`
# is `<: Real` but `isapprox` on it returns a symbolic inequality that throws in
# a boolean context, while `ForwardDiff.Dual` is `<: Real` and compares fine.
# Dispatching on `::Real` here made `cod_tensor` throw on a `TensTI{4, Num}`
# reference; guarding on the type made it throw on `Sym`.
function _is_unit_alignment(d)
    is_hard_numeric(d) && return isapprox(d, one(d); atol = 1.0e-10)
    # Structural branch. Test `d - 1 == 0` rather than `d == 1`: on a
    # `Symbolics.Num` the axis components carry `Float64` literals while `one(d)`
    # carries an `Int`, so `isequal(d, one(d))` is *false* for a genuinely
    # aligned axis. Subtracting first lets the literals collapse. Getting this
    # wrong sent an aligned `TensTI{4, Num}` down the numerical back-end, which
    # then failed inside StaticArrays instead of using the closed form.
    gap = tsimplify(d - one(d))
    verdict = iszero(gap)
    return verdict isa Bool ? verdict : false
end

# TI-aligned dispatch rules — refine Core dispatch for `AbstractCrack` + TensTI{4}.
# Explicit Val{:auto}, Val{:residues}, etc. methods are needed to disambiguate
# against the generic Core rules (which use the same Val{:auto} signature).
if isdefined(TensND, :TensTI)
    @eval function _ti_crack_dispatch(method::Symbol, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4})
        _ti_aligned(C₀, crack_basis(crack)) && return MFH_Core.Analytical()
        method === :decuhr && return MFH_Core.DECUHR()
        method === :nestedquadgk && return MFH_Core.NestedQuadGK()
        method === :residues && return MFH_Core.Residue()
        # Non-aligned : hand over to the shared anisotropic default, which
        # picks a cubature.  Deciding it here again would silently reinstate
        # the residue algorithm as a default — the very thing
        # `Core.dispatch.jl` centralizes to avoid.
        return MFH_Core._aniso_default_algo(C₀)
    end
    @eval MFH_Core._resolve_algo(::Val{:auto}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
        _ti_crack_dispatch(:auto, crack, C₀)
    @eval MFH_Core._resolve_algo(::Val{:analytical}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
        _ti_crack_dispatch(:analytical, crack, C₀)
    @eval MFH_Core._resolve_algo(::Val{:residues}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
        _ti_crack_dispatch(:residues, crack, C₀)
    @eval MFH_Core._resolve_algo(::Val{:decuhr}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
        _ti_crack_dispatch(:decuhr, crack, C₀)
    @eval MFH_Core._resolve_algo(::Val{:nestedquadgk}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
        _ti_crack_dispatch(:nestedquadgk, crack, C₀)
end

# ── Public API ──────────────────────────────────────────────────────────────

"""
    cod_tensor(crack, C₀; method=:auto, abstol=1e-8, reltol=1e-6, maxiters=100_000)
        -> Tens{2,3}

Size-independent crack-opening-displacement (COD) tensor
``\\mathbf B`` defined from the average displacement jump on the crack
surface through

```
(1/S) ∫_S [u] dS = b · B · (Σ · n̂),
```

where ``b`` is the semi-minor in-plane semi-axis (``b\\ge c\\to 0``).
``\\mathbf B`` factors the crack compliance tensor as

- Elliptic: ``\\mathbb H = \\tfrac{3}{4}\\,\\hat{\\mathbf n}
  \\stackrel{s}{\\otimes}\\mathbf B\\stackrel{s}{\\otimes}\\hat{\\mathbf n}``;
- Ribbon:   ``\\mathbb H = \\tfrac{2}{\\pi}\\,\\hat{\\mathbf n}
  \\stackrel{s}{\\otimes}\\mathbf B\\stackrel{s}{\\otimes}\\hat{\\mathbf n}``;

with ``\\mathbb H = \\lim_{c/b\\to 0}(c/b)\\,\\mathbb Q^{-1}`` and
``\\mathbb Q = \\mathbb C - \\mathbb C:\\mathbb P:\\mathbb C`` the
second Hill tensor
([kachanov1992](@cite),
 [sevostianov2002](@cite),
 [barthelemyIJES2021](@cite)).  The elliptic and ribbon
factorizations are related by ``\\mathbf B^{\\mathcal R} =
\\tfrac{3\\pi}{8}\\,\\lim_{\\eta\\to 0}\\mathbf B^{\\mathcal E}``.

For isotropic or aligned-TI matrices the kernel is analytical
([hoenig1978](@cite),
 [kanaun2009](@cite));
for arbitrarily anisotropic matrices the limit ``c/b\\to 0`` is
resolved numerically through the first-order Taylor term of the Hill
tensor ([barthelemyIJSS2009](@cite)).

Alias: [`B_tensor`](@ref).
"""
function cod_tensor(
        crack::MFH_Core.AbstractCrack,
        C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        method::Symbol = :auto,
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    algo = MFH_Core._resolve_algo(Val(method), crack, C₀)
    B = _kernel(crack, C₀, algo; abstol = abstol, reltol = reltol, maxiters = maxiters)
    K_interface === nothing && return B
    return _apply_interface_stiffness(B, K_interface, semi_minor(crack))
end

const B_tensor = cod_tensor

"""
    _apply_interface_stiffness(B::Tens{2,3}, K::Tens{2,3}, b) -> Tens{2,3}

Apply the Sevostianov spring-like-interface correction to the COD
2-tensor ``\\mathbf B`` :

```
B_eff = (b · K + B^{-1})^{-1} = B · (I + b · K · B)^{-1}
```

Limits :
* ``\\mathbf K = 0`` (traction-free)        →  `B_eff = B`
* ``\\mathbf K \\to \\infty`` (rigid bond)   →  `B_eff = 0`

Reference : [sevostianov2002](@cite).
"""
function _apply_interface_stiffness(
        B::TensND.AbstractTens{2, 3},
        K::TensND.AbstractTens{2, 3}, b::Real
    )
    B_M = TensND.get_array(B)        # 3 × 3
    K_M = TensND.get_array(K)
    I3 = Matrix{eltype(B_M)}(LinearAlgebra.I, 3, 3)
    KB = Matrix(K_M) * Matrix(B_M)
    M = I3 + b .* KB
    B_eff_M = Matrix(B_M) / M
    # Symmetrize to remove rounding drift.
    B_eff_M = (B_eff_M + B_eff_M') ./ 2
    return TensND.TensCanonical(B_eff_M)
end

# Order-2 (conductivity) — thermal COD scalar
"""
    cod_tensor(crack, K₀::AbstractTens{2,3}; method=:auto, kw...) -> Real

Size-independent **thermal crack-opening-displacement scalar** ``b``
for a flat crack in a conductor of 2nd-order conductivity tensor
``\\mathbf K_0``.  Analog of the elasticity COD tensor: in the 2nd-
order problem, the temperature jump across the crack is scalar and
only the normal component of the heat flux drives it, so a single
scalar captures the full crack flexibility.  The associated
size-independent resistivity contribution ``\\mathbf R = `` [`compliance_contribution`](@ref)`(crack, K₀)` is

```
R = (3/4) b · n̂ ⊗ n̂   (elliptic)
R = (2/π) b · n̂ ⊗ n̂   (ribbon)
```

The rank-1 direction is the crack normal ``\\hat{\\mathbf n}`` for *any*
``\\mathbf K_0``: the null space of
``\\mathbf K_0-\\mathbf K_0\\mathbf P(0)\\mathbf K_0`` is spanned by it.
Apply [`delta_resistivity`](@ref) to recover the dilute resistivity
correction ``\\Delta\\mathbf R = (4\\pi/3)\\varepsilon^{3\\mathrm d}\\mathbf R``
(elliptic) or ``\\Delta\\mathbf R = \\pi\\,\\varepsilon^{2\\mathrm d}\\mathbf R``
(ribbon).

``b`` is normalized exactly like the elastic ``\\mathbf B`` — by the in-plane
half-width — so ``b = \\chi/(b\\Lambda)`` with ``\\chi^{\\mathcal E} = 2/3`` and
``\\chi^{\\mathcal R} = \\pi/4``. Because the order-2 acoustic form is a
*scalar*, the closed form holds for **every** anisotropy, through a 2×2
eigenvalue problem on ``\\mathrm{adj}\\,\\mathbf K_0``; see the theory page
`docs/src/theory/thermal_cracks.md` and its derivation script
`scripts/16_cod_symbolic_thermal.jl`.

!!! warning "These values changed in v0.4.0"
    The thermal closed forms up to v0.3.2 were too small by ``4\\pi/(3\\eta)``
    (elliptic) and ``\\pi^{2}/4`` (ribbon).
"""
function cod_tensor(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3};
        α_interface::Union{Nothing, Real} = nothing,
        method::Symbol = :auto,
        kw...
    )
    b_th = _cod_thermal(crack, K₀)
    α_interface === nothing && return b_th
    # Sevostianov correction in the conductivity case (scalar form)
    #   b_eff = b / (1 + a · α · b),  a = semi_minor.
    #   α → 0   ⇒ b_eff = b (free crack);
    #   α → ∞   ⇒ b_eff = 0 (perfectly bonded interface).
    a = semi_minor(crack)
    return b_th / (1 + a * α_interface * b_th)
end

# Analytical dispatch: iso vs aniso.
_cod_thermal(crack::EllipticCrack, K₀::TensND.TensISO{2, 3}) =
    _cod_iso_ellipse_thermal(crack, MFH_Core.extract_iso_conductivity(K₀))

_cod_thermal(crack::RibbonCrack, K₀::TensND.TensISO{2, 3}) =
    _cod_iso_ribbon_thermal(crack, MFH_Core.extract_iso_conductivity(K₀))

_cod_thermal(crack::EllipticCrack, K₀::TensND.AbstractTens{2, 3}) =
    _cod_aniso_ellipse_thermal(crack, K₀)

_cod_thermal(crack::RibbonCrack, K₀::TensND.AbstractTens{2, 3}) =
    _cod_aniso_ribbon_thermal(crack, K₀)

"""
    compliance_contribution(crack, K₀::AbstractTens{2,3}; kw...) -> Tens{2,3}

Size-independent **crack resistivity contribution tensor** ``\\mathbf R``
(thermal analog of the elasticity [`compliance_contribution`](@ref)):

- Elliptic crack:  ``\\mathbf R = \\tfrac{3}{4}\\,b\\,
  \\hat{\\mathbf w}\\otimes\\hat{\\mathbf w}``.
- Ribbon crack:    ``\\mathbf R = \\tfrac{2}{\\pi}\\,b\\,
  \\hat{\\mathbf w}\\otimes\\hat{\\mathbf w}``.

with ``b = `` [`cod_tensor`](@ref)`(crack, K₀)` the thermal COD scalar
and ``\\hat{\\mathbf w}\\parallel\\mathbf K_0^{-1/2}\\hat{\\mathbf n}``
(reduces to ``\\hat{\\mathbf n}`` for iso / aligned-TI matrices).
Apply [`delta_resistivity`](@ref)`(crack, R, ε)` to obtain the dilute
resistivity correction ``\\Delta\\mathbf R``.
"""
function compliance_contribution(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    b = cod_tensor(crack, K₀; kw...)
    return _resistivity_from_b(crack, b, K₀)
end

# ── Level 2: _kernel methods ─────────────────────────────────────────────────

# Analytical — isotropic matrix
function _kernel(crack::EllipticCrack, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...)
    E, ν = MFH_Core.extract_iso_moduli(C₀)
    return _cod_iso_ellipse(crack, E, ν)
end

function _kernel(crack::RibbonCrack, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...)
    E, ν = MFH_Core.extract_iso_moduli(C₀)
    return _cod_iso_ribbon(crack, E, ν)
end

# Analytical — TI matrix aligned with n̂
if isdefined(TensND, :TensTI)
    @eval function _kernel(crack::EllipticCrack, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...)
        n̂ = TensND.tens_basis(crack_basis(crack), 3)
        nt = MFH_Core.extract_ti_moduli(C₀, n̂)
        return _cod_ti_ellipse(crack, nt.E, nt.H, nt.ν₁, nt.ν₂, nt.Γ)
    end

    @eval function _kernel(crack::RibbonCrack, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...)
        n̂ = TensND.tens_basis(crack_basis(crack), 3)
        nt = MFH_Core.extract_ti_moduli(C₀, n̂)
        return _cod_ti_ribbon(crack, nt.E, nt.H, nt.ν₁, nt.ν₂, nt.Γ)
    end
end

# Residue — anisotropic matrix
function _kernel(crack::EllipticCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...)
    return _cod_elliptic_numerical(
        crack, C₀, _residue_backend;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

function _kernel(crack::RibbonCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...)
    return _cod_ribbon_numerical(
        crack, C₀,
        (Carr, ξ, n̂; kw2...) -> _Qnn_star_residue(Carr, ξ, n̂);
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

# DECUHR — anisotropic matrix
function _kernel(crack::EllipticCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.DECUHR; kw...)
    return _cod_elliptic_decuhr_direct(
        crack, C₀;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

function _kernel(crack::RibbonCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.DECUHR; kw...)
    return _cod_ribbon_numerical(
        crack, C₀, _decuhr_backend;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

# NestedQuadGK — anisotropic matrix (historical nested-1D-QuadGK path,
# formerly shipped under the DECUHR name).
function _kernel(crack::EllipticCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.NestedQuadGK; kw...)
    return _cod_elliptic_nestedquadgk_direct(
        crack, C₀;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

function _kernel(crack::RibbonCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.NestedQuadGK; kw...)
    return _cod_ribbon_numerical(
        crack, C₀, _nestedquadgk_backend;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end
