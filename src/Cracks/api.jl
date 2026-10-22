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
third basis vector of `TensND.get_basis(C₀)` (which is always ``\\underline{e}_3`` for
a `TensTI{4}` since the underlying basis is canonical — the symmetry
axis is stored separately in the structured container).

The test is TensND's `isparallel`: a tolerance on comparable numbers, an exact
structural test on symbolic ones (no `dot`, no `abs`: they conjugate a `Sym`, and
`Symbolics` does not fold `abs` of a literal).
"""
_ti_aligned(C₀::TensND.TensTI{4}, ℬ_crack::TensND.AbstractBasis) =
    TensND.isparallel(TensND.axis(C₀), TensND.tens_basis(ℬ_crack, 3))

# TI-aligned dispatch rules — refine Core dispatch for `AbstractCrack` + TensTI{4}.
# Explicit Val{:auto}, Val{:residues}, etc. methods are needed to disambiguate
# against the generic Core rules (which use the same Val{:auto} signature).
function _ti_crack_dispatch(method::Symbol, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4})
    # The closed form when it applies and was not ruled out; an explicit
    # numerical method is honored, as for an ellipsoid (`Elasticity.jl`).
    if (method === :auto || method === :analytical) && _ti_aligned(C₀, crack_basis(crack))
        return MFH_Core.Analytical()
    end
    method === :decuhr && return MFH_Core.DECUHR()
    method === :nestedquadgk && return MFH_Core.NestedQuadGK()
    method === :residues && return MFH_Core.Residue()
    # Non-aligned : hand over to the shared anisotropic default, which
    # picks a cubature.  Deciding it here again would silently reinstate
    # the residue algorithm as a default — the very thing
    # `Core.dispatch.jl` centralizes to avoid.
    return MFH_Core._aniso_default_algo(C₀)
end
MFH_Core._resolve_algo(::Val{:auto}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
    _ti_crack_dispatch(:auto, crack, C₀)
MFH_Core._resolve_algo(::Val{:analytical}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
    _ti_crack_dispatch(:analytical, crack, C₀)
MFH_Core._resolve_algo(::Val{:residues}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
    _ti_crack_dispatch(:residues, crack, C₀)
MFH_Core._resolve_algo(::Val{:decuhr}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
    _ti_crack_dispatch(:decuhr, crack, C₀)
MFH_Core._resolve_algo(::Val{:nestedquadgk}, crack::MFH_Core.AbstractCrack, C₀::TensND.TensTI{4}) =
    _ti_crack_dispatch(:nestedquadgk, crack, C₀)

# ── Public API ──────────────────────────────────────────────────────────────

"""
    cod_tensor(crack, C₀; method=:auto, abstol=1e-8, reltol=1e-6, maxiters=100_000)
        -> Tens{2,3}

Size-independent crack-opening-displacement (COD) tensor
``\\boldsymbol{B}`` defined from the average displacement jump on the crack
surface through

```math
\\frac{1}{S}\\int_S [\\![\\underline{u}]\\!]\\,\\mathrm{d}S
= b\\,\\boldsymbol{B}\\cdot(\\boldsymbol{\\Sigma}\\cdot\\underline{n}),
```

where ``b`` is the semi-minor in-plane semi-axis (``b\\ge c\\to 0``).
``\\boldsymbol{B}`` factors the crack compliance tensor as

- Elliptic: ``\\mathbb{H} = \\tfrac{3}{4}\\,\\underline{n}
  \\stackrel{s}{\\otimes}\\boldsymbol{B}\\stackrel{s}{\\otimes}\\underline{n}``;
- Ribbon:   ``\\mathbb{H} = \\tfrac{2}{\\pi}\\,\\underline{n}
  \\stackrel{s}{\\otimes}\\boldsymbol{B}\\stackrel{s}{\\otimes}\\underline{n}``;

with ``\\mathbb{H} = \\lim_{c/b\\to 0}(c/b)\\,\\mathbb{Q}^{-1}`` and
``\\mathbb{Q} = \\mathbb{C} - \\mathbb{C}:\\mathbb{P}:\\mathbb{C}`` the
second Hill tensor
[kachanov1992, sevostianov2002, barthelemyIJES2021](@cite).  The elliptic and ribbon
factorizations are related by ``\\boldsymbol{B}^{\\mathcal{R}} =
\\tfrac{3\\pi}{8}\\,\\lim_{\\eta\\to 0}\\boldsymbol{B}^{\\mathcal{E}}``.

For isotropic or aligned-TI matrices the kernel is analytical
[hoenig1978, kanaun2009](@cite) with `method = :auto` or `:analytical`; an
explicit numerical `method` is honored. For arbitrarily anisotropic
matrices ``\\boldsymbol{B}`` is the inverse of an integral over the directions of
the crack plane of the reduced kernel
``\\hat{\\boldsymbol{Q}}^{\\star}_{nn}`` [barthelemyIJSS2026](@cite), itself a
line integral along the normal [kanaunlevin2009](@cite), computed numerically: by
nested adaptive quadrature (`:nestedquadgk`, the `:auto` choice), by the DECUHR
cubature (`:decuhr`), or with the inner integral by residues (`:residues`). The
flattening route of [barthelemyIJSS2009](@citet), the first-order term of the
Hill tensor of a flat ellipsoid in its aspect ratio, gives the same tensor.

Alias: [`B_tensor`](@ref).
"""
Base.@constprop :aggressive @inline function cod_tensor(
        crack::MFH_Core.AbstractCrack,
        C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        method::Symbol = :auto,
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    return _cod_tensor_entry(crack, C₀, K_interface, method, abstol, reltol, maxiters)
end

# Positional and inlined, as `_hill_tensor_entry`: Julia 1.12 propagates the
# constant default `method = :auto` for inference but compiled this body for a
# run-time `Symbol`, whose other branch returns an abstract type, and boxed the
# result — 64 bytes per call. Inlined, the literal reaches `_with_method`.
Base.@constprop :aggressive @inline function _cod_tensor_entry(
        crack, C₀, K_interface, method::Symbol, abstol, reltol, maxiters
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    B = MFH_Core._with_method(_cod_tensor_at, method, crack, C₀, abstol, reltol, maxiters)
    K_interface === nothing && return B
    return _apply_interface_stiffness(B, K_interface, semi_minor(crack))
end

_cod_tensor_at(m::Val, crack, C₀, abstol, reltol, maxiters) =
    _kernel(crack, C₀, MFH_Core._resolve_algo(m, crack, C₀); abstol, reltol, maxiters)

const B_tensor = cod_tensor

"""
    _apply_interface_stiffness(B::Tens{2,3}, K::Tens{2,3}, b) -> Tens{2,3}

Apply the Sevostianov spring-like-interface correction to the COD
2-tensor ``\\boldsymbol{B}``, with ``\\boldsymbol{K}`` the interface stiffness tensor:

```math
\\boldsymbol{B}^{\\mathrm{hom}}
= \\bigl(b\\,\\boldsymbol{K} + \\boldsymbol{B}^{-1}\\bigr)^{-1}
= \\boldsymbol{B}\\cdot\\bigl(\\boldsymbol{1} + b\\,\\boldsymbol{K}\\cdot\\boldsymbol{B}\\bigr)^{-1}
```

Limits :
* ``\\boldsymbol{K} = \\boldsymbol{0}`` (traction-free)        →  ``\\boldsymbol{B}^{\\mathrm{hom}} = \\boldsymbol{B}``
* ``\\boldsymbol{K} \\to \\infty`` (rigid bond)   →  ``\\boldsymbol{B}^{\\mathrm{hom}} = \\boldsymbol{0}``

Reference : [sevostianov2002](@citet).
"""
function _apply_interface_stiffness(
        B::TensND.AbstractTens{2, 3},
        K::TensND.AbstractTens{2, 3}, b::Real
    )
    # In the basis `B` is written in (the crack's): only `K`, a 3×3, changes
    # basis, and the result stays in that basis. `get_array(K)` was read there
    # as if it were, and the result was relabeled canonical, so an oriented
    # crack with an interface stiffness came back in the wrong frame.
    # The algebra below is matrix algebra, right in an orthonormal basis only:
    # the orthonormal one `B` is written in (read as it is), or its parent.
    ℬ = TensND.relevant_OrthonormalBasis(TensND.get_basis(B))
    B_M = Array(TensND.components(B, ℬ))
    K_M = Array(TensND.components(K, ℬ))
    M = LinearAlgebra.I + b .* (K_M * B_M)
    B_eff_M = B_M / M
    # Symmetrize to remove rounding drift.
    return TensND.Tens((B_eff_M + transpose(B_eff_M)) ./ 2, ℬ)
end

# Order-2 (conductivity) — thermal COD scalar
"""
    cod_tensor(crack, K₀::AbstractTens{2,3}; method=:auto, kw...) -> Real

Size-independent **thermal crack-opening-displacement scalar** ``b``
for a flat crack in a conductor of 2nd-order conductivity tensor
``\\boldsymbol{K}_0``.  Analog of the elasticity COD tensor: in the 2nd-
order problem, the temperature jump across the crack is scalar and
only the normal component of the heat flux drives it, so a single
scalar captures the full crack flexibility.  The associated
size-independent resistivity contribution ``\\boldsymbol{R}``, returned by [`inverse_property_contribution`](@ref)`(crack, K₀)`, is

```math
\\begin{aligned}
\\boldsymbol{R} &= \\tfrac{3}{4}\\,b\\,\\underline{n}\\otimes\\underline{n} &&(\\mathrm{elliptic}),\\\\
\\boldsymbol{R} &= \\tfrac{2}{\\pi}\\,b\\,\\underline{n}\\otimes\\underline{n} &&(\\mathrm{ribbon}).
\\end{aligned}
```

The rank-1 direction is the crack normal ``\\underline{n}`` for *any*
``\\boldsymbol{K}_0``: the null space of
``\\boldsymbol{K}_0-\\boldsymbol{K}_0\\cdot\\boldsymbol{P}(0)\\cdot\\boldsymbol{K}_0`` is spanned by it.
Apply [`delta_inverse_property`](@ref) to recover the dilute resistivity
correction ``\\Delta\\boldsymbol{R} = (4\\pi/3)\\,\\varepsilon^{3\\mathrm{d}}\\,\\boldsymbol{R}``
(elliptic) or ``\\Delta\\boldsymbol{R} = \\pi\\,\\varepsilon^{2\\mathrm{d}}\\,\\boldsymbol{R}``
(ribbon).

``b`` is normalized exactly like the elastic ``\\boldsymbol{B}`` — by the in-plane
half-width — so ``b = \\chi/(b\\Lambda)`` with ``\\chi^{\\mathcal{E}} = 2/3`` and
``\\chi^{\\mathcal{R}} = \\pi/4``. Because the order-2 acoustic form is a
*scalar*, the closed form holds for **every** anisotropy, through a 2×2
eigenvalue problem on ``\\mathrm{adj}\\,\\boldsymbol{K}_0``; see the theory page
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
    _cod_iso_ellipse_thermal(crack, first(TensND.get_data(K₀)))

_cod_thermal(crack::RibbonCrack, K₀::TensND.TensISO{2, 3}) =
    _cod_iso_ribbon_thermal(crack, first(TensND.get_data(K₀)))

_cod_thermal(crack::EllipticCrack, K₀::TensND.AbstractTens{2, 3}) =
    _cod_aniso_ellipse_thermal(crack, K₀)

_cod_thermal(crack::RibbonCrack, K₀::TensND.AbstractTens{2, 3}) =
    _cod_aniso_ribbon_thermal(crack, K₀)

"""
    inverse_property_contribution(crack, K₀::AbstractTens{2,3}; kw...) -> Tens{2,3}

Size-independent **crack resistivity contribution tensor** ``\\boldsymbol{R}``
(thermal analog of the elasticity [`inverse_property_contribution`](@ref)):

- Elliptic crack:  ``\\boldsymbol{R} = \\tfrac{3}{4}\\,b\\,
  \\underline{w}\\otimes\\underline{w}``.
- Ribbon crack:    ``\\boldsymbol{R} = \\tfrac{2}{\\pi}\\,b\\,
  \\underline{w}\\otimes\\underline{w}``.

with ``b`` the thermal COD scalar, returned by [`cod_tensor`](@ref)`(crack, K₀)`,
and ``\\underline{w}\\parallel\\boldsymbol{K}_0^{-1/2}\\cdot\\underline{n}``
(reduces to ``\\underline{n}`` for iso / aligned-TI matrices).
Apply [`delta_inverse_property`](@ref)`(crack, R, ε)` to obtain the dilute
resistivity correction ``\\Delta\\boldsymbol{R}``.
"""
function inverse_property_contribution(
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
    E, ν = TensND.E_nu(C₀)
    return _cod_iso_ellipse(crack, E, ν)
end

function _kernel(crack::RibbonCrack, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...)
    E, ν = TensND.E_nu(C₀)
    return _cod_iso_ribbon(crack, E, ν)
end

# Analytical — TI matrix aligned with n̂
function _kernel(crack::EllipticCrack, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...)
    E, ν₁, ν₂, H, Γ = TensND.arg_TI_Hoenig(inv(C₀))
    return _cod_ti_ellipse(crack, E, H, ν₁, ν₂, Γ)
end

function _kernel(crack::RibbonCrack, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...)
    E, ν₁, ν₂, H, Γ = TensND.arg_TI_Hoenig(inv(C₀))
    return _cod_ti_ribbon(crack, E, H, ν₁, ν₂, Γ)
end

# Residue — anisotropic matrix
function _kernel(crack::EllipticCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...)
    MFH_Core._residues_in_floats(crack, C₀)
    return _cod_elliptic_numerical(
        crack, C₀, _residue_backend;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 100_000)
    )
end

function _kernel(crack::RibbonCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...)
    MFH_Core._residues_in_floats(crack, C₀)
    return _cod_ribbon_numerical(
        crack, C₀, _residue_backend;
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

# The ribbon's reduced kernel is a single 1-D integral, and DECUHR integrates
# over hyperrectangles of dimension two or more: it returned `Failure` on every
# ribbon. The integral is evaluated by Gauss–Kronrod, as `:residues` on a
# cylinder falls back to the quadrature that applies.
function _kernel(crack::RibbonCrack, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.DECUHR; kw...)
    return _cod_ribbon_numerical(
        crack, C₀, _nestedquadgk_backend;
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
