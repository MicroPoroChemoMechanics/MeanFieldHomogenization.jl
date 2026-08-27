# =============================================================================
#  compliance.jl — crack compliance / resistivity contribution tensors.
#
#  Public API returns the size-independent contribution tensor H (or R in
#  the thermal case), consistent with the Echoes convention.  Helpers
#  `delta_compliance` / `delta_resistivity` apply the Budiansky density
#  factor to recover the full dilute compliance correction ΔS (or ΔR).
# =============================================================================

"""
    compliance_contribution(crack, C₀; method=:auto, kw...) -> Tens{4,3}

Size-independent **crack compliance contribution tensor**
``\\mathbb H`` (Echoes convention).  Assembled from the COD tensor
``\\mathbf B = `` [`cod_tensor`](@ref) through the factorization

- Elliptic crack:  ``\\mathbb H = \\tfrac{3}{4}\\,\\hat{\\mathbf n}
  \\stackrel{s}{\\otimes}\\mathbf B\\stackrel{s}{\\otimes}\\hat{\\mathbf n}``.
- Ribbon crack:   ``\\mathbb H = \\tfrac{2}{\\pi}\\,\\hat{\\mathbf n}
  \\stackrel{s}{\\otimes}\\mathbf B\\stackrel{s}{\\otimes}\\hat{\\mathbf n}``.

The two geometric prefactors follow from the single definition
``\\mathbb H = \\lim_{c/b\\to 0}(c/b)\\,\\mathbb Q^{-1}
= (cS/V)\\,\\hat{\\mathbf n}\\stackrel{s}{\\otimes}\\mathbf B
\\stackrel{s}{\\otimes}\\hat{\\mathbf n}`` applied with
``S/V = 3/(4c)`` (elliptic, ``S=\\pi ab``, ``V=\\tfrac{4}{3}\\pi abc``)
or ``S/V = 2/(\\pi c)`` (ribbon, ``S=4ab``, ``V=2\\pi abc``,
``a\\to\\infty``).  The Kachanov–Echoes factorization of the elliptic
case is recovered for ``\\eta=1`` (penny).

Apply [`delta_compliance`](@ref)`(crack, H, ε)` to obtain the dilute
compliance correction ``\\Delta\\mathbb S``:

```
ΔS = (4π/3) ε³ᵈ H   (elliptic, ε³ᵈ = N a b²)
ΔS =    π   ε²ᵈ H   (ribbon,   ε²ᵈ = N b²)
```

See [kachanov1992](@cite),
[sevostianov2002](@cite),
[barthelemyIJES2021](@cite).
"""
function compliance_contribution(
        crack::MFH_Core.AbstractCrack,
        C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        kw...
    )
    B = cod_tensor(crack, C₀; K_interface = K_interface, kw...)
    return _compliance_from_B(crack, B)
end

"""
    _compliance_from_B_elliptic(crack, B) -> Tens{4,3}

Elliptic: ``\\mathbb H = \\tfrac{3}{4}(\\hat n ⊗ˢ \\mathbf B ⊗ˢ \\hat n)``.
"""
function _compliance_from_B_elliptic(crack::MFH_Core.AbstractCrack, B)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    T = eltype(B)
    return (3 * one(T) / 4) * (n̂ ⊗ˢ B ⊗ˢ n̂)
end

"""
    _compliance_from_B_ribbon(crack, B) -> Tens{4,3}

Ribbon: ``\\mathbb H = \\tfrac{2}{\\pi}(\\hat n ⊗ˢ \\mathbf B ⊗ˢ \\hat n)``.
"""
function _compliance_from_B_ribbon(crack::MFH_Core.AbstractCrack, B)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    T = eltype(B)
    return (2 * one(T) / T(π)) * (n̂ ⊗ˢ B ⊗ˢ n̂)
end

# ── Shape-trait dispatch ─────────────────────────────────────────────────────
#
#  The B → ℍ algebra and the Budiansky prefactors depend only on the *shape
#  family*, not on the concrete struct.  Keying them on `shape_trait` rather
#  than on `EllipticCrack` / `RibbonCrack` means a user-defined crack — e.g.
#  one whose COD tensor comes out of a finite-element solve — inherits the
#  whole contribution chain (ℍ, ℕ, 𝐑, 𝐍_K, the bundle and the four `delta_*`)
#  by implementing `cod_tensor` alone, provided it declares
#  `shape_trait` ∈ {`Penny`, `EllipticShape`, `Ribbon`}.

function _unsupported_crack_shape(S, what)
    return throw(
        ArgumentError(
            "no $what rule for crack shape trait `$S`. Declare " *
                "`shape_trait` as one of `Penny`, `EllipticShape` (elliptical " *
                "algebra) or `Ribbon` (ribbon algebra), or add your own method."
        )
    )
end

_compliance_from_B(crack::MFH_Core.AbstractCrack, B) =
    _compliance_from_B(MFH_Core.shape_trait(crack), crack, B)
_compliance_from_B(::Type{<:CrackShape}, crack, B) = _compliance_from_B_elliptic(crack, B)
_compliance_from_B(::Type{Ribbon}, crack, B) = _compliance_from_B_ribbon(crack, B)
_compliance_from_B(S::Type, _crack, _B) = _unsupported_crack_shape(S, "compliance")

# =============================================================================
#  Thermal (2nd-order) — R from the scalar COD b
# =============================================================================

"""
    _resistivity_from_b_elliptic(crack, b, K₀) -> Tens{2,3}

Elliptic: ``\\mathbf R = \\tfrac{3}{4}\\,b\\,\\hat{\\mathbf n}
\\otimes\\hat{\\mathbf n}``.

The rank-1 direction is always the crack normal ``\\hat{\\mathbf n}``
(null space of ``\\mathbf K_0 - \\mathbf K_0\\mathbf P(0)\\mathbf K_0``
with the correct V-formula Hill tensor limit
``\\mathbf P(0) = \\hat{\\mathbf n}\\otimes\\hat{\\mathbf n}/k_{nn}``).
"""
function _resistivity_from_b_elliptic(crack::MFH_Core.AbstractCrack, b, _K₀)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    T = eltype(n̂)
    return (T(3) / T(4) * b) * (n̂ ⊗ n̂)
end

"""
    _resistivity_from_b_ribbon(crack, b, K₀) -> Tens{2,3}

Ribbon: ``\\mathbf R = \\tfrac{2}{\\pi}\\,b\\,\\hat{\\mathbf n}
\\otimes\\hat{\\mathbf n}``.
"""
function _resistivity_from_b_ribbon(crack::MFH_Core.AbstractCrack, b, _K₀)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    T = eltype(n̂)
    return (T(2) / T(π) * b) * (n̂ ⊗ n̂)
end

_resistivity_from_b(crack::MFH_Core.AbstractCrack, b, K₀) =
    _resistivity_from_b(MFH_Core.shape_trait(crack), crack, b, K₀)
_resistivity_from_b(::Type{<:CrackShape}, crack, b, K₀) =
    _resistivity_from_b_elliptic(crack, b, K₀)
_resistivity_from_b(::Type{Ribbon}, crack, b, K₀) =
    _resistivity_from_b_ribbon(crack, b, K₀)
_resistivity_from_b(S::Type, _crack, _b, _K₀) = _unsupported_crack_shape(S, "resistivity")

# =============================================================================
#  Budiansky density helpers — dilute compliance / resistivity corrections.
# =============================================================================

"""
    delta_compliance(crack, H, ε) -> Tens{4,3}

Dilute compliance correction ``\\Delta\\mathbb S`` of a family of
identical parallel cracks of Budiansky density ``\\varepsilon`` from the
size-independent contribution tensor ``\\mathbb H``:

- Elliptic: ``\\Delta\\mathbb S = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm d}\\,\\mathbb H``
  with ``\\varepsilon^{3\\mathrm d} = N a b^{2}``.
- Ribbon:   ``\\Delta\\mathbb S = \\pi\\,\\varepsilon^{2\\mathrm d}\\,\\mathbb H``
  with ``\\varepsilon^{2\\mathrm d} = N b^{2}``.

See [budiansky1976](@cite),
[sevostianov2002](@cite).
"""
delta_compliance(crack::MFH_Core.AbstractCrack, H, ε) =
    crack_density_factor(crack) * ε * H

"""
    delta_resistivity(crack, R, ε) -> Tens{2,3}

Dilute resistivity correction ``\\Delta\\mathbf R`` of a family of
identical parallel cracks of Budiansky density ``\\varepsilon`` from the
size-independent contribution tensor ``\\mathbf R``:

- Elliptic: ``\\Delta\\mathbf R = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm d}\\,\\mathbf R``.
- Ribbon:   ``\\Delta\\mathbf R = \\pi\\,\\varepsilon^{2\\mathrm d}\\,\\mathbf R``.
"""
delta_resistivity(crack::MFH_Core.AbstractCrack, R, ε) =
    crack_density_factor(crack) * ε * R

"""
    crack_density_factor(crack) -> Real

Geometric prefactor relating a **density-like amount** to the dilute
effective correction, i.e. the single number shared by the four
three-argument seams
[`delta_compliance`](@ref), [`delta_stiffness`](@ref),
[`delta_conductivity`](@ref) and [`delta_resistivity`](@ref):

```
Δ = crack_density_factor(crack) · ε · X
```

- `4π/3` for an elliptical (or penny-shaped) crack, whose Budiansky density
  is ``\\varepsilon^{3\\mathrm d} = N a b^{2}``;
- `π` for a ribbon crack, whose density is ``\\varepsilon^{2\\mathrm d} = N b^{2}``.

Dispatched on [`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait), so a
user-defined crack inherits the right prefactor for free.  A flat morphology
with a different density convention overrides this single method rather than
the four `delta_*` ones.

See [budiansky1976](@cite).
"""
crack_density_factor(crack::MFH_Core.AbstractCrack) =
    _crack_density_factor(MFH_Core.shape_trait(crack))

_crack_density_factor(::Type{<:CrackShape}) = 4 * Float64(π) / 3
_crack_density_factor(::Type{Ribbon}) = Float64(π)
_crack_density_factor(S::Type) = _unsupported_crack_shape(S, "density-factor")

# =============================================================================
#  Stiffness / conductivity contribution for cracks (API symmetry with
#  ellipsoids).  For a flat crack the dilute expansion of `inv(S₀+ΔS) - C₀`
#  at first order in the density ε gives N_crack = -C₀ : H : C₀, so both
#  contribution flavors are related by a simple ± C₀ : (·) : C₀ mapping.
# =============================================================================

"""
    stiffness_contribution(crack, C₀; kw...) -> Tens{4,3}

Size-independent **crack stiffness contribution tensor**
``\\mathbb N = -\\mathbb C_0 : \\mathbb H : \\mathbb C_0``, where
``\\mathbb H`` is the crack compliance contribution tensor
([`compliance_contribution`](@ref)).  Provided for API symmetry with
solid inclusions; the associated dilute correction is
``\\Delta\\mathbb C = (4\\pi/3)\\,\\varepsilon^{3\\mathrm d}\\,\\mathbb N``
(elliptic) or ``\\pi\\,\\varepsilon^{2\\mathrm d}\\,\\mathbb N`` (ribbon),
assembled by [`delta_stiffness`](@ref)`(crack, N, ε)`.
"""
function MFH_Core.stiffness_contribution(
        crack::MFH_Core.AbstractCrack,
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    H = compliance_contribution(crack, C₀; kw...)
    return -(C₀ ⊡ H ⊡ C₀)
end

"""
    conductivity_contribution(crack, K₀; kw...) -> Tens{2,3}

Size-independent **crack conductivity contribution tensor**
``\\mathbf N_K = -\\mathbf K_0 \\cdot \\mathbf R \\cdot \\mathbf K_0``.
"""
function MFH_Core.conductivity_contribution(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    R = compliance_contribution(crack, K₀; kw...)
    return -(K₀ ⋅ R ⋅ K₀)
end

"""
    delta_stiffness(crack, N, ε) -> Tens{4,3}

Dilute stiffness correction ``\\Delta\\mathbb C`` from the
size-independent crack contribution tensor ``\\mathbb N`` and the
Budiansky density ``\\varepsilon``:

- Elliptic: ``\\Delta\\mathbb C = \\tfrac{4\\pi}{3}\\,\\varepsilon^{3\\mathrm d}\\,\\mathbb N``.
- Ribbon:   ``\\Delta\\mathbb C = \\pi\\,\\varepsilon^{2\\mathrm d}\\,\\mathbb N``.
"""
MFH_Core.delta_stiffness(crack::MFH_Core.AbstractCrack, N, ε) =
    crack_density_factor(crack) * ε * N

"""
    delta_conductivity(crack, N_K, ε) -> Tens{2,3}

Dilute conductivity correction from the crack contribution tensor
``\\mathbf N_K`` and the Budiansky density, with the same prefactors as
[`delta_stiffness`](@ref).
"""
MFH_Core.delta_conductivity(crack::MFH_Core.AbstractCrack, N, ε) =
    crack_density_factor(crack) * ε * N

# =============================================================================
#  Bundled compliance + stiffness contribution
#
#  `stiffness_contribution(crack, C₀)` above is DEFINED as
#  `-(C₀ ⊡ compliance_contribution(crack, C₀) ⊡ C₀)`, and
#  `compliance_contribution` is `_compliance_from_B(crack, cod_tensor(...))`.
#  A scheme that needs both therefore solves `cod_tensor` — the most
#  expensive object in the package for an anisotropic matrix (adaptive
#  quadrature with a full residue/polynomial solve at every node) — twice.
#
#  One `cod_tensor` yields both, exactly: the expressions below are copied
#  verbatim, so the results are bitwise identical.
# =============================================================================

"""
    compliance_and_stiffness_contribution(crack, C₀; K_interface, kw...) -> (H, N)
    compliance_and_stiffness_contribution(crack, K₀; kw...)              -> (R, N_K)

Bundled `(compliance_contribution, stiffness_contribution)` — resp.
`(compliance_contribution, conductivity_contribution)` — for a flat crack,
sharing the single [`cod_tensor`](@ref) solve.

Bitwise identical to calling the two functions separately.
"""
function compliance_and_stiffness_contribution(
        crack::MFH_Core.AbstractCrack,
        C₀::TensND.AbstractTens{4, 3};
        K_interface::Union{Nothing, TensND.AbstractTens{2, 3}} = nothing,
        kw...
    )
    B = cod_tensor(crack, C₀; K_interface = K_interface, kw...)
    H = _compliance_from_B(crack, B)
    return (H, -(C₀ ⊡ H ⊡ C₀))
end

function compliance_and_stiffness_contribution(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    b = cod_tensor(crack, K₀; kw...)
    R = _resistivity_from_b(crack, b, K₀)
    return (R, -(K₀ ⋅ R ⋅ K₀))
end
