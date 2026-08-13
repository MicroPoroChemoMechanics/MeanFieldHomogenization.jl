# =============================================================================
#  cod_analytical_thermal.jl — closed-form thermal COD scalars.
#
#  Analog of `cod_analytical.jl` for the 2nd-order (conductivity /
#  diffusion) problem.  Returns a **scalar** `b` instead of a symmetric
#  2nd-order tensor: since the temperature jump across a flat crack is a
#  scalar driven only by the normal component of the heat flux, a single
#  scalar suffices.  The associated resistivity contribution is then
#  assembled as
#
#      R = (3/4) b · n̂ ⊗ n̂   (elliptic crack)
#      R = (2/π) b · n̂ ⊗ n̂   (ribbon crack)
#
#  in [`compliance.jl`](src/Cracks/compliance.jl) — the same geometric
#  factors as the elastic ℍ, because they come from `cS/V` and know
#  nothing about the physics.  Mathematical derivation and reference
#  conventions live in
#  [`docs/src/theory/thermal_cracks.md`](docs/src/theory/thermal_cracks.md).
#
#  ── Normalization (the trap) ────────────────────────────────────────────
#  `b` is defined exactly like the elastic 𝐁: through the crack-surface
#  average of the jump, normalized by the in-plane HALF-WIDTH `b`,
#
#      ⟨⟦T⟧⟩_I / b = b_scalar · (𝛔·n̂)   with   𝛔 ≡ -q ,
#
#  so `b = χ / (bΛ)` with `χᴱ = 2/3`, `χᴿ = π/4`, exactly as in
#  elasticity.  Up to v0.3.2 these formulas carried a different
#  normalization and were too small by `4π/(3η)` (elliptic) and `π²/4`
#  (ribbon); see the CHANGELOG for v0.4.0.  Three independent checks pin
#  the values below — the ξₙ-integral chain of
#  `scripts/16_cod_symbolic_thermal.jl`, the flattening limit
#  `lim_{ω→0} ω·Λ⁻¹` (`test/Cracks/test_thermal.jl`), and the textbook
#  temperature jump of an insulating penny crack.
# =============================================================================

# -----------------------------------------------------------------------------
#  ISO matrix
# -----------------------------------------------------------------------------

"""
    _cod_iso_ellipse_thermal(c::EllipticCrack, k₀) -> Real

Closed-form thermal COD scalar ``b`` of an elliptic crack of aspect
ratio ``\\eta = b/a`` in an isotropic conductor ``\\mathbf K_0 =
k_0\\,\\mathbf 1``:

```
b = 4 / (3 · k₀ · 𝓔_η) ,   𝓔_η = 𝓔(√(1-η²))
```

with ``\\mathcal E_\\eta`` the complete elliptic integral of the second
kind ([Abramowitz & Stegun 1972](@cite abramowitz1972)). Penny-crack
limit ``\\eta=1``: ``b = 8/(3\\pi k_0)``, which is the surface average of
the textbook jump ``[\\![T]\\!](r) = \\frac{4\\sigma_n a}{\\pi k_0}
\\sqrt{1-r^2/a^2}`` divided by ``a``.

This is the ``\\mathbf K_0 = k_0\\mathbf 1`` case of
[`_cod_aniso_ellipse_thermal`](@ref) (there ``\\lambda_1 = k_0^2`` and
``\\eta' = \\eta``); it is kept as a specialization only to skip the
adjugate and the 2×2 eigenvalue problem.
"""
function _cod_iso_ellipse_thermal(c::EllipticCrack{T}, k₀) where {T <: Number}
    η = aspect_ratio(c)
    ℰ = ell_E(one(η) - η^2)
    return T(4) / (T(3) * k₀ * ℰ)
end

"""
    _cod_iso_ribbon_thermal(c::RibbonCrack, k₀) -> Real

Closed-form thermal COD scalar of a ribbon (tunnel) crack in an
isotropic conductor:

```
b = π / (2 k₀) .
```

The 2-D counterpart of the elliptic formula: the contour integral
collapses to the single direction ``\\hat{\\mathbf m}`` and the shape
coefficient is ``\\chi^{\\mathcal R} = \\pi/4`` instead of
``\\chi^{\\mathcal E} = 2/3``
([Sevostianov & Kachanov 2002](@cite sevostianov2002),
 [Kachanov 2018](@cite kachanov2018)).
"""
function _cod_iso_ribbon_thermal(c::RibbonCrack{T}, k₀) where {T <: Number}
    return T(π) / (T(2) * k₀)
end

# -----------------------------------------------------------------------------
#  Anisotropic matrix — the adjugate form
# -----------------------------------------------------------------------------

"""
    _adj3_sym(K) -> (a11, a22, a33, a12, a13, a23)

The six independent entries of the adjugate
``\\mathrm{adj}\\,\\mathbf K = \\det\\mathbf K\\;\\mathbf K^{-1}`` of a
**symmetric** 3×3, by cofactors. Division-free, hence stable at small
``\\det\\mathbf K`` and evaluable on `ForwardDiff.Dual` and symbolic scalars —
which `det` composed with `inv` would also be, but at the cost of a division
the crack formulas immediately cancel.
"""
@inline function _adj3_sym(K)
    return @inbounds (
        K[2, 2] * K[3, 3] - K[2, 3] * K[2, 3],      # a11
        K[1, 1] * K[3, 3] - K[1, 3] * K[1, 3],      # a22
        K[1, 1] * K[2, 2] - K[1, 2] * K[1, 2],      # a33
        K[1, 3] * K[2, 3] - K[1, 2] * K[3, 3],      # a12
        K[1, 2] * K[2, 3] - K[1, 3] * K[2, 2],      # a13
        K[1, 2] * K[1, 3] - K[1, 1] * K[2, 3],      # a23
    )
end

"""
    _cod_aniso_ellipse_thermal(c::EllipticCrack, K₀) -> Real

Closed-form thermal COD scalar of an elliptic crack in an **arbitrarily
anisotropic** conductor. Unlike elasticity, the order-2 problem needs no
symmetry assumption: the acoustic form
``\\underline\\xi\\cdot\\mathbf K_0\\cdot\\underline\\xi`` is a scalar, so the
kernel integral closes in every case,

```math
\\hat Q^{\\star}_{nn}(\\underline\\xi^{\\star})
= \\tfrac12\\sqrt{(\\underline n\\wedge\\underline\\xi^{\\star})\\cdot
                 \\mathrm{adj}\\,\\mathbf K_0\\cdot
                 (\\underline n\\wedge\\underline\\xi^{\\star})} .
```

On the crack contour this leaves the 2×2 form

```math
\\mathbf Q_2 = \\begin{pmatrix}
  \\eta^{2} A & -\\eta B\\\\ -\\eta B & C\\end{pmatrix},
\\quad
A = \\hat{\\mathbf m}\\cdot\\mathrm{adj}\\mathbf K_0\\hat{\\mathbf m},\\;
B = \\hat{\\mathbf m}\\cdot\\mathrm{adj}\\mathbf K_0\\hat{\\boldsymbol\\ell},\\;
C = \\hat{\\boldsymbol\\ell}\\cdot\\mathrm{adj}\\mathbf K_0\\hat{\\boldsymbol\\ell},
```

whose eigenvalues ``\\lambda_1\\ge\\lambda_2>0`` give

```math
b = \\frac{4}{3\\sqrt{\\lambda_1}\\,\\mathcal E_{\\eta'}},
\\qquad \\eta' = \\sqrt{\\lambda_2/\\lambda_1} .
```

So an anisotropic conductor behaves as an **isotropic** one of conductivity
``\\sqrt{\\lambda_1}`` around a crack of **effective** aspect ratio ``\\eta'``;
``\\eta'\\ne\\eta`` in general, so even a circular crack acquires an effective
ellipticity. Reduces to ``4/(3k_0\\mathcal E_\\eta)`` for
``\\mathbf K_0 = k_0\\mathbf 1`` and to
``b = 4/(3\\sqrt{k_tk_n}\\,\\mathcal E_\\eta)`` for a TI conductor aligned with
``\\hat{\\mathbf n}`` (the geometric mean of the two conductivities).

Being a 2×2 eigenvalue problem, this is closed form and **type-generic** —
`ForwardDiff.Dual` and symbolic scalars included. It replaces the
``\\mathbf K_0^{-1/2}`` route of [Giraud et al. 2019](@cite giraudMOM2019),
which is equivalent but needs `eigen` on a 3×3 and `svdvals`, and so is
restricted to `Float64`. Derived in
`scripts/16_cod_symbolic_thermal.jl`.
"""
function _cod_aniso_ellipse_thermal(c::EllipticCrack, K₀)
    η = aspect_ratio(c)

    # Frame consistency, the same rule as `_crack_local_frame` in the elastic
    # kernels: express K₀ in the crack basis, so that (ℓ̂, m̂, n̂) are that
    # basis' own (e₁, e₂, e₃) and no two quantities live in different frames.
    # `adj` is a tensor under rotation, so the adjugate of the rotated
    # components is the rotated adjugate — the order does not matter.
    Kloc = TensND.change_tens(K₀, crack_basis(c))
    adj = _adj3_sym(Kloc)

    A = adj[2]           # m̂·adj(K₀)·m̂
    B = adj[4]           # m̂·adj(K₀)·ℓ̂   (the (1,2) entry, adj being symmetric)
    C = adj[1]           # ℓ̂·adj(K₀)·ℓ̂

    # Eigenvalues of Q₂ = [η²A  -ηB; -ηB  C], ordered λ₁ ≥ λ₂ > 0.
    tr = η^2 * A + C
    dt = η^2 * (A * C - B^2)
    gap = sqrt(tr^2 - 4 * dt)
    λ₁ = (tr + gap) / 2
    λ₂ = (tr - gap) / 2

    return 4 / (3 * sqrt(λ₁) * ell_E(one(λ₁) - λ₂ / λ₁))
end

# -----------------------------------------------------------------------------
#  Anisotropic matrix — ribbon crack
# -----------------------------------------------------------------------------

"""
    _cod_aniso_ribbon_thermal(c::RibbonCrack, K₀) -> Real

Closed-form thermal COD scalar of a ribbon crack in an arbitrarily
anisotropic conductor.  The contour integral collapses to the single
direction ``\\hat{\\mathbf m}``, so
``\\hat Q^{\\star}_{nn}(\\hat{\\mathbf m})
= \\tfrac12\\sqrt{\\det(\\mathbf K_0|_{(\\hat{\\mathbf m},\\hat{\\mathbf n})})}``
and, with ``\\chi^{\\mathcal R} = \\pi/4``,

```
b = π / (2 · √det(K₀|_{(m̂,n̂)})) .
```

Only the 2×2 block of ``\\mathbf K_0`` restricted to the plane spanned by the
in-plane crack direction and the crack normal enters — the transverse plane of
the tunnel.  Reduces to ``b = \\pi/(2k_0)`` for an isotropic conductor.
"""
function _cod_aniso_ribbon_thermal(c::RibbonCrack, K₀)
    T_mat = eltype(K₀)

    # Express K₀ in the crack basis: indices 2, 3 give the (m̂, n̂) block.
    K₀_loc = TensND.change_tens(K₀, crack_basis(c))
    K_mm = T_mat(K₀_loc[2, 2])
    K_nn = T_mat(K₀_loc[3, 3])
    K_mn = T_mat(K₀_loc[2, 3])

    det_K⊥ = K_mm * K_nn - K_mn * K_mn
    return T_mat(π) / (T_mat(2) * sqrt(det_K⊥))
end
