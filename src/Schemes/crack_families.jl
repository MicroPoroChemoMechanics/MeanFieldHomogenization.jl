# =============================================================================
#  crack_families.jl — per-family decomposition of the macroscopic compliance.
#
#  WHY THIS EXISTS.  A scheme returns one tensor, `C_hom`.  A model that lets
#  fractures evolve needs more than that: it needs to know how much of the
#  macroscopic compliance *each* crack family is responsible for, because that
#  is what fixes the family's average opening.  Writing, for family `i` of
#  normal `n̂_i`,
#
#      S_hom = S_solid + Σ_i (4π/3) d_i 𝕊_i ,
#
#  the average strain carried by family `i` under a macroscopic stress `Σ` is
#  `f_i ⟨ε⟩_i = (4π/3) d_i 𝕊_i : Σ`, and since `f_i = (4π/3) d_i ω_i` with a
#  constant radius, `⟨ε⟩_{i,nn} = Δω_i / ω_i` gives the aperture rate
#
#      Δω_i = n̂_i · (𝕊_i : ΔΣ) · n̂_i .                                    (★)
#
#  This is the short, checkable form of eq. (10) of Barthélémy & Daniel
#  (ARMA 2011).  Its virtue is that the *same* 𝕊_i appears in the assembly of
#  `S_hom` and in the aperture update, so a single identity — the one this file
#  guarantees — validates both at once.  Transcribing eq. (10) literally would
#  instead be an independent, unchecked chain of operators.
#
#  THE SUBTLETY.  `𝕊_i` is NOT simply `compliance_contribution(crack, C_hom)`.
#  Each scheme post-multiplies ℍ_i by its own concentration correction, because
#  the stress seen by an inclusion in the Eshelby auxiliary problem is not the
#  macroscopic one.  The correction is derived here from the scheme body itself
#  rather than copied from the paper, and `crack_family_residual` measures the
#  resulting identity so a wrong factor cannot pass unnoticed.
# =============================================================================

"""
    crack_family_compliances(rve, scheme, C_hom; property = :C, kw...)

Decompose the macroscopic compliance of `rve` into a solid part and one
contribution per crack family, such that

```math
\\mathbb{S}^{\\rm hom} = \\mathbb{S}^{\\rm solid}
  + \\sum_i \\frac{4\\pi}{3}\\,d_i\\,\\mathbb{S}_i .
```

Returns a `NamedTuple`:

- `solid` — the solid part ``\\mathbb{S}^{\\rm solid}`` (for a single-matrix RVE,
  the matrix compliance ``\\mathbb{S}_{\\rm s}``);
- `families` — a `Dict{Symbol,…}` mapping each crack phase name to its
  ``\\mathbb{S}_i``, **normalized per unit** ``(4\\pi/3)\\,d_i``, so that the
  scaled contribution is recovered with
  [`delta_compliance`](@ref MeanFieldHomogenization.Core.delta_compliance).

`C_hom` **must** be the estimate that `scheme` itself returned for this `rve`
(`homogenize(rve, scheme, property)`): the decomposition is an identity of the
scheme *at its own fixed point*, and evaluating it anywhere else is meaningless.
[`crack_family_residual`](@ref) checks exactly that.

The ``\\mathbb{S}_i`` are what drives an aperture update — see the derivation of
``(\\star)`` at the top of this file, and
[`MeanFieldHomogenization.Constitutive`](@ref MeanFieldHomogenization.Constitutive),
which consumes them.

# Supported schemes

| Scheme | ``\\mathbb{S}^{\\rm solid}`` | correction |
|---|---|---|
| [`SelfConsistent`](@ref) | ``\\langle\\mathbb{A}\\rangle : \\langle\\mathbb{C}\\mathbb{A}\\rangle^{-1}`` | ``\\mathbb{C}^{\\rm hom} : \\langle\\mathbb{C}\\mathbb{A}\\rangle^{-1}`` |
| [`MoriTanaka`](@ref) | ``\\mathbb{S}_{\\rm s}`` | none (identity) |

For Mori-Tanaka the identity ``\\mathbb{S}^{\\rm MT} = \\mathbb{S}_{\\rm s} + \\sum_i
(4\\pi/3) d_i \\mathbb{H}_i(\\mathbb{C}_{\\rm s})`` is exact — a two-line consequence of
the `𝔹 : 𝔸⁻¹` body — but **only when the matrix is the sole phase carrying
volume**. An RVE with solid inclusions *and* cracks is rejected rather than
answered approximately.

!!! warning "Unsymmetrized families only"
    ``(\\star)`` is a statement about one orientation. A crack phase declared
    with `symmetrize = :iso` or `:ti` has already had its ℍ averaged over an
    orbit of orientations, and no single ``\\underline{n}_i`` remains to project onto.
    Such a phase raises an `ArgumentError` here instead of returning a plausible
    but meaningless tensor. Discrete fracture families — the ARMA setting — are
    unsymmetrized by construction.

See also [`crack_family_residual`](@ref), [`homogenize`](@ref).
"""
function crack_family_compliances(
        rve::RVE, scheme::HomogenizationScheme,
        C_hom::TensND.AbstractTens{4, 3}; property::Symbol = :C, kw...
    )
    return throw(
        ArgumentError(
            "crack_family_compliances is not implemented for $(typeof(scheme)); " *
                "supported schemes are SelfConsistent and MoriTanaka (the latter for " *
                "an RVE whose only volumetric phase is the matrix). The decomposition " *
                "has to be derived from each scheme body separately — see " *
                "src/Schemes/crack_families.jl."
        )
    )
end

function crack_family_compliances(
        rve::RVE, ::SelfConsistent,
        C_hom::TensND.AbstractTens{4, 3}; property::Symbol = :C, kw...
    )
    names = _crack_family_names(rve)
    # At the SC fixed point the body `C_hom = B_E : A_E⁻¹` with
    #   A_E = ⟨A⟩ : S_hom + Σ_i (4π/3) d_i ℍ_i ,   B_E = ⟨CA⟩ : S_hom
    # rearranges — multiply `A_E = S_hom : B_E` on the right by C_hom, then by
    # ⟨CA⟩⁻¹ — into
    #   S_hom = ⟨A⟩:⟨CA⟩⁻¹  +  (Σ_i (4π/3) d_i ℍ_i) : C_hom : ⟨CA⟩⁻¹ ,
    # which is the decomposition, with the post-factor read off directly.
    A_avg, CA_avg = _sc_solid_averages(rve, C_hom, property; kw...)
    inv_CA = inv(CA_avg)
    W = C_hom ⊡ inv_CA
    families = Dict{Symbol, Any}(
        name => _crack_family_H(rve, name, property, C_hom; kw...) ⊡ W
            for name in names
    )
    return (solid = A_avg ⊡ inv_CA, families = families)
end

function crack_family_compliances(
        rve::RVE, scheme::MoriTanaka,
        C_hom::TensND.AbstractTens{4, 3}; property::Symbol = :C, kw...
    )
    m = matrix_name(scheme, rve)
    names = _crack_family_names(rve)
    solid_inclusions = [
        n for n in inclusion_phase_names(rve, m)
            if !(rve.amounts[n] isa CrackDensity)
    ]
    isempty(solid_inclusions) || throw(
        ArgumentError(
            "crack_family_compliances(::MoriTanaka) requires the matrix to be the " *
                "only phase carrying volume, but $(solid_inclusions) also do. The " *
                "exact `S_MT = s_s + Σ (4π/3) d_i ℍ_i` collapse relies on that; with " *
                "solid inclusions present the decomposition is different. Use " *
                "SelfConsistent, or compute the contributions yourself and check " *
                "them with crack_family_residual."
        )
    )
    # `C_MT = c_s : (𝕀 + ℍ:c_s)⁻¹` for a matrix + cracks RVE, hence
    # `S_MT = (𝕀 + ℍ:c_s) : s_s = s_s + ℍ`: no correction factor at all, and the
    # contributions are evaluated in the MATRIX, not in `C_hom`.
    C_s = phase_property(rve, m, property)
    families = Dict{Symbol, Any}(
        name => _crack_family_H(rve, name, property, C_s; kw...) for name in names
    )
    return (solid = inv(C_s), families = families)
end

"""
    crack_family_residual(rve, scheme, C_hom; property = :C, kw...) -> Real

Relative Frobenius residual of the identity guaranteed by
[`crack_family_compliances`](@ref):

```math
\\frac{\\bigl\\| \\mathbb{S}^{\\rm hom} - \\mathbb{S}^{\\rm solid}
- \\sum_i (4\\pi/3) d_i \\mathbb{S}_i \\bigr\\|}{\\bigl\\|\\mathbb{S}^{\\rm hom}\\bigr\\|}
```

A model self-check, and the regression test of the decomposition. For an
iterative scheme the residual is bounded by the solver tolerance, not by machine
epsilon: `SelfConsistent` returns `step(x)` on convergence, so with the default
`reltol = 1e-8` expect a residual of that order. Tighten `abstol`/`reltol` in
the scheme to tighten the residual, and do not enable `select_best` (which
returns an earlier iterate and so breaks the identity by that iterate's
residual).
"""
function crack_family_residual(
        rve::RVE, scheme::HomogenizationScheme,
        C_hom::TensND.AbstractTens{4, 3}; property::Symbol = :C, kw...
    )
    dec = crack_family_compliances(rve, scheme, C_hom; property = property, kw...)
    S_hom = inv(C_hom)
    acc = dec.solid
    for (name, S_i) in dec.families
        geom = rve.phases[name].geometry
        acc = acc + delta_compliance(geom, S_i, crack_density(rve, name))
    end
    return sqrt(_frob_sq(S_hom - acc) / _frob_sq(S_hom))
end

# ── Internals ────────────────────────────────────────────────────────────────

# Crack phases of an RVE, rejecting the orientation-averaged ones for which a
# per-family law has no meaning (see the docstring above).
function _crack_family_names(rve::RVE)
    names = Symbol[]
    for name in rve.phase_names
        rve.amounts[name] isa CrackDensity || continue
        sym = phase_symmetrize(rve, name)
        sym isa NoSymmetrize || throw(
            ArgumentError(
                "crack phase :$(name) declares symmetrize = $(sym), so its " *
                    "compliance contribution has already been averaged over an " *
                    "orientation orbit and no single crack normal remains. The " *
                    "per-family compliance is only defined for an unsymmetrized " *
                    "family; declare the orientations as separate phases instead."
            )
        )
        push!(names, name)
    end
    return names
end

# Raw, *unscaled* compliance contribution ℍ_i of one crack family in the
# reference medium `P₀` — no density factor, no symmetrize (there is none, by
# the guard above), but the interface stiffness of a partially closed crack is
# honored, since that is a property of the family and not of the scheme.
function _crack_family_H(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    K_int = _crack_interface_K4(rve, name)
    return compliance_contribution(geom, P₀; K_interface = K_int, kw...)
end
