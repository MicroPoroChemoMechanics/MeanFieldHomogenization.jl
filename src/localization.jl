# =============================================================================
#  localization.jl — generic localization tensors of the dilute Eshelby
#  problem (Kachanov-Sevostianov / Barthélémy et al. 2021).
#
#  Four tensors relate the fields within an inclusion of stiffness `C₁`
#  (or conductivity `K₁`) embedded in an infinite matrix `C₀` (`K₀`) to
#  the remote far-field `ε∞` / `σ∞`:
#
#      strain_strain_loc : ε_inc = A_εε : ε∞
#      stress_strain_loc : σ_inc = A_σε : ε∞
#      strain_stress_loc : ε_inc = A_εσ : σ∞
#      stress_stress_loc : σ_inc = A_σσ : σ∞
#
#  The pivot formula (dilute) is
#
#      A_εε = [𝕀 + ℙ(incl, C₀) : (C₁ - C₀)]⁻¹,
#
#  where ℙ = `hill_tensor(incl, C₀)`.  The three remaining tensors are
#  derived algebraically:
#
#      A_σε = C₁ : A_εε
#      A_εσ = A_εε : S₀    (S₀ = C₀⁻¹)
#      A_σσ = C₁ : A_εε : S₀ = A_σε : S₀
#
#  Conductivity analogs (2-tensor fields, 2-tensor Hill/moduli) use
#  the same formulas with `·` in place of `:`.
#
#  Type-genericity: the implementation works for Float64, BigFloat,
#  ForwardDiff.Dual, SymPy.Sym and Symbolics.Num so long as `hill_tensor`
#  does; it relies only on TensND algebra (`+`, `-`, `⊡`, `inv`).
# =============================================================================

"""
    _identity_4sym(::Type{T}) -> TensISO{4,3}

Symmetric 4-tensor identity `𝕀_{ijkl} = ½(δ_{ik}δ_{jl} + δ_{il}δ_{jk})`
in its most compact `TensISO` form (3D).  `𝕀 ⊡ X = X` for any symmetric
`Tens{4,3}`.
"""
_identity_4sym(::Type{T}) where {T <: Number} = TensISO{3}(one(T), one(T))

"""
    _identity_2(::Type{T}) -> TensISO{2,3}

Identity 2-tensor `δ_{ij}` in 3D (`TensISO{2,3}`).  `𝟙 · x = x` for any
`Tens{2,3}` or 3-vector.
"""
_identity_2(::Type{T}) where {T <: Number} = TensISO{3}(one(T))

# =============================================================================
#  Elastic localization (4-tensor fields)
# =============================================================================

"""
    strain_strain_loc(incl, C₁, C₀; kw...) -> Tens{4,3}

Dilute **strain-strain localization tensor** `A_εε`: connects the
average strain in an `AbstractInclusion` of stiffness `C₁` to the remote
strain `ε∞`:

```
ε_inc = A_εε : ε∞,
A_εε  = [𝕀 + ℙ(incl, C₀) : (C₁ - C₀)]⁻¹.
```

Keyword arguments are forwarded to [`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor).

See also [`stress_strain_loc`](@ref), [`strain_stress_loc`](@ref),
[`stress_stress_loc`](@ref).
"""
function strain_strain_loc(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    T = promote_type(eltype(C₁), eltype(C₀))
    P = hill_tensor(incl, C₀; kw...)
    δC = C₁ - C₀
    return inv(_identity_4sym(T) + (P ⊡ δC))
end

"""
    stress_strain_loc(incl, C₁, C₀; kw...) -> Tens{4,3}

Dilute **stress-strain localization tensor** `A_σε`: `σ_inc = A_σε : ε∞`.

!!! warning "The generic method assumes a homogeneous inclusion"
    It evaluates `A_σε = C₁ : A_εε`, which holds only when the inclusion
    carries a **single uniform stiffness**. An internally heterogeneous
    inclusion — one whose [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) is `false` — has
    no such `C₁`: its average stress has to be assembled from the local
    fields, and the type **must** provide its own method (as `LayeredSphere`
    does). [`stress_stress_loc`](@ref) is expressed in terms of this function,
    so overriding it here fixes both.
"""
function stress_strain_loc(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    return C₁ ⊡ strain_strain_loc(incl, C₁, C₀; kw...)
end

"""
    strain_stress_loc(incl, C₁, C₀; kw...) -> Tens{4,3}

Dilute **strain-stress localization tensor** `A_εσ = A_εε : S₀`:
`ε_inc = A_εσ : σ∞`.  `S₀ = C₀⁻¹` is built internally.
"""
function strain_stress_loc(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    return strain_strain_loc(incl, C₁, C₀; kw...) ⊡ inv(C₀)
end

"""
    stress_stress_loc(incl, C₁, C₀; kw...) -> Tens{4,3}

Dilute **stress-stress localization tensor** `A_σσ = A_σε : S₀`:
`σ_inc = A_σσ : σ∞`.

Derived from [`stress_strain_loc`](@ref) rather than from `A_εε` directly, so
that a heterogeneous inclusion which supplies its own stress-side
localization gets a correct `A_σσ` for free. For a homogeneous inclusion the
two routes are the same expression (`⊡` is left-associative), hence bitwise
identical.
"""
function stress_stress_loc(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    return stress_strain_loc(incl, C₁, C₀; kw...) ⊡ inv(C₀)
end

# =============================================================================
#  Conductivity localization (2-tensor fields)
#
#  SIGN CONVENTION.  Hooke carries no minus sign, Fourier and Fick do.  The
#  package removes that asymmetry by taking as the stress analog MINUS the
#  flux,
#
#       σ ≡ -q = K·∇T ,
#
#  which is not bookkeeping: σ·n is then, in both theories, what the exterior
#  transmits to the interior across a surface of normal n — the traction in
#  mechanics, the entering flux in conduction.  Every transport formula is then
#  the elastic one symbol for symbol (∇T ↔ ε, K ↔ ℂ, ℙ ↔ 𝐏, …).
#
#  Consequence for the four functions below: the "flux" they carry is σ = -q.
#  The names are kept aligned with their elastic twins — renaming them would be
#  breaking and would not make the meaning any clearer than saying it here —
#  and each docstring states the quantity explicitly.  See the notation page,
#  section "Elasticity and transport: one set of formulas".
# =============================================================================

"""
    gradient_gradient_loc(incl, K₁, K₀; kw...) -> Tens{2,3}

Dilute **gradient-gradient localization tensor** `A_∇∇` for the 2nd
order transport problem:

```
∇T_inc = A_∇∇ · ∇T∞,
A_∇∇   = [𝟙 + ℙ(incl, K₀) · (K₁ - K₀)]⁻¹.
```

Conductivity analog of [`strain_strain_loc`](@ref).  Keyword arguments
are forwarded to [`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor).
"""
function gradient_gradient_loc(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    T = promote_type(eltype(K₁), eltype(K₀))
    P = hill_tensor(incl, K₀; kw...)
    δK = K₁ - K₀
    return inv(_identity_2(T) + (P ⋅ δK))
end

"""
    flux_gradient_loc(incl, K₁, K₀; kw...) -> Tens{2,3}

Dilute **flux-gradient localization tensor** `A_q∇`: `σ_inc = A_q∇ · ∇T∞`,
with `σ ≡ -q = K·∇T` the stress analog of the package (see the sign note
above this block, and the notation page). It is `A_q∇ = K₁ · A_∇∇`, so the
quantity it produces is `K₁ · ∇T_inc`, i.e. **minus** the flux — exactly what
makes this the transport twin of [`stress_strain_loc`](@ref), symbol for
symbol.

!!! warning "The generic method assumes a homogeneous inclusion"
    The expression `A_q∇ = K₁ · A_∇∇` is valid only for a single uniform
    conductivity. A heterogeneous inclusion must supply its own method;
    [`flux_flux_loc`](@ref) then follows.
"""
function flux_gradient_loc(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    return K₁ ⋅ gradient_gradient_loc(incl, K₁, K₀; kw...)
end

"""
    gradient_flux_loc(incl, K₁, K₀; kw...) -> Tens{2,3}

Dilute **gradient-flux localization tensor** `A_∇q = A_∇∇ · R₀`
(with `R₀ = K₀⁻¹`): `∇T_inc = A_∇q · σ∞`, with `σ ≡ -q = K₀·∇T∞` the stress
analog of the package. Transport twin of [`strain_stress_loc`](@ref).
"""
function gradient_flux_loc(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    return gradient_gradient_loc(incl, K₁, K₀; kw...) ⋅ inv(K₀)
end

"""
    flux_flux_loc(incl, K₁, K₀; kw...) -> Tens{2,3}

Dilute **flux-flux localization tensor** `A_qq = A_q∇ · R₀`:
`σ_inc = A_qq · σ∞`, with `σ ≡ -q` throughout — both sides carry the same
convention, so the tensor itself is the one a reader of
[`stress_stress_loc`](@ref) expects.

Derived from [`flux_gradient_loc`](@ref), for the reason given in
[`stress_stress_loc`](@ref).
"""
function flux_flux_loc(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    return flux_gradient_loc(incl, K₁, K₀; kw...) ⋅ inv(K₀)
end

# =============================================================================
#  Bundled localization + contribution
#
#  Mori-Tanaka and the self-consistent kernels need, per phase, BOTH the
#  dilute concentration tensor `A` and the contribution tensor `N` (resp. the
#  stress average `B`).  Computed separately, each goes through
#  `strain_strain_loc` — hence through `hill_tensor`, the dominant cost of the
#  whole package — with byte-identical arguments.
#
#  The dispatch axis is the INCLUSION CLASS rather than a runtime
#  `is_homogeneous_inclusion` flag: the fallback on `AbstractInclusion` is
#  literally the pair of separate calls, so a future heterogeneous inclusion
#  that forgets to specialize can only be slow, never wrong.  The fast path
#  is restricted to `AbstractEllipsoidalInclusion`, which is homogeneous by
#  construction.
#
#  Every fast path repeats the SAME expressions in the SAME order as the two
#  functions it replaces (`contribution.jl` `stiffness_contribution`,
#  `stress_strain_loc` above), so the results are bitwise identical — the
#  benchmark harness asserts exactly that.
# =============================================================================

# ── Safe fallback: any inclusion, including internally heterogeneous ones ────

loc_and_stiffness(
    incl::AbstractInclusion,
    C₁::TensND.AbstractTens{4, 3},
    C₀::TensND.AbstractTens{4, 3};
    kw...
) = (
    strain_strain_loc(incl, C₁, C₀; kw...),
    stiffness_contribution(incl, C₁, C₀; kw...),
)

loc_and_stiffness(
    incl::AbstractInclusion,
    K₁::TensND.AbstractTens{2, 3},
    K₀::TensND.AbstractTens{2, 3};
    kw...
) = (
    gradient_gradient_loc(incl, K₁, K₀; kw...),
    conductivity_contribution(incl, K₁, K₀; kw...),
)

loc_and_stress_average(
    incl::AbstractInclusion,
    C₁::TensND.AbstractTens{4, 3},
    C₀::TensND.AbstractTens{4, 3};
    kw...
) = (
    strain_strain_loc(incl, C₁, C₀; kw...),
    stress_strain_loc(incl, C₁, C₀; kw...),
)

loc_and_stress_average(
    incl::AbstractInclusion,
    K₁::TensND.AbstractTens{2, 3},
    K₀::TensND.AbstractTens{2, 3};
    kw...
) = (
    gradient_gradient_loc(incl, K₁, K₀; kw...),
    flux_gradient_loc(incl, K₁, K₀; kw...),
)

# ── Homogeneous ellipsoidal fast path: ONE hill_tensor instead of two ────────

function loc_and_stiffness(
        incl::AbstractEllipsoidalInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    A = strain_strain_loc(incl, C₁, C₀; kw...)
    return (A, (C₁ - C₀) ⊡ A)          # verbatim `contribution.jl` :39-40
end

function loc_and_stiffness(
        incl::AbstractEllipsoidalInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    A = gradient_gradient_loc(incl, K₁, K₀; kw...)
    return (A, (K₁ - K₀) ⋅ A)
end

function loc_and_stress_average(
        incl::AbstractEllipsoidalInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    A = strain_strain_loc(incl, C₁, C₀; kw...)
    return (A, C₁ ⊡ A)                 # verbatim `stress_strain_loc`
end

function loc_and_stress_average(
        incl::AbstractEllipsoidalInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    A = gradient_gradient_loc(incl, K₁, K₀; kw...)
    return (A, K₁ ⋅ A)
end
