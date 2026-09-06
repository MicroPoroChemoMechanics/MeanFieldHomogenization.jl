# =============================================================================
#  localfields.jl — POINTWISE reconstruction of the elastic and transport
#  fields inside and around an isotropic `N`-layer `LayeredSphere`, from
#  the per-region mode amplitudes of the state-vector recurrences
#  (`bulk_recurrence.jl`, `shear_recurrence.jl`, `conductivity.jl`).
#
#  Where the rest of the module stops at the volume-averaged concentration
#  tensors (`layer_strain_average`, `strain_strain_loc(…; layer)`), this
#  file gives the field AT A POINT — in any layer and in the surrounding
#  matrix — for an arbitrary remote loading.
#
#  Structure of the answer
#  -----------------------
#  Write `n = x/‖x‖`, `p = n⊗n`, `q = 𝟏 − p`.  The configuration is
#  invariant under every rotation about the center, so the pointwise
#  localization tensor is TRANSVERSELY ISOTROPIC about `n`, with
#  coefficients depending on `r = ‖x‖` alone, and it carries NO major
#  symmetry.  That is exactly `TensND.TensTI{4,T,6}` (Walpole basis
#  `W₁…W₆`), so `𝔸(x)` is six scalars plus an axis, never an 81-component
#  array, and every product below stays inside the closed-form Walpole
#  algebra.
#
#  Elasticity.  Splitting `ε∞ = (tr ε∞/3) 𝟏 + ε∞ᵈ`:
#
#   * spherical part (Y₀) — `u = f(r) n`, `f = Ã r + B̃/r²`, giving
#     `ε = f′ p + (f/r) q` and
#         𝔸_bulk = (1/3)[(Ã − 2B̃/r³) p⊗𝟏 + (Ã + B̃/r³) q⊗𝟏] ;
#
#   * deviatoric part (Y₂) — the general, NON-axisymmetric ℓ=2 solution is
#         u = g(r) (ε∞ᵈ·n) + h(r) (n·ε∞ᵈ·n) n ,
#     which reduces to the module's `u_r = U P₂(cosθ)`, `u_θ = W dP₂/dθ`
#     convention through `U = 2(g+h)`, `W = g`.  Differentiating with
#     `∂n_i/∂x_j = (δ_ij − n_i n_j)/r` gives, with `s = n·ε∞ᵈ·n`,
#         ε = (g/r) ε∞ᵈ + G₂ (ε∞ᵈ·n ⊗ n)ˢ + G₃ s p + G₄ s 𝟏 ,
#         G₂ = g′ − g/r + 2h/r,  G₃ = h′ − 3h/r,  G₄ = h/r .
#
#  In the Walpole basis (`𝕀 = W₁+W₂+W₅+W₆`, `(ε·n⊗n)ˢ = (W₁+W₆/2):ε`,
#  `p⊗𝟏 = W₁+√2W₃`, `q⊗𝟏 = 2W₂+√2W₄`):
#
#      ℓ(𝔸_dev)  = (g/r + G₂+G₃+G₄, g/r, 0, √2 G₄, g/r, g/r + G₂/2)
#      ℓ(𝔸_bulk) = (P, 2Q, √2 P, √2 Q, 0, 0),  P = (Ã−2B̃/r³)/3,
#                                              Q = (Ã+B̃/r³)/3
#      𝔸(x) = 𝔸_bulk + 𝔸_dev ⊡ 𝕂 .
#
#  Consistency with the averaged path.  Averaging `𝔸(x)` over directions
#  gives `Ã_k 𝕁 + β_local(r) 𝕂` with
#  `β_local = (2/3)(g/r) + g′/3 + 2h′/15 + (4/15)(h/r)`; modes 3 and 4
#  contribute zero POINTWISE, mode 1 contributes `a`, and mode 2
#  contributes `7 b r² (3κ+μ)/μ`, whose shell average is exactly
#  `_layer_avg_dev_shear_factor`.  So the pointwise and averaged routes
#  reproduce `α_k` and `β_k` identically — `shell_localization` below
#  exposes that identity, and the test suite pins it.
#
#  Transport (Y₁).  `T = f(r)(n·∇T∞)`, `f = Ã r + B̃/r²`, hence
#  `∇T = [f′ p + (f/r) q] · ∇T∞`, a 2nd-order TI tensor `TensTI{2}`.
#
#  Conventions
#  -----------
#  * Loading is normalized to a unit remote field: the recurrences already
#    solve for `A_∞ = 1` (bulk / transport) and `(a, b)_matrix = (1, 0)`
#    (shear), so no rescaling happens here.
#  * Results are expressed in the CANONICAL basis; the returned `TensTI`
#    carries its own axis `n`, so `get_array` gives canonical components.
#  * Region indexing follows `get_layer`: `1..N` for the layers, `N+1` for
#    the matrix.  Exactly ON an interface the field is genuinely
#    discontinuous (imperfect interfaces make even the displacement jump),
#    so every entry point takes a `side` keyword to say which limit is
#    meant.
# =============================================================================

# ── Region lookup ────────────────────────────────────────────────────────────

"""
    get_layer(sphere::LayeredSphere, r; side = :outer) -> Int

Index of the region containing the radius `r`: `1, …, N` for the layers,
`N+1` for the surrounding matrix.

Layer `k` occupies `r_{k-1} ≤ r < r_k`, so a radius strictly between two
interfaces is unambiguous.  **On** an interface `r = r_k` the field has two
different limits and `side` picks one:

- `side = :outer` (default) — the limit from above, `r_k⁺`: region `k+1`,
  consistent with the half-open convention `[r_{k-1}, r_k)` used by
  [`layer_volume_fraction`](@ref) and by `LayeredSpheroids.get_layer`;
- `side = :inner` — the limit from below, `r_k⁻`: region `k`.

The distinction is not cosmetic. Across a [`SpringInterface`](@ref) the
displacement jumps and across a [`MembraneInterface`](@ref) the traction
jumps, so `local_stress(…, r_k; side = :inner)` and `side = :outer` return
genuinely different tensors — that difference *is* the interface law, and
the test suite checks it against the prescribed jump.

Radii beyond `r_N` always give `N+1`, whatever `side`.

A **symbolic** radius is refused rather than located.  `SymPy.Sym` answers a
comparison it cannot decide with a plain `false` — `r ≥ 1.0` on a positive
symbol returns `Bool(false)`, not an error — so the loop below would silently
return layer 1 for any symbol.  Name the region explicitly instead: every
pointwise entry point takes a `layer` keyword.
"""
function get_layer(sphere::LayeredSphere{T, N}, r; side::Symbol = :outer) where {T, N}
    side === :outer || side === :inner ||
        throw(ArgumentError("get_layer: `side` must be :outer or :inner, got $(side)"))
    is_hard_numeric(typeof(r)) || throw(
        ArgumentError(
            "get_layer: a symbolic radius cannot be located by comparison " *
                "(got $(typeof(r))); name the region with `layer = k` instead " *
                "(1..$(N) for a layer, $(N + 1) for the matrix)"
        )
    )
    k = 1
    if side === :outer
        while k ≤ N && r ≥ sphere.radii[k]
            k += 1
        end
    else
        while k ≤ N && r > sphere.radii[k]
            k += 1
        end
    end
    return k
end

"""
    _resolve_layer(sphere, r; side, layer) -> Int

Region index to evaluate the field in.  `layer` short-circuits the lookup;
otherwise [`get_layer`](@ref) decides from `r`.

A **symbolic** radius cannot be located by comparison — `Symbolics.Num` and
`SymPy.Sym` subtype `Real` without being ordered — so the lookup is refused
with an actionable message rather than a `MethodError` deep in a `while`.
Symbolic work is exactly the case where the region is known up front, so
passing `layer = k` costs nothing and keeps the whole reconstruction
symbolic.
"""
@inline function _resolve_layer(
        sphere::LayeredSphere{T, N}, r; side::Symbol, layer
    ) where {T, N}
    if layer !== nothing
        1 ≤ layer ≤ N + 1 ||
            throw(ArgumentError("layer must be in 1..$(N + 1) (N+1 = matrix), got $(layer)"))
        return Int(layer)
    end
    is_hard_numeric(typeof(r)) || throw(
        ArgumentError(
            "cannot locate a symbolic radius by comparison; pass `layer = k` " *
                "explicitly (1..$(N) for a layer, $(N + 1) for the matrix)"
        )
    )
    return get_layer(sphere, r; side)
end

"""
    _at_origin(r) -> Bool

Whether `r` is the exact origin, where the `1/r³` and `1/r⁵` mode terms are
switched off (their amplitudes vanish identically in the core).  Gated on
[`is_hard_numeric`](@ref) so a symbolic radius always takes the general
branch and keeps its closed form.
"""
@inline _at_origin(r) = is_hard_numeric(typeof(r)) && iszero(r)

# ── Point argument: Cartesian vector or spherical coordinates ────────────────

"""
    _radial_frame(x) -> (r, n)
    _radial_frame(r, θ, φ) -> (r, n)

Radius and unit radial direction of a point.  The Cartesian form accepts
anything `TensND._extract_vec` understands (`Vec{3}`, `NTuple{3}`, an
`AbstractVector`, a 1st-order `AbstractTens`).  The spherical form uses the
package-wide convention `θ` = colatitude measured from `e₃`, `φ` = azimuth
in the `(e₁, e₂)` plane.

At `r = 0` the direction is undefined; `(0, 0, 1)` is returned, which is
harmless because every Walpole coefficient of the field is axis-independent
there (see [`local_strain_strain_loc`](@ref)).
"""
function _radial_frame(x)
    v = TensND._extract_vec(x)
    r = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    T = typeof(r)
    _at_origin(r) && return r, (zero(T), zero(T), one(T))
    return r, (v[1] / r, v[2] / r, v[3] / r)
end

function _radial_frame(r, θ, φ)
    sθ, cθ = sin(θ), cos(θ)
    return r, (sθ * cos(φ), sθ * sin(φ), cθ)
end

# ── Cached solution: elasticity ──────────────────────────────────────────────

"""
    LayeredSphereFields(sphere, C₀)

Precomputed pointwise elastic solution of a [`LayeredSphere`](@ref)
embedded in the isotropic matrix `C₀`: the per-region amplitudes of the
spherical mode pair `(Ã, B̃)` and of the four deviatoric Love modes
`(a, b, c, d)`, for `k = 1..N` the layers and `k = N+1` the matrix, under a
unit remote strain.

Build it once and pass it wherever `(sphere, C₀)` is accepted. Every
`local_*` entry point also takes `(sphere, C₀)` directly, but that form
re-runs the whole recurrence on every call — fine for a handful of points,
wasteful for the thousands a field map needs.

```
sol = LayeredSphereFields(sphere, C₀)
A   = local_strain_strain_loc(sol, x)
```

See also [`LayeredSphereTransportFields`](@ref).
"""
struct LayeredSphereFields{T, N, Np1, S, C}
    sphere::S
    C₀::C
    κμ::NTuple{N, Tuple{T, T}}
    κμ₀::Tuple{T, T}
    AB::NTuple{Np1, Tuple{T, T}}
    abcd::NTuple{Np1, NTuple{4, T}}
end

function LayeredSphereFields(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}
    ) where {T, N}
    κμ = _bulk_layer_moduli(sphere)
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    TP = _bulk_promote(sphere, κμ, κ₀, μ₀)
    radii = sphere.radii

    inside, s_matrix = _bulk_state_seq(sphere, κ₀, μ₀)
    A_inf, B_inf = _bulk_extract_AB(
        TP(radii[N]), TP(κ₀), TP(μ₀), s_matrix[1], s_matrix[2]
    )
    inv_A_inf = one(TP) / A_inf

    # `B₁ = 0` exactly: the core expansion `u = A r + B/r²` must stay finite
    # at the origin.  The seed enforces it, but recovering it through
    # `_bulk_extract_AB` leaves an `O(eps)` residue that `B̃/r³` amplifies
    # without bound as `r → 0`.  Write the zero down instead.
    AB = ntuple(Val(N + 1)) do k
        if k > N
            (one(TP), B_inf * inv_A_inf)
        else
            (κk, μk) = κμ[k]
            Ak, Bk = _bulk_extract_AB(
                TP(radii[k]), TP(κk), TP(μk), inside[k][1], inside[k][2]
            )
            k == 1 ? (Ak * inv_A_inf, zero(TP)) : (Ak * inv_A_inf, Bk * inv_A_inf)
        end
    end

    abcd = _shear_amplitude_seq(sphere, C₀)

    κμ_p = ntuple(k -> (TP(κμ[k][1]), TP(κμ[k][2])), Val(N))
    return LayeredSphereFields{TP, N, N + 1, typeof(sphere), typeof(C₀)}(
        sphere, C₀, κμ_p, (TP(κ₀), TP(μ₀)), AB, abcd
    )
end

layer_count(f::LayeredSphereFields{T, N}) where {T, N} = N
get_layer(f::LayeredSphereFields, r; side::Symbol = :outer) =
    get_layer(f.sphere, r; side)

"""
    _region_moduli(f, k) -> (κ, μ)

Bulk and shear moduli of region `k` (the matrix for `k = N+1`).
"""
@inline _region_moduli(f::LayeredSphereFields{T, N}, k::Int) where {T, N} =
    k ≤ N ? f.κμ[k] : f.κμ₀

"""
    region_stiffness(f::LayeredSphereFields, k) -> TensISO{4,3}

Stiffness tensor of region `k`, the matrix `C₀` for `k = N+1`.
"""
function region_stiffness(f::LayeredSphereFields{T, N}, k::Int) where {T, N}
    κ, μ = _region_moduli(f, k)
    return TensISO{3}(3 * κ, 2 * μ)
end

# ── Radial scalars ───────────────────────────────────────────────────────────

"""
    _radial_scalars(f, k, r) -> (f_over_r, fp, g_over_r, gp, h_over_r, hp)

The six radial functions the pointwise field is built from, in region `k`
at radius `r`:

- spherical: `f/r = Ã + B̃/r³` and `f′ = Ã − 2B̃/r³`;
- deviatoric: `g/r`, `g′`, `h/r`, `h′` with
  `g = a r + b(15x+11) r³ − c/r⁴ + d/r²` and
  `h = −b(6x+17) r³ + (5/2) c/r⁴ + ((3x+1)/2) d/r²`, `x = κ/μ`.

The four deviatoric scalars are evaluated as MODE SUMS, never as a division
of `g(r)` by `r`. In the core `c = d = 0` exactly, so each one reduces to a
polynomial in `r²` — finite and accurate all the way down to `r = 0`, where
forming `g(r)/r` would divide two vanishing quantities.
"""
@inline function _radial_scalars(f::LayeredSphereFields{T}, k::Int, r) where {T}
    κ, μ = _region_moduli(f, k)
    Ã, B̃ = f.AB[k]
    a, b, c, d = f.abcd[k]
    TR = promote_type(T, typeof(r))
    rr = TR(r)

    x = TR(κ) / TR(μ)
    γ = 15 * x + 11
    δ = 6 * x + 17
    η = (3 * x + 1) / 2

    r² = rr * rr
    ir³ = _at_origin(rr) ? zero(TR) : one(TR) / (r² * rr)
    ir⁵ = _at_origin(rr) ? zero(TR) : ir³ / r²

    f_over_r = TR(Ã) + TR(B̃) * ir³
    fp = TR(Ã) - 2 * TR(B̃) * ir³

    bγ = TR(b) * γ
    bδ = TR(b) * δ
    cc = TR(c)
    dd = TR(d)

    g_over_r = TR(a) + bγ * r² - cc * ir⁵ + dd * ir³
    gp = TR(a) + 3 * bγ * r² + 4 * cc * ir⁵ - 2 * dd * ir³
    h_over_r = -bδ * r² + (5 // 2) * cc * ir⁵ + η * dd * ir³
    hp = -3 * bδ * r² - 10 * cc * ir⁵ - 2 * η * dd * ir³

    return f_over_r, fp, g_over_r, gp, h_over_r, hp
end

# ── The pointwise strain localization tensor ─────────────────────────────────

"""
    local_strain_strain_loc(f_or_sphere, [C₀,] x; side = :outer) -> TensTI{4,3}

Pointwise strain-strain localization tensor `𝔸(x)`, so that the local
strain under a remote uniform strain `ε∞` is `ε(x) = 𝔸(x) ⊡ ε∞`. Valid at
every point: inside any layer **and** in the surrounding matrix, with
perfect or imperfect interfaces.

The point `x` is either a Cartesian vector (`Vec{3}`, `NTuple{3}`,
`AbstractVector`, 1st-order `AbstractTens`) or spherical coordinates given
as three arguments `(r, θ, φ)`, `θ` the colatitude from `e₃`.

The result is transversely isotropic about `n = x/‖x‖` — the configuration
is rotation-invariant about the center — and has **no major symmetry**, so
it comes back as a general `TensTI{4,T,6}`, six Walpole scalars plus the
axis, in the canonical basis.

At `x = 0` every Walpole coefficient degenerates to the isotropic pair
`(Ã₁, a₁)`, so the returned tensor is the same for any axis; `(0,0,1)` is
used. Note that `a₁` is the **pointwise** core value and differs in general
from the layer average `β₁`, which folds in the mode-2 term.

`side` disambiguates a point lying exactly on an interface — see
[`get_layer`](@ref).

See also [`local_stress_strain_loc`](@ref), [`local_strain_stress_loc`](@ref),
[`local_stress_stress_loc`](@ref), [`local_strain`](@ref),
[`LayeredSphereFields`](@ref).
"""
function local_strain_strain_loc end

function local_strain_strain_loc(
        f::LayeredSphereFields, x; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    return _local_A(f, r, n, _resolve_layer(f.sphere, r; side, layer))
end

function local_strain_strain_loc(
        f::LayeredSphereFields, r, θ, φ; side::Symbol = :outer, layer = nothing
    )
    rr, n = _radial_frame(r, θ, φ)
    return _local_A(f, rr, n, _resolve_layer(f.sphere, rr; side, layer))
end

function _local_A(f::LayeredSphereFields{T}, r, n, k::Int) where {T}
    f_over_r, fp, g_over_r, gp, h_over_r, hp = _radial_scalars(f, k, r)

    G₂ = gp - g_over_r + 2 * h_over_r
    G₃ = hp - 3 * h_over_r
    G₄ = h_over_r

    TR = typeof(g_over_r)
    sq2 = sqrt(TR(2))
    z = zero(TR)

    P = fp / 3
    Q = f_over_r / 3
    A_bulk = TensND.TensTI{4}(P, 2 * Q, sq2 * P, sq2 * Q, z, z, n)

    A_dev = TensND.TensTI{4}(
        g_over_r + G₂ + G₃ + G₄,
        g_over_r,
        z,
        sq2 * G₄,
        g_over_r,
        g_over_r + G₂ / 2,
        n,
    )

    𝕂 = TensISO{3}(z, one(TR))
    return A_bulk + A_dev ⊡ 𝕂
end

"""
    local_stress_strain_loc(f_or_sphere, [C₀,] x; side = :outer) -> TensTI{4,3}

Pointwise stress-strain localization `ℂ(x) ⊡ 𝔸(x)`: the local stress under
a remote uniform **strain** `ε∞` is `σ(x) = local_stress_strain_loc(…) ⊡ ε∞`.
`ℂ(x)` is the stiffness of the region containing `x` (the matrix outside).

Twin of [`local_strain_strain_loc`](@ref); same arguments and conventions.
"""
function local_stress_strain_loc end

"""
    local_strain_stress_loc(f_or_sphere, [C₀,] x; side = :outer) -> TensTI{4,3}

Pointwise strain-stress localization `𝔸(x) ⊡ 𝕊₀`, `𝕊₀ = C₀⁻¹`: the local
strain under a remote uniform **stress** `σ∞` is
`ε(x) = local_strain_stress_loc(…) ⊡ σ∞`.

Exact because the remote medium is uniform: `ε∞ = 𝕊₀ ⊡ σ∞`.
Twin of [`local_strain_strain_loc`](@ref); same arguments and conventions.
"""
function local_strain_stress_loc end

"""
    local_stress_stress_loc(f_or_sphere, [C₀,] x; side = :outer) -> TensTI{4,3}

Pointwise stress-stress localization `ℂ(x) ⊡ 𝔸(x) ⊡ 𝕊₀`: the local stress
under a remote uniform **stress** `σ∞` is
`σ(x) = local_stress_stress_loc(…) ⊡ σ∞`.

Twin of [`local_strain_strain_loc`](@ref); same arguments and conventions.
"""
function local_stress_stress_loc end

for (name, pre, post) in (
        (:local_stress_strain_loc, true, false),
        (:local_strain_stress_loc, false, true),
        (:local_stress_stress_loc, true, true),
    )
    @eval function $name(
            f::LayeredSphereFields, x; side::Symbol = :outer, layer = nothing
        )
        r, n = _radial_frame(x)
        k = _resolve_layer(f.sphere, r; side, layer)
        return _local_A_scaled(f, r, n, k, Val($pre), Val($post))
    end
    @eval function $name(
            f::LayeredSphereFields, r, θ, φ; side::Symbol = :outer, layer = nothing
        )
        rr, n = _radial_frame(r, θ, φ)
        k = _resolve_layer(f.sphere, rr; side, layer)
        return _local_A_scaled(f, rr, n, k, Val($pre), Val($post))
    end
end

function _local_A_scaled(
        f::LayeredSphereFields, r, n, k::Int, ::Val{pre}, ::Val{post}
    ) where {pre, post}
    A = _local_A(f, r, n, k)
    if pre
        A = region_stiffness(f, k) ⊡ A
    end
    if post
        A = A ⊡ inv(f.C₀)
    end
    return A
end

# ── Field evaluators ─────────────────────────────────────────────────────────

"""
    local_strain(f_or_sphere, [C₀,] x, ε∞; side = :outer) -> Tens{2,3}

Local strain tensor at `x` under the remote uniform strain `ε∞`, i.e.
`local_strain_strain_loc(…) ⊡ ε∞`. `x` is Cartesian, or replace it by three
arguments `(r, θ, φ)`.
"""
function local_strain end

"""
    local_stress(f_or_sphere, [C₀,] x, ε∞; side = :outer) -> Tens{2,3}

Local stress tensor at `x` under the remote uniform strain `ε∞`, i.e.
`ℂ(x) ⊡ local_strain(…)` with `ℂ(x)` the stiffness of the region containing
the point (the matrix `C₀` outside the composite sphere).
"""
function local_stress end

"""
    local_displacement(f_or_sphere, [C₀,] x, ε∞; side = :outer) -> Vec{3}

Local displacement at `x` under the remote uniform strain `ε∞`:

```math
u = \\frac{\\mathrm{tr}\\,ε∞}{3} f(r)\\,n + g(r)\\,(ε∞ᵈ·n) + h(r)\\,(n·ε∞ᵈ·n)\\,n
```

with `f = Ã r + B̃/r²` and `g`, `h` the deviatoric radial functions of the
four Love modes. The rigid-body translation is fixed by `u → ε∞·x` at
infinity, and `u(0) = 0`.

Across a [`SpringInterface`](@ref) the displacement is discontinuous; use
`side` to select the limit (see [`get_layer`](@ref)).
"""
function local_displacement end

function local_strain(
        f::LayeredSphereFields, x, ε∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    return _local_A(f, r, n, _resolve_layer(f.sphere, r; side, layer)) ⊡ ε∞
end

function local_strain(
        f::LayeredSphereFields, r, θ, φ, ε∞; side::Symbol = :outer, layer = nothing
    )
    rr, n = _radial_frame(r, θ, φ)
    return _local_A(f, rr, n, _resolve_layer(f.sphere, rr; side, layer)) ⊡ ε∞
end

function local_stress(
        f::LayeredSphereFields, x, ε∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    k = _resolve_layer(f.sphere, r; side, layer)
    return region_stiffness(f, k) ⊡ (_local_A(f, r, n, k) ⊡ ε∞)
end

function local_stress(
        f::LayeredSphereFields, r, θ, φ, ε∞; side::Symbol = :outer, layer = nothing
    )
    rr, n = _radial_frame(r, θ, φ)
    k = _resolve_layer(f.sphere, rr; side, layer)
    return region_stiffness(f, k) ⊡ (_local_A(f, rr, n, k) ⊡ ε∞)
end

function _local_u(f::LayeredSphereFields, r, n, k::Int, ε∞)
    f_over_r, _, g_over_r, _, h_over_r, _ = _radial_scalars(f, k, r)

    tr_ε = ε∞[1, 1] + ε∞[2, 2] + ε∞[3, 3]
    e_v = tr_ε / 3
    # Deviatoric part of the remote strain, applied to n.
    En = ntuple(i -> sum(ε∞[i, j] * n[j] for j in 1:3) - e_v * n[i], 3)
    s = sum(En[i] * n[i] for i in 1:3)

    fr = f_over_r * r
    gr = g_over_r * r
    hr = h_over_r * r
    return TensND.Vec{3}(
        ntuple(i -> e_v * fr * n[i] + gr * En[i] + hr * s * n[i], 3)
    )
end

function local_displacement(
        f::LayeredSphereFields, x, ε∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    return _local_u(f, r, n, _resolve_layer(f.sphere, r; side, layer), ε∞)
end

function local_displacement(
        f::LayeredSphereFields, r, θ, φ, ε∞; side::Symbol = :outer, layer = nothing
    )
    rr, n = _radial_frame(r, θ, φ)
    return _local_u(f, rr, n, _resolve_layer(f.sphere, rr; side, layer), ε∞)
end

# ── Bridge back to the layer averages ────────────────────────────────────────

"""
    shell_localization(f::LayeredSphereFields, k) -> (α_k, β_k)

Volume-averaged bulk and shear localization of layer `k`, recovered from
the cached pointwise amplitudes: `α_k = Ã_k` (the spherical localization is
uniform inside a layer) and
`β_k = a_k + b_k · _layer_avg_dev_shear_factor(r_{k-1}, r_k, κ_k, μ_k)`.

Identical by construction to `_bulk_localization` and `_shear_localization`,
which is the point: the pointwise reconstruction and the averaged path
share their amplitudes and cannot drift apart.
"""
function shell_localization(f::LayeredSphereFields{T, N}, k::Int) where {T, N}
    1 ≤ k ≤ N || throw(BoundsError(f.sphere, k))
    radii = f.sphere.radii
    κk, μk = f.κμ[k]
    a, b, _, _ = f.abcd[k]
    r_a = k == 1 ? zero(T) : radii[k - 1]
    return f.AB[k][1], a + b * _layer_avg_dev_shear_factor(r_a, radii[k], κk, μk)
end

# ── Cached solution: transport ───────────────────────────────────────────────

"""
    LayeredSphereTransportFields(sphere, K₀)

Precomputed pointwise transport (thermal / electric / Darcy) solution of a
[`LayeredSphere`](@ref) in the isotropic matrix `K₀`: the per-region
amplitudes `(Ã, B̃)` of `T = (Ã r + B̃/r²)(n·∇T∞)`, for `k = 1..N` the layers
and `k = N+1` the matrix, under a unit remote gradient.

Transport twin of [`LayeredSphereFields`](@ref).
"""
struct LayeredSphereTransportFields{T, N, Np1, S, K}
    sphere::S
    K₀::K
    k_layers::NTuple{N, T}
    k₀::T
    AB::NTuple{Np1, Tuple{T, T}}
end

function LayeredSphereTransportFields(
        sphere::LayeredSphere{T, N}, K₀::TensND.TensISO{2, 3}
    ) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    k₀ = _iso_scalar(K₀)
    TP = promote_type(
        T, typeof(k₀), ntuple(k -> typeof(k_layers[k]), N)...,
        interfaces_eltype(sphere.interfaces)
    )
    radii = sphere.radii

    inside, s_matrix = _cond_state_seq(sphere, k₀)
    A_inf, B_inf = _cond_extract_AB(TP(radii[N]), TP(k₀), s_matrix[1], s_matrix[2])
    inv_A_inf = one(TP) / A_inf

    # `B₁ = 0` exactly — same regularity argument as in the elastic case.
    AB = ntuple(Val(N + 1)) do k
        if k > N
            (one(TP), B_inf * inv_A_inf)
        else
            Ak, Bk = _cond_extract_AB(
                TP(radii[k]), TP(k_layers[k]), inside[k][1], inside[k][2]
            )
            k == 1 ? (Ak * inv_A_inf, zero(TP)) : (Ak * inv_A_inf, Bk * inv_A_inf)
        end
    end

    return LayeredSphereTransportFields{TP, N, N + 1, typeof(sphere), typeof(K₀)}(
        sphere, K₀, ntuple(k -> TP(k_layers[k]), Val(N)), TP(k₀), AB
    )
end

layer_count(f::LayeredSphereTransportFields{T, N}) where {T, N} = N
get_layer(f::LayeredSphereTransportFields, r; side::Symbol = :outer) =
    get_layer(f.sphere, r; side)

@inline _region_conductivity(f::LayeredSphereTransportFields{T, N}, k::Int) where {T, N} =
    k ≤ N ? f.k_layers[k] : f.k₀

@inline function _cond_radial_scalars(f::LayeredSphereTransportFields{T}, k::Int, r) where {T}
    Ã, B̃ = f.AB[k]
    TR = promote_type(T, typeof(r))
    rr = TR(r)
    ir³ = _at_origin(rr) ? zero(TR) : one(TR) / (rr * rr * rr)
    return TR(Ã) + TR(B̃) * ir³, TR(Ã) - 2 * TR(B̃) * ir³   # (f/r, f′)
end

"""
    local_gradient_gradient_loc(f_or_sphere, [K₀,] x; side = :outer) -> TensTI{2,3}

Pointwise gradient-gradient localization `𝐀(x)`, so that the local
temperature gradient under a remote uniform gradient `∇T∞` is
`∇T(x) = 𝐀(x) · ∇T∞`. Valid in any layer and in the matrix, with perfect,
Kapitza or surface-conductive interfaces.

`𝐀(x) = f′(r) n⊗n + (f(r)/r)(𝟏 − n⊗n)` is transversely isotropic about
`n = x/‖x‖` and comes back as a `TensTI{2,T,2}`. Transport twin of
[`local_strain_strain_loc`](@ref); same argument forms and same `side`
convention.
"""
function local_gradient_gradient_loc end

"""
    local_flux_gradient_loc(f_or_sphere, [K₀,] x; side = :outer) -> TensTI{2,3}

Pointwise flux-gradient localization `−k(x) 𝐀(x)`: the local flux under a
remote uniform gradient `∇T∞` is `q(x) = local_flux_gradient_loc(…) · ∇T∞`,
with the Fourier / Fick sign convention `q = −k ∇T`.
"""
function local_flux_gradient_loc end

"""
    local_gradient_flux_loc(f_or_sphere, [K₀,] x; side = :outer) -> TensTI{2,3}

Pointwise gradient-flux localization `𝐀(x) · K₀⁻¹`: the local gradient
under a remote uniform **flux** `q∞` is `∇T(x) = local_gradient_flux_loc(…) · q∞`.
"""
function local_gradient_flux_loc end

"""
    local_flux_flux_loc(f_or_sphere, [K₀,] x; side = :outer) -> TensTI{2,3}

Pointwise flux-flux localization `−k(x) 𝐀(x) · K₀⁻¹`: the local flux under
a remote uniform **flux** `q∞` is `q(x) = local_flux_flux_loc(…) · q∞`.
"""
function local_flux_flux_loc end

function _local_K(f::LayeredSphereTransportFields, r, n, k::Int)
    f_over_r, fp = _cond_radial_scalars(f, k, r)
    return TensND.TensTI{2}(f_over_r, fp, n)     # (transverse, axial)
end

for (name, pre, post) in (
        (:local_gradient_gradient_loc, false, false),
        (:local_flux_gradient_loc, true, false),
        (:local_gradient_flux_loc, false, true),
        (:local_flux_flux_loc, true, true),
    )
    @eval function $name(
            f::LayeredSphereTransportFields, x; side::Symbol = :outer, layer = nothing
        )
        r, n = _radial_frame(x)
        k = _resolve_layer(f.sphere, r; side, layer)
        return _local_K_scaled(f, r, n, k, Val($pre), Val($post))
    end
    @eval function $name(
            f::LayeredSphereTransportFields, r, θ, φ; side::Symbol = :outer, layer = nothing
        )
        rr, n = _radial_frame(r, θ, φ)
        k = _resolve_layer(f.sphere, rr; side, layer)
        return _local_K_scaled(f, rr, n, k, Val($pre), Val($post))
    end
end

function _local_K_scaled(
        f::LayeredSphereTransportFields, r, n, k::Int, ::Val{pre}, ::Val{post}
    ) where {pre, post}
    A = _local_K(f, r, n, k)
    if pre
        A = -_region_conductivity(f, k) * A
    end
    if post
        A = A ⋅ inv(f.K₀)
    end
    return A
end

"""
    local_temperature(f_or_sphere, [K₀,] x, ∇T∞; side = :outer) -> Number

Local temperature (or pressure / potential) at `x` under the remote uniform
gradient `∇T∞`: `T = f(r) (n·∇T∞)`, normalized so that `T → ∇T∞·x` at
infinity and `T(0) = 0`.

Discontinuous across a [`KapitzaInterface`](@ref); use `side` to pick the
limit (see [`get_layer`](@ref)).
"""
function local_temperature end

"""
    local_gradient(f_or_sphere, [K₀,] x, ∇T∞; side = :outer) -> Vec{3}

Local temperature gradient at `x` under the remote uniform gradient `∇T∞`,
i.e. `local_gradient_gradient_loc(…) · ∇T∞`.
"""
function local_gradient end

"""
    local_flux(f_or_sphere, [K₀,] x, ∇T∞; side = :outer) -> Vec{3}

Local flux at `x` under the remote uniform gradient `∇T∞`, with the
Fourier / Fick convention `q = −k(x) ∇T(x)`, `k(x)` the conductivity of the
region containing the point (the matrix outside).
"""
function local_flux end

function _local_T(f::LayeredSphereTransportFields, r, n, k::Int, ∇T∞)
    f_over_r, _ = _cond_radial_scalars(f, k, r)
    g = TensND._extract_vec(∇T∞)
    return f_over_r * r * sum(n[i] * g[i] for i in 1:3)
end

function local_temperature(
        f::LayeredSphereTransportFields, x, ∇T∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    return _local_T(f, r, n, _resolve_layer(f.sphere, r; side, layer), ∇T∞)
end

function local_temperature(
        f::LayeredSphereTransportFields, r, θ, φ, ∇T∞;
        side::Symbol = :outer, layer = nothing,
    )
    rr, n = _radial_frame(r, θ, φ)
    return _local_T(f, rr, n, _resolve_layer(f.sphere, rr; side, layer), ∇T∞)
end

# `𝐀 · G` in closed form.  `TensTI{2}` is a *structured* tensor: `⋅` against a
# plain vector falls through to `LinearAlgebra.dot`, which flattens both
# operands and computes an inner product of mismatched lengths.  Writing the
# transverse/axial split out is both correct and cheaper than materializing the
# 3×3 array.
@inline function _ti2_dot(f_over_r, fp, n, g)
    ng = n[1] * g[1] + n[2] * g[2] + n[3] * g[3]
    return TensND.Vec{3}(
        ntuple(i -> f_over_r * (g[i] - ng * n[i]) + fp * ng * n[i], 3)
    )
end

function _local_gradient_vec(f::LayeredSphereTransportFields, r, n, k::Int, ∇T∞)
    f_over_r, fp = _cond_radial_scalars(f, k, r)
    return _ti2_dot(f_over_r, fp, n, TensND._extract_vec(∇T∞))
end

function local_gradient(
        f::LayeredSphereTransportFields, x, ∇T∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    return _local_gradient_vec(f, r, n, _resolve_layer(f.sphere, r; side, layer), ∇T∞)
end

function local_gradient(
        f::LayeredSphereTransportFields, r, θ, φ, ∇T∞;
        side::Symbol = :outer, layer = nothing,
    )
    rr, n = _radial_frame(r, θ, φ)
    return _local_gradient_vec(f, rr, n, _resolve_layer(f.sphere, rr; side, layer), ∇T∞)
end

function local_flux(
        f::LayeredSphereTransportFields, x, ∇T∞; side::Symbol = :outer, layer = nothing
    )
    r, n = _radial_frame(x)
    k = _resolve_layer(f.sphere, r; side, layer)
    return -_region_conductivity(f, k) * _local_gradient_vec(f, r, n, k, ∇T∞)
end

function local_flux(
        f::LayeredSphereTransportFields, r, θ, φ, ∇T∞;
        side::Symbol = :outer, layer = nothing,
    )
    rr, n = _radial_frame(r, θ, φ)
    k = _resolve_layer(f.sphere, rr; side, layer)
    return -_region_conductivity(f, k) * _local_gradient_vec(f, rr, n, k, ∇T∞)
end

# ── Convenience methods taking `(sphere, C₀)` / `(sphere, K₀)` directly ──────
#
# Each rebuilds the recurrence, so they are meant for one-off evaluations.
# Anything sweeping many points should build the cached solution once.

for name in (
        :local_strain_strain_loc, :local_stress_strain_loc,
        :local_strain_stress_loc, :local_stress_stress_loc,
    )
    @eval $name(sphere::LayeredSphere, C₀::TensND.TensISO{4, 3}, args...; kw...) =
        $name(LayeredSphereFields(sphere, C₀), args...; kw...)
end

for name in (:local_strain, :local_stress, :local_displacement)
    @eval $name(sphere::LayeredSphere, C₀::TensND.TensISO{4, 3}, args...; kw...) =
        $name(LayeredSphereFields(sphere, C₀), args...; kw...)
end

for name in (
        :local_gradient_gradient_loc, :local_flux_gradient_loc,
        :local_gradient_flux_loc, :local_flux_flux_loc,
        :local_temperature, :local_gradient, :local_flux,
    )
    @eval $name(sphere::LayeredSphere, K₀::TensND.TensISO{2, 3}, args...; kw...) =
        $name(LayeredSphereTransportFields(sphere, K₀), args...; kw...)
end
