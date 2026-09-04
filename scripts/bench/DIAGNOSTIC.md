# Optimization audit — MeanFieldHomogenization.jl / TensND.jl

Measured diagnostic. Campaign of 2026-07-27, re-verified and extended
2026-08-10.

Machine: znver3, Julia 1.12.6, `JULIA_NUM_THREADS=1`, `opt=2`. Baseline recorded
on a `git` worktree of commit `0cf9fd5` **without instrumentation**; noise floor
1.5–2.5 % depending on the campaign.

Every number below is **measured**, never estimated. Where a measurement turned
out to be wrong it is flagged as such rather than deleted.

---

## 0. Starting point, and what it revealed

`_Qnn_direct` (`src/Core/green_kernel.jl:98-110`) contracted a 4th-order tensor
with the Green kernel through a 6-deep loop `for p,q,r,s2,α,β in 1:3` — 729
iterations × ~12 flops — where the contraction factors into `U = (C·n̂)·ξ` then
`B = U·K⁻¹·Uᵀ` (~100 flops).

Two facts, established by reading and by exhaustive `grep` over `src/`, `ext/`,
`test/`, `docs/`:

1. **The factored form already existed twice in the repository** —
   `Cracks/green_residue.jl:33-72` (`Tncon → V → M·Vᵀ`) and
   `Core/green_helpers.jl:96-119`. `_Qnn_direct` was the naive version left
   behind.
2. **`_Qnn_direct` is called nowhere**, nor is `_acoustic_tensor`, its only
   consumer. Dead code: to delete, not to optimize.

The systematic sweep found the same class of problem elsewhere — and three
categories heavier than the loop that started it.

---

## 1. The dominant cost: `hill_tensor` / `cod_tensor` called twice

Call chain, verified line by line:

- `Schemes/mori_tanaka.jl` `_mt_4` calls, **per phase**,
  `_phase_dilute_concentration` *and* `_phase_stiffness_contribution`;
- the first does `strain_strain_loc(geom, P_i, P₀_proj)`;
- the second goes through `Core.stiffness_contribution` →
  `contribution.jl:33-41` → **`strain_strain_loc(incl, C₁, C₀)`, identical
  arguments**;
- `strain_strain_loc` (`localization.jl:72-82`) does
  `P = hill_tensor(incl, C₀; kw...)` — the expensive object.

An exact factor of 2 on the dominant cost, at every MT evaluation and every SC
iteration. Three further instances of the same pattern:

| instance | file | duplicated cost |
|---|---|---|
| cracks (MT) | `_phase_compliance_contribution` then `Cracks/compliance.jl:168-175` | `cod_tensor` ×2 |
| SC / ASC | `self_consistent.jl:136-140` → `_phase_stress_strain_average` recomputes `A_raw` as soon as `sym ≠ NoSymmetrize` and `P_i` is not isotropic | `hill_tensor` × number of bins |
| `LayeredSphere` | `strain_strain_loc` + `stiffness_contribution` + internal `_membrane_surface_stress` | Hervé-Zaoui recurrences ×3 |
| `LayeredSpheroid` | `conductivity_contribution` already calls both localizations | confocal recurrences ×3 |

### Fix and measured gains (tier 1, bit-for-bit gate)

New generics `Core.loc_and_stiffness` / `Core.loc_and_stress_average`,
dispatched on the **inclusion class** (safe fallback on `AbstractInclusion`,
fast path on `AbstractEllipsoidalInclusion`), plus dedicated bundles for cracks,
`LayeredSphere` and `LayeredSpheroid`.

| case | time | allocations | work (counters) |
|---|---|---|---|
| `schemes/mt.porous.oblate.isosym` | **−50.5 %** | −14.2 % | hill 2→1 |
| `schemes/mt.aniso_matrix` | **−49.9 %** | −49.9 % | hill 2→1, nodes 210→105 |
| `schemes/mt.crack.penny.tri` | **−49.2 %** | −49.9 % | cod 2→1, nodes 210→105 |
| `schemes/mt.crack.penny` | −26.3 % | −21.0 % | cod 2→1 |
| `schemes/mt.theta_binned_ti.n20` | **−18.6 %** | −7.2 % | **hill 40→20** |
| `schemes/mt.iso2.sphere` | −2.3 % | −5.4 % | hill 2→1 |

The counter channel confirms the gain comes from removed work and not from a
change in adaptive-quadrature behavior: the node count tracks the Hill call
count exactly (210→105).

**Gate**: 67/67 cases bit-for-bit identical (sha256 of the `%.17g` rendering),
full suite green (7142 pass, 0 failure).

### Measured downside, owned

On the cheapest cases the deduplication **costs** more than it saves: the
returned tuple `(A, N)` is allocated and boxed (the localization chain is
inferred `Any` by construction, see §5), while the Hill solve saved is
analytical and nearly free.

| case | time | allocations |
|---|---|---|
| `schemes/mt.conductivity.iso2` (analytical 2nd-order iso Hill) | **+11.3 %** (629 → 700 ns) | **+22.2 %** (432 → 528 B) |

Measured directly, 2000 samples, min and median agreeing. It is the only case in
the suite where the balance is negative; the trade (−50 % on the expensive cases
against +70 ns on the cheapest) looks clearly favorable, but it is real and
recorded here rather than passed over.

The other small rises (`asc.stiffness` +3.1 %, `sc.newton` +3.6 %,
`sc.porous.sphere.phi30` +5.3 %) are on bodies where the `sym isa NoSymmetrize`
shortcut **already** did a single solve: nothing to deduplicate, only the tuple
to add.

---

## 2. Pre-existing inconsistency in the reference medium

Found while writing tier 1, unrelated to the original request.

| helper | order | reference actually passed |
|---|---|---|
| `_phase_dilute_concentration` | 4 and 2 | `P₀_proj = _project_matrix(P₀, sym)` |
| `_phase_stiffness_contribution` | 4 | `P₀_proj` |
| `_phase_stiffness_contribution` | **2** | **raw `P₀`** |
| `_phase_compliance_contribution` | **4 and 2** | **raw `P₀`** |

So within one MT scheme, `A_dil` and `N` of the **same phase** see two different
reference media as soon as `symmetrize ≠ NoSymmetrize`.

Invisible to the current tests: every phase carrying a `symmetrize` in `test/`
has an isotropic matrix **and** an isotropic phase property, so
`_project_matrix(P₀, IsoSymmetrize())` is a no-op to ~1e-16.

**Status**: tier 1 preserves current behavior — the bundles concerned fall back
to the two separate calls when `P₀_proj !== P₀` (the guard is object identity,
covering `NoSymmetrize` **and** `TISymmetrize(reference_projection = :none)`).
Unifying them is tier 2, not yet applied.

---

## 3. Redundancy in the integrands (tier 3, fixed)

| site | finding | fix |
|---|---|---|
| `Core/green_helpers.jl:71-80` | `Kns[i,j] = Σ C_{ikjl}(n̂_k ξ_l + ξ_k n̂_l)` in 81 iterations. By major symmetry `C_{ikjl}=C_{jlik}`, the 1st term **is** `Vs[i,j]` and the 2nd `Vs[j,i]`: `Kns == Vs + Vsᵀ`, zero flops. `Ks` symmetric (half the entries computed twice). | drop the `skn` accumulator; `Ks` over `i ≤ j` |
| `Elasticity/hill_3d_aniso_residue.jl:96-111` | the `for α in 1:21` loop passes **the same `Q` and the same `z`** 21 times; each call redoes `derivative(Q)` up to `Q4` and recomputes `sqrt(1+z²)`, `log(z+√)` | cache per (φ, root) |
| `hill_3d_aniso_decuhr.jl:56-66`, `hill_3d_cylinder_aniso.jl:44-56` | 4-deep loop for a **symmetric** `K`, then `Matrix` + LU `inv` per node | `_sym3_inv_acoustic` already exists (`hill_3d_aniso_nestedquadgk.jl:47-68`) |
| `hill_2d_aniso.jl:67-74` | 16 separate `quadgk` calls, each evaluating the full 16-component integrand to keep one | one vector `quadgk` — **16×** |
| `Cracks/green_residue.jl:66-92` | `Bpoly` filled for all 9 `(i,k)` then symmetrized afterwards; `acc = acc + …` allocates a fresh `Polynomial` at each of the 54 terms | loop `i ≤ k`, reused buffer |
| `Cracks/cod_numerical.jl:42-56` | `Tncon`/`A` (invariant in φ) recomputed at every node — the sibling `_direct` functions already hoist them | precomputed context |
| `Schemes/self_consistent.jl:337,341,367,379` | 3 free evaluations of the full SC residual (= one `hill_tensor` per phase) per Newton iteration | carry `r_new` forward |

Cost marker, measured: a triaxial/triclinic `hill_tensor` in `:nestedquadgk`
allocates **103 MB** and takes 83 ms for **13 665 integrand evaluations**. In
`:residues`, 6.8 ms and 105 nodes.

### Measured gains (tier 3, 1e-14 gate)

| case | time | allocations | what changed |
|---|---|---|---|
| `kernels/hill.decuhr.tri.321` | **−62.0 %** | **−80.7 %** | `_sym3_inv_acoustic` instead of `Matrix` + LU `inv` per node |
| `kernels/hill2.aniso` | **−35.2 %** | −18.5 % | one vector `quadgk` instead of 16 scalar ones |
| `kernels/hill.dual.nqgk.tri` | **−21.5 %** | −0.0 % | `Kns = Vs + Vsᵀ` and `Ks` over `i ≤ j` |
| `schemes/sc.newton` | −2.1 % | **−18.4 %** | carrying `r_new` forward, `Tref` from the loop's 1st residual |

The `nodes` counter is **identical** before and after on every quadrature case
(13 665, 21 825, 105, 315 …): the adaptive quadrature did not change behavior,
so the time comparison is over the same work.

Three planned items were measured as unprofitable or too invasive and were
**not** applied: the `prepare_logI`/`prepare_logz` cache of the 21-loop on the
`:residues` path (most invasive, expected gain unverified), the `i ≤ k` loop of
`Cracks/green_residue.jl`, and the precomputed-context form of `_Qnn_star_*`.
The `:residues` path leaves the tier unchanged (`hill.residues.tri.321` −0.4 %,
within noise).

---

## 3 bis. StaticArrays on the hot paths (tier 4, 1e-14 gate)

`_qnn_pair_components!` — the innermost loop of the whole `Cracks` module —
wrote into a caller buffer and built ~10 heap `Matrix{T}` 3×3 **per node α**
(`Vp, Vm, Kp, Km` by diffusion, `iKp, iKm` via `_inv3`, then four temporaries for
the two `(V·iK)·Vᵀ`). It became a **pure** function returning an
`SMatrix{3,3,T}`; `_A_and_Tn`, `_phi_cache` and `_inv3` also return static.

| case | time | allocations |
|---|---|---|
| `kernels/cod.nqgk.ellipse03.tri` | **−85.3 %** (4.35 ms → 639 µs) | **−99.4 %** (18.48 MB → 111 kB) |

The largest single gain of the campaign. The checksum moves by **6.9e-16** —
pure floating-point reassociation, 1 ULP, expected as soon as a generic matrix
product is replaced by its unrolled static form.

Two dead ends abandoned along the way, both in the direction of readability:

- a first `_A_and_Tn` in `mod1`/`fld1` index arithmetic over tuples: unreadable,
  and `ntuple(f, 27)` **without `Val` is type-unstable** — it would have
  introduced exactly the regression it was meant to remove. Replaced by
  `MArray` → `SArray` with the original loops intact;
- a closure `iK = (i,j) -> iKt[…]` in the hot DECUHR loop — the boxing trap
  already met in this repository. Replaced by direct tuple indexing.

Out of scope, owned: the `zeros(T,3,3,3,3)` at kernel tails (cold paths),
`Viscoelasticity/` (everything sized by the number of time steps),
`LayeredSpheroids/` (dynamic Legendre truncation), and the
`Matrix{Polynomial{ComplexF64}}` (3×3 but non-`isbits` elements: an `SMatrix`
would remove no allocation). The `SVector{21,T}` port of the Hill back-end
integrand returns was not done — not needed for the gain, and the
`Integrals`/DECUHR path needs a separate check of buffer mutability.

---

## 4. TensND — `TensOrtho` was slower than the generic path

The sharpest finding of the campaign, measured:

| case | time | allocations |
|---|---|---|
| `tensnd/dcontract.ortho_ortho` | **13 860 ns** | 9 136 B |
| `tensnd/dcontract.iso_ortho` | 10 900 ns | 8 992 B |
| `tensnd/dcontract.gen_gen` (**generic** tensor) | **20 ns** | 928 B |
| `tensnd/dcontract.ti_ti` | 11 ns | 224 B |
| `tensnd/dcontract.iso_iso` | 3 ns | 80 B |

Contracting two **structured orthotropic** tensors was **~690× slower** than
contracting two dense generic ones. Same hierarchy on access:

| case | time | allocations |
|---|---|---|
| `tensnd/getindex.ortho` | **2 792 ns** | 2 304 B |
| `tensnd/getindex.ti` | 17.8 ns | 96 B |
| `tensnd/getindex.iso` | 2.9 ns | 48 B |
| `tensnd/collect.ortho` | **48 810 ns** | 60 448 B |
| `tensnd/collect.iso` | 491 ns | 784 B |

Causes identified:

1. `Base.getindex(t::TensOrtho, i,j,k,l) = get_array(t)[i,j,k,l]`
   (`tens_walpole.jl:1025`) — one `Array{T,4}` allocation and an 81-iteration
   loop at ~40 multiplications **per scalar access**. `TensOrtho` was the
   **only** structured type left on the dense path: `TensISO`,
   `TensTI{4,N=5/6/8}` and `TensTI{2,N=2/3}` all had closed-form `getindex`.
   Since `TensOrtho <: AbstractArray`, any generic traversal became O(81²) for
   an O(81) operation.
2. **No closed-form `dcontract` for `TensOrtho`**: every `⊡` went through
   `same_basis` → `change_tens` → dense `get_array` on **both** operands before
   any arithmetic. Yet the file header (`tens_walpole.jl:863-870`) already
   states the structure: in the material frame the KM matrix is block-diagonal
   `[3×3 sym] ⊕ diag(2C₄₄,2C₅₅,2C₆₆)`, so `A⊡B` is one 3×3 product plus 3 scalar
   products. Exact structural twin of the `inv` already implemented
   (`inv.ortho`: 8.6 ns — proof that the closed form works).

### Fixes and measured gains (tier 5, TensND v0.2.6, 1e-14 gate)

**(a) Closed-form `getindex(::TensOrtho)`.** `_ortho_entry` is factored out of
`get_array` so a scalar access evaluates **one** component instead of 81.

| case | time | allocations |
|---|---|---|
| `tensnd/getindex.ortho` | **−99.8 %** (2 792 → 4 ns) | **−95.8 %** |
| `tensnd/collect.ortho` | **−94.7 %** (48.8 → 2.47 µs) | **−98.6 %** |
| `tensnd/get_array.ortho` | −44.9 % | +0.0 % |

**(b) `tensor_or_array` was type-unstable — the real seam.** `dim` comes from
`size(tab, 1)`, so it is a **runtime** value: writing `Tensor{order, dim}(tab)`
builds a non-concrete type at compile time and construction becomes fully
dynamic — **3 147 ns and 3 120 B** for an 81-element array, against 311 ns to
produce that array. Every structured tensor rejoins the generic route through
this function (`change_tens` → `same_basis` → every binary operation), so the
cost was paid **twice per `⊡`** between structured operands. `dim` is now
channelled through `Val`.

| case | time | allocations |
|---|---|---|
| `tensnd/dcontract.iso_ortho` | **−58.2 %** (10.9 → 4.38 µs) | −46.3 % |
| `tensnd/dcontract.ortho_ortho` | **−57.0 %** (13.9 → 4.77 µs) | −45.5 % |

A wider gain than the `TensOrtho` case originally targeted: it covers **every**
operand pair that falls back to the dense path.

**(c) The closed-form `dcontract`, and the bottleneck it exposed.** Taken up
later, after establishing the convention instead of assuming it.

`inv_KM` had no frame problem: `KM(t)` is the **canonical** Kelvin-Mandel form,
`inv_KM` reads it back as such, and the round trip is exact for all eight forms
(orders 2 and 4, symmetric or not, dim 2 and 3). The prototype did
`inv_KM(Mₐ·M_b)` with `Mₐ`, `M_b` the KM of the **material frame**: it
interpreted material components as canonical ones. In the canonical frame `Q = I`
and the error was invisible (1.8e-12); in a rotated frame it was 6039. The
congruence was missing.

Verified on three frames rather than postulated:

    KM(t) == Q · KM_material(t) · Qᵀ          (5.3e-15)

where `Q` is the Kelvin-Mandel representation of `R ⊠ˢ R`, i.e.
`KM(rot6(θ,ϕ,ψ))` — which TensND already had. `Q` is **orthogonal** (2.2e-16):
exactly what Kelvin-Mandel buys over Voigt, where the analogous matrix is not.

The result is not a `TensOrtho`: the product of two symmetric 3×3 blocks is not
symmetric unless they commute, so `A ⊡ B` is orthotropic **without major
symmetry** — 12 constants for 9 stored, discrepancy measured at 87.6, not noise.
The same widening as `TensTI{4}` N=5 → N=6. The method therefore returns the
`TensCanonical` the generic route produced, to 6.2e-16.

And profiling the closed form, the remaining 1115 ns were **not in the algebra**:

| | before | after |
|---|---|---|
| `frame(A) == frame(B)` | **1117 ns** | 162 ns |
| everything else combined | ~130 ns | ~130 ns |

`AbstractBasis <: AbstractMatrix`, and its `getindex` went through
`vecbasis(ℬ, :cov)` — the `Symbol` overload, which builds `Val(var)` from a
runtime value: dynamic dispatch on **every scalar access** (67 ns). The generic
`==` of `AbstractArray` did 18 of them. That cost was paid by every generic
traversal of a basis and by every `_check_same_reference` — the guard cost more
than the algebra it guarded.

| case | before v0.2.6 | after (b) | after (c) |
|---|---|---|---|
| `dcontract.ortho_ortho` | 13 860 ns / 9 136 B | 4 770 ns | **164 ns / 304 B** |
| `dcontract.iso_ortho` | 10 900 ns | 4 380 ns | **164 ns** |
| `inv_KM` 6×6 | 228 ns | — | **37.5 ns** |

That is **−98.8 %** on what had been named the most profitable remaining
opportunity. The ~690× factor against the generic tensor falls to ~3.3×.

Two `Dual` bugs from §5 were fixed along the way (`TensTI` constructor with mixed
eltypes, and `_ti8_to_ti6`); the two corresponding `@test_broken` became real
`@test`s.

---

## 4 bis. The outer products went through a contraction engine

*Added 2026-08-10 — TensND v0.3.3.*

The audit had noted that several `OMEinsum` calls in
`TensND/src/array_utils.jl` perform **no summation at all**: their output
indices are the union of the input indices. Four functions were affected —
`otimes` (a plain outer product), `otimesu` and `otimesl` (the same product
with the two operands' indices *interleaved*), and `sotimes` (the average of
the first two). Measured:

| case | before | after | change |
|---|---:|---:|---:|
| `otimes(3×3, 3×3)` | 3517 ns / 4144 B | **163 ns / 864 B** | **−95.4 %** |
| `otimes(3, 3)` | 3240 ns / 3200 B | **50.7 ns / 240 B** | **−98.4 %** |
| `otimesu(3×3, 3×3)` | 8337 ns / 7680 B | **158 ns / 864 B** | **−98.1 %** |
| `otimesl(3×3, 3×3)` | 8180 ns / 7632 B | **159 ns / 864 B** | **−98.1 %** |
| `sotimes(3×3, 3×3)` | 8810 ns / 10016 B | **307 ns / 992 B** | **−96.5 %** |

The time was machinery — `EinCode` construction, code selection, dispatch —
around one multiplication per element.

All four are now a single broadcast. The key observation is that both index
lists are **increasing**: `otimesu` sends `t1`'s indices to output positions
`(1…o1−1, o1+1)` and `t2`'s to `(o1, o1+2…)`, neither of which reorders an
operand's own axes. So each operand can be reshaped in place with singleton
axes where the other's indices sit, and the product needs no permutation pass:

```julia
s1, s2 = _otimes_shapes(t1, t2, ec1, ec2, Val(n))
reshape(t1, s1) .* reshape(t2, s2)
```

`sotimes` fuses its two terms into one broadcast, turning three allocations
into one (495 ns / 3136 B before the fusion, 307 ns / 992 B after).

Verified **bit-for-bit against the einsum implementations they replace** on
nine shape combinations per function — non-square, order-3, mixed-order, and a
first-order second operand — with `===` element equality, `ForwardDiff.Dual`
preserved, and the full TensND suite green.

### The regression this introduced, and how it surfaced

A first draft of `otimes` used `vec(t1) .* transpose(vec(t2))` — the same
layout, and **faster** on plain arrays (65 ns) because it is a BLAS-shaped
rank-1 product. It passed the array oracle, the array benchmark and the whole
TensND suite.

It still broke the documentation build:

    ArgumentError: the (no-op) transpose is discontinued for `Tensors.Vec`

`transpose` of a first-order `Tensors` array is deliberately discontinued
upstream, and no test in the suite called `otimes` with a first-order
`Tensors` operand — only `docs/` did, through
`scripts/20_green_function.jl`. The singleton-axis form transposes nothing and
has no such restriction; the 163 ns above is that correction, still 21× faster
than the einsum it replaces.

The lesson is about the oracle, not about the arithmetic: it was written over
`Array` alone, while the function's callers pass `Tensors.Vec`,
`Tensors.Tensor`, `SymmetricTensor` and `Tens`. The check now covers those
types explicitly.

### Not done

`Core._quadgk`-style dead code aside, the remaining `OMEinsum` calls
(`dcontract`, `qcontract`, `contract`, the single `ein"ijl,lk->ijk"` in
`compute_Christoffel`) do contract, so the dependency stays.

## 4 ter. `best_sym_tens` derived candidates it never read

*Added 2026-08-10 — TensND v0.3.3.*

`best_sym_tens` computed **both** the transversely isotropic axis and the
orthotropic frame on every call, each from its own fresh
`Array(get_array(newt))` — so the array was materialized three times — and did
so even when `proj` named neither symmetry.

| case | before | after |
|---|---:|---:|
| `best_sym_tens(C; proj = (:ISO,))` | 11.3 µs / 14016 B | **7.1 µs / 7568 B** (−37 % / −46 %) |
| `best_sym_tens(C)` (all three) | 17.6 µs / 22640 B | **17.3 µs / 21168 B** |

The candidates are now derived only when `proj` asks for the symmetry that
reads them, and from the array already materialized. The full call genuinely
needs both, so it gains only the duplicate materializations; a restricted
`proj` gains the eigen-decomposition it was never going to use
(`_candidate_TI_axis` 2.0 µs, `_candidate_ORTHO_frame` 3.0 µs).

Behavior is unchanged, and pinned: `test_tensor_products.jl` checks that
restricting `proj` does not alter the numbers reported for a given symmetry,
on the nesting ISO ⊂ TI ⊂ ORTHO where restricting legitimately changes *which*
symmetry is reported.

---

## 5. Pre-existing bugs found along the way

Each **verified on a worktree of the reference commit** before being asserted.

1. **`Core._quadgk` is dead code** (`Core/quadrature.jl:17`) — the wrapper that
   "all submodules should use" according to its own header comment. All of them
   call `QuadGK.quadgk` directly. Brings the dead-code list to **15 functions**.
2. **`SelfConsistent` + a `TensTI` phase under `TISymmetrize`** →
   `MethodError: no method matching _hill_3d_ti_coaxial(::Ellipsoid{Spherical}, ::TensTI{4,Float64,8})`.
   The running estimate becomes an **8**-parameter TI (result of exact
   symmetrization) and the analytical TI-coaxial kernel only had methods for 5
   and 6. **Since fixed** (`_ti8_to_ti6`, `hill_3d_ti_coaxial.jl:310-315`).
3. **ForwardDiff through a `TensTI` phase property** → the inner constructor
   `TensTI{order,T,N}(::NTuple{N,T}, ::Tuple{T,T,T})` demanded the **same** `T`
   for the parameters and the axis, while the axis stays `Float64` when the
   parameters become `Dual`. TensND AD bug. **Since fixed.**
4. **Inherent type instability**: `hill_tensor` is not inferable, by design —
   `_resolve_algo(Val(method), incl, C₀)` resolves the algorithm at runtime,
   which is precisely what makes `:auto` and the `NestedQuadGK` fallback under
   `Dual` work. The whole localization chain is therefore `Any`. This is the
   cause of the §1 downside.

---

## 6. Method errors made and corrected

Recorded because they condition how much the numbers should be trusted.

1. **First instrumentation was type-unstable.** `_maybe_count(f)` returned the
   counting closure from a branch: `Union` return type, hence dynamic dispatch
   **per node** — exactly the cost the counters exist to measure. Replaced by a
   `struct _CountingFn{F}` and one branch per `quadgk` call. Verified
   afterwards: allocations **identical to the byte** (103 363 248 B on
   `hill.nqgk.tri`).
2. **`@inferred` on the bundles.** Asserting a property the code never had
   (§5.4). Replaced by the useful property: the bundle allocates strictly less
   than the two separate calls.
3. **`samples=7` in the time channel.** On cases with heavy GC pressure around
   ten µs, a single GC pause lands in the 7 samples and inflates the `minimum`
   by a factor of 2. It produced two phantoms: `asc.stiffness` **+103 %** (real:
   +4.5 %) and `sc.porous.sphere.phi30` **−46 %** (real: +3.6 %). Caught because
   the counter channel showed **unchanged** work. Channels decoupled:
   allocations at 7 samples (deterministic, `min == max` over the 67 cases),
   time up to 10 000 samples within a 2 s budget.
4. **Sub-µs cases are unreliable.** Even after (3), `control/dilute_dual.iso2`
   showed **+63 %** with allocations identical to the byte; direct measurement
   at 2000 samples: **−1.7 %**. Mechanism: the `evals` chosen by `tune!` differs
   between campaigns (175 vs 180), and comparing a min-of-1-call to a
   min-of-N-amortized-calls is not comparing the same statistic. The diff now
   reports `~evalsN→M` and **refuses** to call such a case moved.

Without the counter channel I would have reported a doubling of the ASC cost.
Without the cross-checking direct measurement I would have reported a 63 %
regression on a control case never touched.

---

## 7. Final campaign — summary

`--label=P7-ortho --baseline=baseline.json --gate=1e-14 --repeat-suite=2`,
67 cases, idle machine.

```text
24 moved, 0 gate failures, 1 control "regression",
21 unreliable (differing evals)
noise floor (controls, p90 of |Δt|/t) = 0.8 %
```

**Gains** (beyond the "moved" threshold = max(3×noise, 3 %)):

| case | time | allocations |
|---|---|---|
| `tensnd/getindex.ortho` | −99.7 % | −95.8 % |
| `tensnd/dcontract.ortho_ortho` | **−99.3 %** | **−94.9 %** |
| `tensnd/dcontract.iso_ortho` | **−99.3 %** | **−95.6 %** |
| `tensnd/collect.ortho` | −94.4 % | −98.6 % |
| `tensnd/inv_KM.gen` | **−87.0 %** | +0.0 % |
| `kernels/cod.nqgk.ellipse03.tri` | −85.3 % | **−99.4 %** |
| `kernels/hill.decuhr.tri.321` | −62.0 % | −80.7 % |
| `schemes/mt.aniso_matrix` | −50.8 % | −50.0 % |
| `schemes/mt.porous.oblate.isosym` | −50.2 % | −14.2 % |
| `schemes/mt.crack.penny.tri` | −49.7 % | −50.0 % |
| `schemes/mt.crack.penny` | −49.0 % | −35.3 % |
| `tensnd/get_array.ortho` | −44.4 % | +0.0 % |
| `kernels/hill2.aniso` | −35.2 % | −18.5 % |
| `kernels/hill.dual.nqgk.tri` | −21.5 % | −0.0 % |
| `schemes/mt.theta_binned_ti.n20` | −17.3 % | −7.2 % |

**Correctness**: 63 of 67 cases stay **bit-for-bit identical** (`0.0e+00`). The
four that move all do so by floating-point reassociation —
`cod.nqgk.ellipse03.tri` at 6.9e-16, `hill.decuhr.tri.321` at 1.7e-18 (going
static), `dcontract.iso_ortho` at 4.0e-17 and `dcontract.ortho_ortho` at 1.9e-17
(closed form) — all far below the 1e-14 tolerance.

**The control "regression" is not one.** `control/alv.voigt.n50` comes out at
**−5.6 %**, i.e. *faster*; the harness flags any control deviation without
looking at the sign. Verified rather than assumed: reproducible over five fresh
processes (−4.8 to −6.8 %), allocations identical to the byte, bit-for-bit
checksum, work counters unchanged. An A/B canceling the single `bases.jl` fix
shows **it is not that one**; the remaining candidate is `inv_KM`, which ALV
calls in a loop to convert its Mandel blocks. Never formally isolated.

That is the limit of the control set: it was chosen assuming the tiers would not
touch shared primitives. `inv_KM`, `tensor_or_array` and basis comparison are
global, so a control can legitimately move — in the right direction here.

**What goes up.** The only deterministic item is `schemes/mt.conductivity.iso2`,
+22.2 % allocation — the bundle-tuple downside of §1, on the cheapest case of
the suite (528 B in total). The rest of the +4/+6 % cluster
(`hill.nqgk.tri.321` +4.4 %, `hill.nqgk.oblate.tri` +4.0 %,
`sc.porous.sphere.phi30` +5.0 %, `asc.stiffness` +4.5 %, `alv/trapezoidal.n50`
+5.9 %) is machine noise: the control `differential.iso2` moves by +2.5 % and
`alv/trapezoidal.n50` only traverses deleted code. The suspected mechanism was
checked rather than assumed — `_counted_quadgk` infers to the **same concrete
type** as `QuadGK.quadgk` called directly, so the instrumentation introduces no
instability on those paths.

### End-to-end verification

| | |
|---|---|
| MeanFieldHomogenization suite | **7154 / 7154** |
| TensND suite | green (AD 63/63, TI/ORTHO projections 89/89, NLopt 46/46) |
| `benchmark_strength` | **24 / 24** |
| `benchmark_hill_derivative` | **17 / 17** |
| `benchmark_nlayers` | §1-4, local constraints at 5.8e-16 |
| `benchmark_porous` | 134 / 140 — **identical to the pre-campaign commit**, figure for figure |
| Documenter build | exit 0, no orphaned docstring |

The 6 `benchmark_porous` failures (DifferentialScheme, φ ≥ 0.50, relative error
growing from 2.6e-03 to 6.2e-02) were replayed on a worktree of commit
`0cf9fd5`: **exactly the same values**. A pre-existing discrepancy against
echoes, unrelated to this campaign.

---

## What remains open

Re-verified 2026-08-10 (MFH `main`, TensND `main` / v0.3.3). Lines marked
"since fixed" refer to changes made after the campaign; the others are unchanged
and still current.

| topic | why it is not done |
|---|---|
| 12-parameter orthotropic container | `A ⊡ B` of two `TensOrtho` has no major symmetry (§4c), so the result falls back to `TensCanonical` as before. A dedicated container would keep the structure across a chain of contractions but would cost kernel methods on the MFH side. The precedent cited at audit time — the `MethodError` on `TensTI{4,T,8}` (§5.2) — is **since fixed**, so the risk is known and manageable; it no longer blocks |
| isolating the primitive behind `alv.voigt.n50`'s −5.6 % | investigation closed with no code consequence: the gain is real and beneficial, `bases.jl` ruled out by A/B, `inv_KM` the remaining candidate, never formally isolated. Kept for the record, nothing at stake |
| `prepare_logI` / `prepare_logz` cache on the `:residues` path | the most invasive tier-3 item; the path leaves the campaign unchanged |
| `SVector{21,T}` for Hill integrand returns | not needed for the gain obtained; the `Integrals`/DECUHR path needs a separate check of buffer mutability |
| concrete-type bases | TensND tier 7, not started: the abstract fields `Tens.basis::Basis`, `TensRotated.basis` and `CoorSystemNum.{χ,R,Γ}_func::Function` remain. `best_sym_tens` is **done** (§4 ter) |
| nested `Dual` tags through `NewtonDefault` | the `_sc_newton_seed` fix is in place (578aacb) with a real regression test (`test_self_consistent.jl`, "NewtonDefault ForwardDiff sensitivity (non-matrix phase)"); the NonlinearSolve extension separately avoids nested Duals through an IFT lift (`MeanFieldHomogenizationNonlinearSolveExt.jl:31-40`). The announced "next problem" is documented nowhere — no `@test_broken`, no comment — so it should either be pinned down concretely or closed out |
