# [Adding a homogenization scheme](@id dev-adding-scheme)

!!! note "Which cell does your scheme serve?"
    `homogenize` accepts any
    [`AbstractHomogenizationCell`](@ref MeanFieldHomogenization.Core.AbstractHomogenizationCell)
    — an [`RVE`](@ref) or a `Laminate`. Schemes stay typed on the **concrete**
    cell they serve (`_evaluate(rve::RVE, ::MyScheme, …)`), so a scheme applied
    to a cell it does not serve falls through to the error-raising fallback
    instead of dispatching somewhere wrong. Type the *scheme* argument too:
    leaving it untyped is ambiguous with that fallback.

`MeanFieldHomogenization.Schemes` ships Voigt/Reuss, dilute (direct and dual),
Mori-Tanaka, Maxwell, Ponte-Castañeda–Willis, self-consistent (symmetric and
asymmetric) and differential, each in an elastic and an ageing-viscoelastic
flavor. A new scheme slots in beside them:

1. Create `src/Schemes/<scheme_name>.jl` and `include` it from
   `src/Schemes/Schemes.jl`.
2. Declare the scheme type (`struct MyScheme <: HomogenizationScheme`) in
   `src/Schemes/scheme_types.jl`, and register its symbol alias in
   `SCHEME_ALIAS` (`src/Schemes/homogenize.jl`) if you want
   `homogenize(rve, :myscheme, :C)` to work.
3. Implement `_evaluate(rve, ::MyScheme, ::Val{p}; kw...)`. Provide **one
   method per tensor order** — dispatch on the order of `matrix_property(rve, p)`
   — so the scheme serves elasticity and transport from one implementation, as
   `_mt_dispatch` does in `mori_tanaka.jl`.
4. **Go through `contribution_helpers.jl`.** Never call
   `stiffness_contribution` and friends directly on a phase geometry: the
   `_phase_*` helpers apply the amount, honor the `symmetrize` setting, hand
   the kernel the correctly pre-projected reference medium, and branch on
   [`is_homogeneous_inclusion`](@ref). Two invariants stated in that file's
   header must hold in your kernel too — all helpers of a given evaluation must
   share the same projected `P₀`, and orientation averaging must never be
   folded into a tensor product.
5. Prefer the bundled seams (`_phase_dilute_and_contribution`,
   `_phase_dilute_and_stress_average`, `_phase_compliance_and_contribution`)
   when you need two objects per phase: they share the single expensive solve.
6. Consider what your kernel *requires* of an inclusion. Needing only the
   contribution tensors keeps the scheme usable with inclusions entered through
   gate C of the [inclusion contract](@ref dev-adding-inclusion); needing
   the concentration tensor restricts it.
7. Export through `src/Schemes/Schemes.jl` and re-export from
   `src/MeanFieldHomogenization.jl`.
8. Add a unit test under `test/Schemes/` and a section in
   `docs/src/manual/schemes.md`.

!!! note "A scheme need not act on an RVE"
    Two cells other than `RVE` carry scheme kernels, and both follow the same
    split: the scheme **type** is declared in `src/Schemes/scheme_types.jl` (so
    the hierarchy and `SCHEME_ALIAS` stay in one place) while its `_evaluate`
    method lives with the cell it acts on — `Laminated` in `src/Laminates/`, and
    the two N-body schemes [`ClusterModel`](@ref) / [`EquivalentInclusion`](@ref)
    in `src/Assemblies/`. If your scheme needs microstructural information an
    `RVE` does not carry — positions, in the N-body case — add a cell rather
    than widening `RVE`, and remember that MFH Studio discovers every concrete
    `HomogenizationScheme` automatically: a scheme the interface cannot drive
    must be listed in `ASSEMBLY_SCHEMES` (`tools/mfhstudio/mfhstudio/model.py`)
    so it is filtered out of its catalog.

