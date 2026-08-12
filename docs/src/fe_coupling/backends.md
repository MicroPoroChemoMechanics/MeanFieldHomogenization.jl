# [Ferrite backend](@id fe-backends)

The material contract knows nothing about any finite-element library, so
coupling to one needs almost no adapter. What Ferrite does *not* provide is
per-quadrature-point material state, so that bookkeeping — and an element
routine — is all this extension adds.

Activated by `import Ferrite` **alone**:

```julia
import Ferrite
const EXT = Base.get_extension(
    MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt
)
```

!!! note "Why a second extension"
    `MeanFieldHomogenizationFerriteExt` serves the [opposite
    coupling](@ref man-fe-inclusions) and needs `FerriteGmsh` and `Gmsh` too. A
    structural computation that only wants a homogenized material law has no
    reason to pull a ~100 MB gmsh artifact into its environment — or into a
    documentation build.

## The three helpers

| | |
|:--|:--|
| `mfh_states(mat, dh, cv)` | one [`initial_state`](@ref) per quadrature point, as the nested `[cell][qp]` vector Ferrite loops expect |
| `mfh_element!(Ke, re, cv, mat, ue, states, states_old, Δt; cache)` | element stiffness and internal force, small strain, plane strain |
| `annulus_grid(Ri, Ro, nr, nθ)` | a structured annular sector, built by bending a rectangle — no gmsh |

Keep **two** state arrays and swap only once a step has converged;
[`material_response`](@ref) never mutates its argument, so a rejected Newton
iteration is undone by not swapping.

```julia
states     = EXT.mfh_states(mat, dh, cv)
states_old = EXT.mfh_states(mat, dh, cv)
cache      = MaterialCache()

for cell in CellIterator(dh)
    reinit!(cv, cell)
    fill!(Ke, 0); fill!(re, 0)
    ed = celldofs(cell)
    EXT.mfh_element!(Ke, re, cv, mat, u[ed],
                     states[cellid(cell)], states_old[cellid(cell)], Δt; cache)
    assemble!(assembler, ed, Ke, re)
end
```

A worked model is [`scripts/88_fe_thick_cylinder.jl`](@ref fe-thick-cylinder).

!!! warning "Two traps"
    A `DofHandler` numbers dofs **by cell traversal, not node order** — read
    nodal results with `evaluate_at_grid_nodes(dh, u, :u)`, never by indexing
    `u` with a node number. And a [`MaterialCache`](@ref) is an unlocked `Dict`:
    use one per thread, or none.

## Other codes

Only Ferrite is wired today. The contract itself is backend-agnostic, so a
Gridap, FEniCSx or Abaqus UMAT driver needs the same two pieces — a per-point
state container and an element routine — plus, outside Julia, a way to carry
`ε` and `σ` across the boundary ([`voigt_strain`](@ref) and
[`voigt_stress`](@ref) exist for that). Those are on the
[roadmap](@ref dev-roadmap), not in the package.
