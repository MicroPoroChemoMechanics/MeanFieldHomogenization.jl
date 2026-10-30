# [Ferrite backend](@id fe-backends)

!!! info "Before this page"
    [Materials](@ref fe-materials), for the contract the helpers below call at
    each quadrature point, and
    [The coupled poroelastic problem](@ref fe-poro-coupling) for the equations
    of the two-field element.

The finite-element code owns the mesh, the assembly and the global equilibrium;
at each quadrature point it hands the law the strain (and, for the coupled
element, the pore pressure) with the state of the last converged step, and gets
back the stress, the tangent and the new state. Here the law is an MFH material,
its state the openings of its cracks or whatever else the material carries from
step to step.

The material contract knows nothing about any finite-element library. What a
library needs is split in two: the pieces that do not depend on it are in the
package, and an adapter per library adds what only that library knows.

| in the package (any FE code) | in the Ferrite adapter |
|:--|:--|
| [`QuadratureStates`](@ref), [`commit!`](@ref), [`rollback!`](@ref): the states of the quadrature points, kept twice | [`material_element!`](@ref): the residual and the stiffness of one cell, for any material and any hypothesis |
| [`ThreeD`](@ref), [`PlaneStrain`](@ref), [`Axisymmetric`](@ref): how the strain of an element is a strain of the material | [`material_system`](@ref): the assembly over the mesh and the boundary conditions, for the driver |
| [`quasistatic!`](@ref): the global Newton iteration, with step cutting | [`write_material_vtk`](@ref): the fields of the states in a VTK file |

The adapter is activated by `using Ferrite` (or `import Ferrite`) **alone**,
and its functions are those of the package: no `get_extension` is needed. The
[quickstart](@ref fe-quickstart) uses them end to end, and
[Step by step](@ref fe-step-by-step) explains each.

!!! note "Why a second extension"
    `MeanFieldHomogenizationFerriteExt` serves the [opposite
    coupling](@ref man-fe-inclusions) and needs `FerriteGmsh` and `Gmsh` too. A
    structural computation that only wants a material law has no reason to pull
    a ~100 MB gmsh artifact into its environment, or into a documentation build.

## Lower-level helpers

The first helpers of the adapter remain, reached through the extension module;
the coupled poroelastic element is among them.

```julia
import Ferrite
const EXT = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt)
```

| | |
|:--|:--|
| `mfh_states(mat, dh, cv)` | one [`initial_state`](@ref) per quadrature point, as a nested `[cell][qp]` vector |
| `mfh_element!(Ke, re, cv, mat, ue, states, states_old, Δt; cache)` | element stiffness and internal force, small strain, plane strain; [`material_element!`](@ref) generalizes it |
| `mfh_poro_element!(Ke, re, cvu, cvp, mat, ue, pe, …, Δt, mobility; u_range, p_range, cache)` | the coupled ``(\underline{u}, p)`` element — [the equations](@ref fe-poro-coupling) |
| `annulus_grid(Ri, Ro, nr, nθ)` | a structured annular sector, built by bending a rectangle — no gmsh |
| `cylinder_sector_grid(Ri, Ro, H, nr, nθ, nz; grading)` | its 3-D twin, with geometric radial layers for a well problem |

A worked mechanical model is the [thick-walled cylinder](@ref fe-thick-cylinder);
the coupled element drives the [ARMA 2011 well test](@ref fe-arma2011).

The two-field element takes the mobility ``\boldsymbol{K}/\mu`` as an
**argument**, one value per quadrature point: the permeability follows the
apertures through a self-consistent solve, so the driver evaluates it once per
step from the converged state and the scheme stays implicit in
``(\underline{u}, p)`` and explicit in ``\boldsymbol{K}``.

```julia
for c in eachindex(states), q in eachindex(states[c])
    K = transport_property(mat, states[c][q])
    mob[c][q] = K === nothing ? zero(SymmetricTensor{2,3}) : to_tensors(K) / μ
end
```

!!! warning "Two traps"
    A `DofHandler` numbers dofs **by cell traversal, not node order** — read
    nodal results with `evaluate_at_grid_nodes(dh, u, :u)`, never by indexing
    `u` with a node number. And a [`MaterialCache`](@ref) is an unlocked `Dict`:
    use one per thread, or none.

## Other codes

Only Ferrite has an adapter today. Another code needs the same small piece of
glue, written with the functions of the package: at each quadrature point,
[`mandel_strain`](@ref) turns the strain of the element into the strain of the
material, [`point_update`](@ref) (or [`material_response`](@ref) for any
material of the contract) returns the stress, the consistent tangent and the
new state, read from and written to a [`QuadratureStates`](@ref), and
[`volume_weight`](@ref) gives the factor of the volume element; the virtual
strain of a shape function is [`mandel_strain`](@ref) of its gradient. Wrapping
the assembly in a function `system!(u, t, Δt) -> (K, r)` then gives
[`quasistatic!`](@ref) for free. Outside Julia, [`voigt_strain`](@extref TensND TensND.voigt_strain) and
[`voigt_stress`](@extref TensND TensND.voigt_stress) carry `ε` and `σ` across the boundary.

## Where to go next

The two examples that follow use these helpers:
[Thick-walled cylinder](@ref fe-thick-cylinder) with the mechanical element,
and [A fractured-reservoir well test](@ref fe-arma2011) with the coupled one.
