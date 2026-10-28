# [Step by step](@id fe-step-by-step)

This page builds the [quickstart](@ref fe-quickstart) again, one piece at a
time, saying for each what it does and what goes wrong without it. Two
libraries share the work, and the boundary between them never moves:

| Ferrite does | MeanFieldHomogenization does |
|:--|:--|
| the mesh, the interpolation, the quadrature | the law at each quadrature point, and its consistent tangent |
| the numbering of the degrees of freedom | the element: residual and stiffness from the law ([`material_element!`](@ref)) |
| the boundary conditions, the assembly | the states of the points ([`QuadratureStates`](@ref)), the global Newton iteration ([`quasistatic!`](@ref)) |
| the VTK file format | what goes into it ([`write_material_vtk`](@ref)) |

## 1. The law

```@example fe_step
using MeanFieldHomogenization, Ferrite
a, b, σ₀ = 1.0, 2.0, 250.0
law = J2Plasticity(; E = 200e3, ν = 0.3, σ₀)
```

Any material of the Gauss-point contract fits here: a law of
[`MaterialLaws`](@ref ml-index), written by you or shipped, or a homogenized
material such as [`HomogenizedElastic`](@ref). The rest of the script does not
change.

## 2. The mesh, the interpolation, the quadrature

```@example fe_step
grid = generate_grid(QuadraticQuadrilateral, (8, 6), Vec(0.0, 0.0), Vec(1.0, 1.0))
transform_coordinates!(grid, x -> (r = a + (b - a) * x[1]; θ = π / 2 * x[2]; Vec(r * cos(θ), r * sin(θ))))
ip, ip_geo = Lagrange{RefQuadrilateral, 2}()^2, Lagrange{RefQuadrilateral, 2}()
cv = CellValues(QuadratureRule{RefQuadrilateral}(3), ip, ip_geo)
nothing # hide
```

A square of quadratic quadrilaterals is bent into a quarter of the tube; the
facet sets keep their names, `"left"` becoming the bore and `"right"` the outer
surface.

!!! warning "Quadratic elements for a plastic material"
    Plastic flow conserves the volume. Linear elements cannot represent a
    deformation that conserves the volume element by element and **lock**:
    they come out far too stiff, and a limit load computed with them is too
    high. Quadratic elements (or a mixed formulation) avoid it. The geometric
    interpolation `ip_geo` must then be quadratic too, or `CellValues` refuses
    the curved cells.

## 3. Degrees of freedom and boundary conditions

```@example fe_step
dh = close!(add!(DofHandler(grid), :u, ip))
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, [2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0, [1]))
close!(ch)
```

On the plane ``\theta = 0`` (`"bottom"`) the displacement normal to it, ``u_2``,
vanishes; on ``\theta = \pi/2`` (`"top"`) ``u_1`` does. Without them the quarter
would be free to slide and turn: the stiffness matrix is singular, and the
first Newton step fails.

## 4. The loads

```@example fe_step
fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(3), ip, ip_geo)
f_unit = zeros(ndofs(dh))
for facet in FacetIterator(dh, getfacetset(grid, "left"))
    reinit!(fv, facet)
    fe = zeros(ndofs_per_cell(dh))
    for q in 1:getnquadpoints(fv), i in eachindex(fe)
        fe[i] -= (shape_value(fv, q, i) ⋅ getnormal(fv, q)) * getdetJdV(fv, q)
    end
    assemble!(f_unit, celldofs(facet), fe)
end
nothing # hide
```

A pressure ``p`` pushes on the bore against its outward normal ``\underline{n}``:
the traction is ``-p\,\underline{n}``, and its work on each shape function gives
the nodal forces of a unit pressure, `f_unit`. They are computed once; the
pressure of each step only scales them.

## 5. The states of the quadrature points

```@example fe_step
states = QuadratureStates(law, getncells(grid), getnquadpoints(cv))
nothing # hide
```

Each quadrature point remembers its history (here the plastic strain and the
cumulated plastic strain), twice: `states.old`, at the last converged step,
which the law reads, and `states.new`, which the current iteration writes. A
converged step is committed ([`commit!`](@ref)); a rejected one is rolled back
([`rollback!`](@ref)) and nothing else is needed, since a law never modifies a
state it reads. Reading `states.new` as the start of the next step instead
would integrate every Newton iteration as if it were a step: the plastic strain
would pile up with each iteration.

## 6. The system and its Newton iteration

```@example fe_step
p_lim = 2σ₀ / sqrt(3) * log(b / a)
pressure(t) = 0.97 * p_lim * t
system!, prescribe! = material_system(dh, ch, cv, law, PlaneStrain(), states;
    external_forces! = (r, t) -> (r .-= pressure(t) .* f_unit))
u = zeros(ndofs(dh))
history = quasistatic!(system!, prescribe!, u, states, range(0, 1; length = 26))
nothing # hide
```

- [`material_system`](@ref) returns the two functions the driver needs:
  `system!(u, t, Δt)` assembles, element by element with
  [`material_element!`](@ref), the internal forces minus the external ones and
  the consistent tangent stiffness; `prescribe!(u, t)` sets the prescribed
  displacements.
- [`PlaneStrain`](@ref)`()` says how the two-dimensional strain of an element
  is a strain of the material; [`ThreeD`](@ref) and [`Axisymmetric`](@ref) are
  the other choices.
- [`quasistatic!`](@ref) runs Newton's method on the equilibrium at each time,
  commits the states of a converged step, and cuts a step that does not
  converge into halves.

`history` holds, for each step, the norms of the successive residuals. Here is
the last step, at 97 % of the limit load:

```@example fe_step
norms = history[end][2]
norms ./ norms[1]
```

The first iterations may grow: so close to the limit load, the extent of the
plastic zone is still being found. Once it is, the fall is quadratic, the
number of correct digits about doubling at each iteration, which only a
tangent consistent with the integration of the law gives. A tangent that is
not consistent (an elastic stiffness used in a plastic step, say) converges
linearly, one digit at a time, if it converges at all.

## 7. The results

```@example fe_step
nodal = evaluate_at_grid_nodes(dh, u, :u)
nodal[1][1], pressure(1.0)                       # bore displacement (m) and pressure (MPa)
```

!!! warning "Degrees of freedom are not nodes"
    Ferrite numbers the degrees of freedom by cell, not by node: `u[2k - 1]` is
    **not** the first displacement of node `k`. `evaluate_at_grid_nodes`
    returns the values at the nodes, in the order of the grid.

[`write_material_vtk`](@ref) writes the displacement and the cell averages of
the stress, the strain and the internal variables, by name, for ParaView:

```@example fe_step
dir = mktempdir()
write_material_vtk(joinpath(dir, "tube"), dh, u, law, states)
readdir(dir)
```

## Troubleshooting

| symptom | likely cause | way out |
|:--|:--|:--|
| [`GlobalNewtonError`](@ref) at the first step | a rigid-body motion left free | add the missing boundary conditions |
| [`GlobalNewtonError`](@ref) near the limit load | no equilibrium exists beyond it | load in displacement, or stop below the limit |
| slow, linear convergence | a tangent not consistent with the law | check the law with [`check_material_interface`](@ref) |
| a limit load too high | volumetric locking of linear elements | quadratic elements |
| [`LocalIntegrationError`](@ref) | a step too large for a law | more times, or let `max_cuts` cut the steps |
| results that do not change with the load | `states.old` never committed | let [`quasistatic!`](@ref) drive, or call [`commit!`](@ref) |

A [`MaterialCache`](@ref), which a homogenized material uses to avoid solving
the same microstructure twice, is not thread-safe: one per thread, or none.
