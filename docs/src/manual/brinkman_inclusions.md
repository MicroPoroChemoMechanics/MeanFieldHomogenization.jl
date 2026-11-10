# [Brinkman inclusions](@id man-brinkman)

!!! info "Before this page"
    [Permeability from Stokes flow](@ref th-brinkman) for the problem these
    inclusions solve, and [Custom inclusions](@ref man-custom-inclusions) for the
    interface through which they enter the schemes.

A Brinkman inclusion is a pattern of a porous medium (a grain, a pore, a fiber,
a slit) embedded in an unbounded medium in which the flow obeys Brinkman's
equation. Its concentration tensors depend on its size relative to the screening
length ``\sqrt{k_0}`` of the reference, which is why it carries a size and no Hill
tensor. This page shows how to declare one and how to feed it to the schemes.

## Shapes and media

Four shapes are available. Each is filled with a medium, among a porous medium
of the phase permeability ([`BrinkmanPorous`](@ref), the default), a clear fluid
([`BrinkmanFluid`](@ref)) or an impervious solid ([`BrinkmanSolid`](@ref)):

| Constructor | Pattern | Media |
| :--- | :--- | :--- |
| [`BrinkmanSphere`](@ref)`(R, medium)` | sphere of radius `R` | porous, fluid, solid |
| [`BrinkmanCylinder`](@ref)`(R, medium; euler_angles)` | infinite cylinder of radius `R` | porous, fluid, solid |
| [`BrinkmanLayer`](@ref)`(h, medium; euler_angles)` | planar layer of thickness `2h` | porous, fluid |
| [`BrinkmanCoatedSphere`](@ref)`(R, R_out)` | impervious core in a fluid shell | composite |

The axis of a cylinder and the normal of a layer are the third vector of the
basis given by the ZYZ `euler_angles` (or by `basis`). Only a porous inclusion
reads the phase property; the property declared for a solid, fluid or coated
phase is ignored, and any isotropic tensor will do.

```@example brinkman
using MeanFieldHomogenization, TensND
iso(x) = TensISO{3}(x)
val(t) = first(TensND.get_data(t))

grain = BrinkmanSphere(1.0, BrinkmanSolid())
pore = BrinkmanSphere(1.0, BrinkmanFluid())
fiber = BrinkmanCylinder(0.5, BrinkmanSolid(); euler_angles = (0.3, 0.8, 0.0))
nothing # hide
```

## Resistance or permeability

The phases carry either the resistance ``\boldsymbol{r} = \boldsymbol{k}^{-1}`` or the
permeability ``\boldsymbol{k}``, as chosen by the keyword `form`:

- [`ResistanceForm`](@ref) (default): the property is `:r`, the gradient-side
  concentration is the velocity concentration ``\boldsymbol{A}``, and the flux side is
  the drag density ``\langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle``. A clear-fluid matrix is
  ``\boldsymbol{r}_0 = \boldsymbol{0}``.
- [`PermeabilityForm`](@ref): the property is `:k`, the gradient side is the force
  concentration ``\boldsymbol{B}`` and the flux side the velocity
  ``\boldsymbol{k}\cdot\boldsymbol{B} = \boldsymbol{A}\cdot\boldsymbol{k}_0``. An impervious matrix is
  ``\boldsymbol{k}_0 = \boldsymbol{0}``.

Every inclusion of a cell must use the same form. The two forms give the same
dilute, Mori–Tanaka, self-consistent and differential estimates whenever both are
finite. Maxwell's scheme uses the Hill tensor of the distribution shape, which
is that of conduction only in the permeability form.

The concentration tensors are those of the
[custom-inclusion interface](@ref man-custom-inclusions):

```@example brinkman
r0 = iso(4.0)                                          # reference resistance
A = gradient_gradient_concentration(pore, iso(0.0), r0)   # velocity in a Stokes sphere
rA = flux_gradient_concentration(grain, iso(0.0), r0)     # drag on an impervious sphere
(val(A), val(rA))
```

The drag is Brinkman's, ``3(3 + 3\alpha_0 + \alpha_0^2)/(2R^2)`` with
``\alpha_0 = R\sqrt{r_0} = 2``, that is ``19.5``.

## In a cell

A bed of impervious spheres of porosity ``\varphi = 0.6``, with the pore space
represented by spheres of the grain radius:

```@example brinkman
φ = 0.6
rve = RVE()
add_phase!(rve, :p, pore, Dict(:r => iso(0.0)); fraction = :rest)
add_phase!(rve, :s, grain, Dict(:r => iso(0.0)); fraction = 1 - φ)

r_dil = val(homogenize(rve, Dilute(:p), :r))
r_mt = val(homogenize(rve, MoriTanaka(:p), :r))
r_dif = val(homogenize(rve, DifferentialScheme(:p), :r))
r_sc = val(homogenize(rve, SelfConsistent(algorithm = NewtonDefault(), init = iso(r_mt)), :r))
(dilute = 1 / r_dil, mori_tanaka = 1 / r_mt, differential = 1 / r_dif, self_consistent = 1 / r_sc)
```

These are the permeabilities in units of ``R^2``. The matrix-based schemes use the
pore phase as matrix, ``\boldsymbol{r}_m = \boldsymbol{0}``.

**The self-consistent seed.** A solid phase has an infinite resistance, so the
Voigt average of a granular cell is infinite, and `SelfConsistent(init = :voigt)`
asks for another start. Without `init` the iteration starts from the property
declared for the remainder phase, which is not read by a fluid or solid
inclusion. An explicit seed of the right magnitude, such as the Mori–Tanaka
estimate above, is the robust choice, with `NewtonDefault()` when the result
must converge tightly.

**Orientations.** Cylinders and layers return transversely isotropic tensors;
`symmetrize = :iso` averages them over all orientations. The reference must be
isotropic, so a self-consistent cell holding cylinders needs that average.

```@example brinkman
fib = RVE()
add_phase!(fib, :p, BrinkmanCylinder(0.5, BrinkmanFluid()), Dict(:r => iso(0.0)); fraction = 0.8, symmetrize = :iso)
add_phase!(fib, :s, BrinkmanCylinder(0.5, BrinkmanSolid()), Dict(:r => iso(0.0)); fraction = 0.2, symmetrize = :iso)
1 / val(homogenize(fib, SelfConsistent(algorithm = NewtonDefault(), init = iso(1.0)), :r))
```

## Problems without a solution

Two problems have no Brinkman solution, and the package refuses them:

- an unbounded inclusion (cylinder or layer) in a clear-fluid reference
  (``\boldsymbol{r}_0 = \boldsymbol{0}``): Stokes' paradox;
- an impervious layer, which would block the normal flow.

```@example brinkman
try
    gradient_gradient_concentration(fiber, iso(0.0), iso(0.0))
catch err
    println(sprint(showerror, err))
end
```

## Generic numbers

The inclusions accept `ForwardDiff.Dual` numbers in their sizes and in the
properties, so `derivative` reaches the radius of the grains through the
self-consistent fixed point. The pores keep their radius here, so the result is
not the ``R^{-2}`` scaling of a change of both sizes:

```@example brinkman
using ForwardDiff
d = derivative(rve, SelfConsistent(algorithm = NewtonDefault(), init = iso(r_sc)),
    geometry(:s, :radius); output = :r, indexer = val)
d / r_sc        # relative sensitivity of the resistance to the grain radius
```

The sphere and the layer accept `SymPy` and `Symbolics` numbers as well, and the
cylinder accepts `SymPy` (its Bessel functions stay symbolic). Since the dilute and
Mori–Tanaka schemes have no iteration, they return closed forms:

```@example brinkman
using SymPyPythonCall
@syms R::positive f::positive
sym = RVE(T = Sym)
add_phase!(sym, :p, BrinkmanSphere(R, BrinkmanFluid()), Dict(:r => iso(Sym(0))); fraction = :rest)
add_phase!(sym, :s, BrinkmanSphere(R, BrinkmanSolid()), Dict(:r => iso(Sym(0))); fraction = f)
simplify(val(homogenize(sym, MoriTanaka(:p), :r)))
```

## Where to go next

[Stokes-to-Darcy permeability of porous media](@ref app-stokes-darcy) applies these inclusions
to granular, fibrous, tubular and double-porosity media, and compares the
schemes. The derivation of their concentration tensors is in
[Permeability from Stokes flow](@ref th-brinkman), and the API in
[Brinkman inclusions](@ref api-brinkman).
