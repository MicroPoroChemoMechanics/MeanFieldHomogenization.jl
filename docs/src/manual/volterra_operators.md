# [Aging viscoelastic operators and histories](@id man-volterra-operators)

!!! info "Before this page"
    [Viscoelastic homogenization](@ref man-viscoelasticity), for the laws
    ([`ViscoLaw`](@ref)) and the call to [`homogenize_alv`](@ref).

In aging linear viscoelasticity the stress at time ``t`` depends on the whole
strain history, through a **relaxation operator**:

```math
\boldsymbol{\sigma}(t) = \int_0^{t} \mathbb{R}(t, \tau) : \mathrm{d}\boldsymbol{\varepsilon}(\tau).
```

``\mathbb{R}(t, t')`` is the stress at ``t`` under a unit strain held from
``t'``. *Aging* means that it depends on ``t'``, the age at loading, and not
only on ``t - t'``. A creep compliance ``\mathbb{J}(t, t')``, an aging
conductivity, a permittivity are operators of the same kind.

This page is about the two objects such a computation handles, and the two
faces each of them has:

| object | as a function | on a time grid ``t_1 < \dots < t_n`` |
|:--|:--|:--|
| an operator, [`VolterraOperator`](@ref) | of two times, ``\mathbb{R}(t, t')`` | a matrix acting on the values of a history |
| a history, [`TimeHistory`](@ref) | of one time, ``\boldsymbol{\varepsilon}(t)`` | its values ``\boldsymbol{\varepsilon}(t_i)`` |

Both faces are accepted wherever one is expected, and the package goes from
one to the other.

```@setup vo
using MeanFieldHomogenization, TensND, LinearAlgebra
```

## The time grid

Every operator and every history of one computation lives on the same
instants. A [`TimeGrid`](@ref) holds them; it is built once, and every object
built on it carries it:

```@example vo
grid = TimeGrid([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
```

[`time_grid`](@ref) returns the grid of an operator, a history, or the result of
a computation on them. Two objects on different grids are refused: nothing is
interpolated behind your back.

## Building an operator

From a **law**, any function of ``(t, t')``. Here an aging Maxwell material whose
bulk and shear moduli relax, the stiffness of a phase being ``3k\,\mathbb{J} +
2\mu\,\mathbb{K}`` (`TensISO{3}(3k, 2μ)`):

```@example vo
age(t′) = 1 - 0.5 * exp(-t′)          # the material stiffens with its age at loading
R = VolterraOperator(grid) do t, t′
    age(t′) * TensISO{3}(3 * 10.0 * exp(-(t - t′) / 3), 2 * 6.0 * exp(-(t - t′)))
end
```

From a **constant tensor** (an elastic phase, whose relaxation does not depend
on time):

```@example vo
E = VolterraOperator(TensISO{3}(3 * 30.0, 2 * 20.0), grid)
nothing # hide
```

From the **matrix** of an operator already discretized on the grid
(``6n\times 6n`` at order 4, with ``6\times 6`` Kelvin-Mandel blocks), as one
scale hands it to the next or as another code writes it. `like = C` stores it in
the symmetry class and frame of the tensor `C`, after checking that it belongs
to it:

```@example vo
M = Matrix(R)                                  # 36 × 36
R2 = VolterraOperator(M, grid; like = TensISO{3}(1.0, 1.0))
R2 ≈ R
```

## Computing as with elastic tensors

An operator computes as its elastic counterpart does, with the same symbols:

| elastic | aging viscoelastic |
|:--|:--|
| `C₁ ⊡ C₂` | composition ``\mathbb{R}_1 \circ \mathbb{R}_2`` (it does not commute: keep the order) |
| `inv(C)` | Volterra inverse |
| `C₁ + C₂`, `C₁ - C₂`, `c * C`, `C / c` | sum, difference, product by a number |
| `I` | the identity operator |
| `C ⊡ ε` | the response to a history |

At order 2 (conduction, diffusion, permittivity) `⋅` takes the place of `⊡`.

The **creep operator** of a material is the Volterra inverse of its relaxation
operator:

```@example vo
J = inv(R)
nothing # hide
```

The stress under a strain history, and the strain recovered from it:

```@example vo
ε = TimeHistory(grid) do t
    1.0e-3 * TensND.Tens(Symmetric([1.0 0 0; 0 -0.3 0; 0 0 -0.3]))   # uniaxial, held from t = 0
end
σ = R ⊡ ε
inv(R) ⊡ σ ≈ ε
```

`σ[i]` is the stress at ``t_i``, as its six Kelvin-Mandel components. A
dilute estimate written as on paper, ``\mathbb{R} + f\,(\mathbb{E} -
\mathbb{R}) \circ (\mathbb{I} + \mathbb{P} \circ (\mathbb{E} - \mathbb{R}))^{-1}``,
is one line of Julia; [`homogenize_alv`](@ref) does this, and the rest, for you.

!!! note "An operator is its matrix"
    A [`VolterraOperator`](@ref) is an `AbstractMatrix`: the ``6n\times 6n``
    matrix of the discretized integral, stored compactly. Its matrix product is the
    composition, its matrix inverse the Volterra inverse, so it goes wherever such a
    matrix goes. `Matrix(R)` writes it out in full.

## Symmetry classes: what is stored

When the values ``\mathbb{R}(t, t')`` are isotropic, transversely isotropic
about one axis, orthotropic or cubic in one frame, each of them is a few small
**blocks** (TensND `symmetry_blocks`): two numbers for an isotropic tensor, a
``2\times 2`` matrix and two numbers for a transversely isotropic one. The
operator then stores, for each block, one matrix over the instants, and computes
on those, block by block:

| class | blocks stored | storage, relative to ``6n\times 6n`` |
|:--|:--|:--|
| isotropic | ``(\alpha)``, ``(\beta)`` | 1/18 |
| cubic | three scalars | 1/12 |
| transversely isotropic, any axis | a ``2\times 2``, two scalars | 1/6 |
| orthotropic, any frame | a ``3\times 3``, three scalars | 1/3 |
| none (dense) | the ``6\times 6`` | 1 |

An isotropic product costs two ``n\times n`` products where the dense one costs
a ``6n\times 6n`` one: about a hundred times fewer operations.

```@example vo
R
```

The class is found from the values of the law, and an isotropic operator
combined with another joins its class. [`operator_value`](@ref)`(R, i, j)` is
the value at the pair of instants ``(t_i, t_j)`` as a tensor of the class;
[`convert_class`](@ref) stores an operator in another class or dense
(`DenseClass{4}()`).

!!! warning "A value of the operator is not a value of the kernel"
    The matrix holds the weights of the trapezoidal rule, which mix the
    kernel at neighboring instants. The kernel ``\mathbb{R}(t_i, t_j)`` is given
    back by [`operator_law`](@ref)`(R)`, a law of ``(t, t')`` known on the grid.

## Homogenization

A phase property in [`homogenize_alv`](@ref) may be a law, a
[`VolterraOperator`](@ref) on the grid, or a [`Homogenized`](@ref) cell, which
is then homogenized on the same grid, as in elasticity. The result is a
[`VolterraOperator`](@ref), in the cheapest class that holds it:

```@example vo
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => R); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => E); fraction = 0.3)
Rhom = homogenize_alv(rve, MoriTanaka(:M), :C)
```

No `times` here: the grid is read on the operators of the phases. With laws
only, give it (`times = grid`).

The result is a phase of the next scale as it is, on the same grid; or, without
computing it first, the cell itself:

```@example vo
concrete = RVE()
add_phase!(concrete, :paste, Ellipsoid(1.0), Dict(:C => Homogenized(rve, MoriTanaka(:M))); fraction = :rest)
add_phase!(concrete, :sand, Ellipsoid(1.0), Dict(:C => VolterraOperator(TensISO{3}(3 * 40.0, 2 * 30.0), grid)); fraction = 0.6)
homogenize_alv(concrete, MoriTanaka(:paste), :C)
```

## Order 2

Conduction, diffusion and permittivity work the same way, with second-order
tensors and vector histories (a temperature gradient):

```@example vo
K = VolterraOperator(grid) do t, t′
    TensISO{3}(2.0 + exp(-(t - t′)))
end
g = TimeHistory(t -> [1.0, 0.0, 0.0], grid)
q = K ⋅ g
q[end]
```
