# [Toroidal inclusions](@id man-toroidal-inclusions)

!!! info "Before this page"
    [Toroidal inclusions](@ref th-toroidal-inclusions), where the tensor
    returned below is derived, and [Custom inclusions](@ref man-custom-inclusions)
    for the interface through which it enters the schemes.

A toroidal pore is a ring-shaped cavity: the torus swept by a circle of radius
``r`` whose center describes a circle of radius ``R>r``. [`InsulatingTorus`](@ref)
is this pore in the conduction problem, in an isotropic matrix. Its field is
not uniform and it has no Hill tensor; its concentration tensor is the sum of a
series, exact to rounding.

```@example mantorus
using MeanFieldHomogenization
using TensND
using LinearAlgebra
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))  # hide
nothing # hide
```

## Declaring a torus

The two radii, then the axis, which is the third vector of the basis given by
the ZYZ `euler_angles` (or by `basis`):

```@example mantorus
torus = InsulatingTorus(1.0, 0.35)
tilted = InsulatingTorus(1.0, 0.35; euler_angles = (0.0, π / 3, 0.0))
```

```@example mantorus
plotly_scene(shape_traces(tilted); uid = "man-torus", height = 420, title = "InsulatingTorus(1.0, 0.35), its axis <u><i>n</i></u> tilted by π/3 from <u><i>e</i></u><sub>3</sub>") # hide
```

## Its concentration tensor

The gradient concentration tensor is the mean temperature gradient over the
pore per unit remote gradient. It is transversely isotropic about the axis and
depends on ``r/R`` only:

```@example mantorus
K₀ = TensISO{3}(2.0)                       # conductivity of the matrix
A = gradient_gradient_concentration(torus, K₀, K₀)
canonical_components(A)
```

The property of the phase, the second argument, is not read: no flux passes
through the torus, and its flux concentration is zero. The contributions follow
from these two tensors, ``\boldsymbol{N}=-\boldsymbol{K}_0\cdot\boldsymbol{A}``
and ``\boldsymbol{R}=\boldsymbol{A}\cdot\boldsymbol{K}_0^{-1}`` per unit volume
of the torus:

```@example mantorus
canonical_components(inverse_property_contribution(torus, K₀, K₀))
```

A thin ring has the components ``3/2`` across the axis and ``2`` along it:

```@example mantorus
diag(canonical_components(gradient_gradient_concentration(InsulatingTorus(1.0, 1.0e-3), K₀, K₀)))
```

## In a scheme

The torus is a phase like any other, measured by its volume fraction. Aligned
pores make the material transversely isotropic:

```@example mantorus
function porous(pore; kw...)
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
    add_phase!(rve, :pores, pore, Dict(:K => K₀); fraction = 0.1, kw...)
    return rve
end
diag(canonical_components(homogenize(porous(torus), MoriTanaka(), :K)))
```

Randomly oriented pores give an isotropic material:

```@example mantorus
first(TensND.get_data(homogenize(porous(torus; symmetrize = IsoSymmetrize()), MoriTanaka(), :K)))
```

The reference medium must be isotropic. A self-consistent or a differential
scheme evaluates the pore in its current estimate, which is anisotropic when
the pores are aligned: these schemes need `symmetrize = IsoSymmetrize()` on the
phase.

## Derivatives

The radii are plain numbers of the inclusion, reached by `ForwardDiff` through
the series:

```@example mantorus
using ForwardDiff
axial(r) = canonical_components(gradient_gradient_concentration(InsulatingTorus(1.0, r), K₀, K₀))[3, 3]
ForwardDiff.derivative(axial, 0.35)
```

## What is not available

- an anisotropic reference medium, refused with a message;
- a torus that conducts, or whose cross-section is not a circle: a shape given
  as a region is treated on a grid of Gaussians
  ([Heterogeneities of any shape](@ref man-gaussian-inclusions)), within a few
  percent;
- the elastic problem.

## Where to go next

- [Toroidal inclusions](@ref th-toroidal-inclusions): the derivation.
- [Ring-shaped pores in a conducting matrix](@ref app-insulating-torus): the
  tensor against the ratio of the radii, published values, and the
  conductivity of a matrix with such pores.
- [Choosing an inclusion](@ref man-choosing-inclusion): the torus among the
  other routes.
