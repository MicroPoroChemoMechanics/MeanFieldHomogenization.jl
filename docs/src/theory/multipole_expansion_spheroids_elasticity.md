# [The multipole expansion method for spheroids in elasticity](@id th-mem-spheroids-elasticity)

!!! info "Before this page"
    [The multipole expansion method in elasticity](@ref th-mem-elasticity),
    whose solutions, translation, linear system and stiffness are kept here, and
    [the method for spheroids](@ref th-mem-spheroids) in conduction, for the
    spheroidal harmonics.

The particles are now spheroids of revolution axis ``\underline{e}_3``, all
oriented alike, perfectly bonded or empty, in an isotropic matrix of shear
modulus ``\mu_0`` and Poisson ratio ``\nu_0`` [kushch2013](@cite). Each one keeps
the unknowns of a sphere, the coefficients of the regular spherical solutions
it receives about its center; what changes is its response, which goes
through solutions adapted to its shape, and the reading of the field of a close
neighbor.

## 1. Solutions of the Lamé equation about a spheroid

Let ``F_n^m`` be the decaying spheroidal harmonics of
[the method for spheroids](@ref th-mem-spheroids), for the focal distance ``c``.
Far from the spheroid, ``F_n^m \simeq (-1)^m \kappa_n\,Y_n^m`` with
``\kappa_n = c^{n+1}/(2n+1)!!``, the same for every order; the harmonics

```math
\tilde{F}_n^m = (-1)^m\,F_n^m/\kappa_n
```

therefore tend to the spherical ``Y_n^m``, and they keep the symmetry
``\tilde{F}_n^{-m} = (-1)^m\,\overline{\tilde{F}_n^m}`` of the ``Y_n^m``. The vector
harmonics ``\underline{Y}^{(2)}_{nm}`` and ``\underline{Y}^{(3)}_{nm}`` of
[the method in elasticity](@ref th-mem-elasticity) are combinations of the
``Y_k^l`` of one degree; written with the ``\tilde{F}_k^l`` in their place, they give
harmonic vectors ``\underline{\tilde{F}}^{(2)}_{nm}`` and ``\underline{\tilde{F}}^{(3)}_{nm}``
([kushch2013](@cite), Eq. (7.2)), the first of zero divergence. The decaying
solutions about the spheroid are

```math
\underline{V}^{(1)}_{nm} = \underline{\operatorname{grad}}\,\tilde{F}_n^m , \qquad
\underline{V}^{(2)}_{nm} = \underline{\tilde{F}}^{(2)}_{nm} , \qquad
\underline{V}^{(3)}_{nm} = 4(1-\nu_0)\,\underline{\tilde{F}}^{(3)}_{nm}
  - \underline{\operatorname{grad}}\,\big(\underline{x}\cdot\underline{\tilde{F}}^{(3)}_{nm}\big) :
```

a potential field and a harmonic field of zero divergence solve the Lamé
equation, and the third is the Papkovich–Neuber field of a harmonic vector.
For a sphere they are the solutions ``\underline{U}^{(i)}_{nm}``. For a spheroid
the gradient of ``\tilde{F}_n^m`` is not a finite combination of the vector
harmonics built on the ``\tilde{F}``; taking it as the first family places the
field of a spheroid under a uniform remote strain, Eshelby's, among the
solutions of degree 2 and below, as the term ``-c\,\xi_0^2\,\underline{\operatorname{grad}}\,f_t^s``
of [kushch2013](@citet), Eq. (7.16), does for the regular solutions. The
gradients of the ``\tilde{F}`` that ``\underline{V}^{(1)}`` and ``\underline{V}^{(3)}``
require are obtained by forward-mode differentiation of their evaluation.

## 2. Response of a spheroid

The spheroid receives the regular spherical solutions
``\underline{u}^{(j)}_{nm}`` about its center, with coefficients ``a``, sends the
``\underline{V}^{(i)}_{nm}`` with coefficients ``b``, and holds inside a series of
regular spherical solutions for its own Poisson ratio ``\nu_1``. On its surface,
of unit normal ``\underline{n}``, the displacement and the traction
``\boldsymbol{\sigma}\cdot\underline{n}`` are continuous; on a cavity the traction is
zero. The spheroid and its phases being unchanged by a rotation about
``\underline{e}_3``, a field of order ``m`` keeps its order, and is known everywhere
from its values on one meridian. For each order the two conditions are
imposed by least squares at the Gauss points of the meridian ``\varphi = 0``,
the traction brought to a displacement by the factor ``R/\mu_0``, ``R`` the
largest semi-axis; the solution gives ``b = \mathbf{Z}\,a``. With
``n_{\max} = 2`` it is exact under a uniform remote strain, which the example
checks against [`MoriTanaka`](@ref):

```@example mem_sph_el
using MeanFieldHomogenization, TensND, LinearAlgebra
iso(κ, μ) = TensISO{3}(3κ, 2μ)
stiffness(asm, scheme) = TensND.mandel(homogenize(asm, scheme, :C), TensND.CanonicalBasis{3, Float64}())
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(2.0, 2.0, 0.4)))
add_matrix!(asm, Dict(:C => iso(1.0, 0.6)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(1.0, 1.0, 0.2), Dict(:C => iso(10.0, 6.0)))
norm(stiffness(asm, MultipoleExpansion(; nmax = 2)) - stiffness(asm, MoriTanaka())) / norm(stiffness(asm, MoriTanaka()))
```

## 3. From one spheroid to another

Outside the sphere that holds its focal segment or disk, the field a spheroid
sends is a series of the decaying spherical solutions
``\underline{U}^{(i)}_{nm}``; its coefficients are fitted by least squares, order
by order, on a sphere of twice its largest semi-axis, up to the degree
``n_{\max} + 40``. The response is then written as the matrix that turns the
received spherical coefficients into the sent spherical ones, and the linear
system, the translation, the periodic sums and the stiffness of
[the method in elasticity](@ref th-mem-elasticity) apply unchanged. As in
conduction, the series of a sender converges about a receiver only if their
distance exceeds the sum of their focal distances; below 2.5 times that sum
the field of the sender is evaluated at points of the surface of the receiver
and projected there, by least squares, on the regular spherical solutions,
which holds at any distance between particles that do not overlap. On a
periodic cell the images within that distance are treated so and left out of
the lattice sums, and the decaying solutions of high degree are summed
directly, as in conduction.

## 4. Spheroids on an orthogonal lattice

[kushch2013](@citet), Table 7.6, gives ``C^{\mathrm{hom}}_{3333}/\mu_0`` of spheroids of
semi-axes ``(1, 1, \varepsilon)`` on a lattice of periods proportional to them,
with ``\nu_0 = \nu_1 = 0.3``. For ``\varepsilon = 0.5``, ``\mu_1/\mu_0 = 10`` and the
volume fraction 0.1 it is 4.021:

```@example mem_sph_el
κ(μ) = 2μ * 1.3 / 1.2
semi = (1.0, 1.0, 0.5)
cell = ParticleAssembly(; boundary = PeriodicBox(cbrt(4π / 3 / 0.1) .* semi))
add_matrix!(cell, Dict(:C => iso(κ(1.0), 1.0)))
add_particle!(cell, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:C => iso(κ(10.0), 10.0)))
round(stiffness(cell, MultipoleExpansion(; nmax = 6))[3, 3]; digits = 3)
```

The table is recomputed in the application
[Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids).
