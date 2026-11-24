# [The multipole expansion method with anisotropic constituents](@id th-mem-anisotropic)

!!! info "Before this page"
    [The multipole expansion method for spheroids](@ref th-mem-spheroids) and
    [for ellipsoids](@ref th-mem-ellipsoids), whose harmonics, projection and
    translations are used here unchanged.

The matrix and the particles may now be anisotropic in conduction. An affine
change of variables makes an orthotropic matrix isotropic [kushch2017](@cite);
in the new variables the particles are still ellipsoids, of anisotropic
conductivity, and the response of such a particle is obtained by projecting
the interface conditions on its surface harmonics. Nothing else in the method
changes.

## 1. An orthotropic matrix made isotropic

Let the matrix conductivity be orthotropic in the axes of the assembly,
``\mathbf{K}_0 = \sum_i k_i\,\underline{e}_i \otimes \underline{e}_i``. Write it
``\mathbf{K}_0 = \bar{k}\,\mathbf{L}^2`` with

```math
\bar{k} = (k_1 k_2 k_3)^{1/3} , \qquad
\mathbf{L} = \sum_i \sqrt{k_i/\bar{k}}\;\underline{e}_i \otimes \underline{e}_i , \qquad \det\mathbf{L} = 1 ,
```

and take the new variables ``\underline{v}`` with ``\underline{x} = \mathbf{L}\,\underline{v}``.
Since ``\underline{\operatorname{grad}}_x T = \mathbf{L}^{-1}\,\underline{\operatorname{grad}}_v T``,

```math
\operatorname{div}_x\!\big(\mathbf{K}_0\,\underline{\operatorname{grad}}_x T\big)
  = \bar{k}\,\operatorname{div}_v\!\big(\mathbf{L}^{-1}\mathbf{L}^2\mathbf{L}^{-1}\,\underline{\operatorname{grad}}_v T\big)
  = \bar{k}\,\Delta_v T :
```

in the variables ``\underline{v}`` the matrix is isotropic of conductivity
``\bar{k}``. The energy density reads
``\underline{\operatorname{grad}}_x T\cdot\mathbf{K}\,\underline{\operatorname{grad}}_x T = \underline{\operatorname{grad}}_v T\cdot(\mathbf{L}^{-1}\mathbf{K}\,\mathbf{L}^{-1})\,\underline{\operatorname{grad}}_v T``,
so that a phase of conductivity ``\mathbf{K}`` becomes one of conductivity
``\mathbf{L}^{-1}\mathbf{K}\,\mathbf{L}^{-1}``. An ellipsoid of semi-axes ``a_i`` along
``\underline{e}_i``, ``\sum_i x_i^2/a_i^2 = 1``, becomes the ellipsoid of semi-axes
``a_i/L_i`` with ``L_i = \sqrt{k_i/\bar{k}}``, a center ``\underline{X}`` the center
``\mathbf{L}^{-1}\underline{X}``, and a cell of periods ``\ell_i`` the cell of periods
``\ell_i/L_i``: volumes, hence volume fractions, are kept.

The effective conductivity follows from the energy. A macroscopic gradient
``\underline{G}_v`` in the variables ``\underline{v}`` is the gradient
``\underline{G}_x = \mathbf{L}^{-1}\underline{G}_v`` in the variables ``\underline{x}``, and the
energies of the two problems are equal, the Jacobian of the map being one:
``\underline{G}_x\cdot\mathbf{K}^{\mathrm{hom}}\,\underline{G}_x = \underline{G}_v\cdot\mathbf{K}_v^{\mathrm{hom}}\,\underline{G}_v``
for all ``\underline{G}_v``, that is

```math
\mathbf{K}^{\mathrm{hom}} = \mathbf{L}\,\mathbf{K}_v^{\mathrm{hom}}\,\mathbf{L} .
```

A sphere in a transversely isotropic matrix is a spheroid in the variables
``\underline{v}``, a spheroid may become a sphere, and an ellipsoid stays one; the
solvers of the previous pages apply, provided the particles are aligned with
the axes of the matrix. [kushch2017](@citet) also treat particles and matrices
whose axes differ, through a rotation of each particle; the ellipsoids are
then no longer aligned, which the solver of this package does not take.

## 2. A particle of anisotropic conductivity

In an isotropic matrix of conductivity ``k_0``, a perfectly bonded particle has
the conductivity ``\mathbf{K}_1``, factored by Cholesky as
``\mathbf{K}_1 = \boldsymbol{\Lambda}\,\boldsymbol{\Lambda}^{\!T}``. For any harmonic
function ``g``, the field ``T(\underline{x}) = g(\underline{u})``, ``\underline{u} = \boldsymbol{\Lambda}^{-1}\underline{x}``,
satisfies the conduction equation in the particle: with
``\underline{\operatorname{grad}}_x T = \boldsymbol{\Lambda}^{-T}\underline{\operatorname{grad}}_u g``,

```math
\mathbf{K}_1\,\underline{\operatorname{grad}}_x T = \boldsymbol{\Lambda}\,\underline{\operatorname{grad}}_u g ,
\qquad
\operatorname{div}_x\!\big(\boldsymbol{\Lambda}\,\underline{\operatorname{grad}}_u g\big) = \Delta_u g = 0 .
```

The field inside is therefore taken as ``\sum_j d_j\,y_j(\boldsymbol{\Lambda}^{-1}\underline{x})``, the
``y_j`` being the regular solid harmonics of degrees 1 to ``n_{\max}`` of
[the method for spheres](@ref th-mem) (a constant adds nothing to the flux and
drops out). Outside, the field is ``\sum_k a_k f_k + \sum_k b_k F_k`` in the regular
and decaying harmonics ``f_k``, ``F_k`` of the particle, spheroidal or ellipsoidal.
On the surface ``\underline{x}\cdot\mathbf{B}\,\underline{x} = 1`` of the particle, with
``\mathbf{B} = \sum_i a_i^{-2}\,\underline{e}_i \otimes \underline{e}_i``, the outer normal is along
``\mathbf{B}\,\underline{x}``, and the continuity of temperature and of normal flux read

```math
\sum_k a_k f_k + \sum_k b_k F_k = \sum_j d_j\,y_j(\boldsymbol{\Lambda}^{-1}\underline{x}) , \qquad
k_0\,(\mathbf{B}\underline{x})\cdot\underline{\operatorname{grad}}\Big(\sum_k a_k f_k + \sum_k b_k F_k\Big)
  = \sum_j d_j\,(\mathbf{B}\underline{x})\cdot\boldsymbol{\Lambda}\,\underline{\operatorname{grad}}_u y_j .
```

Both are projected on the surface harmonics of the particle with the
projection ``\Pi`` of the spheroids and ellipsoids, which reads a surface field
as the coefficients of the regular harmonics having that trace
([kushch2017](@cite), Eqs. (3.7)-(3.8)). The gradient of a regular solid
harmonic is made of regular solid harmonics of degree one less, in the
normalization of [the method for spheres](@ref th-mem):

```math
\partial_1 y_n^m = \tfrac12\big(y_{n-1}^{m-1} - y_{n-1}^{m+1}\big) , \qquad
\partial_2 y_n^m = \tfrac{\mathrm{i}}{2}\big(y_{n-1}^{m-1} + y_{n-1}^{m+1}\big) , \qquad
\partial_3 y_n^m = y_{n-1}^m ;
```

the derivative of ``F_k`` along ``\mathbf{B}\underline{x}`` is obtained by forward-mode
differentiation of its evaluation at the surface points.

The response ``\mathbf{Z}``, ``\underline{b} = \mathbf{Z}\,\underline{a}``, follows by taking as
regular field each spherical harmonic ``y_j`` in turn: its coefficients on the
``f_k`` are ``\Pi\,y_j``, and the two projected conditions give the ``b_k``, hence

```math
\mathbf{Z} = \big[\,\underline{b}(y_1) \;\cdots\; \underline{b}(y_N)\,\big]\,\big[\,\Pi y_1 \;\cdots\; \Pi y_N\,\big]^{-1} .
```

``\mathbf{Z}`` is full: an anisotropic particle couples the degrees and the orders.
For ``\mathbf{K}_1 = k_1\boldsymbol{1}`` the exact response of the previous pages lies in the
spaces of the projection, and is recovered. One particle in a domain of its
own shape is Mori–Tanaka, here a rotated orthotropic spheroid:

```@example mem_aniso
using MeanFieldHomogenization, TensND, LinearAlgebra
Q = [cos(0.4) -sin(0.4) 0; sin(0.4) cos(0.4) 0; 0 0 1]
K₁ = Tens(Q * Diagonal([5.0, 2.0, 8.0]) * transpose(Q))
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(2.0, 2.0, 1.0)))
add_matrix!(asm, Dict(:K => TensISO{3}(1.0)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(1.0, 1.0, 0.5), Dict(:K => K₁))
mem = canonical_components(homogenize(asm, MultipoleExpansion(; nmax = 3), :K))
mt = canonical_components(homogenize(asm, MoriTanaka(), :K))
norm(mem - mt) / norm(mt)
```

## 3. Spheres of anisotropic conductivity

The solver of [the method for spheres](@ref th-mem) holds one scalar response
per degree. A sphere of anisotropic conductivity, or a spheroid that the map
of section 1 turns into a sphere, goes instead through the solver of shaped
particles of the spheroids, its harmonics being the solid ones: the
translation of the spheres holds at any distance between particles that do
not overlap, and replaces the projection on the receiving surface.

## 4. Spheroids in a transversely isotropic matrix

[kushch2014](@citet) tabulate the anisotropy parameter
``\varpi = (k^{\mathrm{hom}}_{33}/k_{33})/(k^{\mathrm{hom}}_{11}/k_{11})`` of
spheroids of semi-axes ``(1, 1, e)`` and conductivity ``10\,k_{11}`` in a matrix of
conductivities ``k_{11}`` across and ``k_{33}`` along ``\underline{e}_3``, on an
orthorhombic lattice of periods proportional to the semi-axes. At ``e = 2``,
``k_{33}/k_{11} = 0.5`` and the fraction 0.3 they give 1.376:

```@example mem_aniso
semi = (1.0, 1.0, 2.0)
cell = ParticleAssembly(; boundary = PeriodicBox(cbrt(4π / 3 / 0.3) .* semi))
add_matrix!(cell, Dict(:K => TensTI{2}(1.0, 0.5, (0.0, 0.0, 1.0))))
add_particle!(cell, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(10.0)))
k = diag(canonical_components(homogenize(cell, MultipoleExpansion(; nmax = 6), :K)))
round((k[3] / 0.5) / k[1]; digits = 3)
```

The whole table is recomputed in the application
[Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids).

## Where to go next

The last page on the method,
[The multipole expansion method for penny-shaped cracks](@ref th-mem-cracks),
takes the spheroid to its flat limit.
