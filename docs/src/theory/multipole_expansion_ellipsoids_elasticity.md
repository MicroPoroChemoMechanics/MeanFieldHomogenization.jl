# [The multipole expansion method for ellipsoids in elasticity](@id th-mem-ellipsoids-elasticity)

!!! info "Before this page"
    [The multipole expansion method for spheroids in elasticity](@ref th-mem-spheroids-elasticity),
    whose response by least squares, reading of close neighbors and spherical
    form are kept here, and [the method for ellipsoids](@ref th-mem-ellipsoids)
    for the Lamé functions and the ellipsoidal harmonics.

The particles are ellipsoids of three different semi-axes
``a_1 > a_2 > a_3``, aligned alike with the axes of the assembly, perfectly
bonded, bonded by springs, or empty, in an isotropic matrix of shear modulus
``\mu_0`` and Poisson ratio ``\nu_0`` [kushch2019a, kushch2019b](@cite). As for
spheroids, each one keeps the unknowns of a sphere; its own solutions are now
built from the decaying ellipsoidal harmonics ``\mathbb{F}_k``, degree 0
included.

## 1. Solutions of the Lamé equation about an ellipsoid

The displacement outside is written with the potentials of Papkovich and
Neuber [kushch2019a](@cite),

```math
\underline{u} = 4(1-\nu_0)\,\underline{B} - \underline{\operatorname{grad}}\,\big(\underline{x}\cdot\underline{B} + B_0\big) ,
\qquad \underline{B} = \sum_{j} B_j\,\underline{i}_j ,
```

the four potentials ``B_0, B_1, B_2, B_3`` being harmonic and decaying. Each
``\mathbb{F}_k`` gives the solutions ``B_j = \mathbb{F}_k``,
``\underline{u} = (3-4\nu_0)\,\mathbb{F}_k\,\underline{i}_j - x_j\,\underline{\operatorname{grad}}\,\mathbb{F}_k``,
and ``B_0 = \mathbb{F}_k``, ``\underline{u} = -\underline{\operatorname{grad}}\,\mathbb{F}_k``.
Three potentials suffice [kushch2019a](@cite); the fourth shortens the series.
It is kept at the degree 2 alone: there it brings the field of an ellipsoid
under a uniform remote strain, Eshelby's, among the solutions of degree 2 and
below, while at the other degrees it adds no accuracy and makes the four
families nearly dependent. Kept at every degree, the least squares of the next
section give stiffnesses that differ by ``10^{-6}`` between two computations in
different arithmetics (plain numbers and numbers carrying a derivative); kept
at the degree 2, by ``10^{-13}``.

## 2. Response of an ellipsoid

The ellipsoid receives the regular spherical solutions about its center and
holds inside a series of regular spherical solutions for its Poisson ratio
``\nu_1``. On its surface, of unit normal ``\underline{n}``, the traction is
continuous and the displacement jumps by the compliance of the interface,

```math
\underline{u}_0 - \underline{u}_1 = \mathbf{S}\cdot\big(\boldsymbol{\sigma}_1\cdot\underline{n}\big) , \qquad
\mathbf{S} = s_n\,\underline{n}\otimes\underline{n} + s_t\,\big(\boldsymbol{1} - \underline{n}\otimes\underline{n}\big) ,
```

``s_n = 1/D_n`` and ``s_t = 1/D_t`` the compliances of a
[`SpringInterface`](@ref) of stiffnesses ``D_n``, ``D_t``, zero for a perfect
bond; on a cavity the traction is zero. No symmetry keeps the order here: the
conditions are imposed by least squares at the quadrature points of the
surface of [the method for ellipsoids](@ref th-mem-ellipsoids). One ellipsoid in
a domain of its own shape is Mori–Tanaka from ``n_{\max} = 2``:

```@example mem_ell_el
using MeanFieldHomogenization, TensND, LinearAlgebra
iso(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)                # (3κ, 2μ)
stiffness(asm, scheme) = TensND.mandel(homogenize(asm, scheme, :C), TensND.CanonicalBasis{3, Float64}())
semi = (1.5, 1.0, 0.5)
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.5))...)))
add_matrix!(asm, Dict(:C => iso(1.0, 0.3)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:C => iso(100.0, 0.2)))
norm(stiffness(asm, MultipoleExpansion(; nmax = 2)) - stiffness(asm, MoriTanaka())) / norm(stiffness(asm, MoriTanaka()))
```

## 3. Spherical form

With ``\mathbb{F}_k = \sum \mu_{kl}\,Y_k^l`` outside the sphere of radius ``2a_1``,
each solution is a Papkovich–Neuber field of ``\underline{i}_j Y_k^l`` or of
``Y_k^l``. In the complex basis of
[the method in elasticity](@ref th-mem-elasticity),
``\underline{i}_1 = \underline{e}_1 + \underline{e}_2``,
``\underline{i}_2 = \mathrm{i}\,(\underline{e}_2 - \underline{e}_1)``,
``\underline{i}_3 = \underline{e}_3``, and ``\underline{e}_1 Y_k^{m-1}``, ``\underline{e}_2 Y_k^{m+1}``,
``\underline{e}_3 Y_k^m`` are combinations of the vector harmonics
``\underline{Y}^{(1)}_{k-1,m}``, ``\underline{Y}^{(2)}_{k,m}`` and
``\underline{Y}^{(3)}_{k+1,m}``, obtained by inverting their definitions. With
``\underline{x}\cdot\underline{Y}^{(1)}_{nm} = -(n+1)\,Y_n^m`` and
``\underline{x}\cdot\underline{Y}^{(2)}_{nm} = 0``, the Papkovich–Neuber fields of these
vector harmonics are decaying spherical solutions:

```math
\underline{Y}^{(1)}_{k-1,m} \mapsto \big(4(1-\nu_0) + k\big)\,\underline{U}^{(1)}_{k-1,m} , \qquad
\underline{Y}^{(2)}_{k,m} \mapsto 4(1-\nu_0)\,\underline{U}^{(2)}_{k,m} , \qquad
\underline{Y}^{(3)}_{k+1,m} \mapsto \underline{U}^{(3)}_{k+1,m} ,
```

and ``\underline{\operatorname{grad}}\,Y_k^l = \underline{U}^{(1)}_{k,l}``. The response then enters
the solver of the spheres as for spheroids, close neighbors being read on the
surface of the receiver. The problem is solved in the frame of the axes of the
ellipsoids ordered by decreasing semi-axis, and the stiffness turned back to
the frame of the assembly.

## 4. A spring interface

[kushch2019a](@citet), Table 5, gives the Maxwell estimate
``C^{\mathrm{hom}}_{ijkl}/\mu_0`` of such ellipsoids at the volume fraction 0.5, with
``\nu_0 = 0.3``, ``\nu_1 = 0.2``, ``\mu_1/\mu_0 = 100`` and ``D_n = D_t = D``. The text
defines ``D = D_n\,a_1/\mu_0``; the table is recovered with ``D_n = D\,\mu_0``. For
``D = 10`` it gives ``C^{\mathrm{hom}}_{1111}/\mu_0 = 7.8673``:

```@example mem_ell_el
spring = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.5))...)))
add_matrix!(spring, Dict(:C => iso(1.0, 0.3)))
add_particle!(spring, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., SpringInterface(10.0, 10.0)), Dict(:C => iso(100.0, 0.2)))
round(stiffness(spring, MultipoleExpansion(; nmax = 8))[1, 1]; digits = 4)
```

The whole table is recomputed in the application
[Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids).
[kushch2019b](@citet) extend the estimate to clusters of interacting
ellipsoids, which the expansion treats as any assembly.
