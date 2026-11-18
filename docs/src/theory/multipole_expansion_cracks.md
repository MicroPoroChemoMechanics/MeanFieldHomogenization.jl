# [The multipole expansion method for penny-shaped cracks](@id th-mem-cracks)

!!! info "Before this page"
    [The multipole expansion method for spheroids](@ref th-mem-spheroids) and
    [for spheroids in elasticity](@ref th-mem-spheroids-elasticity), whose
    harmonics, solutions, translations and linear systems are kept here.

A penny-shaped crack of radius ``a`` and normal ``\underline{e}_3`` enters the
expansion as a spheroid does, through its response to the harmonics it receives
and the field it sends [kushch2013](@cite). It is not taken as the limit of a
flat spheroid: it is the focal disk of the oblate coordinates of focal distance
``a``, on which the spheroidal harmonics are finite, and its two lips are told
apart by those coordinates.

## 1. The crack as a focal disk

In the oblate coordinates ``(\varphi, p, \tau)`` of the chart of focal distance
``a`` ([`OblateSpheroidalChart`](@extref TensND TensND.OblateSpheroidalChart)),

```math
x_1 + \mathrm{i}x_2 = a\sqrt{1-p^2}\,\sqrt{1+\tau^2}\;e^{\mathrm{i}\varphi} , \qquad
x_3 = a\,p\,\tau ,
```

the surfaces ``\tau = \tau_0 > 0`` are oblate spheroids and ``\tau = 0`` is the
disk ``x_3 = 0``, ``\rho \le a``, run over twice: its upper lip is ``p > 0``, its
lower lip ``p < 0``, and its edge ``p = 0``. The radial functions of the second
kind ``\mathcal{Q}_n^m(\tau)`` are finite at ``\tau = 0``
([`radial_Q`](@extref TensND TensND.radial_Q)), and so are the decaying
harmonics of the spheroid, which take on the two lips the values of
``P_n^m(\pm|p|)``: the harmonics of ``n - m`` even are continuous across the
disk, those of ``n - m`` odd jump, as the opening of a crack does.

On the disk ``\partial\tau/\partial x_3 = 1/(a p)`` and
``\partial p/\partial x_3 = 0``, from ``\mathrm{d}x_3 = a\,p\,\mathrm{d}\tau`` at
``\tau = 0``: the normal derivative of a field is its derivative along ``\tau``
divided by ``a p``.

## 2. An insulating crack

An insulating crack makes the normal flux vanish on both lips. With the
regular harmonics ``f_n^m`` and the decaying ones ``F_n^m`` of
[the method for spheroids](@ref th-mem-spheroids), whose radial factors are
``P_n^m`` and ``Q_n^m`` of ``\xi = \mathrm{i}\tau``, the field
``\sum a_n^m f_n^m + b_n^m F_n^m`` has a zero derivative along ``\tau`` at
``\tau = 0`` if and only if, harmonic by harmonic,

```math
b_n^m = -\,\frac{P_n^{m\prime}(\mathrm{i}0)}{Q_n^{m\prime}(\mathrm{i}0)}\;a_n^m ,
```

the response of a cavity taken on the coordinate surface itself
[kushch2013; Eq. (6.50)](@cite). At ``\xi = \mathrm{i}0``, ``P_n^{m\prime}``
vanishes for ``n - m`` even: the crack answers only the harmonics odd in
``x_3``, those whose normal derivative does not vanish on its plane, and a
uniform gradient along its plane is left undisturbed.

The values of a regular field on the disk carry nothing of its odd part, which
vanishes there. A close neighbor is therefore read through the normal
derivative on the upper lip, where

```math
a\,p\,\partial_3\Big(\sum_{k,l} a_k^l f_k^l\Big) = \mathrm{i}\sum_{k,l} a_k^l\,\bar P_k^{l\prime}(\mathrm{i}0)\,\chi_k^l(p, \varphi) ,
```

``\chi_k^l`` the surface harmonics; only the terms of ``k - |l|`` odd remain, and
the functions of one parity being orthogonal on ``p \in (0, 1)`` with half their
norm on ``(-1, 1)``,

```math
a_k^l = \frac{2(2k+1)}{4\pi\,\mathrm{i}\,P_k^{|l|\prime}(\mathrm{i}0)}
  \int_0^1\!\!\int_0^{2\pi} a\,p\,\partial_3 u\;\overline{\chi_k^l}\,\mathrm{d}\varphi\,\mathrm{d}p
```

for ``k - |l|`` odd; the even coefficients, which the crack does not answer,
are left at zero. A single crack in a ball closed by Maxwell's scheme gives
the contribution ``-\tfrac{8}{3}\,k_0\,a^3/V\;\underline{e}_3\otimes\underline{e}_3``
of the dilute crack, and the expansion recovers Maxwell's estimate at any
degree:

```@example mem_cracks
using MeanFieldHomogenization, TensND, LinearAlgebra
a, R = 0.7, 2.0
asm = ParticleAssembly(; boundary = MixedBC(R))
add_matrix!(asm, Dict(:K => TensISO{3}(1.0)))
add_particle!(asm, :c, (0.0, 0.0, 0.0), EllipticCrack(a, a), Dict{Symbol, Any}())
σ = -8a^3 / (3 * (4π * R^3 / 3))
canonical_components(homogenize(asm, MultipoleExpansion(; nmax = 3), :K))[3, 3] - (1 + σ / (1 - σ / 3))
```

## 3. A traction-free crack

In elasticity the crack sends the solutions ``\underline{V}^{(i)}_{nm}`` of
[the method for spheroids in elasticity](@ref th-mem-spheroids-elasticity),
built on the decaying harmonics of focal distance ``a``, and its lips are free
of traction, ``\boldsymbol{\sigma}\cdot\underline{e}_3 = \underline{0}`` at
``p > 0`` and ``p < 0``. A point of a lip has the same Cartesian coordinates as
the facing point of the other lip; the solutions are therefore evaluated as
functions of the oblate coordinates, smooth on the disk, and their Cartesian
gradients follow from

```math
\underline{\operatorname{grad}}\,f = \Big(\frac{\partial\underline{x}}{\partial\underline{\xi}}\Big)^{-\mathsf{T}}\,\frac{\partial f}{\partial\underline{\xi}} ,
```

the Jacobian of the chart being regular on the lips off the axis and the edge.
The response is the least-squares solution of the traction-free condition at
the Gauss points in ``p`` of one meridian, both lips together.

The crack reads a close neighbor through the traction on its plane. That
reading has a null space: the rigid motions, the uniform stresses in the plane
of the crack and other regular solutions carry no traction there. It is
immaterial: the response of the crack is a function of the traction alone,
``\hat{b} = \mathbf{G}\,\mathbf{T}\,\hat{a}``, so any received coefficients
``\hat{a}`` with the traction read give the same sent field. The coefficients
are solved for on a basis of the solutions that carry a traction, chosen by a
QR factorization with column pivoting. One crack in a ball is again Maxwell's
estimate, against the cracked medium of the package:

```@example mem_cracks
C₀ = TensISO{3}(3 * 5 / 3, 2 * 1.0)
asm = ParticleAssembly(; boundary = MixedBC(R))
add_matrix!(asm, Dict(:C => C₀))
add_particle!(asm, :c, (0.0, 0.0, 0.0), EllipticCrack(a, a), Dict{Symbol, Any}())
rve = RVE(; distribution_shape = Ellipsoid(1.0))
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C₀); density = a^3 / (4π * R^3 / 3))
km(C) = mandel(C, CanonicalBasis{3, Float64}())
norm(km(homogenize(asm, MultipoleExpansion(; nmax = 2), :C)) - km(homogenize(rve, Maxwell()))) / norm(km(C₀))
```

## 4. Arrays of cracks

[kushch2013](@citet), Tables 6.3, 6.4 and 7.7 to 7.10, gives the conductivity
and the moduli of solids with arrays of cracks of unit radius on the lattice of
periods ``(a_1, a_1, a_3)``. The overlap check of an assembly treats cracks as
disks, so that parallel cracks in distinct planes may be stacked as closely as
wished. For ``a_1 = 3`` and ``a_3 = a_1/2`` the table gives
``\lambda^*_{33}/\lambda_0 = 0.830``:

```@example mem_cracks
cell = ParticleAssembly(; boundary = PeriodicBox((3.0, 3.0, 1.5)))
add_matrix!(cell, Dict(:K => TensISO{3}(1.0)))
add_particle!(cell, :c, (0.0, 0.0, 0.0), EllipticCrack(1.0, 1.0), Dict{Symbol, Any}())
round(canonical_components(homogenize(cell, MultipoleExpansion(; nmax = 8), :K))[3, 3]; digits = 3)
```

The tables are recomputed in the application
[Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids).
