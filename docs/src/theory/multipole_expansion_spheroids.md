# [The multipole expansion method for spheroids](@id th-mem-spheroids)

!!! info "Before this page"
    [The multipole expansion method](@ref th-mem) for spheres, whose structure
    is kept here, and [Layered spheroid](@ref th-layered-spheroid) for the
    confocal coordinates, the transfer matrices and the imperfect interfaces of
    a spheroid.

The particles of the assembly are now spheroids of revolution axis
``\underline{e}_3``, all oriented alike, possibly layered and with imperfect
interfaces, in an isotropic matrix of conductivity ``k_0``; they may differ in
size, aspect ratio and conductivity. The method is that of the spheres
[kushch2013](@cite): about each particle the field is a series of harmonics, the
series of one particle is rewritten about the others, and the coefficients
solve a linear system. Three things change: the harmonics are spheroidal, the
response of a particle couples the degrees, and the translation from one
particle to another cannot always go through the spherical harmonics. The
scheme is the same [`MultipoleExpansion`](@ref); an assembly of spheroids
selects this page, an assembly of spheres the previous one.

## 1. Spheroidal solid harmonics

Each particle carries the confocal coordinates ``(p, q, \varphi)`` of
[Layered spheroid](@ref th-layered-spheroid) about its center, with the half
focal distance ``c`` of its outer surface:

```math
x_1 + \mathrm{i}\,x_2 = c\,\sqrt{1-p^2}\,\sqrt{q^2-1}\;\mathrm{e}^{\mathrm{i}\varphi} ,
\qquad x_3 = c\,p\,q ,
```

with ``c`` real and ``q > 1`` for a prolate spheroid, and ``c = -\mathrm{i}\,c_0``,
``q = \mathrm{i}\,\tau`` for an oblate one of focal radius ``c_0``. The outer surface is
``q = q_0``. The surface harmonic is that of the spheres,
``\chi_n^m(p, \varphi) = (-1)^m\,P_n^m(p)\,\mathrm{e}^{\mathrm{i}m\varphi}``,
and the regular and decaying spheroidal solid harmonics are
[kushch2013](@cite)

```math
\hat{y}_n^m = \frac{(n-m)!}{(n+m)!}\,P_n^m(q)\,\chi_n^m(p, \varphi) , \qquad
\hat{Y}_n^m = \frac{(n-m)!}{(n+m)!}\,Q_n^m(q)\,\chi_n^m(p, \varphi) ,
```

the radial factor of a negative order being that of ``|m|``. ``P_n^m`` and
``Q_n^m`` are the associated Legendre functions of the first and second kind
of [Layered spheroid](@ref th-layered-spheroid), Ferrers' on ``[-1, 1]`` and
Hobson's off the cut, now needed at every order. ``P_n^m(q)`` grows upward in
``n`` from ``P_m^m(q) = (2m-1)!!\,(q^2-1)^{m/2}``; ``Q_n^m(q)`` is built at the
order 0 and raised in ``m`` by the exact relation

```math
\sqrt{q^2-1}\;Q_n^{m+1}(q) = (n-m)\,q\,Q_n^m(q) - (n+m)\,Q_{n-1}^m(q) ,
```

started from the closed form of ``Q_0^m``, the ``m``-th derivative of
``\operatorname{arccoth} q`` times ``(q^2-1)^{m/2}``.

The two families are tied to the spherical harmonics of the
[previous page](@ref th-mem) about the same center by
[kushch2013](@cite) (Eqs. (6.21)–(6.26)):

```math
y_n^m = \sum_{k} K^{(2)}_{nk}(c)\;\hat{y}_k^m , \qquad
\hat{Y}_n^m = (-1)^m \sum_{k \ge n} K^{(3)}_{nk}(c)\;Y_k^m \quad (r > |c|) ,
```

the sums running over the degrees ``k`` of the parity of ``n``, with

```math
K^{(2)}_{nk}(c) = \Big(\frac{c}{2}\Big)^n \frac{\sqrt{\pi}\,(k + \tfrac12)}{\Gamma\big(\tfrac{n-k}{2}+1\big)\,\Gamma\big(\tfrac{n+k}{2}+\tfrac32\big)} , \qquad
K^{(3)}_{nk}(c) = \Big(\frac{c}{2}\Big)^{k+1} \frac{\sqrt{\pi}}{\Gamma\big(\tfrac{k-n}{2}+1\big)\,\Gamma\big(\tfrac{k+n}{2}+\tfrac32\big)} .
```

The first relation is a finite identity between polynomials; the second
converges outside the sphere of radius ``|c|`` that holds the focal segment, or
the focal disk, of the spheroid. In particular ``y_1^m = c\,\hat{y}_1^m``: a
uniform gradient is a combination of the ``\hat{y}_1^m``.

## 2. Response of a spheroid

A spheroid placed in the regular field ``\sum a_n^m\,\hat{y}_n^m`` sends the
decaying field ``\sum b_n^m\,\hat{Y}_n^m``. The transfer matrices of
[Layered spheroid](@ref th-layered-spheroid), written there at the orders 0
and 1 and the odd degrees, hold at any order ``m`` and any set of degrees; the
product ``S`` of the matrices of the layers maps the regular amplitudes of the
core to the regular and decaying amplitudes in the matrix, and the response at
the order ``m`` is the matrix

```math
b^m = \mathbf{Z}^m\,a^m , \qquad \mathbf{Z}^m = S_{21}\,S_{11}^{-1} ,
```

up to the normalization ``(n-m)!/(n+m)!`` of the harmonics. A perfect interface
gives a diagonal ``\mathbf{Z}^m``. A Kapitza or a surface-conductive interface on
``q = q_0`` couples the degrees of the same parity through the integrals of the
interface, which at the order ``m`` read, with ``P_i = P_i^m``,

```math
\int_{-1}^{1}\frac{P_i\,P_j}{\sqrt{q_0^2-x^2}}\,\mathrm{d}x
\quad\text{(Kapitza)} , \qquad
\int_{-1}^{1}\frac{(1-x^2)\,P_i'\,P_j'}{\sqrt{q_0^2-x^2}}\,\mathrm{d}x
 + \frac{m^2}{q_0^2-1}\int_{-1}^{1}\frac{\sqrt{q_0^2-x^2}\,P_i\,P_j}{1-x^2}\,\mathrm{d}x
\quad\text{(surface conduction)} .
```

The second follows from the weak form of the surface conduction: on
``q = q_0`` the area element is ``c^2\sqrt{q_0^2-1}\sqrt{q_0^2-p^2}\,\mathrm{d}p\,\mathrm{d}\varphi``
and the surface gradient has the components
``\sqrt{1-p^2}\,\partial_p T/(c\sqrt{q_0^2-p^2})`` and
``\partial_\varphi T/(c\sqrt{q_0^2-1}\sqrt{1-p^2})``, whose products with those of a
harmonic of the same order give the two integrals, the second one carrying
``m^2`` from the two derivatives in ``\varphi``. At the orders 0 and 1 they are the
matrices ``J`` and ``K + L/(q^2-1)`` of
[Layered spheroid](@ref th-layered-spheroid).
[`harmonic_response`](@ref) returns ``\mathbf{Z}^m``:

```@example mem_sph
using MeanFieldHomogenization, TensND, LinearAlgebra
K₀ = TensISO{3}(1.0)
oblate = LayeredSpheroid((0.2,), (0.5,), (TensISO{3}(50.0),); interfaces = (KapitzaInterface(0.05),))
Z = harmonic_response(oblate, K₀, 1, 5)          # the order 1, degrees 1 to 5
round.(abs.(Z); digits = 3)
```

The degrees 1, 3, 5 couple, and the even ones apart.

## 3. From one spheroid to another

The decaying harmonics of particle ``J``, centered at ``-\underline{R}`` from
particle ``I``, are a regular field about ``I``:

```math
\hat{Y}_t^s(\underline{x} - \underline{X}_J) = \sum_{k,l} \eta_{tksl}(\underline{R})\;\hat{y}_k^l(\underline{x} - \underline{X}_I) ,
\qquad \underline{R} = \underline{X}_I - \underline{X}_J .
```

**Far particles.** Rewriting ``\hat{Y}_t^s`` on the spherical harmonics about
``J``, translating these to ``I`` and rewriting the result on the
``\hat{y}_k^l`` gives, with ``c_J``, ``c_I`` the half focal distances of the two
particles,

```math
\eta_{tksl} = (-1)^{s+k+l}\,\pi\,\big(k+\tfrac12\big)\Big(\frac{c_J}{2}\Big)^{t+1}\Big(\frac{c_I}{2}\Big)^{k}
\sum_{r \ge 0} Y_{t+k+2r}^{s-l}(\underline{R})
\sum_{j=0}^{r}\frac{(c_J/2)^{2(r-j)}\,(c_I/2)^{2j}}{(r-j)!\,\Gamma\big(t+r-j+\tfrac32\big)\,j!\,\Gamma\big(k+j+\tfrac32\big)} .
```

The series in ``r`` behaves as ``\big((|c_I|+|c_J|)/|\underline{R}|\big)^{2r}``: it
diverges when ``|\underline{R}| < |c_I| + |c_J|`` [kushch2013](@cite), although
the particles may be far from touching. Two oblate spheroids stacked along
their axis are the common case: their focal disks are wide and their centers
close.

**Near particles.** The coefficients can be read on the surface ``q = q_0`` of
particle ``I``, where ``\hat{y}_k^l = \frac{(k-l)!}{(k+l)!}P_k^l(q_0)\,\chi_k^l`` and the
``\chi_k^l`` are orthogonal [kushch2013](@cite) (Eq. (6.58)):

```math
\eta_{tksl} = \frac{2k+1}{4\pi\,P_k^{|l|}(q_0)}\int_0^{2\pi}\!\!\int_{-1}^{1}
\hat{Y}_t^s(\underline{x} - \underline{X}_J)\;\overline{\chi_k^l(p, \varphi)}\;\mathrm{d}p\,\mathrm{d}\varphi .
```

The field of ``J`` is harmonic inside particle ``I``, so this is the
coefficient whatever the distance, as long as the particles do not overlap. The
integral is computed by Gauss–Legendre in ``p`` and the trapezoidal rule in
``\varphi``, with ``2n_{\max}+24`` and twice as many points.

The near form is used below ``|\underline{R}| = \tfrac52\,(|c_I|+|c_J|)``, the far
one above, where twenty terms of the series in ``r`` leave it below the
rounding. Moving the threshold between ``1.6`` and ``4`` times ``|c_I|+|c_J|`` leaves
the effective conductivity unchanged.

## 4. Periodic arrays

In a periodic assembly, held in a rectangular cell of periods
``(L_1, L_2, L_3)`` ([`PeriodicBox`](@ref)`((L₁, L₂, L₃))`), each particle stands for
the lattice of its images, the images of particle ``I`` included. The images of
``J`` within the threshold of particle ``I`` are translated one by one by the
near form; the others enter the far form through the sums of
``Y_N^M(\underline{R} + \underline{L})`` over them.

Forming these sums as the sums over the whole lattice minus the near images
fails: the near images contribute terms larger than the result by a factor that
grows as ``(r_{\mathrm{near}}/r_{\min})^N`` with the degree, ``r_{\min}`` being the
distance of the closest one, and the difference loses its digits as the degree
grows. The sums are formed in two ways instead. Up to
the degree 8, where those of the lowest degrees converge only conditionally,
Ewald's splitting of the [previous page](@ref th-mem) runs with the near images
excluded: their short-range terms are skipped and the long-range part they
carry, ``(1 - w_N)\,Y_N``, is taken off, the splitting parameter being lowered so
that this part stays small. Above the degree 8 the sums converge absolutely and
are formed directly over the images up to eight times the threshold. On the
lattice of section 6, both are accurate to about ``10^{-8}`` of the sums.

The background of the lattice adds, as for spheres, the uniform gradient
``-4\pi\,\underline{d}/(3V)`` per spherical dipole ``\underline{d}`` of the lattice
(section 5 below).

## 5. Effective conductivity

The average flux follows from the surface integrals of the
[previous page](@ref th-mem), which do not depend on the surface chosen in the
matrix: on a sphere enclosing the spheroid, the field it sends is a spherical
series, whose dipoles alone contribute. Since
``\hat{Y}_1^m = (-1)^m\,(c^2/3)\,Y_1^m + \dots``, the spherical dipoles of particle
``I`` are

```math
b_{I,1}^{\mathrm{sph},m} = (-1)^m\,\frac{c_I^2}{3}\,b_{I,1}^m ,
```

with ``b_{I,1}^m`` the first row of ``\mathbf{Z}^m`` applied to what the particle
receives, and the formulas of the spheres apply: ``k^{\mathrm{hom}}`` closed by
Maxwell's argument for a finite assembly,
``\boldsymbol{K}^{\mathrm{hom}} = \boldsymbol{K}_0 + \boldsymbol{\Sigma}`` for a periodic
one. One spheroid in a domain of its own shape is
[Mori–Tanaka](@ref th-scheme-mori-tanaka) at any degree:

```@example mem_sph
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(1.0, 1.0, 0.4)))
add_matrix!(asm, Dict(:K => K₀))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.2), Dict(:K => TensISO{3}(8.0)))
k(scheme) = round.(diag(canonical_components(homogenize(asm, scheme, :K))); digits = 10)
k(MultipoleExpansion(; nmax = 3)), k(MoriTanaka())
```

## 6. Oblate spheroids on a lattice

[kushch2015](@citet) tabulate oblate spheroids of aspect ratio ``1/3`` on the
proportional lattice of periods ``(1, 1, 1/3)``, conductivity ratio 1000. At
``c = 0.5`` the spheroids nearly touch, by a gap of ``0.005`` along their axis;
the authors report ``(k_{11}, k_{33})/k_0 = (4.561, 3.251)`` with the degree 1 and
``(6.117, 4.326)`` with the degree 3:

```@example mem_sph
l₁ = cbrt(3 * 0.5 / (4π))
lattice = ParticleAssembly(; boundary = PeriodicBox((1.0, 1.0, 1 / 3)))
add_matrix!(lattice, Dict(:K => K₀))
add_particle!(lattice, :p, (0.0, 0.0, 0.0), Ellipsoid(l₁, l₁, l₁ / 3), Dict(:K => TensISO{3}(1000.0)))
[round.(diag(canonical_components(homogenize(lattice, MultipoleExpansion(; nmax), :K)))[[1, 3]]; digits = 3)
    for nmax in (1, 3)]
```

The application [Spheroids and ellipsoids by the multipole expansion](@ref app-mem-spheroids)
compares the method with the other tables of the article, interfaces included.
The fields of each spheroid, and the spheroids in elasticity, are listed in the
[roadmap](@ref dev-roadmap).
