# [The multipole expansion method for spheroids and ellipsoids](@id th-mem-shapes)

!!! info "Before this page"
    [The multipole expansion method](@ref th-mem) for spheres, whose structure
    is kept here, and [Layered spheroid](@ref th-layered-spheroid) for the
    confocal coordinates, the transfer matrices and the imperfect interfaces of
    a spheroid.

The method of the [spheres](@ref th-mem) carries over to particles of other
shapes: about each particle the field is expanded on harmonics adapted to its
shape. Part A treats spheroids in conduction and part B in elasticity; parts C
and D do the same for ellipsoids of three different semi-axes.

## [A. Spheroids in conduction](@id th-mem-spheroids)

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
selects this part, an assembly of spheres the [method for spheres](@ref th-mem).

### 1. Spheroidal solid harmonics

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
[method for spheres](@ref th-mem) about the same center by
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

### 2. Response of a spheroid

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

### 3. From one spheroid to another

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

### 4. Periodic arrays

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
Ewald's splitting of the [method for spheres](@ref th-mem) runs with the near images
excluded: their short-range terms are skipped and the long-range part they
carry, ``(1 - w_N)\,Y_N``, is taken off, the splitting parameter being lowered so
that this part stays small. Above the degree 8 the sums converge absolutely and
are formed directly over the images up to eight times the threshold. On the
lattice of section 6, both are accurate to about ``10^{-8}`` of the sums.

The background of the lattice adds, as for spheres, the uniform gradient
``-4\pi\,\underline{d}/(3V)`` per spherical dipole ``\underline{d}`` of the lattice
(section 5 below).

### 5. Effective conductivity

The average flux follows from the surface integrals of the
[method for spheres](@ref th-mem), which do not depend on the surface chosen in the
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

### 6. Oblate spheroids on a lattice

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
Part B treats the same particles in elasticity; the fields of each spheroid are
listed in the [roadmap](@ref dev-roadmap).

## [B. Spheroids in elasticity](@id th-mem-spheroids-elasticity)

Part B keeps the solutions, the translation, the linear system and the
stiffness of the [spheres in elasticity](@ref th-mem-elasticity), and the
spheroidal harmonics of part A.

The particles are now spheroids of revolution axis ``\underline{e}_3``, all
oriented alike, bonded perfectly or by springs, or empty, in an isotropic
matrix of shear
modulus ``\mu_0`` and Poisson ratio ``\nu_0`` [kushch2013](@cite). Each one keeps
the unknowns of a sphere, the coefficients of the regular spherical solutions
it receives about its center; what changes is its response, which goes
through solutions adapted to its shape, and the reading of the field of a close
neighbor.

### 1. Solutions of the Lamé equation about a spheroid

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

### 2. Response of a spheroid

The spheroid receives the regular spherical solutions
``\underline{u}^{(j)}_{nm}`` about its center, with coefficients ``a``, sends the
``\underline{V}^{(i)}_{nm}`` with coefficients ``b``, and holds inside a series of
regular spherical solutions for its own Poisson ratio ``\nu_1``. On its surface,
of unit normal ``\underline{n}``, the traction ``\boldsymbol{\sigma}\cdot\underline{n}`` is
continuous and the displacement jumps by the compliance of the interface,
``\underline{u}_0 - \underline{u}_1 = \mathbf{S}\cdot(\boldsymbol{\sigma}_1\cdot\underline{n})``,
``\mathbf{S} = s_n\,\underline{n}\otimes\underline{n} + s_t\,(\boldsymbol{1} - \underline{n}\otimes\underline{n})``
for a [`SpringInterface`](@ref) of compliances ``s_n``, ``s_t``, zero for a
perfect bond; on a cavity the traction is zero. The spheroid and its phases being unchanged by a rotation about
``\underline{e}_3``, a field of order ``m`` keeps its order, and is known everywhere
from its values on one meridian. For each order the two conditions are
imposed by least squares at the Gauss points of the meridian ``\varphi = 0``,
the traction brought to a displacement by the factor ``R/\mu_0``, ``R`` the
largest semi-axis; the solution gives ``b = \mathbf{Z}\,a``. With
``n_{\max} = 2`` it is exact under a uniform remote strain, which the example
checks against [`MoriTanaka`](@ref). With springs it is no longer exact, and
converges with ``n_{\max}`` to the response of the
[layered spheroid](@ref th-spheroid-elasticity) obtained by its own route, the
transfer of the Papkovich–Neuber potentials: to ``3\cdot 10^{-7}`` at
``n_{\max} = 8``.

```@example mem_sph_el
using MeanFieldHomogenization, TensND, LinearAlgebra
iso(κ, μ) = TensISO{3}(3κ, 2μ)
stiffness(asm, scheme) = TensND.mandel(homogenize(asm, scheme, :C), TensND.CanonicalBasis{3, Float64}())
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(2.0, 2.0, 0.4)))
add_matrix!(asm, Dict(:C => iso(1.0, 0.6)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(1.0, 1.0, 0.2), Dict(:C => iso(10.0, 6.0)))
norm(stiffness(asm, MultipoleExpansion(; nmax = 2)) - stiffness(asm, MoriTanaka())) / norm(stiffness(asm, MoriTanaka()))
```

### 3. From one spheroid to another

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

### 4. Spheroids on an orthogonal lattice

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

## [C. Ellipsoids in conduction](@id th-mem-ellipsoids)

Part C keeps the structure of part A: the response of a particle over its own
harmonics, the translation by projection on the receiving surface, the far
images of a lattice through the spherical harmonics.

The particles are now ellipsoids of three different semi-axes, all oriented
alike, with a perfect or a Kapitza interface, in an isotropic matrix of
conductivity ``k_0`` [kushch2020](@cite). The harmonics adapted to an ellipsoid
are those of Lamé, built here from the theory of [dassios2012](@citet). An
assembly of [`Ellipsoid`](@ref)s, or of [`InterfacedEllipsoid`](@ref)s for an
imperfect interface, selects this part.

### 1. Ellipsoidal coordinates and Lamé functions

Let ``a_1 > a_2 > a_3`` be the semi-axes, along ``x_1, x_2, x_3``, and
``h_1^2 = a_2^2 - a_3^2``, ``h_2^2 = a_1^2 - a_3^2``, ``h_3^2 = a_1^2 - a_2^2``. The
ellipsoidal coordinates ``(\rho, \mu, \nu)``, with
``0 \le \nu^2 \le h_3^2 \le \mu^2 \le h_2^2 \le \rho^2``, are the three roots ``t`` of

```math
\frac{x_1^2}{t} + \frac{x_2^2}{t - h_3^2} + \frac{x_3^2}{t - h_2^2} = 1 , \qquad t \in \{\rho^2, \mu^2, \nu^2\} ,
```

and the surface of the particle is ``\rho = a_1`` [dassios2012](@cite). Laplace's
equation separates as ``E(\rho)\,E(\mu)\,E(\nu)``, ``E`` being a Lamé function of
degree ``n``, a solution of

```math
(x^2 - h_3^2)(x^2 - h_2^2)\,E'' + x\,(2x^2 - h_3^2 - h_2^2)\,E' + \big[(h_3^2 + h_2^2)\,p - n(n+1)\,x^2\big]\,E = 0
```

of the form ``\psi(x)\,P(x)``: ``P`` a polynomial of every other power and ``\psi`` one
of ``1``, ``\sqrt{|x^2-h_3^2|}``, ``\sqrt{|x^2-h_2^2|}`` and their product, the four
classes K, L, M, N. Writing ``P = \sum_k c_k\,x^{d-2k}`` turns the equation into a
three-term recurrence on the ``c_k`` whose truncation is a tridiagonal
eigenproblem for the separation constant ``p`` [dassios2012](@cite); its
eigenvalues give the ``2n+1`` Lamé functions of the degree, ``c_0 = 1``. The
square roots take the signs of the octant of the point.

The interior and exterior ellipsoidal harmonics are

```math
\mathbb{E}_n^m = E_n^m(\rho)\,E_n^m(\mu)\,E_n^m(\nu) , \qquad
\mathbb{F}_n^m = F_n^m(\rho)\,E_n^m(\mu)\,E_n^m(\nu) , \qquad
F(\rho) = (2n+1)\,E(\rho)\int_\rho^\infty \frac{\mathrm{d}t}{E(t)^2\sqrt{t^2-h_3^2}\sqrt{t^2-h_2^2}} ,
```

the first a polynomial of degree ``n``, the second decaying as ``\rho^{-n-1}``. The
integral is computed as ``\rho\int_0^1 \mathrm{d}s/[E(\rho/s)^2\sqrt{\rho^2-h_3^2s^2}\sqrt{\rho^2-h_2^2s^2}]``.
Its last factor falls to ``\kappa = \sqrt{\rho^2-h_2^2}`` at ``s = 1``, that is ``a_3`` on
the surface, so that for an elongated ellipsoid the integrand varies over a
length ``(\kappa/h_2)^2`` there. On ``[1/2, 1]`` the variable ``\tau`` defined by
``1 - s^2 = (\kappa/h_2)^2\sinh^2\tau`` turns that factor into ``\kappa\cosh\tau``; the
integrands on ``[0, 1/2]`` in ``s`` and on the image of ``[1/2, 1]`` in ``\tau`` are
then smooth and integrated by Gauss–Legendre. At the degree 1 the three classes K, L, M give
``\mathbb{E}_1 = h_2h_3\,x_1``, ``h_1h_3\,x_2``, ``h_1h_2\,x_3``, and the
corresponding ``\mathbb{F}_1`` tend to the same constants times ``x_j/r^3``.

On a surface ``\rho = \mathrm{const}`` the surface harmonics ``S_n^m = E_n^m(\mu)\,E_n^m(\nu)``
are orthogonal for the weight

```math
\mathrm{d}\Omega = \frac{(\mu^2 - \nu^2)\,\mathrm{d}\mu\,\mathrm{d}\nu}
  {\sqrt{\mu^2-h_3^2}\sqrt{h_2^2-\mu^2}\sqrt{h_3^2-\nu^2}\sqrt{h_2^2-\nu^2}} ,
\qquad \int S_n^m S_{n'}^{m'}\,\mathrm{d}\Omega = \gamma_n^m\,\delta_{nn'}\delta_{mm'} .
```

The substitutions ``\mu^2 = h_3^2 + (h_2^2 - h_3^2)\sin^2\alpha``, ``\alpha \in [0, \pi]``, and
``\nu = h_3\sin\beta``, ``\beta \in [0, 2\pi)``, cover the eight octants with their signs
and remove the singular factors: ``\mathrm{d}\Omega = (\mu^2 - \nu^2)/(\mu\sqrt{h_2^2-\nu^2})\,\mathrm{d}\alpha\,\mathrm{d}\beta``,
integrated by Gauss–Legendre in ``\alpha`` and the trapezoidal rule in ``\beta``. At
the degree 1, ``\gamma_1^m = (4\pi/3)\,H_m^2`` with ``H_m = h_1h_2h_3/h_m``.

### 2. Response of an ellipsoid

In the regular field ``\sum a_n^m\,\mathbb{E}_n^m`` the ellipsoid sends
``\sum b_n^m\,\mathbb{F}_n^m``. On ``\rho = a_1`` both are ``E(a_1)\,S`` and ``F(a_1)\,S``, and the
normal derivative is ``h_\rho^{-1}\,\partial_\rho`` with the same metric coefficient
``h_\rho`` on both sides, so that a perfect interface answers harmonic by harmonic:

```math
b_n^m = \frac{\tilde k - 1}{F'/E' - \tilde k\,F/E}\,a_n^m , \qquad \tilde k = k_1/k_0 ,
```

the functions being taken at ``a_1``. A Kapitza interface of resistance
``\rho_{\mathrm{K}}``, ``[\![T]\!] = -\rho_{\mathrm{K}}\,q_n``, keeps the flux continuous harmonic by
harmonic, ``d = (b\,F' + a\,E')/(\tilde k\,E')`` inside, while the jump, projected on
``S_j``, couples the harmonics through the metric:

```math
\gamma_j\,\big(b_j F_j + a_j E_j - d_j E_j\big)
  = \rho_{\mathrm{K}}\,k_1 \sum_k \Big(\int \frac{S_j\,S_k}{h_\rho}\,\mathrm{d}\Omega\Big)\,E'_k\,d_k ,
```

a linear system that gives the response matrix ``\mathbf{Z}``, ``b = \mathbf{Z}\,a``.
[kushch2020](@citet) tabulates the Maxwell estimate ``k_{11}/k_0`` of ellipsoids of
semi-axes ``(1.5, 1, 0.5)`` at the fraction 0.5, ``k_1/k_0 = 100``, against the
conductance ``1/\rho_{\mathrm{K}}`` of the interface and the degree; for
``1/\rho_{\mathrm{K}} = 1`` it gives 1.5541708 at the degree 1 and 1.6079410 at the degree 3.
One ellipsoid in a domain of its own shape is that estimate:

```@example mem_ell
using MeanFieldHomogenization, TensND, LinearAlgebra
semi = (1.5, 1.0, 0.5)
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.5))...)))
add_matrix!(asm, Dict(:K => TensISO{3}(1.0)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., KapitzaInterface(1.0)), Dict(:K => TensISO{3}(100.0)))
[round(canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K))[1, 1]; digits = 7) for nmax in (1, 3)]
```

### 3. From one ellipsoid to another

The coefficients of the field of particle ``J`` on the interior harmonics of
particle ``I`` are read on the surface of ``I``, as for spheroids:

```math
\eta_{(n,m)(k,l)} = \frac{1}{E_n^m(a_1)\,\gamma_n^m}\int \mathbb{F}_k^l(\underline{x} - \underline{X}_J)\,S_n^m\,\mathrm{d}\Omega
```

([kushch2020](@cite), Eq. (33)), which holds whatever the distance as long as the
particles do not overlap. Far apart, the exterior harmonics of ``J`` are expanded
on the decaying spherical ones about its center, by projection on a sphere of
radius ``2a_1``, which holds the focal ellipse; these are translated as in
[the method for spheres](@ref th-mem), and the regular spherical harmonics are
expanded on the ``\mathbb{E}`` of ``I`` by projection on its surface, exactly since
they are polynomials ([kushch2020](@cite), Eqs. (38), (41)). The series of this
far form converges as that of spheroids; it is used beyond
``\tfrac52\,(h_{2,I} + h_{2,J})`` with thirty spherical degrees beyond ``n_{\max}``,
and agrees there with the projection to ``10^{-9}``. On a periodic cell the far
images go through the lattice sums of [the spheroids](@ref th-mem-spheroids),
with the same background.

The coefficients of the translation along ``\underline{e}_3`` tabulated by
[kushch2020](@citet) are recovered by the projection to five digits; at
``|\underline{R}| = 3a_1`` the table gives ``\eta_{(1,1)(1,1)} = 0.917124\times 10^{-2}``:

```@example mem_ell
A = MeanFieldHomogenization.Assemblies
d = A._mem_ellipsoid_data(semi, 2.0, 7, false)
i₁ = A._mem_index(1, -1)
round(real(A._mem_eta_near(d, d, [0.0, 0.0, 4.5], 7)[i₁, i₁]); sigdigits = 6)
```

### 4. Orientation and effective conductivity

The ellipsoids of an assembly share their orientation, their axes along those
of the assembly; the problem is solved in the frame of their axes ordered by
decreasing semi-axis, and the effective conductivity turned back to the frame
of the assembly. The dipole of a particle is read on its harmonics of degree 1,
``\underline{d} = (h_2h_3\,b_1^{\mathrm K}, h_1h_3\,b_1^{\mathrm L}, h_1h_2\,b_1^{\mathrm M})``, and the
formulas of the [spheres](@ref th-mem) apply. The constituents are isotropic
here; anisotropic ones [kushch2017](@cite) are treated in
[The multipole expansion method with anisotropic constituents](@ref th-mem-anisotropic),
and the elasticity of ellipsoids in part D.

## [D. Ellipsoids in elasticity](@id th-mem-ellipsoids-elasticity)

Part D keeps the response by least squares, the reading of close neighbors and
the spherical form of part B, and the Lamé functions and ellipsoidal harmonics
of part C.

The particles are ellipsoids of three different semi-axes
``a_1 > a_2 > a_3``, aligned alike with the axes of the assembly, perfectly
bonded, bonded by springs, or empty, in an isotropic matrix of shear modulus
``\mu_0`` and Poisson ratio ``\nu_0`` [kushch2019a, kushch2019b](@cite). As for
spheroids, each one keeps the unknowns of a sphere; its own solutions are now
built from the decaying ellipsoidal harmonics ``\mathbb{F}_k``, degree 0
included.

### 1. Solutions of the Lamé equation about an ellipsoid

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

### 2. Response of an ellipsoid

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

### 3. Spherical form

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

### 4. A spring interface

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

## Where to go next

The matrix and the particles of these pages are isotropic. The next page,
[The multipole expansion method with anisotropic constituents](@ref th-mem-anisotropic),
lifts that restriction in conduction.
