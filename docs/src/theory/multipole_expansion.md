# [The multipole expansion method](@id th-mem)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity) for the transport problem
    and its notation, [Maxwell](@ref th-scheme-maxwell) for the closure used on
    a finite assembly, [Layered sphere](@ref th-layered-sphere) for the transfer
    matrices of a sphere with layers and imperfect interfaces, and
    [Layered spheroid](@ref th-layered-spheroid), whose notation for the
    harmonics is kept here.

[The cluster model](@ref th-cluster) and
[the equivalent inclusion method](@ref th-eim) describe each inclusion by one
uniform polarization. The multipole expansion method of
[kushch2013](@citet) keeps the whole field instead: around each inclusion the
field is a series of harmonics, the series of one inclusion is rewritten about
the center of the others, and the coefficients solve a linear system. Truncated
at the degree ``n_{\max}``, the solution tends to the exact field of the
assembly as ``n_{\max}`` grows. The scheme is
[`MultipoleExpansion`](@ref), on a [`ParticleAssembly`](@ref) of spheres in an
isotropic matrix.

![The series of decaying harmonics of one sphere is a regular field in the ball about the center of another sphere that reaches the center of the first; it is rewritten there on the regular harmonics.](../assets/schemes/scheme_mem.svg)

Part A treats conduction and [part B](@ref th-mem-elasticity) elasticity.

## [A. Conduction](@id th-mem-conduction)

The temperature ``T`` is
harmonic in each phase and the flux is
``\underline{q} = -k\,\underline{\operatorname{grad}}\,T``, with ``k_0`` the
conductivity of the matrix. The assembly holds ``N`` spheres of centers
``\underline{X}_I`` and radii ``R_I``, ``1 \le I \le N``, and is submitted to
the remote gradient ``\underline{H}``.

### 1. Solid harmonics

Let ``(r, \theta, \varphi)`` be the spherical coordinates of
``\underline{x}`` about ``\underline{e}_3``, and ``P_n^m`` the associated
Legendre function of the first kind, of degree ``n`` and order ``m``, as in
[barthelemyBignonnetIJES2020](@citet): ``P_1^1(p) = -\sqrt{1-p^2}`` for
``|p| \le 1``. The surface harmonic of degree ``n \ge 0`` and order
``0 \le m \le n`` is taken as

```math
\chi_n^m(\theta, \varphi) = (-1)^m\,P_n^m(\cos\theta)\,\mathrm{e}^{\mathrm{i}m\varphi} ,
```

so that ``\chi_1^1 = \sin\theta\,\mathrm{e}^{\mathrm{i}\varphi}``. Two families
of harmonic functions follow, one regular at the origin and one decaying at
infinity, with the normalization of [kushch2013](@citet):

```math
y_n^m(\underline{x}) = \frac{r^n}{(n+m)!}\,\chi_n^m , \qquad
Y_n^m(\underline{x}) = \frac{(n-m)!}{r^{n+1}}\,\chi_n^m .
```

The negative orders are defined by
``y_n^{-m} = (-1)^m\,\overline{y_n^m}`` and
``Y_n^{-m} = (-1)^m\,\overline{Y_n^m}``. The first functions are
``y_0^0 = 1``, ``y_1^0 = x_3``, ``y_1^1 = (x_1 + \mathrm{i}x_2)/2`` and
``Y_0^0 = 1/r``, ``Y_1^0 = x_3/r^3``, ``Y_1^1 = (x_1 + \mathrm{i}x_2)/r^3``:
a uniform gradient is a combination of the ``y_1^m`` and the field of a dipole
a combination of the ``Y_1^m``.

[`regular_solid_harmonics`](@extref TensND TensND.regular_solid_harmonics) and [`irregular_solid_harmonics`](@extref TensND TensND.irregular_solid_harmonics) return the table
of one family at a point:

```@example mem
using MeanFieldHomogenization, TensND, LinearAlgebra
x = [0.3, -0.7, 0.45]
y = regular_solid_harmonics(x, 2)
Y = irregular_solid_harmonics(x, 2)
y[1, 0], y[1, 1], Y[1, 0] * norm(x)^3
```

### 2. Response of a sphere to one harmonic

Around a sphere centered at the origin, the temperature in the matrix is a sum
over the degrees and the orders of terms

```math
a_n^m\,y_n^m + b_n^m\,Y_n^m ,
```

the first being the field the sphere receives and the second the field it
sends back; as in [barthelemyBignonnetIJES2020](@citet), ``a`` is the
coefficient of the regular function and ``b`` that of the singular one. The
sphere is isotropic, so the harmonics do not mix, and the ratio of the two
radial functions ``(n-m)!\,b_n^m\,r^{-(n+1)}`` and ``a_n^m\,r^n/(n+m)!``
depends on the degree alone. This defines the response ``\zeta_n`` of the
sphere to the degree ``n``:

```math
(n-m)!\,(n+m)!\;b_n^m = \zeta_n\,a_n^m .
```

For a homogeneous sphere of radius ``R`` and conductivity ``k_1`` behind a
perfect interface, the continuity of ``T`` and of the normal flux at ``r = R``
gives

```math
\zeta_n = -\,\frac{n\,(k_1 - k_0)}{n\,k_1 + (n+1)\,k_0}\,R^{2n+1} .
```

For a sphere with layers and interfaces, the state
``(\hat T, \hat q_n)`` of the harmonic is carried from the center to the outer
radius by the transfer matrices of the [layered sphere](@ref th-layered-sphere),
written with the radial functions ``r^n`` and ``r^{-(n+1)}`` in place of ``r``
and ``r^{-2}``; the interface models act as for the uniform gradient, the
surface conductance ``k^{\mathrm{s}}`` entering through the surface Laplacian
of ``\chi_n^m``, ``\Delta_S\,\chi_n^m = -n(n+1)\,\chi_n^m/r^2``. A Kapitza
resistance ``\rho`` at the surface of a homogeneous sphere thus replaces
``k_1`` by ``k_1/(1 + n\,\rho\,k_1/R)``, and a surface conductance by
``k_1 + (n+1)\,k^{\mathrm{s}}/R``: the apparent conductivity of the sphere
depends on the degree. [`harmonic_response`](@ref) returns ``\zeta_n``:

```@example mem
K₀, K₁ = TensISO{3}(1.0), TensISO{3}(10.0)
plain = LayeredSphere((0.5,), (K₁,))
resisting = LayeredSphere((0.5,), (K₁,); interfaces = (KapitzaInterface(0.2),))
[harmonic_response(plain, K₀, n) for n in 1:3], [harmonic_response(resisting, K₀, n) for n in 1:3]
```

### 3. From one center to another

A decaying harmonic centered at a point is, away from that point, a regular
field. About a second center, at ``\underline{R}`` from the first, it expands
on the regular harmonics [kushch2013](@cite):

```math
Y_{n'}^{m'}(\underline{r} + \underline{R}) =
  \sum_{n \ge 0}\,\sum_{|m| \le n} (-1)^{n+m}\,Y_{n'+n}^{m'-m}(\underline{R})\;y_n^m(\underline{r}) ,
\qquad \|\underline{r}\| < \|\underline{R}\| ,
```

a harmonic whose order exceeds its degree being zero. The normalization of the
two families is the one for which the coefficients are decaying harmonics of
the vector joining the centers, without any other factor. The series converges
inside the ball that reaches the first center:

```@example mem
R, r = [1.1, -0.6, 0.9], [0.15, 0.2, -0.1]
YR, yr = irregular_solid_harmonics(R, 22), regular_solid_harmonics(r, 20)
series = sum((-1)^(n + m) * YR[2 + n, 1 - m] * yr[n, m] for n in 0:20 for m in -n:n)
series ≈ irregular_solid_harmonics(R + r, 2)[2, 1]
```

### 4. The linear system

In the matrix the temperature is the remote field and one series of decaying
harmonics per sphere,

```math
T(\underline{x}) = \underline{H}\cdot\underline{x}
  + \sum_{J=1}^{N}\,\sum_{n' \ge 1}\,\sum_{|m'| \le n'} b_{J,n'}^{m'}\,Y_{n'}^{m'}(\underline{x} - \underline{X}_J) ,
```

the degree 0 being absent because no sphere is a source of heat. About the
center of sphere ``I``, the remote field and the series of the other spheres
are regular: by the formula of the previous section, with
``\underline{R} = \underline{X}_I - \underline{X}_J``, the coefficient
received on ``y_n^m(\underline{x} - \underline{X}_I)`` is

```math
a_{I,n}^m = h_n^m + \sum_{J \ne I}\,\sum_{n',m'} (-1)^{n+m}\,
  Y_{n'+n}^{m'-m}(\underline{X}_I - \underline{X}_J)\,b_{J,n'}^{m'} ,
```

with ``h_n^m`` the coefficients of ``\underline{H}\cdot\underline{x}``, which
has the degree 1 only:

```math
\underline{H}\cdot\underline{x} = \sum_{|m| \le 1} h_1^m\,y_1^m , \qquad
h_1^0 = H_3 , \quad h_1^1 = H_1 - \mathrm{i}H_2 , \quad h_1^{-1} = -(H_1 + \mathrm{i}H_2) .
```

The sphere answers by its response ``\zeta_{I,n}``:

```math
(n-m)!\,(n+m)!\;b_{I,n}^m = \zeta_{I,n}\,a_{I,n}^m , \qquad 1 \le n \le n_{\max} .
```

These two relations are ``N\,n_{\max}(n_{\max}+2)`` equations for as many
coefficients ``a_{I,n}^m``. The interaction enters by the decaying harmonics of
the vectors joining the centers, which decrease as
``\|\underline{X}_I - \underline{X}_J\|^{-(n'+n+1)}``: the closer two spheres,
the higher the degree needed. With ``n_{\max} = 1`` the spheres exchange
dipoles only.

The coefficients are scaled before the system is solved,
``a_{I,n}^m\,R_I^n/\sqrt{(n-m)!\,(n+m)!}``, which keeps the entries of the
system of order one at every degree [kushch2013](@cite).

### 5. Effective conductivity

Let ``V`` be a volume that holds the spheres. In the matrix
``\underline{q} + k_0\,\underline{\operatorname{grad}}\,T = \underline{0}``, so
that the average of this vector over ``V`` reduces to the spheres, where both
terms are turned into integrals over the surface ``S_I`` of each sphere, taken
on the side of the matrix (``\underline{n}`` is the outward normal and the flux
is divergence-free):

```math
\langle\underline{q}\rangle + k_0\,\langle\underline{\operatorname{grad}}\,T\rangle
 = \frac{1}{V}\sum_I \oint_{S_I}\big[(\underline{x} - \underline{X}_I)\,(\underline{q}\cdot\underline{n})
   + k_0\,T\,\underline{n}\big]\,\mathrm{d}S .
```

Defined in this way, the two averages hold whatever the interface between a
sphere and the matrix. In each integral the field of the matrix is the regular
field received by the sphere plus its own series. The regular part contributes
nothing, for it is a solution of the conduction problem of the homogeneous
matrix inside ``S_I``. In the series, the surface harmonics of degree other
than 1 are orthogonal to the components of ``\underline{n}``, and the degree 1
is the field of a dipole: with
``\underline{r} = \underline{x} - \underline{X}_I``,

```math
\sum_{|m| \le 1} b_{I,1}^m\,Y_1^m(\underline{r}) = \frac{\underline{d}_I\cdot\underline{r}}{r^3} ,
\qquad \underline{d}_I = \big(b_{I,1}^{1} - b_{I,1}^{-1},\ \mathrm{i}\,(b_{I,1}^{1} + b_{I,1}^{-1}),\ b_{I,1}^{0}\big) .
```

On the sphere ``r = R_I`` it gives
``T = \underline{d}_I\cdot\underline{n}/R_I^2`` and
``\underline{q}\cdot\underline{n} = 2k_0\,\underline{d}_I\cdot\underline{n}/R_I^3``,
and with ``\oint \underline{n}\otimes\underline{n}\,\mathrm{d}S = (4\pi R_I^2/3)\,\boldsymbol{1}``
the integral is ``4\pi k_0\,\underline{d}_I``. Hence

```math
\langle\underline{q}\rangle = -k_0\,\langle\underline{\operatorname{grad}}\,T\rangle
  + \frac{4\pi k_0}{V}\sum_I \underline{d}_I .
```

The degree 1 received by sphere ``I`` is a uniform gradient
``\underline{H}_I``, of components ``a_{I,1}^0 = H_{I,3}`` and
``a_{I,1}^1 = H_{I,1} - \mathrm{i}H_{I,2}``, and the relation between
``b_{I,1}^m`` and ``a_{I,1}^m`` reads
``\underline{d}_I = \zeta_{I,1}\,\underline{H}_I``. The system being linear,
``\underline{H}_I = \boldsymbol{A}_I\cdot\underline{H}``, with
``\boldsymbol{A}_I`` obtained from the three unit remote gradients, and the
spheres contribute to the flux, per unit volume, by

```math
\boldsymbol{\Sigma} = -\,\frac{4\pi k_0}{V}\sum_I \zeta_{I,1}\,\boldsymbol{A}_I ,
\qquad \langle\underline{q}\rangle = -k_0\,\langle\underline{\operatorname{grad}}\,T\rangle
  - \boldsymbol{\Sigma}\cdot\underline{H} .
```

For one sphere, ``\boldsymbol{A}_1 = \boldsymbol{1}`` and
``\boldsymbol{\Sigma}`` is the volume fraction times the
[contribution tensor](@ref th-conductivity) of the sphere.

The same regular field gives the averages over each sphere. The harmonics of
degree other than 1 have a zero mean gradient over a ball, so the mean gradient
and the mean flux of sphere ``I`` are those of the sphere alone under the
uniform gradient ``\underline{H}_I`` it receives:
[`concentration_tensors`](@ref) returns them, layers and interfaces included.

What remains is to relate ``\underline{H}`` to the average gradient, which
depends on what surrounds the assembly.

**Finite assembly.** Under [`MixedBC`](@ref), the spheres lie in a domain
``\Omega`` of volume ``V``, itself in the infinite matrix. Following
[Maxwell](@ref th-scheme-maxwell), the domain filled with the effective medium
must send far away the same dipole as the spheres it replaces: with
``\boldsymbol{P}_\Omega`` the Hill tensor of the shape of ``\Omega`` in the
matrix,

```math
\boldsymbol{K}^{\mathrm{hom}} = \boldsymbol{K}_0
  + \boldsymbol{\Sigma}\cdot\big(\boldsymbol{1} - \boldsymbol{P}_\Omega\cdot\boldsymbol{\Sigma}\big)^{-1} ,
\qquad \boldsymbol{K}_0 = k_0\,\boldsymbol{1} ,
```

the formula of the Maxwell scheme, in which the contributions of the inclusions
now account for their interactions [kushch2013](@cite).

**Periodic assembly.** Under a [`PeriodicBox`](@ref) the temperature is
``\underline{H}\cdot\underline{x}`` plus a periodic field, so that
``\langle\underline{\operatorname{grad}}\,T\rangle = \underline{H}`` and

```math
\boldsymbol{K}^{\mathrm{hom}} = \boldsymbol{K}_0 + \boldsymbol{\Sigma} ,
```

without any closure: the estimate is the exact conductivity of the periodic
medium, up to the truncation.

### 6. Periodic arrays

In a periodic assembly each sphere stands for the lattice of its images, and
the decaying harmonics of the system are replaced by their sums over the
lattice. The sum of the harmonics of degree 2, which decrease as ``r^{-3}``,
converges conditionally, a difficulty met by [rayleigh1892](@citet) and settled
by [mcphedran1978](@citet) for the simple cubic array. It is avoided here by
starting from a function that is periodic by construction: the potential ``S``
of a unit source repeated on the lattice of periods ``(L_1, L_2, L_3)``, at the
points ``\underline{x}_i``, and neutralized by a uniform background
[hasimoto1959](@cite),

```math
\Delta S = -4\pi\Big[\sum_{i}\delta(\underline{x} - \underline{x}_i) - \frac{1}{V}\Big] ,
\qquad V = L_1 L_2 L_3 .
```

A decaying harmonic is a derivative of ``1/r``,
``Y_n^m = \mathcal{D}_n^m(1/r)`` with ``\mathcal{D}_n^m`` a harmonic polynomial
of the three partial derivatives, and its periodic counterpart is the same
derivative of ``S``, ``Y_n^{*m} = \mathcal{D}_n^m S``, returned by
[`lattice_solid_harmonics`](@extref TensND TensND.lattice_solid_harmonics). ``S`` is computed by the splitting of
[ewald1921](@citet) into a sum over the lattice and a sum over the reciprocal
lattice, both converging as a Gaussian; the derivative passes through each of
them term by term, at any degree.

The background has one consequence. Near a lattice point, ``S - 1/r`` is
regular but not harmonic: its Laplacian is ``4\pi/V``, so its expansion holds
the term ``2\pi r^2/(3V)`` next to the harmonic ones. A periodic dipole of
moment ``\underline{d}`` therefore sends to every sphere, on top of the
harmonics given by the formula of section 3 with ``Y^{*}`` in place of ``Y``,
the uniform gradient

```math
-\,\frac{4\pi}{3V}\,\underline{d} ,
```

the same in the three directions and whatever the lattice. This term is what
the dipoles alone need to recover a classical result: on a cubic array the
harmonics of degree 2 of the lattice vanish by symmetry, and with
``n_{\max} = 1`` the system reduces, for one sphere of radius ``R``,
conductivity ``k_1`` and response
``\zeta_1 = -\beta R^3``, ``\beta = (k_1 - k_0)/(k_1 + 2k_0)``, to
``\underline{d} = \zeta_1\big(\underline{H} - 4\pi\,\underline{d}/(3V)\big)``,
that is, with ``f = 4\pi R^3/(3V)`` the volume fraction,

```math
\frac{k^{\mathrm{hom}}}{k_0} = \frac{1 + 2f\beta}{1 - f\beta} ,
```

the estimate of Maxwell and of [Mori–Tanaka](@ref th-scheme-mori-tanaka):

```@example mem
matrix, spheres = Dict(:K => TensISO{3}(1.0)), Dict(:K => TensISO{3}(10.0))
asm = cubic_lattice(:sc, matrix, spheres; fraction = 0.3)
k(scheme) = canonical_components(homogenize(asm, scheme, :K))[1, 1]
k(MultipoleExpansion(; nmax = 1)), k(MoriTanaka())
```

### 7. Convergence with the degree

The higher harmonics carry what the arrangement adds to that estimate. On the
simple cubic array of fraction 0.3 and contrast 10, the value tabulated by
[kushch2014](@citet) is 1.888:

```@example mem
[(nmax, round(k(MultipoleExpansion(; nmax)); digits = 5)) for nmax in (1, 3, 5, 9)]
```

The degree needed grows with the fraction and with the contrast, the spheres
being closer and their interaction stronger: for perfectly conducting spheres
at the fraction 0.5, where the gap between neighbors is 1.5 % of the period,
``n_{\max} = 9`` still misses the converged value 5.891 by more than 1 %. The
manual page [Particle assemblies and N-body schemes](@ref man-assemblies) gives
the syntax of the scheme and of the assemblies it acts on, and the application
[Arrays and clusters of spheres by the multipole expansion](@ref app-mem-spheres)
follows these convergences against the published tables.

## [B. Elasticity](@id th-mem-elasticity)

Part B follows the steps of part A with the displacement in place of the
temperature, and keeps its notation: spheres ``I`` of centers
``\underline{X}_I`` and radii ``R_I``, harmonics of degree ``n`` and order
``m``, coefficients ``a`` of the regular functions and ``b`` of the decaying
ones.

In an isotropic medium of shear modulus ``\mu`` and Poisson ratio ``\nu``, the
displacement ``\underline{u}`` solves the Lamé equation

```math
\frac{2(1-\nu)}{1-2\nu}\,\underline{\operatorname{grad}}\,(\operatorname{div}\,\underline{u})
  - \underline{\operatorname{curl}}\,(\underline{\operatorname{curl}}\,\underline{u}) = \underline{0} .
```

The matrix has the moduli ``\mu_0`` and ``\nu_0``, and each layer of a sphere
its own. A harmonic function is replaced by three vector solutions per degree
and order, the response of a sphere by a matrix, and the translation formula by
one that mixes the three solutions [kushch2013](@cite). The scheme is the same
[`MultipoleExpansion`](@ref), on a finite [`ParticleAssembly`](@ref) of spheres.

### 1. Vector solid harmonics

With ``y_n^m`` and ``Y_n^m`` the [solid harmonics](@ref th-mem), three families
of harmonic vectors are built on each of them [kushch2013](@cite). The decaying
ones are

```math
\underline{Y}^{(1)}_{nm} = \underline{\operatorname{grad}}\,Y_n^m , \qquad
\underline{Y}^{(2)}_{nm} = \frac{1}{n}\,\underline{\operatorname{grad}}\,Y_n^m\times\underline{x} ,
```

and a third family ``\underline{Y}^{(3)}_{nm}``, whose Cartesian components are
harmonics of degree ``n-1``, such that

```math
\underline{\operatorname{curl}}\,\underline{Y}^{(3)}_{nm} = \underline{Y}^{(2)}_{nm} ,\qquad
\operatorname{div}\,\underline{Y}^{(3)}_{nm} = -Y_n^m , \qquad
\underline{x}\cdot\underline{Y}^{(3)}_{nm} = \frac{r^2\,Y_n^m}{2n-1} .
```

The first two families are a gradient and a rotation field, both
divergence-free. The regular families ``\underline{y}^{(i)}_{nm}`` are built in
the same way on ``y_n^m``: ``\underline{y}^{(1)}_{nm} =
\underline{\operatorname{grad}}\,y_n^m``,
``\underline{y}^{(2)}_{nm} = -\underline{\operatorname{grad}}\,y_n^m\times\underline{x}/(n+1)``,
and ``\underline{y}^{(3)}_{nm}`` with the same curl and divergence relations
and ``\underline{x}\cdot\underline{y}^{(3)}_{nm} = -r^2\,y_n^m/(2n+3)``. The
components of all of them are combinations of scalar solid harmonics, which is
how [`regular_vector_harmonic`](@extref TensND TensND.regular_vector_harmonic) and [`irregular_vector_harmonic`](@extref TensND TensND.irregular_vector_harmonic)
evaluate them.

### 2. Solutions of the Lamé equation

The first two families are divergence-free and their curl is a gradient or
vanishes, so they solve the Lamé equation as they are, whatever ``\nu``. The
third solution is the field of Papkovich and Neuber built on the third family:
for any harmonic vector ``\underline{B}``,
``4(1-\nu)\,\underline{B} - \underline{\operatorname{grad}}\,(\underline{x}\cdot\underline{B})``
solves the equation, hence the three decaying solutions

```math
\underline{U}^{(1)}_{nm} = \underline{Y}^{(1)}_{nm} , \qquad
\underline{U}^{(2)}_{nm} = \underline{Y}^{(2)}_{nm} , \qquad
\underline{U}^{(3)}_{nm} = 4(1-\nu)\,\underline{Y}^{(3)}_{nm}
  - \underline{\operatorname{grad}}\,\big(\underline{x}\cdot\underline{Y}^{(3)}_{nm}\big) ,
```

and the three regular ones ``\underline{u}^{(i)}_{nm}``, built in the same way
on ``\underline{y}^{(i)}_{nm}``. Written in this form, the third solution is
that of [kushch2013](@citet) without its biharmonic function, and the Poisson
ratio enters through the factor ``4(1-\nu)`` alone. [`regular_lame_solution`](@extref TensND TensND.regular_lame_solution) and
[`irregular_lame_solution`](@extref TensND TensND.irregular_lame_solution) return the solutions at a point.

The lowest degrees are the fields of the uniform strain and of the rigid
motions: ``\underline{u}^{(3)}_{00}`` is the dilatation,
``\underline{u}^{(1)}_{1m}`` the translations, ``\underline{u}^{(2)}_{1m}`` the
rotations and ``\underline{u}^{(1)}_{2m}`` the five uniform deviatoric strains.
Among the decaying ones, ``\underline{U}^{(1)}_{00}`` is the center of
dilatation, ``\underline{U}^{(3)}_{1m}`` the point forces and
``\underline{U}^{(2)}_{1m}`` the point torques.

```@example mem_el
using MeanFieldHomogenization, TensND, LinearAlgebra
x, ν = [0.3, -0.7, 0.45], 0.3
regular_lame_solution(3, 0, 0, x, ν) ≈ (4ν - 2) / 3 * x, regular_lame_solution(1, 2, 0, x, ν) ≈ [-x[1] / 2, -x[2] / 2, x[3]]
```

### 3. Response of a sphere

On a sphere of radius ``r`` the displacement and the traction of each solution
are combinations of the three vector surface harmonics of its degree and order,

```math
\underline{S}^{(1)}_{nm} = r\,\underline{\operatorname{grad}}\,\chi_n^m , \qquad
\underline{S}^{(2)}_{nm} = r\,\underline{\operatorname{curl}}\,(\chi_n^m\,\underline{e}_r) , \qquad
\underline{S}^{(3)}_{nm} = \chi_n^m\,\underline{e}_r ,
```

the tangential gradient, the tangential rotation and the normal, which are
orthogonal on the sphere. In a medium of moduli ``\mu`` and ``\nu``, with
``\bar n = n`` for the regular solutions and ``\bar n = -(n+1)`` for the
decaying ones, and up to the factorial of the normalization
[kushch2013](@cite),

| solution | displacement | traction ``/\,2\mu`` |
| :-- | :-- | :-- |
| first | ``r^{\bar n-1}\,(\underline{S}^{(1)}_{nm} + \bar n\,\underline{S}^{(3)}_{nm})`` | ``(\bar n-1)/r`` times the displacement |
| second | ``-r^{\bar n}\,\underline{S}^{(2)}_{nm}/(\bar n+1)`` | ``(\bar n-1)/(2r)`` times the displacement |
| third | ``r^{\bar n+1}\,(\beta_{\bar n}\,\underline{S}^{(1)}_{nm} + \gamma_{\bar n}\,\underline{S}^{(3)}_{nm})`` | ``r^{\bar n}\,(\beta'_{\bar n}\,\underline{S}^{(1)}_{nm} + \gamma'_{\bar n}\,\underline{S}^{(3)}_{nm})`` |

```math
\beta_{\bar n} = \frac{\bar n+5-4\nu}{(\bar n+1)(2\bar n+3)} , \quad
\gamma_{\bar n} = \frac{\bar n-2+4\nu}{2\bar n+3} , \quad
\beta'_{\bar n} = (\bar n+1)\,\beta_{\bar n} - \frac{2(1-\nu)}{\bar n+1} , \quad
\gamma'_{\bar n} = (\bar n+1)\,\gamma_{\bar n} - 2\nu .
```

The state of a harmonic at a radius is the vector of its six components,
``(u_1, u_2, u_3)`` for the displacement and ``(\tau_1, \tau_2, \tau_3)`` for
the traction, on ``(\underline{S}^{(1)}_{nm}, \underline{S}^{(2)}_{nm},
\underline{S}^{(3)}_{nm})``. As for the
[layered sphere](@ref th-layered-sphere) under a uniform strain, it is carried
from the center to the outer radius, through each layer by the six solutions of
that layer and across each interface by the jump of its model. A spring
interface of compliances ``s_n`` and ``s_t`` adds ``s_t\,\tau_1``,
``s_t\,\tau_2`` and ``s_n\,\tau_3`` to the three components of the
displacement. A membrane of moduli ``\kappa^{\mathrm{s}}`` and
``\mu^{\mathrm{s}}`` adds to the traction the surface divergence of its stress,
which on the harmonic of degree ``n`` reads [kushch2013](@cite)

```math
\begin{aligned}
[\![\tau_1]\!] &= \big[-2\kappa^{\mathrm{s}}\,u_3 + \big(n(n+1)(\kappa^{\mathrm{s}}+\mu^{\mathrm{s}}) - 2\mu^{\mathrm{s}}\big)\,u_1\big]/r^2 ,\\
[\![\tau_2]\!] &= \big(n(n+1) - 2\big)\,\mu^{\mathrm{s}}\,u_2/r^2 ,\\
[\![\tau_3]\!] &= \big[4\kappa^{\mathrm{s}}\,u_3 - 2\,n(n+1)\,\kappa^{\mathrm{s}}\,u_1\big]/r^2 ,
\end{aligned}
```

``[\![\cdot]\!]`` being the jump across the interface, outer side minus inner
side. At ``n = 0`` and ``n = 2`` these are the jumps of the bulk and shear
recurrences of the layered sphere. Matching the state at the outer radius with
the six solutions of the matrix, of moduli ``\mu_0`` and ``\nu_0``, gives the
response of the sphere, a matrix ``\mathbf{Z}_n`` independent of the order
since the sphere is isotropic: receiving the regular solutions with the
coefficients ``a^{(j)}_{nm}``, it sends the decaying ones with the coefficients
``b^{(i)}_{nm}``,

```math
\underline{u} = \sum_{j=1}^{3} a^{(j)}_{nm}\,\underline{u}^{(j)}_{nm}
  + \sum_{i=1}^{3} b^{(i)}_{nm}\,\underline{U}^{(i)}_{nm} ,
\qquad (n-m)!\,(n+m)!\;b^{(i)}_{nm} = \sum_{j=1}^{3} (\mathbf{Z}_n)_{ij}\,a^{(j)}_{nm} .
```

[`harmonic_response`](@ref) returns ``\mathbf{Z}_n``. A rigid motion sends
nothing back: the first two columns of ``\mathbf{Z}_1`` vanish, so that a
sphere in equilibrium carries neither a net force nor a net torque.

```@example mem_el
C(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)        # (3k, 2μ)
C₀, sphere = C(1.0, 0.3), LayeredSphere((0.8,), (C(7.0, 0.2),))
Z₁ = harmonic_response(sphere, C₀, 1)
norm(Z₁[:, 1:2]) < 1e-12 * norm(Z₁)
```

### 4. From one center to another

The decaying vector harmonics expand about a second center, at
``\underline{R}`` from the first, on the regular ones of the same and of the
lower families [kushch2013](@cite):

```math
\underline{Y}^{(i)}_{n'm'}(\underline{r} + \underline{R}) =
  \sum_{j \le i}\,\sum_{n,m} (-1)^{n+m}\,\alpha^{(i)(j)}_{n'n\,m'm}(\underline{R})\;\underline{y}^{(j)}_{nm}(\underline{r}) ,
\qquad \|\underline{r}\| < \|\underline{R}\| ,
```

```math
\begin{aligned}
\alpha^{(1)(1)}_{n'n\,m'm} &= \alpha^{(2)(2)}_{n'n\,m'm} = \alpha^{(3)(3)}_{n'n\,m'm} = Y_{n'+n}^{m'-m} , \qquad
\alpha^{(2)(1)}_{n'n\,m'm} = -\alpha^{(3)(2)}_{n'n\,m'm} = \mathrm{i}\,\Big(\frac{m}{n} + \frac{m'}{n'}\Big)\,Y_{n'+n-1}^{m'-m} ,\\
\alpha^{(3)(1)}_{n'n\,m'm} &= \Big[1 + \frac{m\,m'}{n\,n'} - \frac{n'^2-m'^2}{n'(2n'-1)} - \frac{n^2-m^2}{n(2n-1)}\Big]\,Y_{n'+n-2}^{m'-m} ,
\end{aligned}
```

the harmonics being taken at ``\underline{R}``. The first two Lamé solutions
translate as these vectors do. For the third, written in the matrix with
``\nu = \nu_0``, the position from the first center is
``\underline{x} = \underline{r} + \underline{R}``: with
``\underline{B} = \underline{Y}^{(3)}_{n'm'}(\underline{r} + \underline{R})``,

```math
\underline{U}^{(3)}_{n'm'} = \big[4(1-\nu_0)\,\underline{B} - \underline{\operatorname{grad}}\,(\underline{r}\cdot\underline{B})\big]
  - \underline{\operatorname{grad}}\,(\underline{R}\cdot\underline{B}) .
```

``\underline{B}`` being expanded as above, the bracket is term by term a
regular Lamé solution about the second center:
``\underline{u}^{(3)}_{nm}`` for ``\underline{y}^{(3)}_{nm}``;
``4(1-\nu_0)\,\underline{u}^{(2)}_{nm}`` for ``\underline{y}^{(2)}_{nm}``, which
is tangent to the spheres; and ``(4(1-\nu_0) - n)\,\underline{u}^{(1)}_{nm}``
for ``\underline{y}^{(1)}_{nm} = \underline{\operatorname{grad}}\,y_n^m``, since
``\underline{r}\cdot\underline{\operatorname{grad}}\,y_n^m = n\,y_n^m``. The
last term is the gradient of the harmonic function
``\underline{R}\cdot\underline{B}``, whose expansion follows from the scalar
translation formula applied to the Cartesian components of
``\underline{Y}^{(3)}_{n'm'}``: with ``H_{n'n\,m'm}`` its coefficient on
``(-1)^{n+m}\,y_n^m``, it contributes
``-H_{n'n\,m'm}\,\underline{u}^{(1)}_{nm}``. Hence

```math
\underline{U}^{(i)}_{n'm'}(\underline{r} + \underline{R}) =
  \sum_{j \le i}\,\sum_{n,m} (-1)^{n+m}\,\eta^{(i)(j)}_{n'n\,m'm}(\underline{R})\;\underline{u}^{(j)}_{nm}(\underline{r}) ,
\qquad \|\underline{r}\| < \|\underline{R}\| ,
```

with ``\eta^{(i)(j)}_{n'n\,m'm} = \alpha^{(i)(j)}_{n'n\,m'm}`` for the first two
solutions (``i = 1, 2``) and

```math
\eta^{(3)(3)}_{n'n\,m'm} = \alpha^{(3)(3)}_{n'n\,m'm} , \qquad
\eta^{(3)(2)}_{n'n\,m'm} = 4(1-\nu_0)\,\alpha^{(3)(2)}_{n'n\,m'm} , \qquad
\eta^{(3)(1)}_{n'n\,m'm} = \big(4(1-\nu_0) - n\big)\,\alpha^{(3)(1)}_{n'n\,m'm} - H_{n'n\,m'm} ,
```

```math
H_{n'n\,m'm} = \frac{(R_1 + \mathrm{i}R_2)\,(n'+m'-1)(n'+m')\,Y_{n'-1+n}^{m'-1-m}
  - (R_1 - \mathrm{i}R_2)\,(n'-m'-1)(n'-m')\,Y_{n'-1+n}^{m'+1-m}
  + 2R_3\,(n'-m')(n'+m')\,Y_{n'-1+n}^{m'-m}}{2\,n'\,(2n'-1)} .
```

At ``n = 0`` the only regular solution is ``\underline{u}^{(3)}_{00}``. Every
coefficient is affine in ``4(1-\nu_0)``; [`lame_solution_translation`](@extref TensND TensND.lame_solution_translation) returns its
two parts. The formula is checked here against the direct evaluation of a
solution:

```@example mem_el
R, r, nmax = [1.1, -0.6, 0.9], [0.12, 0.2, -0.1], 24
Y = irregular_solid_harmonics(R, nmax + 3)
series = zeros(ComplexF64, 3)
for n in 0:nmax, m in -n:n, j in 1:3
    a, b = lame_solution_translation(3, j, 2, n, 1, m, Y, R)
    global series += (-1)^(n + m) * (a + 4 * (1 - ν) * b) * regular_lame_solution(j, n, m, r, ν)
end
series ≈ irregular_lame_solution(3, 2, 1, R + r, ν)
```

### 5. The linear system

Under the remote strain ``\boldsymbol{E}``, the displacement of the matrix is

```math
\underline{u}(\underline{x}) = \boldsymbol{E}\cdot\underline{x}
  + \sum_{J=1}^{N}\,\sum_{n',m'}\,\sum_{i=1}^{3} b^{(i)}_{J,n'm'}\,\underline{U}^{(i)}_{n'm'}(\underline{x} - \underline{X}_J) ,
```

the solutions being those of the matrix (``\nu = \nu_0``), and sphere ``I``
receives on ``\underline{u}^{(j)}_{nm}(\underline{x} - \underline{X}_I)`` the
coefficient

```math
a^{(j)}_{I,nm} = e^{(j)}_{nm} + \sum_{J \ne I}\,\sum_{i,n',m'} (-1)^{n+m}\,
  \eta^{(i)(j)}_{n'n\,m'm}(\underline{X}_I - \underline{X}_J)\,b^{(i)}_{J,n'm'} ,
```

to which it answers by its response,
``(n-m)!\,(n+m)!\;b^{(i)}_{I,nm} = \sum_j (\mathbf{Z}_{I,n})_{ij}\,a^{(j)}_{I,nm}``.
The coefficients ``e^{(j)}_{nm}`` are those of the remote field, up to a rigid
translation that sends nothing back. It has the degrees 0 and 2 only
[kushch2013](@cite):

```math
\boldsymbol{E}\cdot\underline{x} = e^{(3)}_{00}\,\underline{u}^{(3)}_{00} + \sum_{|m| \le 2} e^{(1)}_{2m}\,\underline{u}^{(1)}_{2m} ,
```

```math
e^{(3)}_{00} = \frac{\operatorname{tr}\boldsymbol{E}}{4\nu_0 - 2} , \quad
e^{(1)}_{20} = \frac{2E_{33} - E_{11} - E_{22}}{3} , \quad
e^{(1)}_{21} = E_{13} - \mathrm{i}E_{23} , \quad
e^{(1)}_{22} = E_{11} - E_{22} - 2\mathrm{i}E_{12} ,
```

with ``e^{(1)}_{2,-m} = (-1)^m\,\overline{e^{(1)}_{2m}}``. There are
``1 + 3\,n_{\max}(n_{\max}+2)`` unknowns per sphere, each scaled to a
displacement at the surface of its sphere.

### 6. Effective stiffness

Let ``\boldsymbol{\tau} = \boldsymbol{\sigma} - \mathbb{C}_0 : \boldsymbol{\varepsilon}``
be the polarization, which vanishes in the matrix, of stiffness
``\mathbb{C}_0``. As for the flux in conduction, its average over a volume
``V`` holding the spheres reduces to integrals over their surfaces, taken on
the side of the matrix, in which the regular part of the field contributes
nothing and the series contributes by its dipoles alone: the center of
dilatation and the five solutions of the third family at the degree 2
[kushch2013](@cite),

```math
\begin{aligned}
\operatorname{tr}\langle\boldsymbol{\tau}\rangle &= \frac{24\pi\mu_0(1-\nu_0)}{(1-2\nu_0)\,V}\sum_I b^{(1)}_{I,00} , &
2\langle\tau_{33}\rangle - \langle\tau_{11}\rangle - \langle\tau_{22}\rangle &= -\frac{32\pi\mu_0(1-\nu_0)}{V}\sum_I b^{(3)}_{I,20} ,\\
\langle\tau_{13}\rangle - \mathrm{i}\langle\tau_{23}\rangle &= -\frac{16\pi\mu_0(1-\nu_0)}{V}\sum_I b^{(3)}_{I,21} , &
\langle\tau_{11}\rangle - \langle\tau_{22}\rangle - 2\mathrm{i}\langle\tau_{12}\rangle &= -\frac{64\pi\mu_0(1-\nu_0)}{V}\sum_I b^{(3)}_{I,22} .
\end{aligned}
```

Solving for the six unit strains gives the tensor ``\boldsymbol{\Sigma}`` such
that ``\langle\boldsymbol{\tau}\rangle = \boldsymbol{\Sigma} : \boldsymbol{E}``,
and a finite assembly in a domain ``\Omega``, of Hill tensor
``\mathbb{P}_\Omega`` in the matrix, is closed as in conduction,

```math
\mathbb{C}^{\mathrm{hom}} = \mathbb{C}_0 + \boldsymbol{\Sigma} :
(\mathbb{I} - \mathbb{P}_\Omega : \boldsymbol{\Sigma})^{-1} .
```

For one sphere at the center of a ball the estimate is that of Maxwell, hence
of [Mori–Tanaka](@ref th-scheme-mori-tanaka), at any degree, the sphere
receiving the uniform strain alone:

```@example mem_el
asm = ParticleAssembly(; boundary = MixedBC(2.0))
add_matrix!(asm, Dict(:C => C₀))
add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(1.1), Dict(:C => C(100.0, 0.2)))
components(C) = (M = mandel(C, CanonicalBasis{3, Float64}()); round.((M[1, 1], M[1, 2], M[4, 4]); digits = 8))
components(homogenize(asm, MultipoleExpansion(; nmax = 4), :C)), components(homogenize(asm, MoriTanaka(), :C))
```

With several spheres the arrangement enters. For 27 nearly rigid spheres on a
cubic block of ``3\times3\times3`` sites, at the fraction 0.2 in the ball of the
same fraction, [kushch2013](@citet) reports ``k^{\mathrm{hom}}/k_0 = 1.4064`` and
two shear moduli, 1.6568 for ``(C_{1111} - C_{1122})/2`` and 1.4719 for
``C_{1212}``, in units of ``\mu_0``: the block is cubic, which no one-site
estimate sees.

```@example mem_el
function block(N, c, C₁)
    R = cbrt(3c / (4π))
    asm = ParticleAssembly(; boundary = MixedBC(R * cbrt(N^3 / c)))
    add_matrix!(asm, Dict(:C => C₀))
    for i in 1:N, j in 1:N, k in 1:N
        add_particle!(asm, Symbol(:p, i, j, k), (i, j, k) .- (N + 1) / 2, Ellipsoid(R), Dict(:C => C₁))
    end
    return asm
end
M = mandel(homogenize(block(3, 0.2, C(1e6, 0.3)), MultipoleExpansion(; nmax = 4), :C), CanonicalBasis{3, Float64}())
k₀ = 2 * 1.3 / (3 * 0.4)
round.(((M[1, 1] + 2M[1, 2]) / 3k₀, (M[1, 1] - M[1, 2]) / 2, M[4, 4] / 2); digits = 4)
```

The degree 4 gives the bulk modulus and misses the last digits of the two shear
moduli, which the degree 6 recovers. The application
[Arrays and clusters of spheres by the multipole expansion](@ref app-mem-spheres)
gives the convergence of the three moduli.

### 7. Periodic arrays

In a periodic assembly each sphere stands for the lattice of its images, as in
[conduction](@ref th-mem), and each decaying solution is replaced by its sum
over the lattice. The first two families are made of decaying harmonics and
are summed by the lattice harmonics ``Y_n^{*m} = \mathcal{D}_n^m S`` of
conduction. The third one is not: it holds
``\underline{x}\cdot\underline{Y}^{(3)}_{n'm'} = r^2\,Y_{n'}^{m'}/(2n'-1)``,
which decreases one degree slower than ``Y_{n'}^{m'}``.

The way out is to see every decaying solution as a differential operator with
constant coefficients applied to the two functions ``1/r`` and ``r``. The
harmonic polynomial ``\mathcal{D}_n^m`` of the partial derivatives acts on a
radial function as ``(1/r)\,\mathrm{d}/\mathrm{d}r`` applied ``n`` times
(Hobson's theorem), so that

```math
\mathcal{D}_n^m\,r = -\frac{r^2\,Y_n^m}{2n-1} ,
\qquad\text{hence}\qquad
\underline{U}^{(3)}_{n'm'} = 4(1-\nu_0)\,\underline{Y}^{(3)}_{n'm'}
  + \underline{\operatorname{grad}}\,\mathcal{D}_{n'}^{m'}\,r ,
```

the components of ``\underline{Y}^{(3)}_{n'm'}`` being harmonics of degree
``n'-1``, that is derivatives of ``1/r``. The only property of the pair
``(1/r, r)`` the Lamé equation uses away from the source is
``\Delta(1/r) = 0`` and ``\Delta r = 2/r``. Its periodic counterpart is the pair
``(S, \rho)``, with ``\rho`` the periodic function of zero mean such that

```math
\Delta\rho = 2S ,
```

and the sum over the images of a decaying solution is the same operator
applied to ``S`` and ``\rho``. The derivatives ``B_n^{*m} = \mathcal{D}_n^m\rho``
replace ``-r^2\,Y_n^m/(2n-1)`` and are returned with the lattice harmonics by
[`lattice_solid_biharmonics`](@extref TensND TensND.lattice_solid_biharmonics). They are computed by Ewald's splitting of
``r``: the short-range part
``\psi(r) = r\,\operatorname{erfc}(\alpha r) - e^{-\alpha^2 r^2}/(\alpha\sqrt{\pi})``,
for which ``(1/r)\,\mathrm{d}\psi/\mathrm{d}r = \operatorname{erfc}(\alpha r)/r``
is the short-range part of ``1/r``, and the long-range part, the Fourier series
of coefficients
``-8\pi\,(1 + k^2/4\alpha^2)\,e^{-k^2/4\alpha^2}/(V k^4)``, ``k \ne 0``.
Both converge as a Gaussian, and the result does not depend on ``\alpha``.

In the translation of section 4, the harmonics ``Y_N^M`` taken at
``\underline{R}`` become ``Y_N^{*M}``, and the products of ``H_{n'n\,m'm}`` by the
components of ``\underline{R}`` become their periodic counterparts through the
identities

```math
\begin{aligned}
\frac{x_1 + \mathrm{i}x_2}{2}\,Y_N^M &= \frac{r^2\,Y_{N+1}^{M+1} - (N-M)(N-M-1)\,Y_{N-1}^{M+1}}{2(2N+1)} , \qquad
\frac{x_1 - \mathrm{i}x_2}{2}\,Y_N^M = \frac{-r^2\,Y_{N+1}^{M-1} + (N+M)(N+M-1)\,Y_{N-1}^{M-1}}{2(2N+1)} ,\\
x_3\,Y_N^M &= \frac{r^2\,Y_{N+1}^M + (N-M)(N+M)\,Y_{N-1}^M}{2N+1} ,
\end{aligned}
```

in which ``r^2\,Y_K^\mu`` becomes ``-(2K-1)\,B_K^{*\mu}``.

The background adds uniform terms, as the dipoles did in conduction. About any
point, ``\Delta S = 4\pi/V`` off the sources, so ``S`` differs from a harmonic
function by ``c\,|\underline{y}|^2`` with ``c = 2\pi/(3V)``, and ``\rho`` differs
from a function with the properties of ``r`` by ``d\,|\underline{y}|^4`` with
``d = \pi/(15V)``. The operators of the decaying solutions applied to these two
polynomials give, by Hobson's theorem again, fields of degree at most 1, and
only at the lowest degrees. The sphere ``I`` thus receives, from every sphere
``J`` of the cell, itself included, on top of the translated series,

```math
\begin{aligned}
&\frac{4\pi}{3V}\,b^{(1)}_{J,00}\,\boldsymbol{1}
  &&\text{(a uniform dilatation, from the center of dilatation)},\\
&\frac{8\pi\,(5\nu_0 - 4)}{45\,V}\,(2-m)!\,(2+m)!\;b^{(3)}_{J,2m}\,\underline{u}^{(1)}_{2m}
  &&\text{(a uniform deviatoric strain, from the five dipoles of degree 2)},
\end{aligned}
```

the first one written as the strain it imposes, and a translation and a
rotation from the solutions of degree 1, to which no sphere responds. A sphere
in equilibrium sends no ``\underline{U}^{(3)}_{1m}``, the field of a point
force, which would add a quadratic field.

The mean strain of the periodic sums vanishes, the mean strain is the applied
``\boldsymbol{E}``, and the polarization of section 6 gives the stiffness
without closure,

```math
\mathbb{C}^{\mathrm{hom}} = \mathbb{C}_0 + \boldsymbol{\Sigma} .
```

For the simple cubic array at ``c = 0.4``, with ``\mu_1/\mu_0 = 1000`` and
``\nu_0 = \nu_1 = 0.3``, [kushch2013](@citet) reports
``C^{\mathrm{hom}}_{1133}/(2\mu_0) = 1.12681``,
``C^{\mathrm{hom}}_{3333}/(2\mu_0) = 4.79979`` and
``C^{\mathrm{hom}}_{1212}/\mu_0 = 2.24677`` at the degree 20:

```@example mem_el
asm = cubic_lattice(:sc, Dict(:C => C(1.0, 0.3)), Dict(:C => C(1000.0, 0.3)); fraction = 0.4)
for nmax in (4, 8, 12)
    M = mandel(homogenize(asm, MultipoleExpansion(; nmax), :C), CanonicalBasis{3, Float64}())
    println("nmax = ", nmax, ": ", round.((M[1, 3] / 2, M[3, 3] / 2, M[6, 6] / 2); digits = 5))
end
```

The fields of each sphere in elasticity are listed in the
[roadmap](@ref dev-roadmap).

## Where to go next

The next page,
[The multipole expansion method for spheroids and ellipsoids](@ref th-mem-shapes),
replaces the spherical harmonics by those of the spheroid and of the
ellipsoid; the expansion itself, its translations and its linear system are
kept.
