# [The multipole expansion method](@id th-mem)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity) for the transport problem
    and its notation, [Maxwell](@ref th-scheme-maxwell) for the closure used on
    a finite assembly, and [Layered sphere](@ref th-layered-sphere) for the
    transfer matrices of a sphere with layers and imperfect interfaces.

[The cluster model](@ref th-cluster) and
[the equivalent inclusion method](@ref th-eim) describe each inclusion by one
uniform polarization. The multipole expansion method of
[kushch2013](@citet) keeps the whole field instead: around each inclusion the
field is a series of harmonics, the series of one inclusion is rewritten about
the center of the others, and the coefficients solve a linear system. Truncated
at the degree ``t_{\max}``, the solution tends to the exact field of the
assembly as ``t_{\max}`` grows. The scheme is
[`MultipoleExpansion`](@ref), on a [`ParticleAssembly`](@ref) of spheres in an
isotropic matrix.

![The series of decaying harmonics of sphere p is a regular field in the ball about the center of sphere q that reaches the center of p; it is rewritten there on the regular harmonics.](../assets/schemes/scheme_mem.svg)

This page treats conduction, and
[the next one](@ref th-mem-elasticity) elasticity: the temperature ``T`` is harmonic in each phase
and the flux is ``\underline{q} = -k\,\underline{\nabla}T``, with ``k_0`` the
conductivity of the matrix.

## 1. Solid harmonics

Let ``(r, \theta, \varphi)`` be the spherical coordinates of
``\underline{x}`` about ``\underline{e}_3``, and
``\chi_t^s = P_t^s(\cos\theta)\,\mathrm{e}^{\mathrm{i}s\varphi}`` the surface
harmonic of degree ``t \ge 0`` and order ``|s| \le t``, ``P_t^s`` being the
associated Legendre function with ``P_1^1 = \sin\theta``. Two families of
harmonic functions follow, one regular at the origin and one decaying at
infinity, normalized as in [kushch2013](@citet):

```math
y_t^s(\underline{x}) = \frac{r^t}{(t+s)!}\,\chi_t^s , \qquad
Y_t^s(\underline{x}) = \frac{(t-s)!}{r^{t+1}}\,\chi_t^s .
```

The first ones are ``y_0^0 = 1``, ``y_1^0 = x_3``,
``y_1^1 = (x_1 + \mathrm{i}x_2)/2`` and ``Y_0^0 = 1/r``,
``Y_1^0 = x_3/r^3``, ``Y_1^1 = (x_1 + \mathrm{i}x_2)/r^3``: a uniform gradient
is a combination of the ``y_1^s`` and the field of a dipole a combination of
the ``Y_1^s``. Negative orders follow from
``y_t^{-s} = (-1)^s\,\overline{y_t^s}``, and the same for ``Y_t^s``.

[`regular_harmonics`](@ref) and [`irregular_harmonics`](@ref) return the table
of one family at a point:

```@example mem
using MeanFieldHomogenization, TensND, LinearAlgebra
x = [0.3, -0.7, 0.45]
y = regular_harmonics(x, 2)
Y = irregular_harmonics(x, 2)
y[1, 0], y[1, 1], Y[1, 0] * norm(x)^3
```

## 2. Response of a sphere to one harmonic

Around a sphere of radius ``R`` centered at the origin, the temperature in the
matrix is a sum over the harmonics of terms

```math
T = \big(A_\infty\,r^t + B_\infty\,r^{-(t+1)}\big)\,\chi_t^s ,
```

the first being the field the sphere receives and the second the field it
sends back. The sphere is isotropic, so the harmonics do not mix and the ratio

```math
b_t = \frac{B_\infty}{A_\infty}
```

depends on the degree alone; it is the response of the sphere to that degree. For a homogeneous sphere of conductivity ``k_1``
behind a perfect interface, the continuity of ``T`` and of the normal flux at
``r = R`` gives

```math
b_t = -\,\frac{t\,(k_1 - k_0)}{t\,k_1 + (t+1)\,k_0}\,R^{2t+1} .
```

For a sphere with layers and interfaces, the state
``(\hat T, \hat q_n)`` of the harmonic is carried from the center to the outer
radius by the transfer matrices of the [layered sphere](@ref th-layered-sphere),
written with the radial functions ``r^t`` and ``r^{-(t+1)}`` in place of ``r``
and ``r^{-2}``; the interface models act as for the uniform gradient, the
surface conductance ``k^{\mathrm{s}}`` entering through the surface Laplacian
``-t(t+1)/r^2`` of ``\chi_t^s``. A Kapitza resistance ``\rho`` at the surface
of a homogeneous sphere thus replaces ``k_1`` by
``k_1/(1 + t\,\rho\,k_1/R)``, and a surface conductance by
``k_1 + (t+1)\,k^{\mathrm{s}}/R``: the apparent conductivity of the sphere
depends on the degree. [`harmonic_response`](@ref) returns ``b_t``:

```@example mem
K₀, K₁ = TensISO{3}(1.0), TensISO{3}(10.0)
plain = LayeredSphere((0.5,), (K₁,))
resisting = LayeredSphere((0.5,), (K₁,); interfaces = (KapitzaInterface(0.2),))
[harmonic_response(plain, K₀, t) for t in 1:3], [harmonic_response(resisting, K₀, t) for t in 1:3]
```

## 3. From one center to another

A decaying harmonic centered at a point is, away from that point, a regular
field. About a second center, at ``\underline{R}`` from the first, it expands
on the regular harmonics [kushch2013](@cite):

```math
Y_k^l(\underline{r} + \underline{R}) =
  \sum_{t \ge 0}\,\sum_{|s| \le t} (-1)^{t+s}\,Y_{k+t}^{l-s}(\underline{R})\;y_t^s(\underline{r}) ,
\qquad \|\underline{r}\| < \|\underline{R}\| .
```

The normalization of the two families is the one for which the coefficients are
decaying harmonics of the vector joining the centers, without any other factor.
The series converges inside the ball that reaches the first center:

```@example mem
R, r = [1.1, -0.6, 0.9], [0.15, 0.2, -0.1]
YR, yr = irregular_harmonics(R, 22), regular_harmonics(r, 20)
series = sum((-1)^(t + s) * YR[2 + t, 1 - s] * yr[t, s] for t in 0:20 for s in -t:t)
series ≈ irregular_harmonics(R + r, 2)[2, 1]
```

## 4. The linear system

Take ``N`` spheres of centers ``\underline{X}_q`` in a matrix submitted to the
remote gradient ``\underline{G}``. In the matrix,

```math
T(\underline{x}) = \underline{G}\cdot\underline{x}
  + \sum_{p=1}^{N}\,\sum_{k \ge 1}\,\sum_{|l| \le k} A_{kl}^{(p)}\,Y_k^l(\underline{x} - \underline{X}_p) ,
```

the term ``k = 0`` being absent because no sphere is a source of heat. About the
center of sphere ``q``, the remote field and the series of the other spheres
are regular: by the formula of the previous section, the amplitude received on
``y_t^s`` is

```math
c_{ts}^{(q)} + \sum_{p \ne q}\,\sum_{k,l} (-1)^{t+s}\,
  Y_{k+t}^{l-s}(\underline{X}_q - \underline{X}_p)\,A_{kl}^{(p)} ,
```

with ``c_{ts}^{(q)}`` the coefficients of ``\underline{G}\cdot\underline{x}``,
which has the degree 1 only:
``c_{10} = G_3``, ``c_{11} = G_1 - \mathrm{i}G_2``,
``c_{1,-1} = -(G_1 + \mathrm{i}G_2)``. The sphere answers by its response
``b_t^{(q)}``. Since the radial functions of ``y_t^s`` and ``Y_t^s`` are
``r^t/(t+s)!`` and ``(t-s)!/r^{t+1}``,

```math
(t-s)!\,(t+s)!\;A_{ts}^{(q)} = b_t^{(q)}\Big[c_{ts}^{(q)}
  + \sum_{p \ne q}\,\sum_{k,l} (-1)^{t+s}\,
  Y_{k+t}^{l-s}(\underline{X}_q - \underline{X}_p)\,A_{kl}^{(p)}\Big] ,
\qquad 1 \le t \le t_{\max} .
```

These are ``N\,t_{\max}(t_{\max}+2)`` equations for as many coefficients. The
interaction enters by the decaying harmonics of the vectors joining the
centers, which decrease as ``\|\underline{X}_q - \underline{X}_p\|^{-(k+t+1)}``:
the closer two spheres, the higher the degree needed. With ``t_{\max} = 1`` the
spheres exchange dipoles only.

The coefficients are scaled before the system is solved,
``x_{ts}^{(q)} = \sqrt{(t-s)!\,(t+s)!}\;A_{ts}^{(q)}/R_q^{t+1}``, which keeps
its entries of order one at every degree [kushch2013](@cite).

## 5. Effective conductivity

Let ``V`` be a volume that holds the spheres. In the matrix
``\underline{q} + k_0\,\underline{\nabla}T = \underline{0}``, so that the
average of this vector over ``V`` reduces to the spheres, where both terms are
turned into integrals over the surface ``S_q``, taken on the side of the matrix
(``\underline{n}`` is the outward normal and the flux is divergence-free):

```math
\langle\underline{q}\rangle + k_0\,\langle\underline{\nabla}T\rangle
 = \frac{1}{V}\sum_q \oint_{S_q}\big[\underline{x}\,(\underline{q}\cdot\underline{n})
   + k_0\,T\,\underline{n}\big]\,\mathrm{d}S .
```

Defined in this way, the two averages hold whatever the interface between a
sphere and the matrix. In each integral the field of the matrix is the regular
field received by the sphere plus its own series. The regular part contributes
nothing, for it is a solution of the conduction problem of the homogeneous
matrix inside ``S_q``. In the series, the surface harmonics of degree other
than 1 are orthogonal to the components of ``\underline{n}``, and the degree 1
is the field of a dipole,

```math
\sum_{|s| \le 1} A_{1s}^{(q)}\,Y_1^s(\underline{r}) = \frac{\underline{d}^{(q)}\cdot\underline{r}}{r^3} ,
\qquad \underline{d}^{(q)} = \big(A_{11} - A_{1,-1},\ \mathrm{i}\,(A_{11} + A_{1,-1}),\ A_{10}\big) .
```

On the sphere ``r = R`` it gives ``T = \underline{d}\cdot\underline{n}/R^2`` and
``\underline{q}\cdot\underline{n} = 2k_0\,\underline{d}\cdot\underline{n}/R^3``,
and with ``\oint \underline{n}\otimes\underline{n}\,\mathrm{d}S = (4\pi R^2/3)\,\boldsymbol{1}``
the integral is ``4\pi k_0\,\underline{d}^{(q)}``. Hence

```math
\langle\underline{q}\rangle = -k_0\,\langle\underline{\nabla}T\rangle
  + \frac{4\pi k_0}{V}\sum_q \underline{d}^{(q)} .
```

The system being linear, ``\sum_q\underline{d}^{(q)} = \boldsymbol{D}\cdot\underline{G}``,
with ``\boldsymbol{D}`` obtained from the three unit gradients, and the spheres
contribute to the flux, per unit volume, by

```math
\boldsymbol{\Sigma} = -\,\frac{4\pi k_0}{V}\,\boldsymbol{D} .
```

For one sphere, ``\boldsymbol{D} = b_1\,\boldsymbol{1}`` and
``\boldsymbol{\Sigma}`` is the volume fraction times the
[contribution tensor](@ref th-conductivity) of the sphere. What remains is to
relate ``\underline{G}`` to the average gradient, which depends on what
surrounds the assembly.

The same regular field gives the averages over each sphere. The harmonics of
degree other than 1 have a zero mean gradient over a ball, so the mean gradient
and the mean flux of sphere ``q`` are those of the sphere alone under the
uniform gradient ``\underline{G}^{(q)}`` it receives, with
``\underline{d}^{(q)} = b_1^{(q)}\,\underline{G}^{(q)}``:
[`concentration_tensors`](@ref) returns them, layers and interfaces included.

**Finite assembly.** Under [`MixedBC`](@ref), the spheres lie in a domain
``\Omega`` of volume ``V``, itself in the infinite matrix. Following
[Maxwell](@ref th-scheme-maxwell), the domain filled with the effective medium
must send far away the same dipole as the spheres it replaces: with
``\boldsymbol{P}_\Omega`` the Hill tensor of the shape of ``\Omega``,

```math
\boldsymbol{K}^{\mathrm{hom}} = \boldsymbol{K}_0
  + \boldsymbol{\Sigma}\cdot\big(\boldsymbol{1} - \boldsymbol{P}_\Omega\cdot\boldsymbol{\Sigma}\big)^{-1} ,
```

the formula of the Maxwell scheme, in which the contributions of the inclusions
now account for their interactions [kushch2013](@cite).

**Periodic assembly.** Under a [`PeriodicBox`](@ref) the temperature is
``\underline{G}\cdot\underline{x}`` plus a periodic field, so that
``\langle\underline{\nabla}T\rangle = \underline{G}`` and

```math
\boldsymbol{K}^{\mathrm{hom}} = \boldsymbol{K}_0 + \boldsymbol{\Sigma} ,
```

without any closure: the estimate is the exact conductivity of the periodic
medium, up to the truncation.

## 6. Periodic arrays

In a periodic assembly each sphere stands for the lattice of its images, and
the decaying harmonics of the system are replaced by their sums over the
lattice. The sum of the harmonics of degree 2, which decrease as ``r^{-3}``,
converges conditionally, a difficulty met by [rayleigh1892](@citet) and settled
by [mcphedran1978](@citet) for the simple cubic array. It is avoided here by
starting from a function that is periodic by construction: the potential ``S``
of a unit source repeated on the lattice of periods ``(L_1, L_2, L_3)`` and
neutralized by a uniform background [hasimoto1959](@cite),

```math
\Delta S = -4\pi\Big[\sum_{\underline{n}}\delta(\underline{x} - \underline{x}_{\underline{n}}) - \frac{1}{V}\Big] ,
\qquad V = L_1 L_2 L_3 .
```

A decaying harmonic is a derivative of ``1/r``, ``Y_t^s = \mathcal{D}_t^s(1/r)``
with ``\mathcal{D}_t^s`` a harmonic polynomial of the gradient, and its periodic
counterpart is the same derivative of ``S``,
``Y_t^{*s} = \mathcal{D}_t^s S``, returned by [`lattice_harmonics`](@ref).
``S`` is computed by the splitting of [ewald1921](@citet) into a sum over the
lattice and a sum over the reciprocal lattice, both converging as a Gaussian;
the derivative passes through each of them term by term, at any degree.

The background has one consequence. Near a lattice point, ``S - 1/r`` is
regular but not harmonic: its Laplacian is ``4\pi/V``, so its expansion holds
the term ``2\pi r^2/(3V)`` next to the harmonic ones. A periodic dipole of
moment ``\underline{d}`` therefore sends to every sphere, on top of the
harmonics given by the formula of section 3 with ``Y^*`` in place of ``Y``, the
uniform gradient

```math
-\,\frac{4\pi}{3V}\,\underline{d} ,
```

the same in the three directions and whatever the lattice. This term is what
the dipoles alone need to recover a classical result: on a cubic array the
harmonics of degree 2 of the lattice vanish by symmetry, and with
``t_{\max} = 1`` the system reduces to
``\underline{d} = b_1\big(\underline{G} - 4\pi\,\underline{d}/(3V)\big)``,
that is, with ``f = 4\pi R^3/(3V)`` and
``\beta = (k_1 - k_0)/(k_1 + 2k_0)``,

```math
\frac{k^{\mathrm{hom}}}{k_0} = \frac{1 + 2f\beta}{1 - f\beta} ,
```

the estimate of Maxwell and of [Mori–Tanaka](@ref th-scheme-mori-tanaka):

```@example mem
matrix, spheres = Dict(:K => TensISO{3}(1.0)), Dict(:K => TensISO{3}(10.0))
asm = cubic_lattice(:sc, matrix, spheres; fraction = 0.3)
k(scheme) = canonical_components(homogenize(asm, scheme, :K))[1, 1]
k(MultipoleExpansion(; tmax = 1)), k(MoriTanaka())
```

## 7. Convergence with the degree

The higher harmonics carry what the arrangement adds to that estimate. On the
simple cubic array of fraction 0.3 and contrast 10, the value tabulated by
[kushch2014](@citet) is 1.888:

```@example mem
[(tmax, round(k(MultipoleExpansion(; tmax)); digits = 5)) for tmax in (1, 3, 5, 9)]
```

The degree needed grows with the fraction and with the contrast, the spheres
being closer and their interaction stronger: for perfectly conducting spheres
at the fraction 0.5, where the gap between neighbors is 1.5 % of the period,
``t_{\max} = 9`` still misses the converged value 5.891 by more than 1 %. The
manual page [Particle assemblies and N-body schemes](@ref man-assemblies) gives
the syntax of the scheme and of the assemblies it acts on.
