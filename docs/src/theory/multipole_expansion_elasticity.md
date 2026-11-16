# [The multipole expansion method in elasticity](@id th-mem-elasticity)

!!! info "Before this page"
    [The multipole expansion method](@ref th-mem), which sets the method out in
    conduction: solid harmonics, response of a sphere, translation, linear
    system, closure. This page follows the same steps with the displacement in
    place of the temperature, and keeps its notation: spheres ``I`` of centers
    ``\underline{X}_I`` and radii ``R_I``, harmonics of degree ``n`` and order
    ``m``, coefficients ``a`` of the regular functions and ``b`` of the
    decaying ones.

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

## 1. Vector solid harmonics

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
how [`vector_regular_harmonic`](@ref) and [`vector_irregular_harmonic`](@ref)
evaluate them.

## 2. Solutions of the Lamé equation

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
ratio enters through the factor ``4(1-\nu)`` alone. [`lame_regular`](@ref) and
[`lame_irregular`](@ref) return the solutions at a point.

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
lame_regular(3, 0, 0, x, ν) ≈ (4ν - 2) / 3 * x, lame_regular(1, 2, 0, x, ν) ≈ [-x[1] / 2, -x[2] / 2, x[3]]
```

## 3. Response of a sphere

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

## 4. From one center to another

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
coefficient is affine in ``4(1-\nu_0)``; [`lame_translation`](@ref) returns its
two parts. The formula is checked here against the direct evaluation of a
solution:

```@example mem_el
R, r, nmax = [1.1, -0.6, 0.9], [0.12, 0.2, -0.1], 24
Y = irregular_harmonics(R, nmax + 3)
series = zeros(ComplexF64, 3)
for n in 0:nmax, m in -n:n, j in 1:3
    a, b = lame_translation(3, j, 2, n, 1, m, Y, R)
    global series += (-1)^(n + m) * (a + 4 * (1 - ν) * b) * lame_regular(j, n, m, r, ν)
end
series ≈ lame_irregular(3, 2, 1, R + r, ν)
```

## 5. The linear system

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

## 6. Effective stiffness

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
gives the convergence of the three moduli. The periodic assembly, whose sums
over the images involve more than the lattice harmonics of conduction, and the
fields of each sphere are listed in the [roadmap](@ref dev-roadmap).
