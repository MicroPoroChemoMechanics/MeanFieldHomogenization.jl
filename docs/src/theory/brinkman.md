# [Permeability from Stokes flow](@id th-brinkman)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity), whose Darcy row is the
    starting point here, and
    [Concentration and contribution tensors](@ref th-concentration) for the
    tensors that the schemes combine.

In the dictionary of [Extension to conductivity](@ref th-conductivity), Darcy's
law makes the permeability a conductivity. That reading holds when every phase
is itself a porous medium of known permeability (Darcy–Darcy upscaling). It
does not hold at the scale of the pores, where the fluid obeys the Stokes
equations and the solid admits no flow at all. [bignonnetIJES2022](@citet)
showed that the estimates of continuum micromechanics nonetheless carry over to
this Stokes-to-Darcy upscaling. Their auxiliary problem is no longer Eshelby's
but a *Brinkman inhomogeneity problem*, whose solution depends on the size of the
inclusion. This page states that formulation in the notation of the package.
The syntax is given in [Brinkman inclusions](@ref man-brinkman), and the
estimates are compared in the application
[Stokes-to-Darcy permeability of porous media](@ref app-stokes-darcy).

## 1. Two scales of flow

At the scale of the material, the filtration velocity ``\underline{V}`` and the
pressure ``P`` of a fluid of viscosity ``\mu`` obey Darcy's law,

```math
\nabla\cdot\underline{V} = 0 ,
\qquad
\underline{V} = -\frac{1}{\mu}\,\boldsymbol{k}^{\mathrm{hom}}\cdot\nabla P .
```

At the scale of the pores, a representative volume ``\Omega`` holds a fluid
domain ``\Omega_f`` and a rigid solid ``\Omega_s``. The pressure gradient
``\underline{F} = \nabla P`` acts as a uniform load on the Stokes flow in the
pores, with no slip on the solid and periodic fluctuations:

```math
\nabla\cdot\boldsymbol{\sigma} = \underline{0} ,\quad
\boldsymbol{\sigma} = -p\,\boldsymbol{1} + 2\mu\,\nabla^{s}\underline{v} ,\quad
\nabla\cdot\underline{v} = 0 \quad\text{in } \Omega_f ,
\qquad \underline{v} = \underline{0} \quad\text{on } \partial\Omega_s .
```

The problem is linear in ``\underline{F}``, so
``\underline{v} = -\mu^{-1}\,\boldsymbol{\kappa}\cdot\underline{F}`` with a
velocity concentration field ``\boldsymbol{\kappa}``, set to zero in the solid.
Averaging the velocity then yields the permeability,
``\boldsymbol{k}^{\mathrm{hom}} = \langle\boldsymbol{\kappa}\rangle_\Omega``.

## 2. Extension to the solid

Elasticity splits a volume average over the phases. To do the same here, the
Stokes problem is extended to the whole of ``\Omega`` by a body force
``\underline{f}`` exerted on a fictitious fluid occupying the solid. The force
vanishes in the true fluid, keeps the velocity zero in the solid, and its
average balances the pressure gradient, ``\langle\underline{f}\rangle_\Omega = \underline{F}``.
Under a prescribed average velocity ``\underline{V}``, the velocity and the
force are linear in ``\underline{V}``:

```math
\underline{v} = \boldsymbol{a}\cdot\underline{V} ,\quad
\underline{f} = -\mu\,\boldsymbol{\rho}\cdot\underline{V} ,
\qquad
\langle\boldsymbol{a}\rangle_\Omega = \boldsymbol{1} ,\quad
\langle\boldsymbol{\rho}\rangle_\Omega = \boldsymbol{r}^{\mathrm{hom}} = \left(\boldsymbol{k}^{\mathrm{hom}}\right)^{-1} .
```

For any uniform tensor ``\boldsymbol{r}_0``, the homogenized resistance then
splits over regions ``\Omega_i`` of volume fractions ``\varphi_i``:

```math
\boldsymbol{r}^{\mathrm{hom}}
  = \boldsymbol{r}_0 + \sum_i \varphi_i\,\big\langle \boldsymbol{\rho} - \boldsymbol{r}_0\cdot\boldsymbol{a} \big\rangle_{\Omega_i} .
```

This is the counterpart of the decomposition of the effective stiffness on which
every scheme rests. What remains is to estimate each regional average from an
auxiliary problem.

## 3. The Brinkman inhomogeneity problem

The auxiliary problem places one pattern ``\omega_i`` of region ``\Omega_i`` in an
unbounded reference medium of permeability ``\boldsymbol{k}_0 = k_0\,\boldsymbol{1}``.
In the reference medium and in a porous inclusion of permeability ``k_1``, the
body force is Brinkman's drag [brinkmanASR1949](@cite):

```math
\nabla\cdot\boldsymbol{\sigma} + \underline{f} = \underline{0} ,\qquad
\underline{f} = -\mu\,\boldsymbol{k}^{-1}\cdot\underline{v} ,\qquad
\boldsymbol{\sigma} = -p\,\boldsymbol{1} + 2\mu\,\nabla^{s}\underline{v} ,\qquad
\nabla\cdot\underline{v} = 0 ,
```

with ``\underline{v} \to \underline{V}_0`` far from the inclusion. A clear fluid is
the limit ``\boldsymbol{k}^{-1} = \boldsymbol{0}`` (Stokes flow), and a solid the limit
``\boldsymbol{k} = \boldsymbol{0}`` (no flow, and an undetermined force whose resultant
is the traction on the solid). Following [bignonnetIJES2022](@citet), the
viscosity is the same in every medium, and the velocity and the traction
``\boldsymbol{\sigma}\cdot\underline{n}`` are continuous across every interface. Both
assumptions are restrictions, since an effective viscosity and a stress jump at
a clear-fluid–porous interface have been proposed, but neither comes with a
predictive value.

The averages over the inclusion define the velocity concentration and the
average drag per unit ``\mu\underline{V}_0``:

```math
\langle\underline{v}\rangle_{\omega} = \boldsymbol{A}\cdot\underline{V}_0 ,
\qquad
-\frac{1}{\mu}\,\langle\underline{f}\rangle_{\omega} = \langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle\cdot\underline{V}_0 ,
```

and their combination is the *resistance contribution tensor*

```math
\boldsymbol{R} = \langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle - \boldsymbol{r}_0\cdot\boldsymbol{A} ,
```

symmetric by the reciprocity of the Brinkman problem. With the velocity in the
role of the gradient and the drag density in that of the flux, these are exactly
the gradient-side concentration, the flux-side concentration and the
contribution tensor that the schemes consume. The package therefore enters
Brinkman inhomogeneities through the concentration gate of the
[custom-inclusion contract](@ref man-custom-inclusions).

**Why the size enters.** Brinkman's equation compares the viscous term
``\mu\Delta\underline{v}`` with the drag ``\mu\underline{v}/k_0``: they balance over the
screening length ``\ell_0 = \sqrt{k_0}``. An inclusion of size ``R`` is therefore
characterized by ``\alpha_0 = R/\ell_0`` (and ``\alpha_1 = R/\sqrt{k_1}`` inside a porous
one). The problem is not invariant under a change of scale, so its solution is
not a property of the shape alone and there is no Hill tensor. As
``\alpha_0, \alpha_1 \to \infty`` the viscous boundary layers become thin and Darcy's
law is recovered, together with the conduction inclusion problem.

**Two formulations.** Under a prescribed remote force instead of a velocity,
the force concentration ``\boldsymbol{B}`` (``\langle\underline{f}\rangle_\omega = \boldsymbol{B}\cdot\underline{F}_0``)
and the average velocity ``\langle\boldsymbol{k}\cdot\boldsymbol{B}\rangle`` play the same
roles in terms of permeability, and the two descriptions are related pointwise by
``\boldsymbol{k}\cdot\boldsymbol{B} = \boldsymbol{A}\cdot\boldsymbol{k}_0``. The resistance form is
finite for a clear-fluid reference (``\boldsymbol{r}_0 = \boldsymbol{0}``) and the
permeability form for an impervious one (``\boldsymbol{k}_0 = \boldsymbol{0}``). In the
permeability form the gradient-side quantity is the force density, which
derives from a potential far from the inclusion, as the gradient of
conduction does. Its Darcy limit is therefore exactly the conduction inclusion
problem with ``\boldsymbol{k}`` as conductivity. This is the form in which the Hill
tensor of conduction may be used, as Maxwell's scheme requires.

## 4. The sphere

The derivation below is the one carried out symbolically in
`scripts/71_brinkman_permeability.jl` with the curvilinear operators of TensND,
along the route of [The elastic sphere, solved symbolically](@extref TensND tut-sphere-problems)
(operators: [Curvilinear differential calculus](@extref TensND th-curvilinear)).

**Fields.** Under a remote velocity ``V\underline{e}_3``, the axial symmetry and the
linearity fix the angular dependence in spherical coordinates
``(r, \theta, \phi)``. Incompressibility then gives the orthoradial velocity in
terms of the radial one:

```math
\underline{v} = u(r)\cos\theta\,\underline{e}_r
  - \Big(u(r) + \frac{r\,u'(r)}{2}\Big)\sin\theta\,\underline{e}_\theta ,
\qquad
p = P(r)\cos\theta .
```

**Radial equation.** The two components of the momentum balance lose their
angular factor. The orthoradial one gives ``P`` in terms of ``u``, and the radial
one becomes a fourth-order equation in a medium of screening length ``\ell``:

```math
\ell^2\big(r^3 u'''' + 8r^2 u''' + 8r\,u'' - 8u'\big) = r^3 u'' + 4r^2 u' .
```

Its solutions are spanned by

```math
1 ,\qquad r^{-3} ,\qquad \frac{(\ell - r)\,e^{r/\ell}}{r^3} ,\qquad \frac{(\ell + r)\,e^{-r/\ell}}{r^3} ,
```

whose Wronskian ``-6/(r^8\ell^5)`` does not vanish. The first two are the
solutions of Stokes flow, and the last two decay or grow over the screening
length. Outside the inclusion, boundedness at infinity and the remote velocity
retain ``1``, ``r^{-3}`` and the decaying exponential term. Inside, regularity at
the origin excludes ``r^{-3}`` and each exponential term alone. Only their
difference remains finite, since
``\big[(\ell - r)e^{r/\ell} - (\ell + r)e^{-r/\ell}\big]/r^3 \to -2/(3\ell^2)``.
Inside a Stokes sphere (``\ell \to \infty``), the regular solutions are ``1`` and
``r^2``.

**Continuity and average.** The four constants follow from the continuity of
``u``, ``u'`` and of the two traction components
``\underline{e}_r\cdot\boldsymbol{\sigma}\cdot\underline{e}_r`` and
``\underline{e}_\theta\cdot\boldsymbol{\sigma}\cdot\underline{e}_r`` at ``r = R``. The average
velocity needs no integration: by the divergence theorem,
``\langle\underline{v}\rangle_\omega = |\omega|^{-1}\int_{\partial\omega}\underline{x}\,(\underline{v}\cdot\underline{n})\,dS``,
which for this field reduces to ``u(R)\,\underline{e}_3``. The velocity
concentration of a porous sphere is then

```math
\begin{gathered}
\boldsymbol{A} = 3\,\frac{N}{D}\,\boldsymbol{1} ,\qquad
\beta = 3\,(\alpha_1^2 - \alpha_0^2)(1 + \alpha_0) ,\\
N = (\alpha_0^3\alpha_1^2 - \beta)\tanh\alpha_1 + (\alpha_0^2\alpha_1^2 + \beta)\,\alpha_1 ,\\
D = (\alpha_0^3\alpha_1^2 + 2\alpha_0\alpha_1^4 - \beta)\tanh\alpha_1 + (\alpha_0^2\alpha_1^2 + 2\alpha_1^4 + \beta)\,\alpha_1 ,
\end{gathered}
```

the closed form of [bignonnetIJES2022](@citet). Its limits are the three
inclusions most often needed:

```math
\begin{gathered}
\boldsymbol{A}^{\mathrm{Stokes}} = 3\,\frac{15 + 15\alpha_0 + 6\alpha_0^2 + \alpha_0^3}{45 + 45\alpha_0 + 6\alpha_0^2 + \alpha_0^3}\,\boldsymbol{1} ,\\
\langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle^{\mathrm{solid}} = \frac{3\,(3 + 3\alpha_0 + \alpha_0^2)}{2R^2}\,\boldsymbol{1} ,
\qquad
\boldsymbol{A}^{\mathrm{Darcy}} = \frac{3k_1}{k_1 + 2k_0}\,\boldsymbol{1} .
\end{gathered}
```

The first is the Stokes sphere (``\alpha_1 \to 0``). The second is the drag on an
impervious sphere, obtained from the traction on it as
``\langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle = (T_r + 2T_\theta)/(\mu R)``. The third is
Darcy's limit (``R \to \infty`` at fixed ``k_1/k_0``).

**Cylinder and layer.** An infinite cylinder of axis ``\underline{n}`` carries two
problems. The axial one is driven by a body force along ``\underline{n}``: its
velocity obeys a modified Bessel equation, solved by ``I_0(r/\ell)`` inside and
``K_0(r/\ell)`` outside. The transverse one is solved by ``1``, ``r^{-2}``,
``K_1(r/\ell)/r`` and ``I_1(r/\ell)/r``. Writing ``\rho_0 = K_0(\alpha_0)/K_1(\alpha_0)`` and
``q_1 = I_1(\alpha_1)/I_0(\alpha_1)``, the average velocity of a porous cylinder is
``\boldsymbol{A} = A_n\,\underline{n}\otimes\underline{n} + A_t\,(\boldsymbol{1} - \underline{n}\otimes\underline{n})``
with

```math
\begin{gathered}
A_n = \frac{\alpha_0\,C}{\alpha_1^3\,(\alpha_1\rho_0 q_1 + \alpha_0)} ,
\qquad
C = \big[2(\alpha_1^2 - \alpha_0^2) + \alpha_0\alpha_1^2\rho_0\big]\,q_1 + \alpha_1\alpha_0^2 ,\\
A_t = \frac{2\alpha_0\,C}
  {\big[2\alpha_0(\alpha_1^2 - \alpha_0^2) + \alpha_1^2(\alpha_1^2 + \alpha_0^2)\rho_0\big]\,q_1 + \alpha_0\alpha_1(\alpha_0^2 + \alpha_1^2)} .
\end{gathered}
```

A planar layer of half-thickness ``R`` and normal ``\underline{n}`` transmits the
normal flow unchanged (``A_n = 1``), and its in-plane component follows from a
one-dimensional Brinkman problem solved by ``\cosh`` and decaying exponentials:

```math
A_t = \frac{\alpha_0\big[\big(\alpha_1^2(1 + \alpha_0) - \alpha_0^2\big)\tanh\alpha_1 + \alpha_0^2\alpha_1\big]}
  {\alpha_1^3\,(\alpha_1\tanh\alpha_1 + \alpha_0)} .
```

Their Stokes limits are ``A_n = 1 + \alpha_0^2/8 + \alpha_0\rho_0/2`` for the tube and
``A_t = 1 + \alpha_0 + \alpha_0^2/3`` for the slit. In an impervious matrix
(``k_0 \to 0``) the corresponding velocities ``\boldsymbol{A}\cdot\boldsymbol{k}_0`` tend to
Poiseuille's ``(R^2/8)\,\underline{n}\otimes\underline{n}`` and
``(R^2/3)\,(\boldsymbol{1} - \underline{n}\otimes\underline{n})``.

**Coated sphere.** An impervious core of radius ``R`` in a Stokes shell of outer
radius ``R/\eta`` has six constants: no slip on the core and continuity at
``R/\eta``. Its averages over the whole coated sphere, ``\boldsymbol{A}`` and
``\langle\boldsymbol{r}\cdot\boldsymbol{A}\rangle``, are ratios of polynomials of degree three in
``\alpha_0`` with coefficients polynomial in ``\eta``, given by
[bignonnetIJES2022](@citet). Used alone in a self-consistent scheme, this
pattern is the effective-medium counterpart of the cell model of
[happelAIChE1958](@citet).

## 5. The schemes

With the contribution tensors ``\boldsymbol{R}_i`` of the patterns evaluated in the
appropriate reference, the schemes read as in conduction:

```math
\begin{aligned}
&\text{dilute:} &&\boldsymbol{r} = \boldsymbol{r}_m + \textstyle\sum_i \varphi_i\,\boldsymbol{R}_i(\boldsymbol{r}_m) ,\\
&\text{Mori-Tanaka:} &&\boldsymbol{r} = \boldsymbol{r}_m + \Big(\textstyle\sum_i \varphi_i\,\boldsymbol{R}_i(\boldsymbol{r}_m)\Big)\cdot\Big(\textstyle\sum_i \varphi_i\,\boldsymbol{A}_i(\boldsymbol{r}_m)\Big)^{-1} ,\\
&\text{self-consistent:} &&\textstyle\sum_i \varphi_i\,\boldsymbol{R}_i(\boldsymbol{r}) = \boldsymbol{0} ,\\
&\text{differential:} &&\dfrac{d\boldsymbol{r}}{d\Phi} = \dfrac{1}{1 - \Phi}\,\textstyle\sum_{i} \gamma_i\,\boldsymbol{R}_i(\boldsymbol{r}) ,
\end{aligned}
```

where ``m`` labels the matrix, ``\Phi`` the current fraction of inclusions and
``\gamma_i`` their filling rates. The differential scheme with impervious spheres
in a clear fluid is the iterated dilute approximation of
[wilkinsonPF1985](@citet). Maxwell's scheme requires the contribution of an
equivalent inclusion of the shape of the distribution. For an ellipsoid much
larger than the screening length, that contribution is the Darcy one, given by
the Hill tensor of conduction in the permeability form.

Three consequences hold whatever the microstructure. First, a scheme built on a
clear-fluid matrix (dilute, Mori–Tanaka) captures only the leading term of the
drag on a sparse bed. The self-consistent scheme, whose reference is the porous
effective medium itself, reproduces the correction in ``\sqrt{1 - \varphi}``
predicted for random beds of spheres
[howellsJFM1974, hinchJFM1977, kimRusselJFM1985](@cite).
Second, the drag on an impervious cylinder vanishes in a clear fluid
(Stokes' paradox): no dilute, Mori–Tanaka or differential estimate exists for
fibers in a fluid matrix. Third, the self-consistent scheme percolates at a
threshold set by the shapes chosen for the pores, ``\varphi_c = 1/3`` for spherical
grains and pores, and it does not percolate at all with cylindrical pores.

## 6. Numerical evaluation

Each formulation evaluates its closed forms in its own variable:
``\alpha_0 = R\sqrt{r_0}`` in the resistance form and ``s = \sqrt{k_0}/R = 1/\alpha_0`` in
the permeability form. Each is a rational function that stays finite at its own
end of the range. The porous closed forms subtract nearly equal terms as
``\alpha_1 \to 0``, and read ``0/0`` at ``\alpha_1 = 0``. They are rewritten with

```math
\begin{gathered}
\frac{\tanh b}{b} ,\qquad \frac{b - \tanh b}{b^3} ,\qquad \frac{b^3 - 3b + 3\tanh b}{b^5} ,\\
\frac{q(b)}{b} ,\qquad \frac{b - 2q(b)}{b^3} ,\qquad q = \frac{I_1}{I_0} ,
\end{gathered}
```

which tend to ``1``, ``1/3``, ``2/5``, ``1/2`` and ``1/8``. For ``b \le 1`` these are
summed from the Taylor series of ``\tanh`` and of ``I_1/I_0``. The coefficients
follow from the Riccati equations ``y' = 1 - y^2`` and
``q' = 1 - q/b - q^2`` that the two functions satisfy. A nearly fluid porous
inclusion therefore keeps full accuracy, as the tests check against 256-bit
evaluations of the published closed forms.

## Where to go next

The next chapter changes the time dependence rather than the physics:
[The Laplace-Carson route](@ref th-laplace-carson) reduces a non-aging
viscoelastic problem to an elastic one.

- [Brinkman inclusions](@ref man-brinkman): the inclusions, the two formulations
  and the choice of the self-consistent seed.
- [Stokes-to-Darcy permeability of porous media](@ref app-stokes-darcy): the estimates on
  granular, fibrous, tubular and double-porosity media.
- [Custom inclusions](@ref man-custom-inclusions): the contract through which
  these inclusions enter the schemes.
