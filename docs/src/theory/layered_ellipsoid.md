# [Layered ellipsoid — confocal layers in conduction](@id th-layered-ellipsoid)

!!! info "Before this page"
    [The Eshelby inclusion problem](@ref th-eshelby-problem) §3 for the Hill
    tensor of conduction, and [Layered spheroid](@ref th-layered-spheroid),
    which treats the spheroid by a series of harmonics.

[`LayeredEllipsoid`](@ref) is an inclusion of ``N`` isotropic layers bounded
by confocal ellipsoids of three unequal axes, in an isotropic medium, for the
conduction problem. Its mean gradients satisfy ``N`` linear relations whose
coefficients are Hill tensors of ellipsoids: no series is needed. This page
derives them. The result is that of [bonfohAMM2020](@cite), obtained there by
a recurrence on the layers; the derivation below goes through the integral
equation of the problem.

## 1. The problem

The ellipsoid ``\Omega_J``, ``J=1,\dots,N``, has the semi-axes
``\sqrt{a_i^2+\lambda_J}`` along three common axes, with
``0=\lambda_1<\dots<\lambda_N``: the ellipsoids are confocal and nested. The
layer ``V_J`` lies between ``\Omega_{J-1}`` and ``\Omega_J`` (``V_1=\Omega_1``
is the core) and has the conductivity ``k_J``; the medium outside
``\Omega_N`` has the conductivity ``k_0``. The temperature gradient
``\underline{g}=\underline{\operatorname{grad}}\,T`` tends to
``\underline{g}^0`` at infinity, and the temperature and the normal flux are
continuous across each boundary. The unknowns are the mean gradients
``\bar{\underline{g}}^K`` of the layers. With ``|\Omega|`` the volume of
``\Omega``,

```math
\varphi_K=\frac{|V_K|}{|\Omega_N|},\qquad
\omega_J=\frac{|\Omega_J|}{|\Omega_N|}=\sum_{K\le J}\varphi_K .
```

## 2. The mean gradient of an ellipsoid

Let ``\kappa`` be a conductivity, to be chosen. The flux is
``-\kappa\,\underline{g}-\underline{\tau}`` with the polarization
``\underline{\tau}=(k-\kappa)\,\underline{g}``, and its divergence vanishes:
the gradient is that of a medium of conductivity ``\kappa`` in which
``\underline{\tau}`` is prescribed. The polarization tends to
``\underline{\tau}^0=(k_0-\kappa)\,\underline{g}^0`` at infinity; a uniform
polarization of the whole space produces a uniform gradient, so that, with
``\boldsymbol{\Gamma}_\kappa=-\tfrac{1}{4\pi\kappa}\,\underline{\operatorname{grad}}\,\underline{\operatorname{grad}}\,\tfrac{1}{\lVert\underline{x}-\underline{x}'\rVert}``
the Green operator of that medium,

```math
\underline{g}(\underline{x})=\underline{g}^\star-\int\boldsymbol{\Gamma}_\kappa(\underline{x}-\underline{x}')\cdot\underline{\tau}_c(\underline{x}')\,\mathrm{d}V',
\qquad\underline{\tau}_c=\underline{\tau}-\underline{\tau}^0 ,
```

where ``\underline{g}^\star`` is a constant vector and the integral converges,
``\underline{\tau}_c`` decaying as the perturbation of the gradient. The mean
over an ellipsoid ``\Omega`` is

```math
\bar{\underline{g}}^{\Omega}=\underline{g}^\star-\frac{1}{|\Omega|}\int\boldsymbol{P}_\kappa(\Omega;\underline{x}')\cdot\underline{\tau}_c(\underline{x}')\,\mathrm{d}V',
\qquad
\boldsymbol{P}_\kappa(\Omega;\underline{x}')=\int_{\Omega}\boldsymbol{\Gamma}_\kappa(\underline{x}-\underline{x}')\,\mathrm{d}V
=-\frac{1}{4\pi\kappa}\,\underline{\operatorname{grad}}\,\underline{\operatorname{grad}}\,U_\Omega(\underline{x}') ,
```

``U_\Omega`` being the Newtonian potential of the homogeneous ellipsoid,
``U_\Omega(\underline{x}')=\int_\Omega\mathrm{d}V/\lVert\underline{x}-\underline{x}'\rVert``.
Two properties of that potential are used.

**Inside** the ellipsoid it is a polynomial of degree two:
``\boldsymbol{P}_\kappa(\Omega;\underline{x}')`` is uniform and is the Hill
tensor ``\boldsymbol{P}_\kappa(\Omega)`` of the ellipsoid.

**Outside** it is [kellogg1929](@cite)

```math
U_\Omega(\underline{x}')=\pi\,a_1a_2a_3\int_{\lambda(\underline{x}')}^{\infty}\Bigl(1-\sum_i\frac{x_i'^2}{a_i^2+s}\Bigr)\frac{\mathrm{d}s}{\Delta(s)},
\qquad\Delta(s)=\prod_i\sqrt{a_i^2+s} ,
```

where ``\lambda(\underline{x}')`` is the positive root of
``\sum_ix_i'^2/(a_i^2+\lambda)=1``. For a confocal ellipsoid of semi-axes
``\sqrt{a_i^2+\mu}`` the change of variable ``s\to s+\mu`` gives the same
integrand and the same lower limit: only the factor ``a_1a_2a_3`` changes.
The exterior potentials of two confocal ellipsoids are in the ratio of their
volumes, hence, at a point outside both,

```math
\frac{\boldsymbol{P}_\kappa(\Omega_{J+1};\underline{x}')}{|\Omega_{J+1}|}=\frac{\boldsymbol{P}_\kappa(\Omega_{J};\underline{x}')}{|\Omega_{J}|} .
```

## 3. The relation between a coating and what it surrounds

Take ``\kappa=k_{J+1}``, the conductivity of the layer ``V_{J+1}``, and
subtract the means over ``\Omega_J`` and over ``\Omega_{J+1}``. The constant
``\underline{g}^\star`` cancels, and the points ``\underline{x}'`` outside
``\Omega_{J+1}`` do not contribute, by the last relation. In ``V_{J+1}``,
``\underline{\tau}=0`` and ``\underline{\tau}_c=-\underline{\tau}^0``; in
``\Omega_J`` both tensors are uniform. Since
``\int_{V_{J+1}}\boldsymbol{P}_\kappa(\Omega_J;\underline{x}')\,\mathrm{d}V'=|\Omega_J|\,(\boldsymbol{P}_\kappa(\Omega_{J+1})-\boldsymbol{P}_\kappa(\Omega_J))``
(the integral over ``\underline{x}\in\Omega_J`` of the Green operator integrated
over ``V_{J+1}``), the terms in ``\underline{\tau}^0`` are

```math
\bigl(\boldsymbol{P}_\kappa(\Omega_{J+1})-\boldsymbol{P}_\kappa(\Omega_J)\bigr)\cdot\underline{\tau}^0
-\bigl(\boldsymbol{P}_\kappa(\Omega_{J+1})-\boldsymbol{P}_\kappa(\Omega_J)\bigr)\cdot\underline{\tau}^0=0 ,
```

and there remains

```math
\bar{\underline{g}}^{\Omega_{J+1}}-\bar{\underline{g}}^{\Omega_{J}}
=\Bigl(\frac{\boldsymbol{P}_{J+1}(\Omega_J)}{|\Omega_J|}-\frac{\boldsymbol{P}_{J+1}(\Omega_{J+1})}{|\Omega_{J+1}|}\Bigr)\cdot\int_{\Omega_J}\underline{\tau}\,\mathrm{d}V,
\qquad
\int_{\Omega_J}\underline{\tau}\,\mathrm{d}V=|\Omega_N|\sum_{K\le J}\varphi_K\,(k_K-k_{J+1})\,\bar{\underline{g}}^K ,
```

where ``\boldsymbol{P}_{J+1}`` is the Hill tensor in the medium of
conductivity ``k_{J+1}``. With
``\omega_{J+1}\,\bar{\underline{g}}^{\Omega_{J+1}}=\omega_J\,\bar{\underline{g}}^{\Omega_J}+\varphi_{J+1}\,\bar{\underline{g}}^{J+1}``
it gives the mean gradient of the coating from those of the layers it
surrounds: for ``J<N``,

```math
\bar{\underline{g}}^{J+1}=\frac{1}{\omega_J}\sum_{K\le J}\varphi_K\,\bigl[\boldsymbol{1}-(k_{J+1}-k_K)\,\boldsymbol{T}_{J+1}\bigr]\cdot\bar{\underline{g}}^K,
\qquad
\boldsymbol{T}_{J+1}=\boldsymbol{P}_{J+1}(\Omega_{J+1})-\frac{\omega_{J+1}}{\varphi_{J+1}}\bigl(\boldsymbol{P}_{J+1}(\Omega_{J+1})-\boldsymbol{P}_{J+1}(\Omega_J)\bigr) .
```

No assumption has been made on the gradient of the layers, which is not
uniform.

## 4. The whole inclusion, and the solution

With ``\kappa=k_0`` the polarization vanishes outside ``\Omega_N`` and
``\underline{g}^\star=\underline{g}^0``; the mean over ``\Omega_N`` is

```math
\sum_K\varphi_K\,\bigl[\boldsymbol{1}+(k_K-k_0)\,\boldsymbol{P}_0(\Omega_N)\bigr]\cdot\bar{\underline{g}}^K=\underline{g}^0 .
```

The relations of section 3 give every mean gradient from that of the core,
``\bar{\underline{g}}^K=\boldsymbol{r}_K\cdot\bar{\underline{g}}^1`` with
``\boldsymbol{r}_1=\boldsymbol{1}``, and this one then gives
``\bar{\underline{g}}^1``. The concentration tensors of the inclusion follow,

```math
\boldsymbol{A}=\sum_K\varphi_K\,\boldsymbol{A}_K,\qquad
\boldsymbol{B}=\sum_K\varphi_K\,k_K\,\boldsymbol{A}_K,\qquad
\boldsymbol{N}=\boldsymbol{B}-k_0\,\boldsymbol{A},
\qquad\bar{\underline{g}}^K=\boldsymbol{A}_K\cdot\underline{g}^0 ,
```

all diagonal in the axes of the ellipsoids, where the Hill tensor in a medium
of conductivity ``\kappa`` is that of the depolarization factors divided by
``\kappa``.

```@example thlayell
using MeanFieldHomogenization, TensND
k = (TensISO{3}(100.0), TensISO{3}(10.0), TensISO{3}(0.5))
ellipsoid = layered_ellipsoid_from_fractions((1.0, 2.0, 5.0), (0.5, 0.2, 0.3), k)
K₀ = TensISO{3}(1.0)
canonical_components(gradient_gradient_concentration(ellipsoid, K₀, K₀))
```

For a spheroid the result is that of the series of
[Layered spheroid](@ref th-layered-spheroid), here on a prolate core of aspect
ratio 5 with one coating:

```@example thlayell
disk = (1.0, 1.02)
axis = sqrt.(disk .^ 2 .+ 24.0)
pair = (
    LayeredEllipsoid(map((d, a) -> (d, d, a), disk, axis), k[1:2]),
    LayeredSpheroid(axis, disk, k[1:2]; Nseries = 10),
)
[canonical_components(gradient_gradient_concentration(p, K₀, K₀))[i, i] for p in pair, i in (1, 3)]
```

## 5. Scope

- **Isotropic layers, isotropic medium, perfect interfaces.** An interface
  with a resistance or a surface conductance adds a polarization carried by
  the surface, which the relations of section 3 do not contain: the confocal
  spheroid of [Layered spheroid](@ref th-layered-spheroid) treats it.
- **Conduction only.** In elasticity the Green operator also involves the
  biharmonic potential of the ellipsoid, whose exterior values are not in the
  ratio of the volumes of confocal ellipsoids: the relation of section 3 is
  then an approximation, measured in
  [Coated inclusions: closures on the mean strains, and the exact solution](@ref app-coated-estimates).
- **Confocal boundaries.** Similar ellipsoids are not confocal; for them, as
  for any other shape of the layers, the response is computed on a
  finite-element cell ([Finite-element inclusions](@ref man-fe-inclusions)).
