# [Heterogeneities of any shape](@id th-gaussian-volume)

!!! info "Before this page"
    [Concentration and contribution tensors](@ref th-concentration) for the
    tensors a scheme needs from a heterogeneity;
    [Two-inclusion interaction tensors](@ref th-interaction), §1, for the
    Lippmann-Schwinger equation and the Green operator ``\mathbb{G}^0``;
    [Approximation by Gaussians](@ref th-gaussian-approximation) for the
    generating function ``\varphi``, the step ``h`` and the shape parameter
    ``H``.

Eshelby's result gives the concentration tensor of an ellipsoid in closed form
because the strain inside it is uniform. In a heterogeneity of another shape
(a torus, a concave pore, a particle read from a tomography) the strain varies
from point to point and has to be computed. This page solves the integral
equation of a heterogeneity by the Gaussian approximation, an approach developed
by Kanaun and coworkers ([in conduction](@cite kanaun2021),
[in elasticity](@cite markov2020)). The equation is written here for the
polarization, which gives a single symmetric system for every contrast.

## 1. One equation for every contrast

A heterogeneity of stiffness ``\mathbb{C}_1`` occupies the region ``\Omega`` of
an infinite matrix of stiffness ``\mathbb{C}_0``, under the remote strain
``\boldsymbol{E}``. With the polarization
``\boldsymbol{\tau} = (\mathbb{C}_1-\mathbb{C}_0):\boldsymbol{\varepsilon}`` in
``\Omega`` and zero outside, the strain is
([Lippmann-Schwinger equation](@ref th-interaction))

```math
\boldsymbol{\varepsilon}(\underline{x}) = \boldsymbol{E}
- \int_{\Omega}\mathbb{G}^0(\underline{x}-\underline{x}'):\boldsymbol{\tau}(\underline{x}')\,\mathrm{d}\Omega' .
```

Inside ``\Omega``, ``\boldsymbol{\varepsilon} = \mathbb{M}:\boldsymbol{\tau}`` with
``\mathbb{M} = (\mathbb{C}_1-\mathbb{C}_0)^{-1}`` the compliance of the contrast,
so that the polarization is the solution of

```math
\boxed{\;
\mathbb{M}:\boldsymbol{\tau}(\underline{x})
+ \int_{\Omega}\mathbb{G}^0(\underline{x}-\underline{x}'):\boldsymbol{\tau}(\underline{x}')\,\mathrm{d}\Omega'
= \boldsymbol{E}
\qquad (\underline{x}\in\Omega).
\;}
\tag{1}
```

The operator on the left is symmetric. ``\mathbb{M}`` is finite at both ends of
the range of contrasts: ``\mathbb{M} = \mathbb{0}`` for a rigid inclusion and
``\mathbb{M} = -\mathbb{S}_0`` for a pore, where the strain in the first and the
stress in the second would not do as unknowns. The Green operator being positive
and bounded by ``\mathbb{S}_0``, the operator of (1) is positive for a stiffer
heterogeneity and negative for a softer one.

**What the schemes need.** Let ``\mathbb{T}`` be the mean of
``\boldsymbol{\tau}`` over ``\Omega`` per unit ``\boldsymbol{E}``. The two
concentration tensors of the [custom-inclusion interface](@ref man-custom-inclusions)
follow, without the inverse of ``\mathbb{C}_1``:

```math
\langle\boldsymbol{\varepsilon}\rangle_{\Omega} = \mathbb{M}:\mathbb{T}:\boldsymbol{E},
\qquad
\langle\boldsymbol{\sigma}\rangle_{\Omega}
= \bigl(\mathbb{C}_0:\mathbb{M}:\mathbb{T}+\mathbb{T}\bigr):\boldsymbol{E} .
```

**Conduction** is the same equation with tensors of order 2: the gradient of the
potential for the strain, the conductivity for the stiffness.

## 2. The Green operator smoothed by a Gaussian

The polarization is expanded on the Gaussians of the nodes
``\underline{x}_m`` of a cubic grid of step ``h``
([Approximation by Gaussians](@ref th-gaussian-approximation)),
``\boldsymbol{\tau}(\underline{x}) = \sum_m\boldsymbol{\tau}_m\,\varphi(\underline{x}-\underline{x}_m)``
with ``\varphi(\underline{x}) = (\pi H)^{-3/2}\exp(-|\underline{x}|^{2}/(Hh^{2}))``.
The integral of (1) becomes
``\sum_m\mathbb{G}^0_h(\underline{x}-\underline{x}_m):\boldsymbol{\tau}_m``, with

```math
\mathbb{G}^0_h = \mathbb{G}^0 * \varphi ,
```

the Green operator smoothed by the Gaussian. ``\mathbb{G}^0`` is homogeneous of
degree ``-3`` and ``\varphi`` has the integral ``h^{3}``:
``\mathbb{G}^0_h`` depends on ``\underline{x}/h`` only.

In an isotropic matrix ``\mathbb{G}^0`` is made of derivatives of two
potentials, ``1/(4\pi r)`` and ``r/(8\pi)``, and smoothing the operator amounts
to smoothing them. With ``\rho = |\underline{x}|/h``,

```math
\Phi_0(\rho) = \frac{\operatorname{erf}(\rho/\sqrt{H})}{4\pi\rho},
\qquad
\Phi_2(\rho) = \frac{1}{4\pi\rho}\left[\Bigl(\frac{\rho^{2}}{2}+\frac{H}{4}\Bigr)
  \operatorname{erf}\frac{\rho}{\sqrt{H}}
  + \frac{\rho}{2}\sqrt{\frac{H}{\pi}}\;\mathrm{e}^{-\rho^{2}/H}\right],
```

the first being the potential of a Gaussian charge, the second the solution of
``\Delta\Phi_2 = \Phi_0`` that behaves as ``\rho/(8\pi)`` at infinity
(``(\rho\Phi_2)'' = \rho\,\Phi_0`` is integrated twice). Then, in grid units,

```math
\begin{aligned}
\text{conduction, } \boldsymbol{K}_0 = k_0\boldsymbol{1}:\quad
\boldsymbol{G}^0_h &= -\frac{1}{k_0}\,\operatorname{grad}\operatorname{grad}\Phi_0 ,\\
\text{elasticity, } (\mu,\nu):\quad
\mathbb{G}^0_h &= -\frac{1}{\mu}\left[
  \boldsymbol{1}\stackrel{s}{\boxtimes}\operatorname{grad}\operatorname{grad}\Phi_0
  - \frac{1}{2(1-\nu)}\,\operatorname{grad}^{4}\Phi_2\right],
\end{aligned}
```

where the first term of the bracket is symmetrized on its two pairs of indices.
The gradients of a function of ``\rho`` in the direction
``\underline{n}`` are, with ``\boldsymbol{N} = \underline{n}\otimes\underline{n}``,

```math
\operatorname{grad}\operatorname{grad}\Phi_0 = \alpha\,\boldsymbol{1}+\beta\,\boldsymbol{N},
\qquad
\alpha = \frac{\Phi_0'}{\rho},\quad \beta = \Phi_0''-\frac{\Phi_0'}{\rho},
```

and a combination of ``\boldsymbol{1}\otimes\boldsymbol{1}+2\mathbb{I}``,
``\boldsymbol{1}\otimes\boldsymbol{N}+\boldsymbol{N}\otimes\boldsymbol{1} +
2(\boldsymbol{1}\stackrel{s}{\boxtimes}\boldsymbol{N} +
\boldsymbol{N}\stackrel{s}{\boxtimes}\boldsymbol{1})`` and
``\boldsymbol{N}\otimes\boldsymbol{N}`` for the fourth gradient
(see `radial_fourth_gradient`).

Two limits check the result. At the node, the Gaussian is a spherically
symmetric source and ``\mathbb{G}^0_h(\underline{0}) = \varphi(\underline{0})\,\mathbb{P}``
with ``\mathbb{P}`` the [Hill tensor](@ref th-hill-tensors) of the sphere. Far
from the node, ``\mathbb{G}^0_h\to\mathbb{G}^0``: exactly in conduction beyond a
few steps, since a Gaussian charge is seen from outside as a point charge, and
up to a term of relative size ``H/\rho^{2}`` in elasticity.

```@example gaussvolume
using MeanFieldHomogenization
using TensND, LinearAlgebra
using Plots

G = MeanFieldHomogenization.GaussianApproximation
K₀ = TensISO{3}(1.0)
ρ = 0:12
smoothed = [G.smoothed_green_operator(K₀, (r, 0, 0), 2.0)[1, 1] for r in ρ]
exact = [get_array(green_operator_iso(K₀, [r, 0.0, 0.0]))[1, 1] for r in ρ[2:end]]
plot(ρ, smoothed; marker = :circle, xlabel = "distance to the node, in steps",
    ylabel = "component 11 along the axis 1", label = "smoothed Green operator")
plot!(ρ[2:end], exact; ls = :dash, label = "Green operator", ylims = (-0.2, 0.05))
```

## 3. The system

Equation (1) is written at the nodes that ``\Omega`` contains. The matrix is the
sum of a block ``\mathbb{M}`` on each node and of the convolution by
``\mathbb{G}^0_h``, evaluated by fast Fourier transform as for
[cracks](@ref th-gaussian-cracks); it is solved by the conjugate gradient (with
the sign changed for a softer heterogeneity), or by GMRES when the contrast is
stiffer in some directions and softer in others.

**Cells cut by the surface.** A cell covered by ``\Omega`` over a fraction
``\theta<1`` carries the polarization of a laminate of the two materials: in
parallel for a softer heterogeneity, ``\mathbb{M}/\theta``, which stays finite
for a pore; in series for a stiffer one,
``(\mathbb{M}+\mathbb{S}_0)/\theta-\mathbb{S}_0``, which stays finite for a
rigid inclusion.

**Two grids.** The concentration tensors are computed on the grids of ``n`` and
``n/2`` nodes and extrapolated linearly to ``h = 0``.

## 4. Accuracy

The Gaussians of the nodes next to the surface spread the polarization across
it, and the depolarizing field is reduced in a layer of the thickness of a few
steps. The error this leaves is larger than for a crack, and not as regular: it
is what limits the method. Against Eshelby's solution for a sphere in
conduction:

```@example gaussvolume
K₀ = TensISO{3}(2.0)
relative_error(A, Aref) = norm(get_array(A) - get_array(Aref)) / norm(get_array(Aref))
ns = (8, 12, 16, 24)
plt = plot(; xscale = :log10, yscale = :log10, xlabel = "nodes across the sphere, n",
    ylabel = "relative error on the concentration tensor", legend = :bottomleft)
for (label, k₁) in (("ten times more conducting", 20.0), ("ten times less", 0.2)), extrapolate in (false, true)
    K₁ = TensISO{3}(k₁)
    exact = gradient_gradient_concentration(Ellipsoid(1.0), K₁, K₀)
    errors = [relative_error(gradient_gradient_concentration(GaussianInclusion(Ellipsoid(1.0); n, extrapolate), K₁, K₀), exact) for n in ns]
    plot!(plt, collect(ns), errors; marker = :circle, label = label * (extrapolate ? ", two grids" : ", one grid"))
end
plt
```

With the default grid of 24 nodes the error is 1 to 3 % for contrasts between
``10^{-4}`` and ``10^{3}``, on spheres and on ellipsoids of aspect ratio down to
0.4, in conduction and in elasticity. It grows with the contrast on the stiff
side, where the fields inside the heterogeneity are less and less determined
(they are not unique in a rigid body), and so does the number of iterations:
beyond a contrast of ``10^{4}`` the solve is slow, and a contrast of
``10^{3}`` already gives the rigid limit within the error of the method.

## 5. Scope

- The reference medium is **isotropic**; the heterogeneity may be anisotropic.
  A phase is registered in a scheme with `symmetrize = IsoSymmetrize()` when
  its reference would not be.
- Derivatives with respect to the properties of the phases go through the solve
  by the implicit-function theorem, as for [cracks](@ref th-gaussian-cracks),
  §8. A derivative with respect to the shape is refused.

## Where to go next

- [Heterogeneities of any shape](@ref man-gaussian-inclusions), in the manual.
- [Finite-element inclusions](@ref man-fe-inclusions) and
  [Neural-surrogate inclusions](@ref man-neural-inclusions): the two other
  treatments of a shape with no closed form.
