# [Problems of the plane](@id th-plane-problems)

!!! info "Before this page"
    [Conventions](@ref th-notation), whose conventions hold in the plane as in
    space. The tensors of dimension 2, their classes and their two pairs of
    elastic moduli are those of TensND:
    [Tensors of the plane, of space and of any dimension](@extref TensND th-dimensions).

A medium invariant along an axis, or a thin plate loaded in its plane, is
homogenized on a cell of the plane: its inclusions are sections, its properties
tensors of dimension 2, and the chain of the following pages holds with
``d=2``. This page fixes what a tensor of the plane stands for, then lists
where each problem of the plane is treated.

## 1. The cell of the plane

The axis of invariance, or the normal to the plate, is ``\underline{e}_1``; the
plane is that of ``(\underline{e}_2,\underline{e}_3)``, whose vectors are the
first and the second of the basis of the cell, with Greek indices. Two states
reduce a material of space to the plane.

- **Plane strain**, ``\varepsilon_{1i}=0``: a body long along the axis. The
  stiffness of the plane is the block of ``\mathbb{C}`` on the plane,
  ``C_{\alpha\beta\gamma\delta}``.
- **Plane stress**, ``\sigma_{1i}=0``: a thin plate, for the mean fields
  through its thickness. The three strains ``\varepsilon_{1i}`` are condensed
  out: in Kelvin–Mandel form, the stiffness of the plane is the inverse of the
  block on the plane of the compliance ``\mathbb{C}^{-1}``. When the plane is a
  plane of symmetry of the material it reduces to
  ``C_{\alpha\beta\gamma\delta}-C_{\alpha\beta11}\,C_{11\gamma\delta}/C_{1111}``.

Both are tensors of the plane, of any anisotropy, and the problem of the plane
does not depend on the state that produced it. Besides the tensor of no
symmetry, a stiffness of the plane has three classes:

| class | constants of a stiffness | invariant under |
| :--- | :--- | :--- |
| isotropic | ``k``, ``\mu`` | every rotation and reflection of the plane |
| square | 3 | the symmetries of a square, in a frame |
| orthotropic | 4, or 5 without the major symmetry | the reflections across the two axes of a frame |
| no symmetry | 6 | the half turn only |

A rotation of the plane turns every direction of it, except the half turn,
which leaves every tensor of even order unchanged: no rotation is left about a
direction of the plane, and the plane has no transversely isotropic class.

An isotropic stiffness of the plane is

```math
\mathbb{C}=2k\,\mathbb{J}+2\mu\,\mathbb{K},
\qquad
\mathbb{J}=\tfrac12\,\boldsymbol{1}\otimes\boldsymbol{1},\quad
\mathbb{K}=\mathbb{I}-\mathbb{J},
```

so that ``\mathrm{tr}\,\boldsymbol{\sigma}=2k\,\mathrm{tr}\,\boldsymbol{\varepsilon}``:
``k`` is the bulk modulus of the plane and ``\mu`` its shear modulus. In plane
strain of an isotropic material of Lamé constants ``\lambda`` and ``\mu``,
``k=\lambda+\mu``; in plane stress ``\lambda`` is replaced by
``\lambda^*=2\lambda\mu/(\lambda+2\mu)``, and ``k=\lambda^*+\mu=E/(2(1-\nu))``.

The same tensor is read with two pairs of Young's modulus and Poisson's ratio.
Read as the plane strain of a material of space, its ``E`` and ``\nu`` are
those of that material,

```math
\nu=\frac{k-\mu}{2k},\qquad E=2\mu\,(1+\nu) ;
```

read as a material of the plane, its moduli are those of a uniaxial stress in
the plane,

```math
\nu_2=\frac{k-\mu}{k+\mu},\qquad E_2=\frac{4k\mu}{k+\mu},
\qquad
\nu_2=\frac{\nu}{1-\nu},\quad E_2=\frac{E}{1-\nu^2} .
```

For the plane-stress stiffness of an isotropic material, ``E_2`` and
``\nu_2`` are the ``E`` and ``\nu`` of that material. The pair ``(k,\mu)``
does not depend on the reading.

## 2. Where each problem is treated

| Problem of the plane | Where |
| :--- | :--- |
| the Hill tensor of an ellipse, in an isotropic or an anisotropic plane | [Hill polarization tensors](@ref th-hill-plane) |
| the Green operator of an anisotropic plane, and the interaction tensors of two inclusions of the plane | [The Green operator of an anisotropic plane](@ref th-plane-green), [Two-inclusion interaction tensors](@ref th-interaction) |
| the crack of the plane, a segment, as the trace of a ribbon crack | [Crack opening displacement and compliance](@ref th-cod-tensors) |
| holes given by a conformal map, in plane strain and in a plate, in a matrix isotropic in the plane; the stress along their contour | [Holes given by a conformal map](@ref th-conformal-mapping) |
| the elliptic hole in an anisotropic plane; a hole of several terms is not solved there | [The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse), [Holes given by a conformal map](@ref th-conformal-mapping), section 10 |
| the complex potentials of plane elasticity, isotropic and anisotropic | [Complex potentials](@ref th-complex-potentials) |
| concentric rings | [Layered cylinder](@ref th-layered-cylinder) |
| conduction across the axis | [Extension to conductivity](@ref th-conductivity) |
| aging viscoelasticity, in plane strain and in plane stress | [Plane strain and plane stress](@ref th-visco-plane-states) |
| the schemes | [Homogenization schemes](@ref th-homogenization), whose formulas hold in any dimension |

The syntax, and what each inclusion accepts in the plane, are in the manual
page [Plane problems](@ref man-plane-problems).

## Where to go next

The foundations start with
[The Eshelby inclusion problem](@ref th-eshelby-problem), where the Hill tensor
``\mathbb{P}`` is defined, in space and in the plane alike.
