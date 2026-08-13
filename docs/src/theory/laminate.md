# [Periodic multilayer — the laminate cell](@id th-laminate)

A **laminate** is a periodic stack of parallel layers: a unit cell of
*periodic* homogenization, with no matrix, no auxiliary Eshelby problem and
no reference medium. Its effective behavior is **exact and in closed form**,
which sets it apart from every other page of this section: the schemes built
on the Eshelby problem ([Mori-Tanaka](@ref th-homogenization), the
[self-consistent](@ref th-homogenization) family, the
[differential](@ref th-differential-scheme) scheme, …) describe *random*
morphologies and produce *estimates*; the laminate solution **is** the answer.

That makes it useful twice over: as a model of genuinely stratified materials
(coatings, interfacial transition zones, bedded media), and as an exact
reference against which bounds and estimates can be calibrated — the laminate
saturates the Voigt bound in the plane of the layers and the Reuss bound
across them, simultaneously.

The derivation below is written for a general anisotropic stack, in
elasticity and in transport, with the localization tensors and the imperfect
interfaces.

## Setting

Layers are bounded by parallel planes of common unit normal ``\underline{n}``.
The vector plane ``\mathcal{P}\perp\underline{n}`` is spanned by orthonormal
``\underline{\ell}, \underline{m}``, with
``(\underline{\ell},\underline{m},\underline{n})`` positively oriented. The
geometry and the materials are invariant in every direction of ``\mathcal{P}``
and periodic of period ``L`` along ``\underline{n}``. One period contains
``N`` layers ``\mathcal{L}_i``, of uniform stiffness ``\mathbb{C}_i``,
thickness ``h_i`` and volume fraction ``f_i = h_i/L``.

```@setup laminate
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example laminate
plotly_scene(laminate_traces([0.30, 0.15, 0.40, 0.15]); uid = "th-laminate-cell",
    height = 420, title = "One period: four layers of thickness hᵢ, common normal n̂")
```

Only ``\underline{n}`` and the thicknesses matter — the picture is drawn from the
volume fractions ``f_i = h_i/L`` and would be unchanged by any rotation within
``\mathcal{P}``. That invariance is what collapses the problem to one dimension,
and it is why the laminate is solved **exactly** rather than estimated.

The cell is loaded by a macroscopic strain ``\boldsymbol{E}``:

```math
\begin{aligned}
\underline{\mathrm{div}}\,\boldsymbol{\sigma} &= \underline{0},
&\qquad
\boldsymbol{\sigma} &= \mathbb{C}_i : \boldsymbol{\varepsilon}
  \quad (\mathcal{L}_i), \\
\underline{u} &= \boldsymbol{E}\cdot\underline{x} + \tilde{\underline{u}},
&\qquad
\tilde{\underline{u}}(\underline{x}+L\underline{n}) &= \tilde{\underline{u}}(\underline{x}),
\end{aligned}
```

with ``\tilde{\underline{u}}`` continuous and piecewise differentiable, and
the traction ``\boldsymbol{\sigma}\cdot\underline{n}`` periodic. The
macroscopic stress is the average
``\boldsymbol{\Sigma} = \tfrac{1}{L}\int_0^L
\boldsymbol{\sigma}(\underline{x}+z\underline{n})\,\mathrm{d}z``.

## The two continuity conditions

Invariance in the plane makes every field depend on
``\underline{x}\cdot\underline{n}`` alone, so ``\underline{\mathrm{div}}\,
\boldsymbol{\sigma}=\underline{0}`` forces ``\boldsymbol{\sigma}`` to be
uniform between two consecutive interfaces. Searching for a stress uniform in
each layer, the whole problem reduces to two statements.

**Traction continuity.** The stress vector is continuous across every plane,

```math
\underline{T} = \boldsymbol{\Sigma}\cdot\underline{n}
              = \boldsymbol{\sigma}_i\cdot\underline{n},
\qquad \forall\, i .
```

**Strain compatibility.** Continuity of ``\underline{u}`` leaves each layer
strain free only through a rank-one symmetric (Hadamard) term,

```math
\boldsymbol{\varepsilon}_i = \boldsymbol{E}
  + \underline{a}_i \stackrel{s}{\otimes} \underline{n},
\qquad \forall\, i ,
```

for ``N`` unknown vectors ``\underline{a}_i``. Equivalently: **the in-plane
components of the strain are continuous** and equal to those of
``\boldsymbol{E}``.

The two conditions are complementary — three stress components are prescribed
across the layers, three strain components are prescribed within them — and
that split is the whole content of the problem.

## In-plane and out-of-plane subspaces

The space of symmetric second-order tensors splits into the tensors *without*
``\underline{n}`` and those *with* it. With
``\boldsymbol{p} = \boldsymbol{1}-\underline{n}\otimes\underline{n}`` the
in-plane projector, the two order-4 projectors are

```math
\Pi^{\mathcal{I}} = \boldsymbol{p}\stackrel{s}{\boxtimes}\boldsymbol{p},
\qquad
\Pi^{\mathcal{O}} = \mathbb{I} - \Pi^{\mathcal{I}} ,
```

that is, ``\Pi^{\mathcal{O}} = \mathbb{W}_1+\mathbb{W}_6`` and
``\Pi^{\mathcal{I}} = \mathbb{W}_2+\mathbb{W}_5`` in the
[Walpole basis](@ref th-notation) of axis ``\underline{n}``.

In Kelvin-Mandel components written in the frame
``(\underline{\ell},\underline{m},\underline{n})``, this split is a **pure
index partition** — the ``\sqrt2`` weights it would otherwise require are
already carried by the Mandel basis:

```math
\mathcal{I} = \{1,2,6\}
  \;\leftrightarrow\; (\ell\ell,\; mm,\; \sqrt2\,\ell m),
\qquad
\mathcal{O} = \{3,4,5\}
  \;\leftrightarrow\; (nn,\; \sqrt2\,mn,\; \sqrt2\,\ell n) .
```

Writing any stiffness in ``2\times2`` blocks on that partition,

```math
\mathrm{Mat}(\mathbb{C}) =
\begin{pmatrix}
  C_{\mathcal{II}} & C_{\mathcal{IO}} \\
  C_{\mathcal{OI}} & C_{\mathcal{OO}}
\end{pmatrix},
```

the continuity conditions read: ``\varepsilon_{\mathcal{I}}`` is the same in
every layer, and ``\sigma_{\mathcal{O}}`` is the same in every layer.

!!! note "One index reversal to be aware of"
    The out-of-plane Mandel slots ``(3,4,5)`` correspond to the index pairs
    ``((3,3),(2,3),(1,3))``, hence to the acoustic-tensor indices
    ``K_{ab} = C_{3a3b}`` in the **reversed** order ``(3,2,1)``, with the
    Mandel weights ``s = (1,\sqrt2,\sqrt2)``:
    ``\mathrm{Mat}(\mathbb{C})[\mathcal{O}_a,\mathcal{O}_b]
    = s_a s_b K_{\pi(a)\pi(b)}``. The reversal is invisible for isotropic
    layers, where ``\boldsymbol{K}`` is diagonal — which is exactly why the
    implementation guards it with a test on a *triclinic* stiffness.

## Layer tensors

Injecting ``\boldsymbol{\varepsilon}_i`` into the constitutive law and
contracting with ``\underline{n}`` gives the traction in terms of
``\underline{a}_i`` through the **acoustic tensor**
``\boldsymbol{K}_i = \underline{n}\cdot\mathbb{C}_i\cdot\underline{n}``:

```math
\underline{T} = \underline{n}\cdot\mathbb{C}_i : \boldsymbol{E}
  + \boldsymbol{K}_i\cdot\underline{a}_i ,
\qquad
\underline{a}_i = \left(\boldsymbol{K}_i^{-1}\stackrel{s}{\otimes}\underline{n}\right)
  : \left(\boldsymbol{\Sigma}-\mathbb{C}_i:\boldsymbol{E}\right) .
```

Hence the Hadamard term is generated by an order-4 tensor,

```math
\underline{a}_i \stackrel{s}{\otimes} \underline{n}
 = \mathbb{P}_i : (\boldsymbol{\Sigma}-\mathbb{C}_i:\boldsymbol{E}),
\qquad
\boxed{\;\mathbb{P}_i
 = \underline{n}\stackrel{s}{\otimes}\boldsymbol{K}_i^{-1}
   \stackrel{s}{\otimes}\underline{n}\;}
```

``\mathbb{P}_i`` is the **Hill polarization tensor of a flat inclusion**: the
limit of the Hill tensor of an ellipsoid embedded in a medium of stiffness
``\mathbb{C}_i`` as its smallest aspect ratio tends to zero (see
[Hill tensors](@ref th-hill-tensors)). Having minor and major symmetries, it operates
**only within out-of-plane second-order tensors**. The companion tensor

```math
\mathbb{Q}_i = \mathbb{C}_i - \mathbb{C}_i : \mathbb{P}_i : \mathbb{C}_i
```

is the second Hill tensor, and operates only within in-plane tensors.

In block form these two statements become identities, which is the cleanest
way to see them:

```math
\mathrm{Mat}(\mathbb{P}_i) =
\begin{pmatrix} 0 & 0 \\ 0 & C_{\mathcal{OO}}^{-1}\end{pmatrix},
\qquad
\mathrm{Mat}(\mathbb{Q}_i) =
\begin{pmatrix}
  C_{\mathcal{II}} - C_{\mathcal{IO}}C_{\mathcal{OO}}^{-1}C_{\mathcal{OI}}
  & 0 \\ 0 & 0
\end{pmatrix} .
```

The out-of-plane block of ``\mathbb{P}_i`` is exactly the **inverse of the
out-of-plane block of ``\mathbb{C}_i``**, and ``\mathbb{Q}_i`` is exactly the
in-plane **Schur complement** of ``\mathbb{C}_i``. Two consequences used
throughout: ``\mathbb{P}:\mathbb{C}:\mathbb{P} = \mathbb{P}`` and
``\mathbb{Q}:\mathbb{P} = 0``.

## [The pseudo-inverse](@id th-laminate-pinv)

Averaging the compatibility condition over the cell, ``\langle
\boldsymbol{\varepsilon}\rangle = \boldsymbol{E}`` requires
``\sum_i f_i\,\underline{a}_i\stackrel{s}{\otimes}\underline{n} = \boldsymbol{0}``,
that is

```math
\langle\mathbb{P}\rangle : \boldsymbol{\Sigma}
  = \langle\mathbb{P}:\mathbb{C}\rangle : \boldsymbol{E},
\qquad
\langle\,\cdot\,\rangle = \sum_{i=1}^N f_i\,(\cdot)_i .
```

This determines only the *out-of-plane* part of ``\boldsymbol{\Sigma}``, and
for a good reason: ``\langle\mathbb{P}\rangle`` is supported on the
out-of-plane subspace, so it is not invertible. What is needed is its
**Moore-Penrose pseudo-inverse**, which for an out-of-plane-supported tensor
is simply the inverse of that block, embedded back:

```math
\mathrm{Mat}(\langle\mathbb{P}\rangle^{\dagger}) =
\begin{pmatrix} 0 & 0 \\ 0 & P_{\mathcal{OO}}^{-1}\end{pmatrix},
\qquad
\langle\mathbb{P}\rangle^{\dagger} : \langle\mathbb{P}\rangle
  = \Pi^{\mathcal{O}} .
```

It exists as soon as ``P_{\mathcal{OO}}`` is invertible, that is as soon as
every acoustic tensor ``\boldsymbol{K}_i`` is definite.

!!! note "Never a generic pinv"
    The implementation does not call `LinearAlgebra.pinv`: the pseudo-inverse
    is the ordinary inverse of a ``3\times3`` block (a scalar reciprocal in
    transport), taken in closed form. An SVD would be differentiable by
    neither `ForwardDiff` nor a symbolic backend, and would in any case be
    wasted on an exactly-rank-3 input.

## Effective stiffness

Substituting back,

```math
\boxed{\;
\mathbb{C}^{\hom} = \langle\mathbb{Q}\rangle
  + \langle\mathbb{C}:\mathbb{P}\rangle
    : \langle\mathbb{P}\rangle^{\dagger}
    : \langle\mathbb{P}:\mathbb{C}\rangle \; }
```

Written on the ``\mathcal{I}/\mathcal{O}`` partition, this collapses to the
form of [backus1962](@cite), which is what the implementation evaluates —
four block products and two ``3\times3`` inversions per layer, no
factorization anywhere:

```math
\begin{aligned}
C^{\hom}_{\mathcal{OO}} &= \big\langle C_{\mathcal{OO}}^{-1}\big\rangle^{-1}, \\
C^{\hom}_{\mathcal{IO}} &= \big\langle C_{\mathcal{IO}}C_{\mathcal{OO}}^{-1}\big\rangle\,
                            C^{\hom}_{\mathcal{OO}}, \\
C^{\hom}_{\mathcal{OI}} &= C^{\hom}_{\mathcal{OO}}\,
                            \big\langle C_{\mathcal{OO}}^{-1}C_{\mathcal{OI}}\big\rangle, \\
C^{\hom}_{\mathcal{II}} &= \big\langle C_{\mathcal{II}}
                            - C_{\mathcal{IO}}C_{\mathcal{OO}}^{-1}C_{\mathcal{OI}}\big\rangle
   + \big\langle C_{\mathcal{IO}}C_{\mathcal{OO}}^{-1}\big\rangle\,
     C^{\hom}_{\mathcal{OO}}\,
     \big\langle C_{\mathcal{OO}}^{-1}C_{\mathcal{OI}}\big\rangle .
\end{aligned}
```

Two of these are **exact bound saturations**, valid for arbitrary anisotropy:

```math
\big(\underline{n}\cdot\mathbb{C}^{\hom}\cdot\underline{n}\big)^{-1}
  = \sum_i f_i \big(\underline{n}\cdot\mathbb{C}_i\cdot\underline{n}\big)^{-1}
\quad\text{(Reuss, out of plane)},
```

```math
\mathrm{Schur}_{\mathcal{I}}(\mathbb{C}^{\hom})
  = \sum_i f_i\, \mathrm{Schur}_{\mathcal{I}}(\mathbb{C}_i)
\quad\text{(Voigt, in plane)} .
```

They are the two oracles the test suite is built on, and they remain exact
when imperfect interfaces are added — each picking up one interface family
and ignoring the other.

### Symmetry of the result

The effective tensor is in general **monoclinic** about ``\underline{n}``: for
layers of arbitrary anisotropy there is no more symmetry than the geometry
itself provides.

It is **exactly** transversely isotropic about ``\underline{n}`` in one case
only: when every layer is isotropic, or transversely isotropic *about that
very axis* — the TI tensors of a common axis being closed under product and
inversion. Any other class breaks it, including an orthotropic layer whose
axes happen to coincide with the laminate frame (orthotropic is not TI in the
plane), and a TI layer whose axis is not ``\underline{n}``.

The implementation decides this *structurally*, from the declared symmetry
classes of the layers, never by testing the output numerically: the conclusion
is then exact and survives symbolic and `Dual` element types alike. It is also
deliberately conservative — a transversely isotropic but **non-major-symmetric**
layer (`TensTI{4,T,8}`, what the exact rotation-group average produces, with
``\ell_3 \neq \ell_4`` and the antisymmetric azimuthal couplings) does *not*
qualify, because the five-coefficient Walpole read-off would silently discard
that content. Such a laminate returns the generic tensor, losslessly.

### Bilayer of isotropic layers

For ``N=2`` isotropic layers of Lamé coefficients ``(\lambda_i,\mu_i)`` the
formulas above reduce to the classical long-wave average of
[backus1962](@cite):

```math
C^{\hom}_{3333} = \Big\langle \tfrac{1}{\lambda+2\mu}\Big\rangle^{-1},
\qquad
C^{\hom}_{2323} = \Big\langle \tfrac{1}{\mu}\Big\rangle^{-1},
\qquad
C^{\hom}_{1212} = \big\langle \mu \big\rangle,
```

```math
C^{\hom}_{1133} = \Big\langle \tfrac{1}{\lambda+2\mu}\Big\rangle^{-1}
                  \Big\langle \tfrac{\lambda}{\lambda+2\mu}\Big\rangle,
\qquad
C^{\hom}_{1111} = \Big\langle \tfrac{4\mu(\lambda+\mu)}{\lambda+2\mu}\Big\rangle
  + \Big\langle \tfrac{1}{\lambda+2\mu}\Big\rangle^{-1}
    \Big\langle \tfrac{\lambda}{\lambda+2\mu}\Big\rangle^{2} .
```

The out-of-plane response is a harmonic (Reuss) mean, the in-plane shear an
arithmetic (Voigt) one — the two saturations above, read off a closed form.
`scripts/38_laminate_symbolic.jl` derives these from the code itself, with
`SymPy`.

## Localization

Since ``\boldsymbol{\Sigma} = \mathbb{C}^{\hom}:\boldsymbol{E}``, the layer
strain follows directly:

```math
\boldsymbol{\varepsilon}_i = \mathbb{A}_i : \boldsymbol{E},
\qquad
\mathbb{A}_i = \mathbb{I} + \mathbb{P}_i : (\mathbb{C}^{\hom}-\mathbb{C}_i),
\qquad
\sum_i f_i\,\mathbb{A}_i = \mathbb{I},
```

and the layer stress from
``\mathbb{B}_i = \mathbb{C}_i : \mathbb{A}_i : (\mathbb{C}^{\hom})^{-1}``,
with ``\sum_i f_i\,\mathbb{B}_i = \mathbb{I}``. Because ``\mathbb{P}_i`` is
out-of-plane, ``\mathbb{A}_i`` has an in-plane block equal to the identity and
a vanishing in-plane/out-of-plane coupling — the macroscopic in-plane strain
reaches every layer unchanged, which is the compatibility condition read
backwards. (These tensors are absent from the original note.)

## Transport

The transposition is immediate: the in-plane gradient is continuous, the
normal flux is continuous. With ``\boldsymbol{K}_i`` the conductivity (or
diffusivity, or permeability) of layer ``i``, the partition of a
second-order property in the frame ``(\underline{\ell},\underline{m},
\underline{n})`` is ``\mathcal{I}=\{1,2\}`` / ``\mathcal{O}=\{3\}`` and the
whole algebra carries over with the **one-dimensional** out-of-plane
subspace, where the pseudo-inverse is a scalar reciprocal:

```math
\frac{1}{k^{\hom}_{nn}} = \sum_i \frac{f_i}{k_{i,nn}},
\qquad
K^{\hom}_{\mathcal{II}} = \sum_i f_i\,
  \Big(K_{i,\mathcal{II}} - \frac{K_{i,\mathcal{I}n}K_{i,n\mathcal{I}}}{k_{i,nn}}\Big)
  + \ldots
```

— series across the layers, parallel within them.

## [Imperfect interfaces](@id th-laminate-interfaces)

The four interface models of the [layered sphere](@ref th-layered-sphere) are
reused unchanged; a planar interface is simply the curvature-free case, and
the algebra collapses to **two additive terms**. They enter with the weight
``1/L``: an interface *density*. At fixed volume fractions, doubling the
period halves the correction, and ``L\to\infty`` recovers perfect bonding —
this is the size effect that makes the absolute period, not just the
fractions, physically meaningful.

| | primal (field jump) | dual (surface stiffness) |
| :--- | :--- | :--- |
| elasticity | `SpringInterface(kn, kt)` | `MembraneInterface(κs, μs)` |
| transport | `KapitzaInterface(ρ)` | `SurfaceConductiveInterface(ks)` |

Unlike the spherical case, **nothing forces these to be isotropic**. The
spherical-harmonic recurrence of the layered sphere only closes if the jump
conditions share the symmetry of the geometry, which is why its interfaces
carry two scalars each. A plane has a normal and an arbitrary in-plane
texture, so ``\boldsymbol{\mathcal{K}}`` may be any symmetric second-order
compliance and ``\mathbb{C}^{s}`` any 2-D surface stiffness (six independent
coefficients) — the formulas below are written for the general case, and the
implementation provides both a scalar and a tensor-valued type per family. The
primal *transport* condition ``[\![T]\!] = \rho\,q_n`` relates two scalars
and is already general.

### Primal: a jump of the field

A spring interface imposes a displacement jump driven by the traction, which
stays continuous:

```math
[\![\underline{u}]\!] = \boldsymbol{\mathcal{K}}\cdot
  (\boldsymbol{\sigma}\cdot\underline{n}) ,
```

where ``\boldsymbol{\mathcal{K}}`` is a symmetric second-order **compliance**
tensor. It is the limit of a layer of vanishing thickness whose out-of-plane
compliance stays finite and whose in-plane stiffness vanishes, so it
contributes to ``\langle\mathbb{P}\rangle`` **and to nothing else**:

```math
\langle\mathbb{P}\rangle \;\longleftarrow\;
  \sum_i f_i\,\mathbb{P}_i + \frac{1}{L}\sum_j \mathbb{P}^{\rm int}_j ,
\qquad
\mathbb{P}^{\rm int} = \underline{n}\stackrel{s}{\otimes}
  \boldsymbol{\mathcal{K}}\stackrel{s}{\otimes}\underline{n} .
```

The out-of-plane oracle therefore becomes, still exactly,

```math
\big(\underline{n}\cdot\mathbb{C}^{\hom}\cdot\underline{n}\big)^{-1}
  = \sum_i f_i \big(\underline{n}\cdot\mathbb{C}_i\cdot\underline{n}\big)^{-1}
  + \frac{1}{L}\sum_j \boldsymbol{\mathcal{K}}_j ,
```

while the in-plane oracle is left untouched. Limits: ``k\to0`` recovers
perfect bonding, ``k\to\infty`` decouples the layers
(``\underline{n}\cdot\mathbb{C}^{\hom}\cdot\underline{n}\to\boldsymbol{0}``).
The same statement in transport reads
``1/k^{\hom}_{nn} = \sum_i f_i/k_{i,nn} + \sum_j \rho_j/L`` — the interfacial
(Kapitza) resistances simply add to the series law.

### Dual: a surface stiffness

Here the flat geometry does something the sphere does not. A Gurtin-Murdoch
membrane carries a surface stress
``\boldsymbol{\sigma}^s = \lambda_s\,\mathrm{tr}(\boldsymbol{\varepsilon}^s)
\boldsymbol{p} + 2\mu_s\,\boldsymbol{\varepsilon}^s``, and the traction jump
it produces is ``[\![\boldsymbol{\sigma}\cdot\underline{n}]\!] =
-\mathrm{div}_s\,\boldsymbol{\sigma}^s``. On a **plane** interface with a
uniform in-plane strain this divergence vanishes: *there is no traction jump
at all*. The surface stress is driven by the in-plane strain, which is
continuous and equal to ``\boldsymbol{E}``, so it adds straight to the
macroscopic stress:

```math
\mathbb{C}^{\hom} \;\longleftarrow\;
  \mathbb{C}^{\hom} + \frac{1}{L}\sum_j \mathbb{C}^{s}_j ,
```

acting in the in-plane block alone. With ``\kappa_s = \lambda_s+\mu_s`` the
surface dilatation modulus (the convention of `LayeredSpheres` and of Echoes'
`DUALDISC`), the in-plane Mandel block of ``\mathbb{C}^s`` is

```math
\begin{pmatrix}
\kappa_s+\mu_s & \kappa_s-\mu_s & 0\\
\kappa_s-\mu_s & \kappa_s+\mu_s & 0\\
0 & 0 & 2\mu_s
\end{pmatrix},
```

so ``C^{\hom}_{1212}`` gains exactly ``\mu_s/L`` and the out-of-plane response
is untouched. In transport, a highly conductive surface layer adds
``k_s(\boldsymbol{1}-\underline{n}\otimes\underline{n})/L`` to the in-plane
conductivity.

The two families are therefore *complementary*: the primal one moves the
out-of-plane law and leaves the in-plane one alone, the dual one does the
reverse. That is what makes the laminate the sharpest available check of the
package's interface conventions.

## Ageing viscoelasticity

The whole solution is products of Kelvin-Mandel matrices and one inversion
restricted to the out-of-plane subspace. Replacing each scalar by a
discretized Volterra operator therefore transposes it verbatim to
[ageing linear viscoelasticity](@ref th-viscoelasticity): the matrices become
``(6n\times6n)`` (resp. ``(3n\times3n)``) in ``n`` time blocks, products
become Volterra products, and the ``3\times3`` (resp. scalar) inversion
becomes `volterra_inverse` on the out-of-plane restriction. The elastic limit
— a Heaviside law per layer — returns the elastic laminate in every diagonal
time block, and the two exact saturations survive the transposition.

## Relation to the rest of the package

- ``\mathbb{P}_i`` is the flat limit of the [Hill tensor](@ref th-hill-tensors): a
  laminate is what a stack of infinitely flat inclusions becomes when they
  fill space.
- The laminate is a **cell**, not an inclusion: it is homogenized, not
  embedded. Embedding a laminated inclusion in a matrix would require *its*
  Hill tensor, which is a separate problem.
- As an [`AbstractHomogenizationCell`](@ref) it takes part in the multiscale
  chain like any `RVE` — see [Multiscale models](@ref man-multiscale).

## References

The isotropic bilayer closed form is [backus1962](@cite); the flat-inclusion
limit of the Hill tensor is discussed in [barthelemyIJES2021](@cite); the
interface models are those of [herveLuanco2014](@cite), specialized to a
plane.
