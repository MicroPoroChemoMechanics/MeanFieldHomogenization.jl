# [Interacting cracks](@id th-interacting-cracks)

!!! info "Before this page"
    [Crack opening displacement and compliance](@ref th-cod-tensors) for the
    isolated crack, and [Problems of the plane](@ref th-plane-problems) for the
    crack of a plane.

The schemes of a later chapter account for the other cracks through an average
medium ([Cracks seen as fractures](@ref th-fracture-compliance)). When a few
cracks lie close to each other, the stress intensity factor at a tip depends on
where exactly the others are: two collinear cracks amplify each other at their
inner tips, two stacked cracks shield each other. This page solves the problem
of ``N`` straight cracks of a plane by the method of average tractions of
[kachanov1987](@citet), for an isotropic plane and, following
[mauge1994](@citet), for an anisotropic one.

## 1. The problem of the tractions

Crack ``k``, of center ``\underline c_k``, half-length ``l_k``, direction
``\underline m_k`` and normal ``\underline n_k``, is the segment
``\underline c_k + \xi\,\underline m_k``, ``-l_k < \xi < l_k``. Under the remote
stress ``\boldsymbol\Sigma``, the solution is the uniform field
``\boldsymbol\Sigma`` plus the field of a body whose crack faces carry the
tractions that cancel it. By linearity, this field is the sum of the fields of
each crack alone in the plane, its faces loaded by the traction
``\underline t_k(\xi)`` of the uncracked continuum on its line: the remote
traction ``\boldsymbol\Sigma\cdot\underline n_k`` plus the tractions all the other
cracks send there,

```math
\underline t_k(\xi) = \boldsymbol\Sigma\cdot\underline n_k
  + \sum_{j \neq k} \boldsymbol\sigma^{(j)}\big(\underline c_k + \xi\,\underline m_k;\ \underline t_j\big)\cdot\underline n_k ,
```

``\boldsymbol\sigma^{(j)}(\underline x;\ \underline t_j)`` the stress created at
``\underline x`` by crack ``j`` alone under the face tractions ``\underline t_j``. The
unknowns are functions: an integral equation.

## 2. The approximation of average tractions

Write ``\underline t_j = \langle\underline t_j\rangle + (\underline t_j - \langle\underline t_j\rangle)``,
``\langle\cdot\rangle`` the mean over the crack. The second part has a zero mean: it
is a self-equilibrated loading, whose field decays faster away from crack
``j``. Kachanov's approximation drops it in the field crack ``j`` sends onto the
others, keeping its uniform mean. A crack under a uniform traction has its
field in closed form, linear in the two components
``\langle t_j\rangle_n = \underline n_j\cdot\langle\underline t_j\rangle`` (opening) and
``\langle t_j\rangle_m = \underline m_j\cdot\langle\underline t_j\rangle`` (sliding). The mean
over crack ``k`` of the traction it sends defines a ``2\times 2`` block
``\boldsymbol\Lambda_{kj}``, and averaging the equation of §1 over crack ``k``
closes the problem on the means:

```math
\langle\underline t_k\rangle = \boldsymbol\Sigma\cdot\underline n_k
  + \sum_{j \neq k} \boldsymbol\Lambda_{kj}\cdot\langle\underline t_j\rangle ,
\qquad k = 1, \dots, N ,
```

a linear system of size ``2N``. Once it is solved, the traction ``\underline t_k(\xi)``
of §1, its fields sent by the means, is known along each crack, and the factors
at the tips ``\xi = \pm l`` of a crack of half-length ``l`` loaded by the
tractions ``t_n(\xi)``, ``t_m(\xi)`` are those of a crack alone (eq. 14 of
[kachanov1987](@citet)):

```math
K_{\mathrm I}^{\pm} + \mathrm i\,K_{\mathrm{II}}^{\pm}
  = \frac{1}{\sqrt{\pi l}}\int_{-l}^{l}\big(t_n(\xi) + \mathrm i\,t_m(\xi)\big)
    \sqrt{\frac{l \pm \xi}{l \mp \xi}}\,\mathrm d\xi .
```

With ``\xi = \mp l\cos\theta``, the weight becomes
``\sqrt{(l\pm\xi)/(l\mp\xi)}\,\mathrm d\xi = l\,(1 - \cos\theta)\,\mathrm d\theta``, regular on
``[0, \pi]``, and a uniform traction ``t`` gives back ``K = t\sqrt{\pi l}``.

The approximation is exact for one crack and for the means of cracks far apart;
it loses precision when the gap between two cracks becomes small against their
lengths, the neglected part of the traction being concentrated near the tips
that face each other.

## 3. The field of a loaded crack

**Isotropic plane.** In the frame ``(\underline m, \underline n)`` of a crack of
half-length ``l`` and with ``z = x + \mathrm i\,y``, the uniform face tractions
``p`` (opening) and ``\tau`` (sliding) create the stresses of Westergaard's
functions ``Z = p\,(z/\sqrt{z^2 - l^2} - 1)`` and ``Z = \tau\,(z/\sqrt{z^2 - l^2} - 1)``:

```math
\begin{aligned}
\sigma_{xx} &= p\,(\operatorname{Re}Z - y\operatorname{Im}Z') + \tau\,(2\operatorname{Im}Z + y\operatorname{Re}Z'),\\
\sigma_{yy} &= p\,(\operatorname{Re}Z + y\operatorname{Im}Z') - \tau\,y\operatorname{Re}Z',\\
\sigma_{xy} &= -p\,y\operatorname{Re}Z' + \tau\,(\operatorname{Re}Z - y\operatorname{Im}Z'),
\end{aligned}
\qquad Z' = -\frac{l^2}{(z^2 - l^2)^{3/2}} ,
```

with ``Z = z/\sqrt{z^2 - l^2} - 1`` here, the root ``\sqrt{z - l}\sqrt{z + l}`` being cut
along the crack and equal to ``z`` at infinity. On the faces ``\operatorname{Re}Z = -1``
and ``y = 0``: ``\sigma_{yy} = -p`` and ``\sigma_{xy} = -\tau``, the loading; at infinity
the field vanishes. These stresses do not depend on the elastic constants.

**Anisotropic plane.** The field is that of a free slit, the ellipse of
[The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse) flattened
(map ``z = \zeta + 1/\zeta``, of half-length 2), under the remote stress
``p\,\underline e_2\otimes\underline e_2 + \tau\,(\underline e_1\otimes\underline e_2 + \underline e_2\otimes\underline e_1)``
of the frame of the crack, minus that uniform stress: the free faces then
carry, in the difference, the tractions ``-p`` and ``-\tau``. The characteristic
roots are those of the stiffness written in the frame of the crack. The stress
on the line of a crack, ahead of its tips, is the same as in an isotropic
plane, so that collinear cracks interact as they do there [mauge1994](@cite);
any other arrangement depends on the anisotropy.

## 4. In the package

[`sif`](@ref) on a list of `center => crack` pairs, each crack a
[`RibbonCrack`](@ref), returns the factors at the two tips and the mean tractions
of each crack; the stiffness of the plane selects the field of §3. The syntax
is in [Cracks](@ref man-cracks), and the application
[Interacting cracks](@ref app-interacting-cracks) compares the method with the
exact solutions Kachanov quotes.

## Where to go next

The last page of the chapter turns to the shapes that degenerate into cracks:
[Degenerate shapes](@ref th-degenerate-shapes).
