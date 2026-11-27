# [Cracks seen as fractures](@id th-fracture-compliance)

!!! info "Before this page"
    [Crack opening displacement and compliance](@ref th-cod-tensors), which define the
    compliance contribution ``\mathbb H`` of a crack, and the scheme pages of this
    chapter, whose estimates are decomposed here.

A family of parallel cracks changes the compliance of a solid as a set of
parallel fractures does, and seismology describes such a set by two numbers per
unit length, its normal and tangential compliances. This page writes the excess
compliance of a crack family under each scheme, reads the compliance of a
fracture set on it, and adds up sets of fractures; the waves of the resulting
medium are those of [TensND](@extref TensND th-plane-waves).

## 1. The excess compliance of a crack family

A scheme gives one effective stiffness ``\mathbb C^{\mathrm{hom}}``. Writing its
compliance as a solid part plus one term per crack family ``i`` of density
``\varepsilon_i``,

```math
\mathbb S^{\mathrm{hom}} = \mathbb S^{\mathrm{solid}}
  + \sum_i \frac{4\pi}{3}\,\varepsilon_i\,\mathbb S_i ,
```

defines the excess compliance ``\Delta\mathbb S_i = (4\pi/3)\,\varepsilon_i\,\mathbb S_i``
of family ``i``. The decomposition is not unique in general: it is read on the
body of each scheme, at its own estimate.

- **Mori–Tanaka**, a matrix ``\mathbb C_s`` and cracks only:
  ``\mathbb C^{\mathrm{MT}} = \mathbb C_s:(\mathbb I + \mathbb H:\mathbb C_s)^{-1}``
  with ``\mathbb H = \sum_i (4\pi/3)\,\varepsilon_i\,\mathbb H_i``, hence
  ``\mathbb S^{\mathrm{MT}} = \mathbb S_s + \mathbb H``: each family adds its own
  ``\mathbb H_i``, computed in the matrix.
- **Stress-driven dilute scheme**: the compliance contributions add up in the
  matrix, ``\mathbb S^{\mathrm{hom}} = \mathbb S_0 + \sum_j \Delta\mathbb S_j``, and the
  same ``\mathbb H_i`` comes out.
- **Dilute, Maxwell, Ponte Castañeda–Willis**: these schemes read
  ``\mathbb C^{\mathrm{hom}} = \mathbb C_0 + \boldsymbol\Sigma:\mathbb R`` with
  ``\boldsymbol\Sigma = \sum_j \mathbb N_j`` the contributions of the phases per unit
  volume and ``\mathbb R = (\mathbb I - \mathbb P_d:\boldsymbol\Sigma)^{-1}``
  (``\mathbb P_d = \mathbb 0`` for the dilute scheme). Since
  ``\mathbb S^{\mathrm{hom}} - \mathbb S_0 = \mathbb S^{\mathrm{hom}}:(\mathbb C_0 - \mathbb C^{\mathrm{hom}}):\mathbb S_0``,

  ```math
  \mathbb S^{\mathrm{hom}} = \mathbb S_0 - \sum_j \mathbb S^{\mathrm{hom}}:\mathbb N_j:\mathbb R:\mathbb S_0 ,
  ```

  and a crack family, ``\mathbb N_i = -(4\pi/3)\,\varepsilon_i\,\mathbb C_0:\mathbb H_i:\mathbb C_0``, has
  ``\mathbb S_i = \mathbb S^{\mathrm{hom}}:\mathbb C_0:\mathbb H_i:\mathbb C_0:\mathbb R:\mathbb S_0``.
- **Self-consistent scheme**: the decomposition of
  [`crack_family_compliances`](@ref MeanFieldHomogenization.Schemes.crack_family_compliances)
  rests on its fixed point, ``\mathbb S_i = \mathbb H_i(\mathbb C^{\mathrm{hom}}):\mathbb C^{\mathrm{hom}}:\langle\mathbb C:\mathbb A\rangle^{-1}``.

## 2. The compliance of a fracture set

A set of parallel fractures of normal ``\underline n``, one every length ``L``,
each of compliance ``\mathbb K`` (the jump of displacement per unit traction across
it), changes the compliance by [schoenberg1995](@cite)

```math
\Delta\mathbb S = \operatorname{sym}(\underline n\otimes\boldsymbol Z\otimes\underline n),
\qquad
\boldsymbol Z = \mathbb K/L ,
```

the symmetrization being over the minor symmetries:
``\Delta S_{ijkl} = \tfrac14(n_i Z_{jk} n_l + n_j Z_{ik} n_l + n_i Z_{jl} n_k + n_j Z_{il} n_k)``.
One set is exactly the periodic laminate of layers of thickness ``L`` with a
spring interface of compliance ``\mathbb K`` ([Laminates](@ref th-laminate)).

Conversely, ``\boldsymbol Z`` is read on ``\Delta\mathbb S``. With
``\boldsymbol M = \underline n\cdot\Delta\mathbb S\cdot\underline n``
(``M_{jk} = n_i\,\Delta S_{ijkl}\,n_l``), the four terms above give

```math
\boldsymbol M = \tfrac14\big(\boldsymbol Z + \underline n\otimes(\boldsymbol Z\cdot\underline n)
  + (\boldsymbol Z\cdot\underline n)\otimes\underline n + Z_N\,\underline n\otimes\underline n\big),
\qquad Z_N = \underline n\cdot\boldsymbol Z\cdot\underline n ,
```

whence ``\underline n\cdot\boldsymbol M\cdot\underline n = Z_N``,
``\boldsymbol Z\cdot\underline n = 2\,\boldsymbol M\cdot\underline n - Z_N\,\underline n`` and

```math
\boldsymbol Z = 4\,\boldsymbol M - \underline n\otimes(\boldsymbol Z\cdot\underline n)
  - (\boldsymbol Z\cdot\underline n)\otimes\underline n - Z_N\,\underline n\otimes\underline n .
```

A set invariant by rotation about its normal has
``\boldsymbol Z = Z_N\,\underline n\otimes\underline n + Z_T\,(\boldsymbol 1 - \underline n\otimes\underline n)``,
with ``Z_T = (\operatorname{tr}\boldsymbol Z - Z_N)/2``. When the excess compliance of a
crack family has the form ``\operatorname{sym}(\underline n\otimes\boldsymbol Z\otimes\underline n)``,
which is the case of flat cracks, whose contribution
``\mathbb H = k\,\underline n\stackrel{s}{\otimes}\boldsymbol B\stackrel{s}{\otimes}\underline n`` has it
([Crack opening displacement and compliance](@ref th-cod-tensors)), these formulas read on
``\Delta\mathbb S_i`` the fracture compliance ``\boldsymbol Z_i`` of the family *as the
scheme sees it*: the stress-driven dilute scheme and Mori–Tanaka give that of
isolated cracks in the matrix, the other schemes one corrected by their own
estimate of the surroundings of the cracks.

## 3. Several sets, and three orthogonal ones

The changes of several sets of fractures add up,
``\mathbb S = \mathbb S_b + \sum_s \operatorname{sym}(\underline n_s\otimes\boldsymbol Z_s\otimes\underline n_s)``,
the sets not interacting. For three orthogonal sets of normals the vectors
``\underline e_i`` of a frame, each invariant about its normal, a set ``i`` changes
in that frame ``\Delta S_{iiii}`` by ``Z_N^i`` and, for ``j \neq i``, ``\Delta S_{ijij}`` by
``Z_T^i/4`` (the formula of §2 with ``\underline n = \underline e_i``), and no other
component. Hence

```math
Z_N^i = \Delta S_{iiii},
\qquad
4\,\Delta S_{ijij} = Z_T^i + Z_T^j ,
\qquad
Z_T^i = 2\,(\Delta S_{ijij} + \Delta S_{ikik} - \Delta S_{jkjk}) ,
```

``(i, j, k)`` a permutation of ``(1, 2, 3)``: the inversion of
[rasolofosaonGJI2000](@citet), eq. (3), who write it in the two-index notation
of compliances, ``\Delta S_{44} = 4\,\Delta S_{2323} = Z_T^2 + Z_T^3``. The other
components of ``\Delta\mathbb S`` in the frame (the couplings
``\Delta S_{iijj}``, and anything outside the orthotropic pattern) are left out.
The frame is that of the best orthotropic approximation of the cracked medium
([`project`](@extref TensND TensND.project) with `:ORTHO` in TensND).

## 4. Mass density

The waves need the mass density of the cell, the volume average of the
densities of its phases: the Voigt average of the phase property ``\rho``. A crack
family, of no volume, contributes nothing.

## 5. In the package

[`fracture_compliance`](@ref) reads ``\boldsymbol Z``, ``Z_N`` and ``Z_T`` of a crack family
under a scheme, [`slip_compliance`](@ref) the formula of §2 on a given change of
compliance, [`linear_slip_change`](@ref) and [`fractured_stiffness`](@ref) build the
medium of §3, [`orthogonal_fracture_compliances`](@ref) inverts three orthogonal
sets, and [`average_density`](@ref) gives ``\rho``. Their syntax is in
[Seismic signatures](@ref man-seismic).

## Where to go next

The last page of the chapter, [Levin's theorem](@ref th-levin), draws from the
concentration tensors of the same schemes the response to free strains and
prestresses, among which the pressure of a fluid in the cracks. The cracks of a
plane interact explicitly, through the tractions each sends onto the others, in
[Interacting cracks](@ref th-interacting-cracks).
