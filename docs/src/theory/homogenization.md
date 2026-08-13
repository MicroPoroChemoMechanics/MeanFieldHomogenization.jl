# [Homogenization schemes](@id th-homogenization)

`MeanFieldHomogenization.Schemes` computes the *effective* property tensor of a
multi-phase medium from (i) the phase geometries, (ii) the phase properties and
(iii) the phase volume fractions or crack densities.

## Notation

The Representative Volume Element (RVE) consists of:

- a *matrix* phase of property tensor ``\mathbb C_0`` (or ``\boldsymbol{K}_0``
  for the 2nd-order conductivity problem),
- one or more *inclusion* phases of property tensors
  ``\mathbb C_i``, geometries ``\mathcal G_i``, and amounts
  ``f_i`` (volume fraction) or ``\varepsilon_i`` (crack density).

![An RVE loaded by ``\underline u = \boldsymbol E\cdot\underline x`` is replaced by a sum of single-inclusion problems in the infinite matrix ``\mathbb C_m``, each loaded by ``\underline u = \boldsymbol E^0\cdot\underline x`` (from the Echoes book [echoes](@cite))](../assets/schemes/rve_decomposition.png)

That picture *is* the mean-field approximation, and every scheme below is one
answer to the single question it leaves open: **what is ``\boldsymbol E^0``, and
in which medium is each inclusion embedded?** Take ``\boldsymbol E^0 = \boldsymbol E``
and the matrix as the reference and you have the dilute estimate; require the
matrix average to come out right and you have Mori–Tanaka; embed each inclusion
in the *unknown* effective medium and you have the self-consistent scheme.

For each inclusion the **dilute strain concentration tensor**
``\mathbb A_\mathrm{dil}^{(i)}`` and the **size-independent stiffness
contribution** ``\mathbb N_i = (\mathbb C_i - \mathbb C_0):
\mathbb A_\mathrm{dil}^{(i)}`` are the natural building blocks
([Kachanov & Sevostianov 2018](@cite kachanov2018)). The dual
**compliance contribution** ``\mathbb H_i = (\mathbb S_i - \mathbb S_0):
\mathbb A_\sigma^{(i)}`` is more natural for cracks (whose stiffness
contribution is the rank-1 limit of a divergent eigenvalue).

## Bounds

| Scheme | Formula |
| --- | --- |
| **Voigt** | ``\langle \mathbb C \rangle = \sum_i f_i \mathbb C_i`` (upper bound, [Hill 1965](@cite hill1965)) |
| **Reuss** | ``\langle \mathbb S \rangle^{-1}`` (lower bound) |

Cracks are ignored in both bounds: their volume contribution vanishes in
the penny limit (``c \to 0``) while their density stays finite.

## One-shot schemes (require a matrix)

Writing ``\mathbb N_\Sigma = \sum_i f_i \mathbb N_i`` for the total dilute
stiffness contribution and ``\mathbb S_0 = \mathbb C_0^{-1}``:

| Scheme | Effective stiffness |
| --- | --- |
| **Dilute** | ``\mathbb C_0 + \mathbb N_\Sigma`` (first order in ``f``) |
| **DiluteDual** | ``\big(\mathbb S_0 + \sum_i f_i \mathbb H_i\big)^{-1}`` |
| **Mori-Tanaka** | ``\mathbb C_0 + \mathbb N_\Sigma : \big(f_m\,\mathbb I + \sum_i f_i \mathbb A_\mathrm{dil}^{(i)}\big)^{-1}`` ([Mori-Tanaka 1973](@cite mori1973), [Christensen 1990](@cite christensen1990)) |
| **Maxwell** | ``\mathbb C_0 + \mathbb N_\Sigma : (\mathbb I - \mathbb P_d : \mathbb N_\Sigma)^{-1}`` with ``\mathbb P_d`` the Hill tensor of the *outer distribution shape* |
| **PCW** | identical algebraic form, distribution-shape-aware ensemble interpretation ([Ponte-Castañeda & Willis 1995](@cite ponte1995)) |

### The second shape: Maxwell and PCW

Maxwell and PCW differ from the first three rows in that a **second** shape
enters, describing how the inclusions are *placed* rather than what they look
like. Maxwell reads it as one equivalent inclusion swallowing a cluster; PCW
reads it as a safety ellipsoid around each inclusion:

| Maxwell | Ponte Castañeda–Willis |
| :---: | :---: |
| ![A cluster of inclusions is equivalent to a single inclusion Ω, both loaded remotely](../assets/schemes/rve_maxwell.png) | ![Each flat inclusion sits inside its own spatial-distribution ellipsoid](../assets/schemes/rve_pcw.png) |

The **distribution shape** is stored at the RVE level (default: unit
sphere ⇒ Mori-Tanaka limit). Any `AbstractInclusion` can be used; the
hierarchy [`AbstractDistributionShape`](@ref) leaves room for a future
`PairwiseDistribution` extension following [Willis 1982](@cite willis1982).

## Iterative schemes

The one-shot schemes all need a phase to play the role of the matrix. When no
phase does — a polycrystal, a granular assembly, a saturated foam — the
reference medium has to be the effective medium itself, and the estimate becomes
a fixed point:

![No phase plays the role of a matrix: the reference medium is the effective medium being sought (from the Echoes book [echoes](@cite))](../assets/schemes/rve_self_consistent.png)

| Scheme | Iteration |
| --- | --- |
| **SelfConsistent** ([McLaughlin 1977](@cite mclaughlin1977)) | ``\mathbb C^{(n+1)} = \big(\sum_i f_i \mathbb C_i : \mathbb A_\mathrm{dil}^{(i)}(\mathbb C^{(n)})\big) : \big(\sum_i f_i \mathbb A_\mathrm{dil}^{(i)}(\mathbb C^{(n)})\big)^{-1}`` |
| **AsymmetricSelfConsistent** | switches between stiffness- and compliance-form iteration based on the matrix-vs-Voigt-bound contrast |

The default solver is a damped Picard fixed point (Anderson with memory
1, Dual-safe). Loading `NonlinearSolve.jl` activates the
`MeanFieldHomogenizationNonlinearSolveExt` extension, which accepts every SciML
non-linear algorithm (`NewtonRaphson()`, `TrustRegion()`,
`Anderson()`, …) via the `algorithm` keyword of [`SelfConsistent`](@ref).

## Differential scheme

The **DifferentialScheme** integrates the multi-phase Norris ODE
([Norris 1985](@cite norris1985)) on a fictitious incorporation time
``\tau \in [0, 1]``,

```math
\frac{\mathrm d \mathbb C^{hom}}{\mathrm d \tau}
  = \sum_i \dot\varphi_i \, \mathbb N_i(\mathbb C^{hom}) ,
\qquad
\dot\varphi_i = \dot f_i + \frac{f_i}{f_0} \sum_j \dot f_j ,
```

the increments ``\dot\varphi_i`` following from the volume balance by
Sherman-Morrison, along a user-selectable trajectory
([`Proportional`](@ref), [`Sequential`](@ref), [`CustomPath`](@ref),
[`Path`](@ref)). The dual form on the compliance is available through
`formulation = :compliance`, and cracks — which have no volume but a
finite density — enter with a balance of their own.

The trajectories agree in the dilute limit (``f \to 0``) and diverge like
``f`` at finite fractions — a *physical* feature of the scheme.

The full derivation, the crack case, the closed form of the homothetic
trajectory and the SciML resolution are in
[The differential scheme](differential_scheme.md).

## [N-body schemes (require positions)](@id th-nbody)

Every scheme above — bounds, one-shot, iterative, differential — sees *one*
inclusion in a reference medium and accounts for the others only through that
reference: the interaction is treated in an average sense. Two schemes drop that
one-site assumption and resolve the interaction inclusion by inclusion, which
needs strictly more information than an `RVE` carries — the positions. They act
on a [`ParticleAssembly`](@ref) instead, and share one ingredient, the
[two-inclusion interaction tensor](@ref th-interaction) ``\mathbb{T}^{ab}``.

| Scheme | Unknowns | Reference |
| --- | --- | --- |
| **ClusterModel** | mean strain of every family, from ``\sum_K \mathbb{M}_{IK} : \mathbb{A}^K = \mathbb{I}`` — see [the cluster model](@ref th-cluster) | [Molinari & El Mouden 1996](@cite molinari1996) |
| **EquivalentInclusion** | polarization of every inclusion, from a Galerkin discretization of the weak Lippmann-Schwinger equation — see [the equivalent inclusion method](@ref th-eim) | [Brisard et al. 2014](@cite brisard2014) |

The two are the *same* linear system on a periodic assembly and differ only in
how the far field is closed. Both degenerate **exactly** onto Mori-Tanaka when
the interaction is switched off — the sharpest available check that their
assembly is right — and the equivalent inclusion method additionally returns a
rigorous bound on the apparent stiffness.

## Number-type compatibility

Every scheme is mandated to support:

- `Float64` — default;
- `ForwardDiff.Dual` — sensitivity analysis through fractions, moduli,
  geometric parameters;
- `Complex{Float64}` — frequency-domain viscoelasticity (parity with
  the C++ ECHOES library, templated on `T = double | complex<double>`);
- `SymPy.Sym`, `Symbolics.Num`, `BigFloat` — best-effort, with explicit
  documentation of any limitation (the iterative SC solvers are not
  symbolic-friendly because the linear-system Jacobian must be numeric).
