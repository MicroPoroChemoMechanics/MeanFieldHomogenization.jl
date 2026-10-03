# [Conductivity](@id man-conductivity)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity), where the order-2
    counterparts of the elastic tensors are derived, and
    [Hill polarization tensors](@ref th-hill-tensors), whose section on
    conductivity gives the closed form used below.

Everything the elastic API does, the conduction API does with order-2 tensors
instead of order-4. Heat conduction, mass diffusion, electric conduction and
Darcy flow are the same mathematical problem, so the same functions serve all
four — the dispatch is on the **order of the property tensor you pass**, not on
a keyword.

```julia
using MeanFieldHomogenization, TensND

C₀ = TensISO{3}(3 * 70.0, 2 * 30.0)   # order 4 → elasticity
K₀ = TensISO{3}(5.0)                  # order 2 → conduction

hill_tensor(Ellipsoid(1.0), C₀)       # order-4 Hill tensor ℙ
hill_tensor(Ellipsoid(1.0), K₀)       # order-2 Hill tensor 𝐏
```

!!! note "The stress analog is minus the flux"
    Fourier's and Fick's laws carry a minus sign that Hooke's does not, so the
    package takes ``\boldsymbol{\sigma} \equiv -\underline{q} =
    \boldsymbol{K}\cdot\nabla T`` as the stress analog: on a surface of
    normal ``\underline{n}``, ``\boldsymbol{\sigma}\cdot\underline{n}`` is then
    in *both* theories what the exterior transmits to the interior. That is why
    one implementation serves both physics with no sign anywhere, and why the
    tensors named `flux_*` below carry ``\boldsymbol{\sigma}``, not
    ``\underline{q}``. Full statement:
    [Elasticity and transport: one set of formulas](@ref th-notation-sigma-q).

## Building a conductivity tensor

Isotropic, transversely isotropic and fully anisotropic conductivities are
`TensND` order-2 tensors:

```julia
using MeanFieldHomogenization, TensND, LinearAlgebra

K_iso   = TensISO{3}(2.5)                              # isotropic
K_ti    = TensND.TensTI{2}(1.0, 4.0, (0., 0., 1.))     # transverse, axial, axis
K_aniso = TensND.Tens(Matrix(Diagonal([3.2, 0.5, 0.6]))) # orthotropic
```

## The Hill and Eshelby tensors

```julia
ell = Ellipsoid(3.0, 1.0, 1.0)        # prolate spheroid

P = hill_tensor(ell, K_iso)           # 𝐏(𝐀, 𝐊)
S = eshelby_tensor(ell, K_iso)        # Eshelby tensor, P ⋅ K
```

For a **sphere in an isotropic matrix** ``\boldsymbol{K}_0 = k_0\boldsymbol{1}``
these are ``\boldsymbol{P} = \boldsymbol{1}/(3k_0)`` and
``\boldsymbol{S}^{\mathrm{E}} = \boldsymbol{1}/3``, independent of ``k_0``.

!!! tip "Anisotropy is cheap here"
    Unlike the order-4 case, the order-2 Hill tensor has a **closed form for any
    matrix anisotropy**, via the square-root transformation
    ``\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K}) = \boldsymbol{K}^{-1/2}\cdot
    \boldsymbol{I}^{\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2}}\cdot
    \boldsymbol{K}^{-1/2}`` [giraudMOM2019](@cite). Passing an anisotropic
    ``\boldsymbol{K}`` costs no more than an isotropic one — no cubature is
    involved, and the result stays ForwardDiff-compatible.

## Homogenization

An RVE carries its conduction properties under the `:K` key, and every scheme
works unchanged:

```julia
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(20.0)); fraction = 0.3)

K_MT = homogenize(rve, MoriTanaka(), :K)
K_SC = homogenize(rve, SelfConsistent(), :K)
K_D  = homogenize(rve, DifferentialScheme(), :K)
```

The third argument selects the physics: `:C` for elasticity, `:K` for
conduction. A single RVE may carry both, and the two are homogenized
independently.

## Cracks

The transport analog of the crack compliance is the **crack resistivity
contribution** ``\boldsymbol{R}``, a rank-1 order-2 tensor: only the normal
component of the flux can jump across a crack. It is assembled from a scalar
COD ``b``, with the same geometric factors ``3/4`` (elliptic) and ``2/\pi``
(ribbon) as in elasticity.

```julia
pc = PennyCrack(1.0)
K₀ = TensISO{3}(2.5)

b  = cod_tensor(pc, K₀)                    # scalar COD, = 2/(π²K) for a penny
R  = compliance_contribution(pc, K₀)       # 𝐑 = (3/4)·b·(ŵ⊗ŵ)
ΔR = delta_resistivity(pc, R, 0.05)        # dilute correction at density ε
```

Theory: [Thermal cracks](@ref th-thermal-cracks).

## Imperfect interfaces

In conduction, composite particles with imperfect interfaces are provided by:

- [`LayeredSphere`](@ref) — concentric shells, any of the four interface models
  (the elastic ones are described in [Layered inclusions](@ref man-layered));
- [`LayeredSpheroid`](@ref) — confocal spheroidal shells, with the
  low-conducting ([`KapitzaInterface`](@ref)) and highly-conducting
  ([`SurfaceConductiveInterface`](@ref)) models.

```julia
s = LayeredSpheroid(
    (1.0,), (2.0,), (TensISO{3}(1e-6),);            # insulating oblate core
    interfaces = (MeanFieldHomogenization.SurfaceConductiveInterface(2.0),),
    Nseries = 5,
)

A = gradient_gradient_loc(s, TensISO{3}(1e-6), TensISO{3}(1.0))  # ⟨∇T⟩ = 𝐀_Ω·𝐇
B = flux_gradient_loc(s, TensISO{3}(1e-6), TensISO{3}(1.0))      # ⟨𝐊∇T⟩ = 𝐁_Ω·𝐇
k_eq = B[1, 1] / A[1, 1]                                          # equivalent particle
```

Theory: [Layered sphere](@ref th-layered-sphere) and
[Layered spheroid](@ref th-layered-spheroid). Worked examples:
[geometry and effective conductivity](@ref tut-layered-spheroid-effective)
(Kapitza sweep and equivalent particle),
[what an interface does to the local fields](@ref tut-layered-spheroid-interfaces),
and [highly conducting interfaces](@ref tut-layered-spheroid-hc).

## Cross-property links

Conduction and elasticity are not independent — a microstructure that stiffens a
material also changes how it conducts. Explicit cross-property correlations for
two-phase composites are given in [sevostianov2002](@citet); the
[Transport properties](@ref tut-transport) application page works
through one.

## Where to go next

[Viscoelastic homogenization](@ref man-viscoelasticity) passes from a second
physics to time dependence, on the same `RVE` machinery. The tutorial
[Transport properties](@ref tut-transport) applies the calls of this page to
effective conductivities, and
[Cement paste: chloride diffusivity and elasticity](@ref app-cement-paste-diffusion)
computes both properties on one microstructure.
