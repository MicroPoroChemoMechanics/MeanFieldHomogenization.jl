# [Getting started](@id getting-started)

This page takes a first calculation from the installation of the package to an
effective stiffness: a matrix carrying spherical inclusions is described, one
homogenization scheme is applied to it, and the effective moduli are read back.
No familiarity with mean-field homogenization is assumed beyond linear
elasticity; each quantity is defined in the chapter
[Theory](@ref th-index), to which the text refers where the quantity is first
needed.

## Installation

`MeanFieldHomogenization` is registered in Julia's General registry and is
installed with the package manager, either from the Pkg mode of the REPL
(entered by typing `]`)

```julia
pkg> add MeanFieldHomogenization
```

or, equivalently, through the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("MeanFieldHomogenization")
```

No additional registry is required: the dependencies (`TensND.jl`,
`OrdinaryDiffEq.jl`, `Elliptic.jl`, `QuadGK.jl`, `Polynomials.jl`,
`PolynomialRoots.jl`, `Tensors.jl`, …) come from General as well, and the
type-generic elliptic integrals are bundled as the
[`MeanFieldHomogenization.Elliptic`](@ref MeanFieldHomogenization.Elliptic)
submodule.

### Optional package extensions

Nothing below is needed for the core. Each extension loads by itself when its
trigger packages are imported, and the feature it unlocks is the only thing that
is unavailable without it.

| Load | Unlocks |
| :--- | :--- |
| `DECUHR`, `Integrals` | the adaptive-cubature backend `method = :decuhr` for arbitrary anisotropy; the bundled `method = :nestedquadgk` covers the same cases |
| `NonlinearSolve` | any SciML algorithm for the self-consistent fixed points ([Iterative solvers](@ref man-schemes)) |
| `SymPy` | symbolic closed forms of the elliptic integrals |
| `Ferrite`, `FerriteGmsh`, `Gmsh` | the reference finite-element backend for [FE inclusions](@ref man-fe-inclusions) |
| `Gridap`, `GridapGmsh` | the second FE backend for the same morphologies |
| `Ferrite` (alone) | the material interface of the [finite-element coupling](@ref fe-coupling) — a microstructure as a Gauss-point law |
| `Lux`, `Optimisers`, `Zygote` | *training* a [neural surrogate](@ref man-neural-inclusions); evaluating a shipped model needs none of them |

## A first estimate

A matrix of bulk modulus ``k_0 = 30`` GPa and shear modulus ``\mu_0 = 10`` GPa
carries spherical inclusions twice as stiff, at a volume fraction ``f = 0.2``.
The microstructure is described by a **representative volume element** (RVE),
a container to which each phase is added with its geometry, its properties and
its amount:

```@example getting-started
using MeanFieldHomogenization

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(60.0, 20.0)); fraction = 0.2)
rve
```

`Ellipsoid(1.0)` is a sphere, `iso_stiffness(k, mu)` the isotropic stiffness
tensor of bulk modulus `k` and shear modulus `mu`, and `fraction = :rest` gives
the matrix whatever volume the other phases leave. The RVE says nothing about
which phase plays the role of a matrix: that is decided by the scheme
([Who is the matrix?](@ref man-who-is-the-matrix)).

The Mori–Tanaka scheme takes the matrix as the reference medium, of stiffness
``\mathbb C_0``, and localizes each inclusion on the average strain of the
matrix. With the dilute concentration tensor of the inclusion phase ``i``,

```math
\mathbb A_i^{\mathrm{dil}} = \big[\mathbb I + \mathbb P_i:(\mathbb C_i - \mathbb C_0)\big]^{-1},
```

where ``\mathbb P_i`` is the Hill polarization tensor of the inclusion shape in
the reference medium ([Hill polarization tensors](@ref th-hill-tensors)), the
estimate of the effective stiffness reads

```math
\mathbb C^{\mathrm{hom}} = \mathbb C_0
+ \Big(\sum_i f_i\,(\mathbb C_i - \mathbb C_0):\mathbb A_i^{\mathrm{dil}}\Big)
: \Big(f_0\,\mathbb I + \sum_i f_i\,\mathbb A_i^{\mathrm{dil}}\Big)^{-1},
```

with ``f_0`` the volume fraction of the matrix. [`homogenize`](@ref) evaluates
it for the property `:C` of the RVE, and [`k_mu`](@ref) reads back the bulk and
shear moduli of the isotropic result:

```@example getting-started
C_hom = homogenize(rve, MoriTanaka(), :C)
k_mu(C_hom)
```

Changing the estimate means changing the scheme argument only: `Dilute()`,
`SelfConsistent()`, `DifferentialScheme()` and the bounds `Voigt()` and
`Reuss()` are called the same way, and what each assumes is the subject of
[Homogenization schemes](@ref th-homogenization).

## Where to go next

The documentation can be entered from three directions, depending on what the
reader already knows.

- A reader new to mean-field homogenization is best served by the chapter
  [Theory](@ref th-index) read in its stated order, from
  [The Eshelby problem](@ref th-eshelby-problem) to
  [Homogenization schemes](@ref th-homogenization), before the tutorial
  [A first homogenization](@ref tut-first-estimate), which compares two schemes
  on the RVE above.
- A reader who knows what is to be computed can go to
  [Schemes and RVEs](@ref man-schemes), then to the page of the
  [Manual](@ref man-index) that matches the morphology at hand.
- A user of Echoes will find the correspondence between the two libraries in
  [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes).

## Citing MeanFieldHomogenization

When `MeanFieldHomogenization` is used in published work, it is to be cited as
follows; `CITATION.cff` in the repository root carries the same metadata in a
machine-readable form.

```bibtex
@software{meanfieldhomogenization_jl,
  author = {Barthélémy, Jean-François},
  title  = {MeanFieldHomogenization.jl: Mean-field homogenization of heterogeneous materials},
  doi    = {10.5281/zenodo.21884243},
  url    = {https://doi.org/10.5281/zenodo.21884243},
}
```
