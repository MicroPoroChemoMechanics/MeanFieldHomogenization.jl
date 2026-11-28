# [Cracks](@id man-cracks)

!!! info "Before this page"
    [Crack opening displacement and compliance](@ref th-cod-tensors), where the
    tensors ``\boldsymbol{B}`` and ``\mathbb{H}`` computed below are derived, and
    [Thermal cracks](@ref th-thermal-cracks) for their conduction counterparts.

A crack is an inclusion of **zero volume**: the ``c \to 0`` limit of an
ellipsoid. Two consequences run through this page — the amount of cracking is a
*density*, not a volume fraction, and everything is written in the crack's own
frame ``(\underline{\ell}, \underline{m}, \underline{n})``,
whose third vector is the normal.

```@example mancracks
using MeanFieldHomogenization
using TensND
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
nothing # hide
```

```@example mancracks
plotly_scene(shape_traces(PennyCrack(1.0)); uid = "man-crack-penny", height = 400,
    title = "Penny-shaped crack: a = b, c = 0, normal <u><i>n</i></u>")
```

```@example mancracks
plotly_scene(shape_traces(RibbonCrack(0.5)); uid = "man-crack-ribbon", height = 400,
    title = "Ribbon crack: half-width b, unbounded along <u><i>ℓ</i></u>")
```

The full set of shapes, tilted cracks included, is in
[The inclusion zoo](@ref man-inclusion-gallery); the geometry and the symbols
are defined in [Crack opening displacement](@ref th-cod-tensors).

```julia
using MeanFieldHomogenization, TensND
E, ν = 210.0, 0.3
k = E/(3*(1-2ν)); μ = E/(2*(1+ν))
C₀ = TensISO{3}(3k, 2μ)

# Penny-shaped crack — size-independent COD tensor B
pc = PennyCrack(1.0)
B  = cod_tensor(pc, C₀)

# Size-independent compliance contribution tensor H = (3/4) n̂ ⊗ˢ B ⊗ˢ n̂
H  = inverse_property_contribution(pc, C₀)

# Dilute compliance correction ΔS from the Budiansky density ε³ᵈ = N a b²
ε³ᵈ = 0.05
ΔS  = delta_inverse_property(pc, H, ε³ᵈ)      # = (4π/3) ε³ᵈ H

# Ribbon crack — same pattern, ε²ᵈ = N b² and ΔS = π ε²ᵈ H
r   = RibbonCrack(0.5)
H_r = inverse_property_contribution(r, C₀)    # H = (2/π) n̂ ⊗ˢ B ⊗ˢ n̂
ΔS_r = delta_inverse_property(r, H_r, 0.05)

# Thermal / conductivity — scalar COD b and rank-1 resistivity tensor R
K₀ = TensISO{3}(1.0)
b  = cod_tensor(pc, K₀)                  # scalar
R  = inverse_property_contribution(pc, K₀)     # R = (3/4) b (ŵ⊗ŵ)
ΔR = delta_inverse_property(pc, R, 0.05)      # = (4π/3) ε³ᵈ R
```

## Cracks with finite interface stiffness

A flat crack carrying a **spring-like interface elasticity** with stiffness
tensor ``\boldsymbol{K}`` (order 2, ``3\times 3`` symmetric — e.g. isotropic with
a normal stiffness ``k_n`` and a tangential one ``k_t``) modifies the COD tensor
``\boldsymbol{B}`` via

```math
\boldsymbol{B}^{\mathrm{hom}}
= \bigl(b\,\boldsymbol{K} + \boldsymbol{B}^{-1}\bigr)^{-1}
= \boldsymbol{B}\cdot\bigl(\boldsymbol{1} + b\,\boldsymbol{K}\cdot\boldsymbol{B}\bigr)^{-1},
```

where ``b`` is the in-plane half-width, `semi_minor(crack)`. The two limits are
the familiar ones: ``\boldsymbol{K} = \boldsymbol{0}`` gives a traction-free
crack (recovering ``\boldsymbol{B}``), and
``\boldsymbol{K}\to\infty`` a rigid bond
(``\boldsymbol{B}^{\mathrm{hom}}\to\boldsymbol{0}``, i.e. no crack at all).

!!! warning "An approximation, exact in its two limits"
    The correction is not derived from the integral equation of a crack with
    spring-like faces, whose opening is not that of a traction-free crack: it is an
    approximation, and no published source is known to the package for it. The
    aging version has the same status ([theory](@ref th-visco-cracks)).

```julia
# Elasticity : iso interface stiffness K = 5·𝟏
B_eff = cod_tensor(pc, C₀; K_interface = TensISO{3}(5.0))
H_eff = inverse_property_contribution(pc, C₀; K_interface = TensISO{3}(5.0))

# Conductivity (Kapitza scalar interface conductance α)
b_eff = cod_tensor(pc, K₀; α_interface = 1.0)
R_eff = inverse_property_contribution(pc, K₀; α_interface = 1.0)
```

When building an `RVE` for a `homogenize` call, attach the interface
data as **phase properties** so the dispatcher picks them up
automatically :

```julia
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀, :K => K₀); fraction = :rest)
add_phase!(rve, :CRACK, PennyCrack(1.0),
            Dict(:C => C₀,
                  :K_interface => TensISO{3}(5.0),  # elastic interface
                  :K => K₀,
                  :α_interface => 1.0);             # Kapitza scalar
            density = 0.10, symmetrize = :iso)

C_eff = homogenize(rve, MoriTanaka(), :C)
K_eff = homogenize(rve, MoriTanaka(), :K)
```

For SC-type schemes on cracked RVEs, which form applies depends on the
orientation distribution:

- **single orientation** — the symmetric [`SelfConsistent`](@ref) raises a
  `SingularException`: its strain-concentration tensor degenerates for a phase
  with no volume and no orientation average to smooth it. Use
  [`AsymmetricSelfConsistent`](@ref).
- **isotropic or TI distribution** (`symmetrize = IsoSymmetrize()` or
  `TISymmetrize(axis)`) — both forms run, and they solve **different fixed
  points**: the symmetric one iterates on the stiffness, the asymmetric one on
  the compliance. Both **percolate**, but not at the same crack density: for
  randomly oriented penny cracks the compliance form reaches zero at the classical
  Budiansky–O'Connell value ``\varepsilon = 9/16`` exactly, the stiffness one at
  ``\varepsilon \approx 1.158`` — both independent of ``\nu_0``. Between the two thresholds they therefore
  disagree qualitatively, not just numerically. Which one to pick is a modeling
  decision, worked out with numbers in
  [Crack distributions: isotropic or parallel](@ref tut-crack-distributions).

For the **time-dependent** (ALV) version with `Rn(t,t')` and
`Rt(t,t')` aging interface kernels, see the
[Viscoelasticity manual](@ref man-visco-cracks).
References: [sevostianov2002, barthelemyIJES2019](@citet).

## [Cracks from their intensity factors](@id man-rice-cracks)

A crack whose intensity factors are known has a compliance contribution, by
integrating its energy release rate along its growth
([theory](@ref th-rice-contributions)). Three cracks are built on it, in an
isotropic matrix:

| type | geometry | density | contributions |
| :--- | :------- | :------ | :------------ |
| [`AnnularCrack`](@ref)`(a, c)` | penny of radius ``a`` bonded on the disc of radius ``c`` | ``N a^3`` | ``\boldsymbol{B}``, ``b``, ``\mathbb{H}``, ``\boldsymbol{R}``, `sif` at both fronts |
| [`CircularArcCrack`](@ref)`(R, α)` | arc of circle of radius ``R``, half-angle ``\alpha``, unbounded along the axis | ``N (R\alpha)^2`` | ``\mathbb{H}``, ``\boldsymbol{R}`` (closed forms), `sif` at both tips |
| [`saturn_ring_compliance`](@ref)`(a, c, C₀)` | spherical pore of radius ``a`` with a ring crack of length ``c`` | — | ``V H_{nnnn}`` only |

```@example mancracks
C₀ = iso_stiffness_E_nu(1.0, 0.25)
K₀ = TensISO{3}(1.0)
ann = AnnularCrack(1.0, 0.5)
plotly_scene(shape_traces(ann); uid = "man-crack-annulus", height = 380,
    title = "Annular crack: outer radius a, bonded disc of radius c")
```

```@example mancracks
B = cod_tensor(ann, C₀)                        # Rice's formula along the growth of the inner front
B_open = cod_tensor(ann, C₀; method = :opening) # the mean jumps of the final crack, one solve
b = cod_tensor(ann, K₀)                        # conduction
TensND.canonical_components(B), TensND.canonical_components(B_open), b
```

`AnnularCrack` and `CircularArcCrack` enter the schemes like any crack, by
their density; the intensity factors being those of an isotropic matrix, a
self-consistent reference must stay isotropic (`symmetrize = IsoSymmetrize()`).

A user who knows the intensity factors of another flat crack whose fronts are
circles gives them to [`RiceCrack`](@ref); for any flat front,
[`rice_cod_tensor`](@ref) takes the front point by point. Worked examples:
[Cracks from their intensity factors](@ref tut-rice-cracks).

## [Interacting cracks of a plane](@id man-interacting-cracks)

A few cracks of a plane load each other: their intensity factors follow from
the method of average tractions ([Interacting cracks](@ref th-interacting-cracks)),
by [`sif`](@ref) on a list of `center => crack` pairs. Two collinear cracks of
half-length 0.45, their inner tips 0.1 apart, under a tension normal to them:

```@example cracks_interaction
using MeanFieldHomogenization, TensND
C₀ = TensISO{2}(2.0, 1.0)                       # isotropic plane: the result does not depend on it
Σ = Tens([0.0 0.0; 0.0 1.0])
pair = [[-0.55, 0.0] => RibbonCrack(0.45; angle = 0.0), [0.55, 0.0] => RibbonCrack(0.45; angle = 0.0)]
r = sif(pair, C₀, Σ)
K₀ = sqrt(π * 0.45)
(inner = r[2].minus[1] / K₀, outer = r[2].plus[1] / K₀, mean_traction = r[2].traction[1])
```

The stiffness of an anisotropic plane gives the fields of its slit instead:

```@example cracks_interaction
C_ort = from_mandel([10.0 2.0 0.0; 2.0 3.0 0.0; 0.0 0.0 3.0])
stack = [[0.0, 0.0] => RibbonCrack(0.5; angle = 0.0), [0.0, 0.4] => RibbonCrack(0.5; angle = 0.0)]
K₀ = sqrt(π * 0.5)
(isotropic = sif(stack, C₀, Σ)[1].plus / K₀, orthotropic = sif(stack, C_ort, Σ)[1].plus / K₀)
```

## Where to go next

[Layered inclusions](@ref man-layered) leaves the ellipsoid family for
composite patterns, which have no Hill tensor. The tutorial
[Cracks and crack density](@ref tut-cracks) drives the calls above through the
schemes, and
[Crack distributions: isotropic or parallel](@ref tut-crack-distributions)
quantifies the choice between the two self-consistent forms discussed above.
