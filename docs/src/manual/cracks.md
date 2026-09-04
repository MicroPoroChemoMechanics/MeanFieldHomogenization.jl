# [Cracks](@id man-cracks)

A crack is an inclusion of **zero volume**: the ``c \to 0`` limit of an
ellipsoid. Two consequences run through this page — the amount of cracking is a
*density*, not a volume fraction, and everything is written in the crack's own
frame ``(\hat{\underline{\ell}}, \hat{\underline{m}}, \hat{\underline{n}})``,
whose third vector is the normal.

```@setup mancracks
using MeanFieldHomogenization
using TensND
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example mancracks
plotly_scene(shape_traces(PennyCrack(1.0)); uid = "man-crack-penny", height = 400,
    title = "Penny-shaped crack: a = b, c = 0, normal n̂")
```

```@example mancracks
plotly_scene(shape_traces(RibbonCrack(0.5)); uid = "man-crack-ribbon", height = 400,
    title = "Ribbon crack: half-width b, unbounded along ℓ̂")
```

The full set of shapes, tilted cracks included, is in
[The inclusion zoo](@ref man-inclusion-gallery); the geometry and the symbols
are defined in [Crack opening displacement](../theory/cod_tensors.md).

```julia
using MeanFieldHomogenization, TensND
E, ν = 210.0, 0.3
k = E/(3*(1-2ν)); μ = E/(2*(1+ν))
C₀ = TensISO{3}(3k, 2μ)

# Penny-shaped crack — size-independent COD tensor B
pc = PennyCrack(1.0)
B  = cod_tensor(pc, C₀)

# Size-independent compliance contribution tensor H = (3/4) n̂ ⊗ˢ B ⊗ˢ n̂
H  = compliance_contribution(pc, C₀)

# Dilute compliance correction ΔS from the Budiansky density ε³ᵈ = N a b²
ε³ᵈ = 0.05
ΔS  = delta_compliance(pc, H, ε³ᵈ)      # = (4π/3) ε³ᵈ H

# Ribbon crack — same pattern, ε²ᵈ = N b² and ΔS = π ε²ᵈ H
r   = RibbonCrack(0.5)
H_r = compliance_contribution(r, C₀)    # H = (2/π) n̂ ⊗ˢ B ⊗ˢ n̂
ΔS_r = delta_compliance(r, H_r, 0.05)

# Thermal / conductivity — scalar COD b and rank-1 resistivity tensor R
K₀ = TensISO{3}(1.0)
b  = cod_tensor(pc, K₀)                  # scalar
R  = compliance_contribution(pc, K₀)     # R = (3/4) b (ŵ⊗ŵ)
ΔR = delta_resistivity(pc, R, 0.05)      # = (4π/3) ε³ᵈ R
```

## Cracks with finite interface stiffness (Sevostianov)

A flat crack carrying a **spring-like interface elasticity** with stiffness
tensor ``\boldsymbol{K}`` (order 2, ``3\times 3`` symmetric — e.g. isotropic with
a normal stiffness ``K_n`` and a tangential one ``K_t``) modifies the COD tensor
``\boldsymbol{B}`` via

```math
\boldsymbol{B}_{\text{eff}}
= \bigl(b\,\boldsymbol{K} + \boldsymbol{B}^{-1}\bigr)^{-1}
= \boldsymbol{B}\cdot\bigl(\boldsymbol{1} + b\,\boldsymbol{K}\cdot\boldsymbol{B}\bigr)^{-1},
```

where ``b`` is the in-plane half-width, `semi_minor(crack)`. The two limits are
the familiar ones: ``\boldsymbol{K} = \boldsymbol{0}`` gives a traction-free
crack (recovering ``\boldsymbol{B}``), and
``\boldsymbol{K}\to\infty`` a rigid bond
(``\boldsymbol{B}_{\text{eff}}\to\boldsymbol{0}``, i.e. no crack at all).

```julia
# Elasticity : iso interface stiffness K = 5·𝟏
B_eff = cod_tensor(pc, C₀; K_interface = TensISO{3}(5.0))
H_eff = compliance_contribution(pc, C₀; K_interface = TensISO{3}(5.0))

# Conductivity (Kapitza scalar interface conductance α)
b_eff = cod_tensor(pc, K₀; α_interface = 1.0)
R_eff = compliance_contribution(pc, K₀; α_interface = 1.0)
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
`Rt(t,t')` ageing interface kernels, see the
[Viscoelasticity manual](viscoelasticity.md#5-cracks-in-alv).
References: [sevostianov2002](@cite), [barthelemyIJES2019](@cite).
