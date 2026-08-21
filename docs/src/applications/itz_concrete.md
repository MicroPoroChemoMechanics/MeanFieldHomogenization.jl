# [The interfacial transition zone in concrete](@id app-itz-concrete)

Around every aggregate in a concrete there is a thin shell of cement paste that
is **more porous than the bulk paste**: packing of cement grains against a rigid
surface leaves more room for water, and the resulting *interfacial transition
zone* (ITZ) is softer and weaker than the material around it. How much
stiffness does that cost? Microstructural model of
[konigsberger2013](@cite).

!!! note "Scope: the elastic effect only"
    [konigsberger2013](@cite) go further and use the ITZ to explain the **elastic
    limit** of concrete: the ITZ is where local failure initiates, so the
    macroscopic proportionality limit is reached long before the bulk paste
    fails. That criterion needs two ingredients `MeanFieldHomogenization` does not provide
    — an orientation-resolved Drucker-Prager check with strain second moments,
    and the Hadamard jump conditions across a plane aggregate/paste interface.
    Only the elastic part of the model is reproduced here. The strength side of
    the same family of models is treated, without an ITZ, in
    [Quasi-brittle strength](strength.md).

## The three scales

The paste scale is the one used in [Quasi-brittle strength](strength.md) — the
hydrate foam and cement paste model of [pichler2011](@cite), shared as
[`scripts/common/quasibrittle_strength.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/common/quasibrittle_strength.jl).
The new ingredient is the fourth phase geometry: **coated** aggregates.

| Scale | RVE | Phases | Scheme |
|:-----:|:----|:-------|:------:|
| 1 | Hydrate foam (HF) | oriented hydrate needles + water + air | Self-Consistent |
| 2 | Cement paste (CP) | HF matrix + clinker grains | Mori-Tanaka |
| 3 | ITZ | CP + extra capillary porosity | Mori-Tanaka |
| 4 | Concrete | CP matrix + ITZ-coated aggregates | Mori-Tanaka |

```@example itz
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

# Densities (relative to water) and phase moduli of pichler2011.
const d_clin, d_hyd, d_san = 3.15, 2.073, 2.648
const K_clin, μ_clin = 116.7, 53.8
const K_hyd, μ_hyd = 18.7, 11.8
const K_san, μ_san = 37.8, 44.3

# Powers hydration model: volume fractions from w/c and hydration degree α.
f_clin(wc, α) = (1 - α) / (1 + d_clin * wc)
f_w(wc, α) = d_clin * (wc - 0.42 * α) / (1 + d_clin * wc)
f_hyd(wc, α) = 1.42 * d_clin / d_hyd * α / (1 + d_clin * wc)
fh_san(wc, sc) = sc / d_san / (1 / d_clin + wc + sc / d_san)

const WC, ALPHA, SC_RATIO = 0.5, 1.0, 3.0

@printf("w/c = %.2f, α = %.1f, s/c = %.1f → aggregate volume fraction %.4f\n",
        WC, ALPHA, SC_RATIO, fh_san(WC, SC_RATIO))
```

## Scale 1–2: the bulk cement paste

The hydrate foam is a self-consistent polycrystal of needle-shaped hydrates
(``\omega = 10^4``) discretized into ``N`` polar orientation bins, plus water
and air; clinker grains are then embedded by Mori-Tanaka. Derivation and
cross-check against Echoes: [Quasi-brittle strength](strength.md).

```@example itz
const N_BINS = 20
const ω_needle = 1.0e4
const TINY = 1.0e-3      # regularizes the exactly-zero water/air stiffness

function hydrate_foam(wc, α; N = N_BINS, ω = ω_needle)
    fclin = f_clin(wc, α)
    fw = f_w(wc, α)
    fhyd = f_hyd(wc, α)
    fair = max(0.0, 1 - fclin - fw - fhyd)
    ## Fractions renormalized to the foam, from which clinker is excluded.
    scale = 1 / (1 - fclin)
    C_hyd = TensISO{3}(3K_hyd, 2μ_hyd)
    C_fluid = TensISO{3}(3TINY, 2TINY)
    ez = (0.0, 0.0, 1.0)

    rve = RVE(:M)
    ## Zero-volume matrix phase: a self-consistent seed only, since the
    ## inclusion fractions already sum to 1.
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_hyd); symmetrize = :iso)
    for (i, bin) in enumerate(polar_orientation_bins(N))
        add_phase!(
            rve, Symbol(:HYD, i), Spheroid(ω; euler_angles = (bin.θ, 0.0, 0.0)),
            Dict(:C => C_hyd); fraction = fhyd * scale * bin.weight,
            symmetrize = TISymmetrize(ez),
        )
    end
    add_phase!(rve, :W, Ellipsoid(1.0), Dict(:C => C_fluid);
        fraction = fw * scale, symmetrize = :iso)
    add_phase!(rve, :AIR, Ellipsoid(1.0), Dict(:C => C_fluid);
        fraction = fair * scale, symmetrize = :iso)
    return homogenize(
        rve, SelfConsistent(; abstol = 1.0e-8, maxiters = 1000, damping = 0.5),
        :C; select_best = true,
    )
end
nothing # hide
```

The converged foam estimate carries a residual transverse isotropy about the
binning axis — an artifact of discretizing a uniform orientation distribution,
not a physical texture. It is projected back to isotropy before the aggregate
scale, which is also what the composite-sphere recurrence of the next section
requires.

```@example itz
C_hf = best_fit_ti(hydrate_foam(WC, ALPHA), (0.0, 0.0, 1.0))

rve_cp = RVE(:HF)
add_matrix!(rve_cp, Ellipsoid(1.0), Dict(:C => C_hf))
add_phase!(
    rve_cp, :CLIN, Ellipsoid(1.0),
    Dict(:C => TensISO{3}(3K_clin, 2μ_clin)); fraction = f_clin(WC, ALPHA),
)
k_cp, μ_cp = k_mu(TensND.proj_tens(Val(:ISO), get_array(homogenize(rve_cp, MoriTanaka(), :C)))[1])
C_cp = TensISO{3}(3k_cp, 2μ_cp)

@printf("bulk cement paste : E = %.2f GPa, ν = %.3f\n", E_nu(C_cp)...)
```

## Scale 3: what an ITZ is made of

The ITZ is modeled as the **same paste carrying additional capillary
porosity** ``\varphi_{\rm ITZ}``, introduced as spherical voids by Mori-Tanaka.
No empirical stiffness–porosity relation is involved: the softening comes out of
the same scheme used everywhere else on this page.

```@example itz
function itz_stiffness(C_cp, φ)
    φ ≤ 0 && return C_cp
    rve = RVE(:CP)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_cp))
    add_phase!(rve, :PORE, Ellipsoid(1.0), Dict(:C => TensISO{3}(0.0, 0.0)); fraction = φ)
    return homogenize(rve, MoriTanaka(), :C)
end

for φ in (0.1, 0.3, 0.5)
    @printf("φ_ITZ = %.1f → E_ITZ = %5.2f GPa  (bulk paste %.2f GPa)\n",
            φ, E_nu(itz_stiffness(C_cp, φ))[1], E_nu(C_cp)[1])
end
```

## Scale 4: coated aggregates

An aggregate of radius ``R`` surrounded by an ITZ of thickness ``t`` is a
two-layer [`LayeredSphere`](@ref) — a *composite sphere*, which
`MeanFieldHomogenization` accepts as an RVE phase directly: it enters the scheme through
its concentration tensors, with no Hill tensor of its own.

Two bookkeeping points, both easy to get wrong:

- the volume fraction to declare is that of the **whole coated particle**,
  ``f_{\rm agg}\,(1+t/R)^3``, not of the aggregate core alone;
- the matrix around it stays the *bulk* paste — the ITZ is already inside the
  inclusion, and counting it twice would double the softening.

```@example itz
const C_SAN = TensISO{3}(3K_san, 2μ_san)
const F_AGG = fh_san(WC, SC_RATIO)

function E_concrete(φ_itz, t_over_R)
    coated = LayeredSphere((1.0, 1.0 + t_over_R), (C_SAN, itz_stiffness(C_cp, φ_itz)))
    rve = RVE(:CP)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_cp))
    add_phase!(
        rve, :AGG, coated, Dict(:C => C_SAN);
        fraction = F_AGG * (1 + t_over_R)^3,
    )
    return E_nu(homogenize(rve, MoriTanaka(), :C))[1]
end

E_ref = E_concrete(0.0, 0.05)   # ITZ as stiff as the bulk paste ⇒ no ITZ at all
@printf("concrete without ITZ contrast : E = %.2f GPa\n", E_ref)
```

## How much stiffness does the ITZ cost?

```@example itz
φs = range(0.0, 0.5; length = 21)
plt = plot(
    xlabel = "additional ITZ porosity φ_ITZ", ylabel = "E_concrete / E (no ITZ)",
    legend = :bottomleft, title = "Elastic cost of a porous ITZ (w/c = $WC, α = $ALPHA)",
)
for t in (0.02, 0.05, 0.10)
    plot!(plt, φs, [E_concrete(φ, t) / E_ref for φ in φs];
          lw = 2, marker = :circle, ms = 2, label = "t/R = $t")
end
plt
```

Three readings of the figure:

- the loss is **roughly linear** in ``\varphi_{\rm ITZ}`` and grows with
  thickness, as one would expect of a thin compliant shell in series with the
  load path;
- it is **modest for realistic ITZs**: at ``t/R = 0.05`` and
  ``\varphi_{\rm ITZ} = 0.3`` — a 5 % shell with a third of it porous — concrete
  stiffness falls by about 11 %. A porous ITZ does not, on its own, explain a
  concrete that is half as stiff as expected;
- the sensitivity to thickness is what makes the ITZ hard to identify from
  stiffness measurements alone: ``t/R`` and ``\varphi_{\rm ITZ}`` trade off
  against each other along the curves.

That last point is the reason [konigsberger2013](@cite) turn to the elastic
*limit* rather than the elastic *modulus*: the ITZ is a far more conspicuous
feature of where concrete starts to fail than of how stiff it is. Reproducing
that argument would need the strength machinery listed in the scope note above.

```@example itz
@printf("t/R = 0.05, φ_ITZ = 0.30 → E/E₀ = %.3f\n", E_concrete(0.3, 0.05) / E_ref)
```
