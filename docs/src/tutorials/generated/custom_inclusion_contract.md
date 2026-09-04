```@meta
EditURL = "../../../../scripts/80_custom_inclusion_contract.jl"
```

# The custom-inclusion contract: three entry gates, one answer

`MeanFieldHomogenization` lets an arbitrary morphology take part in every homogenization
scheme, in elasticity and in transport, without touching the package. This is
the counterpart of the `user_inclusion` mechanism of the C++/Python `echoes`
codebase, but leveled: you implement the **lowest gate you can reach** and
the package derives everything above it.

| Gate | You supply | You get |
|---|---|---|
| **A** | the Hill tensor ``\mathbb P`` | the 8 localization tensors, the 4 contribution tensors, every scheme |
| **B** | the localization tensor ``\mathbb A`` (both sides of it if the inclusion is heterogeneous) | the derived localizations, the contributions, every scheme |
| **C** | the contribution tensors ``\mathbb N``, ``\mathbb H`` | the contribution-based schemes |

This script drives the same physical inclusion through all three gates and
checks that the schemes cannot tell them apart. The point is not the numbers
— they are identical by construction — but that *the plumbing is identical*.

See the manual page *Custom inclusions* and the developer page
*Adding a new inclusion* for the full contract.

````@example custom_inclusion_contract
using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra
````

## The reference: a tilted prolate spheroid

Tilted, so that the effective tensor is genuinely anisotropic and a dropped
rotation cannot hide behind an isotropic answer.

````@example custom_inclusion_contract
C₀ = TensISO{3}(3 * 30.0, 2 * 10.0)      # matrix
C₁ = TensISO{3}(3 * 60.0, 2 * 20.0)      # inclusion
K₀ = TensISO{3}(2.0)                     # matrix conductivity
K₁ = TensISO{3}(7.0)

ell = Ellipsoid(3.0, 1.0, 1.0; euler_angles = (0.3, 0.7, 0.0))
semi = (3.0, 1.0, 1.0)
basis = MeanFieldHomogenization.inclusion_basis(ell)
````

## Gate A — the Hill tensor

One callback. Everything else is derived: the eight localization tensors, the
four contribution tensors, and every scheme.

````@example custom_inclusion_contract
gate_A = CustomInclusion(
    semi; basis,
    hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...),
)
````

`P₀` is a 4th-order stiffness in elasticity and a 2nd-order conductivity in
transport, so a single callback serves both physics.

## Gate B — the localization tensor

For a morphology whose ``\mathbb P`` has no convenient closed form, or when
an external solver returns ``\mathbb A`` directly.

The inclusion here is **homogeneous**, so the strain-side tensor alone is
enough: the stress side follows from
``\mathbb A_{\sigma\varepsilon} = \mathbb C_1:\mathbb A_{\varepsilon\varepsilon}``.
A heterogeneous inclusion has no such ``\mathbb C_1`` and must supply
`stress_strain_loc` (and `flux_gradient_loc`) as well — `LayeredSphere`
implements all four. Omitting them is silent: `Dilute` and `MoriTanaka` stay
right, only the self-consistent schemes drift.

````@example custom_inclusion_contract
gate_B = CustomInclusion(
    semi; basis,
    strain_strain_loc = (P₁, P₀; kw...) -> strain_strain_loc(ell, P₁, P₀; kw...),
    gradient_gradient_loc = (P₁, P₀; kw...) -> gradient_gradient_loc(ell, P₁, P₀; kw...),
)
````

## Gate C — the contribution tensors

The last resort, and the one that buys the least: contribution tensors alone
do not determine the dilute concentration tensor ``\mathbb A``, so
Mori-Tanaka and the self-consistent schemes stay out of reach.

````@example custom_inclusion_contract
gate_C = CustomInclusion(
    semi; basis,
    stiffness_contribution = (P₁, P₀; kw...) -> stiffness_contribution(ell, P₁, P₀; kw...),
    compliance_contribution = (P₁, P₀; kw...) -> compliance_contribution(ell, P₁, P₀; kw...),
    conductivity_contribution = (P₁, P₀; kw...) -> conductivity_contribution(ell, P₁, P₀; kw...),
    resistivity_contribution = (P₁, P₀; kw...) -> resistivity_contribution(ell, P₁, P₀; kw...),
)
````

## Checking conformance before wiring anything

`check_inclusion_interface` reports the level-0 methods, the gate it found,
and — this is the useful part — what that gate leaves unavailable.

````@example custom_inclusion_contract
println("="^78)
check_inclusion_interface(gate_A)
println()
check_inclusion_interface(gate_C)
println("="^78)
````

## The schemes cannot tell them apart

````@example custom_inclusion_contract
function rve_with(geom, prop_m, prop_i, key)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(key => prop_m); fraction = :rest)
    add_phase!(r, :I, geom, Dict(key => prop_i); fraction = 0.25)
    return r
end

c1111(C) = get_array(C)[1, 1, 1, 1]
k11(K) = get_array(K)[1, 1]

ALL = (
    "Dilute" => Dilute(), "DiluteDual" => DiluteDual(),
    "MoriTanaka" => MoriTanaka(), "Maxwell" => Maxwell(),
    "PCW" => PonteCastanedaWillis(), "SelfConsistent" => SelfConsistent(),
    "ASC" => AsymmetricSelfConsistent(), "Differential" => DifferentialScheme(),
)
GATE_C_OK = ("Dilute", "DiluteDual", "Maxwell", "PCW", "Differential")

println("\nElasticity — C_eff[1,1,1,1]\n", "-"^78)
@printf "%-16s %14s %14s %14s %14s\n" "scheme" "Ellipsoid" "gate A" "gate B" "gate C"
for (name, sch) in ALL
    ref = c1111(homogenize(rve_with(ell, C₀, C₁, :C), sch, :C))
    vA = c1111(homogenize(rve_with(gate_A, C₀, C₁, :C), sch, :C))
    vB = c1111(homogenize(rve_with(gate_B, C₀, C₁, :C), sch, :C))
    if name in GATE_C_OK
        vC = c1111(homogenize(rve_with(gate_C, C₀, C₁, :C), sch, :C))
        @printf "%-16s %14.8f %14.8f %14.8f %14.8f\n" name ref vA vB vC
    else
        @printf "%-16s %14.8f %14.8f %14.8f %14s\n" name ref vA vB "n/a"
    end
end

println("\nConduction — K_eff[1,1]\n", "-"^78)
@printf "%-16s %14s %14s %14s %14s\n" "scheme" "Ellipsoid" "gate A" "gate B" "gate C"
for (name, sch) in (
        "Dilute" => Dilute(), "DiluteDual" => DiluteDual(),
        "MoriTanaka" => MoriTanaka(), "Maxwell" => Maxwell(),
    )
    ref = k11(homogenize(rve_with(ell, K₀, K₁, :K), sch, :K))
    vA = k11(homogenize(rve_with(gate_A, K₀, K₁, :K), sch, :K))
    vB = k11(homogenize(rve_with(gate_B, K₀, K₁, :K), sch, :K))
    if name in GATE_C_OK
        vC = k11(homogenize(rve_with(gate_C, K₀, K₁, :K), sch, :K))
        @printf "%-16s %14.8f %14.8f %14.8f %14.8f\n" name ref vA vB vC
    else
        @printf "%-16s %14.8f %14.8f %14.8f %14s\n" name ref vA vB "n/a"
    end
end
````

Every column agrees to the last digit: the gates are algebraically equivalent
reformulations of the same information, and the schemes consume whichever is
available.

## Level 2 — the amount × contribution seam

A scheme adds up **an amount times a size-independent contribution**. For a
volume fraction that is simply ``f\,\mathbb N``, and there is nothing to
implement. For a *flat* object measured by a **density** the prefactor is
geometric — ``4\pi/3`` for an elliptical crack of Budiansky density
``\varepsilon^{3\mathrm d} = N a b^2`` — and that is exactly the seam
`density_factor` fills.

````@example custom_inclusion_contract
crack = EllipticCrack(1.0, 0.25)
flat = CustomInclusion(
    (1.0, 0.25, 0.0);
    basis = MeanFieldHomogenization.inclusion_basis(crack),
    density_factor = 4π / 3,
    compliance_contribution = (P₀; kw...) -> compliance_contribution(crack, P₀; kw...),
    stiffness_contribution = (P₀; kw...) -> stiffness_contribution(crack, P₀; kw...),
)

function cracked(geom, ε)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(r, :cr, geom, Dict(:C => C₀); density = ε)
    return r
end

println("\nFlat object with a density amount — C_eff[1,1,1,1]\n", "-"^78)
@printf "%-8s %18s %18s\n" "ε" "EllipticCrack" "CustomInclusion"
for ε in (0.0, 0.02, 0.05, 0.1)
    @printf "%-8.2f %18.10f %18.10f\n" ε c1111(homogenize(cracked(crack, ε), MoriTanaka(), :C)) c1111(homogenize(cracked(flat, ε), MoriTanaka(), :C))
end
````

## What is free: orientation averaging

`IsoSymmetrize` / `TISymmetrize` are applied by the *scheme*, after the
callback returns — so a custom inclusion inherits them at no cost. The only
obligations are to return tensors in the **global** frame, and to know that
under symmetrization the scheme hands the kernel a pre-projected reference
medium (isotropic by default).

````@example custom_inclusion_contract
function oriented(geom_at, nbins, f)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
    for (i, bin) in enumerate(polar_orientation_bins(nbins))
        add_phase!(
            r, Symbol(:I, i), geom_at(bin.θ), Dict(:C => C₁);
            fraction = f * bin.weight, symmetrize = TISymmetrize((0.0, 0.0, 1.0))
        )
    end
    return r
end

spheroid_at(θ) = Ellipsoid(3.0, 1.0, 1.0; euler_angles = (θ, 0.0, 0.0))
custom_at(θ) = let e = spheroid_at(θ)
    CustomInclusion(
        semi; basis = MeanFieldHomogenization.inclusion_basis(e),
        hill_tensor = (P₀; kw...) -> hill_tensor(e, P₀; kw...)
    )
end

C_ref = homogenize(oriented(spheroid_at, 12, 0.2), MoriTanaka(), :C)
C_cus = homogenize(oriented(custom_at, 12, 0.2), MoriTanaka(), :C)

println("\nOrientation distribution (12 polar bins) + TISymmetrize\n", "-"^78)
println(
    "  native  : ", typeof(C_ref).name.name, "   C₃₃₃₃ = ",
    round(get_array(C_ref)[3, 3, 3, 3], digits = 10)
)
println(
    "  custom  : ", typeof(C_cus).name.name, "   C₃₃₃₃ = ",
    round(get_array(C_cus)[3, 3, 3, 3], digits = 10)
)
@printf "  max |Δ| = %.3e\n" maximum(abs, get_array(C_cus) .- get_array(C_ref))
````

The exact transverse-isotropic average is applied identically in both cases:
the custom inclusion never had to know that orientation averaging existed.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

