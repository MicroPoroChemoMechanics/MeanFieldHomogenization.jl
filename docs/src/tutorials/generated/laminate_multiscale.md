```@meta
EditURL = "../../../../scripts/36_laminate_multiscale.jl"
```

# [Multiscale chaining: explicit and declarative, side by side](@id tut-laminate-multiscale)

A multiscale model chains homogenizations: what one scale computes becomes a
phase property at the next. `MeanFieldHomogenization` supports two ways of writing that
chain, and they compute the same thing to the last bit.

- **explicit** — one call per scale, in order; the author hands each result
  to the next. The style of most of the Applications chapters.
- **declarative** — a property value *is* another cell plus a scheme
  ([`Homogenized`](@ref)), resolved lazily. The whole model is one object.

This script writes the same three-scale model both ways, checks that they
agree exactly, and shows what the declarative form buys: a sensitivity that
crosses every scale in a single `ForwardDiff` pass, and one microstructure
answering several physics.

Neither style is deprecated — see [Multiscale models](@ref man-multiscale)
for when each is the right choice.

````@example laminate_multiscale
using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra

iso(k, μ) = TensISO{3}(3k, 2μ)
````

## The model

Three scales, mixing both cell types on purpose:

1. **micro** — a porous solid, an `RVE` closed by Mori-Tanaka;
2. **meso** — a `Laminate` alternating that porous solid with a stiff layer,
   solved exactly;
3. **macro** — an `RVE` embedding the laminated material as spherical
   aggregates in a soft binder.

````@example laminate_multiscale
C_solid = iso(3.0, 1.2)
C_pore = iso(1.0e-9, 1.0e-9)
C_stiff = iso(0.5, 0.2)
C_binder = iso(1.0, 0.4)
K_solid, K_pore, K_stiff, K_binder = TensISO{3}(2.0), TensISO{3}(1.0e-9), TensISO{3}(0.3), TensISO{3}(1.0)

function micro_cell(; φ = 0.2)
    r = RVE()
    add_phase!(r, :SOLID, Ellipsoid(1.0), Dict(:C => C_solid, :K => K_solid); fraction = :rest)
    add_phase!(r, :pore, Ellipsoid(1.0), Dict(:C => C_pore, :K => K_pore); fraction = φ)
    return r
end
````

## Explicit chaining

````@example laminate_multiscale
function explicit_chain(; φ = 0.2, f_lay = 0.4, f_agg = 0.3)
    C_micro = homogenize(micro_cell(; φ = φ), MoriTanaka(), :C)

    meso = Laminate(; normal = (0, 0, 1))
    add_layer!(meso, :POROUS, Dict(:C => C_micro); fraction = f_lay)
    add_layer!(meso, :STIFF, Dict(:C => C_stiff); fraction = 1 - f_lay)
    C_meso = homogenize(meso, Laminated(), :C)

    macro_rve = RVE()
    add_phase!(macro_rve, :BINDER, Ellipsoid(1.0), Dict(:C => C_binder); fraction = :rest)
    add_phase!(
        macro_rve, :agg, Ellipsoid(1.0), Dict(:C => C_meso);
        fraction = f_agg, symmetrize = :iso
    )
    return homogenize(macro_rve, MoriTanaka(), :C)
end
````

## Declarative chaining

The same three scales, but the model is assembled once and evaluated once.
Note that nothing is homogenized while the object is being built.

````@example laminate_multiscale
function declarative_model(; φ = 0.2, f_lay = 0.4, f_agg = 0.3)
    meso = Laminate(; normal = (0, 0, 1))
    add_layer!(
        meso, :POROUS,
        Dict(:C => Homogenized(micro_cell(; φ = φ), MoriTanaka()));   # ← scale 1 → 2
        fraction = f_lay
    )
    add_layer!(meso, :STIFF, Dict(:C => C_stiff); fraction = 1 - f_lay)

    macro_rve = RVE()
    add_phase!(macro_rve, :BINDER, Ellipsoid(1.0), Dict(:C => C_binder); fraction = :rest)
    add_phase!(
        macro_rve, :agg, Ellipsoid(1.0),
        Dict(:C => Homogenized(meso, Laminated()));                    # ← scale 2 → 3
        fraction = f_agg, symmetrize = :iso
    )
    return macro_rve
end

C_x = explicit_chain()
C_d = homogenize(declarative_model(), MoriTanaka(), :C)

kx, μx = k_mu(C_x)
kd, μd = k_mu(C_d)
@printf "explicit    : k = %.10f   μ = %.10f\n" kx μx
@printf "declarative : k = %.10f   μ = %.10f\n" kd μd
@printf "difference  : %.2e / %.2e\n" abs(kx - kd) abs(μx - μd)
````

## What the declarative form buys, 1: sensitivities across every scale

[`NestedParameter`](@ref) addresses a scalar *inside* a nested cell. Two
levels down, the shear modulus of the solid skeleton:

````@example laminate_multiscale
model = declarative_model()
p_μ = nested(:agg, :C, nested(:POROUS, :C, property(:SOLID, :C, :shear)))
@printf "\nget_param(nested ×2) = %.4f   (2μ_solid = %.4f)\n" get_param(model, p_μ) (2 * 1.2)

dμ = derivative(model, MoriTanaka(), p_μ; indexer = C -> k_mu(C)[2])
````

Against a central finite difference through the explicit chain:

````@example laminate_multiscale
function μ_macro_explicit(x)
    C_micro = let r = RVE()
        add_phase!(r, :SOLID, Ellipsoid(1.0), Dict(:C => iso(3.0, x / 2)); fraction = :rest)
        add_phase!(r, :pore, Ellipsoid(1.0), Dict(:C => C_pore); fraction = 0.2)
        homogenize(r, MoriTanaka(), :C)
    end
    meso = Laminate(; normal = (0, 0, 1))
    add_layer!(meso, :POROUS, Dict(:C => C_micro); fraction = 0.4)
    add_layer!(meso, :STIFF, Dict(:C => C_stiff); fraction = 0.6)
    m = RVE()
    add_phase!(m, :BINDER, Ellipsoid(1.0), Dict(:C => C_binder); fraction = :rest)
    add_phase!(
        m, :agg, Ellipsoid(1.0), Dict(:C => homogenize(meso, Laminated(), :C));
        fraction = 0.3, symmetrize = :iso
    )
    return k_mu(homogenize(m, MoriTanaka(), :C))[2]
end

h = 1.0e-6
fd = (μ_macro_explicit(2 * 1.2 + h) - μ_macro_explicit(2 * 1.2 - h)) / (2h)
@printf "∂μ_macro/∂(2μ_solid) : AD = %.10f   FD = %.10f   |Δ| = %.2e\n" dμ fd abs(dμ - fd)
````

A laminate thickness, three scales up, works the same way — and so does a
gradient mixing lenses from different scales.

````@example laminate_multiscale
p_h = nested(:agg, :C, thickness(:POROUS))
p_φ = nested(:agg, :C, nested(:POROUS, :C, amount(:pore)))
g = gradient(model, MoriTanaka(), [p_μ, p_h, p_φ]; indexer = C -> k_mu(C)[2])
@printf "\ngradient over three scales : ∂μ/∂(2μ_solid) = %.6f\n" g[1]
@printf "                             ∂μ/∂h_POROUS   = %.6f\n" g[2]
@printf "                             ∂μ/∂φ_pore     = %.6f\n" g[3]
````

## What the declarative form buys, 2: one microstructure, several physics

`Homogenized(cell, scheme)` without an explicit `property` **inherits the key
it is stored under**. The same nested object therefore answers `:C` with the
inner effective stiffness and `:K` with the inner effective conductivity —
the microstructure is described once, not once per physics.

````@example laminate_multiscale
h_micro = Homogenized(micro_cell(), MoriTanaka())
meso = Laminate(; normal = (0, 0, 1))
add_layer!(meso, :POROUS, Dict(:C => h_micro, :K => h_micro); fraction = 0.4)
add_layer!(meso, :STIFF, Dict(:C => C_stiff, :K => K_stiff); fraction = 0.6)

h_meso = Homogenized(meso, Laminated())
macro_rve = RVE()
add_phase!(macro_rve, :BINDER, Ellipsoid(1.0), Dict(:C => C_binder, :K => K_binder); fraction = :rest)
add_phase!(
    macro_rve, :agg, Ellipsoid(1.0), Dict(:C => h_meso, :K => h_meso);
    fraction = 0.3, symmetrize = :iso
)

k_eff, μ_eff = k_mu(homogenize(macro_rve, MoriTanaka(), :C))
K_eff = homogenize(macro_rve, MoriTanaka(), :K)
@printf "\none model, two physics : k = %.6f  μ = %.6f\n" k_eff μ_eff
println("                        K = ", K_eff)
````

## The cost: the memoization

Within one `homogenize` call each `(cell, key)` pair is evaluated exactly
once, however many times a scheme reads it. That matters for the iterative
schemes: `SelfConsistent` reads the phase properties once per iteration, so
without the call-scoped cache the nested cells would be re-solved on every
one of them.

````@example laminate_multiscale
t_mt = @elapsed homogenize(declarative_model(), MoriTanaka(), :C)
t_sc = @elapsed homogenize(declarative_model(), SelfConsistent(), :C)
@printf "\nMori-Tanaka      : %.4f s\n" t_mt
@printf "self-consistent  : %.4f s   (the nested cells are solved ONCE, not once per iteration)\n" t_sc

C_sc_d = homogenize(declarative_model(), SelfConsistent(), :C)
C_sc_x = let
    m = RVE()
    add_phase!(m, :BINDER, Ellipsoid(1.0), Dict(:C => C_binder); fraction = :rest)
    meso_x = Laminate(; normal = (0, 0, 1))
    add_layer!(
        meso_x, :POROUS, Dict(:C => homogenize(micro_cell(), MoriTanaka(), :C));
        fraction = 0.4
    )
    add_layer!(meso_x, :STIFF, Dict(:C => C_stiff); fraction = 0.6)
    add_phase!(
        m, :agg, Ellipsoid(1.0), Dict(:C => homogenize(meso_x, Laminated(), :C));
        fraction = 0.3, symmetrize = :iso
    )
    homogenize(m, SelfConsistent(), :C)
end
@printf "SC explicit vs declarative : |Δk| = %.2e\n" abs(k_mu(C_sc_x)[1] - k_mu(C_sc_d)[1])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

