# [Ageing creep of solidifying cementitious materials](@id app-ageing-creep)

The ageing-creep model of [sanahuja2013](@cite): one phase **solidifies
progressively** — as C-S-H does during hydration — so ``\mathbb R^{\rm hom}(t,
t')`` depends on the observation time ``t`` and the loading time ``t'``
*independently*. Laplace–Carson no longer applies; the homogenization runs
directly in the time domain, through [`homogenize_alv`](@ref).

The composite has three phase types:

| Phase | Fraction | Stiffness | Rheology |
|:------|:--------:|:----------|:---------|
| Matrix | ``f_0 = 0.6`` | ``E_0=1,\ \nu_0=0.2`` | Maxwell |
| Solidifying inclusions | ``f_\infty = 0.3`` | ``E_1=5,\ \nu_1=0.3`` | Maxwell (per-layer setting time) |
| Pore | ``1-f_0-f_\infty`` | ``E_p\approx0`` | elastic |

Both viscoelastic phases obey a **Maxwell** relaxation law with separate bulk and
shear characteristic times.

```@example creep
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

# Matrix
const E0, ν0, f0 = 1.0, 0.2, 0.6
const k0, μ0 = E0 / (3(1 - 2ν0)), E0 / (2(1 + ν0))
const η0, γ0 = 0.2, 0.133          # bulk / shear relaxation times
# Solidifying phase
const E1, ν1, finf = 5.0, 0.3, 0.3
const k1, μ1 = E1 / (3(1 - 2ν1)), E1 / (2(1 + ν1))
const η1, γ1 = 1.0, 1.67
# Pore (elastic, near-zero)
const Ep, νp = 1.0e-8, 0.2
const kp, μp = Ep / (3(1 - 2νp)), Ep / (2(1 + νp))
const fp = 1 - f0 - finf
const C_p = TensISO{3}(3kp, 2μp)

make_R0() = maxwell_iso(k0, μ0, η0, γ0)
make_R1() = maxwell_iso(k1, μ1, η1, γ1)
nothing # hide
```

## Solidification kinetics

The solidified fraction grows as ``f(t) = f_\infty\, t^\alpha/(1+t^\alpha)``; the
setting time of the layer carrying midpoint fraction ``f_k = (k+\tfrac12)
f_\infty/N`` is ``t_k = (f_k/(f_\infty-f_k))^{1/\alpha}``.

```@example creep
function setting_times(N, α)
    F = [(i + 0.5) * finf / N for i in 0:(N - 1)]
    return [(f / (finf - f))^(1 / α) for f in F]
end
nothing # hide
```

## Per-layer relaxation law: history-dependent vs frozen

A newly formed layer is deposited stress-free and creeps only from its setting
time on. **History-dependent** (`fixed = false`): layer ``i`` responds as a solid
only if it had set at the *loading time* ``t'``. **Frozen** (`fixed = true`): the
decision is made once, at the start of the observation window ``t_0`` — a cheaper
but physically approximate model.

```@example creep
function inclusion_law(t_set, t0; fixed)
    if fixed
        t0 ≥ t_set && return make_R1()
        return ViscoLaw((t, tp) -> (t < tp ? zero(C_p) : C_p), :relaxation)
    else
        R1 = make_R1()
        return ViscoLaw(
            function (t, tp)
                t < tp && return zero(C_p)
                tp ≥ t_set ? R1.eval_fun(t, tp) : C_p
            end, :relaxation)
    end
end
nothing # hide
```

## Two equivalent RVE topologies

[sanahuja2013](@cite)'s key contribution is that the ``N`` solidifying shells and
the pore can be packed into a **single composite sphere** instead of ``N+1``
separate inclusions — reducing ``N+1`` Eshelby problems to one. `MeanFieldHomogenization`
supports both: `:whole_pores` (``N`` separate spherical inclusions) and `:layers`
(one [`LayeredSphere`](@ref) whose per-layer moduli are ageing relaxation laws).

```@example creep
function build_rve_whole_pores(N, α, t0; fixed)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => make_R0()); fraction = :rest)
    add_phase!(rve, :PORE, Ellipsoid(1.0), Dict(:C => heaviside_law(C_p)); fraction = fp)
    t_sets = setting_times(N, α)
    for i in 1:N
        add_phase!(rve, Symbol(:INC_, i), Ellipsoid(1.0),
            Dict(:C => inclusion_law(t_sets[i], t0; fixed = fixed)); fraction = finf / N)
    end
    return rve
end

function build_rve_layers(N, α, t0; fixed)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => make_R0()); fraction = :rest)
    t_sets = setting_times(N, α)
    f_layers = vcat([fp], fill(finf / N, N))        # pore innermost, shells outward
    cumulative = cumsum(f_layers)
    radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)
    moduli = ntuple(N + 1) do k
        k == 1 ? heaviside_law(C_p) : inclusion_law(t_sets[N - k + 2], t0; fixed = fixed)
    end
    sphere = LayeredSphere(radii, moduli)
    add_phase!(rve, :INCLUSION, sphere, Dict(:C => heaviside_law(C_p)); fraction = fp + finf)
    return rve
end

build_rve(N, α, t0, model; fixed) =
    model === :layers ? build_rve_layers(N, α, t0; fixed = fixed) :
    build_rve_whole_pores(N, α, t0; fixed = fixed)
nothing # hide
```

## Time-domain homogenization and effective creep

`homogenize_alv` returns the ``6n\times6n`` block relaxation matrix over the time
grid; its Volterra inverse (`volterra_inverse`) is the creep-compliance matrix,
from which the uniaxial creep ``E_0 J^E_{\rm eff}(t,t_0)`` follows.

```@example creep
function uniaxial_creep(R)
    J = volterra_inverse(R; block_size = 6)
    n = size(J, 1) ÷ 6
    return [sum(J[6(i - 1) + 1, 6(j - 1) + 1] for j in 1:n) for i in 1:n]
end

function creep_curve(N, α, t0, T, model; fixed)
    R = homogenize_alv(build_rve(N, α, t0, model; fixed = fixed), MoriTanaka(), :C; times = T)
    return uniaxial_creep(R)
end
nothing # hide
```

## Results

Following [sanahuja2013](@cite), the effective creep is computed for five loading
ages ``t_0`` (history-dependent, solid `+`; frozen, dashed) with both RVE
topologies, side by side as in the Echoes book. `N = 100` layers are used.

The **elastic reference** ``1/E^{\rm hom}(t)`` (black dotted) is the
instantaneous (glassy) compliance of the microstructure frozen at time ``t``:
every layer past its setting time carries its *elastic* stiffness
``\mathbb C_1``, the others are still pores. It is built from the same topology
but computed by the **purely elastic** pipeline — [`homogenize`](@ref), no
Volterra algebra anywhere — so its agreement with the start of each creep curve
is an independent cross-check of `homogenize_alv`, not a restatement of it.

```@example creep
const N, α_solid, t_max = 100, 4.0, 10 / 3
loading_ages = (1 / 3, 2 / 3, 4 / 3, 2.0, 8 / 3)
cmap = palette(:viridis, length(loading_ages))
const C_0_el, C_1_el = TensISO{3}(3k0, 2μ0), TensISO{3}(3k1, 2μ1)

# Elastic RVE frozen at time t — same two topologies, elastic moduli only.
layer_stiffness(t, t_set) = t ≥ t_set ? C_1_el : C_p

function build_elastic_rve(N, α, t, model)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_0_el); fraction = :rest)
    t_sets = setting_times(N, α)
    if model === :layers
        cumulative = cumsum(vcat([fp], fill(finf / N, N)))
        radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)
        moduli = ntuple(k -> k == 1 ? C_p : layer_stiffness(t, t_sets[N - k + 2]), N + 1)
        add_phase!(rve, :INCLUSION, LayeredSphere(radii, moduli), Dict(:C => C_p);
            fraction = fp + finf)
    else
        add_phase!(rve, :PORE, Ellipsoid(1.0), Dict(:C => C_p); fraction = fp)
        for i in 1:N
            add_phase!(rve, Symbol(:INC_, i), Ellipsoid(1.0),
                Dict(:C => layer_stiffness(t, t_sets[i])); fraction = finf / N)
        end
    end
    return rve
end

function elastic_ref(t, model)
    C_hom = homogenize(build_elastic_rve(N, α_solid, t, model), MoriTanaka(), :C)
    K_hom, μ_hom = TensND.get_data(C_hom)[1] / 3, TensND.get_data(C_hom)[2] / 2
    return E0 / max(9K_hom * μ_hom / (3K_hom + μ_hom), 1.0e-12)
end

function creep_panel(model, title)
    p = plot(; xlabel = "t", ylabel = "E₀ · J^E_eff(t, t₀)", legend = :topleft,
        framestyle = :box, xlims = (0, t_max), ylims = (0, 20), title = title)
    for (k, t0) in enumerate(loading_ages)
        T = collect(range(t0, t_max; length = 31))
        Jh = creep_curve(N, α_solid, t0, T, model; fixed = false)
        Jf = creep_curve(N, α_solid, t0, T, model; fixed = true)
        plot!(p, T, E0 .* Jh; lw = 2, color = cmap[k], marker = :+, ms = 2,
            label = "history t₀=$(round(t0, digits = 2))")
        plot!(p, T, E0 .* Jf; lw = 1.5, color = cmap[k], ls = :dash, label = "")
    end
    # Sample the setting times (where 1/E^hom jumps) *and* the loading ages,
    # so the start of each curve is read off the reference without
    # interpolation error.
    T_ref = sort(vcat([1e-3], filter(≤(t_max), setting_times(N, α_solid)),
        collect(loading_ages), [t_max]))
    plot!(p, T_ref, [elastic_ref(t, model) for t in T_ref]; lw = 2, color = :black,
        ls = :dot, label = "1/E^hom(t)")
    return p
end

plot(creep_panel(:layers, "model = :layers"),
    creep_panel(:whole_pores, "model = :whole_pores");
    layout = (2, 1), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (860, 880))
```

Three observations, all reproducing [sanahuja2013](@cite):

1. **Ageing** — early loading ages (``t_0`` small) give much larger creep because
   many layers have not yet solidified; the compliance decreases toward the
   elastic limit as ``t_0`` grows.
2. **History vs frozen** — the frozen approach overestimates creep at early ages
   (it ignores solidification before ``t_0``) and converges with the
   history-dependent result at late ages. Each curve starts on the dotted elastic
   reference of its own panel.
3. **Morphology** — the two panels do **not** coincide: `:whole_pores` is
   systematically more compliant than `:layers`.

### Elastic cross-check

``J^E_{\rm eff}(t_0, t_0) = 1/E^{\rm hom}(t_0)`` must hold exactly: the
trapezoidal block ``(1,1)`` of every phase kernel is ``\mathbb R(t_0, t_0)``, its
glassy modulus, and the Volterra products, inverses and layered recurrences all
preserve that block. The two sides below come from disjoint code paths — the
time-domain ALV pipeline and the elastic Mori–Tanaka estimate — so the agreement
validates one against the other.

```@example creep
using Printf
for model in (:layers, :whole_pores), t0 in (2 / 3, 2.0)
    alv = E0 * creep_curve(N, α_solid, t0, [t0, t_max], model; fixed = false)[1]
    ela = elastic_ref(t0, model)
    @printf("%-13s t₀ = %.3f   ALV %9.6f   elastic %9.6f   rel. err. %.1e\n",
        model, t0, alv, ela, abs(alv - ela) / ela)
end
```

## A note on the two topologies

The `:layers` composite sphere and the `:whole_pores` collection of ``N+1``
separate inclusions are **different morphologies** — the first packs the pore and
the solidifying shells concentrically (as hydrates deposit around a pore), the
second scatters them independently in the matrix.

!!! note "Reproducing Echoes: the two topologies genuinely differ"
    `MeanFieldHomogenization` reproduces Echoes for **both** topologies to better than 1 %
    (`:layers` ``E_0 J`` ≈ 1.60 → 11.06; `:whole_pores` ≈ 1.96 → 17.54 at
    ``t_0 = 2/3``), and the two differ by ``\approx 6.5`` in ``E_0 J``.

    That gap is a modeling result, not a discrepancy: the composite-sphere
    packing of [sanahuja2013](@cite) is an efficient model — one Eshelby problem
    instead of ``N+1`` — not an exact reformulation of the separate-inclusion
    RVE. Choosing between them is choosing a morphology.
