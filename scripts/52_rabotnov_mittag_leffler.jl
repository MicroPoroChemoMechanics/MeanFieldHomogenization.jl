# =============================================================================
#  52_rabotnov_mittag_leffler.jl
#
#  Closed-form validation of the ALV homogenization pipeline against the
#  Rabotnov / Mittag-Leffler benchmark of @barthelemyIJES2019 §5.
#
#  The matrix is a non-ageing fractional Maxwell with shear relaxation
#
#      μ_M(t,t') = μ_0 · [ 1 + λ_0 · I_Rabotnov(t-t', α_0, β_0) ] · 𝕂
#                + 3·k_0 · 𝕁
#
#  where `I_Rabotnov(τ, α, β) = (1 − E_{α+1, 1}(−β·τ^(α+1))) / β` and
#  `E_{α,β}` is the two-parameter Mittag-Leffler function.  The bulk
#  modulus `k_0` is constant.
#
#  Inclusions are spheres in two limits:
#    1. **Rigid** (`stiff_kmu` × 1e6) — schemes Dilute and Maxwell.
#    2. **Pores**  (`stiff_kmu` × 1e-10) — schemes DiluteDual, Maxwell, NIA.
#
#  For each (scheme, volume fraction) the closed form for the effective
#  shear `L^μ(t)` (creep compliance) and `μ(t)` (relaxation modulus) is
#  available analytically (eqs. (35)–(43) of @barthelemyIJES2019).
#
#  This script computes the homogenized effective shear via
#  `homogenize_alv` and overlays the closed-form curves.
#
#  The Mittag-Leffler function used to come from the ECHOES Python test suite
#  through PyCall, which made this script machine-dependent.  It no longer
#  does.  Two things replaced it:
#
#    * `MittagLeffler.jl` (a weak dependency) when it is loaded — the
#      `Rabotnov` model then evaluates `E_{α+1,1}` directly, and `α ∈ (-1,0)`
#      here puts the order in `(0,1)` where that package is reliable;
#    * otherwise the numerical inversion of the model's **closed-form**
#      Laplace-Carson transform,
#
#          R*(p) = μ₀ (1 + λ₀ / (p^{α+1} + β)),
#
#      which involves no special function at all.  The two agree to ~1e-10,
#      and this script prints the comparison.
#
#  Usage  : julia --project scripts/52_rabotnov_mittag_leffler.jl
#  Output : scripts/figures/52_rabotnov_mittag_leffler.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using MittagLeffler
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── The Rabotnov kernel, from the model library ────────────────────────────
#  I_Rabotnov(τ, α, β) = (1 − E_{α+1,1}(−β τ^{α+1})) / β
#
#  This is the shear channel of a `Rabotnov` model with μ₀ = 1 and λ₀ = 1, so
#  the library evaluates it — through `MittagLeffler.jl` when it is available,
#  and by inverting the closed-form transform when it is not.

I_Rabotnov(t, α, β) = t ≤ 0 ? 0.0 :
    (relaxation(Rabotnov(1.0, 1.0, α, β), t) - 1.0)

# ─── Fractional Maxwell matrix law  (Eq. (5) of @barthelemyIJES2019) ───────

const k_0 = 5.97
const μ_0 = 1.7
const α_0 = -0.46
const β_0 = 0.98
const λ_0 = -0.495

function R_matrix(t, tp)
    α_M = 3 * k_0
    β_M = 2 * μ_0 * (1 + λ_0 * I_Rabotnov(t - tp, α_0, β_0))
    return TensISO{3}(α_M, β_M)
end

const law_matrix = ViscoLaw(R_matrix, :relaxation)

# ─── Closed-form effective shear (eqs. (35)–(43) of [@barthelemyIJES2019]) ──
#
#  Rabotnov / Mittag-Leffler creep formulas transcribed in Julia.
#  `Lmu_*` is the shear creep compliance,
#  `mu_*` is the shear relaxation modulus.

function Lmu_sph_rig_DIL(t, f)
    μ_0d = μ_0 * (1 + 5f / 6 * (3k_0 + 4μ_0) / (k_0 + 2μ_0))
    a0μ = λ_0 * μ_0 * (3 + 5f) / (3μ_0d)
    a1μ = 5f / (2 * (3 + 5f)) * (k_0 / (k_0 + 2μ_0))^2 * a0μ
    β_1 = β_0 + 2 * λ_0 * μ_0 / (k_0 + 2μ_0)
    B = β_0 + a0μ + β_1 + a1μ
    C = a0μ * β_1 + a1μ * β_0 + β_0 * β_1
    sqD = sqrt(B^2 - 4C)
    β_3 = (B - sqD) / 2;  β_4 = (B + sqD) / 2
    a3μ = (β_0 - β_3) * (β_3 - β_1) / (β_3 - β_4)
    a4μ = (β_0 - β_4) * (β_4 - β_1) / (β_4 - β_3)
    return (
        1 + a3μ * I_Rabotnov(t, α_0, β_3)
            + a4μ * I_Rabotnov(t, α_0, β_4)
    ) / μ_0d
end

function mu_sph_rig_DIL(t, f)
    μ_0d = μ_0 * (1 + 5f / 6 * (3k_0 + 4μ_0) / (k_0 + 2μ_0))
    a0μ = λ_0 * μ_0 * (3 + 5f) / (3μ_0d)
    a1μ = 5f / (2 * (3 + 5f)) * (k_0 / (k_0 + 2μ_0))^2 * a0μ
    β_1 = β_0 + 2 * λ_0 * μ_0 / (k_0 + 2μ_0)
    return μ_0d * (
        1 + a0μ * I_Rabotnov(t, α_0, β_0)
            + a1μ * I_Rabotnov(t, α_0, β_1)
    )
end

function Lmu_sph_rig_MAX(t, f)
    μ_0X = μ_0 * (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0)) /
        (6 * (1 - f) * (k_0 + 2μ_0))
    b0μ = 2 * (k_0 + 2μ_0) * (3 + 2f) * λ_0 /
        (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0))
    b1μ = 5f * k_0^2 * λ_0 /
        (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0)) / (k_0 + 2μ_0)
    β_1 = β_0 + 2 * λ_0 * μ_0 / (k_0 + 2μ_0)
    B = β_0 + b0μ + β_1 + b1μ
    C = b0μ * β_1 + b1μ * β_0 + β_0 * β_1
    sqD = sqrt(B^2 - 4C)
    δ_3 = (B - sqD) / 2;  δ_4 = (B + sqD) / 2
    b3μ = (β_0 - δ_3) * (δ_3 - β_1) / (δ_3 - δ_4)
    b4μ = (β_0 - δ_4) * (δ_4 - β_1) / (δ_4 - δ_3)
    return (
        1 + b3μ * I_Rabotnov(t, α_0, δ_3)
            + b4μ * I_Rabotnov(t, α_0, δ_4)
    ) / μ_0X
end

function mu_sph_rig_MAX(t, f)
    μ_0X = μ_0 * (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0)) /
        (6 * (1 - f) * (k_0 + 2μ_0))
    b0μ = 2 * (k_0 + 2μ_0) * (3 + 2f) * λ_0 /
        (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0))
    b1μ = 5f * k_0^2 * λ_0 /
        (6 * (k_0 + 2μ_0) + f * (9k_0 + 8μ_0)) / (k_0 + 2μ_0)
    β_1 = β_0 + 2 * λ_0 * μ_0 / (k_0 + 2μ_0)
    return μ_0X * (
        1 + b0μ * I_Rabotnov(t, α_0, β_0)
            + b1μ * I_Rabotnov(t, α_0, β_1)
    )
end

# ─── Build the RVE for a given inclusion stiffness contrast and scheme ─────

function homogenize_shear(
        scheme, f::Real;
        contrast::Symbol = :rigid, n::Int = 200, t_max::Real = 100.0
    )
    times = vcat(0.0, 10 .^ range(-8, log10(t_max); length = n))

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_matrix); fraction = :rest)
    if contrast === :rigid
        C_inc = 1.0e6 * TensISO{3}(3 * k_0, 2 * μ_0)
    elseif contrast === :pore
        C_inc = 1.0e-10 * TensISO{3}(3 * k_0, 2 * μ_0)
    else
        throw(ArgumentError("contrast must be :rigid or :pore"))
    end
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(C_inc));
        fraction = f, symmetrize = :iso
    )

    R̃ = homogenize_alv(rve, scheme, :C; times = times)
    α, β = iso_params_from_blocks(R̃)            # 3K(t,t'), 2μ(t,t')

    # Trace : "step strain at t = 0" → strain history `e ≡ 1` on [0, t_max].
    # Shear modulus μ(t) = (β · 1) [t]   (the 2μ component evaluated at all times)
    # Shear creep   L^μ(t) = ((β)^{-vol} · 1)[t]
    one_vec = ones(eltype(α), n + 1)
    μ_t = (β * one_vec) ./ 2          # 2μ → μ
    Lμ_t = (volterra_inverse(β; block_size = 1) * one_vec) .* 2
    return times, μ_t, Lμ_t
end

# ─── Run scheme combinations and collect (times, μ, L^μ) tuples ────────────

println("Running ALV homogenization for the Rabotnov benchmark…")

results_rigid = Dict{Tuple{Symbol, Float64}, NamedTuple}()
for f in (0.05, 0.1, 0.2)
    for sch in (Maxwell(), Dilute())
        sch_name = sch isa Maxwell ? :Maxwell : :Dilute
        @printf "  rigid  f=%.2f  scheme=%s\n" f sch_name
        times, μ_t, Lμ_t = homogenize_shear(sch, f; contrast = :rigid)
        results_rigid[(sch_name, f)] = (; times, μ_t, Lμ_t)
    end
end

# ─── Plot creep + relaxation overlays (rigid spheres) ──────────────────────

const N_REF = 200
const COLOR_OF_F = Dict(0.05 => :blue, 0.1 => :black, 0.2 => :red)

p_creep = plot(
    xscale = :log10, xlabel = "t [h]",
    ylabel = raw"L^μ(t)  [GPa^{-1}]",
    title = "Rigid spheres — shear creep",
    legend = :bottomright
)
p_relax = plot(
    xscale = :log10, xlabel = "t [h]",
    ylabel = raw"μ(t)  [GPa]",
    title = "Rigid spheres — shear relaxation",
    legend = :topright
)

for f in (0.05, 0.1, 0.2)
    col = COLOR_OF_F[f]
    # Numerical (markers) — skip t = 0 for log-axis safety.
    res_max = results_rigid[(:Maxwell, f)]
    res_dil = results_rigid[(:Dilute, f)]
    keep = 2:length(res_max.times)
    scatter!(
        p_creep, res_max.times[keep], res_max.Lμ_t[keep];
        label = "", color = col, marker = :x, markersize = 4
    )
    scatter!(
        p_creep, res_dil.times[keep], res_dil.Lμ_t[keep];
        label = "", color = col, marker = :x, markersize = 4
    )
    scatter!(
        p_relax, res_max.times[keep], res_max.μ_t[keep];
        label = "", color = col, marker = :x, markersize = 4
    )
    scatter!(
        p_relax, res_dil.times[keep], res_dil.μ_t[keep];
        label = "", color = col, marker = :x, markersize = 4
    )
    # Closed form (lines) — strictly positive log-grid.
    t_ref = 10 .^ range(-7, log10(100.0); length = N_REF)
    plot!(
        p_creep, t_ref, Lmu_sph_rig_MAX.(t_ref, f);
        label = "MAX f=$(f)", color = col, linestyle = :solid, linewidth = 1.5
    )
    plot!(
        p_creep, t_ref, Lmu_sph_rig_DIL.(t_ref, f);
        label = "DIL f=$(f)", color = col, linestyle = :dash, linewidth = 1.5
    )
    plot!(
        p_relax, t_ref, mu_sph_rig_MAX.(t_ref, f);
        label = "", color = col, linestyle = :solid, linewidth = 1.5
    )
    plot!(
        p_relax, t_ref, mu_sph_rig_DIL.(t_ref, f);
        label = "", color = col, linestyle = :dash, linewidth = 1.5
    )
end

fig = plot(
    p_creep, p_relax;
    layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600)
)

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "52_rabotnov_mittag_leffler.png")
savefig(fig, out)
display(fig)
println("Saved : $out")

println()
println("═══════════════════════════════════════════════════════════════════")
println(" Quantitative agreement (numerical ↔ closed-form) at f = 0.10")
println("═══════════════════════════════════════════════════════════════════")
let res = results_rigid[(:Maxwell, 0.1)]
    times = res.times
    keep_idx = findall(t -> t > 0, times)
    rel_max = -Inf
    for i in keep_idx
        Lμ_ref = Lmu_sph_rig_MAX(times[i], 0.1)
        μ_ref = mu_sph_rig_MAX(times[i], 0.1)
        rel_max = max(
            rel_max,
            abs(res.Lμ_t[i] - Lμ_ref) / abs(Lμ_ref),
            abs(res.μ_t[i] - μ_ref) / abs(μ_ref)
        )
    end
    @printf "  Maxwell  rigid  f=0.10 : max relative error vs closed-form = %.2e\n" rel_max
end
let res = results_rigid[(:Dilute, 0.1)]
    times = res.times
    keep_idx = findall(t -> t > 0, times)
    rel_max = -Inf
    for i in keep_idx
        Lμ_ref = Lmu_sph_rig_DIL(times[i], 0.1)
        μ_ref = mu_sph_rig_DIL(times[i], 0.1)
        rel_max = max(
            rel_max,
            abs(res.Lμ_t[i] - Lμ_ref) / abs(Lμ_ref),
            abs(res.μ_t[i] - μ_ref) / abs(μ_ref)
        )
    end
    @printf "  Dilute   rigid  f=0.10 : max relative error vs closed-form = %.2e\n" rel_max
end
println()
println("Reference : @barthelemyIJES2019 §5, eqs. (35)–(43).")
println("═══════════════════════════════════════════════════════════════════")
