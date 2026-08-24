# # Frequency or time? Two routes to the same viscoelastic composite
#
# `MeanFieldHomogenization` reaches the effective behavior of a linear viscoelastic
# composite by two entirely separate roads:
#
# - the **frequency route** — replace every modulus by its complex counterpart
#   and call [`homogenize`](@ref) unchanged, as in the
#   [viscoelastic composites tutorial](../viscoelasticity.md);
# - the **time route** — [`homogenize_alv`](@ref), which discretizes the
#   Volterra operators on a time grid and never leaves the time domain, as in
#   the [ageing creep application](../../applications/ageing_creep.md).
#
# They share no code. For a **non-ageing** material the correspondence principle
# says they must nevertheless agree, and this page checks that they do —
# quantitatively, and with the discretization error made visible.
#
# ## Why no numerical Laplace inversion is needed
#
# The natural-looking comparison — invert the frequency answer back to the time
# domain — would require a numerical inverse Laplace transform, an ill-posed
# operation whose stability would then be the subject of the page instead of the
# physics. The transform runs the easy way round here: the time route produces a
# sampled relaxation function ``\mu^{\hom}(t)``, and its **Laplace–Carson
# transform**
#
# ```math
# \mu^{*}(p) \;=\; p\int_{0}^{\infty}\mu^{\hom}(t)\,e^{-pt}\,\mathrm{d}t,
# \qquad p = i\omega,
# ```
#
# is a plain quadrature. Evaluated at ``p = i\omega`` it is exactly what the
# frequency route computes directly.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using QuadGK
using Printf
using Plots
gr()  # headless backend; GKSwstype is set to "100" before Literate runs

# ## §1 Two standard-solid phases
#
# Each phase is a **standard solid** (Zener): a relaxation function decaying
# from an instantaneous modulus to a finite long-time modulus,
#
# ```math
# \mu(t) = \mu_\infty + \mu_d\,e^{-t/\tau_\mu},
# ```
#
# with ``\mu_\infty`` the relaxed (long-time) shear modulus, ``\mu_d`` the
# relaxing part, ``\tau_\mu`` the shear relaxation time, and likewise
# ``(k_\infty, k_d, \tau_k)`` in bulk. A finite ``\mu_\infty`` keeps the
# composite from flowing indefinitely, so the relaxation function has a plateau
# that the truncated time grid can actually reach — the one requirement of the
# comparison below.
#
# The Laplace–Carson transform of such a kernel is elementary,
# ``\mathrm{LC}\{a + b\,e^{-t/\tau}\}(p) = a + b\,\dfrac{p\tau}{1+p\tau}``,
# which is what feeds the frequency route.

struct StandardSolid
    k∞::Float64
    k_d::Float64
    τk::Float64
    μ∞::Float64
    μ_d::Float64
    τμ::Float64
end

k_of(z::StandardSolid, t) = z.k∞ + z.k_d * exp(-t / z.τk)
μ_of(z::StandardSolid, t) = z.μ∞ + z.μ_d * exp(-t / z.τμ)

## Time-domain kernel R(t, t') = R(t - t') — non-ageing, hence the difference.
relaxation_law(z::StandardSolid) = ViscoLaw(
    (t, tp) -> t < tp ? TensISO{3}(0.0, 0.0) :
        TensISO{3}(3 * k_of(z, t - tp), 2 * μ_of(z, t - tp)),
    :relaxation,
)

## Laplace-Carson transform of the same kernel, in closed form.
lc(a, b, τ, p) = a + b * p * τ / (1 + p * τ)
stiffness_star(z::StandardSolid, p) = TensISO{3}(
    3 * lc(z.k∞, z.k_d, z.τk, p),
    2 * lc(z.μ∞, z.μ_d, z.τμ, p),
)

const Z_MATRIX = StandardSolid(30.0, 20.0, 1.0, 10.0, 8.0, 0.7)
const Z_INCL = StandardSolid(80.0, 10.0, 2.0, 30.0, 5.0, 1.5)
const F_INCL = 0.3

# ## §2 The frequency route
#
# Nothing viscoelastic happens here: an RVE of `ComplexF64` moduli goes through
# the ordinary Mori-Tanaka scheme, and the effective shear modulus is read off
# the result.

function mu_frequency(ω)
    p = im * ω
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => stiffness_star(Z_MATRIX, p)))
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => stiffness_star(Z_INCL, p));
        fraction = F_INCL
    )
    return TensND.get_data(homogenize(rve, MoriTanaka()))[2] / 2
end

# ## §3 The time route, and how to read a relaxation function out of it
#
# [`homogenize_alv`](@ref) returns the effective operator as a ``6n \times 6n``
# block matrix ``\widetilde{\mathbb{R}}`` acting on a *strain history* sampled on
# the grid — the trapezoidal representation of the Stieltjes integral
# ``\sigma(t_i) = \int_{t_0}^{t_i}\mathbb{R}(t_i,\tau):\mathrm{d}\varepsilon(\tau)``
# ([sanahuja2013](@cite)). Its blocks are *differences* of kernel values, not
# kernel values, so reading ``\mathbb{R}^{\hom}(t)`` off a column would be wrong.
#
# The physical extraction is a relaxation test. Applying a **unit strain step at
# ``t = 0``** means a history vector whose every time slot holds the same strain,
# so the stress at ``t_i`` is the row sum:
#
# ```math
# \mathbb{R}^{\hom}(t_i) \;=\; \sum_j \widetilde{\mathbb{R}}_{ij}.
# ```
#
# [`iso_params_from_blocks`](@ref) splits the block matrix into its two isotropic
# parts ``\alpha = 3k`` and ``\beta = 2\mu``, so the row sums of ``\beta`` give
# ``2\mu^{\hom}(t_i)`` directly.

function mu_relaxation(times)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => relaxation_law(Z_MATRIX)))
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => relaxation_law(Z_INCL));
        fraction = F_INCL
    )
    R̃ = homogenize_alv(rve, MoriTanaka(), :C; times = times)
    _, β = iso_params_from_blocks(R̃)
    return vec(sum(β, dims = 2)) ./ 2      # 2μ(tᵢ) → μ(tᵢ)
end

times = collect(range(0.0, 40.0; length = 401))
μ_t = mu_relaxation(times)

@printf(
    "μ_hom(0) = %.5f  (instantaneous)      μ_hom(T = %.0f) = %.5f  (relaxed)\n",
    μ_t[1], times[end], μ_t[end]
)

# The grid must be long enough for the plateau to be reached: the tail beyond
# ``T`` is then a constant, and its contribution to the transform is the closed
# form ``\mu^{\hom}(T)\,e^{-pT}`` rather than a truncation error.

function mu_from_time(ω, times, μ_t)
    p = im * ω
    interp(t) = begin
        i = searchsortedlast(times, t)
        i ≥ length(times) && return μ_t[end]
        θ = (t - times[i]) / (times[i + 1] - times[i])
        (1 - θ) * μ_t[i] + θ * μ_t[i + 1]
    end
    I, _ = quadgk(t -> interp(t) * exp(-p * t), times[1], times[end]; rtol = 1.0e-11)
    return p * I + μ_t[end] * exp(-p * times[end])
end

# ## §4 The comparison

ωs = exp10.(range(-1.5, 1.5; length = 25))
μ_freq = [mu_frequency(ω) for ω in ωs]
μ_time = [mu_from_time(ω, times, μ_t) for ω in ωs]
rel_err = abs.(μ_freq .- μ_time) ./ abs.(μ_freq)

@printf("\n   ω      Re μ* (freq)  Re μ* (time)  Im μ* (freq)  Im μ* (time)   rel. err\n")
for i in 1:4:length(ωs)
    @printf(
        "%7.3f   %11.5f  %12.5f  %12.5f  %12.5f   %.2e\n",
        ωs[i], real(μ_freq[i]), real(μ_time[i]),
        imag(μ_freq[i]), imag(μ_time[i]), rel_err[i]
    )
end

p1 = plot(
    ωs, real.(μ_freq); xscale = :log10, lw = 2, color = :blue,
    label = "Re — frequency route", xlabel = "ω", ylabel = "μ*_hom(ω)",
    title = "Mori-Tanaka, f = $F_INCL", legend = :right,
)
plot!(p1, ωs, imag.(μ_freq); lw = 2, color = :red, label = "Im — frequency route")
scatter!(p1, ωs, real.(μ_time); marker = :circle, ms = 3, color = :blue,
    markerstrokewidth = 0, label = "Re — time route + LC")
scatter!(p1, ωs, imag.(μ_time); marker = :circle, ms = 3, color = :red,
    markerstrokewidth = 0, label = "Im — time route + LC")

p2 = plot(
    ωs, rel_err; xscale = :log10, yscale = :log10, lw = 2, color = :black,
    xlabel = "ω", ylabel = "relative difference", legend = false,
    title = "Δt = $(round(times[2] - times[1], digits = 3))",
)
plt = plot(
    p1, p2; layout = (1, 2), size = (950, 400),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

# ## §5 The difference is the time step, and nothing else
#
# The two routes do not agree exactly, and they should not: the time route
# integrates the Volterra operator with the trapezoidal rule. Halving ``\Delta t``
# should therefore divide the discrepancy by four — which is a much stronger
# statement than a single small number, because it identifies *what* the
# remaining difference is.

@printf("\n   Δt        relative difference at ω = 1\n")
prev = NaN
for npts in (101, 201, 401)
    tg = collect(range(0.0, 40.0; length = npts))
    e = abs(mu_from_time(1.0, tg, mu_relaxation(tg)) - mu_frequency(1.0)) / abs(mu_frequency(1.0))
    ratio = isnan(prev) ? "" : @sprintf("   (÷ %.2f)", prev / e)
    @printf("%7.3f     %.3e%s\n", tg[2] - tg[1], e, ratio)
    global prev = e
end

# The factor is 4 to two digits: the entire gap between the frequency and the
# time route is the trapezoidal error, and the two implementations agree in the
# continuum limit.
#
# !!! note "This only works because the material does not age"
#     The correspondence principle requires ``\mathbb{R}(t, t')`` to depend on
#     ``t - t'`` alone. When a phase solidifies progressively, ``t`` and ``t'``
#     enter independently, the Laplace–Carson transform no longer factorizes the
#     convolution, and the frequency route simply ceases to exist — which is why
#     [`homogenize_alv`](@ref) is not a redundant second implementation. See the
#     [ageing creep application](../../applications/ageing_creep.md) for a case
#     where only the time route applies.

const figdir = joinpath(@__DIR__, "figures")                        #jl
isdir(figdir) || mkdir(figdir)                                       #jl
figpath = joinpath(figdir, "61_freq_vs_time.png")                    #jl
savefig(plt, figpath)                                                #jl
display(plt)                                                         #jl
@printf "\nSaved : %s\n" figpath                                     #jl
