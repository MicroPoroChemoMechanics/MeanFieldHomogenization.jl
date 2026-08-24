# =============================================================================
#  53_ageing_creep_solid.jl
#
#  Reproduction of the solidifying ageing-creep benchmark
#  (both the **whole-pores** and the **layers** topologies) and the
#  manual chapter `ch09_applications.typ` § "Ageing creep of solidifying
#  cementitious materials".
#
#  Setup : an ageing composite with three phase types
#    * a viscoelastic Maxwell matrix (M),
#    * N solidifying spherical inclusions (each Maxwell, each with its
#      own setting time `t_i^set`),
#    * a single pore (elastic, ~zero stiffness).
#
#  Two topologies (chosen via the `MODEL` constant below) :
#    `:whole_pores` — every layer is a separate ellipsoidal inclusion.
#    `:layers`      — pore + N solidifying shells form a single
#                     `LayeredSphere` inclusion (the *composite-sphere*
#                     topology of @sanahuja2013).  Requires the bulk
#                     and shear ALV recurrences from
#                     `Viscoelasticity/layered_alv.jl`.
#
#  At every loading age t_0 the script computes the effective uniaxial
#  creep compliance `J^E_eff(t, t_0)` via Mori-Tanaka time-domain
#  homogenization, comparing the **history-dependent** (a layer becomes
#  active when its setting time is reached during the experiment) and
#  the **frozen** approach (microstructure is fixed at t_0).
#
#  Usage : julia --project scripts/53_ageing_creep_solid.jl
#  Output : scripts/figures/53_ageing_creep_solid_<model>.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

# ─── Phase moduli ──────────────────────────────────────────────────────────

# Matrix
const E0 = 1.0
const ν0 = 0.2
const k0 = E0 / (3 * (1 - 2ν0))
const μ0_ = E0 / (2 * (1 + ν0))
const f0 = 0.6
const η0 = 0.2     # bulk relaxation time
const γ0 = 0.133   # shear relaxation time

# Solidifying phase
const E1 = 5.0
const ν1 = 0.3
const k1 = E1 / (3 * (1 - 2ν1))
const μ1 = E1 / (2 * (1 + ν1))
const finf = 0.3
const η1 = 1.0
const γ1 = 1.67

# Pore (elastic, near-zero)
const Ep = 1.0e-8 * E0
const νp = 0.2
const kp = Ep / (3 * (1 - 2νp))
const μp = Ep / (2 * (1 + νp))
const fp = 1 - f0 - finf

const C_p_tens = TensISO{3}(3 * kp, 2 * μp)

# Maxwell relaxation kernels.
make_R0() = maxwell_iso(k0, μ0_, η0, γ0)
make_R1() = maxwell_iso(k1, μ1, η1, γ1)

# ─── Solidification kinetics (same as the Python script) ────────────────────

# Volume fraction of solidified material : f(t, α) = finf · t^α / (1 + t^α).
# Layer i is "active" when t' ≥ t_i^set with
#   t_i^set = ((i+0.5)/N · finf / (finf - (i+0.5)/N · finf))^(1/α).
function solidification_setting_times(N::Int, α::Real)
    F = [(i + 0.5) * finf / N for i in 0:(N - 1)]
    return [(f / (finf - f))^(1.0 / α) for f in F]
end

# Inclusion law for layer i.  In `fixed = true` (frozen) mode the
# decision is made at loading time t_0 ; in `fixed = false` (history-
# dependent) mode the layer activates when t' reaches its setting
# time during the actual experiment.
function inclusion_law(t_set::Real, t_0::Real; fixed::Bool)
    if fixed
        # At loading t_0, decide once whether the layer is active.
        if t_0 ≥ t_set
            return make_R1()
        else
            return ViscoLaw(
                (t, tp) -> (t < tp ? zero(C_p_tens) : C_p_tens),
                :relaxation
            )
        end
    else
        # History-dependent : check t' (= tp in our notation).
        R1 = make_R1()
        return ViscoLaw(
            function (t, tp)
                if t < tp
                    return zero(C_p_tens)
                elseif tp ≥ t_set
                    return R1.eval_fun(t, tp)
                else
                    return C_p_tens
                end
            end, :relaxation
        )
    end
end

# ─── Build RVE — whole-pores topology ───────────────────────────────────────

function build_rve_whole_pores(N::Int, α::Real, t_0::Real; fixed::Bool)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => make_R0()))
    # Pore
    add_phase!(
        rve, :PORE, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => heaviside_law(C_p_tens));
        fraction = fp
    )
    # N solidifying spherical layers, each at fraction finf / N.
    t_sets = solidification_setting_times(N, α)
    for i in 1:N
        name = Symbol("INC_$i")
        add_phase!(
            rve, name, Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => inclusion_law(t_sets[i], t_0; fixed = fixed));
            fraction = finf / N
        )
    end
    return rve
end

# ─── Build RVE — layers topology (single LayeredSphere inclusion) ──────────
#
# Mirrors the Python `sphere_nlayers(radius=1, layer_fractions=[fp, finf/N, …])`
# call: the inclusion is a composite sphere with the pore as the
# innermost layer and the N solidifying shells outwards, ordered so the
# outer shell carries `t_set = lT[N-1]` (latest setting time) and the
# innermost solidifying shell carries `t_set = lT[0]` (earliest).

function build_rve_layers(N::Int, α::Real, t_0::Real; fixed::Bool)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => make_R0()))

    t_sets = solidification_setting_times(N, α)

    # Layer thicknesses (in volume fraction) from inside out :
    # pore (fp) + N solidifying shells (finf / N each).
    f_layers = vcat([fp], fill(finf / N, N))
    cumulative = cumsum(f_layers)
    radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)   # absolute scale doesn't matter

    # Per-layer ALV moduli (innermost first).
    moduli = ntuple(N + 1) do k
        if k == 1
            heaviside_law(C_p_tens)
        else
            # k = 2 → t_sets[N] (outermost solidifying = latest setting).
            # k = N+1 → t_sets[1] (innermost solidifying shell = earliest).
            inclusion_law(t_sets[N - k + 2], t_0; fixed = fixed)
        end
    end

    sphere = LayeredSphere(radii, moduli)
    # Volume fraction of the LayeredSphere inclusion in the matrix
    # composite : pore + total solidifying = fp + finf.
    add_phase!(
        rve, :INCLUSION, sphere,
        Dict(:C => heaviside_law(C_p_tens));
        fraction = fp + finf
    )
    return rve
end

function build_rve(N::Int, α::Real, t_0::Real, model::Symbol; fixed::Bool)
    if model === :whole_pores
        return build_rve_whole_pores(N, α, t_0; fixed = fixed)
    elseif model === :layers
        return build_rve_layers(N, α, t_0; fixed = fixed)
    else
        throw(ArgumentError("model must be :whole_pores or :layers (got :$model)"))
    end
end

# ─── Compute uniaxial creep compliance from MT block matrix ─────────────────

# Effective compliance of a 6n × 6n iso-isotropic relaxation matrix R̃ :
# the uniaxial creep compliance is the (1, 1) Mandel block of J̃ = R̃^{-vol}
# summed against a unit-step uniaxial stress
# (cf. ECHOES manual eq-Jeff `J_E^eff(t_i, t_0) = Σ_j [R̃^{-1}]_{11, ij}`).
function uniaxial_creep(R::AbstractMatrix)
    J = volterra_inverse(R; block_size = 6)
    n = size(J, 1) ÷ 6
    return [sum(J[6 * (i - 1) + 1, 6 * (j - 1) + 1] for j in 1:n) for i in 1:n]
end

function creep_curve(
        N::Int, α::Real, t_0::Real, T_grid::AbstractVector{<:Real},
        model::Symbol; fixed::Bool
    )
    rve = build_rve(N, α, t_0, model; fixed = fixed)
    R = homogenize_alv(rve, MoriTanaka(), :C; times = T_grid)
    return uniaxial_creep(R)
end

# ─── Main run ────────────────────────────────────────────────────────────────

# We use N = 20 layers (faster than the Python's N = 100, still enough
# resolution for the qualitative trends) and α = 4.
const N = 20
const α_solid = 4.0
const MODEL = :layers          # :whole_pores or :layers

const loading_ages = (1 / 3, 2 / 3, 4 / 3, 2.0, 8 / 3)
const t_max = 10 / 3
const npts_per_curve = 41

println(
    "Ageing creep — solidifying composite (N = $N layers, α = $α_solid, " *
        "topology = :$MODEL)"
)
println("─"^70)

# Plot.
p = plot(;
    xlabel = "t", ylabel = "E₀ · J^E_{eff}(t, t₀)",
    title = "Ageing creep — Maxwell matrix + Maxwell solidifying inclusions" *
        " ($(MODEL))",
    legend = :topleft, grid = true,
    xlims = (0, t_max), ylims = (0, 15)
)

const colors = [:viridis, :plasma, :magma, :inferno, :turbo]
const cmap = palette(:viridis, length(loading_ages))

J_start = Float64[]     # E₀ · J^E_eff(t_0, t_0), for the elastic cross-check

for (k, t_0) in enumerate(loading_ages)
    @printf "  loading age t_0 = %.4f\n" t_0
    t_eff = t_0 == 0.0 ? 1.0e-4 : t_0
    T_grid = collect(range(t_eff, t_max; length = npts_per_curve))

    J_hist = creep_curve(N, α_solid, t_0, T_grid, MODEL; fixed = false)
    J_frozen = creep_curve(N, α_solid, t_0, T_grid, MODEL; fixed = true)
    push!(J_start, E0 * J_hist[1])

    plot!(p, T_grid, E0 .* J_hist; lw = 2, color = cmap[k], label = "history t₀=$(round(t_0; digits = 2))")
    plot!(p, T_grid, E0 .* J_frozen; lw = 2, color = cmap[k], linestyle = :dash, label = "frozen t₀=$(round(t_0; digits = 2))")
end

# ─── Elastic reference — purely elastic pipeline, same topology ─────────────
#
# `1 / E^hom(t)` is the instantaneous (glassy) compliance of the
# microstructure frozen at time t : every layer whose setting time has
# been reached carries its *elastic* stiffness C₁, the others are still
# pores.  It is computed with the elastic `homogenize`, NOT with
# `homogenize_alv` on a one-point grid — the latter would merely echo the
# starting point of the creep curves, whereas this is an independent
# cross-check of the ALV pipeline.
#
# The reference MUST use the same topology as the creep curves : the
# `:layers` (composite sphere) and `:whole_pores` (N + 1 separate
# ellipsoidal phases) morphologies do **not** give the same MT estimate
# (they differ by ~23 % at t = 2/3 here).

const C_0_el = TensISO{3}(3 * k0, 2 * μ0_)
const C_1_el = TensISO{3}(3 * k1, 2 * μ1)

# Elastic stiffness of solidifying layer i at time t.
layer_stiffness(t::Real, t_set::Real) = (t ≥ t_set) ? C_1_el : C_p_tens

function build_elastic_rve_whole_pores(N::Int, α::Real, t::Real)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_0_el))
    add_phase!(
        rve, :PORE, Ellipsoid(1.0, 1.0, 1.0),
        Dict(:C => C_p_tens); fraction = fp
    )
    t_sets = solidification_setting_times(N, α)
    for i in 1:N
        add_phase!(
            rve, Symbol("INC_$i"), Ellipsoid(1.0, 1.0, 1.0),
            Dict(:C => layer_stiffness(t, t_sets[i])); fraction = finf / N
        )
    end
    return rve
end

function build_elastic_rve_layers(N::Int, α::Real, t::Real)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_0_el))
    t_sets = solidification_setting_times(N, α)
    f_layers = vcat([fp], fill(finf / N, N))
    cumulative = cumsum(f_layers)
    radii = ntuple(k -> cumulative[k]^(1 / 3), N + 1)
    # Same layer ordering as `build_rve_layers`: pore innermost, then the
    # solidifying shells with the latest setting time outermost.
    moduli = ntuple(N + 1) do k
        k == 1 ? C_p_tens : layer_stiffness(t, t_sets[N - k + 2])
    end
    add_phase!(
        rve, :INCLUSION, LayeredSphere(radii, moduli),
        Dict(:C => C_p_tens); fraction = fp + finf
    )
    return rve
end

function elastic_compliance(t::Real, N::Int, α::Real, model::Symbol)
    rve = model === :layers ? build_elastic_rve_layers(N, α, t) :
        build_elastic_rve_whole_pores(N, α, t)
    C_hom = homogenize(rve, MoriTanaka(), :C)
    K_hom, μ_hom = TensND.get_data(C_hom)[1] / 3, TensND.get_data(C_hom)[2] / 2
    E_hom = 9 * K_hom * μ_hom / (3 * K_hom + μ_hom)
    return E0 / max(E_hom, 1.0e-12)
end

t_sets_for_ref = solidification_setting_times(N, α_solid)
# Sample the setting times (where 1 / E^hom jumps) *and* the loading ages,
# so that the start of every creep curve can be read against the reference
# without interpolation error.
T_ref = sort(
    vcat(
        [1.0e-3], filter(t -> t ≤ t_max, t_sets_for_ref),
        collect(loading_ages), [t_max]
    )
)
J_elastic = [elastic_compliance(t, N, α_solid, MODEL) for t in T_ref]
plot!(
    p, T_ref, J_elastic; lw = 2, color = :black, linestyle = :dot,
    label = "1 / E^hom(t)  (elastic)"
)

# Cross-check : the ALV creep curve must start exactly on the elastic
# reference, `J^E_eff(t_0, t_0) = 1 / E^hom(t_0)`.  The two are computed by
# entirely different code paths (time-domain Volterra algebra vs the
# elastic Mori-Tanaka estimate), so agreement is a genuine validation.
println("\nCross-check  E₀ · J^E_eff(t₀, t₀)  vs  E₀ / E^hom(t₀) :")
@printf "  %8s  %12s  %12s  %10s\n" "t₀" "ALV start" "elastic" "rel. err."
for (k, t_0) in enumerate(loading_ages)
    ref = elastic_compliance(t_0, N, α_solid, MODEL)
    @printf "  %8.4f  %12.6f  %12.6f  %10.2e\n" t_0 J_start[k] ref abs(J_start[k] - ref) / ref
end

const figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "53_ageing_creep_solid_$(MODEL).png")
savefig(p, figpath)
display(p)
@printf "\nSaved : %s\n" figpath
