# =============================================================================
#  scripts/bench_echoes/benchmark_nlayers.jl
#
#  Verification of MeanFieldHomogenization.jl `LayeredSphere` against four
#  independent references:
#
#   § 1  **Bulk α_k and shear β_k** vs `echoes.layer_eE` (volume-averaged
#        strain concentration in each layer).
#   § 2  **Internal consistency**: Julia state-vector recurrence vs
#        a direct 8×8 linear-system solver assembled from the same mode
#        formulas (sanity check that the recurrence implements the
#        correct boundary-value problem).
#   § 3  **Analytical limits**: shear concentration `β_k` in degenerate
#        configurations (vanishing core, vanishing shell, core ≡ shell)
#        compared to the closed-form single-layer Eshelby result.
#   § 4  **Local bulk stress profile** `σ_rr(r), σ_θθ(r)` vs
#        `echoes.loc_sS` under remote hydrostatic loading.
#
#  History of the shear (β_k) ECHOES comparison
#  --------------------------------------------
#  An earlier revision of this file compared β_k against ECHOES, found a
#  1–50 % disagreement on genuine multi-layer stacks, attributed it to an
#  `echoes.layer_eE` indexing convention, and fell back to §3's analytical
#  limits.  That conclusion was wrong, and is kept here so nobody redoes
#  the same reasoning:
#
#    * β_k was then computed as the bare mode-1 amplitude a_k, dropping the
#      mode-2 term b_k·F_k (see `_shear_concentration_multi`).  The omission
#      cancels in every degenerate configuration — which is exactly what §3
#      tests — hence "agrees with the analytical limits but not with ECHOES".
#      The same bug was found and fixed in §2's direct 8×8 solver.
#    * The `layer++` explanation cannot hold: α_k and β_k are read from the
#      *same* `layer_eE(k)` matrix, so a layer-index error would have broken
#      α_k too, and α_k matched to 5e-13 throughout.
#    * Independently, the ALV per-layer β(t,t') Volterra blocks are pinned to
#      ECHOES Python at 1e-16 on the diagonal
#      (`test/Viscoelasticity/test_layered_alv.jl`), which is impossible if
#      the elastic β_k were 1–50 % off.
#
#  β_k is therefore compared against ECHOES in §1, like α_k. §3's analytical
#  limits are kept as an independent check, not as a substitute.
#
#  Run from the `MeanFieldHomogenization.jl` package root:
#    julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark_nlayers.jl
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Random
using PythonCall
using Plots

# The pointwise field is public API since v0.10.0, so the local-stress section
# no longer reaches into the recurrence.  Two internals remain, and only for §2,
# which cross-checks the recurrence against an independent 8×8 direct solve —
# that check exists precisely to be written in terms of the raw fundamental
# matrix rather than the API it validates.
import MeanFieldHomogenization.LayeredSpheres: _shear_M_matrix,
    _layer_avg_dev_shear_factor

# ─── Python-side wrappers ────────────────────────────────────────────────────

pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))
const NL = pyimport("nlayers_echoes")

# Julia vectors cross the bridge as Python lists, and every array that comes
# back is a Julia matrix, so that the rest of the script reads as before.
const py_stiff_kmu = NL.py_stiff_kmu
py_make_nlayers(radii, props, Cref) = NL.py_make_nlayers(pylist(radii), pylist(props), Cref)
py_layer_eE(spn, k) = pymat(NL.py_layer_eE(spn, k))
py_eE_total(spn) = pymat(NL.py_eE(spn))
py_sE_total(spn) = pymat(NL.py_sE(spn))
py_layer_frac(spn, k) = pyconvert(Float64, NL.py_layer_fraction(spn, k))
py_conc_sS(spn, r, theta, phi) = pymat(NL.py_conc_sS(spn, r, theta, phi))
py_conc_all(spn, r, theta, phi, layer) = Tuple(pymat(x) for x in NL.py_conc_all(spn, r, theta, phi, layer))
py_make_nlayers_itf(radii, props, ref, tags, p1s, p2s) =
    NL.py_make_nlayers_itf(pylist(radii), pylist(props), ref, pylist(tags), pylist(p1s), pylist(p2s))

# ─── Helpers ─────────────────────────────────────────────────────────────────

# (E, ν) → (K, μ).
function _Kmu_Enu(E, ν)
    K = E / (3 * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    return K, μ
end

# Convert ECHOES 6×6 Voigt-Mandel stiffness array (standard `Cref.array`) to
# Julia (3K, 2μ) iso-data: `α = (KM[1,1] + 2 KM[1,2]) = 3K`, similarly for β.
function _iso_data_from_echoes6x6(KM::AbstractMatrix)
    α = KM[1, 1] + 2 * KM[1, 2]   # = 3K
    β = KM[1, 1] - KM[1, 2]       # = 2μ
    return α, β
end

# Convert layer fractions to ascending radii given outer R.
function _radii_from_fractions(f::AbstractVector{<:Real}, R::Real)
    f_norm = f ./ sum(f)
    cum = zero(R); radii = similar(f_norm, typeof(R))
    for k in eachindex(f_norm)
        cum += f_norm[k] * R^3
        radii[k] = cbrt(cum)
    end
    return radii
end

relerr(a, b) = (abs(a) + abs(b) < 1.0e-14) ? 0.0 : abs(a - b) / max(abs(a), abs(b))

# ─── §1 + §2  Random n-layer cross-check ────────────────────────────────────

println("="^78)
println("§1  Bulk α_k and shear β_k vs ECHOES layer_eE (random n-layer configs)")
println("="^78)

const rtol_match = 1.0e-8

Random.seed!(20260426)

const N_CONFIGS = 30
const N_LAYERS_RANGE = 2:8

n_pass_α = 0; n_fail_α = 0
worst_α_err = 0.0
n_pass_β = 0; n_fail_β = 0
worst_β_err = 0.0

for cfg in 1:N_CONFIGS
    n = rand(N_LAYERS_RANGE)
    R_outer = 1.0 + 4.0 * rand()

    fractions = rand(n) .+ 0.05
    radii = _radii_from_fractions(fractions, R_outer)

    E_lay = 1.0 .+ 99.0 .* rand(n)
    ν_lay = 0.05 .+ 0.4 .* rand(n)
    Kμ_lay = [_Kmu_Enu(E_lay[k], ν_lay[k]) for k in 1:n]

    E_ref = 1.0 + 99.0 * rand()
    ν_ref = 0.05 + 0.4 * rand()
    K_ref, μ_ref = _Kmu_Enu(E_ref, ν_ref)

    C_layers_jl = ntuple(k -> TensISO{3}(3 * Kμ_lay[k][1], 2 * Kμ_lay[k][2]), n)
    C_ref_jl = TensISO{3}(3 * K_ref, 2 * μ_ref)
    sphere_jl = LayeredSphere(Tuple(radii), C_layers_jl)

    C_layers_py = [py_stiff_kmu(Kμ_lay[k][1], Kμ_lay[k][2]) for k in 1:n]
    C_ref_py = py_stiff_kmu(K_ref, μ_ref)
    spn_py = py_make_nlayers(radii, C_layers_py, C_ref_py)

    cfg_α_err = 0.0
    cfg_β_err = 0.0
    for k in 1:n
        A_jl = gradient_gradient_concentration(sphere_jl, C_ref_jl; layer = k)
        α_jl, β_jl = TensND.get_data(A_jl)
        eE_py = py_layer_eE(spn_py, k - 1)
        α_py, β_py = _iso_data_from_echoes6x6(eE_py)
        cfg_α_err = max(cfg_α_err, relerr(α_jl, α_py))
        cfg_β_err = max(cfg_β_err, relerr(β_jl, β_py))
    end

    pass_α = cfg_α_err ≤ rtol_match
    pass_α ? (global n_pass_α += 1) : (global n_fail_α += 1)
    global worst_α_err = max(worst_α_err, cfg_α_err)

    pass_β = cfg_β_err ≤ rtol_match
    pass_β ? (global n_pass_β += 1) : (global n_fail_β += 1)
    global worst_β_err = max(worst_β_err, cfg_β_err)
end

@printf "  α_k :  %d/%d configs within rtol = %.0e (worst rerr = %.3e)\n" n_pass_α (n_pass_α + n_fail_α) rtol_match worst_α_err
@printf "  β_k :  %d/%d configs within rtol = %.0e (worst rerr = %.3e)\n" n_pass_β (n_pass_β + n_fail_β) rtol_match worst_β_err
println()

# ─── §2  Internal consistency : Julia recurrence vs direct 8×8 solver ───────

println("="^78)
println("§2  β_k self-consistency : recurrence vs direct 8×8 linear-system")
println("="^78)

# Direct 8×8 solver for the 2-layer Y₂-harmonic shear problem.
# Unknowns x = (a₁, b₁, a₂, b₂, c₂, d₂, c_∞, d_∞);  c₁ = d₁ = 0 enforced.
# BC at r = ∞ : matrix mode-1 amplitude = 1, mode-2 amplitude = 0.
#
# β_layer1 is the layer-VOLUME-AVERAGED deviatoric strain concentration, not
# the bare mode-1 amplitude a₁: the core carries both the uniform mode 1 (a₁)
# and the r³-varying mode 2 (b₁), whose Y₂-projected volume average adds
# `b₁ · F₁` with the Christensen-Lo factor F₁ = _layer_avg_dev_shear_factor.
# (Earlier this returned a₁ alone — correct only in the degenerate limits
# where b₁ → 0, so it agreed with §3 but disagreed with the recurrence by a
# few % on general configs.)
function direct_2layer_β_layer1(r1, r2, κc, μc, κs, μs, κm, μm)
    Mc_r1 = _shear_M_matrix(r1, κc, μc)
    Ms_r1 = _shear_M_matrix(r1, κs, μs)
    Ms_r2 = _shear_M_matrix(r2, κs, μs)
    Mm_r2 = _shear_M_matrix(r2, κm, μm)
    A = zeros(8, 8); b = zeros(8)
    A[1:4, 1:2] = Mc_r1[:, 1:2]
    A[1:4, 3:6] = -Ms_r1
    A[5:8, 3:6] = Ms_r2
    A[5:8, 7:8] = -Mm_r2[:, 3:4]
    b[5:8] = Mm_r2[:, 1]
    x = A \ b
    a₁, b₁ = x[1], x[2]
    return a₁ + b₁ * _layer_avg_dev_shear_factor(0.0, r1, κc, μc)
end

n_pass_self = 0; worst_self_err = 0.0
for cfg in 1:20
    Kc = 1 + 99 * rand(); μc = 0.5 + 49.5 * rand()
    Ks = 1 + 99 * rand(); μs = 0.5 + 49.5 * rand()
    Km = 1 + 99 * rand(); μm = 0.5 + 49.5 * rand()
    r1 = 0.1 + 0.8 * rand()

    sphere = LayeredSphere(
        (r1, 1.0),
        (TensISO{3}(3Kc, 2μc), TensISO{3}(3Ks, 2μs))
    )
    C0 = TensISO{3}(3Km, 2μm)
    _, β_recurrence = TensND.get_data(gradient_gradient_concentration(sphere, C0; layer = 1))
    β_direct = direct_2layer_β_layer1(r1, 1.0, Kc, μc, Ks, μs, Km, μm)
    err = relerr(β_recurrence, β_direct)
    err < rtol_match && (global n_pass_self += 1)
    global worst_self_err = max(worst_self_err, err)
end
@printf "  %d/20 configs within rtol = %.0e (worst rerr = %.3e)\n" n_pass_self rtol_match worst_self_err
println()

# ─── §3  Analytical-limit check : β in degenerate configurations ────────────

println("="^78)
println("§3  β_layer1 vs analytical Eshelby in degenerate limits")
println("="^78)

# Single-layer Eshelby strain concentration for a sphere of moduli (μ₁) in
# matrix (κ₀, μ₀):  β_∞ = 1 / (1 + α_dev (μ₁/μ₀ − 1))  with
# α_dev = 6(κ₀+2μ₀) / (5(3κ₀+4μ₀)).
function β_eshelby_sphere(μ1, κ0, μ0)
    α_dev = 6 * (κ0 + 2μ0) / (5 * (3κ0 + 4μ0))
    return 1 / (1 + α_dev * (μ1 / μ0 - 1))
end

const Kc, μc = 80.0, 30.0
const Ks, μs = 20.0, 8.0
const Km, μm = 50.0, 20.0
const C_ref_lim = TensISO{3}(3Km, 2μm)
const C_core = TensISO{3}(3Kc, 2μc)
const C_shell = TensISO{3}(3Ks, 2μs)

# Limit 1 : shell ≡ matrix → β_layer1 = single-layer Eshelby (core in matrix).
let
    sphere = LayeredSphere((0.5, 1.0), (C_core, C_ref_lim))
    _, β_jl = TensND.get_data(gradient_gradient_concentration(sphere, C_ref_lim; layer = 1))
    β_an = β_eshelby_sphere(μc, Km, μm)
    @printf "  shell ≡ matrix      :  Julia β = %.10f   analytical = %.10f   relerr = %.2e\n" β_jl β_an relerr(β_jl, β_an)
end

# Limit 2 : core ≡ shell → β_layer1 = single-layer Eshelby for sphere of
# core moduli at full radius (radius 1) in matrix.
let
    sphere = LayeredSphere((0.5, 1.0), (C_core, C_core))
    _, β_jl = TensND.get_data(gradient_gradient_concentration(sphere, C_ref_lim; layer = 1))
    β_an = β_eshelby_sphere(μc, Km, μm)
    @printf "  core ≡ shell        :  Julia β = %.10f   analytical = %.10f   relerr = %.2e\n" β_jl β_an relerr(β_jl, β_an)
end

# Limit 3 : tiny core (r₁ → 0) → β_layer1 = β_inner_Eshelby × β_shell_Eshelby
let
    r1 = 1.0e-4
    sphere = LayeredSphere((r1, 1.0), (C_core, C_shell))
    _, β_jl = TensND.get_data(gradient_gradient_concentration(sphere, C_ref_lim; layer = 1))
    β_shell = β_eshelby_sphere(μs, Km, μm)
    β_core_in_shell = β_eshelby_sphere(μc, Ks, μs)
    β_an = β_core_in_shell * β_shell
    @printf "  r₁ = 1e-4 (vanishing core) :  Julia β = %.10f   analytical = %.10f   relerr = %.2e\n" β_jl β_an relerr(β_jl, β_an)
end

# Limit 4 : N=3 homogeneous (all layers equal to matrix) → β_k = 1 ∀ k.
let
    sphere = LayeredSphere((0.3, 0.7, 1.0), (C_ref_lim, C_ref_lim, C_ref_lim))
    βs = [TensND.get_data(gradient_gradient_concentration(sphere, C_ref_lim; layer = k))[2] for k in 1:3]
    @printf "  homogeneous N=3     :  β = (%.10f, %.10f, %.10f)   (expected 1.0)\n" βs[1] βs[2] βs[3]
end
println()

# ─── §4  Local bulk profile vs `conc_sS` under hydrostatic load ──────────────

println("="^78)
println("§4  Local stress profile (hydrostatic) vs ECHOES conc_sS")
println("="^78)

# Use the Christensen-style 2-layer setup of script 32_local_nlayers.jl.
const Eo, νo = 30.0, 0.3
const Ei, νi = 100.0, 0.3
const Eitz, νitz = 0.1 * Ei, 0.2

K_o, μ_o = _Kmu_Enu(Eo, νo)
K_i, μ_i = _Kmu_Enu(Ei, νi)
K_itz, μ_itz = _Kmu_Enu(Eitz, νitz)

const C_ref_local_jl = TensISO{3}(3 * K_o, 2 * μ_o)
const C_ref_local_py = py_stiff_kmu(K_o, μ_o)
const C_layers_local_py = [py_stiff_kmu(K_i, μ_i), py_stiff_kmu(K_itz, μ_itz)]

const R_inner = 1.0
const ep_layer = 2.0
const radii_local = [R_inner, R_inner + ep_layer]
const sphere_local_jl = LayeredSphere(
    (R_inner, R_inner + ep_layer),
    (
        TensISO{3}(3 * K_i, 2 * μ_i),
        TensISO{3}(3 * K_itz, 2 * μ_itz),
    )
)
const spn_local_py = py_make_nlayers(radii_local, C_layers_local_py, C_ref_local_py)

# Radial stress profile straight from the public pointwise API.  This used to
# reach into `_bulk_state_seq` / `_bulk_extract_AB` and re-derive the (A, B)
# coefficients by hand, and it covered the hydrostatic part only because the
# deviatoric amplitudes were not exposed.  Both limitations are gone.
const sol_local_jl = LayeredSphereFields(sphere_local_jl, C_ref_local_jl)

"Radial and hoop stress at radius `r` under a remote hydrostatic strain `ε_v 𝟙`."
function bulk_stresses(sol, r; ε_v::Real = 1.0)
    # At (r, 0, 0) the first canonical axis is radial and the other two are hoop.
    σ = local_flux(sol, [r, 0.0, 0.0], ε_v * TensISO{3}(1.0))
    return σ[1, 1], σ[2, 2]
end

# Far-field hydrostatic strain ε_v ⇒ remote uniaxial in *each* direction.
# In ECHOES Voigt-Mandel convention, `conc_sS(r, θ, φ)` returns a 6×6
# matrix σ_local = M · σ_∞.  For hydrostatic σ∞ = 3K₀ ε_v · 𝟙, a
# Voigt 6-vector representation is `(3 K₀ ε_v, 3 K₀ ε_v, 3 K₀ ε_v, 0, 0, 0)`.
# In the spherical-symmetric basis at point (r, 0, 0) (i.e. on the x axis),
# `σ_rr_local = M · S` projected on Voigt index 1 (xx).
const ε_v = 1.0
const σ_far = 3 * K_o * ε_v
const Σ_inf_voigt = [σ_far, σ_far, σ_far, 0.0, 0.0, 0.0]   # σ∞ in Voigt-6

const lr_check = collect(range(0.05 * R_inner, 5 * (R_inner + ep_layer); length = 80))
σ_rr_jl = similar(lr_check); σ_θθ_jl = similar(lr_check)
σ_rr_py = similar(lr_check); σ_θθ_py = similar(lr_check)

for (i, r) in enumerate(lr_check)
    σrr, σθθ = bulk_stresses(sol_local_jl, r; ε_v = ε_v)
    σ_rr_jl[i] = σrr; σ_θθ_jl[i] = σθθ
    # ECHOES at (r, θ=0, φ=0) — local frame has e_z radial, so σ_rr ↔ Voigt index 3.
    M = py_conc_sS(spn_local_py, r, 0.0, 0.0)
    σ_local = M * Σ_inf_voigt
    σ_rr_py[i] = σ_local[3]   # zz in local frame = rr at (θ=0, φ=0)
    σ_θθ_py[i] = σ_local[1]   # xx local = θθ
end

# Relative error sweep.
errs_rr = [relerr(σ_rr_jl[i], σ_rr_py[i]) for i in eachindex(lr_check)]
errs_θθ = [relerr(σ_θθ_jl[i], σ_θθ_py[i]) for i in eachindex(lr_check)]
@printf "Local stress max relerr — σ_rr : %.3e, σ_θθ : %.3e (over %d points)\n\n" maximum(errs_rr) maximum(errs_θθ) length(lr_check)

# Plot Julia vs ECHOES.
p_local = plot(;
    xlabel = "r", ylabel = "σ_ij / σ∞",
    title = "Local stress profile — hydrostatic far-field",
    legend = :topright, grid = true
)
plot!(p_local, lr_check, σ_rr_jl ./ σ_far; lw = 2, color = :red, label = "σ_rr (Julia)")
plot!(
    p_local, lr_check, σ_rr_py ./ σ_far; lw = 0, marker = :circle, ms = 4,
    color = :red, label = "σ_rr (ECHOES)"
)
plot!(p_local, lr_check, σ_θθ_jl ./ σ_far; lw = 2, color = :blue, label = "σ_θθ (Julia)")
plot!(
    p_local, lr_check, σ_θθ_py ./ σ_far; lw = 0, marker = :diamond, ms = 4,
    color = :blue, label = "σ_θθ (ECHOES)"
)
hline!(p_local, [1.0]; lw = 1, color = :black, linestyle = :dot, label = "σ∞")
vline!(p_local, [R_inner]; lw = 1, color = :black, linestyle = :dash, label = "")
vline!(p_local, [R_inner + ep_layer]; lw = 1, color = :black, linestyle = :dash, label = "")

# =============================================================================
#  §5  POINTWISE concentration tensors — the four couplings, three interface
#      families.  This is the check that pins `LayeredSpheres/localfields.jl`
#      against an independent implementation: whole 6×6 Kelvin-Mandel matrices,
#      inside every layer, in the matrix, and exactly ON an interface with the
#      region forced on both sides.
#
#      Conventions reconciled here: ECHOES returns rows in the local spherical
#      basis, hence the `rotation_tensor4(θ, φ)` on the Python side; and its `PRIMALDISC`
#      takes interface STIFFNESSES, the same convention as `SpringInterface`
#      since v0.10.0, so the same numbers go to both.
# =============================================================================

println("\n" * "="^78)
println("§5  Pointwise concentration — conc_eE / conc_sE / conc_eS / conc_sS")
println("="^78)

const KN_SPRING, KT_SPRING = 47.6, 27.1
const KS_MEMB, MS_MEMB = 1.3, 0.7
const P_PERF = PerfectInterface{Float64}()

# (label, Julia interfaces, Python tags, first parameters, second parameters)
const ITF_CASES = (
    ("perfect ", (P_PERF, P_PERF), ["perfect", "perfect"], [0.0, 0.0], [0.0, 0.0]),
    (
        "spring  ",
        (SpringInterface(KN_SPRING, KT_SPRING), P_PERF),
        ["spring", "perfect"], [KN_SPRING, 0.0], [KT_SPRING, 0.0],
    ),
    (
        "membrane",
        (P_PERF, MembraneInterface(KS_MEMB, MS_MEMB)),
        ["perfect", "membrane"], [0.0, KS_MEMB], [0.0, MS_MEMB],
    ),
)

# (r, θ, φ, layer)  — `layer` is 0-based ECHOES / `nothing` = let both decide.
const LOCAL_POINTS = (
    (0.4 * R_inner, 0.7, 0.9, nothing),
    (0.999 * R_inner, 1.2, -0.4, nothing),
    (0.5 * (R_inner + radii_local[2]), 0.3, 2.1, nothing),
    (0.9999 * radii_local[2], 2.4, 0.15, nothing),
    (1.3 * radii_local[2], 1.0, 0.6, nothing),
    (4.0 * radii_local[2], 0.45, -1.3, nothing),
    (R_inner, 0.7, 0.9, 0),          # on the inner interface, inside
    (R_inner, 0.7, 0.9, 1),          # on the inner interface, outside
    (radii_local[2], 0.7, 0.9, 1),     # on the outer interface, inside
    (radii_local[2], 0.7, 0.9, 2),     # on the outer interface, matrix side
)

@printf "  %-9s  %10s  %10s  %10s  %10s\n" "interface" "conc_eE" "conc_sE" "conc_eS" "conc_sS"
println("  " * "-"^58)

worst_local = 0.0
for (label, itf_jl, tags, p1s, p2s) in ITF_CASES
    sph = LayeredSphere(
        (R_inner, radii_local[2]),
        (TensISO{3}(3 * K_i, 2 * μ_i), TensISO{3}(3 * K_itz, 2 * μ_itz));
        interfaces = itf_jl,
    )
    sol = LayeredSphereFields(sph, C_ref_local_jl)
    spn = py_make_nlayers_itf(radii_local, C_layers_local_py, C_ref_local_py, tags, p1s, p2s)

    w = zeros(4)
    for (r, θ, φ, lay) in LOCAL_POINTS
        py_lay = lay === nothing ? -1 : lay
        jl_lay = lay === nothing ? nothing : lay + 1        # Julia is 1-based
        ref = py_conc_all(spn, r, θ, φ, py_lay)
        got = (
            Matrix(mandel(local_gradient_gradient_concentration(sol, r, θ, φ; layer = jl_lay))),
            Matrix(mandel(local_flux_gradient_concentration(sol, r, θ, φ; layer = jl_lay))),
            Matrix(mandel(local_gradient_flux_concentration(sol, r, θ, φ; layer = jl_lay))),
            Matrix(mandel(local_flux_flux_concentration(sol, r, θ, φ; layer = jl_lay))),
        )
        for q in 1:4
            scale = max(1.0, maximum(abs, ref[q]))
            w[q] = max(w[q], maximum(abs, got[q] .- ref[q]) / scale)
        end
    end
    global worst_local = max(worst_local, maximum(w))
    @printf "  %-9s  %10.2e  %10.2e  %10.2e  %10.2e\n" label w[1] w[2] w[3] w[4]
end
@printf "\n  worst pointwise discrepancy over all cases : %.3e\n" worst_local
println(worst_local < 1.0e-11 ? "  PASS" : "  FAIL — investigate before releasing")

const figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "benchmark_nlayers.png")
savefig(p_local, figpath)
@printf "\nSaved : %s\n" figpath
