# =============================================================================
#  31_local_nlayers.jl
#
#  Radial stress profile σ_rr(r), σ_θθ(r) inside a 2-layer composite
#  sphere (matrix–core–shell) under a remote hydrostatic far-field
#  σ∞ = -p₀ · 𝟙, with hydrostatic
#  rather than uniaxial loading: the displacement field is purely
#  radial `u_r = A_k r + B_k / r²`, every quantity is axisymmetric and
#  reduces to a single 2×2 state-vector recurrence already implemented
#  in `MeanFieldHomogenization.LayeredSpheres.bulk_recurrence`.
#
#  The deviatoric (Y₂-harmonic) profile under a uniaxial far-field
#  needs the per-radius evaluation of the 4×4 shear-recurrence state
#  vector, which is not yet exposed by `MeanFieldHomogenization`.  This script
#  intentionally restricts itself to the bulk part — see
#  `scripts/bench_echoes/benchmark_nlayers.jl` for a cross-check of
#  effective and per-layer averages against the C++ reference.
#
#  Usage:  julia --project scripts/31_local_nlayers.jl
#  Saves : scripts/figures/31_local_nlayers.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# Internal helpers we reuse for the local-state evaluation.
const LS = MeanFieldHomogenization.LayeredSpheres
import MeanFieldHomogenization.LayeredSpheres: _iso_bulk_shear, _bulk_state_seq,
    _bulk_layer_transfer, _bulk_extract_AB

# ─── Per-layer (A, B) coefficients under unit far-field A_∞ = 1 ─────────────

"""
    bulk_AB(sphere, C₀) -> (AB, B_inf)

Per-layer coefficients `(A_k, B_k)` of the bulk displacement
`u_r = A_k r + B_k/r²` inside a `LayeredSphere`, normalized so that
the matrix far-field amplitude `A_∞ = 1`.  Returns `B_inf` (matrix-
side `B`-coefficient) as the second element.
"""
function bulk_AB(sphere::LayeredSphere{T, N}, C₀::TensISO{4, 3}) where {T, N}
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    inside, s_mat = _bulk_state_seq(sphere, κ₀, μ₀)
    radii = sphere.radii
    A_inf, B_inf = _bulk_extract_AB(radii[N], κ₀, μ₀, s_mat[1], s_mat[2])
    inv_A_inf = 1 / A_inf
    AB = ntuple(N) do k
        κ_k, μ_k = _iso_bulk_shear(sphere.moduli[k])
        Ak, Bk = _bulk_extract_AB(radii[k], κ_k, μ_k, inside[k][1], inside[k][2])
        (Ak * inv_A_inf, Bk * inv_A_inf)
    end
    return AB, B_inf * inv_A_inf
end

# ─── Stress at radius r under remote hydrostatic strain ε_v ─────────────────

"""
    bulk_stresses(sphere, C₀, r; ε_v=1.0)

Return `(σ_rr, σ_θθ)` at radius `r` under a remote hydrostatic
strain `ε∞ = ε_v · 𝟙`.  `r > 0` may be inside the composite sphere
(any layer) or in the matrix.  Continuity of σ_rr across perfect
interfaces is automatic; explicit jumps for imperfect interfaces are
not modeled here.
"""
function bulk_stresses(
        sphere::LayeredSphere{T, N}, C₀::TensISO{4, 3}, r;
        ε_v::Real = 1.0
    ) where {T, N}
    AB, B_inf = bulk_AB(sphere, C₀)
    radii = sphere.radii
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    if r ≤ radii[N]
        # Find the layer k such that r_{k-1} < r ≤ r_k.
        k = findfirst(rk -> r ≤ rk, collect(radii))::Int
        Ak, Bk = AB[k]
        κ_k, μ_k = _iso_bulk_shear(sphere.moduli[k])
        # ε∞ = ε_v · 𝟙  ⇒  A_∞_with_unit = ε_v  (because ε_rr_∞ = A_∞).
        # All linear in A_∞: scale (A_k, B_k) by ε_v.
        scale = ε_v
        σ_rr = 3 * κ_k * Ak * scale - 4 * μ_k * Bk * scale / r^3
        σ_θθ = 3 * κ_k * Ak * scale + 2 * μ_k * Bk * scale / r^3
    else
        # Outside the composite sphere — far-field perturbation.
        σ_rr = 3 * κ₀ * ε_v - 4 * μ₀ * (B_inf * ε_v) / r^3
        σ_θθ = 3 * κ₀ * ε_v + 2 * μ₀ * (B_inf * ε_v) / r^3
    end
    return σ_rr, σ_θθ
end

# ─── Setup — Christensen-style three-phase model ─────────────────────────────

# Matrix
const Eo, νo = 30.0, 0.3
const Ko, μo = Eo / (3 * (1 - 2νo)), Eo / (2 * (1 + νo))
const C₀ = TensISO{3}(3 * Ko, 2 * μo)

# Stiff core (e.g. inclusion)
const Ei, νi = 100.0, 0.3
const Ki, μi = Ei / (3 * (1 - 2νi)), Ei / (2 * (1 + νi))
const Ci = TensISO{3}(3 * Ki, 2 * μi)

# Compliant ITZ shell (e.g. interfacial transition zone)
const Eitz, νitz = 0.1 * Ei, 0.2
const Kitz = Eitz / (3 * (1 - 2νitz))
const μitz = Eitz / (2 * (1 + νitz))
const Citz = TensISO{3}(3 * Kitz, 2 * μitz)

const R = 1.0
const ep = 2.0
const sphere2 = LayeredSphere((R, R + ep), (Ci, Citz))
const sphere1 = LayeredSphere((R,), (Ci,))    # single-layer reference

# ─── Apply far-field hydrostatic strain ε_v ─────────────────────────────────

const ε_v = 1.0   # arbitrary unit far-field volumetric strain
const lr = collect(range(1.0e-6, 6 * R; length = 600))

σ_rr_2 = similar(lr); σ_θθ_2 = similar(lr)
σ_rr_1 = similar(lr); σ_θθ_1 = similar(lr)
for (i, r) in enumerate(lr)
    σrr, σθθ = bulk_stresses(sphere2, C₀, r; ε_v = ε_v)
    σ_rr_2[i] = σrr; σ_θθ_2[i] = σθθ
    σrr, σθθ = bulk_stresses(sphere1, C₀, r; ε_v = ε_v)
    σ_rr_1[i] = σrr; σ_θθ_1[i] = σθθ
end

# Far-field reference σ∞_ii = 3 K₀ ε_v.
σ_far = 3 * Ko * ε_v

# ─── Tabular check : continuity of σ_rr, jump of σ_θθ at interfaces ──────────

println("Two-layer composite sphere (core + ITZ shell) under unit hydrostatic strain")
println("─"^78)
@printf "  Matrix (E, ν)        : (%.2f, %.3f)  → K=%.4f, μ=%.4f\n" Eo νo Ko μo
@printf "  Core   (E, ν)        : (%.2f, %.3f)  → K=%.4f, μ=%.4f\n" Ei νi Ki μi
@printf "  ITZ    (E, ν)        : (%.2f, %.3f)  → K=%.4f, μ=%.4f\n" Eitz νitz Kitz μitz
@printf "  R = %.2f, ep = %.2f  → core radius %.2f, shell outer %.2f\n" R ep R R + ep
@printf "  Far-field stress σ∞_ii = 3 K₀ ε_v = %.6f\n\n" σ_far

println("Stress at selected radii (2-layer sphere):")
@printf "  %-8s   %12s   %12s\n" "r" "σ_rr" "σ_θθ"
for r in (0.5 * R, R, R + 0.5 * ep, R + ep, 2 * (R + ep), 4 * (R + ep))
    σrr, σθθ = bulk_stresses(sphere2, C₀, r; ε_v = ε_v)
    @printf "  %-8.4f   %+12.6f   %+12.6f\n" r σrr σθθ
end
println()

# Verify that σ_rr is continuous across interfaces (perfect interfaces).
for r_int in (R, R + ep)
    σ_minus, _ = bulk_stresses(sphere2, C₀, r_int - 1.0e-9; ε_v = ε_v)
    σ_plus, _ = bulk_stresses(sphere2, C₀, r_int + 1.0e-9; ε_v = ε_v)
    @printf "  σ_rr continuity at r=%.2f : Δ = %.3e\n" r_int abs(σ_plus - σ_minus)
end
println()

# ─── Plot ────────────────────────────────────────────────────────────────────

p1 = plot(;
    xlabel = "r", ylabel = "σ_ij / (3 K₀ ε_v)",
    title = "Two-layer sphere (core + ITZ) — hydrostatic far-field",
    legend = :topright, grid = true
)
plot!(p1, lr, σ_rr_2 ./ σ_far; lw = 2, color = :red, label = "σ_rr")
plot!(p1, lr, σ_θθ_2 ./ σ_far; lw = 2, color = :blue, label = "σ_θθ")
hline!(p1, [1.0]; lw = 1, color = :black, linestyle = :dot, label = "σ∞")
vline!(p1, [R]; lw = 1, color = :black, linestyle = :dash, label = "r = R")
vline!(p1, [R + ep]; lw = 1, color = :black, linestyle = :dash, label = "r = R+ep")

p2 = plot(;
    xlabel = "r", ylabel = "σ_ij / (3 K₀ ε_v)",
    title = "Single-layer (core only) — same matrix C₀",
    legend = :topright, grid = true
)
plot!(p2, lr, σ_rr_1 ./ σ_far; lw = 2, color = :red, label = "σ_rr")
plot!(p2, lr, σ_θθ_1 ./ σ_far; lw = 2, color = :blue, label = "σ_θθ")
hline!(p2, [1.0]; lw = 1, color = :black, linestyle = :dot, label = "σ∞")
vline!(p2, [R]; lw = 1, color = :black, linestyle = :dash, label = "r = R")

p_full = plot(
    p1, p2; layout = (1, 2), size = (1400, 500),
    plot_title = "Local bulk stress profile in n-layer sphere"
)

const figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "31_local_nlayers.png")
savefig(p_full, figpath)
display(p_full)
@printf "Saved : %s\n" figpath
