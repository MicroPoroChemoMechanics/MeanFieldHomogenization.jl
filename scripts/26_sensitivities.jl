# =============================================================================
#  26_sensitivities.jl
#
#  Tour of the autodiff sensitivity API of MeanFieldHomogenization:
#    * derivative(rve, scheme, lens; indexer)        — scalar sensitivity,
#    * gradient(rve, scheme, lenses; indexer)        — multi-parameter gradient,
#    * jacobian(rve, scheme, lenses)                 — full Jacobian.
#
#  Four lens types illustrated:
#    * `amount(:I)`                  — phase volume fraction,
#    * `property(:I, :C, :bulk)`     — scalar coefficient of a tensor,
#    * `geometry(:I, :semi_axes, 3)` — scalar geometry field,
#    * `shape_param(:semi_axes, 1)`  — distribution-shape field (PCW).
#
#  The script ends with a cross-check against the Christensen 1990 closed
#  form for ∂k_MT/∂f.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using ForwardDiff
using TensND
using Printf

println("="^78)
println("MeanFieldHomogenization — autodiff sensitivities tour")
println("="^78)

# 2-phase RVE with spherical inclusions ──────────────────────────────────────
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
add_phase!(
    rve, :I, Ellipsoid(1.0),
    Dict(:C => TensISO{3}(60.0, 20.0));
    fraction = 0.2
)

println("\nRVE :")
display(rve)
println()

# Canonical indexer: the (1,1,1,1) component of the effective tensor.
const idxC = C -> get_array(C)[1, 1, 1, 1]

# ── 1) Scalar sensitivity w.r.t. the volume fraction ────────────────────────
∂_f = derivative(rve, MoriTanaka(), amount(:I); indexer = idxC)
@printf "\n[1] ∂C[1111]/∂f_I    (Mori-Tanaka)  = %.6f\n" ∂_f

# ── 2) Sensitivity w.r.t. the inclusion bulk modulus ────────────────────────
∂_K = derivative(rve, MoriTanaka(), property(:I, :C, :bulk); indexer = idxC)
@printf "[2] ∂C[1111]/∂K_I    (Mori-Tanaka)  = %.6f\n" ∂_K

# ── 3) Geometry sensitivity (sphere → the effect is zero, but the machinery
#       runs and demonstrates the lens API).
∂_a3 = derivative(rve, MoriTanaka(), geometry(:I, :semi_axes, 3); indexer = idxC)
@printf "[3] ∂C[1111]/∂a3_I   (sphere, 0)    = %.6e\n" ∂_a3

# ── 4) Multi-parameter gradient ─────────────────────────────────────────────
ps = [
    amount(:I),
    property(:I, :C, :bulk),
    property(:I, :C, :shear),
    property(:M, :C, :bulk),
    property(:M, :C, :shear),
]
∇ = gradient(rve, MoriTanaka(), ps; indexer = idxC)
println("\n[4] gradient(C[1111]) on 5 parameters [f_I, K_I, μ_I, K_M, μ_M] :")
@printf "    [%.4f, %.4f, %.4f, %.4f, %.4f]\n" ∇[1] ∇[2] ∇[3] ∇[4] ∇[5]

# ── 5) Full Jacobian of C_eff w.r.t. (f_I, K_I) ─────────────────────────────
J = jacobian(rve, MoriTanaka(), [amount(:I), property(:I, :C, :bulk)])
@printf "\n[5] jacobian shape = (%d × %d)   ;   J[1, :] = [%.4f, %.6f]\n" size(J, 1) size(J, 2) J[1, 1] J[1, 2]

# ── 6) Cross-check vs Christensen 1990 closed form ──────────────────────────
println("\n[6] Cross-check ∂k_MT/∂f vs Christensen 1990 closed form :")
println("    ─────────────────────────────────────────────────────────")
@printf "    %6s   %15s   %15s   %12s\n" "f" "AD" "Closed form" "rel. error"
@printf "    %s\n" ("─"^60)
k_m, μ_m = 10.0, 5.0
k_i, μ_i = 40.0, 20.0
ζm = k_m + 4 * μ_m / 3
Δk = k_i - k_m
bulk = C -> begin
    a = get_array(C)
    s = zero(eltype(a))
    for i in 1:3, j in 1:3
        s += a[i, i, j, j]
    end
    return s / 9
end
for f in (0.05, 0.1, 0.2, 0.3, 0.4)
    rve_f = RVE()
    add_phase!(rve_f, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k_m, 2μ_m)); fraction = :rest)
    add_phase!(rve_f, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k_i, 2μ_i)); fraction = f)
    ∂_ad = derivative(rve_f, MoriTanaka(), amount(:I); indexer = bulk)
    D = ζm + (1 - f) * Δk
    ∂_cf = Δk * ζm * (ζm + Δk) / D^2
    @printf "    %6.3f   %15.8f   %15.8f   %12.2e\n" f ∂_ad ∂_cf abs(∂_ad - ∂_cf) / ∂_cf
end

# ── 7) sensitivity() — closure fallback for cases the lenses cannot express ─
println("\n[7] sensitivity(closure) — generic fallback:")
f_eval = K_inc -> begin
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3k_m, 2μ_m)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_inc, 2μ_i)); fraction = 0.2)
    return bulk(homogenize(r, MoriTanaka()))
end
∂ = sensitivity(f_eval, 3k_i)
@printf "    ∂k_MT/∂(3K_I)|f=0.2  = %.6f\n" ∂

println("\nDone.")
