# =============================================================================
#  58_alv_kernel_types.jl
#
#  Demonstration of the structured ALV kernel types
#  `ALVKernelISO / ALVKernelTI / ALVKernelOrtho` (`<: AbstractMatrix{T}`)
#  introduced in v0.7.0 of `MeanFieldHomogenization.jl`.
#
#  These types parallel `TensND.TensISO / TensTI / TensOrtho` for the
#  *time-discretized Volterra* algebra.  Each one stores the symmetry
#  class compactly (2 / 6 / 12 `n × n` matrices respectively, instead
#  of one `(6n × 6n)`) and the algebra closures `+, *, volterra_inverse`
#  remain in the structured class without ever materializing the full
#  block matrix.
#
#  Usage  : julia --project scripts/58_alv_kernel_types.jl
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

# ─── 1) ALVKernelISO from a Maxwell iso law ────────────────────────────────

println("="^78)
println(" 1) ALVKernelISO — iso ALV kernel from a Maxwell relaxation law")
println("="^78)

# Maxwell iso : R(t,t') = (3K∞ + (3K₀ - 3K∞) e^{-(t-t')/τ_K}) 𝕁
#             + (2μ∞ + (2μ₀ - 2μ∞) e^{-(t-t')/τ_μ}) 𝕂
const k₀ = 5.0;  const μ₀ = 2.0
const k∞ = 3.0;  const μ∞ = 1.0
const τK = 1.0;  const τμ = 0.5

function R_iso(t, tp)
    α = 3 * (k∞ + (k₀ - k∞) * exp(-(t - tp) / τK))
    β = 2 * (μ∞ + (μ₀ - μ∞) * exp(-(t - tp) / τμ))
    return TensISO{3}(α, β)
end
const law_M = ViscoLaw(R_iso, :relaxation)

times = collect(range(0.0, 2.0; length = 8))
n = length(times)

# `trapezoidal_matrix` gives the (6n × 6n) iso block matrix.
M_full = trapezoidal_matrix(law_M, times)
@printf "Full block matrix: %d × %d entries  (%d Float64)\n" size(M_full)... length(M_full)

# Wrap it as a structured `ALVKernelISO`.  This *extracts* the (α, β)
# parameter matrices via `iso_params_from_blocks`.
K_iso = ALVKernelISO(M_full)
println(
    "\nALVKernelISO storage : ",
    2 * length(K_iso.α), " Float64 entries  → ",
    round(length(M_full) / (2 * length(K_iso.α)); digits = 1), "× cheaper"
)

@assert isapprox(Matrix(K_iso), M_full; atol = 1.0e-12)  # round-trip check

# ─── 2) Algebra closure — multiplication stays iso, no materialization ─────

println("\n" * "="^78)
println(" 2) Algebra closure : iso × iso → iso  (no (6n × 6n) materialisation)")
println("="^78)

K1 = ALVKernelISO(trapezoidal_matrix(law_M, times))
K2 = ALVKernelISO(
    trapezoidal_matrix(
        maxwell_iso(
            2 * k₀, 2 * μ₀, 0.5 * τK,
            0.5 * τμ
        ),
        times
    )
)

# `*` invokes the iso fast path internally.
K_prod = K1 * K2
println("type(K1 * K2) = ", typeof(K_prod))           # ALVKernelISO
println("K_prod is type-stable iso : ", K_prod isa ALVKernelISO)

# Sanity check vs the materialized generic GEMM.
M_prod_full = Matrix(K1) * Matrix(K2)
@assert isapprox(Matrix(K_prod), M_prod_full; atol = 1.0e-12)

# ─── 3) Volterra inverse stays iso ─────────────────────────────────────────

println("\n" * "="^78)
println(" 3) volterra_inverse — iso closure, no block-LU on (6n × 6n)")
println("="^78)

K_inv = volterra_inverse(K1)
println("type(volterra_inverse(K1)) = ", typeof(K_inv))
@assert K_inv isa ALVKernelISO

# Check K1 ∘ K1^{-vol} ≈ block identity.
M_id = Matrix(K1) * Matrix(K_inv)
H_id = zeros(6n, 6n)
for i in 1:n
    rows = (6 * (i - 1) + 1):(6 * i)
    H_id[rows, rows] = Matrix{Float64}(I, 6, 6)
end
err_id = maximum(abs, M_id .- H_id)
@printf "max|K1 ∘ K1^{-vol} - I| = %.2e\n" err_id
@assert err_id < 1.0e-10

# ─── 4) ALVKernelTI from an iso law (axis = e_3) ───────────────────────────

println("\n" * "="^78)
println(" 4) ALVKernelTI — iso ⊂ TI : promotion auto via constructor")
println("="^78)

K_TI = ALVKernelTI(K_iso)         # iso → TI promotion
println("type(K_TI)   = ", typeof(K_TI))
println("K_TI.axis    = ", K_TI.axis)
println(
    "storage      : 6 × n² = ", 6 * n^2, " entries  (vs 36n² = ",
    36 * n^2, " for full)"
)

# Algebra : iso × TI → TI (auto-promote)
K_mixed = K_iso * K_TI
println("type(K_iso * K_TI) = ", typeof(K_mixed))    # ALVKernelTI
@assert K_mixed isa ALVKernelTI

# ─── 5) ALVKernelOrtho — full ladder iso ⊂ TI ⊂ ortho ──────────────────────

println("\n" * "="^78)
println(" 5) ALVKernelOrtho — symmetry ladder iso ⊂ TI ⊂ ortho")
println("="^78)

K_O = ALVKernelOrtho(K_TI)        # TI → ortho promotion
println("type(K_O)    = ", typeof(K_O))
println("K_O.axes     = canonical (e₁, e₂, e₃)")
println("storage      : 12 × n² = ", 12 * n^2, " entries")

# Mixed arithmetic : iso × ortho → ortho
K_io_o = K_iso + K_O
println("type(K_iso + K_O) = ", typeof(K_io_o))
@assert K_io_o isa ALVKernelOrtho

# Check that round-trip iso → TI → ortho → Matrix matches the generic path.
M_iso = Matrix(K_iso)
M_via_TI = Matrix(ALVKernelTI(K_iso))
M_via_O = Matrix(ALVKernelOrtho(K_iso))
@assert isapprox(M_iso, M_via_TI; atol = 1.0e-12)
@assert isapprox(M_iso, M_via_O; atol = 1.0e-12)
println("round-trip iso → TI → ortho → (6n×6n) preserved (≤ 1e-12)")

# ─── 6) AbstractMatrix interface — works in generic Julia code ────────────

println("\n" * "="^78)
println(" 6) AbstractMatrix interface — istril, getindex, multiplications…")
println("="^78)

println(
    "istril(K_iso) = ", istril(K_iso),
    " (Volterra causality is built-in)"
)
println("size(K_iso, 1) = ", size(K_iso, 1))
println("K_iso[1, 1] = ", K_iso[1, 1])

# Mixed `K * Matrix` works via the AbstractMatrix interface (falls back
# to `Matrix(K) * other`).  Use this to plug structured kernels into
# generic Julia matrix code without reimplementing every operator.
v = randn(6n)
println("K_iso * v size = ", size(Matrix(K_iso) * v))   # 6n vector

println("\nSummary :")
@printf "  ALVKernelISO   :  2·n² = %d entries (saves %.0f%% vs %d for (6n×6n))\n" 2 * n^2 (1 - 2 / (36)) * 100 36 * n^2
@printf "  ALVKernelTI    :  6·n² = %d entries (saves %.0f%%)\n"                          6 * n^2 (1 - 6 / 36) * 100
@printf "  ALVKernelOrtho : 12·n² = %d entries (saves %.0f%%)\n"                         12 * n^2 (1 - 12 / 36) * 100

println("\n--- end of demo ---")
