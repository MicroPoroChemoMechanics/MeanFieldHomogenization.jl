# =============================================================================
#  bench_layered_alv.jl — Julia counterpart of `bench_layered_alv.py`.
#
#  Loads the per-layer ALV concentration tensors computed by ECHOES Python
#  (the trusted reference) and compares them to those produced by
#  `MeanFieldHomogenization.jl`'s `bulk_concentration_alv` / `shear_concentration_alv`.
#
#  Run order :
#    1. python scripts/bench_echoes/bench_layered_alv.py
#    2. julia --project scripts/bench_echoes/bench_layered_alv.jl
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Printf

const HERE = @__DIR__
const json_path = joinpath(HERE, "bench_layered_alv_python.json")
isfile(json_path) || error("Run `bench_layered_alv.py` first; missing: $json_path")
ref = JSON.parsefile(json_path)

# ── Reproduce the Python setup in Julia ─────────────────────────────────────

T_grid = collect(Float64.(ref.T))
n = length(T_grid)

# Coerce the JSON-loaded scalars to Float64 (JSON keeps integers as Int).
const k0 = Float64(ref.k0)
const mu0 = Float64(ref.mu0)
const eta_k0 = Float64(ref.eta_k0)
const eta_mu0 = Float64(ref.eta_mu0)
const k1 = Float64(ref.k1)
const mu1 = Float64(ref.mu1)
const k2 = Float64(ref.k2)
const mu2 = Float64(ref.mu2)

# Matrix Maxwell relaxation kernel.
make_R0() = maxwell_iso(k0, mu0, eta_k0, eta_mu0)

# Two elastic Heaviside-relaxation layers.
C1 = TensISO{3}(3.0 * k1, 2.0 * mu1)
C2 = TensISO{3}(3.0 * k2, 2.0 * mu2)

# Layer fractions [0.5, 0.5] (within the sphere) → cumulative radii.
layer_fracs = collect(Float64.(ref.layer_fracs))
@assert length(layer_fracs) == 2
cumulative = cumsum(layer_fracs)
radii = ntuple(k -> cumulative[k]^(1 / 3), length(layer_fracs))

# Build the LayeredSphere.
moduli = (heaviside_law(C1), heaviside_law(C2))
sphere = LayeredSphere(radii, moduli)
println("LayeredSphere : ", layer_count(sphere), " layers, radii = ", sphere.radii)

# ── Compute Julia per-layer alpha, beta Volterra blocks ─────────────────────

C0_law = make_R0()
α_jl = bulk_concentration_alv(sphere, C0_law, T_grid)
β_jl = shear_concentration_alv(sphere, C0_law, T_grid)

println("Julia α[1] (n×n) :")
display(α_jl[1])
println("\nJulia β[1] (n×n) :")
display(β_jl[1])
println("\nJulia α[2] (n×n) :")
display(α_jl[2])
println("\nJulia β[2] (n×n) :")
display(β_jl[2])

# ── Compare against the ECHOES reference ─────────────────────────────────────

# JSON nested arrays are loaded as vectors of vectors; convert to matrices.
to_matrix(v) = reduce(hcat, [collect(Float64.(row)) for row in v])'

α_py_0 = Matrix(to_matrix(ref.alpha0))
β_py_0 = Matrix(to_matrix(ref.beta0))
α_py_1 = Matrix(to_matrix(ref.alpha1))
β_py_1 = Matrix(to_matrix(ref.beta1))

println("\nECHOES Python α[layer=0] (n×n) :")
display(α_py_0)
println("\nECHOES Python β[layer=0] (n×n) :")
display(β_py_0)

# ── Print element-by-element diff ───────────────────────────────────────────

function compare(label, A_jl, A_py)
    @printf "\n── %s ──\n" label
    @printf "  size jl = %s ; size py = %s\n" string(size(A_jl)) string(size(A_py))
    if size(A_jl) != size(A_py)
        println("  SIZE MISMATCH — cannot diff.")
        return
    end
    diff = A_jl .- A_py
    rel = maximum(abs, diff) / max(maximum(abs, A_py), 1.0e-300)
    @printf "  max |Δ|       = %.3e\n" maximum(abs, diff)
    @printf "  relative      = %.3e\n" rel
    println("  diagonal (jl − py) :")
    for i in 1:size(diff, 1)
        @printf "    [%d, %d] : Δ = %+.3e   (jl = %+.6e ; py = %+.6e)\n" i i diff[i, i] A_jl[i, i] A_py[i, i]
    end
    return
end

compare("layer 1 (core)  α", α_jl[1], α_py_0)
compare("layer 1 (core)  β", β_jl[1], β_py_0)
compare("layer 2 (shell) α", α_jl[2], α_py_1)
compare("layer 2 (shell) β", β_jl[2], β_py_1)
