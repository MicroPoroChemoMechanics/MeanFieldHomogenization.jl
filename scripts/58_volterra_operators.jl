# =============================================================================
#  scripts/58_volterra_operators.jl
#
#  Aging viscoelastic operators and histories (`VolterraOperator`,
#  `TimeHistory`) on one shared time grid (`TimeGrid`), step by step:
#
#    1. an operator from a law, from a constant tensor, from a matrix;
#    2. computing as with elastic tensors: creep = inverse, response to a history;
#    3. what is stored: the blocks of the symmetry class, and what it saves;
#    4. homogenization with operators as phase properties, over two scales.
#
#  Run from the package root:
#    julia --project=. scripts/58_volterra_operators.jl
# =============================================================================

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

# ── 1. Operators ────────────────────────────────────────────────────────────

# The instants of the whole computation, built once.
grid = TimeGrid(collect(range(0.0, 30.0; length = 40)))

# An aging Maxwell paste: its moduli relax, and it stiffens with its age at
# loading t′. The stiffness of an isotropic phase is 3k 𝕁 + 2μ 𝕂.
age(t′) = 1 - 0.6 * exp(-t′ / 3)
paste = VolterraOperator(grid) do t, t′
    age(t′) * TensISO{3}(3 * (2.0 + 10.0 * exp(-(t - t′) / 5)), 2 * (1.0 + 6.0 * exp(-(t - t′) / 2)))
end
println(paste)

# An elastic aggregate: a constant tensor is an operator too.
sand = VolterraOperator(TensISO{3}(3 * 40.0, 2 * 30.0), grid)

# ── 2. Computing as with elastic tensors ───────────────────────────────────

# The creep operator is the Volterra inverse of the relaxation one.
creep = inv(paste)

# A uniaxial stress of 1 held from t = 0: its strain is the creep operator applied
# to it; the relaxation operator gives the stress back.
σ = TimeHistory(t -> TensND.Tens(Symmetric([1.0 0 0; 0 0 0; 0 0 0])), grid)
ε = creep ⊡ σ
@printf "axial strain at t = %4.1f: %.4f, at t = %4.1f: %.4f\n" grid[2] ε[2][1] grid[end] ε[end][1]
@printf "stress given back by the relaxation operator, to %.1e\n" maximum(abs, Matrix(paste ⊡ ε) - Matrix(σ))

# ── 3. What is stored ───────────────────────────────────────────────────────

# Isotropic values: two n×n matrices, where the full matrix is 6n×6n.
n = length(grid)
stored = sum(length, paste.blocks)
@printf "stored entries: %d, full matrix: %d (%.0f times fewer)\n" stored 36n^2 36n^2 / stored

# The same algebra on the full matrices, for comparison.
D = Matrix(paste)
paste * paste; D * D; inv(paste); volterra_inverse(D)
t_blocks = minimum(@elapsed(inv(paste * paste)) for _ in 1:5)
t_dense = minimum(@elapsed(volterra_inverse(D * D)) for _ in 1:5)
@printf "inverse of a composition: %.2e s by blocks, %.2e s dense (%.0f×)\n" t_blocks t_dense t_dense / t_blocks
@printf "same result: %.1e\n" norm(Matrix(inv(paste * paste)) - volterra_inverse(D * D)) / norm(volterra_inverse(D * D))

# ── 4. Homogenization over two scales ──────────────────────────────────────

# A mortar: the paste with 30 % of stiff spheres. Its phases are operators, so
# the grid is read on them.
mortar = RVE()
add_phase!(mortar, :paste, Ellipsoid(1.0), Dict(:C => paste); fraction = :rest)
add_phase!(mortar, :sand, Ellipsoid(1.0), Dict(:C => sand); fraction = 0.3)
R_mortar = homogenize_alv(mortar, MoriTanaka(:paste), :C)
println("mortar: ", R_mortar)

# A concrete: the mortar, not computed beforehand, with 40 % of prolate
# aggregates along e₃. The result is transversely isotropic about e₃.
concrete = RVE()
add_phase!(concrete, :mortar, Ellipsoid(1.0), Dict(:C => Homogenized(mortar, MoriTanaka(:paste))); fraction = :rest)
add_phase!(concrete, :gravel, Ellipsoid(1.0, 1.0, 3.0), Dict(:C => sand); fraction = 0.4)
R_concrete = homogenize_alv(concrete, MoriTanaka(:mortar), :C)
println("concrete: ", R_concrete)

# Its axial creep under the same held stress, against the paste's.
ε_c = inv(R_concrete) ⊡ σ
@printf "axial strain at t = %4.1f: paste %.4f, concrete %.4f\n" grid[end] ε[end][1] ε_c[end][1]
