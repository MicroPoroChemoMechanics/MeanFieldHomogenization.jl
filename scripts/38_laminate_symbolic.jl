# =============================================================================
#  38_laminate_symbolic.jl
#
#  The laminate kernel run on SYMBOLIC moduli, so that the classical closed
#  forms come out of the code itself rather than being checked against it.
#
#  For two isotropic layers the general formula
#
#      ℂ_hom = ⟨ℚ⟩ + ⟨ℂ:ℙ⟩ : ⟨ℙ⟩† : ⟨ℙ:ℂ⟩
#
#  must simplify EXACTLY to Backus (1962). Getting there requires two
#  implementation choices that a purely numerical script would never expose:
#
#    * the pseudo-inverse is a cofactor inverse of the 3×3 out-of-plane block,
#      never `LinearAlgebra.pinv` — an SVD is not symbolically evaluable;
#    * every intermediate is an `SMatrix`, never an `MMatrix`, which cannot
#      even be CONSTRUCTED for a non-isbits element type such as `SymPy.Sym`.
#
#  NOT published to the documentation gallery: SymPy-heavy scripts are kept
#  out of the Literate build (repo policy, see scripts/README.md). Its content
#  is covered by the hand-written tutorial
#  `docs/src/tutorials/symbolic_laminate.md`, by `docs/src/theory/laminate.md`
#  and by `test/Laminates/test_laminate_symbolic.jl`.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using StaticArrays
using SymPy
using Printf

const MFHC = MeanFieldHomogenization.Core

println("="^78)
println("Periodic multilayer — symbolic derivation of the closed forms")
println("="^78)

# ── §1  Two isotropic layers, general fractions ──────────────────────────────

@syms λ₁::positive μ₁::positive λ₂::positive μ₂::positive f₁::positive
f₂ = 1 - f₁

function iso_km(λ, μ)
    z = zero(λ)
    return SMatrix{6, 6}(
        [
            λ+2μ λ λ z z z
            λ λ+2μ λ z z z
            λ λ λ+2μ z z z
            z z z 2μ z z
            z z z z 2μ z
            z z z z z 2μ
        ]
    )
end

Z = SMatrix{6, 6}(zeros(Sym, 6, 6))
Ch = MFHC.laminate_stiffness((iso_km(λ₁, μ₁), iso_km(λ₂, μ₂)), (f₁, f₂), Z, Z)

println("\n§1  Effective coefficients, simplified by SymPy")
println("─"^78)
@printf "  C₃₃₃₃ = %s\n" string(simplify(Ch[3, 3]))
@printf "  C₂₃₂₃ = %s\n" string(simplify(Ch[4, 4] / 2))
@printf "  C₁₂₁₂ = %s\n" string(simplify(Ch[6, 6] / 2))
@printf "  C₁₁₃₃ = %s\n" string(simplify(Ch[1, 3]))

# The two structural results, read directly off the symbolic output:
#   * out of plane, a HARMONIC (Reuss) mean — the laminate saturates Reuss;
#   * in plane, an ARITHMETIC (Voigt) mean  — and saturates Voigt.
# The whole effective matrix, in the five averages the answer is built from.
# `A` and `G` enter INVERTED — the harmonic (series, out-of-plane) part; `E`
# and `F` enter directly — the arithmetic (parallel, in-plane) part; `B`
# couples them.
avg0(g) = f₁ * g(λ₁, μ₁) + f₂ * g(λ₂, μ₂)
A = avg0((l, m) -> 1 / (l + 2m))          # harmonic building block
B = avg0((l, m) -> l / (l + 2m))          # the coupling weight
E = avg0((l, m) -> 2m * l / (l + 2m))
F = avg0((l, m) -> m)                     # arithmetic mean of the shear moduli
G = avg0((l, m) -> 1 / m)                 # harmonic mean of the shear moduli

M_claim = [
    E+B^2/A+2F E+B^2/A B/A 0 0 0
    E+B^2/A E+B^2/A+2F B/A 0 0 0
    B/A B/A 1/A 0 0 0
    0 0 0 2/G 0 0
    0 0 0 0 2/G 0
    0 0 0 0 0 2F
]

println("\n§1b  The effective Kelvin-Mandel matrix, in five averages")
println("─"^78)
println("     A = ⟨1/(λ+2μ)⟩   B = ⟨λ/(λ+2μ)⟩   E = ⟨2μλ/(λ+2μ)⟩   F = ⟨μ⟩   G = ⟨1/μ⟩")
println()
println("     ⎡ E+B²/A+2F   E+B²/A     B/A     .      .      .  ⎤")
println("     ⎢ E+B²/A      E+B²/A+2F  B/A     .      .      .  ⎥")
println("     ⎢ B/A         B/A        1/A     .      .      .  ⎥")
println("     ⎢ .           .          .      2/G     .      .  ⎥")
println("     ⎢ .           .          .       .     2/G     .  ⎥")
println("     ⎣ .           .          .       .      .     2F  ⎦")
println()
@printf "  residual against the code : %s\n" string(maximum(abs, simplify.(Matrix(Ch) - M_claim)))
println("  A and G appear INVERTED (harmonic, series, out of plane);")
println("  E and F appear directly (arithmetic, parallel, in plane).")

println("\n§2  The two exact bound saturations")
println("─"^78)
d_reuss = simplify(1 / Ch[3, 3] - (f₁ / (λ₁ + 2μ₁) + f₂ / (λ₂ + 2μ₂)))
d_voigt = simplify(Ch[6, 6] / 2 - (f₁ * μ₁ + f₂ * μ₂))
@printf "  1/C₃₃₃₃ − ⟨1/(λ+2μ)⟩ = %s      (Reuss, out of plane)\n" string(d_reuss)
@printf "  C₁₂₁₂   − ⟨μ⟩         = %s      (Voigt, in plane)\n" string(d_voigt)

# ── §3  The full Backus (1962) set ───────────────────────────────────────────

avg(g) = f₁ * g(λ₁, μ₁) + f₂ * g(λ₂, μ₂)
r₃₃ = 1 / avg((l, m) -> 1 / (l + 2m))
rλ = avg((l, m) -> l / (l + 2m))

backus = (
    ("C₁₁₁₁", Ch[1, 1], avg((l, m) -> 4m * (l + m) / (l + 2m)) + r₃₃ * rλ^2),
    ("C₁₁₂₂", Ch[1, 2], avg((l, m) -> 2m * l / (l + 2m)) + r₃₃ * rλ^2),
    ("C₁₁₃₃", Ch[1, 3], r₃₃ * rλ),
    ("C₃₃₃₃", Ch[3, 3], r₃₃),
    ("C₂₃₂₃", Ch[4, 4] / 2, 1 / avg((l, m) -> 1 / m)),
    ("C₁₂₁₂", Ch[6, 6] / 2, avg((l, m) -> m)),
)

println("\n§3  Against Backus (1962), component by component")
println("─"^78)
for (name, got, want) in backus
    @printf "  simplify(%s − Backus) = %s\n" name string(simplify(got - want))
end

# ── §4  A spring interface, still in closed form ─────────────────────────────

@syms kn::positive L::positive
z = zero(kn)
𝕂 = SMatrix{3, 3}([z z z; z z z; z z kn])          # normal compliance only
P_int = MFHC._op_embed(MFHC.compliance_op_block(𝕂)) / L
Chi = MFHC.laminate_stiffness((iso_km(λ₁, μ₁), iso_km(λ₂, μ₂)), (f₁, f₂), P_int, Z)

println("\n§4  With a normal spring interface of compliance kn over a period L")
println("─"^78)
@printf "  1/C₃₃₃₃ = %s\n" string(simplify(1 / Chi[3, 3]))
d_itf = simplify(1 / Chi[3, 3] - (f₁ / (λ₁ + 2μ₁) + f₂ / (λ₂ + 2μ₂) + kn / L))
@printf "  minus ⟨1/(λ+2μ)⟩ + kn/L = %s   (the compliance simply adds)\n" string(d_itf)
@printf "  C₁₂₁₂ unchanged : %s\n" string(simplify(Chi[6, 6] / 2 - (f₁ * μ₁ + f₂ * μ₂)))

# ── §5  Through the Laminate cell itself ─────────────────────────────────────
#
# Not just the bare kernel: the cell (property dicts, derived fractions, the
# frame, the exact-TI return ladder) carries symbolic entries end to end. Note
# that NOTHING is declared — no `T = Sym`: the moduli, the fractions and the
# frame each carry their own element type, and the canonical frame is read for
# its axis exactly, as `(0, 0, 1)` and not `(0.0, 0.0, 1.0)`. Were it not, that
# float would reappear as a symbolic `1.0` in front of every coefficient below,
# because a `TensTI` rebuilds its components from the Walpole basis of its axis.

@syms κ₁::positive κ₂::positive
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = f₂)
Csym = homogenize(lam, Laminated(), :C)

println("\n§5  `homogenize` on a symbolic Laminate")
println("─"^78)
println("  returned type : ", typeof(Csym))
println("  (isotropic layers ⇒ EXACTLY transversely isotropic about n)")
@printf "  axis          : %s   (exact integers, not 0.0 / 1.0)\n" string(TensND.axis(Csym))
Msym = KM(Csym)
lame(κ, μ) = κ - 2μ / 3
@printf "  1/C₃₃₃₃ − ⟨1/(λ+2μ)⟩ = %s\n" string(
    simplify(1 / Msym[3, 3] - (f₁ / (lame(κ₁, μ₁) + 2μ₁) + f₂ / (lame(κ₂, μ₂) + 2μ₂)))
)
@printf "  C₂₃₂₃                 = %s\n" string(simplify(Msym[4, 4] / 2))

# ── §5b  A symbolic FRAME, not just symbolic moduli ──────────────────────────
#
# The normal itself may be symbolic. It is completed into an orthonormal
# (ℓ, m, n̂) by plain Gram-Schmidt against a reference axis — no trigonometry and
# no atan2, so the frame stays as readable as the normal it came from. The
# in-plane reference defaults to e₁ and is overridable with `in_plane = …`; the
# choice is physically immaterial, the answer being invariant under rotation
# about n.

θ = symbols("theta", real = true)
lam_tilt = Laminate(; normal = (0, sin(θ), cos(θ)))
add_layer!(lam_tilt, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
add_layer!(lam_tilt, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = f₂)
Ctilt = homogenize(lam_tilt, Laminated(), :C)

println("\n§5b  A symbolic normal")
println("─"^78)
println("  returned type : ", typeof(Ctilt))
@printf "  axis          : %s\n" string(simplify.(TensND.axis(Ctilt)))
same = all(
    iszero,
    simplify.(collect(TensND.get_data(Ctilt)) .- collect(TensND.get_data(Csym)))
)
@printf "  same Walpole coefficients as the canonical stack : %s\n" string(same)
println("  (frame covariance, as an identity rather than a tolerance)")

# ── §6  Conduction ───────────────────────────────────────────────────────────

@syms k₁::positive k₂::positive ρ::positive
K3s = (
    SMatrix{3, 3}(Sym[k₁ 0 0; 0 k₁ 0; 0 0 k₁]),
    SMatrix{3, 3}(Sym[k₂ 0 0; 0 k₂ 0; 0 0 k₂]),
)
Z3 = SMatrix{3, 3}(zeros(Sym, 3, 3))
Pρ = SMatrix{3, 3}(Sym[0 0 0; 0 0 0; 0 0 ρ]) / L

Kh = MFHC.laminate_conductivity(K3s, (f₁, f₂), Z3, Z3)
Khρ = MFHC.laminate_conductivity(K3s, (f₁, f₂), Pρ, Z3)

println("\n§6  Conduction")
println("─"^78)
@printf "  k_⊥      = %s        (series)\n" string(simplify(Kh[3, 3]))
@printf "  k_∥      = %s        (parallel)\n" string(simplify(Kh[1, 1]))
@printf "  1/k_⊥ with Kapitza ρ, minus ⟨1/k⟩ + ρ/L = %s\n" string(
    simplify(1 / Khρ[3, 3] - (f₁ / k₁ + f₂ / k₂ + ρ / L))
)

println("\n" * "="^78)
println("Every difference above is 0: the code reproduces the closed forms exactly.")
println("="^78)
