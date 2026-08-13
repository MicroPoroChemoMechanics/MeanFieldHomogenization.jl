# =============================================================================
#  test_laminate_symbolic.jl — the laminate kernel on symbolic element types.
#
#  Number types are part of the contract. The laminate kernel is written so
#  that it stays evaluable symbolically, which pins two implementation
#  choices that a purely numerical test would let rot:
#
#   * the pseudo-inverse goes through the cofactor `Core._inv3`, never through
#     `LinearAlgebra.pinv` (an SVD is not symbolically evaluable);
#   * every intermediate is an `SMatrix`, never an `MMatrix`
#     (`MMatrix{6,6,T}(undef)` is not constructible for a non-`isbits` `T`).
#
#  The pay-off is that the classical closed forms come out of the code itself:
#  for two isotropic layers the kernel must simplify EXACTLY to Backus (1962).
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using SymPy

const MFHC_S = MeanFieldHomogenization.Core

# Kelvin-Mandel matrix of an isotropic stiffness, built symbolically.
function _iso_km_sym(λ, μ)
    z = zero(λ)
    return SMatrix{6, 6}(
        [
            λ + 2μ λ λ z z z
            λ λ + 2μ λ z z z
            λ λ λ + 2μ z z z
            z z z 2μ z z
            z z z z 2μ z
            z z z z z 2μ
        ]
    )
end

@testset "Symbolic — bilayer reproduces Backus (1962) exactly" begin
    @syms λ₁::positive μ₁::positive λ₂::positive μ₂::positive f₁::positive
    f₂ = 1 - f₁

    C6s = (_iso_km_sym(λ₁, μ₁), _iso_km_sym(λ₂, μ₂))
    fs = (f₁, f₂)
    Z = SMatrix{6, 6}(zeros(Sym, 6, 6))

    Ch = MFHC_S.laminate_stiffness(C6s, fs, Z, Z)

    avg(g) = f₁ * g(λ₁, μ₁) + f₂ * g(λ₂, μ₂)
    r33 = 1 / avg((l, m) -> 1 / (l + 2m))
    rλ = avg((l, m) -> l / (l + 2m))

    # Every independent coefficient, simplified to zero difference.
    @test simplify(Ch[3, 3] - r33) == 0                                   # C₃₃₃₃
    @test simplify(Ch[4, 4] / 2 - 1 / avg((l, m) -> 1 / m)) == 0          # C₂₃₂₃
    @test simplify(Ch[6, 6] / 2 - avg((l, m) -> m)) == 0                  # C₁₂₁₂
    @test simplify(Ch[1, 3] - r33 * rλ) == 0                              # C₁₁₃₃
    @test simplify(Ch[1, 1] - (avg((l, m) -> 4m * (l + m) / (l + 2m)) + r33 * rλ^2)) == 0
    @test simplify(Ch[1, 2] - (avg((l, m) -> 2m * l / (l + 2m)) + r33 * rλ^2)) == 0

    # The out-of-plane law is a harmonic (Reuss) mean, in closed form.
    @test simplify(
        1 / Ch[3, 3] - (f₁ / (λ₁ + 2μ₁) + f₂ / (λ₂ + 2μ₂))
    ) == 0
    # ... and the in-plane shear an arithmetic (Voigt) one.
    @test simplify(Ch[6, 6] / 2 - (f₁ * μ₁ + f₂ * μ₂)) == 0

    # The structural zeros survive symbolically.
    for (i, j) in ((1, 4), (1, 5), (1, 6), (3, 4), (3, 6), (4, 5), (4, 6), (5, 6))
        @test simplify(Ch[i, j]) == 0
    end
end

@testset "Symbolic — a spring interface, in closed form" begin
    @syms λ₁::positive μ₁::positive λ₂::positive μ₂::positive
    @syms f₁::positive kn::positive L::positive
    f₂ = 1 - f₁

    C6s = (_iso_km_sym(λ₁, μ₁), _iso_km_sym(λ₂, μ₂))
    fs = (f₁, f₂)
    Z = SMatrix{6, 6}(zeros(Sym, 6, 6))
    z = zero(kn)
    𝕂 = SMatrix{3, 3}([z z z; z z z; z z kn])          # normal compliance only
    P_int = MFHC_S._op_embed(MFHC_S.compliance_op_block(𝕂)) / L

    Ch = MFHC_S.laminate_stiffness(C6s, fs, P_int, Z)

    # The exact out-of-plane series law, with the interface compliance added.
    @test simplify(
        1 / Ch[3, 3] - (f₁ / (λ₁ + 2μ₁) + f₂ / (λ₂ + 2μ₂) + kn / L)
    ) == 0
    # A normal spring leaves the in-plane shear untouched.
    @test simplify(Ch[6, 6] / 2 - (f₁ * μ₁ + f₂ * μ₂)) == 0
end

@testset "Symbolic — conduction bilayer" begin
    @syms k₁::positive k₂::positive f₁::positive ρ::positive L::positive
    f₂ = 1 - f₁

    K3s = (
        SMatrix{3, 3}(Sym[k₁ 0 0; 0 k₁ 0; 0 0 k₁]),
        SMatrix{3, 3}(Sym[k₂ 0 0; 0 k₂ 0; 0 0 k₂]),
    )
    fs = (f₁, f₂)
    Z = SMatrix{3, 3}(zeros(Sym, 3, 3))
    Pi = SMatrix{3, 3}(Sym[0 0 0; 0 0 0; 0 0 ρ]) / L

    Kh = MFHC_S.laminate_conductivity(K3s, fs, Z, Z)
    @test simplify(1 / Kh[3, 3] - (f₁ / k₁ + f₂ / k₂)) == 0     # series
    @test simplify(Kh[1, 1] - (f₁ * k₁ + f₂ * k₂)) == 0         # parallel

    KhR = MFHC_S.laminate_conductivity(K3s, fs, Pi, Z)
    @test simplify(1 / KhR[3, 3] - (f₁ / k₁ + f₂ / k₂ + ρ / L)) == 0
end

# `Symbolics` became a declared test dependency in v0.4.0 (see
# `Cracks/test_cod_symbolic.jl`), so this guard now always takes the `true`
# branch on CI; it is kept so the file still runs in a bare environment.
# `import`, not `using` — both `SymPy` and `Symbolics` export `@syms`, and this
# file uses SymPy's bare macro above.
const LAM_HAS_SYMBOLICS = try
    @eval import Symbolics
    true
catch
    false
end

if LAM_HAS_SYMBOLICS
    # `@eval` defers macro expansion of `Symbolics.@variables` until after the
    # conditional import above has actually run — without it the macro is
    # expanded while the file is being lowered, when `Symbolics` is not yet
    # bound in this module.
    @eval @testset "Symbolic — Symbolics.jl backend" begin
        Symbolics.@variables λ₁ μ₁ λ₂ μ₂ f₁
        f₂ = 1 - f₁
        C6s = (_iso_km_sym(λ₁, μ₁), _iso_km_sym(λ₂, μ₂))
        Z = SMatrix{6, 6}(zeros(Symbolics.Num, 6, 6))
        Ch = MFHC_S.laminate_stiffness(C6s, (f₁, f₂), Z, Z)

        # Same two closed forms as under SymPy — the kernel is backend-agnostic.
        d33 = Symbolics.simplify(
            1 / Ch[3, 3] - (f₁ / (λ₁ + 2μ₁) + f₂ / (λ₂ + 2μ₂)); expand = true
        )
        @test isequal(d33, 0)
        d66 = Symbolics.simplify(Ch[6, 6] / 2 - (f₁ * μ₁ + f₂ * μ₂); expand = true)
        @test isequal(d66, 0)
    end
else
    @info "Symbolics.jl not available — skipping the second symbolic backend"
end

@testset "Symbolic — through the Laminate cell itself" begin
    # Not just the kernel: the whole cell (property dicts, fractions, the
    # exact-TI return ladder) must carry symbolic entries.
    @syms κ₁::positive μ₁::positive κ₂::positive μ₂::positive f₁::positive

    lam = Laminate(; T = Sym)
    add_layer!(lam, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
    add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = 1 - f₁)

    Ch = homogenize(lam, Laminated(), :C)
    @test Ch isa TensND.TensTI{4}          # isotropic layers ⇒ exactly TI

    λ(κ, μ) = κ - 2μ / 3
    M = KM(Ch)
    @test simplify(
        1 / M[3, 3] - (f₁ / (λ(κ₁, μ₁) + 2μ₁) + (1 - f₁) / (λ(κ₂, μ₂) + 2μ₂))
    ) == 0
    @test simplify(M[6, 6] / 2 - (f₁ * μ₁ + (1 - f₁) * μ₂)) == 0

    # Voigt and Reuss stay symbolic too.
    @test simplify(KM(homogenize(lam, Voigt(), :C))[6, 6] / 2 - (f₁ * μ₁ + (1 - f₁) * μ₂)) == 0
end
