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
import LinearAlgebra

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

    # The CELL, not just the kernel, on `Symbolics.Num`. `Num <: Real` while
    # `SymPy.Sym` is not, so the `h isa Real && h < 0` guards in `add_layer!` /
    # `validate_laminate` / `validate_rve` used to reach `h < 0`, get a `Num`
    # back and throw `TypeError: non-boolean (Num) used in boolean context`.
    # The guards now go through `is_hard_numeric`.
    @eval @testset "Symbolic — Symbolics.jl fractions through the cell" begin
        Symbolics.@variables κ₁ μ₁ κ₂ μ₂ f₁
        lam = Laminate(; T = Symbolics.Num)
        add_layer!(lam, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
        add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = 1 - f₁)
        @test validate_laminate(lam) === lam
        @test isequal(Symbolics.simplify(laminate_period(lam) - 1), 0)

        rve = RVE(; T = Symbolics.Num)
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = f₁)
        @test MeanFieldHomogenization.Schemes.validate_rve(rve) === rve
    end
else
    @info "Symbolics.jl not available — skipping the second symbolic backend"
end

@testset "Symbolic — the frame must not contaminate the result" begin
    # THE regression test for the stray `1.0`.
    #
    # A `TensTI` converts its axis to the element type of its DATA, and the
    # components are rebuilt from the Walpole basis of that axis. So an axis read
    # off a `CanonicalBasis{3,Float64}` — `(0.0, 0.0, 1.0)` — became
    # `(Sym(0.0), Sym(0.0), Sym(1.0))` and put a `1.0` in front of every
    # coefficient of an otherwise exact answer:
    #
    #     C₂₃₂₃ = 1.0*mu_A*mu_B/(f*mu_B + mu_A*(1 - f))
    #
    # Note that `Sym(1.0) == Sym(1)` is `true`, so no `simplify(… - …) == 0`
    # test can see this: `1.0*x - x` IS zero. `is_Integer` is what distinguishes
    # the exact `1` from the float `1.0`, which is why the assertion is on the
    # axis rather than on the coefficients.
    @syms k_A::positive mu_A::positive k_B::positive mu_B::positive f::positive

    for lam in (
            Laminate(; normal = (0, 0, 1)),      # no `T = Sym` — the reported form
            Laminate(),
            Laminate(; T = Sym, normal = (0, 0, 1)),
            Laminate(; T = Sym),
        )
        add_layer!(lam, :A, Dict(:C => TensISO{3}(3k_A, 2mu_A)); fraction = f)
        add_layer!(lam, :B, Dict(:C => TensISO{3}(3k_B, 2mu_B)); fraction = 1 - f)
        C = homogenize(lam, Laminated(), :C)

        @test C isa TensND.TensTI{4}
        @test eltype(C) <: Sym
        # exact `0`, `0`, `1` — not `0.0`, `0.0`, `1.0`
        @test all(x -> x.is_Integer, TensND.axis(C))

        a = get_array(C)
        @test iszero(simplify(a[2, 3, 2, 3] - mu_A * mu_B / (f * mu_B + (1 - f) * mu_A)))
        # …and the user-visible symptom itself: no float in the printed form.
        @test !occursin("1.0", string(simplify(a[2, 3, 2, 3])))
        @test !occursin("1.0", string(simplify(a[3, 3, 3, 3])))
    end
end

@testset "Symbolic — a laminate with a symbolic normal" begin
    # The frame is completed by Gram-Schmidt against `e₁`, purely algebraically,
    # so no `atan2` ever appears and the answer stays as readable as the normal.
    @syms κ₁::positive μ₁::positive κ₂::positive μ₂::positive f₁::positive
    θ = symbols("theta", real = true)

    mklam(frame_kw) = begin
        lam = Laminate(; frame_kw...)
        add_layer!(lam, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
        add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = 1 - f₁)
        return lam
    end

    C0 = homogenize(mklam((T = Sym,)), Laminated(), :C)
    for kw in (
            (normal = (0, sin(θ), cos(θ)),),
            (normal = (cos(θ), 0, sin(θ)), in_plane = (0, 1, 0)),
            (euler_angles = (θ, 0, 0),),
        )
        lam = mklam(kw)
        C = homogenize(lam, Laminated(), :C)
        @test C isa TensND.TensTI{4}
        @test eltype(C) <: Sym
        # The axis moved; the Walpole coefficients did NOT. That is the whole
        # content of frame covariance, and it holds symbolically.
        @test all(
            iszero,
            simplify.(collect(TensND.get_data(C)) .- collect(TensND.get_data(C0)))
        )
        # …and the frame really is an orthonormal, right-handed one about `n`.
        R = MeanFieldHomogenization.Core._frame_matrix(laminate_basis(lam))
        @test all(iszero, simplify.(R' * R - LinearAlgebra.I))
        @test iszero(simplify(LinearAlgebra.det(R) - 1))
        @test all(iszero, simplify.(collect(TensND.axis(C)) .- R[:, 3]))
    end
end

@testset "Symbolic — a TI layer about a symbolic axis is recognized" begin
    # `_parallel` used to answer `false` for every non-`Real` element type, so a
    # symbolic TI layer was never seen as coaxial and the exact `TensTI` return
    # degraded to a generic `Tens`. The symbolic branch decides collinearity
    # structurally — conservative, never a false positive.
    @syms a₁::positive a₂::positive a₃::positive a₅::positive a₆::positive
    @syms μ::positive κ::positive f₁::positive
    θ = symbols("theta", real = true)
    n = (Sym(0), sin(θ), cos(θ))

    lam = Laminate(; normal = n)
    add_layer!(lam, :A, Dict(:C => TensTI{4}(a₁, a₂, a₃, a₅, a₆, n)); fraction = f₁)
    add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ, 2μ)); fraction = 1 - f₁)
    @test homogenize(lam, Laminated(), :C) isa TensND.TensTI{4}

    # A TI layer about a DIFFERENT symbolic axis must still fall through.
    lam2 = Laminate(; normal = n)
    add_layer!(
        lam2, :A, Dict(:C => TensTI{4}(a₁, a₂, a₃, a₅, a₆, (sin(θ), Sym(0), cos(θ))));
        fraction = f₁
    )
    add_layer!(lam2, :B, Dict(:C => TensISO{3}(3κ, 2μ)); fraction = 1 - f₁)
    @test !(homogenize(lam2, Laminated(), :C) isa TensND.TensTI)
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


@testset "Symbolic — the four localization generics on a laminate" begin
    # The two mixed tensors are new; the identities that define them must hold
    # as exact symbolic zeros, not as tolerances.
    #
    # Two choices keep this cheap enough to belong in a test suite. The
    # comparison is made on the Walpole coefficients rather than on the 6×6
    # Kelvin-Mandel matrices — a laminate of isotropic layers returns
    # localization tensors that are exactly `TensTI{4,T,6}`, so six scalars
    # describe each of them. And only ONE layer is symbolic: the compliance
    # `𝕊ʰᵒᵐ`, which the two stress-side tensors carry, is a rational function of
    # every modulus, and asking SymPy to simplify it in four free moduli at once
    # costs minutes for nothing the extra freedom would establish.
    @syms κ₁::positive μ₁::positive f₁::positive
    κ₂ = Sym(1)
    μ₂ = Sym(1) / 2

    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => TensISO{3}(3κ₁, 2μ₁)); fraction = f₁)
    add_layer!(lam, :B, Dict(:C => TensISO{3}(3κ₂, 2μ₂)); fraction = 1 - f₁)

    Ch = homogenize(lam, Laminated(), :C)
    Sh = inv(Ch)
    @test Ch isa TensND.TensTI{4, <:Sym, 5}        # major-symmetric

    # `tsimplify` on the difference of two tensors: exact zero on every
    # canonical coefficient. (It used to be a silent no-op on structured
    # tensors — see the TensND v0.4.0 changelog.)
    zero_tens(X) = all(iszero, TensND.get_data(tsimplify(X)))

    A = strain_strain_loc(lam, :A)
    C_A = TensISO{3}(3κ₁, 2μ₁)
    @test A isa TensND.TensTI{4, <:Sym, 6}         # NOT major-symmetric
    @test zero_tens(A - layer_strain_localization(lam, :A))
    @test zero_tens(stress_stress_loc(lam, :A) - layer_stress_localization(lam, :A))
    @test zero_tens(stress_strain_loc(lam, :A) - C_A ⊡ A)
    @test zero_tens(strain_stress_loc(lam, :A) - A ⊡ Sh)
    @test zero_tens(stress_stress_loc(lam, :A) - stress_strain_loc(lam, :A) ⊡ Sh)

    # The two strain-side sum rules. The stress-side ones follow from them by
    # right-multiplication with `𝕊ʰᵒᵐ` and are covered numerically.
    Id4 = TensND.tens_Id4(Val(3), Val(Sym))
    wsum(g) = f₁ * g(:A) + (1 - f₁) * g(:B)
    @test zero_tens(wsum(nm -> strain_strain_loc(lam, nm)) - Id4)
    @test zero_tens(wsum(nm -> stress_strain_loc(lam, nm)) - Ch)

    # The one closed form worth reading off: the macroscopic in-plane strain
    # reaches every layer unchanged, so ℓ₅ of 𝔸ᵢ — the in-plane deviatoric
    # coefficient — is exactly 1.
    @test iszero(tsimplify(TensND.get_ℓ(A)[5] - 1))
end
