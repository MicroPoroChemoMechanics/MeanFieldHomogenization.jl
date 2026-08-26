# =============================================================================
#  test_laminate_oracles.jl — the physics of the periodic multilayer.
#
#  Two EXACT closed-form oracles, valid for anisotropic layers, pin the whole
#  kernel between them — including the `1/L` interface weight:
#
#   O1 (out-of-plane, sensitive to the PRIMAL interfaces only)
#       (n·ℂ_hom·n)⁻¹ = Σ_i f_i (n·ℂ_i·n)⁻¹ + Σ_j 𝕂_j / L
#     because ℚ_i has an identically zero out-of-plane row and column and
#     (ℂ_iℙ_i) restricted to the out-of-plane block is the identity.
#
#   O2 (in-plane, sensitive to the DUAL interfaces only)
#       Schur_IP(ℂ_hom) = Σ_i f_i Schur_IP(ℂ_i) + Σ_j ℂˢ_j / L
#     because the ⟨ℙ⟩ factors cancel identically.
#
#  Being complementary, the pair separates a spring bug from a membrane bug.
#
#  Further coverage: Backus (1962) bilayer closed form, N = 1 degeneracy,
#  identical layers, layer-order invariance, exact `TensTI` output, bounds
#  bracketing with the in-plane Voigt / out-of-plane Reuss exactness, the
#  localization sum rules, and a deliberately asymmetric anisotropic case.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using Random

const MFHC_O = MeanFieldHomogenization.Core
const ATOL_LAM = 1.0e-11

_iso(k, μ) = TensISO{3}(3k, 2μ)
_lame(k, μ) = (k - 2μ / 3, μ)

# Out-of-plane 3×3 acoustic tensor of a laminate result / of a layer property.
_acoustic(t, basis) = MFHC_O.acoustic_tensor(SMatrix{6, 6}(KM(t, basis)))

# In-plane Schur complement of a 4th-order property, in the layer frame.
function _schur_ip(t, basis)
    M = SMatrix{6, 6}(KM(t, basis))
    A = MFHC_O._ip_block(M)
    B = SMatrix{3, 3}(M[MFHC_O.KM_IP, MFHC_O.KM_OP])
    C = SMatrix{3, 3}(M[MFHC_O.KM_OP, MFHC_O.KM_IP])
    return A - B * MFHC_O._inv3(MFHC_O._op_block(M)) * C
end

# A random, genuinely triclinic, positive-definite stiffness.
function _rand_stiffness(rng)
    A = randn(rng, 6, 6)
    return Tens(inv_KM(Matrix(A * A' + 8I)))
end

@testset "Laminate — O1: exact out-of-plane series law (anisotropic)" begin
    rng = MersenneTwister(1001)
    for N in (2, 3, 5)
        Cs = [_rand_stiffness(rng) for _ in 1:N]
        hs = rand(rng, N) .+ 0.2
        lam = Laminate(; normal = (0, 0, 1))
        for i in 1:N
            add_layer!(lam, Symbol("L", i), Dict(:C => Cs[i]); thickness = hs[i])
        end
        Ch = homogenize(lam, Laminated(), :C)
        b = laminate_basis(lam)
        lhs = inv(_acoustic(Ch, b))
        rhs = sum(layer_volume_fraction(lam, Symbol("L", i)) * inv(_acoustic(Cs[i], b)) for i in 1:N)
        @test lhs ≈ rhs atol = ATOL_LAM
    end
end

@testset "Laminate — O2: exact in-plane Schur law (anisotropic)" begin
    rng = MersenneTwister(1002)
    for N in (2, 3, 5)
        Cs = [_rand_stiffness(rng) for _ in 1:N]
        hs = rand(rng, N) .+ 0.2
        lam = Laminate(; normal = (0, 0, 1))
        for i in 1:N
            add_layer!(lam, Symbol("L", i), Dict(:C => Cs[i]); thickness = hs[i])
        end
        Ch = homogenize(lam, Laminated(), :C)
        b = laminate_basis(lam)
        lhs = _schur_ip(Ch, b)
        rhs = sum(layer_volume_fraction(lam, Symbol("L", i)) * _schur_ip(Cs[i], b) for i in 1:N)
        @test lhs ≈ rhs atol = ATOL_LAM
    end
end

@testset "Laminate — Backus (1962) bilayer closed form" begin
    k1, μ1, k2, μ2 = 2.0, 0.8, 0.5, 0.2
    λ1, _ = _lame(k1, μ1)
    λ2, _ = _lame(k2, μ2)
    f1, f2 = 0.3, 0.7

    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => _iso(k1, μ1)); fraction = f1)
    add_layer!(lam, :B, Dict(:C => _iso(k2, μ2)); fraction = f2)
    Ch = homogenize(lam, Laminated(), :C)
    M = Matrix(KM(Ch))

    avg(g) = f1 * g(λ1, μ1) + f2 * g(λ2, μ2)
    r33 = 1 / avg((l, m) -> 1 / (l + 2m))
    rλ = avg((l, m) -> l / (l + 2m))

    @test M[3, 3] ≈ r33 atol = ATOL_LAM                                  # C₃₃₃₃
    @test M[4, 4] / 2 ≈ 1 / avg((l, m) -> 1 / m) atol = ATOL_LAM         # C₂₃₂₃
    @test M[6, 6] / 2 ≈ avg((l, m) -> m) atol = ATOL_LAM                 # C₁₂₁₂
    @test M[1, 3] ≈ r33 * rλ atol = ATOL_LAM                             # C₁₁₃₃
    @test M[1, 1] ≈ avg((l, m) -> 4m * (l + m) / (l + 2m)) + r33 * rλ^2 atol = ATOL_LAM
    @test M[1, 2] ≈ avg((l, m) -> 2m * l / (l + 2m)) + r33 * rλ^2 atol = ATOL_LAM

    # Isotropic layers ⇒ the answer is EXACTLY transversely isotropic about n,
    # and is returned as such (structural detection, not a fit).
    @test Ch isa TensND.TensTI{4}
    @test all(isapprox.(TensND.axis(Ch), (0.0, 0.0, 1.0); atol = 1.0e-14))
end

@testset "Laminate — degeneracies and invariances" begin
    k1, μ1, k2, μ2 = 2.0, 0.8, 0.5, 0.2

    # N = 1 with perfect interfaces: the layer property object itself.
    lam1 = Laminate()
    add_layer!(lam1, :A, Dict(:C => _iso(k1, μ1)); fraction = 1.0)
    @test homogenize(lam1, Laminated(), :C) === layer_property(lam1, :A, :C)
    @test homogenize(lam1, Voigt(), :C) ≈ _iso(k1, μ1)
    @test homogenize(lam1, Reuss(), :C) ≈ _iso(k1, μ1)

    # Identical layers, unequal fractions.
    lam2 = Laminate()
    add_layer!(lam2, :A, Dict(:C => _iso(k1, μ1)); thickness = 0.2)
    add_layer!(lam2, :B, Dict(:C => _iso(k1, μ1)); thickness = 0.5)
    add_layer!(lam2, :C, Dict(:C => _iso(k1, μ1)); thickness = 0.3)
    @test Matrix(KM(homogenize(lam2, Laminated(), :C))) ≈ Matrix(KM(_iso(k1, μ1))) atol = ATOL_LAM

    # Layer ORDER is irrelevant with perfect interfaces (the cell is periodic
    # and every layer sees the same traction).
    lam_a = Laminate()
    add_layer!(lam_a, :A, Dict(:C => _iso(k1, μ1)); thickness = 0.3)
    add_layer!(lam_a, :B, Dict(:C => _iso(k2, μ2)); thickness = 0.7)
    lam_b = Laminate()
    add_layer!(lam_b, :B, Dict(:C => _iso(k2, μ2)); thickness = 0.7)
    add_layer!(lam_b, :A, Dict(:C => _iso(k1, μ1)); thickness = 0.3)
    @test Matrix(KM(homogenize(lam_a, Laminated(), :C))) ≈
        Matrix(KM(homogenize(lam_b, Laminated(), :C))) atol = ATOL_LAM

    # The result depends on the fractions, not on the absolute period, when
    # every interface is perfect.
    lam_c = Laminate()
    add_layer!(lam_c, :A, Dict(:C => _iso(k1, μ1)); thickness = 3.0)
    add_layer!(lam_c, :B, Dict(:C => _iso(k2, μ2)); thickness = 7.0)
    @test Matrix(KM(homogenize(lam_c, Laminated(), :C))) ≈
        Matrix(KM(homogenize(lam_a, Laminated(), :C))) atol = ATOL_LAM
end

@testset "Laminate — frame covariance" begin
    # The same physical laminate, described in a rotated frame, must give the
    # same physical answer: layer tensors and normal rotated together.
    k1, μ1, k2, μ2 = 2.0, 0.8, 0.5, 0.2
    lam0 = Laminate(; normal = (0, 0, 1))
    add_layer!(lam0, :A, Dict(:C => _iso(k1, μ1)); fraction = 0.3)
    add_layer!(lam0, :B, Dict(:C => _iso(k2, μ2)); fraction = 0.7)
    C0 = homogenize(lam0, Laminated(), :C)

    # Isotropic layers are invariant under rotation, so only the normal moves:
    # the result must be the same TI tensor about the new axis.
    for n in ((1, 0, 0), (0, 1, 0), (1, 1, 1))
        lam = Laminate(; normal = n)
        add_layer!(lam, :A, Dict(:C => _iso(k1, μ1)); fraction = 0.3)
        add_layer!(lam, :B, Dict(:C => _iso(k2, μ2)); fraction = 0.7)
        C = homogenize(lam, Laminated(), :C)
        @test C isa TensND.TensTI{4}
        nrm = sqrt(sum(x -> x^2, n))
        @test all(isapprox.(TensND.axis(C), n ./ nrm; atol = 1.0e-12))
        # same Walpole coefficients: the physics does not know about the frame
        @test collect(TensND.get_data(C)) ≈ collect(TensND.get_data(C0)) atol = ATOL_LAM
    end
end

@testset "Laminate — bounds bracket the exact answer" begin
    # And two of the bracketings are EQUALITIES, which is the sharpest
    # statement available about a laminate:
    #   in-plane shear  C₁₂₁₂  is exactly Voigt,
    #   out-of-plane    (n·ℂ·n) is exactly Reuss.
    k1, μ1, k2, μ2 = 2.0, 0.8, 0.5, 0.2
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => _iso(k1, μ1)); fraction = 0.3)
    add_layer!(lam, :B, Dict(:C => _iso(k2, μ2)); fraction = 0.7)

    Mh = Matrix(KM(homogenize(lam, Laminated(), :C)))
    Mv = Matrix(KM(homogenize(lam, Voigt(), :C)))
    Mr = Matrix(KM(homogenize(lam, Reuss(), :C)))

    @test Mr[3, 3] ≤ Mh[3, 3] + ATOL_LAM
    @test Mh[3, 3] ≤ Mv[3, 3] + ATOL_LAM
    @test Mh[6, 6] ≈ Mv[6, 6] atol = ATOL_LAM      # in-plane shear: Voigt exact
    b = laminate_basis(lam)
    @test _acoustic(homogenize(lam, Laminated(), :C), b) ≈
        _acoustic(homogenize(lam, Reuss(), :C), b) atol = ATOL_LAM
end

@testset "Laminate — localization sum rules and block structure" begin
    rng = MersenneTwister(1003)
    Cs = [_rand_stiffness(rng) for _ in 1:3]
    hs = [0.2, 0.45, 0.35]
    lam = Laminate(; normal = (0, 0, 1))
    for i in 1:3
        add_layer!(lam, Symbol("L", i), Dict(:C => Cs[i]); thickness = hs[i])
    end
    names = layer_names(lam)
    fs = [layer_volume_fraction(lam, nm) for nm in names]
    As = [layer_strain_localization(lam, nm) for nm in names]
    Bs = [layer_stress_localization(lam, nm) for nm in names]

    @test Matrix(KM(sum(fs[i] * As[i] for i in 1:3))) ≈ Matrix(1.0I, 6, 6) atol = ATOL_LAM
    @test Matrix(KM(sum(fs[i] * Bs[i] for i in 1:3))) ≈ Matrix(1.0I, 6, 6) atol = ATOL_LAM

    Ch = homogenize(lam, Laminated(), :C)
    # ⟨ℂ:𝔸⟩ = ℂ_hom
    @test Matrix(KM(sum(fs[i] * (Cs[i] ⊡ As[i]) for i in 1:3))) ≈
        Matrix(KM(Ch)) atol = ATOL_LAM

    b = laminate_basis(lam)
    for A in As
        MA = SMatrix{6, 6}(KM(A, b))
        # The macroscopic in-plane strain reaches every layer unchanged:
        # ε_i = E + a_i ⊗ˢ n.
        @test MFHC_O._ip_block(MA) ≈ I atol = ATOL_LAM
        @test maximum(abs, MA[MFHC_O.KM_IP, MFHC_O.KM_OP]) < ATOL_LAM
    end

    # The two Hill tensors and their identities.
    P, Q = laminate_hill(lam, :L1)
    @test Matrix(KM(P ⊡ Cs[1] ⊡ P)) ≈ Matrix(KM(P)) atol = ATOL_LAM
    @test maximum(abs, Matrix(KM(Q ⊡ P))) < ATOL_LAM
end

@testset "Laminate — the four localization generics" begin
    rng = MersenneTwister(1004)
    Cs = [_rand_stiffness(rng) for _ in 1:3]
    Ks = [Tens(Symmetric(randn(rng, 3, 3) * 0.1 + 3.0I)) for _ in 1:3]
    hs = [0.2, 0.45, 0.35]
    lam = Laminate(; normal = (0.0, 0.6, 0.8))
    for i in 1:3
        add_layer!(lam, Symbol("L", i), Dict(:C => Cs[i], :K => Ks[i]); thickness = hs[i])
    end
    names = layer_names(lam)
    fs = [layer_volume_fraction(lam, nm) for nm in names]
    Ch = homogenize(lam, Laminated(), :C)
    Kh = homogenize(lam, Laminated(), :K)
    Sh = inv(Ch)
    Rh = inv(Kh)
    Id4 = Matrix(1.0I, 6, 6)
    Id2 = Matrix(1.0I, 3, 3)

    # The two aliases are the very same object as the `layer_*` names.
    for (i, nm) in enumerate(names)
        @test Matrix(KM(strain_strain_loc(lam, nm))) ==
            Matrix(KM(layer_strain_localization(lam, nm)))
        @test Matrix(KM(stress_stress_loc(lam, nm))) ==
            Matrix(KM(layer_stress_localization(lam, nm)))
        @test Matrix(gradient_gradient_loc(lam, nm)) ==
            Matrix(layer_gradient_localization(lam, nm))
        @test Matrix(flux_flux_loc(lam, nm)) == Matrix(layer_flux_localization(lam, nm))

        # The two mixed tensors, against their definitions.
        A = strain_strain_loc(lam, nm)
        @test Matrix(KM(stress_strain_loc(lam, nm))) ≈ Matrix(KM(Cs[i] ⊡ A)) atol = ATOL_LAM
        @test Matrix(KM(strain_stress_loc(lam, nm))) ≈ Matrix(KM(A ⊡ Sh)) atol = ATOL_LAM
        @test Matrix(KM(stress_stress_loc(lam, nm))) ≈
            Matrix(KM(stress_strain_loc(lam, nm) ⊡ Sh)) atol = ATOL_LAM

        a = gradient_gradient_loc(lam, nm)
        @test Matrix(flux_gradient_loc(lam, nm)) ≈ Matrix(Ks[i] ⋅ a) atol = ATOL_LAM
        @test Matrix(gradient_flux_loc(lam, nm)) ≈ Matrix(a ⋅ Rh) atol = ATOL_LAM
    end

    # The four sum rules, at both orders.
    _sum(g) = sum(fs[i] * g(names[i]) for i in eachindex(names))
    @test Matrix(KM(_sum(nm -> strain_strain_loc(lam, nm)))) ≈ Id4 atol = ATOL_LAM
    @test Matrix(KM(_sum(nm -> stress_stress_loc(lam, nm)))) ≈ Id4 atol = ATOL_LAM
    @test Matrix(KM(_sum(nm -> stress_strain_loc(lam, nm)))) ≈
        Matrix(KM(Ch)) atol = ATOL_LAM
    @test Matrix(KM(_sum(nm -> strain_stress_loc(lam, nm)))) ≈
        Matrix(KM(Sh)) atol = ATOL_LAM
    @test Matrix(_sum(nm -> gradient_gradient_loc(lam, nm))) ≈ Id2 atol = ATOL_LAM
    @test Matrix(_sum(nm -> flux_flux_loc(lam, nm))) ≈ Id2 atol = ATOL_LAM
    @test Matrix(_sum(nm -> flux_gradient_loc(lam, nm))) ≈ Matrix(Kh) atol = ATOL_LAM
    @test Matrix(_sum(nm -> gradient_flux_loc(lam, nm))) ≈ Matrix(Rh) atol = ATOL_LAM

    # The symmetry class of the answer, which is the point of `_wrap4_general`.
    #
    # A laminate of layers all TI about its own normal has localization tensors
    # TI about that same normal — but NOT major-symmetric, `𝔸ᵢ` being a product
    # of major-symmetric tensors. So the exact type is `TensTI{4,T,6}`, and the
    # 5-coefficient read-off `_wrap4` uses for the effective stiffness would
    # silently replace `ℓ₃` and `ℓ₄` by their half-sum.
    lam_ti = Laminate(; normal = (0.0, 0.6, 0.8))
    add_layer!(lam_ti, :A, Dict(:C => _iso(3.0, 1.2), :K => TensISO{3}(2.0)); fraction = 0.35)
    add_layer!(lam_ti, :B, Dict(:C => _iso(0.7, 0.3), :K => TensISO{3}(0.4)); fraction = 0.65)

    @test homogenize(lam_ti, Laminated(), :C) isa TensND.TensTI{4, Float64, 5}
    for g in (strain_strain_loc, stress_strain_loc, strain_stress_loc, stress_stress_loc)
        A = g(lam_ti, :A)
        @test A isa TensND.TensTI{4, Float64, 6}
        @test collect(TensND.axis(A)) ≈ [0.0, 0.6, 0.8]
        # ℓ₇ and ℓ₈ are structurally zero — the six-coefficient Walpole span is
        # closed under product and inverse — so nothing is dropped by N = 6.
        p8 = TensND.ti8_params_from_KM(Matrix(KM(A, laminate_basis(lam_ti))))
        @test abs(p8[7]) < ATOL_LAM
        @test abs(p8[8]) < ATOL_LAM
        # ℓ₃ ≠ ℓ₄: the tensor really is outside the major-symmetric span, so a
        # 5-coefficient read-off would have been lossy rather than merely tight.
        ℓ = TensND.get_ℓ(A)
        @test abs(ℓ[3] - ℓ[4]) > 1.0e-3
    end
    for g in (gradient_gradient_loc, flux_gradient_loc, gradient_flux_loc, flux_flux_loc)
        @test g(lam_ti, :A) isa TensND.TensTI{2, Float64, 2}
    end

    # An anisotropic layer breaks it, structurally, and an unstructured tensor
    # is returned rather than a wrong TI claim.
    lam_gen = Laminate(; normal = (0, 0, 1))
    add_layer!(lam_gen, :A, Dict(:C => _rand_stiffness(MersenneTwister(7))); fraction = 0.4)
    add_layer!(lam_gen, :B, Dict(:C => _iso(0.7, 0.3)); fraction = 0.6)
    @test !(strain_strain_loc(lam_gen, :A) isa TensND.TensTI)

    # A primal (spring) interface carries part of the macroscopic strain in the
    # displacement jumps, so the STRAIN-side rule is expected to fail while the
    # stress-side one still holds — this is the documented caveat, asserted.
    lam_s = Laminate(; normal = (0, 0, 1))
    add_layer!(
        lam_s, :A, Dict(:C => _iso(3.0, 1.2)); thickness = 0.4,
        interface = SpringInterface(0.5, 0.3)
    )
    add_layer!(lam_s, :B, Dict(:C => _iso(0.7, 0.3)); thickness = 0.6)
    fs_s = [layer_volume_fraction(lam_s, nm) for nm in layer_names(lam_s)]
    As_s = [strain_strain_loc(lam_s, nm) for nm in layer_names(lam_s)]
    Bs_s = [stress_stress_loc(lam_s, nm) for nm in layer_names(lam_s)]
    @test !isapprox(
        Matrix(KM(sum(fs_s[i] * As_s[i] for i in 1:2))), Id4; atol = 1.0e-6
    )
    @test Matrix(KM(sum(fs_s[i] * Bs_s[i] for i in 1:2))) ≈ Id4 atol = ATOL_LAM
end

@testset "Laminate — the exact-TI claim, class by class" begin
    # The result is transversely isotropic about `n` **only** when every layer
    # is isotropic or major-symmetric TI about that same axis. Anything else —
    # a TI layer with another axis, an orthotropic layer (even one whose axes
    # coincide with the laminate frame), a triclinic one — must fall through to
    # a generic `Tens`, and the returned object must then genuinely NOT be TI.
    n = (0.0, 0.0, 1.0)
    ortho_km = [
        25.0 8.0 7.0 0.0 0.0 0.0
        8.0 20.0 6.0 0.0 0.0 0.0
        7.0 6.0 15.0 0.0 0.0 0.0
        0.0 0.0 0.0 10.0 0.0 0.0
        0.0 0.0 0.0 0.0 9.0 0.0
        0.0 0.0 0.0 0.0 0.0 8.0
    ]
    C_ti_z = TensTI{4}(20.0, 25.0, 7.0, 6.0, 5.0, n)
    C_ti_x = TensTI{4}(20.0, 25.0, 7.0, 6.0, 5.0, (1.0, 0.0, 0.0))

    ti_expected = (
        (true, [_iso(2.0, 0.8), _iso(0.5, 0.2)]),
        (true, [_iso(2.0, 0.8), C_ti_z]),
        (false, [_iso(2.0, 0.8), C_ti_x]),
        (false, [_iso(2.0, 0.8), TensOrtho(ortho_km, TensND.RotatedBasis(0.3, 0.5, 0.2))]),
        (false, [_iso(2.0, 0.8), TensOrtho(ortho_km, TensND.CanonicalBasis{3, Float64}())]),
    )

    for (want_ti, Cs) in ti_expected
        lam = Laminate(; normal = n)
        for (i, Ci) in enumerate(Cs)
            add_layer!(lam, Symbol("L", i), Dict(:C => Ci); fraction = 1 / length(Cs))
        end
        Ch = homogenize(lam, Laminated(), :C)
        @test (Ch isa TensND.TensTI{4}) == want_ti

        # ... and the claim matches reality: check the TI identities on the
        # component matrix in the layer frame.
        M = Matrix(KM(Ch, laminate_basis(lam)))
        resid = max(
            abs(M[1, 1] - M[2, 2]), abs(M[1, 3] - M[2, 3]), abs(M[4, 4] - M[5, 5]),
            abs(M[6, 6] - (M[1, 1] - M[1, 2])),
            maximum(abs, [M[1, 4], M[1, 5], M[1, 6], M[2, 4], M[3, 6], M[4, 5], M[4, 6]])
        )
        want_ti ? (@test resid < ATOL_LAM) : (@test resid > 1.0e-3)
    end
end

@testset "Laminate — a non-major-symmetric TI layer is not claimed as TI" begin
    # `TensTI{4,T,8}` is transversely isotropic but NOT major-symmetric: it is
    # what the exact rotation-group average produces (ℓ₃ ≠ ℓ₄, plus the
    # antisymmetric azimuthal couplings ℓ₇, ℓ₈). Sending it through the
    # 5-parameter Walpole read-off would silently discard that content, so the
    # TI branch must refuse it and the generic wrapper must be lossless.
    n = (0.0, 0.0, 1.0)
    W8 = TensTI{4}(20.0, 25.0, 7.0, 9.0, 6.0, 5.0, 0.4, 0.3, n)
    @test W8 isa TensND.TensTI{4, Float64, 8}
    @test !isapprox(Matrix(KM(W8)), Matrix(KM(W8))'; atol = 1.0e-12)   # not major-symmetric

    lam = Laminate(; normal = n)
    add_layer!(lam, :A, Dict(:C => W8); fraction = 0.5)
    add_layer!(lam, :B, Dict(:C => _iso(0.5, 0.2)); fraction = 0.5)
    Ch = homogenize(lam, Laminated(), :C)
    @test !(Ch isa TensND.TensTI)

    # Lossless: the wrapped result equals the raw kernel output exactly.
    b = laminate_basis(lam)
    C6s = [
        SMatrix{6, 6}(KM(W8, b)),
        SMatrix{6, 6}(KM(_iso(0.5, 0.2), b)),
    ]
    Z = zero(SMatrix{6, 6, Float64})
    raw = MFHC_O.laminate_stiffness(C6s, [0.5, 0.5], Z, Z)
    @test Matrix(KM(Ch, b)) ≈ Matrix(raw) atol = ATOL_LAM
end

@testset "Laminate — asymmetric anisotropic case" begin
    # The mandatory asymmetric case: unequal fractions, one orthotropic layer
    # in a rotated frame and one TI layer NOT coaxial with the laminate
    # normal, so no degeneracy can hide a convention error. The result must be
    # a generic `Tens` (the TI detection is structural, and it must say no).
    C_ortho_km = [
        25.0 8.0 7.0 0.0 0.0 0.0
        8.0 20.0 6.0 0.0 0.0 0.0
        7.0 6.0 15.0 0.0 0.0 0.0
        0.0 0.0 0.0 10.0 0.0 0.0
        0.0 0.0 0.0 0.0 9.0 0.0
        0.0 0.0 0.0 0.0 0.0 8.0
    ]
    C_ortho = TensOrtho(C_ortho_km, TensND.RotatedBasis(0.3, 0.5, 0.2))
    C_ti = TensTI{4}(20.0, 25.0, 7.0, 6.0, 5.0, (1.0, 0.0, 0.0))    # axis ⟂ n
    C_iso = _iso(2.0, 0.8)

    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :O, Dict(:C => C_ortho); thickness = 0.15)
    add_layer!(lam, :T, Dict(:C => C_ti); thickness = 0.55)
    add_layer!(lam, :I, Dict(:C => C_iso); thickness = 0.3)

    Ch = homogenize(lam, Laminated(), :C)
    @test !(Ch isa TensND.TensTI)                # structural detection says no
    @test !(Ch isa TensND.TensISO)

    b = laminate_basis(lam)
    names = layer_names(lam)
    fs = [layer_volume_fraction(lam, nm) for nm in names]
    props = [layer_property(lam, nm, :C) for nm in names]

    # Both oracles still hold — that is the point of the asymmetric case.
    @test inv(_acoustic(Ch, b)) ≈ sum(fs[i] * inv(_acoustic(props[i], b)) for i in 1:3) atol = ATOL_LAM
    @test _schur_ip(Ch, b) ≈ sum(fs[i] * _schur_ip(props[i], b) for i in 1:3) atol = ATOL_LAM

    # Major symmetry is preserved.
    M = Matrix(KM(Ch, b))
    @test M ≈ M' atol = ATOL_LAM

    As = [layer_strain_localization(lam, nm) for nm in names]
    @test Matrix(KM(sum(fs[i] * As[i] for i in 1:3))) ≈ Matrix(1.0I, 6, 6) atol = ATOL_LAM
end
