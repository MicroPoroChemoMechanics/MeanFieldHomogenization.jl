# =============================================================================
#  test_laminate_interfaces.jl — imperfect interfaces of a periodic laminate.
#
#  A planar interface is the curvature-free case of the spherical one, so the
#  five `LayeredSpheres` types are reused unchanged and the algebra collapses
#  to two additive terms. That makes the laminate the sharpest available test
#  of the package's interface conventions:
#
#   * PRIMAL (spring, Kapitza) adds `Σ_j 𝕂_j / L` to the out-of-plane
#     compliance average and nothing else — oracle O1;
#   * DUAL (membrane, surface-conductive) adds `Σ_j ℂˢ_j / L` to the in-plane
#     block and nothing else — oracle O2.
#
#  Being complementary, the two oracles separate a spring bug from a membrane
#  bug. The `1/L` weight is an interface *density*: at fixed volume fractions,
#  doubling the period halves the correction.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using Random
using ForwardDiff
# `Tensors.SymmetricTensor` is used below. `using TensND` does not bring the
# name in, so without this the file only ran because an earlier file in the
# suite had leaked it into `Main` — passing in the full run and erroring on
# its own, which is the worst way for a test to be wrong.
import Tensors

const MFHC_I = MeanFieldHomogenization.Core
const ATOL_ITF = 1.0e-11

_isoi(k, μ) = TensISO{3}(3k, 2μ)
_acoustic_i(t, basis) = MFHC_I.acoustic_tensor(SMatrix{6, 6}(KM(t, basis)))

function _schur_ip_i(t, basis)
    M = SMatrix{6, 6}(KM(t, basis))
    A = MFHC_I._ip_block(M)
    B = SMatrix{3, 3}(M[MFHC_I.KM_IP, MFHC_I.KM_OP])
    C = SMatrix{3, 3}(M[MFHC_I.KM_OP, MFHC_I.KM_IP])
    return A - B * MFHC_I._inv3(MFHC_I._op_block(M)) * C
end

# Two-layer reference cell, parameterised by its interfaces and its period.
function _bilayer(; itf1 = PerfectInterface(), itf2 = PerfectInterface(), scale = 1.0)
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(
        lam, :A, Dict(:C => _isoi(2.0, 0.8), :K => TensISO{3}(2.0));
        thickness = 0.3scale, interface = itf1
    )
    add_layer!(
        lam, :B, Dict(:C => _isoi(0.5, 0.2), :K => TensISO{3}(0.3));
        thickness = 0.7scale, interface = itf2
    )
    return lam
end

@testset "Interfaces — the null limits are the perfect interface" begin
    ref = homogenize(_bilayer(), Laminated(), :C)
    refK = homogenize(_bilayer(), Laminated(), :K)

    @test homogenize(_bilayer(; itf1 = SpringInterface(0.0, 0.0)), Laminated(), :C) ≈ ref
    @test homogenize(_bilayer(; itf1 = MembraneInterface(0.0, 0.0)), Laminated(), :C) ≈ ref
    @test homogenize(_bilayer(; itf1 = KapitzaInterface(0.0)), Laminated(), :K) ≈ refK
    @test homogenize(
        _bilayer(; itf1 = SurfaceConductiveInterface(0.0)), Laminated(), :K
    ) ≈ refK

    # An elastic interface leaves transport alone and vice versa.
    @test homogenize(_bilayer(; itf1 = SpringInterface(1.0e-1, 1.0e-1)), Laminated(), :K) ≈ refK
    @test homogenize(_bilayer(; itf1 = KapitzaInterface(1.0e-1)), Laminated(), :C) ≈ ref
end

@testset "Interfaces — O1: spring enters the out-of-plane law, exactly" begin
    kn, kt = 0.013, 0.021
    L = 1.0
    lam = _bilayer(; itf1 = SpringInterface(kn, kt), itf2 = SpringInterface(2kn, 3kt))
    Ch = homogenize(lam, Laminated(), :C)
    b = laminate_basis(lam)

    𝕂tot = Diagonal([kt + 3kt, kt + 3kt, kn + 2kn])       # Σ_j 𝕂_j, in (ℓ, m, n)
    lhs = inv(_acoustic_i(Ch, b))
    rhs = sum(
        layer_volume_fraction(lam, nm) * inv(_acoustic_i(layer_property(lam, nm, :C), b))
            for nm in layer_names(lam)
    ) + 𝕂tot / L
    @test lhs ≈ rhs atol = ATOL_ITF

    # ... and the in-plane Schur law is UNTOUCHED by a spring (oracle O2).
    @test _schur_ip_i(Ch, b) ≈ sum(
        layer_volume_fraction(lam, nm) * _schur_ip_i(layer_property(lam, nm, :C), b)
            for nm in layer_names(lam)
    ) atol = ATOL_ITF
end

@testset "Interfaces — O2: membrane enters the in-plane law, exactly" begin
    κs, μs = 0.07, 0.04
    L = 1.0
    lam = _bilayer(; itf1 = MembraneInterface(κs, μs))
    Ch = homogenize(lam, Laminated(), :C)
    ref = homogenize(_bilayer(), Laminated(), :C)
    b = laminate_basis(lam)

    Cs = [κs + μs κs - μs 0.0; κs - μs κs + μs 0.0; 0.0 0.0 2μs]
    @test _schur_ip_i(Ch, b) ≈ _schur_ip_i(ref, b) + Cs / L atol = ATOL_ITF
    # A planar membrane produces NO traction jump, so the out-of-plane law is
    # untouched (oracle O1) — this is what distinguishes the flat case from
    # the spherical one.
    @test _acoustic_i(Ch, b) ≈ _acoustic_i(ref, b) atol = ATOL_ITF

    # Read off directly: the in-plane shear gains exactly μs/L.
    M, Mr = Matrix(KM(Ch)), Matrix(KM(ref))
    @test M[6, 6] / 2 - Mr[6, 6] / 2 ≈ μs / L atol = ATOL_ITF
    @test M[3, 3] ≈ Mr[3, 3] atol = ATOL_ITF
end

@testset "Interfaces — the 1/L size effect" begin
    kn, kt = 0.05, 0.05
    ref = Matrix(KM(homogenize(_bilayer(), Laminated(), :C)))
    itf = SpringInterface(kn, kt)

    C1 = Matrix(KM(homogenize(_bilayer(; itf1 = itf, scale = 1.0), Laminated(), :C)))
    C2 = Matrix(KM(homogenize(_bilayer(; itf1 = itf, scale = 2.0), Laminated(), :C)))

    # The interface correction lives in the out-of-plane compliance and scales
    # like 1/L: doubling every thickness at fixed fractions halves it.
    d1 = 1 / C1[3, 3] - 1 / ref[3, 3]
    d2 = 1 / C2[3, 3] - 1 / ref[3, 3]
    @test d2 ≈ d1 / 2 atol = ATOL_ITF

    # L → ∞ recovers the perfect interface.
    Cbig = Matrix(KM(homogenize(_bilayer(; itf1 = itf, scale = 1.0e8), Laminated(), :C)))
    @test Cbig ≈ ref rtol = 1.0e-7
end

@testset "Interfaces — spring softens monotonically, and decouples" begin
    prev = Inf
    for kn in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e2)
        lam = _bilayer(; itf1 = SpringInterface(kn, kn))
        C33 = Matrix(KM(homogenize(lam, Laminated(), :C)))[3, 3]
        @test C33 < prev                       # strictly softer as kn grows
        prev = C33
    end
    # kn → ∞: the layers decouple, the cell carries no normal stress.
    lam = _bilayer(; itf1 = SpringInterface(1.0e12, 1.0e12))
    Ch = homogenize(lam, Laminated(), :C)
    @test norm(_acoustic_i(Ch, laminate_basis(lam))) < 1.0e-8
    # ... but the in-plane stiffness is untouched (O2 again).
    @test Matrix(KM(Ch))[6, 6] ≈ Matrix(KM(homogenize(_bilayer(), Laminated(), :C)))[6, 6] atol = ATOL_ITF
end

@testset "Interfaces — the displacement jump" begin
    kn, kt = 1.0e-2, 3.0e-3
    lam = _bilayer(; itf1 = SpringInterface(kn, kt))
    Ch = homogenize(lam, Laminated(), :C)

    # Pure normal macroscopic strain: the jump is normal, of size kn·Σ₃₃.
    E = Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> (i == 3 && j == 3) ? 1.0e-3 : 0.0))
    j = interface_jump(lam, 1, E)
    Σ = Ch ⊡ E
    @test j[1] ≈ 0 atol = 1.0e-14
    @test j[2] ≈ 0 atol = 1.0e-14
    @test j[3] ≈ kn * Matrix(components(Σ))[3, 3] atol = 1.0e-14

    # Pure out-of-plane shear: the jump is tangential, of size kt·Σ₁₃.
    Es = Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> ((i, j) == (1, 3) || (i, j) == (3, 1)) ? 5.0e-4 : 0.0
        )
    )
    js = interface_jump(lam, 1, Es)
    Σs = Ch ⊡ Es
    @test js[1] ≈ kt * Matrix(components(Σs))[1, 3] atol = 1.0e-14
    @test js[3] ≈ 0 atol = 1.0e-14

    # A perfect or a dual interface produces no jump at all.
    @test all(interface_jump(_bilayer(), 1, E) .== 0)
    @test all(interface_jump(_bilayer(; itf1 = MembraneInterface(0.1, 0.1)), 1, E) .== 0)
end

@testset "Interfaces — conduction: Kapitza and the surface layer" begin
    ρ, ks, L = 0.11, 0.06, 1.0
    kA, kB, fA, fB = 2.0, 0.3, 0.3, 0.7

    lamK = _bilayer(; itf1 = KapitzaInterface(ρ))
    KhK = Matrix(components(homogenize(lamK, Laminated(), :K)))
    # The series law with the interfacial resistance added — closed form.
    @test 1 / KhK[3, 3] ≈ fA / kA + fB / kB + ρ / L atol = ATOL_ITF
    @test KhK[1, 1] ≈ fA * kA + fB * kB atol = ATOL_ITF     # in-plane untouched

    lamS = _bilayer(; itf1 = SurfaceConductiveInterface(ks))
    KhS = Matrix(components(homogenize(lamS, Laminated(), :K)))
    @test KhS[1, 1] ≈ fA * kA + fB * kB + ks / L atol = ATOL_ITF
    @test KhS[2, 2] ≈ KhS[1, 1] atol = ATOL_ITF
    @test 1 / KhS[3, 3] ≈ fA / kA + fB / kB atol = ATOL_ITF  # out-of-plane untouched

    # Both at once, and two interfaces: the terms are additive.
    lamB = _bilayer(; itf1 = KapitzaInterface(ρ), itf2 = SurfaceConductiveInterface(ks))
    KhB = Matrix(components(homogenize(lamB, Laminated(), :K)))
    @test 1 / KhB[3, 3] ≈ fA / kA + fB / kB + ρ / L atol = ATOL_ITF
    @test KhB[1, 1] ≈ fA * kA + fB * kB + ks / L atol = ATOL_ITF

    # ρ → ∞ insulates the stack; ks → ∞ short-circuits the plane.
    lamI = _bilayer(; itf1 = KapitzaInterface(1.0e12))
    @test Matrix(components(homogenize(lamI, Laminated(), :K)))[3, 3] < 1.0e-9
end

@testset "Interfaces — conduction with anisotropic layers" begin
    rng = MersenneTwister(2001)
    K3 = [Tens(Matrix(A * A' + 3I)) for A in (randn(rng, 3, 3), randn(rng, 3, 3))]
    ρ = 0.09
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:K => K3[1]); thickness = 0.4, interface = KapitzaInterface(ρ))
    add_layer!(lam, :B, Dict(:K => K3[2]); thickness = 0.6)
    Kh = Matrix(components(homogenize(lam, Laminated(), :K)))
    fs = [layer_volume_fraction(lam, nm) for nm in layer_names(lam)]
    kn = [Matrix(components(K3[i]))[3, 3] for i in 1:2]
    @test 1 / Kh[3, 3] ≈ sum(fs[i] / kn[i] for i in 1:2) + ρ / laminate_period(lam) atol = ATOL_ITF
    @test !(homogenize(lam, Laminated(), :K) isa TensND.TensTI)   # structural: not TI
end

# =============================================================================
#  Anisotropic interfaces — a plane, unlike a sphere, imposes no symmetry on
#  the interface, so the laminate accepts a full tensor. These check that the
#  tensor-valued types reproduce the scalar ones on isotropic input, that both
#  oracles stay exact for a genuinely anisotropic interface, and that the
#  exact-TI claim correctly refuses such a cell.
# =============================================================================

@testset "Anisotropic interfaces — reduce to the isotropic ones" begin
    kn, kt = 0.013, 0.021
    𝕂 = SMatrix{3, 3}(Diagonal([kt, kt, kn]))          # (ℓ, m, n) frame
    a = homogenize(_bilayer(; itf1 = AnisotropicSpringInterface(𝕂)), Laminated(), :C)
    b0 = homogenize(_bilayer(; itf1 = SpringInterface(kn, kt)), Laminated(), :C)
    @test Matrix(KM(a)) ≈ Matrix(KM(b0)) atol = ATOL_ITF

    κs, μs = 0.07, 0.04
    Cs = SMatrix{3, 3}(κs + μs, κs - μs, 0.0, κs - μs, κs + μs, 0.0, 0.0, 0.0, 2μs)
    am = homogenize(_bilayer(; itf1 = AnisotropicMembraneInterface(Cs)), Laminated(), :C)
    bm = homogenize(_bilayer(; itf1 = MembraneInterface(κs, μs)), Laminated(), :C)
    @test Matrix(KM(am)) ≈ Matrix(KM(bm)) atol = ATOL_ITF

    ks = 0.06
    Ks = SMatrix{3, 3}(ks, 0.0, 0.0, 0.0, ks, 0.0, 0.0, 0.0, 0.0)
    ak = homogenize(_bilayer(; itf1 = AnisotropicSurfaceConductiveInterface(Ks)), Laminated(), :K)
    bk = homogenize(_bilayer(; itf1 = SurfaceConductiveInterface(ks)), Laminated(), :K)
    @test Matrix(components(ak)) ≈ Matrix(components(bk)) atol = ATOL_ITF

    # The null tensor is still the perfect interface.
    Z3 = zero(SMatrix{3, 3, Float64})
    @test Matrix(KM(homogenize(_bilayer(; itf1 = AnisotropicSpringInterface(Z3)), Laminated(), :C))) ≈
        Matrix(KM(homogenize(_bilayer(), Laminated(), :C))) atol = ATOL_ITF
end

@testset "Anisotropic interfaces — both oracles stay exact" begin
    # A full, non-diagonal compliance: normal and tangential directions with
    # their own compliances AND coupled.
    𝕂 = SMatrix{3, 3}(
        3.0e-3, 5.0e-4, 7.0e-4,
        5.0e-4, 8.0e-3, 2.0e-4,
        7.0e-4, 2.0e-4, 1.1e-2
    )
    @test 𝕂 ≈ 𝕂'                                        # a compliance is symmetric
    lam = _bilayer(; itf1 = AnisotropicSpringInterface(𝕂))
    Ch = homogenize(lam, Laminated(), :C)
    b = laminate_basis(lam)

    # O1 — the full tensor simply adds to the out-of-plane series law.
    @test inv(_acoustic_i(Ch, b)) ≈ sum(
        layer_volume_fraction(lam, nm) * inv(_acoustic_i(layer_property(lam, nm, :C), b))
            for nm in layer_names(lam)
    ) + 𝕂 / laminate_period(lam) atol = ATOL_ITF
    # O2 — a spring, anisotropic or not, is invisible in the plane.
    @test _schur_ip_i(Ch, b) ≈ sum(
        layer_volume_fraction(lam, nm) * _schur_ip_i(layer_property(lam, nm, :C), b)
            for nm in layer_names(lam)
    ) atol = ATOL_ITF

    # A full in-plane surface stiffness, with a shear-extension coupling.
    Cs = SMatrix{3, 3}(0.2, 0.05, 0.01, 0.05, 0.09, 0.02, 0.01, 0.02, 0.06)
    @test Cs ≈ Cs'
    lamm = _bilayer(; itf1 = AnisotropicMembraneInterface(Cs))
    Chm = homogenize(lamm, Laminated(), :C)
    ref = homogenize(_bilayer(), Laminated(), :C)
    @test _schur_ip_i(Chm, b) ≈ _schur_ip_i(ref, b) + Cs / laminate_period(lamm) atol = ATOL_ITF
    @test _acoustic_i(Chm, b) ≈ _acoustic_i(ref, b) atol = ATOL_ITF   # no traction jump

    # An anisotropic surface conductivity adds to the in-plane conduction only.
    Ks = SMatrix{3, 3}(0.09, 0.02, 0.0, 0.02, 0.04, 0.0, 0.0, 0.0, 0.0)
    lamk = _bilayer(; itf1 = AnisotropicSurfaceConductiveInterface(Ks))
    Kh = Matrix(components(homogenize(lamk, Laminated(), :K)))
    Kr = Matrix(components(homogenize(_bilayer(), Laminated(), :K)))
    @test (Kh - Kr)[1:2, 1:2] ≈ Matrix(Ks)[1:2, 1:2] atol = ATOL_ITF
    @test Kh[3, 3] ≈ Kr[3, 3] atol = ATOL_ITF
end

@testset "Anisotropic interfaces — the exact-TI claim refuses them" begin
    # Isotropic layers, but an anisotropic interface: the stack is NOT
    # transversely isotropic, and claiming so would project away the in-plane
    # texture the interface carries.
    𝕂 = SMatrix{3, 3}(3.0e-3, 5.0e-4, 7.0e-4, 5.0e-4, 8.0e-3, 2.0e-4, 7.0e-4, 2.0e-4, 1.1e-2)
    lam = _bilayer(; itf1 = AnisotropicSpringInterface(𝕂))
    Ch = homogenize(lam, Laminated(), :C)
    @test !(Ch isa TensND.TensTI)
    M = Matrix(KM(Ch, laminate_basis(lam)))
    @test abs(M[4, 4] - M[5, 5]) > 1.0e-6          # genuinely not TI

    # ... while the scalar interface types keep the TI claim.
    @test homogenize(_bilayer(; itf1 = SpringInterface(1.0e-2, 2.0e-2)), Laminated(), :C) isa
        TensND.TensTI{4}
    @test homogenize(_bilayer(; itf1 = MembraneInterface(0.07, 0.04)), Laminated(), :C) isa
        TensND.TensTI{4}
end

@testset "Anisotropic interfaces — TensND-valued fields and a rotated frame" begin
    # The tensor may be given as a plain matrix (read in the layer frame) or as
    # a TensND tensor carrying its own basis, converted on use.
    𝕂 = SMatrix{3, 3}(3.0e-3, 5.0e-4, 7.0e-4, 5.0e-4, 8.0e-3, 2.0e-4, 7.0e-4, 2.0e-4, 1.1e-2)
    lam = Laminate(; normal = (1, 1, 1))
    b = laminate_basis(lam)
    add_layer!(
        lam, :A, Dict(:C => _isoi(2.0, 0.8)); thickness = 0.3,
        interface = AnisotropicSpringInterface(Tens(Matrix(𝕂), b))
    )
    add_layer!(lam, :B, Dict(:C => _isoi(0.5, 0.2)); thickness = 0.7)
    Ch = homogenize(lam, Laminated(), :C)

    @test inv(_acoustic_i(Ch, b)) ≈ sum(
        layer_volume_fraction(lam, nm) * inv(_acoustic_i(layer_property(lam, nm, :C), b))
            for nm in layer_names(lam)
    ) + 𝕂 / laminate_period(lam) atol = ATOL_ITF

    # A plain matrix in the same (layer) frame gives the same answer.
    lam2 = Laminate(; normal = (1, 1, 1))
    add_layer!(
        lam2, :A, Dict(:C => _isoi(2.0, 0.8)); thickness = 0.3,
        interface = AnisotropicSpringInterface(𝕂)
    )
    add_layer!(lam2, :B, Dict(:C => _isoi(0.5, 0.2)); thickness = 0.7)
    @test Matrix(KM(homogenize(lam2, Laminated(), :C), b)) ≈ Matrix(KM(Ch, b)) atol = ATOL_ITF
end

@testset "Anisotropic interfaces — ForwardDiff reaches a tensor entry" begin
    base = [3.0e-3 5.0e-4 7.0e-4; 5.0e-4 8.0e-3 2.0e-4; 7.0e-4 2.0e-4 0.0]
    f = function (x)
        K = SMatrix{3, 3}(base + [0.0 0 0; 0 0 0; 0 0 x])
        return Matrix(KM(homogenize(_bilayer(; itf1 = AnisotropicSpringInterface(K)), Laminated(), :C)))[3, 3]
    end
    h = 1.0e-8
    @test ForwardDiff.derivative(f, 1.1e-2) ≈ (f(1.1e-2 + h) - f(1.1e-2 - h)) / (2h) rtol = 1.0e-5
end
