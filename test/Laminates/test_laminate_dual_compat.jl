# =============================================================================
#  test_laminate_dual_compat.jl — ForwardDiff through the laminate.
#
#  Mirrors `test/Schemes/test_dual_compat.jl`: every lens × every scheme, each
#  derivative checked against a central finite difference. Number types are
#  part of the contract, and the package's rule is that a `Dual` must be
#  exercised THROUGH `derivative`, not merely fed to a kernel.
#
#  What this pins, beyond "it runs":
#   * the layer thickness reaches both the volume fractions and the period, so
#     it also carries the interface size effect;
#   * an interface compliance is differentiable even though the layers and the
#     thicknesses stay `Float64` (the element type has to be promoted with the
#     interfaces, not just with the layers);
#   * the cofactor `_inv3` / `_inv_km6` route keeps the derivative exact where
#     an SVD-based pseudo-inverse would not be differentiable at all.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const RTOL_AD = 1.0e-5
const FD_H = 1.0e-6

_isod(k, μ) = TensISO{3}(3k, 2μ)

# Central finite difference of a scalar function.
_fd(f, x; h = FD_H) = (f(x + h) - f(x - h)) / (2h)

# The reference cell, rebuilt from scratch for each finite-difference point so
# that nothing is shared between evaluations.
function _cell(; hA = 0.3, hB = 0.7, μA = 0.8, sn = 0.0, st = 0.0, κs = 0.0, μs = 0.0)
    lam = Laminate(; normal = (0, 0, 1))
    itf = (sn == 0 && st == 0) ? PerfectInterface() : SpringInterface(; sn = sn, st = st)
    itf2 = (κs == 0 && μs == 0) ? PerfectInterface() : MembraneInterface(κs, μs)
    add_layer!(
        lam, :A, Dict(:C => _isod(2.0, μA), :K => TensISO{3}(2.0));
        thickness = hA, interface = itf
    )
    add_layer!(
        lam, :B, Dict(:C => _isod(0.5, 0.2), :K => TensISO{3}(0.3));
        thickness = hB, interface = itf2
    )
    return lam
end

_C33(C) = Matrix(KM(C))[3, 3]
_C66(C) = Matrix(KM(C))[6, 6]

@testset "AD — layer thickness, every scheme" begin
    for scheme in (Laminated(), Voigt(), Reuss())
        f = x -> _C33(homogenize(_cell(; hA = x), scheme, :C))
        ad = derivative(_cell(), scheme, thickness(:A); indexer = _C33)
        @test ad ≈ _fd(f, 0.3) rtol = RTOL_AD
        @test !iszero(ad)
    end
end

@testset "AD — layer modulus, every scheme" begin
    for scheme in (Laminated(), Voigt(), Reuss())
        # `TensISO` stores (3κ, 2μ), so the lens variable is 2μ.
        f = x -> _C33(homogenize(_cell(; μA = x / 2), scheme, :C))
        ad = derivative(_cell(), scheme, property(:A, :C, :shear); indexer = _C33)
        @test ad ≈ _fd(f, 2 * 0.8) rtol = RTOL_AD
        @test !iszero(ad)
    end
end

@testset "AD — interface compliance and surface moduli" begin
    # A spring interface: only the interface carries the `Dual`, while the
    # layers and thicknesses stay `Float64`.
    sn0, st0 = 1.0e-2, 5.0e-3
    lam = _cell(; sn = sn0, st = st0)

    ad_sn = derivative(lam, Laminated(), interface_param(1, :sn); indexer = _C33)
    @test ad_sn ≈ _fd(x -> _C33(homogenize(_cell(; sn = x, st = st0), Laminated(), :C)), sn0) rtol = RTOL_AD
    @test ad_sn < 0                                    # a softer interface softens the stack

    ad_st = derivative(lam, Laminated(), interface_param(1, :st); indexer = C -> Matrix(KM(C))[4, 4])
    @test ad_st ≈ _fd(
        x -> Matrix(KM(homogenize(_cell(; sn = sn0, st = x), Laminated(), :C)))[4, 4], st0
    ) rtol = RTOL_AD

    # The stiffness spelling differentiates too, and the chain rule through
    # `sn = 1/kn` pins the two against each other: ∂C/∂kn = −sn² ∂C/∂sn.  This
    # is what makes `interface_param(i, :kn)` and `interface_param(i, :sn)` both
    # legitimate rather than one of them a silent alias of the other.
    ad_kn = derivative(lam, Laminated(), interface_param(1, :kn); indexer = _C33)
    @test ad_kn ≈ -sn0^2 * ad_sn rtol = 1.0e-10
    @test ad_kn > 0                                    # a stiffer interface stiffens the stack

    ad_kt = derivative(lam, Laminated(), interface_param(1, :kt); indexer = C -> Matrix(KM(C))[4, 4])
    @test ad_kt ≈ -st0^2 * ad_st rtol = 1.0e-10

    # A membrane surface modulus (the second interface of the cell).
    lamm = _cell(; κs = 0.07, μs = 0.04)
    ad_μs = derivative(lamm, Laminated(), interface_param(2, :μs); indexer = _C66)
    @test ad_μs ≈ _fd(x -> _C66(homogenize(_cell(; κs = 0.07, μs = x), Laminated(), :C)), 0.04) rtol = RTOL_AD
    # ∂C₁₂₁₂/∂μs = 2/L exactly (the membrane term is additive, in Mandel form).
    @test ad_μs ≈ 2 / laminate_period(lamm) rtol = 1.0e-8
end

@testset "AD — thickness carries the interface size effect" begin
    # With a spring interface the thickness derivative differs from the pure
    # volume-fraction one: changing hA also changes L, hence the 1/L weight.
    lam = _cell(; sn = 5.0e-2)
    ad = derivative(lam, Laminated(), thickness(:A); indexer = _C33)
    @test ad ≈ _fd(x -> _C33(homogenize(_cell(; hA = x, sn = 5.0e-2), Laminated(), :C)), 0.3) rtol = RTOL_AD

    # Scaling the whole cell changes nothing without interfaces …
    f_perf = s -> _C33(homogenize(_cell(; hA = 0.3s, hB = 0.7s), Laminated(), :C))
    @test abs(_fd(f_perf, 1.0)) < 1.0e-9
    # … but is a genuine size effect with one.
    f_spring = s -> _C33(homogenize(_cell(; hA = 0.3s, hB = 0.7s, sn = 5.0e-2), Laminated(), :C))
    @test abs(_fd(f_spring, 1.0)) > 1.0e-3
end

@testset "AD — conduction" begin
    for scheme in (Laminated(), Voigt(), Reuss())
        ad = derivative(
            _cell(), scheme, property(:A, :K, 1);
            output = :K, indexer = K -> Matrix(components(K))[3, 3]
        )
        f = function (x)
            lam = Laminate(; normal = (0, 0, 1))
            add_layer!(lam, :A, Dict(:K => TensISO{3}(x)); thickness = 0.3)
            add_layer!(lam, :B, Dict(:K => TensISO{3}(0.3)); thickness = 0.7)
            return Matrix(components(homogenize(lam, scheme, :K)))[3, 3]
        end
        @test ad ≈ _fd(f, 2.0) rtol = RTOL_AD
    end
end

@testset "AD — gradient over several lenses at once" begin
    lam = _cell(; sn = 1.0e-2)
    ps = [thickness(:A), property(:A, :C, :shear), interface_param(1, :sn)]
    # Qualified: `gradient` is exported by Tensors, Ferrite, Symbolics and
    # Zygote too, all of which are loaded by the time the full suite runs.
    g = MeanFieldHomogenization.gradient(lam, Laminated(), ps; indexer = _C33)
    @test length(g) == 3
    @test g[1] ≈ _fd(x -> _C33(homogenize(_cell(; hA = x, sn = 1.0e-2), Laminated(), :C)), 0.3) rtol = RTOL_AD
    @test g[2] ≈ _fd(x -> _C33(homogenize(_cell(; μA = x / 2, sn = 1.0e-2), Laminated(), :C)), 1.6) rtol = RTOL_AD
    @test g[3] ≈ _fd(x -> _C33(homogenize(_cell(; sn = x), Laminated(), :C)), 1.0e-2) rtol = RTOL_AD
end

@testset "AD — the lenses that must refuse" begin
    lam = _cell()
    # A laminate stores thicknesses; the fractions are derived. Silently
    # reinterpreting an `AmountParameter` would be a footgun.
    @test_throws ArgumentError get_param(lam, amount(:A))
    @test_throws ArgumentError set_param(lam, amount(:A), 0.5)
    @test_throws ArgumentError get_param(lam, thickness(:Z))
    @test_throws ArgumentError get_param(lam, interface_param(1, :nope))
    # A spring stores compliances and exposes stiffnesses, so its parameter set
    # is {kn, kt, sn, st}; anything else must be refused on the WRITE path too,
    # not only on the read one.
    @test_throws ArgumentError set_param(lam, interface_param(1, :nope), 1.0)
    @test_throws ArgumentError set_param(lam, interface_param(1, :κs), 1.0)
end

@testset "AD — anisotropic layers, second derivative" begin
    # The kernel is smooth: `_inv3` and `_inv_km6` are rational in their
    # entries, so nested Duals go through and match a finite difference of
    # the first derivative.
    f = x -> _C33(homogenize(_cell(; μA = x), Laminated(), :C))
    d1 = x -> ForwardDiff.derivative(f, x)
    @test ForwardDiff.derivative(d1, 0.8) ≈ _fd(d1, 0.8; h = 1.0e-4) rtol = 1.0e-4
end

@testset "ForwardDiff — a TILTED laminate" begin
    # Every other test here uses the canonical frame. A tilted one exercises a
    # different storage path entirely, and used to be BLOCKED — not by anything
    # in the laminate, but upstream:
    #
    #   TensND.KM(TensISO{4,3,Dual}, RotatedBasis{3,Float64})  →  9×9, not 6×6
    #
    # Writing a structured tensor about a non-canonical axis leaves minor-
    # antisymmetric round-off of order 1e-16, so the components landed on the
    # 81-component `Tensor` instead of the 36-component `SymmetricTensor`.
    # `TensND._store_symmetric` absorbs exactly that residue, but its tolerant
    # method was keyed on `AbstractFloat`, which `ForwardDiff.Dual` is not.
    # Fixed in TensND 0.3.6 by keying it on `ApproxType` instead — the union
    # that package already defined for precisely this distinction.
    tilted(x) = begin
        lam = Laminate(; normal = (1, 1, 1))
        add_layer!(lam, :A, Dict(:C => _isod(2.0, x)); thickness = 0.3)
        add_layer!(lam, :B, Dict(:C => _isod(0.5, 0.2)); thickness = 0.7)
        return TensND.get_data(homogenize(lam, Laminated(), :C))[5]   # 2·C₂₃₂₃
    end

    # The value: the out-of-plane shear is an exact harmonic mean, whatever the
    # frame — so it is known in closed form and pins the tilted path itself.
    @test tilted(0.8) ≈ 2 / (0.3 / 0.8 + 0.7 / 0.2) rtol = 1.0e-12
    @test ForwardDiff.derivative(tilted, 0.8) ≈ _fd(tilted, 0.8) rtol = RTOL_AD

    # …and a thickness, whose derivative also moves the volume fractions.
    tilted_h(h) = begin
        lam = Laminate(; normal = (0.3, -0.7, 0.2))
        add_layer!(lam, :A, Dict(:C => _isod(2.0, 0.8)); thickness = h)
        add_layer!(lam, :B, Dict(:C => _isod(0.5, 0.2)); thickness = 0.7)
        return TensND.get_data(homogenize(lam, Laminated(), :C))[5]
    end
    @test ForwardDiff.derivative(tilted_h, 0.3) ≈ _fd(tilted_h, 0.3) rtol = RTOL_AD
end
