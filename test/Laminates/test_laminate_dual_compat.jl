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
function _cell(; hA = 0.3, hB = 0.7, μA = 0.8, kn = 0.0, kt = 0.0, κs = 0.0, μs = 0.0)
    lam = Laminate(; normal = (0, 0, 1))
    itf = (kn == 0 && kt == 0) ? PerfectInterface() : SpringInterface(kn, kt)
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
    # A spring compliance: only the interface carries the `Dual`, while the
    # layers and thicknesses stay `Float64`.
    lam = _cell(; kn = 1.0e-2, kt = 5.0e-3)
    ad_kn = derivative(lam, Laminated(), interface_param(1, :kn); indexer = _C33)
    @test ad_kn ≈ _fd(x -> _C33(homogenize(_cell(; kn = x, kt = 5.0e-3), Laminated(), :C)), 1.0e-2) rtol = RTOL_AD
    @test ad_kn < 0                                    # a softer interface softens the stack

    ad_kt = derivative(lam, Laminated(), interface_param(1, :kt); indexer = C -> Matrix(KM(C))[4, 4])
    @test ad_kt ≈ _fd(
        x -> Matrix(KM(homogenize(_cell(; kn = 1.0e-2, kt = x), Laminated(), :C)))[4, 4], 5.0e-3
    ) rtol = RTOL_AD

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
    lam = _cell(; kn = 5.0e-2)
    ad = derivative(lam, Laminated(), thickness(:A); indexer = _C33)
    @test ad ≈ _fd(x -> _C33(homogenize(_cell(; hA = x, kn = 5.0e-2), Laminated(), :C)), 0.3) rtol = RTOL_AD

    # Scaling the whole cell changes nothing without interfaces …
    f_perf = s -> _C33(homogenize(_cell(; hA = 0.3s, hB = 0.7s), Laminated(), :C))
    @test abs(_fd(f_perf, 1.0)) < 1.0e-9
    # … but is a genuine size effect with one.
    f_spring = s -> _C33(homogenize(_cell(; hA = 0.3s, hB = 0.7s, kn = 5.0e-2), Laminated(), :C))
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
    lam = _cell(; kn = 1.0e-2)
    ps = [thickness(:A), property(:A, :C, :shear), interface_param(1, :kn)]
    # Qualified: `gradient` is exported by Tensors, Ferrite, Symbolics and
    # Zygote too, all of which are loaded by the time the full suite runs.
    g = MeanFieldHomogenization.gradient(lam, Laminated(), ps; indexer = _C33)
    @test length(g) == 3
    @test g[1] ≈ _fd(x -> _C33(homogenize(_cell(; hA = x, kn = 1.0e-2), Laminated(), :C)), 0.3) rtol = RTOL_AD
    @test g[2] ≈ _fd(x -> _C33(homogenize(_cell(; μA = x / 2, kn = 1.0e-2), Laminated(), :C)), 1.6) rtol = RTOL_AD
    @test g[3] ≈ _fd(x -> _C33(homogenize(_cell(; kn = x), Laminated(), :C)), 1.0e-2) rtol = RTOL_AD
end

@testset "AD — the lenses that must refuse" begin
    lam = _cell()
    # A laminate stores thicknesses; the fractions are derived. Silently
    # reinterpreting an `AmountParameter` would be a footgun.
    @test_throws ArgumentError get_param(lam, amount(:A))
    @test_throws ArgumentError set_param(lam, amount(:A), 0.5)
    @test_throws ArgumentError get_param(lam, thickness(:Z))
    @test_throws ArgumentError get_param(lam, interface_param(1, :nope))
end

@testset "AD — anisotropic layers, second derivative" begin
    # The kernel is smooth: `_inv3` and `_inv_km6` are rational in their
    # entries, so nested Duals go through and match a finite difference of
    # the first derivative.
    f = x -> _C33(homogenize(_cell(; μA = x), Laminated(), :C))
    d1 = x -> ForwardDiff.derivative(f, x)
    @test ForwardDiff.derivative(d1, 0.8) ≈ _fd(d1, 0.8; h = 1.0e-4) rtol = 1.0e-4
end

@testset "ForwardDiff — a TILTED laminate (blocked upstream, in TensND)" begin
    # Every other test here uses the canonical frame. A tilted one does NOT
    # currently differentiate, and the obstruction is entirely upstream: it
    # involves no laminate code at all.
    #
    #   TensND.KM(TensISO{4,3,Dual}, RotatedBasis{3,Float64})  →  9×9, not 6×6
    #
    # Writing a structured tensor about a non-canonical axis leaves minor-
    # antisymmetric round-off of order 1e-16, so `Tensors.issymmetric` says no
    # and the components land on the 81-component `Tensor` instead of the
    # 36-component `SymmetricTensor`. `TensND._store_symmetric` (`src/tens.jl`)
    # absorbs exactly that residue — but its tolerant method is written
    # `T <: AbstractFloat`, and `ForwardDiff.Dual` is not. Its own docstring
    # justifies the exact branch by "there is no round-off to absorb", which is
    # true of a symbolic or rational element type and false of a `Dual`.
    #
    # The fix belongs in TensND (`_store_symmetric` should key on `ApproxType`,
    # which that package already defines as
    # `Union{AbstractFloat, Complex{<:AbstractFloat}, ForwardDiff.Dual}`), so it
    # is recorded here rather than worked around: `@test_broken` will fail
    # loudly, and this testset can be promoted, the day it lands.
    tilted(x) = begin
        lam = Laminate(; normal = (1, 1, 1))
        add_layer!(lam, :A, Dict(:C => _isod(2.0, x)); thickness = 0.3)
        add_layer!(lam, :B, Dict(:C => _isod(0.5, 0.2)); thickness = 0.7)
        return TensND.get_data(homogenize(lam, Laminated(), :C))[5]   # 2·C₂₃₂₃
    end

    # The value itself is fine — only the derivative is blocked.
    @test tilted(0.8) ≈ 2 / (0.3 / 0.8 + 0.7 / 0.2) rtol = 1.0e-12
    @test_broken try
        isapprox(ForwardDiff.derivative(tilted, 0.8), _fd(tilted, 0.8); rtol = RTOL_AD)
    catch
        false
    end
end

