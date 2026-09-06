# =============================================================================
#  test_laminate_cell.jl — the `Laminate` container itself.
#
#  Coverage:
#   1. Two-step construction, accessors, stacking order.
#   2. `thickness` xor `fraction`, and the period they imply.
#   3. Frame construction from a normal / Euler angles / an explicit basis.
#   4. `validate_laminate` failure modes.
#   5. Element-type promotion (a `Dual` thickness in a plain `Laminate()`).
#   6. The `AbstractHomogenizationCell` contract hooks.
#   7. `show`.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy

const _ISO_L(k, μ) = TensISO{3}(3k, 2μ)

@testset "Laminate — construction and accessors" begin
    lam = Laminate(; normal = (0, 0, 1))
    @test layer_count(lam) == 0
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 0.3)
    add_layer!(lam, :B, Dict(:C => _ISO_L(0.5, 0.2)); thickness = 0.7)

    @test layer_names(lam) == [:A, :B]          # stacking order is insertion order
    @test layer_count(lam) == 2
    @test layer_thickness(lam, :A) == 0.3
    @test laminate_period(lam) ≈ 1.0
    @test layer_volume_fraction(lam, :A) ≈ 0.3
    @test layer_volume_fraction(lam, :B) ≈ 0.7
    @test layer_property(lam, :A, :C) == _ISO_L(2.0, 0.8)
    @test laminate_normal(lam) == (0.0, 0.0, 1.0)
    @test layer_interface(lam, 1) isa PerfectInterface
    @test length(lam.interfaces) == layer_count(lam)   # one per layer, periodic

    @test_throws ArgumentError add_layer!(lam, :A, Dict(:C => _ISO_L(1.0, 1.0)); thickness = 1.0)
    @test_throws ArgumentError layer_property(lam, :Z, :C)
    @test_throws ArgumentError layer_property(lam, :A, :nope)
    @test_throws ArgumentError layer_thickness(lam, :Z)
end

@testset "Laminate — thickness xor fraction, and the period" begin
    # `fraction` is read against the reference period (1 by default), so a
    # fraction-specified stack has L = 1 …
    lam = Laminate()
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); fraction = 0.25)
    add_layer!(lam, :B, Dict(:C => _ISO_L(0.5, 0.2)); fraction = 0.75)
    @test laminate_period(lam) ≈ 1.0
    @test layer_volume_fraction(lam, :A) ≈ 0.25

    # … while thicknesses set an absolute period, which is what carries the
    # interface size effect.
    lam2 = Laminate()
    add_layer!(lam2, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 0.5)
    add_layer!(lam2, :B, Dict(:C => _ISO_L(0.5, 0.2)); thickness = 1.5)
    @test laminate_period(lam2) ≈ 2.0
    @test layer_volume_fraction(lam2, :A) ≈ 0.25          # same fractions as `lam`

    lam3 = Laminate()
    @test_throws ArgumentError add_layer!(lam3, :A, Dict(:C => _ISO_L(1.0, 1.0)))
    @test_throws ArgumentError add_layer!(
        lam3, :A, Dict(:C => _ISO_L(1.0, 1.0)); thickness = 0.5, fraction = 0.5
    )
    @test_throws ArgumentError add_layer!(
        lam3, :A, Dict(:C => _ISO_L(1.0, 1.0)); thickness = -0.5
    )
end

@testset "Laminate — the frame" begin
    # n = e₃ keeps the canonical basis (and lets the kernel skip the rotation)
    @test laminate_basis(Laminate()) isa TensND.CanonicalBasis
    @test laminate_basis(Laminate(; normal = (0, 0, 1))) isa TensND.CanonicalBasis
    @test laminate_basis(Laminate(; normal = (0, 0, 5))) isa TensND.CanonicalBasis

    for n in ((1, 0, 0), (0, 1, 0), (1, 1, 1), (0.3, -0.7, 0.2))
        lam = Laminate(; normal = n)
        nn = laminate_normal(lam)
        nrm = sqrt(sum(x -> x^2, n))
        @test all(isapprox.(nn, n ./ nrm; atol = 1.0e-12))
        # the frame is orthonormal and right-handed
        R = MeanFieldHomogenization.Core._basis_matrix(laminate_basis(lam))
        @test R' * R ≈ I atol = 1.0e-12
        @test det(R) ≈ 1 atol = 1.0e-12
    end

    @test laminate_basis(Laminate(; euler_angles = (0.3, 0.4, 0.1))) isa TensND.RotatedBasis
    b = TensND.RotatedBasis(0.2, 0.5, 0.1)
    @test laminate_basis(Laminate(; basis = b)) === b

    @test_throws ArgumentError Laminate(; normal = (0, 0, 1), euler_angles = (0.1,))
    @test_throws ArgumentError Laminate(; normal = (0, 0, 0))
    @test_throws ArgumentError Laminate(; normal = (0, 1))
    # `in_plane` only makes sense with `normal`: the other two routes already
    # fix the whole frame, so accepting it there would silently ignore it.
    @test_throws ArgumentError Laminate(; euler_angles = (0.3,), in_plane = (1, 0, 0))
    @test_throws ArgumentError Laminate(; in_plane = (1, 0, 0))
    @test_throws ArgumentError Laminate(; normal = (1, 1, 1), in_plane = (1, 0))
end

@testset "Laminate — `in_plane` fixes the in-plane axis" begin
    # The physics never sees this choice (the result is invariant under rotation
    # about `n`), but the STORED frame does, and an anisotropic interface given
    # as a plain matrix is read in it — so it has to be honored exactly.
    lam = Laminate(; normal = (0, 0, 1), in_plane = (1, 1, 0))
    R = MeanFieldHomogenization.Core._basis_matrix(laminate_basis(lam))
    @test R[:, 1] ≈ [1, 1, 0] ./ sqrt(2) atol = 1.0e-12
    @test R[:, 3] ≈ [0, 0, 1] atol = 1.0e-12
    @test R' * R ≈ I atol = 1.0e-12
    @test det(R) ≈ 1 atol = 1.0e-12

    # Asking for the canonical normal WITH a reference axis is a genuine
    # request for a rotated frame, so it must not short-circuit to canonical.
    @test !(laminate_basis(lam) isa TensND.CanonicalBasis)
    @test laminate_basis(Laminate(; normal = (0, 0, 1))) isa TensND.CanonicalBasis
end

@testset "Laminate — the canonical frame follows the declared element type" begin
    # `T` is a floor for the thicknesses AND the element type of a canonical
    # frame. The two ways of asking for `n = e₃` must agree, or the axis of the
    # returned `TensTI` differs between them — which is exactly how a `Float64`
    # `1.0` used to end up multiplying every coefficient of a symbolic result.
    @test eltype(laminate_basis(Laminate())) === Float64
    @test eltype(laminate_basis(Laminate(; normal = (0, 0, 1)))) === Float64
    @test eltype(laminate_basis(Laminate(; T = BigFloat))) === Float64   # Real ⇒ Float64
    @test typeof(laminate_basis(Laminate(; T = Sym, normal = (0, 0, 1)))) ===
        typeof(laminate_basis(Laminate(; T = Sym)))
    @test eltype(laminate_basis(Laminate(; T = Sym))) <: Sym
end

@testset "Laminate — a symbolic frame" begin
    θ = symbols("theta", real = true)

    # From a symbolic normal. Gram-Schmidt against `e₁` by default: purely
    # algebraic, so the frame stays as readable as the normal it came from.
    lam = Laminate(; normal = (0, sin(θ), cos(θ)))
    b = laminate_basis(lam)
    @test b isa TensND.RotatedBasis
    @test eltype(b) <: Sym
    R = MeanFieldHomogenization.Core._frame_matrix(b)
    @test all(iszero, simplify.(R' * R - I))          # orthonormal
    @test iszero(simplify(det(R) - 1))                # right-handed
    @test all(iszero, simplify.(R[:, 3] - [0, sin(θ), cos(θ)]))   # 3rd axis is n̂
    @test all(iszero, simplify.(collect(laminate_normal(lam)) - [0, sin(θ), cos(θ)]))

    # An explicit reference, which is what a normal along e₁ needs.
    lamP = Laminate(; normal = (cos(θ), 0, sin(θ)), in_plane = (0, 1, 0))
    RP = MeanFieldHomogenization.Core._frame_matrix(laminate_basis(lamP))
    @test all(iszero, simplify.(RP' * RP - I))
    @test all(iszero, simplify.(RP[:, 1] - [0, 1, 0]))

    # Symbolic ZYZ angles, the other route.
    lamE = Laminate(; euler_angles = (θ, 0, 0))
    @test laminate_basis(lamE) isa TensND.RotatedBasis
    @test eltype(laminate_basis(lamE)) <: Sym
    @test all(iszero, simplify.(collect(laminate_normal(lamE)) - [sin(θ), 0, cos(θ)]))

    # `_basis_matrix` pins to Float64 and would throw here; `_frame_matrix` is
    # the accessor the laminate paths must use.
    @test_throws Exception MeanFieldHomogenization.Core._basis_matrix(b)

    # `show` has neither a `float` nor a `round` to apply to a symbolic axis.
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); fraction = 0.4)
    add_layer!(lam, :B, Dict(:C => _ISO_L(0.5, 0.2)); fraction = 0.6)
    str = sprint(show, MIME"text/plain"(), lam)
    @test occursin("theta", str)
    @test occursin("2 layer(s)", str)
end

@testset "Laminate — validation" begin
    @test_throws ArgumentError validate_laminate(Laminate())
    @test_throws ArgumentError homogenize(Laminate(), Laminated(), :C)

    lam = Laminate()
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 0.4)
    @test validate_laminate(lam) === lam
    @test validate_cell(lam) === lam

    # A scheme that needs a matrix must say so rather than dispatch elsewhere.
    @test_throws ErrorException homogenize(lam, MoriTanaka(), :C)
    # ... and symmetrically, `Laminated` on an RVE.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _ISO_L(2.0, 0.8)); fraction = :rest)
    @test_throws ErrorException homogenize(rve, Laminated(), :C)
end

@testset "Laminate — element-type promotion" begin
    lam = Laminate()                                   # declared floor Float64
    @test eltype(typeof(lam)) === Float64
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 1)   # Int
    @test layer_thickness(lam, :A) isa Float64
    @test eltype(lam) === Float64

    # A `Dual` thickness lives in a plain `Laminate()` — the floor is a floor,
    # not a constraint, exactly as for an RVE amount.
    d = ForwardDiff.Dual{:tag}(0.3, 1.0)
    lam2 = Laminate()
    add_layer!(lam2, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = d)
    add_layer!(lam2, :B, Dict(:C => _ISO_L(0.5, 0.2)); thickness = 0.7)
    @test layer_thickness(lam2, :A) isa ForwardDiff.Dual
    @test eltype(lam2) <: ForwardDiff.Dual
    @test laminate_period(lam2) isa ForwardDiff.Dual
end

@testset "Laminate — the cell contract" begin
    lam = Laminate()
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 0.3)
    add_layer!(lam, :B, Dict(:C => _ISO_L(0.5, 0.2)); thickness = 0.7)

    @test Laminate <: MeanFieldHomogenization.Core.AbstractHomogenizationCell
    @test MeanFieldHomogenization.Core.cell_member_names(lam) == [:A, :B]
    @test MeanFieldHomogenization.Core.cell_container_property(lam, :A, :C) == _ISO_L(2.0, 0.8)

    lam2 = MeanFieldHomogenization.Core.cell_set_property(lam, :A, :C, _ISO_L(9.0, 3.0))
    @test layer_property(lam2, :A, :C) == _ISO_L(9.0, 3.0)
    @test layer_property(lam, :A, :C) == _ISO_L(2.0, 0.8)      # no mutation
    @test layer_thickness(lam2, :A) == 0.3
    @test laminate_period(lam2) ≈ laminate_period(lam)
end

@testset "Laminate — show" begin
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => _ISO_L(2.0, 0.8)); thickness = 0.3)
    add_layer!(
        lam, :B, Dict(:C => _ISO_L(0.5, 0.2)); thickness = 0.7,
        interface = SpringInterface(; sn = 1.0e-3, st = 2.0e-3)
    )
    s = sprint(show, MIME"text/plain"(), lam)
    @test occursin("Laminate{Float64}", s)
    @test occursin("2 layer(s)", s)
    @test occursin(":A", s) && occursin(":B", s)
    @test occursin("SpringInterface", s)
    @test occursin("Laminate{Float64}", sprint(show, lam))
end
