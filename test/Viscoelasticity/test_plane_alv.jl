using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Symbolics

# =============================================================================
#  Aging viscoelasticity on two-dimensional cells: the operators of the plane
#  (`VolterraOperator` on `TensISO{4,2}`, `TensISO{2,2}` and tensors of the
#  plane) and `homogenize_alv` on cells of disks and ellipses.
#
#  References, independent of the code under test:
#   1. the dense path, which knows nothing of the blocks: the full trapezoidal
#      matrix of the law, its matrix products and its block forward substitution;
#   2. the elastic limit: with constant kernels every operator is block diagonal
#      and its blocks are the elastic estimate of the plane (`homogenize`);
#   3. a kernel proportional to one scalar function, C(t, t′) = φ(t, t′) C̄, with
#      holes: the scalar operator commutes with the tensors, so the estimate is
#      φ ⊗ C̄ʰᵒᵐ, C̄ʰᵒᵐ the elastic one, and the modulus of a sheet with holes is
#      E (1 − f)/(1 + 2f) by Mori-Tanaka, E (1 − 3f) self-consistently;
#   4. the same medium in three dimensions: aligned circular cylinders of aging
#      materials, whose plane-strain block is the cell of disks of the plane
#      whose kernels are the plane-strain blocks of those of space.
# =============================================================================

const PA_TIMES = [0.0, 0.4, 1.0, 2.0, 5.0, 9.0]
const PA_N = length(PA_TIMES)
pa_decay(t, tp) = exp(-(t - tp) / 2)

# An aging isotropic kernel of the plane, and one that is orthotropic in the
# canonical frame of the plane, given as a general tensor.
pa_plane_ortho(a, b, c, d) = Tens(
    [
        (i == j == k == l == 1) * a + (i == j == k == l == 2) * b + (i == j && k == l && i != k) * c + (i != j && k != l) * d
            for i in 1:2, j in 1:2, k in 1:2, l in 1:2
    ]
)
const PA_ISO4 = ViscoLaw((t, tp) -> TensISO{2}(2 * (3 + pa_decay(t, tp)), 2 * (1 + pa_decay(t, tp)) + 0.1 * tp))
const PA_DENSE4 = ViscoLaw((t, tp) -> pa_plane_ortho(7 + pa_decay(t, tp), 5.0, 2 + 0.1 * tp, 1.5 + pa_decay(t, tp)))
const PA_ISO2 = ViscoLaw((t, tp) -> TensISO{2}(2 + pa_decay(t, tp) + 0.05 * tp))
const PA_DENSE2 = ViscoLaw((t, tp) -> Tens([3 + pa_decay(t, tp) 0.4; 0.4 2.0 + 0.1 * tp]))

@testset "VolterraOperator of the plane: the algebra against the dense matrices" begin
    for (name, law, m, class) in (
            ("iso order 4", PA_ISO4, 3, TensISO{4, 2}), ("dense order 4", PA_DENSE4, 3, MeanFieldHomogenization.Viscoelasticity.DenseClass{4, 2}),
            ("iso order 2", PA_ISO2, 2, TensISO{2, 2}), ("dense order 2", PA_DENSE2, 2, MeanFieldHomogenization.Viscoelasticity.DenseClass{2, 2}),
        )
        @testset "$name" begin
            R = VolterraOperator(law, PA_TIMES)
            M = trapezoidal_matrix(law, PA_TIMES)
            @test R.class isa class
            @test size(M) == (m * PA_N, m * PA_N)
            @test size(R) == size(M)
            @test Matrix(R) ≈ M rtol = 1.0e-14
            @test Matrix(inv(R)) ≈ volterra_inverse(M; block_size = m) rtol = 1.0e-12
            @test Matrix(R * R) ≈ M * M rtol = 1.0e-13
            @test Matrix(R + 2 * R) ≈ 3 * M rtol = 1.0e-14
            # the identity of the order and of the plane
            @test Matrix(inv(R) * R) ≈ Matrix(I, m * PA_N, m * PA_N) atol = 1.0e-12
            # the response to a history, against the full matrix times the stacked values
            h = TimeHistory(t -> m == 3 ? Tens([1 + t 0.3; 0.3 -t]) : Tens([1 + t, -t]), PA_TIMES)
            @test size(Matrix(h)) == (m, PA_N)
            @test vec(Matrix(R * h)) ≈ M * vec(Matrix(h)) rtol = 1.0e-13
            # the full matrix back into its class
            like = m == 3 ? TensISO{2}(1.0, 1.0) : TensISO{2}(1.0)
            if R.class isa TensISO
                @test VolterraOperator(M, PA_TIMES; like) ≈ R rtol = 1.0e-14
            else
                @test_throws ArgumentError VolterraOperator(M, PA_TIMES; like)
                @test VolterraOperator(M, PA_TIMES; like = law(1.0, 0.0)).class isa class
            end
        end
    end
    # isotropic and dense operators combine, in the dense class of the plane
    A, B = VolterraOperator(PA_ISO4, PA_TIMES), VolterraOperator(PA_DENSE4, PA_TIMES)
    @test (A * B).class isa MeanFieldHomogenization.Viscoelasticity.DenseClass{4, 2}
    @test Matrix(A * B) ≈ Matrix(A) * Matrix(B) rtol = 1.0e-13
    @test Matrix(B - A) ≈ Matrix(B) - Matrix(A) rtol = 1.0e-14
    A₂, B₂ = VolterraOperator(PA_ISO2, PA_TIMES), VolterraOperator(PA_DENSE2, PA_TIMES)
    @test Matrix(A₂ + B₂) ≈ Matrix(A₂) + Matrix(B₂) rtol = 1.0e-14
    # a constant tensor of the plane: its value on the diagonal
    C = TensISO{2}(6.0, 4.0)
    K = VolterraOperator(C, PA_TIMES)
    @test K.class isa TensISO{4, 2}
    @test operator_value(K, 4, 4) == C
    @test iszero(operator_value(K, 4, 2))
    # the plane and the space do not mix, nor do the sizes of the plane alone
    @test_throws ArgumentError A + VolterraOperator(TensISO{3}(3.0, 2.0), PA_TIMES)
    @test_throws DimensionMismatch VolterraOperator(Matrix(A), PA_TIMES; like = TensISO{2}(1.0))
end

@testset "VolterraOperator of the plane: isotropic part" begin
    # the isotropic part of a constant orthotropic tensor of the plane is its
    # average over the orientations of the plane, as the elastic cells take it
    C = pa_plane_ortho(7.0, 5.0, 2.0, 1.5)
    iso = MeanFieldHomogenization.Viscoelasticity._iso_part(VolterraOperator(C, PA_TIMES))
    @test iso.class isa TensISO{4, 2}
    ref = MeanFieldHomogenization.Core.isotropify(C)
    @test canonical_components(operator_value(iso, 2, 2)) ≈ canonical_components(ref) rtol = 1.0e-14
    iso2 = MeanFieldHomogenization.Viscoelasticity._iso_part2(VolterraOperator(Tens([3.0 0.4; 0.4 2.0]), PA_TIMES))
    @test only(TensND.get_data(operator_value(iso2, 2, 2))) ≈ 2.5
end

@testset "VolterraOperator of the plane: symbolic moduli" begin
    # the moduli whose Volterra inverses make the Hill kernel of the plane: the
    # longitudinal one λ + 2μ = (α + β)/2 and the shear one β/2, here constant
    Symbolics.@variables α β
    R = VolterraOperator(TensISO{2}(α, β), [0.0, 1.0])
    J_long, J_shear = MeanFieldHomogenization.Viscoelasticity._alv_iso_inverses(R)
    values = Dict(α => 3.0, β => 1.5)
    num(x) = Symbolics.value(Symbolics.substitute(x, values; fold = Val(true)))
    @test num(J_long[2, 2]) ≈ 2 / (3.0 + 1.5)
    @test num(J_shear[2, 2]) ≈ 2 / 1.5
    @test iszero(num(J_long[2, 1]))
    @test num(only(TensND.get_data(operator_value(inv(R), 2, 2))[1:1])) ≈ 1 / 3.0
end

# ── Homogenization of cells of the plane ─────────────────────────────────────

# Plane-strain stiffnesses: the matrix, the inclusions.
const PA_C0 = plane_strain_stiffness(iso_stiffness_young_poisson(70.0, 0.25))
const PA_C1 = plane_strain_stiffness(iso_stiffness_young_poisson(200.0, 0.3))
const PA_DISK = Ellipsoid(1.0; dim = 2)

function pa_cell(c₀, c₁, f; inclusion = PA_DISK, kw...)
    rve = RVE(; distribution_shape = PA_DISK)
    add_phase!(rve, :matrix, PA_DISK, Dict(:C => c₀); fraction = :rest)
    add_phase!(rve, :inclusions, inclusion, Dict(:C => c₁); fraction = f, kw...)
    return rve
end

# The Kelvin-Mandel matrix of a tensor of the plane, and the operator of the
# constant kernel equal to it: that matrix on the diagonal of the time blocks.
pa_km(C) = mandel(canonical_components(C); symmetrize = true)
pa_constant(C) = kron(I(PA_N), pa_km(C))

const PA_SCHEMES = (Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(), PonteCastanedaWillis(), SelfConsistent())

@testset "homogenize_alv on a cell of the plane: the elastic limit" begin
    f = 0.2
    for scheme in PA_SCHEMES
        R = homogenize_alv(pa_cell(PA_C0, PA_C1, f), scheme, :C; times = PA_TIMES)
        E = homogenize(pa_cell(PA_C0, PA_C1, f), scheme, :C)
        # block diagonal, each block the elastic estimate
        @test R.class isa TensISO{4, 2}
        @test Matrix(R) ≈ pa_constant(E) rtol = 1.0e-9
    end
    # aligned ellipses, and ellipses at random in the plane
    ellipse = Ellipsoid(3.0, 1.0)
    for (scheme, kw) in ((MoriTanaka(), (;)), (Dilute(), (;)), (MoriTanaka(), (; symmetrize = IsoSymmetrize())), (SelfConsistent(), (; symmetrize = IsoSymmetrize())))
        R = homogenize_alv(pa_cell(PA_C0, PA_C1, f; inclusion = ellipse, kw...), scheme, :C; times = PA_TIMES)
        E = homogenize(pa_cell(PA_C0, PA_C1, f; inclusion = ellipse, kw...), scheme, :C)
        @test Matrix(R) ≈ pa_constant(E) rtol = 1.0e-9
    end
    # the plane has no axis to average about
    @test_throws ArgumentError homogenize_alv(pa_cell(PA_C0, PA_C1, f; inclusion = ellipse, symmetrize = TISymmetrize((0.0, 0.0, 1.0))), MoriTanaka(), :C; times = PA_TIMES)
    # the Hill kernel is known against an isotropic matrix only
    @test_throws ArgumentError homogenize_alv(pa_cell(PA_DENSE4, PA_C1, f), MoriTanaka(), :C; times = PA_TIMES)
    # the schemes written on full matrices of space are refused
    for scheme in (DifferentialScheme(), AsymmetricSelfConsistent(), SelfConsistent(algorithm = NewtonDefault()))
        @test_throws ArgumentError homogenize_alv(pa_cell(PA_C0, PA_C1, f), scheme, :C; times = PA_TIMES)
    end
end

# A kernel proportional to φ(t, t′): an aging Maxwell-like relaxation function.
pa_φ(t, tp) = exp(-(t - tp) / (1 + tp))
# Young's modulus of a stiffness of the plane read as a sheet: 4kμ/(k + μ), k = α/2, μ = β/2.
pa_sheet_modulus(C::TensISO{4, 2}) = ((α, β) = TensND.get_data(C); k = α / 2; μ = β / 2; 4k * μ / (k + μ))

@testset "homogenize_alv on a cell of the plane: holes in a proportional kernel" begin
    f = 0.1
    law = ViscoLaw((t, tp) -> pa_φ(t, tp) * PA_C0)
    Φ = trapezoidal_matrix(ViscoLaw(pa_φ), PA_TIMES)
    E₀ = pa_sheet_modulus(PA_C0)
    for (scheme, ratio) in ((MoriTanaka(), (1 - f) / (1 + 2f)), (SelfConsistent(reltol = 1.0e-13), 1 - 3f))
        R = homogenize_alv(pa_cell(law, 0 * PA_C0, f), scheme, :C; times = PA_TIMES)
        E = homogenize(pa_cell(PA_C0, 0 * PA_C0, f), scheme, :C)
        @test Matrix(R) ≈ kron(Φ, pa_km(E)) rtol = 1.0e-8
        for i in 1:PA_N
            @test pa_sheet_modulus(operator_value(R, i, i)) ≈ Φ[i, i] * E₀ * ratio rtol = 1.0e-6
        end
    end
    # with respect to the fraction of holes: Φᵢᵢ E₀ d/df[(1 − f)/(1 + 2f)] = −3 Φᵢᵢ E₀/(1 + 2f)²
    sheet(x) = pa_sheet_modulus(operator_value(homogenize_alv(pa_cell(law, 0 * PA_C0, x), MoriTanaka(), :C; times = PA_TIMES), PA_N, PA_N))
    @test ForwardDiff.derivative(sheet, f) ≈ -3 * Φ[end, end] * E₀ / (1 + 2f)^2 rtol = 1.0e-10
end

# The plane-strain block of a stiffness of space: its components in the plane
# (2, 3), the cylinders having the axis e₁.
pa_plane_block(C) = canonical_components(C)[2:3, 2:3, 2:3, 2:3]
# Every value of the operator of the plane `R₂` against the plane-strain block of
# that of the operator of space `R₃`, read in the transversely isotropic class
# about e₁ (its values are then tensors).
const PA_TI_E1 = TensTI{4}(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, (1.0, 0.0, 0.0))
pa_is_plane_block(R₂, R₃) = (
    R₃ = convert_class(R₃, PA_TI_E1); all(
        isapprox(canonical_components(operator_value(R₂, i, j)), pa_plane_block(operator_value(R₃, i, j)); rtol = 1.0e-9, atol = 1.0e-9)
            for i in 1:PA_N for j in 1:i
    )
)

@testset "homogenize_alv on a cell of the plane: the same medium in three dimensions" begin
    f = 0.25
    k(t, tp) = 40 + 20 * pa_decay(t, tp)
    μ(t, tp) = (20 + 10 * pa_φ(t, tp)) * (1 + 0.02 * tp)
    space_law = ViscoLaw((t, tp) -> TensISO{3}(3 * k(t, tp), 2 * μ(t, tp)))
    # the plane-strain block of 3k 𝕁 + 2μ 𝕂: λ + μ = k + μ/3
    plane_law = ViscoLaw((t, tp) -> TensISO{2}(2 * (k(t, tp) + μ(t, tp) / 3), 2 * μ(t, tp)))
    C₁ = iso_stiffness_young_poisson(200.0, 0.3)
    space = RVE(; distribution_shape = Cylinder(1.0, 1.0))
    add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:C => space_law); fraction = :rest)
    add_phase!(space, :inclusions, Cylinder(1.0, 1.0), Dict(:C => C₁); fraction = f)
    plane = pa_cell(plane_law, plane_strain_stiffness(C₁), f)
    for scheme in (Dilute(), MoriTanaka(), Maxwell())
        R₃ = homogenize_alv(space, scheme, :C; times = PA_TIMES)
        R₂ = homogenize_alv(plane, scheme, :C; times = PA_TIMES)
        @test R₂.class isa TensISO{4, 2}
        @test pa_is_plane_block(R₂, R₃)
    end
end

# ── Cracks of the plane ──────────────────────────────────────────────────────

# The normal n and the direction m of a segment, in the axes of the plane.
pa_normal(crack) = MeanFieldHomogenization.Cracks._plane_normal(crack)
pa_direction(crack) = MeanFieldHomogenization.Cracks._plane_direction(crack)

@testset "cracks of the plane in aging viscoelasticity" begin
    crack = RibbonCrack(1.0; angle = 0.3)
    # the elastic limit: the contribution of the segment on the diagonal, faces free or held
    H = inverse_property_contribution_alv(crack, PA_C0, PA_TIMES)
    @test Matrix(H) ≈ pa_constant(inverse_property_contribution(crack, PA_C0)) rtol = 1.0e-12
    kₙ, kₜ = 30.0, 12.0
    n, m = pa_normal(crack), pa_direction(crack)
    Kᵢ = Tens(kₙ * n * transpose(n) + kₜ * m * transpose(m))
    Hᵢ = inverse_property_contribution_alv(crack, PA_C0, PA_TIMES; Rn = heaviside_law(kₙ), Rt = heaviside_law(kₜ))
    @test Matrix(Hᵢ) ≈ pa_constant(inverse_property_contribution(crack, PA_C0; K_interface = Kᵢ)) rtol = 1.0e-12
    @test Matrix(property_contribution_alv(crack, PA_C0, PA_TIMES)) ≈
        pa_constant(property_contribution(crack, PA_C0)) rtol = 1.0e-12
    # a crack of space is not one of the plane
    @test_throws ArgumentError inverse_property_contribution_alv(EllipticCrack(1.0, 0.5), PA_C0, PA_TIMES)

    # cracked sheets, in the elastic limit
    ρ = 0.1
    function cracked(c₀; kw...)
        rve = RVE(; distribution_shape = PA_DISK)
        add_phase!(rve, :matrix, PA_DISK, Dict(:C => c₀); fraction = :rest)
        add_phase!(rve, :cracks, crack, Dict(:C => 0 * PA_C0); density = ρ, kw...)
        return rve
    end
    for (scheme, kw) in ((Dilute(), (;)), (MoriTanaka(), (;)), (DiluteDual(), (;)), (MoriTanaka(), (; symmetrize = IsoSymmetrize())), (SelfConsistent(), (; symmetrize = IsoSymmetrize())))
        R = homogenize_alv(cracked(PA_C0; kw...), scheme, :C; times = PA_TIMES)
        E = homogenize(cracked(PA_C0; kw...), scheme, :C)
        @test Matrix(R) ≈ pa_constant(E) rtol = 1.0e-9
    end

    # the flat limit of a void ellipse in an aging matrix: f = π ρ η of ellipses of
    # semi-axes 1 and η along the segment and its normal against ρ of segments
    law = ViscoLaw((t, tp) -> TensISO{2}(2 * (40 + 20 * pa_decay(t, tp)), 2 * (20 + 10 * pa_φ(t, tp)) * (1 + 0.02 * tp)))
    segment = RibbonCrack(1.0; angle = 0.0)
    η = 1.0e-6
    holes = pa_cell(law, 0 * PA_C0, π * ρ * η; inclusion = Ellipsoid(1.0, η))
    segments = RVE(; distribution_shape = PA_DISK)
    add_phase!(segments, :matrix, PA_DISK, Dict(:C => law); fraction = :rest)
    add_phase!(segments, :cracks, segment, Dict(:C => 0 * PA_C0); density = ρ)
    C₀ = VolterraOperator(law, PA_TIMES)
    ΔC_holes = homogenize_alv(holes, Dilute(), :C; times = PA_TIMES) - C₀
    ΔC_segments = homogenize_alv(segments, Dilute(), :C; times = PA_TIMES) - C₀
    @test Matrix(ΔC_holes) ≈ Matrix(ΔC_segments) rtol = 1.0e-5
end

# ── Coated disks: a LayeredCylinder in a cell of the plane ──────────────────

@testset "concentric rings in aging viscoelasticity" begin
    times = PA_TIMES
    coat = plane_strain_stiffness(iso_stiffness_young_poisson(20.0, 0.35))
    # the aging matrix of the plane, and its kernel in space whose plane-strain block it is
    k(t, tp) = 40 + 20 * pa_decay(t, tp)
    μ(t, tp) = (20 + 10 * pa_φ(t, tp)) * (1 + 0.02 * tp)
    plane_law = ViscoLaw((t, tp) -> TensISO{2}(2 * (k(t, tp) + μ(t, tp) / 3), 2 * μ(t, tp)))
    space_law = ViscoLaw((t, tp) -> TensISO{3}(3 * k(t, tp), 2 * μ(t, tp)))
    for interfaces in ((PerfectInterface(), PerfectInterface()), (PerfectInterface(), SpringInterface(50.0, 30.0)))
        rings = LayeredCylinder((0.8, 1.0), (PA_C1, coat); interfaces)
        # the elastic limit
        ref = heaviside_law(PA_C0)
        @test Matrix(gradient_gradient_concentration_alv(rings, ref, times)) ≈
            pa_constant(gradient_gradient_concentration(rings, PA_C1, PA_C0)) rtol = 1.0e-10
        @test Matrix(property_contribution_alv(rings, ref, times)) ≈
            pa_constant(property_contribution(rings, PA_C1, PA_C0)) rtol = 1.0e-10
        @test Matrix(flux_gradient_concentration_alv(rings, ref, times; layer = 2)) ≈
            pa_constant(flux_gradient_concentration(rings, PA_C0; layer = 2)) rtol = 1.0e-10
        # against the cylinder of axis e₁ in the aging matrix of space
        C₁, C₂ = iso_from_plane_strain.((PA_C1, coat))
        cylinder = LayeredCylinder((0.8, 1.0), (C₁, C₂); interfaces, axis = (1.0, 0.0, 0.0))
        for f in (gradient_gradient_concentration_alv, property_contribution_alv)
            R₂ = f(rings, plane_law, times)
            R₃ = VolterraOperator(f(cylinder, space_law, times), times)
            @test pa_is_plane_block(R₂, R₃)
        end
        # in a cell, by the schemes on the matrix
        plane = pa_cell(plane_law, PA_C1, 0.3; inclusion = rings)
        space = RVE(; distribution_shape = Cylinder(1.0, 1.0))
        add_phase!(space, :matrix, Cylinder(1.0, 1.0), Dict(:C => space_law); fraction = :rest)
        add_phase!(space, :inclusions, cylinder, Dict(:C => C₁); fraction = 0.3)
        for scheme in (Dilute(), MoriTanaka(), Voigt())
            R₂ = homogenize_alv(plane, scheme, :C; times)
            R₃ = homogenize_alv(space, scheme, :C; times)
            @test pa_is_plane_block(R₂, R₃)
        end
        # the inverse of a stiffness of space is not that of its plane-strain block:
        # the Reuss bound against the elastic one of the plane
        constant = pa_cell(PA_C0, PA_C1, 0.3; inclusion = rings)
        @test Matrix(homogenize_alv(constant, Reuss(), :C; times)) ≈ pa_constant(homogenize(constant, Reuss(), :C)) rtol = 1.0e-10
    end
    # the self-consistent iteration with rings runs on full matrices of space: refused
    rings = LayeredCylinder((0.8, 1.0), (PA_C1, coat))
    @test_throws ArgumentError homogenize_alv(pa_cell(heaviside_law(PA_C0), PA_C1, 0.3; inclusion = rings), SelfConsistent(), :C; times)
    alone = RVE()
    add_phase!(alone, :matrix, LayeredCylinder((1.0,), (PA_C0,)), Dict(:C => PA_C0); fraction = :rest)
    add_phase!(alone, :inclusions, rings, Dict(:C => PA_C1); fraction = 0.3)
    @test_throws ArgumentError homogenize_alv(alone, SelfConsistent(), :C; times)
end
