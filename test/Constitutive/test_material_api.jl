using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const CS_MAT = TensISO{3}(3 * 30.0, 2 * 18.0)
const CANON_MAT = TensND.CanonicalBasis{3, Float64}()
# Canonical-frame components — see the frame rule in `Constitutive`.
_g(t::TensND.AbstractTens) = get_array(TensND.change_tens(t, CANON_MAT))

function _composite_rve(; f = 0.2)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_MAT))
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 90.0, 2 * 60.0));
        fraction = f
    )
    return rve
end

# A microstructure with a TILTED crack family, so the homogenized stiffness is
# returned in a rotated basis. This is what catches a material that hands raw
# components to an FE code.
function _tilted_rve(; d = 0.08)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => CS_MAT))
    add_phase!(
        rve, :F, PennyCrack(1.0; euler_angles = (π / 4, 0.0)), Dict(:C => CS_MAT);
        density = d
    )
    return rve
end

@testset "HomogenizedElastic reproduces the scheme" begin
    rve = _composite_rve()
    for scheme in (MoriTanaka(), Dilute(), SelfConsistent())
        mat = HomogenizedElastic(rve, scheme)
        @test _g(stiffness(mat)) ≈ _g(homogenize(rve, scheme)) rtol = 1.0e-12

        ε = Tens(TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> 1.0e-3 * (i == j ? i : 0.0)))
        r = material_response(mat, ε, initial_state(mat), 0.0)
        @test _g(stress(r)) ≈ _g(homogenize(rve, scheme) ⊡ ε) rtol = 1.0e-12
        @test state(r) isa NoState
    end

    # The direct-stiffness constructor is the same material.
    C = homogenize(rve, MoriTanaka())
    @test _g(stiffness(HomogenizedElastic(C))) ≈ _g(C) rtol = 1.0e-14
end

@testset "the material speaks the GLOBAL frame" begin
    # The whole point of the boundary: a scheme fed tilted cracks returns its
    # estimate in a rotated basis, and an FE code reads components as global.
    rve = _tilted_rve()
    C_raw = homogenize(rve, MoriTanaka())
    @test !(TensND.get_basis(C_raw) isa TensND.CanonicalBasis)   # premise of the test

    mat = HomogenizedElastic(rve, MoriTanaka())
    @test TensND.get_basis(stiffness(mat)) isa TensND.CanonicalBasis
    @test _g(stiffness(mat)) ≈ _g(C_raw) rtol = 1.0e-12

    # …and the raw components genuinely differ, so the conversion is not a no-op.
    @test !isapprox(get_array(C_raw), _g(C_raw); atol = 1.0e-8)
end

@testset "check_material_interface" begin
    for rve in (_composite_rve(), _tilted_rve())
        @test check_material_interface(
            HomogenizedElastic(rve, MoriTanaka()); verbose = false
        )
    end
end

@testset "Tensors.jl bridge" begin
    T2 = TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> 1.0 * i + 2.0 * j)
    @test to_tensors(from_tensors(T2)) ≈ T2

    C = homogenize(_tilted_rve(), MoriTanaka())
    CT = to_tensors(C)
    @test CT isa TensND.Tensors.SymmetricTensor{4, 3}
    # `to_tensors` must give the GLOBAL components, not the own-basis ones.
    @test [CT[i, j, k, l] for i in 1:3, j in 1:3, k in 1:3, l in 1:3] ≈ _g(C) atol = 1.0e-12

    # Voigt round trips, with the engineering-shear factor kept straight.
    ε = Tens(TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> 1.0e-3 * (i + j)))
    v = voigt_strain(ε)
    @test v[4] ≈ 2 * ε[1, 2]
    @test _g(strain_from_voigt(v)) ≈ _g(ε) atol = 1.0e-14
    σ = Tens(TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> 1.0 * i * j))
    w = voigt_stress(σ)
    @test w[4] ≈ σ[1, 2]
    @test _g(stress_from_voigt(w)) ≈ _g(σ) atol = 1.0e-14
end

@testset "MaterialCache" begin
    c = MaterialCache()
    calls = Ref(0)
    f = () -> (calls[] += 1; 42)
    @test cached!(f, c, (:a, 1)) == 42
    @test cached!(f, c, (:a, 1)) == 42          # hit
    @test cached!(f, c, (:a, 2)) == 42          # miss
    @test calls[] == 2
    st = cache_stats(c)
    @test st.hits == 1 && st.misses == 2 && st.entries == 2

    reset_cache!(c)
    @test cache_stats(c) == (hits = 0, misses = 0, entries = 0)

    # `nothing` disables memoization without any branching at the call site.
    calls[] = 0
    cached!(f, nothing, (:a, 1)); cached!(f, nothing, (:a, 1))
    @test calls[] == 2
    @test cache_stats(nothing).entries == 0
end

@testset "self-description defaults" begin
    mat = HomogenizedElastic(_composite_rve(), MoriTanaka())
    @test gradient_names(mat) == (:ε,)
    @test flux_names(mat) == (:σ,)
    @test tangent_blocks(mat) == (:σε,)
    @test transport_property(mat, initial_state(mat)) === nothing
end
