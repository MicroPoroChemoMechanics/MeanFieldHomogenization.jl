using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

const C0_PO = TensISO{3}(3 * 30.0, 2 * 18.0)      # k = 30 GPa, μ = 18 GPa
const KS_PO = 1.0e-18                              # tight matrix, m²
const CF_PO = 4.0e-18                              # fracture conductivity
const CANON_PO = TensND.CanonicalBasis{3, Float64}()
_g(t) = get_array(TensND.change_tens(t, CANON_PO))
_ε33(e) = from_tensors(
    TensND.Tensors.SymmetricTensor{2, 3}((i, j) -> (i == j == 3) ? e : 0.0)
)

function _poro_rve(; d = 0.1, conductive = true)
    rve = RVE(:M)
    props = Dict(:C => C0_PO, :K => TensISO{3}(KS_PO))
    add_matrix!(rve, Ellipsoid(1.0), props)
    geom = conductive ? ConductiveCrack(1.0; conductivity = CF_PO) : PennyCrack(1.0)
    add_phase!(rve, :F, geom, props; density = d)
    return rve
end

_mat(; kw...) = FracturedPoroelasticRock(
    _poro_rve(; kw...), MoriTanaka(); ω₀ = (1.0e-3,), k_matrix = KS_PO
)

_step(mat, st, e, p, cache) =
    material_response(mat, (; ε = _ε33(e), p = p), st, 0.0; cache = cache)

@testset "self-description" begin
    mat = _mat()
    @test gradient_names(mat) == (:ε, :p)
    @test flux_names(mat) == (:σ, :φ)
    @test tangent_blocks(mat) == (:σε, :σp, :φε, :φp)
end

@testset "the four tangent blocks are the Biot parameters" begin
    mat = _mat()
    cache = MaterialCache()
    r = _step(mat, initial_state(mat), -1.0e-5, 0.0, cache)

    C_hom = homogenize(_poro_rve(), MoriTanaka())
    par = poroelastic_parameters(C_hom, C0_PO, 0.0)

    @test _g(r.tangents.σε) ≈ _g(C_hom) rtol = 1.0e-10        # ∂Σ/∂E
    @test _g(-r.tangents.σp) ≈ _g(par.B) rtol = 1.0e-10       # ∂Σ/∂p = −B
    @test _g(r.tangents.φε) ≈ _g(par.B) rtol = 1.0e-10        # ∂φ/∂E = B
    @test r.tangents.φp ≈ par.inverse_modulus rtol = 1.0e-10  # ∂φ/∂p = 1/M
end

@testset "pore pressure reopens what compression closed" begin
    # The Terzaghi statement: at FIXED strain, raising p opens the fractures,
    # because the microstructure is driven by Σ' = Σ + pδ.
    mat = _mat()
    cache = MaterialCache()
    st = state(_step(mat, initial_state(mat), -5.0e-4, 0.0, cache))
    ω_dry = apertures(st)[1]
    @test 0 < ω_dry < 1.0e-3                       # closing but still open

    ω_prev = ω_dry
    for p in (5.0e-3, 1.0e-2)
        r = _step(mat, st, -5.0e-4, p, cache)
        ω = apertures(state(r))[1]
        @test ω > ω_prev                           # pressure opens
        ω_prev = ω
    end
end

@testset "the conductivity follows the cubic aperture law" begin
    mat = _mat()
    cache = MaterialCache()
    st = initial_state(mat)
    @test conductivities(st)[1] ≈ CF_PO

    for e in (-2.0e-4, -5.0e-4)
        st = state(_step(mat, st, e, 0.0, cache))
        ω = apertures(st)[1]
        @test conductivities(st)[1] ≈ CF_PO * (ω / 1.0e-3)^3 rtol = 1.0e-12
    end
end

@testset "the permeability follows the apertures" begin
    mat = _mat()
    cache = MaterialCache()
    st = initial_state(mat)
    K_open = _g(transport_property(mat, st))[1, 1]
    @test K_open > KS_PO                            # fractures conduct

    st = state(_step(mat, st, -5.0e-4, 0.0, cache))
    K_tight = _g(transport_property(mat, st))[1, 1]
    @test KS_PO < K_tight < K_open                  # closing lowers it

    # A closed fracture carries no flow at all.
    st = state(_step(mat, st, -1.5e-3, 0.0, cache))
    @test open_set(st) == (false,)
    @test transport_property(mat, st) === nothing
end

@testset "a closed fracture leaves the intact matrix" begin
    # No pore space left: B = 0 and 1/M = 0, so the poroelastic coupling
    # switches itself off — the medium is simply the solid.
    mat = _mat()
    cache = MaterialCache()
    r = _step(mat, initial_state(mat), -1.5e-3, 0.0, cache)
    @test open_set(state(r)) == (false,)
    @test maximum(abs, _g(r.tangents.σp)) < 1.0e-12
    @test abs(r.tangents.φp) < 1.0e-12
    @test _g(r.tangents.σε) ≈ _g(C0_PO) rtol = 1.0e-10
end

@testset "an ordinary crack makes it purely poroelastic" begin
    mat = _mat(; conductive = false)
    cache = MaterialCache()
    r = _step(mat, initial_state(mat), -1.0e-4, 0.0, cache)
    @test transport_property(mat, state(r)) === nothing   # no hydraulic side
    @test r.tangents.φp > 0                               # …but still poroelastic
end

@testset "zero pressure reproduces the mechanical material" begin
    # With p ≡ 0 the poroelastic law must be the microcracked one, which is what
    # makes the extra machinery safe to trust.
    rve = _poro_rve()
    cache1, cache2 = MaterialCache(), MaterialCache()
    poro = FracturedPoroelasticRock(rve, MoriTanaka(); ω₀ = (1.0e-3,), k_matrix = KS_PO)
    mech = MicrocrackedMaterial(rve, MoriTanaka(); ω₀ = (1.0e-3,))

    sp, sm = initial_state(poro), initial_state(mech)
    for e in (-2.0e-4, -5.0e-4, -1.5e-3, 5.0e-4)
        rp = material_response(poro, (; ε = _ε33(e), p = 0.0), sp, 0.0; cache = cache1)
        rm = material_response(mech, _ε33(e), sm, 0.0; cache = cache2)
        sp, sm = state(rp), state(rm)
        @test _g(rp.fluxes.σ) ≈ _g(rm.fluxes.σ) rtol = 1.0e-12
        @test apertures(sp) == apertures(sm)
        @test open_set(sp) == open_set(sm)
    end
end
