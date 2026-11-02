using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using ForwardDiff

# =============================================================================
#  test_laminate_frames.jl — a laminate in any frame, in every physics.
#
#  Reference: frame covariance. A laminate whose normal is turned by a rotation R,
#  with layers whose properties are turned by R as well, must give the result of the
#  laminate along e₃ turned by R — in elasticity and conduction, and in aging
#  viscoelasticity at both orders, time block by time block. The layers are
#  anisotropic (random positive definite tensors) and the aging laws genuine, so that
#  neither an isotropic layer nor the elastic limit can hide a frame error. Three
#  rotations, an in-plane spin included. Interfaces in a tilted frame are held to the
#  elastic limit; a tilted laminate nested as a `Homogenized` phase of an aging cell
#  is held to the explicit two-step chain.
#
#  Aging conduction in a tilted laminate used to be refused, and the plain 3n × 3n
#  matrix an aging laminate returns in conduction was read with 6 × 6 blocks when
#  passed on as a phase property.
# =============================================================================

const _LF_RNG = Xoshiro(20261101)
_lf_spd(n, s) = (A = randn(_LF_RNG, n, n); s * (A * transpose(A) + n * I))
_lf_C(M) = TensND.Tens(TensND.get_array(TensND.from_mandel(M)))
_lf_canonC(X) = TensND.mandel(TensND.canonical_components(X); symmetrize = true)
_lf_canonK(X) = Matrix(TensND.canonical_components(X))
const _LF_TIMES = [0.0, 0.5, 1.5, 3.0]
_lf_age(c, τ) = ViscoLaw((t, tp) -> t >= tp ? (1 + 0.4tp) * (0.6 + 0.4exp(-(t - tp) / τ)) * c : zero(c))
const _LF_CA, _LF_CB = _lf_spd(6, 3.0), _lf_spd(6, 0.5)
const _LF_KA, _LF_KB = _lf_spd(3, 2.0), _lf_spd(3, 0.2)

function _lf_laminate(angles, prop, values; law = identity, interface = PerfectInterface())
    lam = angles === nothing ? Laminate(; normal = (0, 0, 1)) : Laminate(; euler_angles = angles)
    add_layer!(lam, :A, Dict(prop => law(values[1])); thickness = 0.3, interface)
    add_layer!(lam, :B, Dict(prop => law(values[2])); thickness = 0.7)
    return lam
end

@testset "Laminate in any frame — elasticity, conduction, aging, both orders" begin
    n = length(_LF_TIMES)
    for angles in ((0.7, 0.4, 0.0), (1.1, 2.3, 0.9), (0.3, -0.8, 2.0))
        R = vecbasis(RotatedBasis(angles...))
        Q = TensND.mandel_rotation(angles...)
        rotC(M) = Q * M * transpose(Q)
        rotK(M) = R * M * transpose(R)
        Cs0, Cs1 = _lf_C.((_LF_CA, _LF_CB)), _lf_C.(rotC.((_LF_CA, _LF_CB)))
        Ks0, Ks1 = TensND.Tens.((_LF_KA, _LF_KB)), TensND.Tens.(rotK.((_LF_KA, _LF_KB)))

        C0 = _lf_canonC(homogenize(_lf_laminate(nothing, :C, Cs0), Laminated(), :C))
        @test _lf_canonC(homogenize(_lf_laminate(angles, :C, Cs1), Laminated(), :C)) ≈ rotC(C0) rtol = 1.0e-12
        K0 = _lf_canonK(homogenize(_lf_laminate(nothing, :K, Ks0), Laminated(), :K))
        @test _lf_canonK(homogenize(_lf_laminate(angles, :K, Ks1), Laminated(), :K)) ≈ rotK(K0) rtol = 1.0e-12

        ageC, ageK = c -> _lf_age(c, 0.8), k -> _lf_age(k, 1.3)
        alv(angles, prop, values, law) = Matrix(
            homogenize_alv(_lf_laminate(angles, prop, values; law), Laminated(), prop; times = _LF_TIMES)
        )
        Qb, Rb = kron(Matrix(1.0I, n, n), Q), kron(Matrix(1.0I, n, n), R)
        @test alv(angles, :C, Cs1, ageC) ≈ Qb * alv(nothing, :C, Cs0, ageC) * transpose(Qb) rtol = 1.0e-11
        @test alv(angles, :K, Ks1, ageK) ≈ Rb * alv(nothing, :K, Ks0, ageK) * transpose(Rb) rtol = 1.0e-11
    end
end

@testset "Laminate in a tilted frame — interfaces, elastic limit" begin
    angles = (1.1, 2.3, 0.9)
    for (prop, itf, A, B, canon) in (
            (:K, KapitzaInterface(0.4), TensND.Tens(_LF_KA), TensND.Tens(_LF_KB), _lf_canonK),
            (:C, SpringInterface(; sn = 3.0, st = 1.5), _lf_C(_LF_CA), _lf_C(_LF_CB), _lf_canonC),
        )
        E = canon(homogenize(_lf_laminate(angles, prop, (A, B); interface = itf), Laminated(), prop))
        M = Matrix(
            homogenize_alv(
                _lf_laminate(angles, prop, (A, B); law = heaviside_law, interface = itf), Laminated(), prop;
                times = _LF_TIMES
            )
        )
        nc = size(E, 1)
        for i in eachindex(_LF_TIMES)
            @test M[(nc * (i - 1) + 1):(nc * i), (nc * (i - 1) + 1):(nc * i)] ≈ E rtol = 1.0e-11
        end
    end
end

@testset "Laminate in a tilted frame — nested in an aging cell, both orders" begin
    angles = (0.7, 0.4, 0.0)
    m = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(3 * (5 + 3exp(-(t - tp))), 2 * (3 + 2exp(-(t - tp)))) : TensISO{3}(0.0, 0.0))
    mk = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(1.0 + exp(-(t - tp))) : TensISO{3}(0.0))
    R = vecbasis(RotatedBasis(angles...))
    Q = TensND.mandel_rotation(angles...)
    for (prop, matrix_law, values, τ) in (
            (:C, m, _lf_C.((Q * _LF_CA * transpose(Q), Q * _LF_CB * transpose(Q))), 0.8),
            (:K, mk, TensND.Tens.((R * _LF_KA * transpose(R), R * _LF_KB * transpose(R))), 1.3),
        )
        lam = _lf_laminate(angles, prop, values; law = c -> _lf_age(c, τ))
        outer(p) = (
            r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(prop => matrix_law); fraction = :rest);
            add_phase!(r, :L, Ellipsoid(1.0), Dict(prop => p); fraction = 0.25); r
        )
        explicit_matrix = homogenize_alv(lam, Laminated(), prop; times = _LF_TIMES)
        explicit = homogenize_alv(outer(explicit_matrix), MoriTanaka(:M), prop; times = _LF_TIMES)
        @test Matrix(homogenize_alv(outer(Homogenized(lam, Laminated())), MoriTanaka(:M), prop; times = _LF_TIMES)) ≈
            Matrix(explicit) rtol = 1.0e-12
        # The plain matrix of an operator passes on as the operator itself, at both orders.
        as_operator = homogenize_alv(outer(VolterraOperator(explicit_matrix, _LF_TIMES)), MoriTanaka(:M), prop)
        @test Matrix(as_operator) ≈ Matrix(explicit) rtol = 1.0e-13
    end
end

@testset "Laminate in a tilted frame — derivative of aging conduction with respect to the orientation" begin
    # Reference: the elastic `Laminated` scheme, which takes any number type, against
    # the diagonal blocks of the Heaviside laminate (the order-4 test is in
    # test_laminate_alv.jl).
    function lam_at(θ, law)
        lam = Laminate(; normal = (0, sin(θ), cos(θ)))
        add_layer!(lam, :A, Dict(:K => law(TensND.Tens(_LF_KA))); thickness = 0.3)
        add_layer!(lam, :B, Dict(:K => law(TensND.Tens(_LF_KB))); thickness = 0.7)
        return lam
    end
    K_el(θ) = _lf_canonK(homogenize(lam_at(θ, identity), Laminated(), :K))
    blk(θ) = Matrix(homogenize_alv(lam_at(θ, heaviside_law), Laminated(), :K; times = [0.0, 1.0]))[4:6, 4:6]
    θ = 0.4
    @test blk(θ) ≈ K_el(θ) rtol = 1.0e-12
    @test ForwardDiff.derivative(blk, θ) ≈ ForwardDiff.derivative(K_el, θ) rtol = 1.0e-10
end
