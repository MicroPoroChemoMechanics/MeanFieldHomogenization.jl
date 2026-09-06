using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy   # top level on purpose: `@syms` is expanded before a `using`
# nested in a `@testset` body would have run.

const LSpd = MeanFieldHomogenization.LayeredSpheroids

# =============================================================================
#  test_ad.jl — type genericity of `LayeredSpheroid`: ForwardDiff.Dual,
#  BigFloat and SymPy.Sym.
#
#  The module had no such test at all, and every one of the cases below used
#  to throw. Three independent causes, all of the same family — a promotion
#  that forgot part of the data:
#
#    * `spheroid_state_sequence` sized its buffers from `promote_type(Q, k₀)`,
#      leaving out the LAYER conductivities and the INTERFACE parameters;
#    * `_J_int` typed its coupling blocks from `eltype(_J_ext(...))`, which
#      knows nothing about `coef` — the factor carrying the interface
#      parameter;
#    * the constructors bound a single `T` across every semi-axis, so one
#      `Dual` boundary among `Float64` ones was a `MethodError`;
#    * `_bisect_cubic_root` decided by sign tests, which carry no derivative,
#      so `layered_spheroid_from_fractions` returned zero partials.
#
#  Every derivative below is checked against a central difference: a lost or
#  truncated derivative cannot survive that.
# =============================================================================

_K(k) = TensISO{3}(k)

function _cdiff(f, x; h = 1.0e-6)
    δ = h * max(abs(x), one(x))
    return (f(x + δ) - f(x - δ)) / (2δ)
end

# Confocal pairs: `axis² - disk²` is `+1/4` (prolate) or `-1/4` (oblate).
const _AX_P, _DK_P = (1.0, 2.0), (sqrt(0.75), sqrt(3.75))
const _AX_O, _DK_O = (1.0, 2.0), (sqrt(1.25), sqrt(4.25))

"First core coefficient of the state sequence — one scalar the whole solve feeds."
_probe(s, k₀, trans) = real(LSpd.spheroid_state_sequence(s, k₀, trans)[1][1])

@testset "LayeredSpheroid — ForwardDiff" begin

    @testset "one layer conductivity among Float64 ones ($(kind))" for
        (kind, ax, dk) in (("prolate", _AX_P, _DK_P), ("oblate", _AX_O, _DK_O))
        for trans in (false, true)
            f = k₁ -> _probe(LayeredSpheroid(ax, dk, (_K(k₁), _K(5.0))), 2.0, trans)
            @test ForwardDiff.derivative(f, 1.0) ≈ _cdiff(f, 1.0) rtol = 1.0e-5
        end
    end

    @testset "matrix conductivity k₀" begin
        s = LayeredSpheroid(_AX_P, _DK_P, (_K(1.0), _K(5.0)))
        f = k₀ -> _probe(s, k₀, false)
        @test ForwardDiff.derivative(f, 2.0) ≈ _cdiff(f, 2.0) rtol = 1.0e-5
    end

    @testset "one semi-axis among Float64 ones (promoting constructor)" begin
        # `a₁` moves the inner boundary; `b₁` follows to stay confocal.
        f = function (a₁)
            s = LayeredSpheroid((a₁, 2.0), (sqrt(a₁^2 - 0.25), _DK_P[2]), (_K(1.0), _K(5.0)))
            return _probe(s, 2.0, false)
        end
        @test ForwardDiff.derivative(f, 1.0) ≈ _cdiff(f, 1.0; h = 1.0e-7) rtol = 1.0e-5
        # …and the mixed tuple is what used to be a MethodError.
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        @test LayeredSpheroid(
            (D(1.0), 2.0), (D(sqrt(0.75)), _DK_P[2]),
            (_K(1.0), _K(5.0))
        ) isa LayeredSpheroid{D}
    end

    @testset "interface parameters" begin
        perfect = PerfectInterface{Float64}()
        for (name, mk, x₀) in (
                ("Kapitza resistance", α -> KapitzaInterface(α), 0.1),
                ("surface conductance", β -> SurfaceConductiveInterface(β), 0.2),
            )
            @testset "$name" begin
                for trans in (false, true)
                    f = function (p)
                        s = LayeredSpheroid(
                            _AX_P, _DK_P, (_K(1.0), _K(5.0));
                            interfaces = (mk(p), perfect)
                        )
                        return _probe(s, 2.0, trans)
                    end
                    d = ForwardDiff.derivative(f, x₀)
                    @test d ≈ _cdiff(f, x₀) rtol = 1.0e-5
                    @test !iszero(d)      # a silent zero is the failure mode being guarded
                end
            end
        end
    end

    @testset "through the bisection: layered_spheroid_from_fractions" begin
        # Sign tests carry no derivative; the Newton polish is what restores it.
        ffrac = f₁ -> _probe(
            layered_spheroid_from_fractions(3.0, 1.0, (f₁, 1 - f₁), (_K(1.0), _K(5.0))),
            2.0, false
        )
        @test ForwardDiff.derivative(ffrac, 0.3) ≈ _cdiff(ffrac, 0.3) rtol = 1.0e-5

        for (kind, ω₀, trans) in (("prolate", 3.0, false), ("oblate", 0.25, true))
            fω = ω -> _probe(
                layered_spheroid_from_fractions(ω, 1.0, (0.3, 0.7), (_K(1.0), _K(5.0))),
                2.0, trans
            )
            d = ForwardDiff.derivative(fω, ω₀)
            @test d ≈ _cdiff(fω, ω₀) rtol = 1.0e-5
            @test !iszero(d)
        end
    end

    @testset "gradient with respect to several parameters at once" begin
        g = function (p)
            s = LayeredSpheroid(
                _AX_P, _DK_P, (_K(p[1]), _K(p[2]));
                interfaces = (KapitzaInterface(p[3]), PerfectInterface{Float64}())
            )
            return _probe(s, p[4], false)
        end
        p₀ = [1.0, 5.0, 0.1, 2.0]
        ∇ = ForwardDiff.gradient(g, p₀)
        for i in 1:4
            fi = x -> g([j == i ? x : p₀[j] for j in 1:4])
            @test ∇[i] ≈ _cdiff(fi, p₀[i]) rtol = 1.0e-5
        end
    end

    @testset "through the full homogenize path" begin
        f = function (k₁)
            s = LayeredSpheroid(_AX_P, _DK_P, (_K(k₁), _K(5.0)); Nseries = 6)
            r = RVE()
            add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => _K(2.0)); fraction = :rest)
            add_phase!(r, :I, s, Dict(:K => _K(5.0)); fraction = 0.15)
            return get_array(homogenize(r, MoriTanaka(), :K))[1, 1]
        end
        @test ForwardDiff.derivative(f, 1.0) ≈ _cdiff(f, 1.0) rtol = 1.0e-5
    end
end

@testset "_q_recurrence_plan — the numeric/symbolic gate keeps its teeth" begin
    plan = LSpd._q_recurrence_plan
    # The gate that lets a symbolic type through must NOT be read as
    # `is_hard_numeric(Tx)`: an oblate spheroid carries `q = iτ`, so `Tx` is
    # `Complex{Float64}`, which is not hard numeric. Testing `Tx` itself would
    # route every oblate case onto the upward recurrence — the very instability
    # Miller's downward variant is here for.
    @test first(plan(5.0, 12, Float64)) === :downward
    @test first(plan(im * 5.0, 12, Complex{Float64})) === :downward
    @test first(plan(im * 50.0, 12, Complex{Float64})) === :downward
    @test first(plan(big(5.0), 12, BigFloat)) === :downward
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    @test first(plan(D(5.0), 12, D)) === :downward
    @test first(plan(Complex{D}(im * 5.0), 12, Complex{D})) === :downward
    # Nearly degenerate: upward loses almost nothing, Miller would be ruinous.
    @test first(plan(1.0001, 12, Float64)) === :upward
    # Exact types have no cancellation to control, and `eps`/`ceil` are
    # meaningless on them.
    @test first(plan(Sym(5), 12, Sym)) === :upward
end

@testset "LayeredSpheroid — BigFloat" begin
    ax = (big(1.0), big(2.0))
    dk = (sqrt(big(0.75)), sqrt(big(3.75)))
    s = LayeredSpheroid(ax, dk, (_K(big(1.0)), _K(big(5.0))); Nseries = 6)
    v = _probe(s, big(2.0), false)
    @test v isa BigFloat
    sf = LayeredSpheroid(_AX_P, _DK_P, (_K(1.0), _K(5.0)); Nseries = 6)
    @test Float64(v) ≈ _probe(sf, 2.0, false) rtol = 1.0e-12
end

@testset "LayeredSpheroid — SymPy" begin
    SymPy.@syms k₁ᶜ::positive k₂ᶜ::positive

    @testset "symbolic geometry needs an explicit prolate/oblate declaration" begin
        SymPy.@syms aᶜ::positive bᶜ::positive
        # The sign of `a² - b²` is undecidable, and SymPy answers `false` to a
        # comparison it cannot settle — so refuse rather than guess.
        @test_throws ArgumentError LayeredSpheroid((aᶜ,), (bᶜ,), (_K(k₁ᶜ),))
        @test LayeredSpheroid((aᶜ,), (bᶜ,), (_K(k₁ᶜ),); prolate = true) isa LayeredSpheroid
    end

    @testset "symbolic solve, then subs, matches Float64" begin
        s = LayeredSpheroid(
            (Sym(1), Sym(2)), (sqrt(Sym(3)) / 2, sqrt(Sym(15)) / 2),
            (_K(k₁ᶜ), _K(k₂ᶜ)); prolate = true, Nseries = 2
        )
        X = LSpd.spheroid_state_sequence(s, Sym(2), false)
        @test eltype(X[1]) <: Sym
        v = Float64((X[1][1].subs(k₁ᶜ, 1).subs(k₂ᶜ, 5)).evalf())
        sf = LayeredSpheroid(_AX_P, _DK_P, (_K(1.0), _K(5.0)); Nseries = 2)
        @test v ≈ _probe(sf, 2.0, false) rtol = 1.0e-12
    end

    @testset "fully symbolic semi-axes give a closed form" begin
        SymPy.@syms a₁ᶜ::positive a₂ᶜ::positive
        s = LayeredSpheroid(
            (a₁ᶜ, a₂ᶜ), (sqrt(a₁ᶜ^2 - Sym(1) // 4), sqrt(a₂ᶜ^2 - Sym(1) // 4)),
            (_K(k₁ᶜ), _K(k₂ᶜ)); prolate = true, Nseries = 1
        )
        X = LSpd.spheroid_state_sequence(s, Sym(2), false)
        expr = X[1][1]
        @test expr isa Sym
        # It really depends on the symbols — not a constant that slipped through.
        @test !isempty(intersect(SymPy.free_symbols(expr), [a₁ᶜ, a₂ᶜ, k₁ᶜ, k₂ᶜ]))
    end
end
