# =============================================================================
#  homogenized.jl — a composite of nonlinear phases, homogenized by a secant
#  method, as a law of MaterialPoints.
#
#  At a given macroscopic strain, each nonlinear phase is replaced by a linear
#  one whose moduli are its secant moduli at an effective strain of the phase;
#  the linear composite they form (the linear comparison composite) is
#  homogenized by any scheme of the package, which gives the macroscopic stress
#  and, in turn, the effective strains of the phases. The effective strains are
#  the unknowns of a fixed point: they are the internal variables of a
#  ResidualLaw whose residual is that fixed point, so that the consistent
#  tangent, the point driver and the finite-element coupling come with it.
# =============================================================================

import ..Schemes

"""
    SecantMethod

How the effective strain of a phase is measured in a secant method:
[`ClassicalSecant`](@ref) by its average, [`ModifiedSecant`](@ref) by its
second moment.
"""
abstract type SecantMethod end

"""
    ClassicalSecant()

The effective strain of a phase is the equivalent strain of its **average**
strain, ``\\bar\\varepsilon_i = \\varepsilon_{\\mathrm{eq}}(\\mathbb{A}_i : \\boldsymbol{E})``,
``\\mathbb{A}_i`` being the strain concentration tensor of the phase in the linear
comparison composite.
"""
struct ClassicalSecant <: SecantMethod end

"""
    ModifiedSecant()

The effective strain of a phase is the square root of the **second moment** of
its equivalent strain,

```math
\\bar\\varepsilon_i^2 = \\langle \\varepsilon_{\\mathrm{eq}}^2 \\rangle_i
= \\frac{2}{3\\,f_i}\\;\\boldsymbol{E} : \\frac{\\partial\\mathbb{C}^{\\mathrm{hom}}}{\\partial (2\\mu_i)} : \\boldsymbol{E},
```

the derivative of the effective stiffness of the linear comparison composite
with respect to the shear modulus of the phase giving the energy the phase
stores [kreher1990](@cite). It is the method whose estimates coincide with the
variational ones of [ponteCastaneda1991](@citet) for power-law phases (see
[the theory page](@ref mp-theory-secant)).
"""
struct ModifiedSecant <: SecantMethod end

"""
    secant_moduli(law, εeq) -> (k, μ)

The isotropic secant moduli of a phase law without internal variable at the
equivalent strain `εeq`, read on its own response: the shear modulus from a
deviatoric strain ``\\varepsilon_{\\mathrm{eq}}\\,(1, -\\tfrac12, -\\tfrac12)``, of
equivalent strain ``\\varepsilon_{\\mathrm{eq}}``, and the bulk modulus from a volume
change of the same size.
"""
function secant_moduli(law::AbstractPointLaw, εeq)
    T = typeof(εeq)
    none = SVector{0, T}()
    σ_dev = _fluxes(law, Mandel{T}(εeq, -εeq / 2, -εeq / 2, 0, 0, 0), none)
    σ_vol = _fluxes(law, Mandel{T}(εeq, εeq, εeq, 0, 0, 0), none)
    return (k = σ_vol[1] / (3εeq), μ = (σ_dev[1] - σ_dev[2]) / (3εeq))
end

"""
    NonlinearHomogenized(rve, laws; scheme = MoriTanaka(), method = ModifiedSecant())

The law of a composite whose phases follow nonlinear laws, homogenized by a
secant method. `rve` is a cell of the package whose phases carry an isotropic
stiffness `:C` (any value for a phase given a law: it is replaced), `laws`
maps phase names to laws of `MaterialPoints` without internal variable
(nonlinear elastic, e.g. [`PowerLawElasticity`](@ref)); a phase without a law
keeps its linear stiffness. `scheme` is any linear homogenization scheme of the
package, `method` a [`SecantMethod`](@ref).

At the macroscopic strain ``\\boldsymbol{E}``, every phase ``i`` with a law takes
its secant moduli at its effective strain ``\\bar\\varepsilon_i``, the linear
composite they form is homogenized into ``\\mathbb{C}^{\\mathrm{hom}}``, and

```math
\\boldsymbol{\\Sigma} = \\mathbb{C}^{\\mathrm{hom}}(\\bar\\varepsilon_1, \\ldots, \\bar\\varepsilon_n) : \\boldsymbol{E},
\\qquad
\\bar\\varepsilon_i = \\text{the effective strain of phase } i \\text{ in that composite},
```

a fixed point in the effective strains. The result is a
[`ResidualLaw`](@ref) whose internal variables are those effective strains, one
per phase with a law, named after the phases: the consistent tangent of the
composite follows from the implicit-function theorem, and the law runs in
[`point_response`](@ref) and in a finite-element computation like any other.
"""
function NonlinearHomogenized(
        rve::Schemes.RVE, laws::AbstractDict{Symbol};
        scheme = Schemes.MoriTanaka(), method::SecantMethod = ModifiedSecant(), name::Symbol = :NonlinearHomogenized
    )
    phases = Tuple(sort!(collect(keys(laws))))
    for φ in phases
        law = laws[φ]
        law isa AbstractPointLaw && isempty(law.layout) && keys(law.gradients) == (:ε,) || throw(
            ArgumentError(
                "phase :$φ: a secant method takes laws of the strain alone and without internal " *
                    "variable (nonlinear elasticity); got $law"
            )
        )
        Schemes.phase_property(rve, φ, :C) isa TensND.TensISO{4} || throw(
            ArgumentError("phase :$φ must carry an isotropic stiffness :C (TensISO), replaced by its secant one")
        )
    end
    lawsnt = NamedTuple{phases}(Tuple(laws[φ] for φ in phases))
    fractions = NamedTuple{phases}(Tuple(Schemes.volume_fraction(rve, φ) for φ in phases))
    return ResidualLaw(;
        name, stress = _nh_stress, residual = _nh_residual, elastic = _nh_at_rest,
        layout = NamedTuple{phases}(ntuple(_ -> 1, length(phases))),
        y₀ = [_reference_strain(lawsnt[φ]) for φ in phases],
        p = (; rve, scheme, method, laws = lawsnt, fractions)
    )
end

# A starting value of the effective strains: the law's reference strain if it
# has one, else a small strain. Only the start of the first Newton iteration.
_reference_strain(law) = hasproperty(law.p, :ε₀) ? float(law.p.ε₀) : 1.0e-3

# The linear comparison composite at the effective strains y.
function _secant_rve(p, y)
    return foldl(keys(p.laws); init = p.rve) do rve, φ
        s = secant_moduli(p.laws[φ], y[φ])
        rve = Schemes.set_param(rve, Schemes.PropertyParameter(φ, :C, :bulk), 3 * s.k)
        Schemes.set_param(rve, Schemes.PropertyParameter(φ, :C, :shear), 2 * s.μ)
    end
end

_effective_stiffness(rve, scheme) = _mandel(Schemes.homogenize(rve, scheme, :C))

_nh_stress(ε, y, p) = _effective_stiffness(_secant_rve(p, y), p.scheme) * ε

function _nh_residual(y, g, prev, Δt, p)
    rve = _secant_rve(p, y)
    φs = keys(p.laws)
    return NamedTuple{φs}(map(φ -> y[φ] - _phase_strain(p.method, rve, p, φ, g.ε), φs))
end

# No macroscopic strain, no strain in the phases: the effective strains keep
# their starting values (the secant moduli of a law may not exist at zero).
_nh_at_rest(g, prev, Δt, p) = iszero(g.ε)

function _phase_strain(::ClassicalSecant, rve, p, φ, ε)
    A = Schemes.concentration_tensors(rve, p.scheme, :C).strain[φ]
    return equivalent_strain(_mandel(A) * ε)
end

function _phase_strain(::ModifiedSecant, rve, p, φ, ε)
    shear = Schemes.PropertyParameter(φ, :C, :shear)
    # W = ½ E:ℂ:E, whose derivative with respect to 2μ of the phase is ¾ fᵢ ⟨ε_eq²⟩ᵢ.
    energy(x) = _quadratic(ε, _effective_stiffness(Schemes.set_param(rve, shear, x), p.scheme))
    ∂W = ForwardDiff.derivative(energy, Schemes.get_param(rve, shear))
    return sqrt(4 * ∂W / (3 * p.fractions[φ]))
end
