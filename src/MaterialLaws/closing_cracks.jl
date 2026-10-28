# =============================================================================
#  closing_cracks.jl — nonlinear poroelasticity of a rock whose microcracks
#  close progressively under the effective stress, as a law of MaterialLaws.
#
#  Each family of cracks is a population of cracks of different apertures: under
#  a compressive normal traction the thinner ones close first, so that the open
#  density of the family decreases smoothly as the compression grows. What
#  closes the cracks is the Terzaghi effective stress Σ' = Σ + p 𝟏. The drained
#  stiffness, the Biot tensor and the Biot modulus are those of the solid with
#  the open densities; the law is integrated incrementally, the tangent moduli
#  being those of the state at the end of the step.
# =============================================================================

import ..Cracks
import ..Poromechanics

"""
    ExponentialClosure(σc)

The open fraction of a family of cracks whose closure stresses follow an
exponential distribution of mean `σc`: under the normal effective traction
``\\sigma_n`` on the plane of the family (negative in compression),

```math
f(\\sigma_n) = \\exp\\!\\Big(-\\frac{\\langle -\\sigma_n\\rangle}{\\sigma_c}\\Big),
\\qquad \\langle x\\rangle = \\max(x, 0):
```

every crack is open under tension, and a fraction ``1/e`` is still open under a
compression equal to ``\\sigma_c``. The closure stress of a crack being
proportional to its aspect ratio, this is an exponential distribution of the
aspect ratios. Any function `σn -> f`, between 0 and 1, can be given instead.
"""
struct ExponentialClosure{T}
    σc::T
    function ExponentialClosure{T}(σc) where {T}
        σc > 0 || throw(ArgumentError("ExponentialClosure: the mean closure stress σc must be positive; got $σc"))
        return new{T}(σc)
    end
end

ExponentialClosure(σc) = ExponentialClosure{typeof(σc)}(σc)

(f::ExponentialClosure)(σn) = exp(-max(-σn, zero(σn)) / f.σc)

"""
    ClosingCrackPoroelasticity(rve; scheme = MoriTanaka(), closure, porosity = 0, families)

The law of a saturated rock whose microcracks **close progressively** under the
effective stress. `rve` is a cell of the package whose crack phases, each with
a given orientation, are the families; their densities in `rve` are those of
the cracks all open. `closure` gives the open fraction of a family as a function
of the normal effective traction on its plane: an [`ExponentialClosure`](@ref),
any function `σn -> f`, or a `NamedTuple` of them keyed by family. `porosity`
is the porosity occupied by the fluid (the volume of the cracks, small for thin
cracks), which sets the Biot modulus; it is held at its given value.

The microstructure sees the Terzaghi effective stress
``\\boldsymbol{\\Sigma}' = \\boldsymbol{\\Sigma} + p\\,\\boldsymbol{1}``: family ``i``
has the open density ``d_i\\,f_i(\\underline{n}_i\\cdot\\boldsymbol{\\Sigma}'\\cdot\\underline{n}_i)``.
The drained stiffness ``\\mathbb{C}^{\\mathrm{hom}}``, the Biot tensor
``\\boldsymbol{B}`` and the Biot modulus ``M`` of the solid with those open
densities give the rates

```math
\\dot{\\boldsymbol{\\Sigma}}' = \\mathbb{C}^{\\mathrm{hom}}:\\dot{\\boldsymbol{E}} + \\dot p\\,(\\boldsymbol{1} - \\boldsymbol{B}),
\\qquad
\\dot\\varphi = \\boldsymbol{B}:\\dot{\\boldsymbol{E}} + \\frac{\\dot p}{M},
```

integrated by a backward Euler step. Gradients `(ε = 6, p = 1)`, fluxes
`(σ = 6, φ = 1)`, internal variables the effective stress `Σ′` and the fluid
content `φ`. The derivation is in [the theory page](@ref mp-theory-closing-cracks).
"""
function ClosingCrackPoroelasticity(
        rve::Schemes.RVE; scheme = Schemes.MoriTanaka(), closure, porosity::Real = 0,
        families = nothing, name::Symbol = :ClosingCrackPoroelasticity
    )
    fams = families === nothing ? Constitutive._crack_phase_names(rve) : Tuple(families)
    isempty(fams) && throw(ArgumentError("ClosingCrackPoroelasticity needs at least one crack family (a phase with a CrackDensity)"))
    for φ in fams
        rve.amounts[φ] isa Schemes.CrackDensity ||
            throw(ArgumentError("phase :$φ is not a crack family: its amount is not a CrackDensity"))
        get(rve.symmetrize, φ, Schemes.NoSymmetrize()) isa Schemes.NoSymmetrize || throw(
            ArgumentError(
                "crack family :$φ is symmetrized over its orientations: its cracks do not all see the " *
                    "same normal traction. Declare oriented families instead (several, to sample the orientations)."
            )
        )
    end
    normals = NamedTuple{fams}(map(φ -> _normal_projector(rve.phases[φ].geometry), fams))
    closures = NamedTuple{fams}(map(φ -> _family_resistance(closure, φ), fams))
    densities = SVector(map(φ -> Schemes.get_param(rve, Schemes.AmountParameter(φ)), fams))
    C_s = _mandel(TensND.change_tens_canon(Schemes.reference_property(rve, scheme, :C)))
    model = cracked_stiffness(rve, scheme, fams)
    return ResidualLaw(;
        name, fluxes = _cc_fluxes, residual = _cc_residual,
        gradients = (ε = 6, p = 1), flux_names = (σ = 6, φ = 1), layout = (Σ′ = 6, φ = 1),
        p = (; model, closures, normals, densities, C_s, porosity)
    )
end

# n ⊗ n in Mandel components, so that n·Σ·n is its dot product with Σ.
function _normal_projector(geometry)
    n = TensND.components_canon(Cracks.crack_normal(geometry))
    return _mandel(TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> n[i] * n[j])))
end

# The poroelastic moduli of the solid at the effective stress Σ′: the open
# density of every family, the homogenized stiffness, and the Biot parameters.
function _cc_moduli(Σ′, p)
    f = SVector(map(φ -> p.closures[φ](transpose(p.normals[φ]) * Σ′), keys(p.closures)))
    C = _stiffness_at(p.model, p.densities .* f)
    par = Poromechanics.poroelastic_parameters(C, p.C_s, p.porosity)
    return (C = C, B = par.B, invM = par.inverse_modulus)
end

_cc_fluxes(g, y, p) = (σ = y.Σ′ - g.p * identity_mandel(eltype(y.Σ′)), φ = y.φ)

function _cc_residual(y, g, prev, Δt, p)
    m = _cc_moduli(y.Σ′, p)
    Δε, Δp = g.ε - prev.g.ε, g.p - prev.g.p
    return (
        Σ′ = y.Σ′ - prev.y.Σ′ - m.C * Δε - Δp * (identity_mandel(eltype(m.B)) - m.B),
        φ = y.φ - prev.y.φ - transpose(m.B) * Δε - m.invM * Δp,
    )
end
