# =============================================================================
#  frictional_strength.jl — the macroscopic limit stress states of a composite
#  whose matrix obeys a Drucker–Prager criterion with a non-associated flow
#  rule, by the modified secant method (Maghous, Dormieux and Barthélémy 2009).
#
#  The rigid–plastic matrix is the limit, as a → 0, of a viscous matrix with an
#  isotropic prestress, whose secant moduli and prestress depend on its volume
#  and deviatoric strain rates. Each is taken at an effective value, the
#  square root of its second moment over the matrix in the linear comparison
#  composite; the second moments follow from the derivatives of the effective
#  potential of that composite with respect to the moduli of the matrix. The
#  prestress enters exactly: for pores it is a uniform eigenstrain of the whole
#  cell (the pores carry no stress), for rigid inclusions its average is fixed
#  by the macroscopic volume strain rate.
# =============================================================================

"""
    DruckerPrager(; h, friction, dilatancy = friction)

The strength of a Drucker–Prager matrix, ``\\sigma_d + T\\,(\\sigma_m - h) \\le 0``,
with ``\\sigma_m`` the mean stress, ``\\sigma_d = \\sqrt{\\boldsymbol{\\sigma}^d:\\boldsymbol{\\sigma}^d}``
the norm of the stress deviator, `h` the isotropic tensile strength and `friction`
the friction coefficient ``T``. Its plastic flow follows the potential
``\\sigma_d + t\\,\\sigma_m``, `dilatancy` being ``t``, with ``0 < t \\le T``: ``t = T``
is the associated (normality) rule, a smaller ``t`` a less dilatant flow
[maghous2009](@cite).
"""
struct DruckerPrager{T <: Real}
    h::T
    friction::T
    dilatancy::T
    function DruckerPrager{T}(h, friction, dilatancy) where {T}
        # The bounds below, and the limit states, compare values.
        TensND.is_hard_numeric(T) || throw(ArgumentError("DruckerPrager needs numbers that can be compared; got $T"))
        h > 0 && friction > 0 || throw(ArgumentError("DruckerPrager: h = $h and friction = $friction must be positive"))
        0 < dilatancy <= friction || throw(
            ArgumentError(
                "DruckerPrager: the dilatancy t = $dilatancy must satisfy 0 < t ≤ T = $friction. The regularization " *
                    "of the method needs t > 0; the case t = 0 is the limit of small t."
            )
        )
        return new{T}(h, friction, dilatancy)
    end
end

function DruckerPrager(; h, friction, dilatancy = friction)
    T = promote_type(typeof(h), typeof(friction), typeof(dilatancy))
    return DruckerPrager{T}(h, friction, dilatancy)
end

"""
    Heterogeneities

What the phases other than the matrix are in a [`FrictionalComposite`](@ref):
[`Pores`](@ref) or [`RigidInclusions`](@ref).
"""
abstract type Heterogeneities end

"Empty pores: the phases other than the matrix carry no stress."
struct Pores <: Heterogeneities end

"Rigid inclusions perfectly bonded to the matrix: no strain rate inside them."
struct RigidInclusions <: Heterogeneities end

"""
    FrictionalComposite(rve, matrix::DruckerPrager; matrix_phase = :M, heterogeneities,
                        scheme = MoriTanaka(), regularization = 1e-3)

A composite of a Drucker–Prager matrix (phase `matrix_phase` of `rve`) and
`heterogeneities`, [`Pores`](@ref) or [`RigidInclusions`](@ref), which every
other phase of `rve` is. The geometry and the fractions come from `rve`, the
linear estimates from `scheme`; the stiffnesses declared in `rve` are replaced.
`regularization` is the parameter ``a`` of the viscous regularization, relative
to ``t\\,|\\boldsymbol{D}|``, the dilatancy times the size of the macroscopic strain
rate: the limit states are the limits
``a \\to 0``, which the computed ones approach at first order in ``a``.

[`limit_stress`](@ref) gives the macroscopic limit stress state reached under a
macroscopic strain rate. The method and its limits are in
[the theory page](@ref ml-theory-frictional-strength).
"""
struct FrictionalComposite{R, S, H <: Heterogeneities, T, N, E <: Number}
    rve::R
    scheme::S
    matrix_phase::Symbol
    others::NTuple{N, Symbol}
    heterogeneities::H
    matrix::DruckerPrager{T}
    fraction::T
    regularization::T
    # The element type of the cell's data (amounts and stiffnesses), known at
    # construction: the homogenization of a cell is not inferred, and its result
    # is converted to the promotion of this type with the moduli.
    numeric::Type{E}
end

function FrictionalComposite(
        rve::Schemes.RVE, matrix::DruckerPrager; matrix_phase::Symbol = :M,
        heterogeneities::Heterogeneities, scheme = Schemes.MoriTanaka(), regularization::Real = 1.0e-3
    )
    matrix_phase in Schemes.phase_names(rve) || throw(ArgumentError("the RVE has no phase :$matrix_phase"))
    others = Tuple(φ for φ in Schemes.phase_names(rve) if φ != matrix_phase)
    isempty(others) && throw(ArgumentError("FrictionalComposite needs pores or inclusions besides the matrix"))
    for φ in (matrix_phase, others...)
        Schemes.phase_property(rve, φ, :C) isa TensND.TensISO{4} ||
            throw(ArgumentError("phase :$φ must carry an isotropic stiffness :C (TensISO), replaced by the method"))
    end
    0 < regularization < 1 || throw(ArgumentError("the regularization must lie in (0, 1); got $regularization"))
    T = promote_type(typeof(matrix.h), typeof(regularization))
    fraction = T(1 - Schemes.volume_fraction(rve, matrix_phase))
    E = mapreduce(φ -> eltype(Schemes.phase_property(rve, φ, :C)), promote_type, (matrix_phase, others...); init = typeof(fraction))
    return FrictionalComposite(rve, scheme, matrix_phase, others, heterogeneities, matrix, fraction, T(regularization), E)
end

# The moduli of the heterogeneities in the comparison composite: empty pores
# softer than the softest modulus of the matrix, rigid inclusions stiffer than
# its stiffest, by this factor. The matrix grows nearly incompressible as the
# regularization vanishes (k/μ ~ 1/a), so that a contrast set on one modulus
# alone would let the other catch up with it.
const _CONTRAST = 1.0e-8
_heterogeneity_moduli(::Pores, k, μ) = (_CONTRAST * min(k, μ), _CONTRAST * min(k, μ))
_heterogeneity_moduli(::RigidInclusions, k, μ) = (max(k, μ) / _CONTRAST, max(k, μ) / _CONTRAST)

# The stiffness of the comparison composite, the matrix of bulk and shear moduli
# k and μ (energy ½ k d_v² + μ dᵈ:dᵈ).
function _comparison_stiffness(fc::FrictionalComposite, k, μ)
    kₕ, μₕ = _heterogeneity_moduli(fc.heterogeneities, k, μ)
    set(rve, φ, k, μ) = Schemes.set_param(
        Schemes.set_param(rve, Schemes.PropertyParameter(φ, :C, :bulk), 3k), Schemes.PropertyParameter(φ, :C, :shear), 2μ
    )
    rve = foldl((r, φ) -> set(r, φ, kₕ, μₕ), fc.others; init = set(fc.rve, fc.matrix_phase, k, μ))
    E = promote_type(typeof(k), typeof(μ), fc.numeric)
    return SMatrix{6, 6, E, 36}(_effective_stiffness(rve, fc.scheme))::SMatrix{6, 6, E, 36}
end

# The effective potential of the comparison composite under the macroscopic
# strain rate D, with the isotropic prestress σp in the matrix.
function _comparison_potential(::Pores, ℂ, k, σp, D, f)
    # A uniform eigenstrain −σp/(3k) 𝟏 everywhere: the pores carry no stress.
    D′ = D + σp / (3k) * identity_mandel(eltype(D))
    return _quadratic(D′, ℂ) - (1 - f) * σp^2 / (2k)
end
# The strain rate vanishes in the inclusions: the prestress works on D_v alone.
_comparison_potential(::RigidInclusions, ℂ, k, σp, D, f) = _quadratic(D, ℂ) + σp * trace_mandel(D)

_comparison_stress(::Pores, ℂ, k, σp, D) = ℂ * (D + σp / (3k) * identity_mandel(eltype(D)))
_comparison_stress(::RigidInclusions, ℂ, k, σp, D) = ℂ * D + σp * identity_mandel(eltype(D))

# The secant moduli and the prestress of the regularized matrix at the effective
# strain rates (d_v, d_d), with f_a(Y) = c (−Y)³/(a + Y) for −a < Y < 0, zero for
# Y ≥ 0: a C² convex function, infinite at Y = −a. Y = d_v − t d_d.
function _matrix_secant(m::DruckerPrager, dv, dd, a)
    u = m.dilatancy * dd - dv                                # −Y ∈ (0, a)
    c = m.h / a
    λ = c * u^2 * (3a - 2u) / (a - u)^2                      # −f_a′(Y)
    k = c * u * (3a - 2u) / (a - u)^2                        # f_a′(Y) / Y
    μ = m.dilatancy * λ / (2dd)
    σp = m.h - k * (m.dilatancy * u / m.friction + dv)
    return (k = k, μ = μ, σp = σp, u = u)
end

# The fixed point on the effective strain rates: each is the square root of its
# second moment, (1 − f)/2 ⟨d_v²⟩ = ∂W/∂k and (1 − f) ⟨dᵈ:dᵈ⟩ = ∂W/∂μ at fixed
# prestress. Outside the domain of the regularized matrix the residual is not
# finite, and the Newton iteration shortens its correction.
function _frictional_residual(fc::FrictionalComposite, y, D, a)
    m = _matrix_secant(fc.matrix, y[1], y[2], a)
    (0 < m.u < a && y[2] > 0) || return SVector(oftype(m.k, NaN), oftype(m.k, NaN))
    W(x) = _comparison_potential(fc.heterogeneities, _comparison_stiffness(fc, x[1], x[2]), x[1], m.σp, D, fc.fraction)
    ∂W = ForwardDiff.gradient(W, SVector(m.k, m.μ))
    f = fc.fraction
    return SVector(y[1] - sqrt(2 * ∂W[1] / (1 - f)), y[2] - sqrt(∂W[2] / (1 - f)))
end

"""
    limit_stress(fc::FrictionalComposite, D; algorithm = LocalNewton()) -> (; σ, dv, dd)

The macroscopic stress `σ` (Kelvin–Mandel components) of the limit state that
the composite reaches under the macroscopic strain rate `D`, with the effective
volume and deviatoric strain rates of the matrix, `dv` and `dd`. It depends on
the direction of `D` only. For [`Pores`](@ref) every direction gives a point of
the boundary of the strength domain. For [`RigidInclusions`](@ref) the limit
states form a cone whose points share one flow direction: `D` must satisfy the
macroscopic flow rule to within the regularization, its distance to it setting
the point on the cone (see [the example](@ref ml-ex-frictional-strength)); a
strain rate that admits no limit state is refused (an `ArgumentError`, or a
[`LocalIntegrationError`](@ref) when the iteration finds no fixed point at
all).
The iteration compares values, so that symbolic numbers are refused.
"""
function limit_stress(fc::FrictionalComposite, D::StaticVector{6}; algorithm::LocalNewton = LocalNewton(; reltol = 1.0e-10))
    T = promote_type(eltype(D), typeof(fc.matrix.h), typeof(fc.matrix.friction), typeof(fc.matrix.dilatancy), typeof(fc.fraction))
    TensND.is_hard_numeric(T) || throw(ArgumentError("limit_stress solves a fixed point iteratively, which needs numbers that can be compared; got $T"))
    # The volume strain rate d_v = t d_d + Y is a root mean square, nonnegative,
    # so that Y ∈ (−a, 0) needs a below t d_d: a is relative to t |D|.
    a = fc.regularization * fc.matrix.dilatancy * norm(D)
    # A start inside the domain of the regularized matrix: Y = −a/2.
    dd₀ = norm(deviator(TensND.primal.(D))) / (1 - TensND.primal(fc.fraction)) + abs(trace_mandel(TensND.primal.(D)))
    y₀ = SVector(TensND.primal(fc.matrix.dilatancy) * dd₀ - TensND.primal(a) / 2, dd₀)
    y = _root(z -> _frictional_residual(fc, z, D, a), y₀, algorithm, :FrictionalComposite, T)
    m = _matrix_secant(fc.matrix, y[1], y[2], a)
    # A strain rate without a limit state drives the regularized matrix to the
    # edge of its domain, Y → −a, where its stress grows without bound: the
    # fixed point is then no limit state. Genuine ones sit well inside
    # (1 − u/a of order one), spurious ones within rounding of the edge.
    1 - TensND.primal(m.u / a) > sqrt(eps(float(typeof(TensND.primal(a))))) || throw(
        ArgumentError(
            "the macroscopic strain rate $(TensND.primal.(D)) admits no limit state: the stress of the " *
                "regularized composite grows without bound. With rigid inclusions, D must follow the " *
                "macroscopic flow rule to within the regularization."
        )
    )
    σ = _comparison_stress(fc.heterogeneities, _comparison_stiffness(fc, m.k, m.μ), m.k, m.σp, D)
    return (σ = σ, dv = y[1], dd = y[2])
end
