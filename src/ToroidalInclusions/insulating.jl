# ── The insulating torus ─────────────────────────────────────────────────────

"""
    ToroidalShape

Shape tag of the toroidal inclusions.
"""
struct ToroidalShape end

"""
    InsulatingTorus(R, r; euler_angles = (), basis = nothing)

A toroidal pore of the conduction problem: the torus of center-line radius `R`
and tube radius `r < R`, of axis the third vector of `basis` (built from the ZYZ
`euler_angles` when no basis is given), through which no flux passes, in an
isotropic matrix.

The temperature around it is a series of toroidal harmonics; its far field gives
the gradient concentration tensor, transversely isotropic about the axis,

```math
\\boldsymbol{A} = A_t\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n}) + A_n\\,\\underline{n}\\otimes\\underline{n} ,
```

which depends on the ratio `r/R` only. The flux concentration vanishes, so that
the contributions per unit volume of the torus are
``\\boldsymbol{N} = -\\boldsymbol{K}_0\\cdot\\boldsymbol{A}`` and
``\\boldsymbol{R} = \\boldsymbol{A}\\cdot\\boldsymbol{K}_0^{-1}``. For a thin ring
``A_t \\to 3/2`` and ``A_n \\to 2``.

The property of the phase is not read. The reference medium must be isotropic;
`R` and `r` are numbers or duals, the series being summed by iterations that
stop on a value.

# Examples
```jldoctest
julia> using TensND

julia> A = gradient_gradient_concentration(InsulatingTorus(1.0, 1.0e-4), TensISO{3}(0.0), TensISO{3}(1.0));

julia> round.(canonical_components(A)[[1, 5, 9]]; digits = 4)
3-element Vector{Float64}:
 1.5
 1.5
 2.0
```

See also [`ToroidalShape`](@ref) and, in TensND, `ToroidalChart`.
"""
struct InsulatingTorus{T <: Number, B <: TensND.AbstractBasis} <: Core.AbstractCustomInclusion{T}
    R::T
    r::T
    basis::B
    function InsulatingTorus{T, B}(R, r, basis) where {T <: Number, B <: TensND.AbstractBasis}
        TensND.ToroidalChart(R, r)          # refuses radii that are not those of a torus
        return new{T, B}(R, r, basis)
    end
end

function InsulatingTorus(R::Number, r::Number, basis::B) where {B <: TensND.AbstractBasis}
    T = TensND.float_like(promote_type(typeof(R), typeof(r)))
    return InsulatingTorus{T, B}(convert(T, R), convert(T, r), basis)
end
function InsulatingTorus(
        R::Number, r::Number;
        euler_angles::Tuple{Vararg{Real}} = (), basis::Union{Nothing, TensND.AbstractBasis} = nothing,
    )
    T = TensND.float_like(promote_type(typeof(R), typeof(r)))
    return InsulatingTorus(R, r, _torus_basis(basis, euler_angles, T))
end

_torus_basis(::Nothing, euler_angles, ::Type{T}) where {T} =
    TensND.euler_basis(euler_angles; T = TensND.canonical_eltype(T))
_torus_basis(basis::TensND.AbstractBasis, _, _) = basis

Base.show(io::IO, t::InsulatingTorus{T}) where {T} =
    print(io, "InsulatingTorus{", T, "} (center-line radius R=", t.R, ", tube radius r=", t.r, ")")

Core.dimension(::InsulatingTorus) = 3
Core.inclusion_basis(t::InsulatingTorus) = t.basis
Core.shape_trait(::InsulatingTorus) = ToroidalShape()
# No single property describes a pore: the contributions follow from the two
# concentration tensors, not from `K₁`.
Core.is_homogeneous_inclusion(::InsulatingTorus) = false

function _axis(t::InsulatingTorus)
    e = TensND.vecbasis(t.basis)
    return (e[1, 3], e[2, 3], e[3, 3])
end

# The series solve the problem in an isotropic medium.
_require_isotropic(K₀::TensND.TensISO{2, 3}) = K₀
_require_isotropic(K₀::TensND.AbstractTens{2, 3}) = throw(
    ArgumentError(
        "an InsulatingTorus needs an isotropic reference medium (a TensISO{2,3}), got a " *
            "$(nameof(typeof(K₀))). Register the phase with `symmetrize = IsoSymmetrize()`, " *
            "or give an isotropic reference."
    )
)

"""
    _insulating_torus_concentration(α₀) -> (Aₜ, Aₙ)

Transverse and axial components of the gradient concentration tensor of the
insulating torus `α = α₀`, with `cosh α₀ = R/r`.

A unit remote gradient along the axis is the series of sines of amplitudes
`(4√2/π) n` on `Q_{n-1/2}`, and one across it the series of cosines of
amplitudes `-(2√2/π) εₙ` on `Q¹_{n-1/2}`, with the factor `cos ϕ`. The
perturbations `bₙ` of [`_neumann_amplitudes`](@ref) behave far from the torus as
the field of a dipole, whose moment divided by the volume of the torus is

    Aₙ = (4√2/π) (sinh³α₀/cosh α₀) Σ n bₙ,      Aₜ = (2√2/π) (sinh³α₀/cosh α₀) Σ (n² - 1/4) bₙ.
"""
function _insulating_torus_concentration(α₀)
    N = _series_length(α₀)
    c = 4 * sqrt(oftype(α₀, 2)) / π
    shape = sinh(α₀)^3 / cosh(α₀)
    b = _neumann_amplitudes(Val(:sine), 0, [c * n for n in 0:(N + 1)], α₀)
    Aₙ = c * shape * dot(1:N, b)
    b = _neumann_amplitudes(Val(:cosine), 1, [-(iszero(n) ? c / 2 : c) for n in 0:(N + 1)], α₀)
    Aₜ = c / 2 * shape * dot([n^2 - 1 // 4 for n in 0:N], b)
    return (Aₜ, Aₙ)
end

"""
    Core.gradient_gradient_concentration(torus::InsulatingTorus, K₁, K₀)

Mean temperature gradient over the torus, the boundary values being those of
the matrix, per unit remote gradient. `K₁` is not read; `K₀` must be isotropic.
"""
function Core.gradient_gradient_concentration(
        t::InsulatingTorus, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
    )
    _require_isotropic(K₀)
    Aₜ, Aₙ = _insulating_torus_concentration(acosh(t.R / t.r))
    return TensND.TensTI{2}(Aₜ, Aₙ, _axis(t))
end

"""
    Core.flux_gradient_concentration(torus::InsulatingTorus, K₁, K₀)

Zero: no flux passes through an insulating torus.
"""
function Core.flux_gradient_concentration(
        t::InsulatingTorus{T}, ::TensND.AbstractTens{2, 3}, K₀::TensND.AbstractTens{2, 3}; kw...
    ) where {T}
    _require_isotropic(K₀)
    return TensND.TensISO{3}(zero(promote_type(T, eltype(K₀))))
end

# The elastic counterpart of an insulating torus is a toroidal cavity, which
# this type does not solve.
_no_elasticity() = throw(
    ArgumentError(
        "an InsulatingTorus is an inclusion of the conduction problem: it has no elastic response"
    )
)
Core.gradient_gradient_concentration(
    ::InsulatingTorus, ::TensND.AbstractTens{4, 3}, ::TensND.AbstractTens{4, 3}; kw...
) = _no_elasticity()
Core.flux_gradient_concentration(
    ::InsulatingTorus, ::TensND.AbstractTens{4, 3}, ::TensND.AbstractTens{4, 3}; kw...
) = _no_elasticity()
