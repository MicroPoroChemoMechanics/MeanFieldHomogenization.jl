# =============================================================================
#  plane.jl — a crack of a two-dimensional cell: the trace of a `RibbonCrack`
#  in the plane normal to the direction it is unbounded along, a segment of
#  half-length b.
#
#  The properties are tensors of the plane. The frame (ℓ, m, n) of the ribbon
#  has ℓ = e₁, the normal to the plane; the plane (e₂, e₃) is that of the cell,
#  whose axes are numbered 1 and 2. The algebra is that of the ribbon: the
#  density N b², the factor π of the dilute correction, ℍ = (2/π) n ⊗ˢ 𝐁 ⊗ˢ n.
# =============================================================================

const _Plane4 = TensND.AbstractTens{4, 2}
const _Plane2 = TensND.AbstractTens{2, 2}

_not_a_plane_crack(crack) = throw(
    ArgumentError(
        "in a two-dimensional cell a crack is a segment, the trace of a ribbon crack unbounded " *
            "along the normal to the plane: `RibbonCrack(b; angle)`. Got a $(nameof(typeof(crack)))" *
            _other_direction(crack)
    )
)
_other_direction(::RibbonCrack) = " that is unbounded along another direction"
_other_direction(crack) = ""

# Normal of the segment, in the axes of the plane.
function _plane_normal(crack::RibbonCrack)
    e = TensND.vecbasis(crack_basis(crack))
    TensND.isparallel((e[1, 1], e[2, 1], e[3, 1]), (1, 0, 0)) || _not_a_plane_crack(crack)
    return SVector(e[2, 3], e[3, 3])
end

# Direction of the segment, in the axes of the plane: the tip is at ±b along it.
function _plane_direction(crack::RibbonCrack)
    e = TensND.vecbasis(crack_basis(crack))
    return SVector(e[2, 2], e[3, 2])
end

# A tensor of the plane as that of a medium invariant along e₁, in plane strain:
# the block of the plane, and out-of-plane moduli that do not enter the problem
# of the plane (no coupling with the axis, shear moduli along it of the same
# order as the others).
function _space_stiffness(c::_Plane4)
    M = TensND.mandel(c, TensND.CanonicalBasis{2, eltype(c)}())
    s = (M[1, 1] + M[2, 2]) / 2
    o = zero(s)
    return TensND.from_mandel(
        SMatrix{6, 6}(
            s, o, o, o, o, o,
            o, M[1, 1], M[2, 1], M[3, 1], o, o,
            o, M[1, 2], M[2, 2], M[3, 2], o, o,
            o, M[1, 3], M[2, 3], M[3, 3], o, o,
            o, o, o, o, s, o,
            o, o, o, o, o, s,
        )
    )
end
function _space_conductivity(κ::_Plane2)
    K = TensND.canonical_components(κ)
    s = (K[1, 1] + K[2, 2]) / 2
    o = zero(s)
    return TensND.Tens(SMatrix{3, 3}(s, o, o, o, K[1, 1], K[2, 1], o, K[1, 2], K[2, 2]))
end

# 𝐁 of the segment. In an isotropic plane, 2k 𝕁 + 2μ 𝕂, it is χ 𝟏 with
# χ = π (κ + 1)/(8μ) = (π/4)(1/μ + 1/k): π (1 − ν²)/E in plane strain, π/E in
# plane stress. In an anisotropic plane it is the block of the plane of the 𝐁
# of the ribbon in the medium invariant along the axis.
function _plane_cod(crack::RibbonCrack, c₀::TensND.TensISO{4, 2}; kw...)
    _plane_normal(crack)
    α, β = TensND.get_data(c₀)
    T = promote_type(typeof(α), typeof(β), typeof(crack.b))
    return TensND.TensISO{2}(T(π) / 2 * (inv(α) + inv(β)))
end
function _plane_cod(crack::RibbonCrack, c₀::_Plane4; kw...)
    _plane_normal(crack)
    B = TensND.canonical_components(cod_tensor(crack, _space_stiffness(c₀); kw...))
    return TensND.Tens(SMatrix{2, 2}(B[2, 2], B[3, 2], B[2, 3], B[3, 3]))
end
_plane_cod(crack::MFH_Core.AbstractCrack, c₀::_Plane4; kw...) = _not_a_plane_crack(crack)

"""
    cod_tensor(crack::RibbonCrack, c₀::AbstractTens{4,2}; K_interface = nothing, kw...) -> AbstractTens{2,2}
    cod_tensor(crack::RibbonCrack, κ₀::AbstractTens{2,2}; α_interface = nothing) -> Real

The same tensor for a crack of a two-dimensional cell, the segment of
half-length ``b`` that a ribbon crack unbounded along ``\\underline{e}_1``
traces in the plane (`RibbonCrack(b; angle)`), in a reference given by a tensor
of the plane. In the isotropic plane ``2k\\,\\mathbb J + 2\\mu\\,\\mathbb K``,

```math
\\boldsymbol{B} = \\frac{\\pi}{4}\\Bigl(\\frac1\\mu + \\frac1k\\Bigr)\\,\\boldsymbol{1} ,
```

that is ``\\pi(1 - \\nu^2)/E`` in plane strain and ``\\pi/E`` in plane stress; in
conduction ``b = \\pi/(2k_0)``. In a plane that is not isotropic the tensor is
computed as for the ribbon in a medium invariant along its axis, with numbers.
The contributions ([`inverse_property_contribution`](@ref),
`property_contribution`) and the density ``N b^2`` are those of the ribbon.
"""
function cod_tensor(
        crack::MFH_Core.AbstractCrack, c₀::_Plane4;
        K_interface::Union{Nothing, _Plane2} = nothing, kw...
    )
    MFH_Core._bump!(MFH_Core.COD_CALLS)
    B = _plane_cod(crack, c₀; kw...)
    K_interface === nothing && return B
    return _apply_interface_stiffness(B, K_interface, semi_minor(crack))
end

function inverse_property_contribution(
        crack::MFH_Core.AbstractCrack, c₀::_Plane4;
        K_interface::Union{Nothing, _Plane2} = nothing, kw...
    )
    return _plane_compliance(crack, cod_tensor(crack, c₀; K_interface, kw...))
end

# ℍ = (2/π) n ⊗ˢ 𝐁 ⊗ˢ n on the components of the plane.
function _plane_compliance(crack, B)
    n = _plane_normal(crack)
    b = TensND.canonical_components(B)
    k = _H_over_B_factor(crack, eltype(b)) / 4
    return TensND.Tens(
        MFH_Core._sfill(
            (i, j, p, q) -> k * (n[i] * b[j, p] * n[q] + n[j] * b[i, p] * n[q] + n[i] * b[j, q] * n[p] + n[j] * b[i, q] * n[p]),
            Val((2, 2, 2, 2)),
        )
    )
end

function inverse_and_property_contribution(
        crack::MFH_Core.AbstractCrack, c₀::_Plane4;
        K_interface::Union{Nothing, _Plane2} = nothing, kw...
    )
    H = inverse_property_contribution(crack, c₀; K_interface, kw...)
    return (H, -(c₀ ⊡ H ⊡ c₀))
end

# ── Conduction ───────────────────────────────────────────────────────────────

_plane_thermal_cod(crack::RibbonCrack, κ₀::TensND.TensISO{2, 2}) =
    (_plane_normal(crack); _cod_iso_ribbon_thermal(crack, first(TensND.get_data(κ₀))))
_plane_thermal_cod(crack::RibbonCrack, κ₀::_Plane2) =
    (_plane_normal(crack); _cod_aniso_ribbon_thermal(crack, _space_conductivity(κ₀)))
_plane_thermal_cod(crack::MFH_Core.AbstractCrack, κ₀::_Plane2) = _not_a_plane_crack(crack)

function cod_tensor(
        crack::MFH_Core.AbstractCrack, κ₀::_Plane2;
        α_interface::Union{Nothing, Real} = nothing, kw...
    )
    b = _plane_thermal_cod(crack, κ₀)
    α_interface === nothing && return b
    return b / (1 + semi_minor(crack) * α_interface * b)
end

# 𝐑 = (2/π) b n ⊗ n.
function inverse_property_contribution(crack::MFH_Core.AbstractCrack, κ₀::_Plane2; kw...)
    b = cod_tensor(crack, κ₀; kw...)
    n = _plane_normal(crack)
    k = _H_over_B_factor(crack, typeof(b)) * b
    return TensND.Tens(SMatrix{2, 2}(k * n[1] * n[1], k * n[2] * n[1], k * n[1] * n[2], k * n[2] * n[2]))
end

function inverse_and_property_contribution(crack::MFH_Core.AbstractCrack, κ₀::_Plane2; kw...)
    R = inverse_property_contribution(crack, κ₀; kw...)
    return (R, -(κ₀ ⋅ R ⋅ κ₀))
end

# ── Intensity factors ────────────────────────────────────────────────────────

"""
    sif(crack::RibbonCrack, C₀::AbstractTens{4,2}, Σ::AbstractTens{2,2}) -> (𝐊, (Kᴵ, Kᴵᴵ))
    sif(crack::RibbonCrack, K₀::AbstractTens{2,2}, σ∞::AbstractTens{1,2}) -> Real

The intensity factors of a segment of half-length `b` in a two-dimensional
cell, ``\\underline{K} = \\sqrt{\\pi b}\\,\\boldsymbol{\\Sigma}\\cdot\\underline{n}``
with its opening and sliding modes on the normal and on the direction of the
segment, and, in conduction,
``K_T = \\sqrt{\\pi b}\\;\\underline{n}\\cdot\\boldsymbol{\\sigma}^{\\infty}``. They do not
depend on the properties of the medium, which only says that the problem is
that of the plane.
"""
function sif(crack::RibbonCrack, ::_Plane4, Σ::_Plane2; kw...)
    n, m = _plane_normal(crack), _plane_direction(crack)
    𝐊 = sqrt(oftype(float(crack.b), π) * crack.b) * (SMatrix{2, 2}(TensND.canonical_components(Σ)) * n)
    return TensND.Tens(𝐊), (transpose(𝐊) * n, transpose(𝐊) * m)
end
function sif(crack::RibbonCrack, ::_Plane2, σ∞::TensND.AbstractTens{1, 2}; kw...)
    n = _plane_normal(crack)
    return sqrt(oftype(float(crack.b), π) * crack.b) * (transpose(n) * SVector{2}(TensND.canonical_components(σ∞)))
end
