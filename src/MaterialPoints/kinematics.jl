# =============================================================================
#  kinematics.jl — what a law receives: the small-strain tensor, in Kelvin–Mandel
#  components.
#
#  A law is evaluated millions of times in a structural computation, inside a
#  local Newton loop and under automatic differentiation. Its strain and stress
#  are therefore plain static vectors of six numbers, not TensND tensors: the
#  Kelvin–Mandel basis is orthonormal, so the double contraction σ:ε is the dot
#  product of the two vectors, the norm of a tensor is the norm of its vector,
#  and a fourth-order tensor with the minor symmetries is a 6×6 matrix whose
#  products are the tensor products. Nothing in a law has to know about
#  factors of 2 on the shears.
#
#  The conversion to and from TensND tensors happens once, at the boundary
#  (`material_response`), through the conversions of Tensors.jl.
# =============================================================================

"""
    SmallStrain()

The kinematics of a law written in infinitesimal strain: the law receives the
symmetric strain tensor ``\\boldsymbol{\\varepsilon}`` and returns the Cauchy
stress ``\\boldsymbol{\\sigma}``. It is the only kinematics so far; the trait
[`kinematics`](@ref) exists so that a finite-strain one can be added without
changing the laws written for this one.
"""
struct SmallStrain end

"""
    Mandel{T}

A symmetric second-order tensor of dimension 3 in **Kelvin–Mandel** components,
an `SVector{6, T}` ordered

```math
(v_{11},\\ v_{22},\\ v_{33},\\ \\sqrt2\\,v_{23},\\ \\sqrt2\\,v_{13},\\ \\sqrt2\\,v_{12}).
```

The basis is orthonormal: ``\\boldsymbol{\\sigma}:\\boldsymbol{\\varepsilon} =
\\boldsymbol{\\sigma}\\cdot\\boldsymbol{\\varepsilon}`` (`dot(σ, ε)`), the norm of
the tensor is `norm(v)`, and a fourth-order tensor with the minor symmetries is
an `SMatrix{6, 6}` acting by the matrix product. The laws of
[`MaterialPoints`](@ref MeanFieldHomogenization.MaterialPoints) are written on
these vectors.
"""
const Mandel{T} = SVector{6, T}

"""
    identity_mandel(T = Float64) -> Mandel{T}

The second-order identity ``\\boldsymbol{1}``, in Kelvin–Mandel components.
"""
identity_mandel(::Type{T} = Float64) where {T} = Mandel{T}(one(T), one(T), one(T), zero(T), zero(T), zero(T))

"""
    trace_mandel(v::Mandel)

The trace ``\\operatorname{tr}\\boldsymbol{v} = \\boldsymbol{v} : \\boldsymbol{1}``.
"""
trace_mandel(v::Mandel) = v[1] + v[2] + v[3]

"""
    deviator(v::Mandel) -> Mandel

The deviatoric part ``\\boldsymbol{v} - \\tfrac13 (\\operatorname{tr}\\boldsymbol{v})\\,\\boldsymbol{1}``.
"""
deviator(v::Mandel) = v - (trace_mandel(v) / 3) * identity_mandel(eltype(v))

"""
    von_mises(σ::Mandel)

The von Mises equivalent stress ``\\sigma_{eq} = \\sqrt{\\tfrac32\\,\\boldsymbol{s}:\\boldsymbol{s}}``,
``\\boldsymbol{s}`` being the deviator of ``\\boldsymbol{\\sigma}``: equal to
``|\\sigma|`` in uniaxial tension.

!!! note "Not differentiable at zero"
    The square root has no derivative at a vanishing deviator. A law whose
    response must be differentiated there writes its flow rule with
    ``\\sigma_{eq}^2`` instead, as [`NortonViscoplasticity`](@ref) does.
"""
von_mises(σ::Mandel) = sqrt(3 * _sqnorm_deviator(σ) / 2)

"""
    equivalent_strain(ε::Mandel)

The equivalent strain ``\\varepsilon_{eq} = \\sqrt{\\tfrac23\\,\\boldsymbol{e}:\\boldsymbol{e}}``,
``\\boldsymbol{e}`` being the deviator of ``\\boldsymbol{\\varepsilon}``: work-conjugate
to [`von_mises`](@ref), and equal to ``|\\varepsilon|`` for an incompressible
uniaxial strain.
"""
equivalent_strain(ε::Mandel) = sqrt(2 * _sqnorm_deviator(ε) / 3)

# The squared norm of the deviator, from the differences of the normal
# components: exactly zero for a change of volume, whatever the rounding, where
# subtracting the mean would leave a few ulps (and a law with a power of the
# deviatoric strain would take them for a deviatoric strain, or produce 0 × ∞
# in its derivatives).
_sqnorm_deviator(v::Mandel) =
    ((v[1] - v[2])^2 + (v[2] - v[3])^2 + (v[3] - v[1])^2) / 3 + v[4]^2 + v[5]^2 + v[6]^2

# ── The boundary with TensND ─────────────────────────────────────────────────
# Components in the canonical basis (`to_tensors`), whatever basis the tensor is
# written in; then Tensors.jl's Mandel ordering, which is the one above.

_mandel(t::TensND.AbstractTens{2, 3}) = Tensors.tomandel(SArray, to_tensors(t))
_mandel(t::TensND.AbstractTens{4, 3}) = Tensors.tomandel(SArray, to_tensors(t))

_tensor(v::StaticVector{6}) = TensND.from_mandel(v)
_tensor(A::StaticMatrix{6, 6}) = TensND.from_mandel(A)
