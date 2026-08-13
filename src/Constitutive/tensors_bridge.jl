# =============================================================================
#  tensors_bridge.jl — TensND ↔ Tensors.jl conversions.
#
#  Finite-element codes in the Julia ecosystem speak `Tensors.jl`: Ferrite is
#  built on it, and its shape-function gradients, cell values and assemblers all
#  produce and consume `SymmetricTensor{2,3}` / `SymmetricTensor{4,3}`. MFH
#  speaks TensND, which carries a *basis* alongside the components.
#
#  The whole bridge rests on one rule, and getting it wrong is the single most
#  likely source of silent errors in a coupling:
#
#      A `Tensors.jl` tensor has NO basis. It is a bare component array, and in
#      an FE code those components are always in the GLOBAL frame.
#
#  A TensND tensor, in contrast, returns its components in its OWN basis from
#  both `get_array` and `getindex`. A homogenized stiffness whose RVE holds
#  tilted crack families comes back in a rotated basis; handing its raw
#  components to an FE assembler would rotate the material by an arbitrary
#  amount with no error and no warning. Every conversion below therefore goes
#  through the canonical basis explicitly.
# =============================================================================

"""
    to_tensors(t::AbstractTens{2,3}) -> Tensors.SymmetricTensor{2,3}
    to_tensors(t::AbstractTens{4,3}) -> Tensors.SymmetricTensor{4,3}

Convert a TensND tensor to its `Tensors.jl` counterpart, **expressed in the
global (canonical) frame**.

This is the direction an FE code consumes: a stress, or a tangent stiffness,
ready to be contracted with shape-function gradients.

!!! warning "Never bypass this with get_array"
    `get_array(t)` and `t[i,j,k,l]` return the components of `t` in *its own*
    basis. Any homogenized property whose RVE carries oriented inclusions —
    tilted crack families, in particular — is returned in a rotated basis, so
    those components are **not** the global ones. `to_tensors` performs the
    change of basis; reading the array directly silently rotates the material.

The 4th-order conversion assumes minor symmetry, which every stiffness,
compliance and localization tensor in the package satisfies. Major symmetry is
*not* assumed: `Tensors.SymmetricTensor{4,3}` carries 36 independent components
and can hold a concentration tensor unchanged.
"""
function to_tensors(t::TensND.AbstractTens{2, 3})
    A = get_array(TensND.change_tens(t, _canonical_like(t)))
    return Tensors.SymmetricTensor{2, 3}((i, j) -> (A[i, j] + A[j, i]) / 2)
end

function to_tensors(t::TensND.AbstractTens{4, 3})
    A = get_array(TensND.change_tens(t, _canonical_like(t)))
    return Tensors.SymmetricTensor{4, 3}((i, j, k, l) -> A[i, j, k, l])
end

"""
    from_tensors(t::Tensors.SymmetricTensor{2,3}) -> Tens{2,3}
    from_tensors(t::Tensors.SymmetricTensor{4,3}) -> Tens{4,3}

Convert a `Tensors.jl` tensor coming from an FE code into a TensND tensor in
the **canonical basis** — which is what its components mean.

This is the direction an FE code produces: the strain at a quadrature point,
assembled from shape-function gradients in the global frame.
"""
from_tensors(t::Tensors.SymmetricTensor{2, 3}) = TensND.Tens(t)
from_tensors(t::Tensors.SymmetricTensor{4, 3}) = TensND.Tens(t)

# A canonical basis carrying the tensor's own element type, so the conversion
# does not silently narrow a `Dual` or widen a `Float64`.
_canonical_like(t::TensND.AbstractTens) =
    TensND.CanonicalBasis{3, eltype(t)}()

"""
    voigt_stress(σ) -> NTuple{6}
    voigt_strain(ε) -> NTuple{6}

Components of a symmetric 2nd-order tensor in the **Voigt** convention used by
Abaqus-style interfaces, ordered `(11, 22, 33, 12, 13, 23)`, in the global
frame.

The two differ by the engineering-shear factor: `voigt_strain` doubles the
off-diagonal terms (`γ₁₂ = 2ε₁₂`), `voigt_stress` does not. Keeping them as
separate functions is deliberate — a single `voigt` helper is the classic way to
lose a factor of two between the strain that goes in and the stress that comes
out.

Note this is *not* the Kelvin-Mandel convention used internally by TensND and
`Tensors.jl` (which carries `√2` instead, and is an isometry); Voigt appears
here only at the boundary with codes that demand it, Abaqus-style UMATs above
all.
"""
function voigt_stress(σ::TensND.AbstractTens{2, 3})
    A = get_array(TensND.change_tens(σ, _canonical_like(σ)))
    return (A[1, 1], A[2, 2], A[3, 3], A[1, 2], A[1, 3], A[2, 3])
end

function voigt_strain(ε::TensND.AbstractTens{2, 3})
    A = get_array(TensND.change_tens(ε, _canonical_like(ε)))
    return (A[1, 1], A[2, 2], A[3, 3], 2A[1, 2], 2A[1, 3], 2A[2, 3])
end

# The docstring above covers both functions; bind it to the second one as well
# so `@ref` and the `@docs` block resolve it.
@doc (@doc voigt_stress) voigt_strain

"""
    strain_from_voigt(v) -> Tens{2,3}

Inverse of [`voigt_strain`](@ref): rebuild a strain tensor from the six Voigt
components `(ε₁₁, ε₂₂, ε₃₃, γ₁₂, γ₁₃, γ₂₃)`, halving the engineering shears.
"""
function strain_from_voigt(v)
    T = float(eltype(v))
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> i == j ? T(v[i]) :
                (i, j) == (1, 2) || (i, j) == (2, 1) ? T(v[4]) / 2 :
                (i, j) == (1, 3) || (i, j) == (3, 1) ? T(v[5]) / 2 : T(v[6]) / 2
        )
    )
end

"""
    stress_from_voigt(v) -> Tens{2,3}

Inverse of [`voigt_stress`](@ref): rebuild a stress tensor from
`(σ₁₁, σ₂₂, σ₃₃, σ₁₂, σ₁₃, σ₂₃)`, with no shear factor.
"""
function stress_from_voigt(v)
    T = float(eltype(v))
    return TensND.Tens(
        Tensors.SymmetricTensor{2, 3}(
            (i, j) -> i == j ? T(v[i]) :
                (i, j) == (1, 2) || (i, j) == (2, 1) ? T(v[4]) :
                (i, j) == (1, 3) || (i, j) == (3, 1) ? T(v[5]) : T(v[6])
        )
    )
end
