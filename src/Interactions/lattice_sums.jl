# =============================================================================
#  lattice_sums.jl — interaction with all the periodic images of a source
#  inclusion, truncated to a cluster of finite radius.
#
#      𝕋̄^{ab} = Σ_{n ∈ Z^d, ‖r_ab + nL‖ ≤ R_c}  𝕋^{ab}(r_ab + nL)
#
#  Convergence.  The sum is only conditionally convergent if taken over all of
#  Z^d, and the summation order then has to be prescribed — this is the
#  difficulty Brisard et al. (2023) discuss around their Eq. (28), where they
#  have to add a heuristic correction to a naive real-space lattice sum.
#
#  The *cluster* truncation used here does not have that problem, and Molinari
#  & El Mouden (1996) prove why in their Appendix B: the interaction kernel
#  integrates to zero over the exterior of any sphere centered on the receiver,
#
#       ∫_{‖x-x_a‖ > R_c} 𝔾⁰(x - x_a) dV = 0 ,
#
#  so the contribution of the images beyond a *spherical* cutoff tends to zero
#  as R_c grows.  Summing over a sphere of images — not over a cube — is
#  therefore not a detail but the reason the truncation is legitimate, and it
#  is why `periodic_images` enumerates a ball rather than a box.
# =============================================================================

"""
    periodic_images(r, L, R_c; skip_self=false) -> Vector

Translations `r + n·L`, `n ∈ Zᵈ`, whose norm is at most `R_c`, for a cubic
period `L`. With `skip_self = true` the null translation is dropped, which is
what a receiver needs when summing over its *own* family of images.

Enumerating a ball rather than a box is what makes the truncated lattice sum
converge — see the file header and
[molinari1996](@cite), App. B.
"""
function periodic_images(r::AbstractVector, L::Real, R_c::Real; skip_self::Bool = false)
    d = length(r)
    nmax = ceil(Int, (R_c + sqrt(sum(abs2, r))) / L) + 1
    out = Vector{typeof(SVector{d}(float.(r)...))}()
    ranges = ntuple(_ -> (-nmax):nmax, d)
    for n in Iterators.product(ranges...)
        skip_self && all(iszero, n) && iszero(sum(abs2, r)) && continue
        x = SVector{d}(ntuple(i -> r[i] + n[i] * L, d)...)
        nrm² = sum(abs2, x)
        iszero(nrm²) && continue
        nrm² ≤ R_c^2 && push!(out, x)
    end
    return out
end

"""
    lattice_interaction_tensor(incl_a, incl_b, r, P₀, L, R_c; kw...) -> AbstractTens

Sum of [`interaction_tensor`](@ref) over every periodic image of the source
inclusion lying within the cluster radius `R_c` of the receiver, for a cubic
cell of side `L`:

```math
\\bar{\\mathbb{T}}^{ab} = \\sum_{\\|r_{ab} + nL\\| \\le R_c}
   \\mathbb{T}^{ab}(r_{ab} + nL) .
```

When `incl_a` and `incl_b` are the *same* inclusion of the cell (`r = 0`), the
null translation is skipped: the self term is not part of this sum, it is
[`self_interaction_tensor`](@ref).

Returns `zero` of the appropriate order when no image falls inside the cutoff,
which is the physically correct answer (a cluster reduced to the receiver
alone) and is what makes the cluster model degenerate exactly onto
Mori-Tanaka.
"""
function lattice_interaction_tensor(
        incl_a, incl_b, r::AbstractVector, P₀::TensND.AbstractTens,
        L::Real, R_c::Real; kw...
    )
    imgs = periodic_images(r, L, R_c; skip_self = true)
    isempty(imgs) && return _zero_interaction(P₀)
    acc = interaction_tensor(incl_a, incl_b, imgs[1], P₀; kw...)
    for k in 2:length(imgs)
        acc = acc + interaction_tensor(incl_a, incl_b, imgs[k], P₀; kw...)
    end
    return acc
end

# A zero of the same tensor order as the reference property.  Built from the
# reference itself rather than from `zero(::TensTI)`, which would narrow the
# symmetry class (`zero(TensTI)` is a `TensISO`) and produce a value that no
# longer accepts a TI update.
_zero_interaction(P₀::TensND.AbstractTens{4, 3}) = zero(TensND.Tens(zeros(eltype(P₀), 3, 3, 3, 3)))
_zero_interaction(P₀::TensND.AbstractTens{4, 2}) = zero(TensND.Tens(zeros(eltype(P₀), 2, 2, 2, 2)))
_zero_interaction(P₀::TensND.AbstractTens{2, 3}) = zero(TensND.Tens(zeros(eltype(P₀), 3, 3)))
_zero_interaction(P₀::TensND.AbstractTens{2, 2}) = zero(TensND.Tens(zeros(eltype(P₀), 2, 2)))
