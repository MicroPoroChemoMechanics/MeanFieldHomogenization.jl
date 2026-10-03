# =============================================================================
#  hill_alv.jl — discrete ALV Hill polarization tensor kernel.
#
#  For an ellipsoidal inclusion in an **isotropic** ALV matrix, the
#  Hill kernel admits the time-space decoupling formula
#
#      P̃_E(t,t') = ⟨k + 4μ/3⟩^{-vol}(t,t') · U^A
#                + ⟨μ⟩^{-vol}(t,t')        · (V^A − U^A)
#
#  where `U^A` and `V^A` are the purely geometric 4-tensors of the
#  inclusion (already implemented as `tens_UA`, `tens_VA` in the
#  `Elasticity` sub-module) and the scalar Volterra inverses are
#  computed by `volterra_inverse` on the trapezoidal matrices of
#  `(k(t,t') + 4μ(t,t')/3)` and `μ(t,t')` respectively.
#
#  Reference: appendix of the ECHOES manual, "ALV Hill (polarization)
#  tensor kernel" (`viscoelastic_hill_kernel.qmd`),
#  [@barthelemyIJSS2016, §4 ; @barthelemyIJES2019, App. A].
# =============================================================================

"""
    hill_kernel(ell::AbstractEllipsoidalInclusion,
                C0_law::ViscoLaw,
                times::AbstractVector{<:Real}) -> Matrix

Discrete ALV Hill polarization tensor kernel for the inclusion `ell`
in an **isotropic** ALV matrix described by `C0_law` (a `ViscoLaw`
returning a `TensND.TensISO{4,3}` 4-tensor at each ``(t, t')``).

The output is a ``6n\\times 6n`` lower-block-triangular `Matrix{T}` with
``n`` = `length(times)`, in the same Mandel block convention as
`trapezoidal_matrix`.

Implementation follows the time-space decoupling formula of the
ECHOES manual appendix `viscoelastic_hill_kernel.qmd`:

  1. Discretize the matrix kernel `C0_law` on `times` to a ``6n\\times 6n``
     block matrix ``\\widetilde{\\mathbb{C}}_0``.
  2. Extract the iso scalar parameter matrices ``(\\alpha, \\beta)`` of ``\\widetilde{\\mathbb{C}}_0``
     (``\\alpha = 3k``, ``\\beta = 2\\mu`` per Mandel convention).
  3. Build the longitudinal ``M_{\\mathrm{long}} = (\\alpha + 2\\beta)/3`` and the shear
     ``M_{\\mathrm{shear}} = \\beta/2`` ``n\\times n`` matrices.
  4. Take the scalar Volterra inverses
     ``J_{\\mathrm{long}} = M_{\\mathrm{long}}^{-\\circ}`` and ``J_{\\mathrm{shear}} = M_{\\mathrm{shear}}^{-\\circ}``.
  5. Compute the Mandel forms ``\\mathrm{Mat}(\\mathbb{U}^{\\boldsymbol{A}})`` and
     ``\\mathrm{Mat}(\\mathbb{V}^{\\boldsymbol{A}})`` of the elastic
     auxiliary tensors `tens_UA(ell)` and `tens_VA(ell)`.
  6. Assemble block-by-block:
     ``[\\widetilde{\\mathbb{P}}]_{ij} = (J_{\\mathrm{long}})_{ij}\\,\\mathrm{Mat}(\\mathbb{U}^{\\boldsymbol{A}}) + (J_{\\mathrm{shear}})_{ij}\\,\\mathrm{Mat}(\\mathbb{V}^{\\boldsymbol{A}} - \\mathbb{U}^{\\boldsymbol{A}})``.
"""
function hill_kernel(ell, C0_law::ViscoLaw, times::AbstractVector{<:Real})
    n = length(times)
    # 1. Build the 6n×6n RELAXATION matrix of the iso reference (invert
    #    the trapezoidal compliance if the law is in `:creep` mode).
    R_M = trapezoidal_matrix(C0_law, times)
    if visco_mode(C0_law) === :creep
        R_M = volterra_inverse(R_M; block_size = 6)
    end
    # 2-3. Extract iso parameters (α=3K, β=2μ) and build longitudinal / shear.
    α, β = iso_params_from_blocks(R_M)
    M_long = @. (α + 2 * β) / 3      # = K + 4μ/3
    M_shear = β ./ 2                  # = μ
    # 4. Scalar Volterra inverses (n×n each).
    J_long = volterra_inverse(M_long; block_size = 1)
    J_shear = volterra_inverse(M_shear; block_size = 1)
    # 5. Geometric 4-tensors of the inclusion in Mandel form.
    U_A = tens_UA(ell)
    V_A = tens_VA(ell)
    U_M = _tens_to_mandel66(U_A)
    V_M = _tens_to_mandel66(V_A)
    D_M = V_M - U_M
    # 6. Assemble.
    T = promote_type(eltype(J_long), eltype(J_shear), eltype(U_M))
    P = zeros(T, 6 * n, 6 * n)
    @inbounds for i in 1:n, j in 1:i
        block = J_long[i, j] .* U_M .+ J_shear[i, j] .* D_M
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        P[rows, cols] = block
    end
    return P
end
