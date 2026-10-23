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
     ``M_{\\mathrm{shear}} = \\beta/2`` ``n\\times n`` matrices: the two eigenvalues
     of the acoustic tensor, which the Hill tensor inverts. The bulk and shear
     kernels follow linearly, ``k = M_{\\mathrm{long}} - \\tfrac{4}{3} M_{\\mathrm{shear}}``
     and ``\\mu = M_{\\mathrm{shear}}``; their inverses do not (see the theory page).
  4. Take the scalar Volterra inverses
     ``J_{\\mathrm{long}} = M_{\\mathrm{long}}^{-\\circ}`` and ``J_{\\mathrm{shear}} = M_{\\mathrm{shear}}^{-\\circ}``.
  5. Compute the Mandel forms ``\\mathrm{Mat}(\\mathbb{U}^{\\boldsymbol{A}})`` and
     ``\\mathrm{Mat}(\\mathbb{V}^{\\boldsymbol{A}})`` of the elastic
     auxiliary tensors `tens_UA(ell)` and `tens_VA(ell)`.
  6. Assemble block-by-block:
     ``[\\widetilde{\\mathbb{P}}]_{ij} = (J_{\\mathrm{long}})_{ij}\\,\\mathrm{Mat}(\\mathbb{U}^{\\boldsymbol{A}}) + (J_{\\mathrm{shear}})_{ij}\\,\\mathrm{Mat}(\\mathbb{V}^{\\boldsymbol{A}} - \\mathbb{U}^{\\boldsymbol{A}})``.
"""
function hill_kernel(ell, C0_law::ViscoLaw, times::AbstractVector{<:Real})
    # 1. Build the 6n×6n RELAXATION matrix of the iso reference (invert
    #    the trapezoidal compliance if the law is in `:creep` mode).
    R_M = trapezoidal_matrix(C0_law, times)
    if visco_mode(C0_law) === :creep
        R_M = volterra_inverse(R_M; block_size = 6)
    end
    # 2-4. The longitudinal and shear Volterra inverses of the reference.
    J_long, J_shear = _alv_iso_inverses(R_M)
    # 5. Geometric 4-tensors of the inclusion in Mandel form.
    U_M = TensND.mandel66(tens_UA(ell))
    D_M = TensND.mandel66(tens_VA(ell)) - U_M
    # 6. Assemble.
    return _alv_hill_blocks(J_long, J_shear, U_M, D_M)
end

# The longitudinal ``(\alpha + 2\beta)/3`` and shear ``\beta/2`` Volterra
# inverses of an isotropic block matrix: what its Hill kernel is made of.
function _alv_iso_inverses(C::AbstractMatrix)
    α, β = iso_params_from_blocks(C)
    return volterra_inverse(@.((α + 2 * β) / 3); block_size = 1), volterra_inverse(β ./ 2; block_size = 1)
end

# The ALV Hill kernel of an ellipsoid of geometry tensors `U_M`, `D_M = V_M - U_M`
# against an isotropic reference: block (i, j) is J_long[i, j] U_M + J_shear[i, j] D_M.
# Both inverses are lower triangular, so the blocks above the diagonal vanish.
function _alv_hill_blocks(J_long, J_shear, U_M, D_M)
    P = kron(J_long, U_M)
    P .+= kron(J_shear, D_M)
    return P
end
