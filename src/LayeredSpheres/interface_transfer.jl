# =============================================================================
#  interface_transfer.jl — interface transfer matrices for
#  `LayeredSphere`.
#
#  For each interface type, both the bulk (2×2) and shear (4×4) jump
#  matrices are provided.  The perfect-interface limit is the identity;
#  spring and membrane interfaces act in complementary "primal / dual"
#  directions (primal = displacement discontinuous, traction continuous;
#  dual = displacement continuous, traction discontinuous).
# =============================================================================

# ── Bulk (2×2) interface transfer matrices ──────────────────────────────────
#
# State vector: s = (u_r, σ_rr).
#
#   PerfectInterface    : J = I
#   SpringInterface     : J = [1  1/kn ; 0  1]    (bulk uses kn only;
#                                                  kn is a STIFFNESS)
#   MembraneInterface   : J = [1   0 ; 4κs/r²  1] (bulk uses κs only)
#   (Thermal analogs live in `conductivity.jl`.)

"""
    _bulk_interface_T(intf, κ, μ, r) -> Matrix(2×2)

Return the 2×2 jump matrix for the **bulk** (spherical, mode `Y₀`)
state vector `(u_r, σ_rr)` at the interface of type `intf` located at
radius `r` between layers of moduli `(κ, μ)` and `(κ⁺, μ⁺)`.  Only
the interface parameters and the radius enter the bulk jump; the
adjacent layer moduli are provided for type promotion.
"""
function _bulk_interface_T end

# Perfect interface — identity.
function _bulk_interface_T(::PerfectInterface, κ, μ, r)
    T = promote_type(typeof(κ), typeof(μ), typeof(r))
    return T[one(T) zero(T); zero(T) one(T)]
end

# Spring: primal (displacement jump), bulk uses kn only.
function _bulk_interface_T(intf::SpringInterface, κ, μ, r)
    T = promote_type(eltype(intf), typeof(κ), typeof(μ), typeof(r))
    sn, _ = spring_compliances(intf)
    return T[one(T) T(sn); zero(T) one(T)]
end

# Surface-elastic membrane: dual (traction jump), bulk uses κs only.
function _bulk_interface_T(intf::MembraneInterface, κ, μ, r)
    T = promote_type(eltype(intf), typeof(κ), typeof(μ), typeof(r))
    Tr = T(r)
    Tr² = Tr * Tr
    return T[one(T) zero(T); 4 * T(intf.κs) / Tr² one(T)]
end

# ── Shear (4×4) interface transfer matrices ──────────────────────────────────
#
# State vector: S = (U, V, σ_rr, σ_rθ).
# These are imposed at the radius r_k of the interface.  Perfect = I.
#
# For SpringInterface the displacement components jump while tractions are
# continuous.  `kn`, `kt` are STIFFNESSES, so the jump carries their inverse:
#   U⁺ = U⁻ + σ_rr⁻ / kn
#   V⁺ = V⁻ + σ_rθ⁻ / kt
# In matrix form:
#   J = [ 1  0  1/kn  0   ;
#         0  1  0     1/kt;
#         0  0  1     0   ;
#         0  0  0     1   ]
#
# For MembraneInterface the traction components jump:
#   σ_rr⁺ = σ_rr⁻  +  f1(κs, μs, r) · U  +  f2(κs, μs, r) · V
#   σ_rθ⁺ = σ_rθ⁻  +  f3(κs, μs, r) · U  +  f4(κs, μs, r) · V
# The exact coefficients come from the divergence of the 2D surface
# stress on a sphere for the `Y₂`-mode (surface-elasticity model with
# both dilatation and shear surface stiffness).  In matrix form J is a
# 4×4 block-triangular with the linear combination in the lower-left
# 2×2 block.

"""
    _shear_interface_T(intf, κ, μ, r) -> Matrix(4×4)

Return the 4×4 jump matrix for the **shear** (deviatoric, mode `Y₂`)
state vector `(U, V, σ_rr/μ_ref, σ_rθ/μ_ref)` at the interface at
radius `r`.
"""
function _shear_interface_T end

function _shear_interface_T(::PerfectInterface, κ, μ, r)
    T = promote_type(typeof(κ), typeof(μ), typeof(r))
    return Matrix{T}(LinearAlgebra.I, 4, 4)
end

function _shear_interface_T(intf::SpringInterface, κ, μ, r)
    T = promote_type(eltype(intf), typeof(κ), typeof(μ), typeof(r))
    M = Matrix{T}(LinearAlgebra.I, 4, 4)
    # State order: (U, W, σ_rr, σ_rθ).  U jumps by σ_rr/kn, W by σ_rθ/kt.
    sn, st = spring_compliances(intf)
    M[1, 3] = T(sn)
    M[2, 4] = T(st)
    return M
end

function _shear_interface_T(intf::MembraneInterface, κ, μ, r)
    T = promote_type(eltype(intf), typeof(κ), typeof(μ), typeof(r))
    M = Matrix{T}(LinearAlgebra.I, 4, 4)
    κs = T(intf.κs); μs = T(intf.μs)
    Tr = T(r)
    inv_r² = one(T) / (Tr * Tr)
    # Gurtin-Murdoch surface elasticity, Y₂-harmonic traction jump on a
    # sphere of radius r.  From the generalized Young-Laplace balance
    # [σ·n] = −divₛ σˢ with σˢ = λₛ(trₛ ε)𝟏 + 2μₛεˢ and the surface strains
    # of u_r = U P₂, u_θ = W dP₂/dθ (derived symbolically, see the theory
    # page).  In MFH's `κs = λₛ + μₛ` convention (matching Echoes' `ks`):
    #   [σ_rr] = ( 4κs U − 12κs W) / r²
    #   [σ_rθ] = (−2κs U + (6κs + 4μs) W) / r²
    M[3, 1] = 4 * κs * inv_r²
    M[3, 2] = -12 * κs * inv_r²
    M[4, 1] = -2 * κs * inv_r²
    M[4, 2] = (6 * κs + 4 * μs) * inv_r²
    return M
end
