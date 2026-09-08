# =============================================================================
#  cell_driver.jl — the corrected solve on the three-dimensional cell.
#
#  One assembly, one factorization, and 2n right-hand sides: n remote loadings
#  and n unit dipoles, with n = 3 in conduction and 6 in elasticity.  Then one
#  linear closure.  Nothing here mentions a finite-element library.
#
#  ## The closure is one step, not an iteration
#
#  The exact infinite-medium field is `u = E·x + ∫∇G:π`, whose far field
#  collapses onto a single dipole.  Putting that dipole into the boundary data
#  of a finite cell removes the O((a/R)³) truncation bias — at the price of the
#  dipole being an *output* of the problem.  Linearity closes it in one step:
#  solve two families on the same mesh,
#
#      family s:  boundary data = the remote field       → L_s
#      family u:  boundary data = a unit-moment dipole   → L_u
#
#  with `L` the measured localization.  The true solution satisfies
#  `L = L_s + L_u·M` and `M = F·L`, hence
#
#      L = (𝕀 − L_u·F)⁻¹ L_s .
#
#  This is the **pore declination** of the general polarization fixed point of
#  `docs/src/theory/corrected_cell.md`.  It is simpler here because a cavity
#  carries no stress, so its polarization is proportional to the very quantity
#  being measured: the one number the solve produces is at once the answer and
#  the source of its own boundary correction.
#
#  ## Two sign conventions, one rule
#
#  The factor `F` mapping the measured localization to the polarization moment
#  is `-V�ℂ₀` in elasticity and `-k₀V` in conduction, and the apparent
#  inconsistency is structural.  Writing `L` for the operator each Green
#  function inverts,
#
#      elasticity   L u = -div(ℂ₀:ε(u)),  div σ = 0, σ = ℂ₀:ε + π  ⟹  L u = +div π
#      conduction   L T = -div(k₀∇T),     div q = 0, q = -k₀∇T + π ⟹  L T = -div π
#
#  because `q = -k₀∇T` carries a minus where `σ = +ℂ₀:ε` does not: the object
#  playing the role of the stress in conduction is `-q`, not `q`.  Splitting
#  `-q = k₀∇T + π′` gives `π′ = -π`, and it is `π′` that pairs with
#  `T = ∇G·M`.
#
#  The sign was measured before it was explained, and it is worth keeping a test
#  that flips it: a wrong sign leaves the *corrected* answer carrying exactly
#  **twice** the truncation bias instead of none, which reads as a mesh that
#  will not converge rather than as an algebra mistake.
# =============================================================================

"""
    _cell_close_dipole(L_s, L_u, F) -> (L, dipole_norm)

Superpose the remote and unit-dipole families: `L = (𝕀 − L_u F)⁻¹ L_s`.

`dipole_norm = ‖L_u F‖` is the diagnostic to watch rather than the answer: the
dipole term is ``O((a/R)^3)``, so a log-log slope of ``-3`` against `R` says the
correction is doing what it claims.
"""
function _cell_close_dipole(L_s::AbstractMatrix, L_u::AbstractMatrix, F::AbstractMatrix)
    B = L_u * F
    return (LinearAlgebra.I - B) \ L_s, LinearAlgebra.opnorm(B)
end

_cell_close_dipole(L_s::AbstractMatrix, L_u::AbstractMatrix, f::Real) =
    _cell_close_dipole(L_s, L_u, f * Matrix(1.0LinearAlgebra.I, size(L_u, 2), size(L_u, 2)))

"""
    _cell_solver(backend, space, K)

Close over one factorization of the free-free block so that every right-hand
side is a back-substitution. The prescribed dofs are eliminated by hand rather
than applied into the matrix, which would consume it.
"""
function _cell_solver(backend, space, K, χ = nothing)
    ndofs, free, presc = fe_cell_dof_split(backend, space, χ)
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K[free, free]))
    Kfp = K[free, presc]
    return function (f)
        u = zeros(ndofs)
        fe_cell_set_dirichlet!(backend, space, u, f, χ)
        u[free] .= F \ Vector(-Kfp * u[presc])
        return u
    end
end

"""
    _cell_conduction_localization(backend, space, k₀, V; sign = -1)
        -> (; A, A_uncorrected, L_s, L_u, dipole_norm)

The gradient localization tensor of the cavity, corrected and uncorrected, as a
`3×3` in the cell's frame.

Six solves on one factorization: three with the remote gradient
``\\nabla T = \\underline e_i`` and three with a unit-moment dipole. `sign`
selects the sign of the polarization factor and exists so that a test can flip
it and watch the factor of two appear.
"""
function _cell_conduction_localization(
        backend, space, k₀::Real, V::Real; sign::Int = -1, octant::Bool = false
    )
    K = fe_cell_stiffness(backend, space, _iso3(k₀))
    # `V` is the volume of the whole cavity in both modes; only the surface
    # average sees the eighth. Dividing everywhere would weaken the dipole
    # correction eightfold, which reads as a mesh that will not converge.
    grad(u) = fe_cell_mean_gradient(backend, space, u, octant ? V / 8 : V)

    L_s = zeros(3, 3)
    L_u = zeros(3, 3)
    for χ in (octant ? _CELL_COND_CLASSES : (nothing,))
        solve = _cell_solver(backend, space, K, χ)
        mask = χ === nothing ? ones(3) : _cell_mask2(χ)
        for i in 1:3
            e = ntuple(k -> k == i ? 1.0 : 0.0, 3)
            χ === nothing || _cell_parity(e) == χ || continue
            L_s[:, i] .= mask .*
                grad(solve(x -> e[1] * x[1] + e[2] * x[2] + e[3] * x[3]))
            L_u[:, i] .= mask .*
                grad(solve(x -> Core.dipole_temperature_iso(k₀, collect(x), e)))
        end
        solve = nothing              # drop the factor before building the next
    end

    A, dnorm = _cell_close_dipole(L_s, L_u, sign * k₀ * V)
    return (; A, A_uncorrected = L_s, L_s, L_u, dipole_norm = dnorm)
end

_iso3(k₀::Real) = Matrix(Float64(k₀) * LinearAlgebra.I, 3, 3)

"""
    _cell_elastic_localization(backend, space, C₀, μ, ν, V; sign = -1)
        -> (; A, A_uncorrected, L_s, L_u, dipole_norm)

The strain localization tensor of the cavity, corrected and uncorrected, as a
`6×6` in Kelvin-Mandel.

Twelve solves on one factorization: six with the remote strain
``\\varepsilon = \\underline{\\underline e}_i`` of the Kelvin basis — which is
what makes a unit load a unit load, the map being an isometry — and six with a
unit-moment dipole.
"""
function _cell_elastic_localization(
        backend, space, C::Tensors.SymmetricTensor{4, 3, Float64},
        μ::Real, ν::Real, V::Real; sign::Int = -1, octant::Bool = false
    )
    K = fe_cell_stiffness(backend, space, C)
    strain(u) = fe_cell_mean_strain(backend, space, u, octant ? V / 8 : V)

    L_s = zeros(6, 6)
    L_u = zeros(6, 6)
    # One assembly, one factorization per parity class: four in elasticity,
    # against one on the full cell — but on a matrix eight times smaller, so the
    # factorization cost falls by roughly 4/8² and the peak memory is that of a
    # single Cholesky. Hence dropping `solve` at the end of each pass.
    for χ in (octant ? _CELL_ELASTIC_CLASSES : (nothing,))
        solve = _cell_solver(backend, space, K, χ)
        mask = χ === nothing ? ones(6) : _cell_mask(χ)
        for m in 1:6
            E = _cell_kelvin_basis(m)
            χ === nothing || _cell_parity(E) == χ || continue
            L_s[:, m] .= mask .* strain(
                solve(
                    x -> (
                        E[1, 1] * x[1] + E[1, 2] * x[2] + E[1, 3] * x[3],
                        E[2, 1] * x[1] + E[2, 2] * x[2] + E[2, 3] * x[3],
                        E[3, 1] * x[1] + E[3, 2] * x[2] + E[3, 3] * x[3],
                    )
                )
            )
            L_u[:, m] .= mask .* strain(
                solve(x -> Tuple(Core._dipole_displacement_iso(μ, ν, collect(x), E)))
            )
        end
        solve = nothing
    end

    C66 = Matrix{Float64}(TensND.KM(TensND.Tens(C)))
    A, dnorm = _cell_close_dipole(L_s, L_u, (sign * V) .* C66)
    return (; A, A_uncorrected = L_s, L_s, L_u, dipole_norm = dnorm)
end

# ─── Mirror parity, and what an octant is allowed to leave out ───────────────
#
# The cell -- shape, outer sphere, isotropic reference -- is invariant under the
# group `G = {diag(±1,±1,±1)}` of order 8, generated by the three coordinate
# reflections `R_k = I - 2 e_k⊗e_k`. That is *weaker* than cubic symmetry, and
# it is all an octant needs: a superspheroid has it without being cubic.
#
# For a boundary datum `u = E·x`, the field `x ↦ R u(Rx)` solves the same
# problem with datum `(R E R)·x`. So whenever `R_k E R_k = χ_k E`, uniqueness
# gives `u(R_k x) = χ_k R_k u(x)`, and on the plane `x_k = 0`:
#
#     χ_k = +1  ⟹  u_k = 0            (symmetry: normal displacement pinned)
#     χ_k = −1  ⟹  u_l = 0 for l ≠ k  (antisymmetry: tangential pinned)
#
# The conditions left natural are the right ones -- `σ_kl` is odd when `χ_k=+1`,
# `σ_kk` is odd when `χ_k=−1`, so both vanish on the plane of their own accord.
# In transport, `χ_k = +1` needs nothing at all (zero normal flux is natural)
# and `χ_k = −1` needs `T = 0`.
#
# **The dipole correction obeys the same law**, which is what makes the whole
# scheme compatible with an octant rather than only its uncorrected part. From
# the closed form of `Core._dipole_displacement_iso`, `r` and `tr Π` are
# invariant and `n̂(Rx) = R n̂`, so
#
#     R u(Rx; Π) = u(x; R Π R)      and      T(Rx; M) = T(x; R M),
#
# exactly the law obeyed by `E·x` and `M·x`. Since the driver drives both
# families with the *same* Kelvin tensor, remote load and dipole share a `χ`
# for every load case, and the plane conditions above serve both.

"""
    _cell_parity(E) -> NTuple{3, Int8}

Parity of a load case under the three coordinate reflections: `χ[k] = +1` when
`R_k E R_k == E`, `-1` when `R_k E R_k == -E`.

`R_k` flips the sign of every entry with exactly one index equal to `k`, so the
test is whether the off-diagonal row `k` vanishes. Accepts a `3×3` matrix (a
strain or a dipole moment) or a 3-tuple (a gradient or a flux moment).

Every Kelvin basis tensor and every unit vector is an eigenvector of all three
reflections, which is exactly why the octant works: each load case falls into
one parity class and never mixes with another.
"""
function _cell_parity(E::AbstractMatrix)
    return ntuple(3) do k
        Int8(all(l -> l == k || iszero(E[k, l]) && iszero(E[l, k]), 1:3) ? 1 : -1)
    end
end

_cell_parity(v::Union{Tuple, AbstractVector}) =
    ntuple(k -> Int8(iszero(v[k]) ? 1 : -1), 3)

"""
    _cell_mask(χ) -> Vector{Float64}

Kelvin-component mask of the parity class `χ`: `1.0` on the components a load
case of that class can produce, `0.0` on the others.

**An octant average is not one eighth of the whole.** Reflecting the surface
integral over the eight octants gives

    ∫_{∂I} (u ⊗ n)ˢ dS  =  Σ_{g ∈ G} χ(g) · g I_oct g  =  8 · P_χ(I_oct),

so the components whose parity differs from the load case cancel *between*
octants rather than vanishing in each. They are not small in `I_oct`; they are
spurious. Multiplying by eight without projecting keeps them at full amplitude.

The Kelvin basis diagonalizes the action of `G`, so `P_χ` is this diagonal mask.
"""
_cell_mask(χ::NTuple{3, Int8}) =
    [Float64(_cell_parity(_cell_kelvin_basis(i)) == χ) for i in 1:6]

_cell_mask2(χ::NTuple{3, Int8}) =
    [Float64(_cell_parity(ntuple(k -> k == i ? 1.0 : 0.0, 3)) == χ) for i in 1:3]

"""
    _cell_kelvin_basis(i) -> 3×3

The `i`-th Kelvin-Mandel basis tensor, orthonormal for the Frobenius inner
product — which is what makes a unit load in this basis a unit load, and lets
the same six matrices serve both as remote strains and as dipole moments.
"""
function _cell_kelvin_basis(i::Integer)
    s = inv(sqrt(2.0))
    M = zeros(3, 3)
    if i ≤ 3
        M[i, i] = 1.0
    elseif i == 4
        M[2, 3] = M[3, 2] = s
    elseif i == 5
        M[1, 3] = M[3, 1] = s
    else
        M[1, 2] = M[2, 1] = s
    end
    return M
end

"""
The distinct parity classes of the six Kelvin load cases: `(+,+,+)` shared by
the three normal cases, then one class per shear case. Four in elasticity,
three in transport — so an octant assembles `K` once and factorizes it four
(resp. three) times on the reduced dof set, instead of once on the full one.
"""
const _CELL_ELASTIC_CLASSES =
    unique(_cell_parity(_cell_kelvin_basis(i)) for i in 1:6)

const _CELL_COND_CLASSES =
    unique(_cell_parity(ntuple(k -> k == i ? 1.0 : 0.0, 3)) for i in 1:3)
