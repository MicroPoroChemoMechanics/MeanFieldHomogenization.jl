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
function _cell_solver(backend, space, K)
    ndofs, free, presc = fe_cell_dof_split(backend, space)
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K[free, free]))
    Kfp = K[free, presc]
    return function (f)
        u = zeros(ndofs)
        fe_cell_set_dirichlet!(backend, space, u, f)
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
        backend, space, k₀::Real, V::Real; sign::Int = -1
    )
    solve = _cell_solver(backend, space, fe_cell_stiffness(backend, space, _iso3(k₀)))
    grad(u) = fe_cell_mean_gradient(backend, space, u, V)

    L_s = zeros(3, 3)
    for i in 1:3
        e = ntuple(k -> k == i ? 1.0 : 0.0, 3)
        L_s[:, i] .= grad(solve(x -> e[1] * x[1] + e[2] * x[2] + e[3] * x[3]))
    end

    L_u = zeros(3, 3)
    for m in 1:3
        e = ntuple(k -> k == m ? 1.0 : 0.0, 3)
        L_u[:, m] .= grad(solve(x -> Core.dipole_temperature_iso(k₀, collect(x), e)))
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
        μ::Real, ν::Real, V::Real; sign::Int = -1
    )
    solve = _cell_solver(backend, space, fe_cell_stiffness(backend, space, C))
    strain(u) = fe_cell_mean_strain(backend, space, u, V)

    L_s = zeros(6, 6)
    for i in 1:6
        E = _cell_kelvin_basis(i)
        L_s[:, i] .= strain(
            solve(
                x -> (
                    E[1, 1] * x[1] + E[1, 2] * x[2] + E[1, 3] * x[3],
                    E[2, 1] * x[1] + E[2, 2] * x[2] + E[2, 3] * x[3],
                    E[3, 1] * x[1] + E[3, 2] * x[2] + E[3, 3] * x[3],
                )
            )
        )
    end

    L_u = zeros(6, 6)
    for m in 1:6
        Π = _cell_kelvin_basis(m)
        L_u[:, m] .= strain(
            solve(x -> Tuple(Core._dipole_displacement_iso(μ, ν, collect(x), Π)))
        )
    end

    C66 = Matrix{Float64}(TensND.KM(TensND.Tens(C)))
    A, dnorm = _cell_close_dipole(L_s, L_u, (sign * V) .* C66)
    return (; A, A_uncorrected = L_s, L_s, L_u, dipole_norm = dnorm)
end

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
