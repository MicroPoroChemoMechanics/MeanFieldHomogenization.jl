# =============================================================================
#  radial_first_order.jl — a radial problem as a system of the first order,
#  symbolically (SymPyPythonCall).
#
#  A loading mode of a sphere or of a cylinder has a few radial functions as
#  unknowns. Its state Y gathers them and r times the radial functions of the
#  traction. `first_order_rhs` writes r dY/dr as expressions linear in symbols
#  y₁, y₂, … standing for the entries of Y: the constitutive law gives the
#  first derivatives of the unknowns from the tractions, the equilibrium
#  equations give their second derivatives. `first_order` is the matrix M of
#  r dY/dr = M Y, and `first_order_forcing` what remains when Y = 0 (a loading
#  that acts inside the layer). For a constant M, `mode_states` gives the states
#  of the radial modes r^s: the eigenvectors of M.
#
#  Include it with
#
#      include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "radial_first_order.jl"))
# =============================================================================

"""
    first_order_rhs(Y, unknowns, equations, r) -> (rhs, y)

`Y`: the state, expressions in the functions `unknowns` of `r` and their first
derivatives; `equations`: as many equilibrium equations, of the second order.
"""
function first_order_rhs(Y, unknowns, equations, r)
    d₁ = [diff(g, r) for g in unknowns]
    d₂ = [diff(g, r, 2) for g in unknowns]
    y = [symbols("y$i", real = true) for i in eachindex(Y)]
    nu = length(unknowns)
    from_law = solve([Y[nu + i] - y[nu + i] for i in 1:nu], d₁)        # r g′ from the traction
    from_equilibrium = solve(equations, d₂)                             # g″ from equilibrium
    rhs = [r * diff(Yᵢ, r) for Yᵢ in Y]
    rhs = [subs(e, [d₂[i] => from_equilibrium[d₂[i]] for i in 1:nu]...) for e in rhs]
    rhs = [subs(e, [d₁[i] => from_law[d₁[i]] for i in 1:nu]...) for e in rhs]
    rhs = [subs(e, [unknowns[i] => y[i] for i in 1:nu]...) for e in rhs]
    return [expand(simplify(e)) for e in rhs], y
end

"The matrix `M` of `r dY/dr = M Y`."
function first_order(Y, unknowns, equations, r)
    rhs, y = first_order_rhs(Y, unknowns, equations, r)
    return [simplify(diff(e, yⱼ)) for e in rhs, yⱼ in y]
end

"The second member of `r dY/dr = M Y + F`: the right-hand side at `Y = 0`."
function first_order_forcing(Y, unknowns, equations, r)
    rhs, y = first_order_rhs(Y, unknowns, equations, r)
    return [simplify(subs(e, [yⱼ => 0 for yⱼ in y]...)) for e in rhs]
end

"""
    mode_states(M, exponents) -> Matrix

The states of the radial modes of a constant matrix `M`: for each `s` of
`exponents`, in order, a basis of the kernel of `M - s 1` (a solution of
`r dY/dr = M Y` is `r^s y` with `M y = s y`), each vector scaled by its first
entry that is not zero. One column per mode.
"""
function mode_states(M, exponents)
    𝟙 = [i == j ? one(eltype(M)) : zero(eltype(M)) for i in axes(M, 1), j in axes(M, 2)]
    columns = [simplify.(vec(v) ./ v[findfirst(!iszero, v)]) for s in exponents for v in (M - s * 𝟙).nullspace()]
    return reduce(hcat, columns)
end
