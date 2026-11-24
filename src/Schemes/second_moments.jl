# =============================================================================
#  second_moments.jl — second moments of the local fields over a phase.
#
#  The effective energy of a cell under a macroscopic strain 𝐄 is stationary
#  with respect to the local strain field, so that its derivative with respect
#  to the stiffness of one phase is the mean over that phase of the quadratic
#  form the variation of the stiffness defines (Kreher 1990):
#
#      fᵢ ⟨ε : 𝔹 : ε⟩ᵢ = d/dt [ 𝐄 : ℂʰᵒᵐ(ℂᵢ + t 𝔹) : 𝐄 ] at t = 0.
#
#  The derivative is taken by ForwardDiff through the lens `PropertyDirection`
#  of `parameters.jl`, hence for every cell and every scheme `homogenize`
#  accepts. The same statement holds at order 2
#  (conduction) for the gradient of the potential. Theory: the documentation
#  page "Second moments of the local fields".
# =============================================================================

# The quadratic form of an effective property on a macroscopic gradient: twice
# the energy 𝐄 : ℂ : 𝐄 of a strain, or G · 𝐊 · G for the gradient of a potential.
_quadratic_form(C::TensND.AbstractTens{4}, E) = E ⊡ (C ⊡ E)
_quadratic_form(K::TensND.AbstractTens{2}, G) = G ⋅ (K ⋅ G)

# The volume fraction of the member over which the moment is a mean. A family of
# cracks has no volume: its fields are described by the opening of the cracks.
_moment_fraction(rve::RVE, name::Symbol) = _moment_fraction(rve.amounts[name], rve, name)
_moment_fraction(::AbstractAmount, rve::RVE, name::Symbol) = volume_fraction(rve, name)
_moment_fraction(::CrackDensity, ::RVE, name::Symbol) = throw(
    ArgumentError(
        "second_moment: :$name is a family of cracks, which has no volume to average over; " *
            "the second moment of a phase is that of a phase declared by a volume fraction"
    )
)

"""
    second_moment(cell, scheme, phase, E, 𝔹; property = :C, kw...) -> Number
    second_moment(cell, scheme, phase, E; property = :C, kw...) -> AbstractTens

Second moment of the local field over the member `phase` of `cell`, under the
macroscopic gradient `E`, as estimated by `scheme`.

With `property = :C`, `E` is the macroscopic strain (a symmetric tensor of
order 2) and the local field the strain ``\\boldsymbol{\\varepsilon}``:

  - with a tensor `𝔹` of order 4, the mean
    ``\\langle\\boldsymbol{\\varepsilon}:\\mathbb{B}:\\boldsymbol{\\varepsilon}\\rangle_i``
    of one quadratic form, at the cost of one derivative:
    ``\\mathbb{B}=\\mathbb{K}`` gives
    ``\\langle\\boldsymbol{\\varepsilon}^d:\\boldsymbol{\\varepsilon}^d\\rangle_i``
    and ``\\mathbb{B}=3\\mathbb{J}`` gives
    ``\\langle(\\mathrm{tr}\\,\\boldsymbol{\\varepsilon})^2\\rangle_i``;
  - without it, the whole tensor
    ``\\langle\\boldsymbol{\\varepsilon}\\otimes\\boldsymbol{\\varepsilon}\\rangle_i``,
    in the canonical basis.

With a property of order 2 (`property = :K`), `E` is the macroscopic gradient
of the potential, `𝔹` a tensor of order 2, and the results are
``\\langle\\underline{g}\\cdot\\boldsymbol{B}\\cdot\\underline{g}\\rangle_i`` and
``\\langle\\underline{g}\\otimes\\underline{g}\\rangle_i``.

The moment is the derivative of the effective energy with respect to the
property of the phase (Kreher's theorem),
``f_i\\langle\\boldsymbol{\\varepsilon}:\\mathbb{B}:\\boldsymbol{\\varepsilon}\\rangle_i
= \\mathrm{d}\\bigl[\\boldsymbol{E}:\\mathbb{C}^{\\mathrm{hom}}(\\mathbb{C}_i+t\\,\\mathbb{B}):\\boldsymbol{E}\\bigr]/\\mathrm{d}t``
at ``t=0``, taken by automatic differentiation through the lens
[`PropertyDirection`](@ref): `𝔹` may belong to any symmetry class, and the
phase need not. `kw...` are passed to [`homogenize`](@ref). A family of cracks
is refused: it has no volume.

Theory: [Second moments of the local fields](@ref th-second-moments).

# Examples
```jldoctest
julia> using MeanFieldHomogenization, TensND

julia> rve = RVE();

julia> add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(10.0, 6.0)); fraction = :rest);

julia> add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness(40.0, 20.0)); fraction = 0.3);

julia> E = TensISO{3}(1.0);          # a unit strain 𝟏: tr 𝐄 = 3

julia> m = second_moment(rve, Voigt(), :I, E, TensISO{3}(3.0, 0.0));   # 3𝕁: ⟨(tr ε)²⟩, of a uniform strain here

julia> m ≈ 9
true
```
"""
function second_moment(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme, phase::Symbol,
        E, 𝔹::TensND.AbstractTens; property::Symbol = :C, kw...
    )
    T = promote_type(_energy_type(cell, scheme, E, property; kw...), eltype(𝔹))
    return moment_along(cell, scheme, phase, E, 𝔹, T; property = property, kw...)
end

function second_moment(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme, phase::Symbol,
        E::TensND.AbstractTens{order, dim}; property::Symbol = :C, kw...
    ) where {order, dim}
    f = _moment_fraction(cell, phase)
    shape = (Val(order), Val((dim * (dim + 1)) ÷ 2))
    basis = TensND.CanonicalBasis{dim, float(eltype(E))}()
    x₀ = _direction_coordinates(zero(_energy_type(cell, scheme, E, property; kw...)), shape...)
    # One evaluation in dual numbers gives every component: the property of the
    # phase is moved by the tensor of Kelvin-Mandel array `x`.
    lens(x) = PropertyDirection(phase, property, TensND.from_mandel(_direction_array(x, shape...), basis))
    energy(x) = _quadratic_form(homogenize(set_param(cell, lens(x), one(eltype(x))), scheme; property = property, kw...), E)
    return TensND.from_mandel(_moment_array(_energy_gradient(energy, x₀), shape...), basis) / f
end

# The number type of the effective energy, in which the abscissa of the property
# is seeded. It holds the dual numbers of every input of the cell, those of the
# other members included, which the type of the cell does not show: seeded in a
# narrower type, a derivative taken under another one could not order the two.
_energy_type(cell, scheme, E, property; kw...) = typeof(_quadratic_form(homogenize(cell, scheme; property = property, kw...), E))

# The derivative of the energy at the abscissa zero, of number type `T`, and its
# gradient at the origin `x₀`, in dual numbers. A symbolic number does not go
# through ForwardDiff: the Symbolics and SymPy extensions differentiate the
# expression of the energy instead.
_energy_derivative(energy, ::Type{T}) where {T} = ForwardDiff.derivative(energy, zero(T))
_energy_gradient(energy, x₀::SVector) = ForwardDiff.gradient(energy, x₀)

"""
    moment_along(cell, scheme, phase, E, 𝔹, T; property = :C, kw...)

[`second_moment`](@ref) along `𝔹`, the number type `T` of the effective energy
being given instead of computed: for a caller that evaluates the moment many
times and knows that type (the promotion of the number types of every input of
`cell`, of `E` and of `𝔹`), which saves one evaluation of the scheme per call.
"""
function moment_along(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme, phase::Symbol,
        E, 𝔹::TensND.AbstractTens, ::Type{T}; property::Symbol = :C, kw...
    ) where {T}
    f = _moment_fraction(cell, phase)
    lens = PropertyDirection(phase, property, 𝔹)
    energy(t) = _quadratic_form(homogenize(set_param(cell, lens, t), scheme; property = property, kw...), E)
    return _energy_derivative(energy, T) / f
end

# ── The symmetric tensors of the space of the property, by their components ──
#
# A property of order 4 (the field is a strain, of order 2) has a symmetric
# Kelvin-Mandel matrix of size n = d(d + 1)/2, hence n(n + 1)/2 independent
# components, read in the order (1,1), (1,2), (2,2), (1,3), …; a property of
# order 2 (the field is a gradient, of order 1) has a Kelvin-Mandel vector of
# size n.
_sym_index(i, j) = min(i, j) + (max(i, j) * (max(i, j) - 1)) ÷ 2

# A vector of independent components, all equal to `z`.
_direction_coordinates(z, ::Val{1}, ::Val{n}) where {n} = SVector{n}(ntuple(_ -> z, Val(n)))
_direction_coordinates(z, ::Val{2}, ::Val{n}) where {n} = SVector{(n * (n + 1)) ÷ 2}(ntuple(_ -> z, Val((n * (n + 1)) ÷ 2)))

# The Kelvin-Mandel array of the tensor of independent components `x`.
_direction_array(x, ::Val{1}, ::Val{n}) where {n} = x
_direction_array(x, ::Val{2}, ::Val{n}) where {n} =
    SMatrix{n, n}(ntuple(k -> x[_sym_index(fldmod1(k, n)...)], Val(n * n)))

# The Kelvin-Mandel array of the moment from the gradient `g` of the energy: an
# off-diagonal component of the direction stands at two places of its matrix,
# so that its derivative is twice the component of the moment.
_moment_array(g, ::Val{1}, ::Val{n}) where {n} = g
_moment_array(g, ::Val{2}, ::Val{n}) where {n} =
    SMatrix{n, n}(ntuple(k -> g[_sym_index(fldmod1(k, n)...)] / (1 + !=(fldmod1(k, n)...)), Val(n * n)))
