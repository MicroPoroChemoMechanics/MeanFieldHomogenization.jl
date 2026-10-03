# =============================================================================
#  mlp.jl — the dependency-free, type-generic multilayer perceptron used to
#  *evaluate* a trained surrogate.
#
#  Training happens elsewhere (`MeanFieldHomogenizationLuxExt`); what lives here is the
#  smallest object able to reproduce a trained network's forward pass, so that
#  a surrogate loaded from disk needs nothing beyond `LinearAlgebra`.
#
#  Two properties are load-bearing, and neither is negotiable:
#
#  1. **Generic in the element type.**  The forward pass is plain `W * x .+ b`
#     followed by a broadcast, so a `ForwardDiff.Dual` input flows through
#     `Float64` weights and comes out `Dual`.  That is what makes
#     `derivative(rve, scheme, geometry(:phase, :field))` reach a *morphology*
#     parameter through a surrogate — the thing a finite-element solve cannot
#     do (`FiniteElements.jl` refuses it outright).
#  2. **Smooth activations only.**  `tanh` and `softplus` are `C^∞`; ReLU is
#     not even `C¹`.  A kink in the activation is a kink in the Hill tensor,
#     hence a wrong second derivative and a discontinuous sensitivity, so the
#     activation table below deliberately offers no piecewise-linear option.
# =============================================================================

"""
    softplus(x)

Smooth positive activation ``\\log(1 + e^x)``, evaluated in the numerically
stable form ``\\log(1 + e^{-|x|}) + \\max(x, 0)`` so that large ``|x|`` neither
overflows nor loses the linear branch.
"""
softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))

# Name ↔ function table.  The *name* is what gets serialized, so this table is
# the compatibility surface of a stored model: entries may be added, never
# renamed or removed.
const ACTIVATIONS = Dict{Symbol, Any}(
    :tanh => tanh,
    :softplus => softplus,
    :identity => identity,
)

const ACTIVATION_NAMES = Dict{Any, Symbol}(v => k for (k, v) in ACTIVATIONS)

"""
    activation(name::Symbol) -> Function

Look up an activation by the name used in a serialized surrogate. Raises on an
unknown name rather than silently substituting a default, because a wrong
activation is a silently wrong tensor.
"""
function activation(name::Symbol)
    haskey(ACTIVATIONS, name) || throw(
        ArgumentError(
            "unknown activation :$name; the surrogate format knows " *
                "$(sort(collect(keys(ACTIVATIONS)))). A stored model naming an " *
                "activation this version does not have cannot be evaluated."
        )
    )
    return ACTIVATIONS[name]
end

"""
    activation_name(σ) -> Symbol

Reverse lookup, for serialization.
"""
function activation_name(σ)
    haskey(ACTIVATION_NAMES, σ) || throw(
        ArgumentError(
            "activation $σ has no registered name, so a network using it cannot " *
                "be saved. Add it to `NeuralInclusions.ACTIVATIONS` first."
        )
    )
    return ACTIVATION_NAMES[σ]
end

"""
    NNDense(W, b, σ)

One fully connected layer, ``\\mathbf{x} \\mapsto \\sigma(\\mathbf{W}\\mathbf{x} + \\mathbf{b})``, ``\\sigma`` applied componentwise.

Named `NNDense` rather than `Dense` on purpose: the training extension has
`Lux.Dense` in scope, and two `Dense` types in one file is a trap.
"""
struct NNDense{T <: Number, F}
    W::Matrix{T}
    b::Vector{T}
    σ::F
    function NNDense(W::Matrix{T}, b::Vector{T}, σ::F) where {T <: Number, F}
        size(W, 1) == length(b) || throw(
            DimensionMismatch(
                "a layer with $(size(W, 1)) outputs needs $(size(W, 1)) biases, " *
                    "got $(length(b))"
            )
        )
        return new{T, F}(W, b, σ)
    end
end

(l::NNDense)(x::AbstractVector) = l.σ.(l.W * x .+ l.b)

n_in(l::NNDense) = size(l.W, 2)
n_out(l::NNDense) = size(l.W, 1)

"""
    MLP(layers...)

Feed-forward stack of [`NNDense`](@ref) layers, callable on a feature vector.

The layers are held in a `Tuple`, so the forward pass is fully inferred and a
small network costs one allocation per layer and nothing else.
"""
struct MLP{L <: Tuple}
    layers::L
    function MLP(layers::Tuple)
        isempty(layers) && throw(ArgumentError("an `MLP` needs at least one layer"))
        for k in 2:length(layers)
            n_out(layers[k - 1]) == n_in(layers[k]) || throw(
                DimensionMismatch(
                    "layer $k takes $(n_in(layers[k])) inputs but layer $(k - 1) " *
                        "produces $(n_out(layers[k - 1]))"
                )
            )
        end
        return new{typeof(layers)}(layers)
    end
end

MLP(layers::NNDense...) = MLP(layers)

(m::MLP)(x::AbstractVector) = foldl((y, l) -> l(y), m.layers; init = x)

n_in(m::MLP) = n_in(first(m.layers))
n_out(m::MLP) = n_out(last(m.layers))

"""
    layer_widths(m::MLP) -> Vector{Int}

The ``[n_{\\mathrm{in}}, h_1, \\dots, n_{\\mathrm{out}}]`` description of the architecture — what the training
extension needs to build an isomorphic `Lux.Chain`.
"""
layer_widths(m::MLP) = [n_in(m); [n_out(l) for l in m.layers]]

"""
    layer_activations(m::MLP) -> Vector{Symbol}

Activation name of each layer, output layer included.
"""
layer_activations(m::MLP) = [activation_name(l.σ) for l in m.layers]

"""
    nparams(m::MLP) -> Int

Total number of weights and biases.
"""
nparams(m::MLP) = sum(length(l.W) + length(l.b) for l in m.layers)

Base.eltype(m::MLP) = eltype(first(m.layers).W)

function Base.show(io::IO, m::MLP)
    w = join(layer_widths(m), "→")
    return print(io, "MLP($w, $(join(layer_activations(m), ", ")); $(nparams(m)) params)")
end

"""
    glorot_mlp(rng, widths; hidden = :tanh, output = :identity) -> MLP

Fresh network with Glorot-uniform weights and zero biases: layer `k` draws from
``\\pm\\sqrt{6/(n_{\\mathrm{in}}+n_{\\mathrm{out}})}``, the scaling that keeps the
forward variance roughly constant through a `tanh` stack.

`widths` is ``[n_{\\mathrm{in}}, h_1, \\dots, n_{\\mathrm{out}}]``. Every hidden layer takes the `hidden`
activation; the output layer takes `output`, which should stay `:identity` —
the physical range of a Hill-tensor component is handled by the output
transform of the surrogate, not by squashing the last layer.
"""
function glorot_mlp(
        rng::Random.AbstractRNG,
        widths::AbstractVector{<:Integer};
        hidden::Symbol = :tanh,
        output::Symbol = :identity
    )
    length(widths) ≥ 2 || throw(
        ArgumentError("`widths` needs at least an input and an output size")
    )
    all(>(0), widths) || throw(ArgumentError("layer widths must be positive"))
    nl = length(widths) - 1
    layers = ntuple(nl) do k
        nin, nout = widths[k], widths[k + 1]
        lim = sqrt(6 / (nin + nout))
        W = (2 .* rand(rng, Float64, nout, nin) .- 1) .* lim
        NNDense(W, zeros(Float64, nout), activation(k == nl ? output : hidden))
    end
    return MLP(layers)
end

glorot_mlp(widths::AbstractVector{<:Integer}; kw...) =
    glorot_mlp(Random.default_rng(), widths; kw...)
