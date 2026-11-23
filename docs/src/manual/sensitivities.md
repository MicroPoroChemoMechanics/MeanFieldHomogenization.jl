# [Sensitivities](@id man-sensitivities)

!!! info "Before this page"
    [Homogenization schemes](@ref man-schemes), where the cells and the schemes
    differentiated below are built.

The derivative of an effective property with respect to one input of a cell
(a volume fraction, a modulus, a semi-axis) is computed by forward automatic
differentiation: the input is replaced by a dual number of
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), and every
function of the package carries it to the result. This page says how to
designate the input, the *parameter*, and how to ask for the derivative.

## A first derivative

```@example sens
using MeanFieldHomogenization, TensND

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(10.0, 6.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.3), Dict(:C => iso_stiffness(40.0, 30.0)); fraction = 0.2)
C₃₃₃₃(C) = canonical_components(C)[3, 3, 3, 3]

derivative(rve, MoriTanaka(), amount(:I); indexer = C₃₃₃₃)   # ∂C₃₃₃₃ / ∂f_I
```

`amount(:I)` is the parameter, the volume fraction of the phase `:I`;
`indexer` picks the scalar of the effective tensor to differentiate.

## [Parameters](@id man-sensitivities-lenses)

A parameter designates one number of a cell, which `get_param` reads and
`set_param` replaces in a copy of the cell:

| parameter | the number |
| :--- | :--- |
| `amount(:I)` | the volume fraction, or the crack density, of a phase |
| `property(:I, :C, sel)` | a coefficient of the property `:C` of a phase |
| `geometry(:I, :semi_axes, 3)` | a field of the inclusion of a phase |
| `shape_param(:semi_axes, 1)` | a field of the distribution shape |

The laminate adds `thickness` and `interface_param`
([Laminates](@ref man-laminates)), the particle assembly `radius_param` and
`center_param` ([Particle assemblies](@ref man-assemblies)), and `nested` reaches
a parameter of a cell nested in another
([Multiscale models](@ref man-multiscale)).

### The coefficient of a property

A coefficient belongs to a **symmetry class that contains the tensor**. With no
`symmetry` given, it is a coefficient of the tensor's own class:

| class | selectors | meaning |
| :--- | :--- | :--- |
| isotropic, order 4, ``\alpha\,\mathbb{J}+\beta\,\mathbb{K}`` | `:k` (≡ `:bulk`), `:μ` (≡ `:shear`), `:α`, `:β` | ``k`` and ``\mu``, with ``\alpha = d\,k`` in dimension ``d`` and ``\beta = 2\mu``; ``\alpha``, ``\beta`` themselves |
| isotropic, order 2 | `:scalar` | the conductivity |
| transversely isotropic, order 4 | `:ℓ₁` … `:ℓ₆` | the coefficients on the Walpole basis (``\ell_3 = \ell_4`` with major symmetry) |
| transversely isotropic, order 2 | `:transverse`, `:axial` | the two coefficients |
| orthotropic, order 4 | `:C1111`, `:C2222`, `:C3333`, `:C1122`, `:C1133`, `:C2233`, `:C2323`, `:C1313`, `:C1212` | the components in the frame |
| orthotropic, order 2 | `:a1`, `:a2`, `:a3` | the components in the frame |
| cubic | `:α`, `:β`, `:γ` | the coefficients on ``\mathbb{J}``, ``\mathbb{E}``, ``\mathbb{T}`` |
| any | `:Mij` (order 4), `:Mi` (order 2) | the Kelvin-Mandel component in the canonical basis |
| any | an integer `i` | the `i`-th entry of `get_data` |

In the plane, ``k = \lambda + \mu`` is the bulk modulus of the plane. A
Kelvin-Mandel component carries the factors of the Kelvin-Mandel matrix:
``M_{66} = 2\,C_{1212}``, and `:Mij` with ``i \neq j`` sets ``M_{ij}`` and
``M_{ji}`` together, so that the matrix stays symmetric.

The keyword `symmetry` names a larger class, in which the tensor is written
before its coefficient is read:

| `symmetry` | class |
| :--- | :--- |
| `(:TI, n)` | transversely isotropic about the unit vector `n` |
| `(:ORTHO, frame)`, `(:CUBIC, frame)` | orthotropic, cubic, in an orthonormal basis `frame` |
| `:ANISO`, `(:ANISO, basis)` | the Kelvin-Mandel components in `basis` (canonical by default) |

![](../assets/theory/parameter_classes.svg)

Setting a coefficient leaves the other coefficients **of its class** unchanged.
The derivative with respect to it is therefore the partial derivative in that
class, and the perturbed tensor may leave the class of the tensor given: a
derivative with respect to ``\ell_5`` of an isotropic phase, seen as
transversely isotropic about ``\underline{n}``, is the response to a
perturbation that is not isotropic.

```@example sens
n = TensND.normalize([0.0, 1.0, 1.0])
(
    k = derivative(rve, MoriTanaka(), property(:I, :C, :k); indexer = C₃₃₃₃),
    α = derivative(rve, MoriTanaka(), property(:I, :C, :α); indexer = C₃₃₃₃),
    ℓ₅ = derivative(rve, MoriTanaka(), property(:I, :C, :ℓ₅; symmetry = (:TI, n)); indexer = C₃₃₃₃),
    M₁₂ = derivative(rve, MoriTanaka(), property(:I, :C, :M12); indexer = C₃₃₃₃),
)
```

The derivative with respect to ``k`` is three times that with respect to
``\alpha = 3k``. A class that does not contain the tensor is refused, since the
derivative would be taken at another tensor; so is a selector foreign to the
class:

```@example sens
try
    get_param(rve, property(:I, :C, :ℓ₁))
catch err
    println(sprint(showerror, err))
end
```

The tutorial [One derivative, several routes](@ref tut-derivative-routes)
computes the same derivative through each class and checks them against one
another by the chain rule.

## Several parameters

`gradient` takes a vector of parameters and returns one derivative each;
`jacobian` returns the derivatives of every component of the effective tensor,
``81 \times N`` for a tensor of order 4:

```@example sens
ps = [amount(:I), property(:I, :C, :k), property(:M, :C, :μ)]
gradient(rve, MoriTanaka(), ps; indexer = C₃₃₃₃)
```

```@example sens
size(jacobian(rve, MoriTanaka(), ps))
```

## Any other input

What no parameter designates (a function of several inputs, an input that is not
stored in the cell) is differentiated by `sensitivity`, on a function that builds
the cell from a number:

```@example sens
function C₃₃₃₃_of_aspect(ω)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(10.0, 6.0)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0, 1.0, ω), Dict(:C => iso_stiffness(40.0, 30.0)); fraction = 0.2)
    return C₃₃₃₃(homogenize(r, MoriTanaka(), :C))
end
sensitivity(C₃₃₃₃_of_aspect, 0.3)
```

A cell nested in another is differentiated the same way, the dual number
crossing every scale; [Multiscale models](@ref man-multiscale) does it with
`nested` parameters.

## Where to go next

[One derivative, several routes](@ref tut-derivative-routes) checks the
parameters of every symmetry class against one another;
[Derivatives and sensitivities](@ref tut-sensitivities) puts them to work;
[Quasi-brittle strength of cement paste and mortar](@ref app-strength)
differentiates a three-scale model with respect to the modulus of one family
of hydrate needles, the derivative from which [pichler2011](@citet) obtain a
strength. The manual closes with an appendix,
[Elliptic integrals — examples](@ref man-elliptic-examples), on the special
functions behind the closed-form Hill tensors.
