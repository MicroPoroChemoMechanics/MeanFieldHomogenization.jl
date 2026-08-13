# [Finite-element coupling](@id fe-coupling)

MeanFieldHomogenization used **as a constitutive law inside a structural
finite-element computation** — the role an MFront behavior or an Abaqus UMAT
plays. One microstructure stands for one material point: the FE code hands over
a strain, the package hands back a stress and a consistent tangent.

!!! warning "Two opposite couplings — this is not the finite-element inclusion"
    | | who calls whom | what it produces |
    |:--|:--|:--|
    | [Finite-element inclusions](@ref man-fe-inclusions) | **MFH calls** the FE solver | one inclusion's response, when no closed form exists |
    | This section | **the FE code calls MFH** | a material law at every Gauss point |

```mermaid
%%{init: {"flowchart": {"useMaxWidth": false, "nodeSpacing": 30, "rankSpacing": 40}} }%%
flowchart TB
    subgraph FE["structure scale — the FE code"]
        MESH["mesh, element loop"]
        QP["quadrature point:<br/>strain 𝛆, state"]
        ASM["assemble residual<br/>and Jacobian"]
    end
    subgraph MFH["material point — MeanFieldHomogenization"]
        RVE["RVE: matrix + inclusions<br/>+ crack families"]
        SCH["scheme<br/>MT, self-consistent, …"]
        HOM["ℂʰᵒᵐ, 𝐁, 1/M, 𝐊ʰᵒᵐ"]
    end
    MESH --> QP
    QP -->|"𝛆, internal state"| RVE
    RVE --> SCH --> HOM
    HOM -->|"𝛔, ∂𝛔/∂𝛆, new state"| ASM
    ASM --> MESH
```

## Why bother

A closed-form law is written once and fitted to data. A homogenized law is
*derived* from the microstructure, so the same computation also tells you what
each phase is doing — and lets the microstructure evolve. That is what makes the
fractured-reservoir model of [barthelemyARMA2011](@cite) possible: fracture
apertures follow the effective stress, and the permeability follows the
apertures.

## Reading order

Three groups, read in this order: the equations, then how to build a model with
them, then worked models.

| Page | |
|:--|:--|
| **Theory** | |
| [Scale transition](@ref fe-scale-transition) | the equations: incremental format, consistent tangent, tangent blocks |
| [The coupled poroelastic problem](@ref fe-poro-coupling) | the same, with a fluid: two balances, the Biot blocks, the four tangents |
| [Fractured permeability](@ref fe-permeability) | flowing cracks and the effective conductivity of a fracture network |
| **Manual** | |
| [Materials](@ref fe-materials) | the Gauss-point contract, in code |
| [Building a fractured-rock material](@ref fe-fractured-rock) | the ARMA 2011 material: two gradients, two fluxes, evolving permeability |
| [Ferrite backend](@ref fe-backends) | the three helpers a Ferrite driver needs |
| **Examples** | |
| [Thick-walled cylinder](@ref fe-thick-cylinder) | worked model, checked against Lamé, then with closing cracks |
| [A fractured-reservoir well test](@ref fe-arma2011) | the ARMA 2011 well test, end to end |

The Biot machinery the poroelastic coupling builds on is a property of a
microstructure rather than of the coupling, so it lives on its own page:
[Poromechanics](@ref manual-poromechanics).
