```@meta
EditURL = "../../../../scripts/02_hill_elasticity.jl"
```

# Hill polarization tensors in practice

The Hill tensor ``\mathbb{P}`` is the single object every mean-field scheme in
this package is built on. It answers Eshelby's question — *given a uniform
eigenstrain in an ellipsoidal inclusion embedded in an infinite matrix
``\mathbb{C}_0``, what is the resulting strain inside?* — and it depends on the
**shape** of the inclusion and on the **matrix**, never on the inclusion's own
stiffness.

The closed forms are collected in [Hill polarization
tensors](@ref th-hill-tensors); this page is the computational tour. Related
scripts go further on individual points: `01_auxiliary_tensors.jl` for the
geometric tensors ``\mathbb{I}^A, \mathbb{U}^A, \mathbb{V}^A`` underneath,
`03_hill_conductivity.jl` for the 2nd-order transport counterpart,
`06_cylinder.jl` for the infinite-cylinder limit and `07_hill_ti_coaxial.jl`
for the transversely-isotropic closed form.

````@example hill_tensors
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
````

## §0 A reference matrix

Steel, ``E = 210`` GPa, ``\nu = 0.3``, in MPa throughout.

````@example hill_tensors
const E_ref = 210.0e3
const ν_ref = 0.3
const λ_ref = E_ref * ν_ref / ((1 + ν_ref) * (1 - 2ν_ref))
const μ_ref = E_ref / (2 * (1 + ν_ref))
const k_ref = λ_ref + 2μ_ref / 3

const C_iso = TensISO{3}(3k_ref, 2μ_ref)

@printf "λ = %.2f,  μ = %.2f,  k = %.2f MPa\n" λ_ref μ_ref k_ref
````

Two display helpers — Voigt storage for printing, Mandel for the linear
algebra of §5. Double contraction itself is TensND's `⊡`, used as `P ⊡ C_iso`
in §4; there is no need to write index loops for it.

````@example hill_tensors
const voigt_idx = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
const voigt_lab = ["11", "22", "33", "23", "31", "12"]

function print_voigt(C; label = "P", scale = 1.0e6, unit = "×10⁻⁶ MPa⁻¹")
    println("  Voigt[$label] ($unit):")
    print("      ")
    for l in voigt_lab
        @printf "%10s" l
    end
    println()
    for (I, (i, j)) in enumerate(voigt_idx)
        @printf "  %2s | " voigt_lab[I]
        for (k, l) in voigt_idx
            @printf "%9.4f " scale * C[i, j, k, l]
        end
        println()
    end
    return
end

const mandel_f = (1.0, 1.0, 1.0, √2, √2, √2)

mandel(C) = [
    C[i, j, k, l] * mandel_f[I] * mandel_f[J]
        for (I, (i, j)) in enumerate(voigt_idx), (J, (k, l)) in enumerate(voigt_idx)
]
````

## §1 Isotropic matrix, four geometries

One call, [`hill_tensor`](@ref), for every shape. The algorithm is selected
automatically from the symmetry of the matrix and the shape of the inclusion.

### The sphere

The only case with a one-line closed form:

```math
P_{1111} = \frac{1}{5(\lambda + 2\mu)} + \frac{1}{\mu}\left(\frac13 - \frac15\right).
```

````@example hill_tensors
let ell = Ellipsoid(1.0)
    P = hill_tensor(ell, C_iso)
    P1111_theory = 1 / (5 * (λ_ref + 2μ_ref)) + (1 / 3 - 1 / 5) / μ_ref

    @printf "  P[1,1,1,1] = %12.9e MPa⁻¹\n" P[1, 1, 1, 1]
    @printf "  theory     = %12.9e MPa⁻¹   (err = %.2e)\n" P1111_theory abs(P[1, 1, 1, 1] - P1111_theory)
    @printf "  isotropy   : |P[1111] − P[2222]| = %.2e\n\n" abs(P[1, 1, 1, 1] - P[2, 2, 2, 2])
    print_voigt(P)
end
````

### The prolate spheroid — a long fiber

`a = 5, b = c = 1`. The answer must be transversely isotropic about the long
axis, which is a free correctness check.

````@example hill_tensors
let ell = Ellipsoid(5.0, 1.0, 1.0)
    P = hill_tensor(ell, C_iso)
    @printf "  P[1,1,1,1] = %12.9e MPa⁻¹  (axial)\n" P[1, 1, 1, 1]
    @printf "  P[2,2,2,2] = %12.9e MPa⁻¹  (transverse)\n" P[2, 2, 2, 2]
    @printf "  transverse isotropy: |P[2222] − P[3333]| = %.2e\n" abs(P[2, 2, 2, 2] - P[3, 3, 3, 3])
end
````

### The oblate spheroid — a disc

`a = b = 5, c = 1`. The component normal to the disc dominates, which is the
whole reason flat inclusions are such effective reinforcements — and, in the
``c/a \to 0`` limit, why a crack is not a small perturbation.

````@example hill_tensors
let ell = Ellipsoid(5.0, 5.0, 1.0)
    P = hill_tensor(ell, C_iso)
    @printf "  P[1,1,1,1] = %12.9e MPa⁻¹  (in-plane)\n" P[1, 1, 1, 1]
    @printf "  P[3,3,3,3] = %12.9e MPa⁻¹  (normal — dominant)\n" P[3, 3, 3, 3]
    @printf "  ratio P[3333]/P[1111] = %.2f\n" P[3, 3, 3, 3] / P[1, 1, 1, 1]
end
````

### The triaxial ellipsoid

No symmetry left to exploit, so this is where the minor and major symmetries
of ``\mathbb{P}`` are worth checking explicitly.

````@example hill_tensors
let ell = Ellipsoid(4.0, 2.0, 1.0)
    P = hill_tensor(ell, C_iso)
    print_voigt(P; label = "P triaxial")
    err_min = maximum(abs(P[i, j, k, l] - P[j, i, k, l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
    err_maj = maximum(abs(P[i, j, k, l] - P[k, l, i, j]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
    @printf "\n  symmetry: minor %.2e,  major %.2e\n" err_min err_maj
end
````

## §2 Anisotropic matrix — two algorithms, one answer

With an anisotropic ``\mathbb{C}_0`` no closed form survives and
``\mathbb{P}`` becomes a surface integral over the unit sphere. `MeanFieldHomogenization`
offers several independent routes to it — a residue evaluation of the
acoustic-tensor poles, and two cubatures (`:nestedquadgk`, built in, and
`:decuhr` when the DECUHR extension is loaded). Agreement between two of them
is the strongest available check, since they share no code.

The matrix here is cubic — `C11 = 250`, `C12 = 100`, `C44 = 80` GPa — with a
Zener ratio ``A = 2C_{44}/(C_{11}-C_{12})`` just off 1.

````@example hill_tensors
let
    C11, C12, C44 = 250.0e3, 100.0e3, 80.0e3
    C_arr = zeros(3, 3, 3, 3)
    for (I, (i, j)) in enumerate(voigt_idx), (J, (k, l)) in enumerate(voigt_idx)
        v = [
            C11 C12 C12 0 0 0;
            C12 C11 C12 0 0 0;
            C12 C12 C11 0 0 0;
            0 0 0 C44 0 0;
            0 0 0 0 C44 0;
            0 0 0 0 0 C44
        ][I, J]
        C_arr[i, j, k, l] = C_arr[j, i, k, l] = C_arr[i, j, l, k] = C_arr[j, i, l, k] = v
    end
    C_cubic = Tens(C_arr)

    @printf "Zener ratio A = 2C44/(C11−C12) = %.4f   (isotropic ⇔ 1)\n" 2C44 / (C11 - C12)

    let _ell = Ellipsoid(2.0, 1.0, 1.0)                     # warm-up, so the
        hill_tensor(_ell, C_cubic; method = :residues)       # timings below do
        hill_tensor(_ell, C_cubic; method = :nestedquadgk)   # not count compilation
    end

    ell = Ellipsoid(3.0, 1.0, 1.0)
    t_res = @elapsed P_res = hill_tensor(ell, C_cubic; method = :residues, abstol = 1.0e-8, reltol = 1.0e-6)
    t_qgk = @elapsed P_qgk = hill_tensor(ell, C_cubic; method = :nestedquadgk, abstol = 1.0e-8, reltol = 1.0e-6)

    max_err = maximum(abs(P_res[i, j, k, l] - P_qgk[i, j, k, l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
    @printf "  :residues      %.4f s\n  :nestedquadgk  %.4f s\n" t_res t_qgk
    @printf "  max |P_residues − P_nestedquadgk| = %.3e MPa⁻¹\n" max_err
end
````

!!! note "method = :auto does not pick the residues"
    For an anisotropic 3-D elastic reference the automatic choice is a
    cubature, not the residue algorithm: the residue acoustic polynomial
    degenerates when the reference is anisotropic in *type* and isotropic in
    *value*, which is exactly what the self-consistent and differential
    schemes feed back at their first step. `method = :residues` remains
    available explicitly, as above, and `:decuhr` is preferred over
    `:nestedquadgk` whenever `import DECUHR, Integrals` has been run.

## §3 Two dimensions — plane strain

The same entry point, with 2-D inclusions and a 2-D reference.

````@example hill_tensors
let
    C_iso2 = TensISO{2}(3k_ref, 2μ_ref)

    P = hill_tensor(Ellipsoid(1.0; dim = 2), C_iso2)
    @printf "circle   : P[1111] = %.9e,  isotropy err = %.2e\n" P[1, 1, 1, 1] abs(P[1, 1, 1, 1] - P[2, 2, 2, 2])

    P = hill_tensor(Ellipsoid(4.0, 1.0), C_iso2)
    @printf "ellipse  : P[1111] = %.9e (major),  P[2222] = %.9e (minor)\n" P[1, 1, 1, 1] P[2, 2, 2, 2]

    E1, E2, ν12, G12 = 100.0e3, 200.0e3, 0.3, 40.0e3
    ν21 = ν12 * E2 / E1
    D = 1 - ν12 * ν21
    C2_arr = zeros(2, 2, 2, 2)
    C2_arr[1, 1, 1, 1] = E1 / D
    C2_arr[2, 2, 2, 2] = E2 / D
    for idx in ((1, 1, 2, 2), (2, 2, 1, 1))
        C2_arr[idx...] = ν12 * E2 / D
    end
    for idx in ((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1))
        C2_arr[idx...] = G12
    end

    P = hill_tensor(Ellipsoid(4.0, 1.0), Tens(C2_arr); abstol = 1.0e-8, reltol = 1.0e-6)
    @printf "ellipse in an orthotropic matrix: P[1111] = %.9e,  P[1212] = %.9e\n" P[1, 1, 1, 1] P[1, 2, 1, 2]
end
````

## §4 The Eshelby tensor

``\mathbb{S}^{E} = \mathbb{P} : \mathbb{C}_0`` — dimensionless, and for a
sphere in an isotropic matrix equal to Eshelby's 1957 closed forms
[eshelby1957](@cite):

```math
S_{1111} = \frac{7-5\nu}{15(1-\nu)}, \qquad
S_{1122} = \frac{5\nu-1}{15(1-\nu)}, \qquad
S_{1212} = \frac{4-5\nu}{15(1-\nu)} .
```

````@example hill_tensors
let ell = Ellipsoid(1.0)
    S = hill_tensor(ell, C_iso) ⊡ C_iso

    S1111_th = (7 - 5ν_ref) / (15 * (1 - ν_ref))
    S1122_th = (5ν_ref - 1) / (15 * (1 - ν_ref))
    S1212_th = (4 - 5ν_ref) / (15 * (1 - ν_ref))

    println("  component      computed     analytical        error")
    @printf "  S[1,1,1,1]  %12.8f  %12.8f     %.2e\n" S[1, 1, 1, 1] S1111_th abs(S[1, 1, 1, 1] - S1111_th)
    @printf "  S[1,1,2,2]  %12.8f  %12.8f     %.2e\n" S[1, 1, 2, 2] S1122_th abs(S[1, 1, 2, 2] - S1122_th)
    @printf "  S[1,2,1,2]  %12.8f  %12.8f     %.2e\n" S[1, 2, 1, 2] S1212_th abs(S[1, 2, 1, 2] - S1212_th)

    S2 = hill_tensor(Ellipsoid(5.0, 1.0, 1.0), C_iso) ⊡ C_iso
    @printf "\n  prolate a/b = 5 : S[1111] = %.7f (axial),  S[2222] = %.7f (transverse)\n" S2[1, 1, 1, 1] S2[2, 2, 2, 2]
end
````

## §5 From ``\mathbb{P}`` to an effective stiffness

The dilute estimate is the shortest path from a Hill tensor to a homogenized
property:

```math
\mathbb{C}^{\text{eff}} = \mathbb{C}_0
  + f\,\delta\mathbb{C} : \left(\mathbb{I} + \mathbb{P}:\delta\mathbb{C}\right)^{-1},
\qquad \delta\mathbb{C} = \mathbb{C}_1 - \mathbb{C}_0 .
```

For voids ``\mathbb{C}_1 = 0``, so ``\delta\mathbb{C} = -\mathbb{C}_0`` and the
localization tensor collapses to ``(\mathbb{I} - \mathbb{S}^{E})^{-1}``.
Written out by hand in Mandel storage, at 5 % porosity:

````@example hill_tensors
let
    f = 0.05
    P = hill_tensor(Ellipsoid(1.0), C_iso)

    M₀ = mandel(C_iso)
    Mδ = -M₀
    MP = mandel(P)

    I6 = Matrix{Float64}(I, 6, 6)
    M_A = inv(I6 + MP * Mδ)
    M_eff = M₀ + f * Mδ * M_A
    S_eff = inv(M_eff)

    E1_eff = 1 / S_eff[1, 1]
    μ_eff = 1 / (2 * S_eff[6, 6])   # Mandel: the shear block carries the factor 2

    k₀, μ₀ = k_ref, μ_ref
    μ_eff_th = μ₀ * (1 - 5f * (3k₀ + 4μ₀) / (9k₀ + 8μ₀))

    @printf "  f = %.2f\n" f
    @printf "  E_eff = %.1f MPa   (E_eff/E₀ = %.4f)\n" E1_eff E1_eff / E_ref
    @printf "  μ_eff = %.1f MPa   (μ_eff/μ₀ = %.4f)\n" μ_eff μ_eff / μ_ref
    @printf "  analytical dilute μ_eff = %.1f MPa   (err = %.2e)\n" μ_eff_th abs(μ_eff - μ_eff_th)
end
````

In practice none of that is written by hand: an `RVE` plus [`Dilute`](@ref)
does the same algebra, in any symmetry class and for any inclusion family, and
returns a tensor rather than a 6×6 array.

````@example hill_tensors
let
    f = 0.05
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_iso); fraction = :rest)
    add_phase!(rve, :PORE, Ellipsoid(1.0), Dict(:C => TensISO{3}(0.0, 0.0)); fraction = f)

    k_eff, μ_eff = k_mu(homogenize(rve, Dilute(), :C))
    @printf "  homogenize(rve, Dilute(), :C):  k_eff = %.1f MPa,  μ_eff = %.1f MPa\n" k_eff μ_eff
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

