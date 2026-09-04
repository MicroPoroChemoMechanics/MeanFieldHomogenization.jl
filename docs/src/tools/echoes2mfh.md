# [Translating Echoes scripts: `echoes2mfh`](@id tools-echoes2mfh)

[From Echoes to MeanFieldHomogenization](@ref tools-from-echoes) explains the correspondence
between the two APIs. `tools/echoes2mfh/` automates it: point it at an Echoes
Python script and it writes the Julia one.

It is deterministic — the same input always gives the same output — and it is
plain Python 3.10+ with no dependencies beyond the standard library, so it runs
without Julia or Echoes installed.

```bash
cd tools/echoes2mfh

python3 -m echoes2mfh survey    /path/to/echoes_cpp/tests/python
python3 -m echoes2mfh translate porous.py -o 90_porous.jl --report porous.json
python3 -m echoes2mfh check-drift -v
```

## What it does to a script

Take the classic porous benchmark, `echoes_tests/porous.py`:

```python
ver = rve(matrix="SOLID")
ver["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO], prop={"C": Cs})
ver["PORE"]  = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO], prop={"C": Cp})

def Chom(f, sch):
    ver["PORE"].fraction = f ; ver["SOLID"].fraction = 1 - f
    C = homogenize(prop="C", rve=ver, scheme=sch,
                   maxnb=300, epsrel=1.e-10, select_best=True)
    return max(C.k, 0.), max(C.mu, 0.)
```

and the translation:

```julia
function build_ver(f)
    rve = RVE()
    add_phase!(rve, :SOLID, Spheroid(1.0), Dict(:C => Cs); fraction = :rest, symmetrize = IsoSymmetrize())
    add_phase!(
        rve, :PORE, Spheroid(1.0), Dict(:C => Cp);
        fraction = f, symmetrize = IsoSymmetrize()
    )
    return rve
end

function Chom(f, sch)
    # `SOLID` is the matrix: MFH derives its fraction as 1 - Σ f_inclusions
    C = homogenize(build_ver(f), sch, :C)
    return (max(k_mu(C)[1], 0.0), max(k_mu(C)[2], 0.0))
end
```

Three things happened that are not local rewrites:

**The RVE was lifted into a builder.** Echoes mutates one module-level RVE;
MFH's [`RVE`](@ref) is immutable, and its matrix fraction is *derived*
(`1 - Σ f`) rather than settable. So the construction becomes a function of
whatever the script varies. Note that `ver["SOLID"].fraction = 1 - f` is
**dropped, not translated** — in MFH it is not merely redundant, it raises.

**Solver options moved into the scheme.** Echoes passes `epsrel`, `maxnb` and
`select_best` as loose `homogenize` keywords; MFH attaches them to the scheme
instance, so `scheme=SC, epsrel=1e-10, maxnb=300, select_best=True` becomes
`SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)`.
Keywords a given scheme does not read are dropped rather than passed on.

**Accumulate-in-a-loop became a comprehension.** `k = []` followed by
`for x in xs: k.append(f(x))` becomes `k = [f(x) for x in xs]`, and when
several accumulators read components of one call, that call is evaluated once
per step instead of once per accumulator.

## Refusals

Anything the translator cannot render faithfully becomes a block carrying the
original Python, plus a runtime `error(...)`:

```julia
#= UNTRANSLATED  line 26  severity=blocking
   iteration over an RVE (`for x in ver` / `x.data()`)
   -> MFH exposes the phases by name: iterate inclusion_phase_names(rve)
   -> and use volume_fraction(rve, name) plus the localization functions
   Original Python:
     moyA = sum([x.data().factor * isotropify(x.data().eE) for x in ver], 0)
=#
error("echoes2mfh: untranslated construct at line 26 — iteration over an RVE")
```

A partially translated script therefore cannot quietly produce wrong numbers.
This is deliberate: a plausible-looking wrong result ends up in a paper, while
a refusal costs an afternoon of hand-porting.

Exit codes are `0` (clean), `2` (translated, with findings) and `3` (refused —
typically Python 2 syntax, which `2to3` fixes first).

## Tensor conventions

`tensor(...)` is the one place where reading the Echoes C++ sources was
unavoidable, because the parameter-list length selects both the order and the
symmetry class, and the class docstrings do not describe what the generic
builder actually calls:

| params | meaning | MFH / TensND |
| :--- | :--- | :--- |
| 2 | isotropic 4th order, ``\mathbb C = \alpha \mathbb J + \beta \mathbb K`` | `TensISO{3}(α, β)` |
| 3 (+3 angles) | 2nd order, eigenvalues in the rotated frame | `Tens(diag, Basis(P))` |
| 5 (+2 angles) | transversely isotropic, **sym-Walpole coefficients** | `TensTI{4,T,5}(p, n)` |
| 9 (+3 angles) | orthotropic, **Cijkl components** | `inv_KM(M, Basis(P))` |
| 21 | triclinic | refused — component ordering unverified |

Two traps are worth stating. The 5-parameter form is *not* the
``(C_{1111}, C_{1122}, C_{1133}, C_{3333}, C_{2323})`` tuple its class
docstring advertises: the generic builder dispatches to `tensor_ti::build`,
which reads coefficients on the sym-Walpole basis, while those components
belong to `build_Cijkl`, which it never calls. The 9-parameter form, by
contrast, *is* the Cijkl one — the two arities genuinely disagree.

Echoes' sym-Walpole basis turns out to be TensND's Walpole basis with
``\mathbb W_3`` and ``\mathbb W_4`` already symmetrized (``\ell_3 = \ell_4``),
i.e. exactly the `N = 5` layout, so the five coefficients map across in order.
The correspondence was checked numerically to ``9 \times 10^{-16}``.

The complex twins (`rvec`, `ellipsoidc`, `tensorc`, …) exist only because the
C++ templates are instantiated separately for real and complex scalars. Julia's
constructors are generic over the scalar type, so the suffix drops and the RVE
is declared `RVE(; T = ComplexF64)`.

## Coverage

Over the 225 scripts of `echoes_tests/`, `creep/`, `echoes_concrete/` and
`spheroid_nlayers/`:

| | scripts |
| :--- | ---: |
| parse as Python 3 | 214 |
| Python 2 syntax (refused; run `2to3` first) | 11 |
| **generated Julia that parses** | **214 / 214** |
| translate with no findings at all | 36 |
| translate with 1–3 findings | 46 |
| translate with more findings | 132 |

The finding counts are occurrences rather than files, and they are a work queue
rather than a verdict: `survey` ranks the unmapped symbols by how many scripts
each one blocks. `echoes_tests/porous.py` translates, runs, and reproduces the
captured Echoes 1.0 values exactly (Mori-Tanaka at ``\varphi = 0.3``:
``k = 33.460582``, ``\mu = 17.626742``).

## Staying in step

`check-drift` compares the mapping tables against both live APIs: it scrapes
the Echoes symbol list from the pybind11 sources and the MFH export list from
`src/MeanFieldHomogenization.jl`, then reports any symbol that is neither mapped nor
explicitly refused, and any mapping target MFH no longer exports.

## See also

- [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes) — the API correspondence
  the tool automates.
- [Cross-validation against Echoes](@ref dev-validation) — the deliberate
  divergences, including the crack-compliance normalization
  (``\mathbb H_{\text{Echoes}} = \eta\, \mathbb H_{\text{MFH}}``, ``\eta = b/a``),
  which the translator never applies silently.
- [MFH Studio](@ref tools-mfhstudio) — the graphical builder. It shares this
  tool's code generator, so both write the same style of Julia, and its **Open**
  runs this translator when handed a `.py`.
