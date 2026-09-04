# echoes2mfh

A deterministic source-to-source translator from **Echoes** Python scripts to
**MeanFieldHomogenization.jl** Julia scripts. No AI in the loop: the same input always
produces the same output, and the mapping is data you can read and edit.

```bash
cd tools/echoes2mfh

# Classify a corpus: what is portable, and what blocks the rest.
python3 -m echoes2mfh survey /path/to/echoes_cpp/tests/python --json survey.json

# Translate one script.
python3 -m echoes2mfh translate path/to/script.py -o 90_script.jl \
        --report 90_script.report.json

# Validate the mapping tables against both live APIs.
python3 -m echoes2mfh check-drift -v
```

Requires Python 3.10+ and nothing else — no Echoes install, no Julia, no
third-party packages. The corpus scan uses only `ast`.

## What it does

It is **not** a general Python-to-Julia transpiler. It recognizes the
constructs that carry MFH meaning and reconstructs them idiomatically, and it
translates ordinary Python around them expression by expression. The
difference matters most in three places:

**The RVE lift.** Echoes mutates one module-level RVE in place:

```python
ver = rve(matrix="SOLID")
ver["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO], prop={"C": Cs})
ver["PORE"]  = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO], prop={"C": Cp})

def Chom(f, omega):
    ver["PORE"].fraction = f ; ver["SOLID"].fraction = 1 - f
    ver["PORE"].shape = spheroidal(omega)
    return max(homogenize(prop="C", rve=ver, scheme=PCW).mu, 0.)
```

MFH's `RVE` is immutable, `set_param` returns a new one, and the matrix
fraction is *derived* rather than settable. So the construction becomes a
builder parameterized by whatever the script varies — which is also how the
hand-written MFH demos are organized:

```julia
function build_ver(f, omega)
    rve = RVE()
    add_phase!(rve, :SOLID, Spheroid(1.0), Dict(:C => Cs); fraction = :rest, symmetrize = IsoSymmetrize())
    add_phase!(
        rve, :PORE, Spheroid(omega), Dict(:C => Cp);
        fraction = f, symmetrize = IsoSymmetrize()
    )
    return rve
end

function Chom(f, omega)
    # `SOLID` is the matrix: MFH derives its fraction as 1 - Σ f_inclusions
    C = homogenize(build_ver(f, omega), PonteCastanedaWillis(), :C)
    return max(k_mu(C)[2], 0.0)
end
```

Note that `ver["SOLID"].fraction = 1 - f` is *dropped*, not translated: in MFH
it is not merely unnecessary, it throws.

### Updating an RVE in place

Echoes stores the fraction *on the inclusion*; MFH stores it on the RVE, which
is the better design but leaves the question of how to update a
already-assembled RVE. The answer is `set_param`, which returns a **new** RVE
and leaves the original untouched:

```julia
rve2 = set_param(rve,  amount(:PORE), 0.3)                    # volume fraction
rve3 = set_param(rve,  crack_density(:CRACK), 0.6)            # crack density
rve4 = set_param(rve,  geometry(:PORE, :semi_axes, 3), 0.2)   # ellipsoid shape
rve5 = set_param(rve,  property(:PORE, :C, 1), 12.0)          # a stiffness component
```

Both routes are correct: the builder is what a *translated script* wants,
because the sweep varies the model anyway; `set_param` is what you want when
you already hold an RVE and need one knob moved — and it is what `derivative`
/`gradient`/`jacobian` use internally, so it is also the AD path.

**Solver options move into the scheme.** Echoes passes `epsrel`, `maxnb`,
`select_best` as loose `homogenize` keywords; MFH attaches them to the scheme
instance. `homogenize(prop="C", rve=v, scheme=SC, epsrel=1e-10, maxnb=300,
select_best=True)` becomes `homogenize(v, SelfConsistent(; abstol = 1.0e-10,
maxiters = 300, select_best = true), :C)`. Keywords a given MFH scheme does not
accept are dropped rather than passed through.

**Sweeps become comprehensions.** `k = []` followed by `for x in xs:
k.append(f(x))` becomes `k = [f(x) for x in xs]`. When several accumulators
read components of one call, the call is evaluated once per step rather than
once per accumulator.

## Refusals are a feature

Anything the tool cannot translate faithfully becomes a block carrying the
original Python verbatim, plus a runtime `error(...)`:

```julia
#= UNTRANSLATED  line 26  severity=blocking
   iteration over an RVE (`for x in ver` / `x.data()`)
   -> MFH exposes the phases by name: iterate inclusion_phase_names(rve)
   -> and use volume_fraction(rve, name) plus the localization functions
   Original Python:
     moyA=sum([x.data().factor*isotropify(x.data().eE) for x in ver],0)
=#
error("echoes2mfh: untranslated construct at line 26 — iteration over an RVE")
```

A partially translated script therefore cannot quietly produce wrong numbers.
A wrong result that looks plausible is far more expensive than a refusal, so
the tool never guesses: when `ver["PORE"].shape = ...` was briefly treated as
a fraction assignment during development, that was a bug of exactly the kind
this design exists to prevent.

Exit codes: `0` fully translated, `2` translated with findings, `3` refused
outright (e.g. Python 2 syntax — run `2to3` first).

## Measured coverage

Over the 225 scripts in `echoes_tests/`, `creep/`, `echoes_concrete/` and
`spheroid_nlayers/`:

| | scripts |
|---|---|
| parse as Python 3 | 214 |
| Python 2 syntax (refused; run `2to3`) | 11 |
| **generated Julia that parses** | **214 / 214** |
| translate with no findings at all | 33 |
| translate with 1–3 findings | 44 |
| translate with more findings | 137 |

Every generated file is syntactically valid Julia, and `echoes_tests/pcw.py`
translates and *runs* against MFH end to end. The findings counts are
occurrences, not files, and they are the work queue rather than a verdict:
`survey` prints the unmapped symbols ranked by how many scripts they block.

The remaining blockers are dominated by a short list — `einsum`, the generic
`tensor(...)` overloads, `.set_prop` on a phase, `es`, `rvec` — plus scripts
that pull in `sympy`, `nlopt` or `getfem`, which are out of scope by
construction.

## Layout

| file | role |
|---|---|
| `mapping.py` | the correspondence tables — **data, not logic** |
| `model.py` | the IR: a program model, not a token stream |
| `expr.py` | Python expression → Julia expression |
| `extract.py` | AST → IR; recognizes the physical model, lifts the RVE |
| `emit.py` | IR → idiomatic Julia |
| `survey.py` | corpus classifier |
| `__main__.py` | CLI |

To extend the tool, add a row to `mapping.py`. Only genuinely structural
constructs need code in `extract.py`/`emit.py`.

## Conventions it knows about

- `iso_stiffness(k, μ)` takes *physical* moduli, while the raw
  `TensISO{3}(a, b)` constructor takes `(3k, 2μ)` — the tool always emits the
  former.
- `symmetrize=[ISO]` (an exact rotational average inside the kernel,
  → `IsoSymmetrize()`) and `.paramsym(ISO)` (a least-squares reporting
  projection, → `best_fit_iso`) are kept strictly apart; conflating them
  changes the numbers.
- Localization tensors: Echoes stores `eE`/`sE`/`eS`/`sS` on the phase after
  `homogenize`; MFH computes them on demand from the inclusion, its stiffness
  and the *reference medium* — the matrix for matrix-based schemes, the
  homogenized result for self-consistent ones. The tool resolves which from
  the homogenize call in scope.
- **The complex twins.** Echoes exposes a parallel `...c` family (`rvec`,
  `ellipsoidc`, `tensorc`, …) because its C++ templates are instantiated
  separately for real and complex scalars. Julia's constructors are generic
  over the scalar type, so the suffix simply drops and the RVE is declared
  `RVE(; T = ComplexF64)`.
- **`tensor(...)`** is transcribed from the Echoes C++ sources and checked
  numerically against TensND (agreement ~9·10⁻¹⁶). Echoes dispatches on the
  parameter-list length (`tensor_builder.h:466`):

  | params | meaning | MFH / TensND |
  |---|---|---|
  | 2 | isotropic 4th order, `C = αJ + βK` | `TensISO{3}(α, β)` |
  | 3 (+3 angles) | 2nd order, eigenvalues in the rotated frame | `Tens(diag, Basis(P))` |
  | 5 (+2 angles) | TI 4th order, **sym-Walpole coefficients** | `TensTI{4,T,5}(p, n)` |
  | 9 (+3 angles) | orthotropic 4th order, **Cijkl components** | `inv_KM(M, Basis(P))` |
  | 21 | triclinic | refused — ordering unverified |

  Two traps worth recording. First, the 5-parameter form is *not* the
  `(C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃)` tuple the `tensor_ti` class docstring
  advertises: the generic builder calls `tensor_ti::build`, which reads
  sym-Walpole coefficients, while the documented components belong to
  `build_Cijkl`, which it never calls. Second, the 9-parameter form *is* the
  Cijkl one — the two arities genuinely disagree with each other. Echoes'
  sym-Walpole basis turns out to be TensND's Walpole basis with W₃ and W₄
  already symmetrized (ℓ₃ = ℓ₄), i.e. exactly the `N = 5` layout, so the
  five coefficients map across in order.

  The TI axis is taken as `(sinθcosφ, sinθsinφ, cosθ)` — the third column of
  Echoes' rotation matrix — which sidesteps any Euler-convention matching.
  For the rotated 2nd-order and orthotropic forms the rotation matrix is
  transcribed verbatim and passed as `Basis(P)`, whose direction was checked
  against `P·A·Pᵀ`. `KM`/`invKM` map to TensND's `KM`/`inv_KM`.
- Python's 0-based indexing: when a loop variable is used *only* as a
  subscript, the range is emitted 1-based and subscripts are left alone;
  otherwise the range stays 0-based and each subscript gets an explicit `+ 1`.
  Both are correct; the first is idiomatic.

## Known gaps

`check-drift` reports mapping holes automatically. Beyond those:

- **Crack compliance normalization.** MFH normalizes ℍ by the *minor* semi-axis
  `b`, Echoes by the major `a`, so `ℍ_Echoes = η ℍ_MFH` with `η = b/a` (see
  `docs/src/developer/validation.md`). The tool does **not** apply this factor
  silently.
- **`AsymmetricSelfConsistent` is not Echoes' `SC`** for cracked media — it is a
  different fixed point that percolates at about half the crack density.
- **`user_inclusion` subclasses** are refused: MFH's `CustomInclusion` has a
  different contract (three entry gates rather than one `build_all`). See
  `scripts/80_custom_inclusion_contract.jl`.
- Scripts importing `sympy`, `nlopt`, `getfem`, `pandas` or `openpyxl` are
  flagged out of scope.

## Related

- `docs/src/tools/from_echoes.md` — the prose porting guide this tool
  operationalizes. `check-drift` asserts the two stay consistent.
- `scripts/bench_echoes/` — the PyCall cross-check harness for numeric
  validation against a live Echoes.
