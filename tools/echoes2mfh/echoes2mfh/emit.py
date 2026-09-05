"""The IR -> idiomatic Julia.

The emitter is where the output stops looking like transcribed Python. It
hoists RVE construction into a builder function, folds accumulate-in-a-loop
into comprehensions, turns matplotlib's implicit current-axes into an explicit
Plots.jl object, and renders every refusal as a block that carries the original
Python *and* fails at runtime -- so a partially translated script can never
quietly produce wrong numbers.
"""

from __future__ import annotations

import os
from typing import Iterable

from .model import (
    Assign,
    For,
    Helper,
    If,
    JuliaStmt,
    Param,
    PlotCall,
    PrintCall,
    Return,
    RVEDef,
    Script,
    Stmt,
    Sweep,
    TryCatch,
    Untranslated,
    While,
)

IND = "    "

RULE = "# " + "=" * 77

# Activating a fixed `joinpath(@__DIR__, "..")` only works for a script sitting
# in `scripts/`. A translated script goes wherever the user puts it, so the
# environment is located by walking upwards. If nothing is found the active
# environment is left alone, which is what `julia --project=...` expects.
_ACTIVATE = r"""
import Pkg
let d = @__DIR__
    while true
        pt = joinpath(d, "Project.toml")
        if isfile(pt) && occursin("MeanFieldHomogenization", read(pt, String))
            Pkg.activate(d; io = devnull)
            break
        end
        parent = dirname(d)
        parent == d && break
        d = parent
    end
end
"""

# The Echoes `tensor(...)` conventions, transcribed from the C++ sources and
# checked numerically against TensND (agreement to 9e-16):
#
#   echoes_cpp/src/echoes/tensors/tensor_builder.h:466-473   arity dispatch
#   echoes_cpp/src/echoes/tensors/tensor_iso.h:30-60         C = αJ + βK
#   echoes_cpp/src/echoes/tensors/tensor_ti.h:95-105         sym-Walpole build
#   echoes_cpp/src/echoes/tensor_special.cpp:259-264         the basis itself
#   echoes_cpp/src/echoes/tensors/tensor_ortho.h:94-118      Cijkl build
#   echoes_cpp/src/echoes/tensor_angles.cpp:70-82            rotation matrix
#
# The subtlety worth recording: for 5 parameters Echoes dispatches to
# `tensor_ti::build`, which reads them as coefficients on the *sym-Walpole*
# basis -- not as the (C1111, C1122, …) components that the class docstring
# advertises (those belong to `build_Cijkl`, which the generic builder never
# calls). For 9 parameters, by contrast, `tensor_ortho::build` really is the
# Cijkl form. The two arities genuinely disagree.
_TENSOR_HELPER = r"""
# ── Echoes `tensor(...)` ──────────────────────────────────────────────────
#
# Echoes reads the tensor order off the argument: an array's shape gives the
# order, a parameter list's length gives both the order and the symmetry
# class (2 → isotropic 4th order, 3 → 2nd order, 5 → transversely isotropic,
# 9 → orthotropic).
echoes_tensor(t::TensND.AbstractTens) = t
echoes_tensor(M::AbstractMatrix) = size(M) == (3, 3) ? Tens(M) : inv_KM(M)
echoes_tensor(v::AbstractVector, angles...) =
    echoes_tensor(Tuple(v), angles...)

# `build_rotation_mat3` (tensor_angles.cpp:70). Its third column is
# (sinθcosφ, sinθsinφ, cosθ), i.e. the usual spherical direction, which is how
# the transversely isotropic axis is recovered below without relying on any
# Euler-angle convention matching.
function echoes_rotation(θ, φ, ψ)
    ct, st, cf, sf, cp, sp = cos(θ), sin(θ), cos(φ), sin(φ), cos(ψ), sin(ψ)
    return [
        cp*ct*cf-sp*sf  -sp*ct*cf-cp*sf  st*cf
        cp*ct*sf+sp*cf  -sp*ct*sf+cp*cf  st*sf
        -cp*st           sp*st           ct
    ]
end

echoes_axis(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))

# 2 parameters — isotropic 4th order, C = α J + β K (tensor_iso.h).
echoes_tensor(p::NTuple{2}) = TensISO{3}(p[1], p[2])

# 5 parameters — transversely isotropic 4th order. The parameters are the
# coefficients on the sym-Walpole basis, which is TensND's own Walpole basis
# with W₃ and W₄ already symmetrized (ℓ₃ = ℓ₄), i.e. exactly the N = 5 layout.
function echoes_tensor(p::NTuple{5, T}, θ = 0.0, φ = 0.0) where {T}
    return TensTI{4, T, 5}(p, T.(echoes_axis(θ, φ)))
end

# 3 parameters — 2nd order, the three eigenvalues in the rotated frame
# (tensor_builder.h, build_tensor_order2_from_param).
function echoes_tensor(p::NTuple{3}, θ = 0.0, φ = 0.0, ψ = 0.0)
    A = [p[1] 0 0; 0 p[2] 0; 0 0 p[3]]
    return (θ, φ, ψ) == (0.0, 0.0, 0.0) ? Tens(A) :
        Tens(A, Basis(echoes_rotation(θ, φ, ψ)))
end

# 9 parameters — orthotropic 4th order, given as the Cijkl components
# (tensor_ortho.h:94). Note the Kelvin-Mandel factor 2 on the shear diagonal.
function echoes_tensor(p::NTuple{9, T}, θ = 0.0, φ = 0.0, ψ = 0.0) where {T}
    C1111, C1122, C1133, C2222, C2233, C3333, C2323, C3131, C1212 = p
    M = zeros(T, 6, 6)
    M[1, 1] = C1111
    M[1, 2] = M[2, 1] = C1122
    M[1, 3] = M[3, 1] = C1133
    M[2, 2] = C2222
    M[2, 3] = M[3, 2] = C2233
    M[3, 3] = C3333
    M[4, 4] = 2 * C2323
    M[5, 5] = 2 * C3131
    M[6, 6] = 2 * C1212
    return (θ, φ, ψ) == (0.0, 0.0, 0.0) ? inv_KM(M) :
        inv_KM(M, Basis(echoes_rotation(θ, φ, ψ)))
end
"""


def _rule(title: str) -> str:
    bar = "─" * max(3, 72 - len(title))
    return f"# ── {title} {bar}"


class Emitter:
    def __init__(self, script: Script, name: str, literate: bool = False):
        self.s = script
        self.name = name
        self.literate = literate
        self.out: list[str] = []

    # ------------------------------------------------------------------

    def emit(self) -> str:
        self._header()
        self._preamble()
        self._params()
        self._builders()
        self._tensor_helper()
        self._label_helper()
        self._style_helper()
        self._helpers()
        self._body()
        self._footer()
        return "\n".join(self.out).rstrip() + "\n"

    def _w(self, line: str = "") -> None:
        self.out.append(line)

    def _blank(self) -> None:
        if self.out and self.out[-1] != "":
            self.out.append("")

    # ------------------------------------------------------------------

    def _header(self) -> None:
        src = os.path.basename(self.s.source_path)
        if self.literate:
            self._w(f"# # {self.name}")
            self._w("#")
            self._w(f"# Translated from the Echoes script `{src}` by `echoes2mfh`.")
            if self.s.header:
                self._w("#")
                for line in self.s.header.splitlines():
                    self._w(f"# {line}")
            self._w()
            return
        self._w(RULE)
        self._w(f"#  {self.name}.jl")
        self._w("#")
        self._w(f"#  Translated from the Echoes script `{src}` by `echoes2mfh`.")
        if self.s.header:
            self._w("#")
            for line in self.s.header.splitlines():
                self._w(f"#  {line}")
        if self.s.findings:
            self._w("#")
            n_block = self.s.finding_count("blocking")
            n_rev = self.s.finding_count("review")
            self._w(
                f"#  {len(self.s.findings)} construct(s) were not translated "
                f"({n_block} blocking, {n_rev} to review)."
            )
            self._w("#  Each is marked UNTRANSLATED below with the original Python.")
        self._w(RULE)
        self._w()

    def _preamble(self) -> None:
        mark = "  #jl" if self.literate else ""
        # A translated script can be dropped anywhere, so the environment is
        # found by walking up from the script rather than assuming it sits one
        # level below the project root the way `scripts/` does.
        for line in _ACTIVATE.strip("\n").splitlines():
            self._w(f"{line}{mark}")
        self._blank()
        order = ["MeanFieldHomogenization", "TensND", "LinearAlgebra", "Printf", "Plots"]
        mods = [m for m in order if m in self.s.imports]
        for m in self.s.imports:
            if m not in mods:
                mods.append(m)
        if self.s.needs_plots and "Plots" not in mods:
            mods.append("Plots")
        for m in mods:
            self._w(f"using {m}")
        if self.s.needs_plots:
            self._w("gr()")
        self._blank()

    def _params(self) -> None:
        if not self.s.params:
            return
        self._w(_rule("Parameters"))
        for p in self.s.params:
            kw = "const " if p.is_const else ""
            self._w(f"{kw}{p.name} = {p.value.code}")
        self._blank()

    # ------------------------------------------------------------------
    # RVE builders -- the structural heart of the translation
    # ------------------------------------------------------------------

    def _builders(self) -> None:
        if not self.s.rves:
            return
        for rve in self.s.rves:
            self._builder(rve)

    def _builder(self, rve: RVEDef) -> None:
        self._w(_rule(f"RVE `{rve.var}` — matrix :{rve.matrix_name}"))
        self._w("#")
        self._w(
            "# Echoes mutates one module-level RVE in place; MFH's RVE is"
        )
        self._w(
            "# immutable and derives the matrix fraction as 1 - Σ f_inclusions,"
        )
        self._w("# so the construction becomes a builder parameterized by whatever")
        self._w("# the script varies.")
        if rve.complex_valued:
            self._w("#")
            self._w(
                "# Built from Echoes' `rvec`. Julia has no separate complex "
                "family:"
            )
            self._w("# a complex element type is all that is needed.")
        args = ", ".join(rve.params)
        self._w(f"function {rve.builder_name}({args})")
        # No phase name here: an RVE designates no matrix. Echoes' matrix
        # becomes the phase taking up the volume complement, which is exactly
        # what a matrix-based scheme picks up when it names none — so the
        # translated script reproduces Echoes' semantics without the schemes
        # having to be rewritten.
        ctor_kw = ["T = ComplexF64"] if rve.complex_valued else []
        if rve.needs_distribution_shape:
            # Maxwell and PCW read it, and MFH gives it no default: an
            # undeclared shape raises rather than collapsing the scheme onto
            # Mori-Tanaka in silence. Echoes' `ver.self()` is a unit sphere
            # unless the script set one, so declaring it here reproduces the
            # original run exactly -- and makes the degeneracy visible.
            ctor_kw.append("distribution_shape = Ellipsoid(1.0)")
        ctor = f"RVE(; {', '.join(ctor_kw)})" if ctor_kw else "RVE()"
        if rve.needs_distribution_shape:
            self._w(
                f"{IND}# Spherical distribution: Echoes' default. It makes "
                "Maxwell and PCW"
            )
            self._w(f"{IND}# coincide with Mori-Tanaka -- state a spheroid to part them.")
        self._w(f"{IND}rve = {ctor}")
        for pd in rve.phases:
            self._phase(pd, rve)
        self._w(f"{IND}return rve")
        self._w("end")
        self._blank()

    def _phase(self, pd, rve: RVEDef) -> None:
        geom = self._geometry(pd)
        props = self._props(pd)
        opts: list[str] = []
        if pd.symmetrize:
            opts.append(f"symmetrize = {pd.symmetrize}")

        if pd.is_matrix:
            opts.insert(0, "fraction = :rest")
        elif pd.amount:
            kind, expr = pd.amount
            opts.insert(0, f"{kind} = {expr.code}")
        line = f"{IND}add_phase!(rve, :{pd.name}, {geom}, {props}; " + ", ".join(opts) + ")"
        if len(line) <= 92:
            self._w(line)
        else:
            self._w(f"{IND}add_phase!(")
            self._w(f"{IND}{IND}rve, :{pd.name}, {geom}, {props};")
            self._w(f"{IND}{IND}" + ", ".join(opts))
            self._w(f"{IND})")

    def _geometry(self, pd) -> str:
        if pd.kind == "sphere_nlayers":
            radii = _tuple(l.code for l in pd.layers)
            moduli = pd.props.get("__moduli__")
            mods = _tuple((moduli.code.split(", ") if moduli and moduli.code else []))
            if pd.interfaces:
                ifs = _tuple(i.code for i in pd.interfaces)
                return f"LayeredSphere({radii}, {mods}; interfaces = {ifs})"
            return f"LayeredSphere({radii}, {mods})"
        if pd.kind == "spheroid_nlayers":
            radii = _tuple(l.code for l in pd.layers)
            mods_e = pd.props.get("__moduli__")
            mods = _tuple((mods_e.code.split(", ") if mods_e and mods_e.code else []))
            ar = pd.props.get("__aspect_ratio__")
            ns = pd.props.get("__nseries__")
            disk = f"{radii} ./ {ar.code}" if ar else radii
            tail = f"; Nseries = {ns.code}" if ns else ""
            return f"LayeredSpheroid({radii}, {disk}, {mods}{tail})"
        return pd.geometry.code or "Spheroid(1.0)"

    def _props(self, pd) -> str:
        entries = []
        for k, v in pd.props.items():
            if k.startswith("__"):
                continue
            if not v.code:
                continue
            entries.append(f"{k} => {v.code}")
        for k, (law, mode) in pd.visco_props.items():
            entries.append(f"{k} => ViscoLaw({law.code}, {mode})")
        if not entries:
            # layered inclusions carry their moduli in the geometry
            return "Dict{Symbol, Any}()"
        return "Dict(" + ", ".join(entries) + ")"

    # ------------------------------------------------------------------

    def _helpers(self) -> None:
        if not self.s.helpers:
            return
        self._w(_rule("Helpers"))
        for h in self.s.helpers:
            self._helper(h)

    def _helper(self, h: Helper) -> None:
        args = []
        for a in h.args:
            if a in h.defaults:
                args.append(f"{a} = {h.defaults[a].code}")
            else:
                args.append(a)
        # keyword-with-default arguments go after a semicolon in Julia
        pos = [a for a in args if "=" not in a]
        kws = [a for a in args if "=" in a]
        sig = ", ".join(pos)
        if kws:
            sig += "; " + ", ".join(kws)
        self._w(f"function {h.name}({sig})")
        self._stmts(h.body, 1)
        self._w("end")
        self._blank()

    def _tensor_helper(self) -> None:
        if not self.s.needs_tensor_helper:
            return
        for line in _TENSOR_HELPER.strip("\n").splitlines():
            self._w(line)
        self._blank()

    def _label_helper(self) -> None:
        if not self.s.needs_label_helper:
            return
        self._w(_rule("Legend labels"))
        self._w("#")
        self._w("# Echoes' scheme constants are enum values that print as short")
        self._w("# names. MFH's schemes are configured struct instances, so a")
        self._w("# plain `string` would put the whole solver configuration in the")
        self._w("# legend. The type name is the faithful short form.")
        self._w("mfh_label(x) = string(x)")
        self._w("mfh_label(x::AbstractString) = String(x)")
        self._w("mfh_label(x::Symbol) = String(x)")
        self._w(
            "mfh_label(s::MeanFieldHomogenization.HomogenizationScheme) = "
            "string(nameof(typeof(s)))"
        )
        self._blank()

    def _style_helper(self) -> None:
        if not self.s.needs_style_helper:
            return
        self._w(_rule("matplotlib format strings"))
        self._w("#")
        self._w("# The original passed a style like 'g-d' as a *variable*, so it")
        self._w("# cannot be decoded at translation time. This decodes it the way")
        self._w("# matplotlib does, into Plots.jl keywords.")
        self._w("function mpl_style(fmt::AbstractString)")
        self._w(f"{IND}colors = Dict(")
        self._w(
            f"{IND}{IND}'b' => :blue, 'g' => :green, 'r' => :red, 'c' => :cyan,"
        )
        self._w(
            f"{IND}{IND}'m' => :magenta, 'y' => :gold, 'k' => :black, "
            f"'w' => :white,"
        )
        self._w(f"{IND})")
        self._w(f"{IND}markers = Dict(")
        self._w(
            f"{IND}{IND}'+' => :cross, 'o' => :circle, '*' => :star5, "
            f"'d' => :diamond,"
        )
        self._w(
            f"{IND}{IND}'s' => :square, '^' => :utriangle, 'v' => :dtriangle,"
        )
        self._w(f"{IND}{IND}'x' => :xcross, '.' => :circle,")
        self._w(f"{IND})")
        self._w(f'{IND}rest = fmt')
        self._w(f"{IND}ls = :solid")
        self._w(f'{IND}for (pat, style) in ("--" => :dash, "-." => :dashdot, '
                f'"-" => :solid, ":" => :dot)')
        self._w(f"{IND}{IND}if occursin(pat, rest)")
        self._w(f"{IND}{IND}{IND}ls = style")
        self._w(f'{IND}{IND}{IND}rest = replace(rest, pat => ""; count = 1)')
        self._w(f"{IND}{IND}{IND}break")
        self._w(f"{IND}{IND}end")
        self._w(f"{IND}end")
        self._w(f"{IND}color = nothing")
        self._w(f"{IND}marker = nothing")
        self._w(f"{IND}for ch in rest")
        self._w(f"{IND}{IND}haskey(colors, ch) && (color = colors[ch])")
        self._w(f"{IND}{IND}haskey(markers, ch) && (marker = markers[ch])")
        self._w(f"{IND}end")
        self._w(f"{IND}kw = Pair{{Symbol, Any}}[:linestyle => ls]")
        self._w(f"{IND}color === nothing || push!(kw, :color => color)")
        self._w(
            f"{IND}marker === nothing || "
            f"(push!(kw, :marker => marker); push!(kw, :markersize => 4))"
        )
        self._w(f"{IND}return kw")
        self._w("end")
        self._blank()

    def _body(self) -> None:
        if not self.s.body:
            return
        self._w(_rule("Main"))
        if self.s.needs_plots and "p" not in self.s.declared_axes:
            self._w("p = plot()")
        self._stmts(self.s.body, 0)

    def _footer(self) -> None:
        if not self.s.needs_plots:
            return
        self._blank()
        axes = sorted(self.s.declared_axes) or ["p"]
        if self.s.used_subplots and len(axes) > 1:
            self._w(
                "# The original used a subplot grid; Plots.jl composes the "
                "panels at the end."
            )
            self._w(
                f"display(plot({', '.join(axes)}; layout = ({len(axes)}, 1)))"
            )
        elif len(axes) > 1:
            for a in axes:
                self._w(f"display({a})")
        else:
            self._w(f"display({axes[0]})")

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------

    def _stmts(self, stmts: Iterable[Stmt], depth: int) -> None:
        for s in stmts:
            self._stmt(s, depth)

    def _stmt(self, s: Stmt, depth: int) -> None:
        pad = IND * depth
        if isinstance(s, Untranslated):
            self._untranslated(s, depth)
        elif isinstance(s, JuliaStmt):
            for line in s.code.splitlines():
                self._w(pad + line)
        elif isinstance(s, Assign):
            tgt = (
                s.targets[0]
                if len(s.targets) == 1
                else "(" + ", ".join(s.targets) + ")"
            )
            self._w(f"{pad}{tgt} = {s.value.code}")
        elif isinstance(s, Return):
            self._w(f"{pad}return {s.value.code}" if s.value else f"{pad}return")
        elif isinstance(s, If):
            self._w(f"{pad}if {s.test.code}")
            self._stmts(s.body, depth + 1)
            if s.orelse:
                self._w(f"{pad}else")
                self._stmts(s.orelse, depth + 1)
            self._w(f"{pad}end")
        elif isinstance(s, For):
            tgt = (
                s.targets[0]
                if len(s.targets) == 1
                else "(" + ", ".join(s.targets) + ")"
            )
            self._w(f"{pad}for {tgt} in {s.iter.code}")
            self._stmts(s.body, depth + 1)
            self._w(f"{pad}end")
        elif isinstance(s, While):
            self._w(f"{pad}while {s.test.code}")
            self._stmts(s.body, depth + 1)
            self._w(f"{pad}end")
        elif isinstance(s, TryCatch):
            if s.py_src:
                for i, line in enumerate(_wrap(s.py_src, 70)):
                    self._w(f"{pad}# {'NOTE: ' if i == 0 else '      '}{line}")
            # A name bound inside `try` does not escape the block in Julia,
            # whereas in Python it does. Declaring it up front preserves the
            # original semantics.
            bound = sorted(_bound_names(s.body))
            if bound:
                self._w(f"{pad}local " + ", ".join(bound))
            self._w(f"{pad}try")
            self._stmts(s.body, depth + 1)
            self._w(f"{pad}catch")
            self._stmts(s.handler, depth + 1)
            self._w(f"{pad}end")
        elif isinstance(s, Sweep):
            self._sweep(s, depth)
        elif isinstance(s, PlotCall):
            self._plot(s, depth)
        elif isinstance(s, PrintCall):
            self._print(s, depth)
        else:
            self._w(f"{pad}# (unhandled IR node {type(s).__name__})")

    def _sweep(self, s: Sweep, depth: int) -> None:
        pad = IND * depth
        if s.comprehensible and len(s.accumulators) == 1:
            (name, elt), = s.accumulators.items()
            self._w(f"{pad}{name} = [{elt.code} for {s.var} in {s.iterable.code}]")
            return
        if s.comprehensible:
            # Several accumulators over one sweep. When every element is a
            # component of the *same* call, evaluate it once per step -- the
            # naive tuple comprehension would call it once per accumulator,
            # which for a homogenization sweep is a 2x cost for nothing.
            tmp = "_sweep"
            common = _common_call(list(s.accumulators.values()))
            if common is not None:
                self._w(f"{pad}{tmp} = [{common} for {s.var} in {s.iterable.code}]")
                for i, name in enumerate(s.accumulators):
                    self._w(f"{pad}{name} = [t[{i + 1}] for t in {tmp}]")
                return
            elts = ", ".join(e.code for e in s.accumulators.values())
            self._w(
                f"{pad}{tmp} = [({elts}) for {s.var} in {s.iterable.code}]"
            )
            for i, name in enumerate(s.accumulators):
                self._w(f"{pad}{name} = [t[{i + 1}] for t in {tmp}]")
            return
        for name in s.accumulators:
            self._w(f"{pad}{name} = Float64[]")
        self._w(f"{pad}for {s.var} in {s.iterable.code}")
        for name, elt in s.accumulators.items():
            self._w(f"{pad}{IND}push!({name}, {elt.code})")
        self._w(f"{pad}end")

    def _plot(self, s: PlotCall, depth: int) -> None:
        pad = IND * depth
        args = ", ".join(a.code for a in s.args)
        opts = dict(s.style)
        for k, v in s.kwargs.items():
            opts[k] = v.code
        parts = [f"{k} = {v}" for k, v in opts.items()]
        if s.runtime_style:
            parts.append(f"mpl_style({s.runtime_style})...")
        kw = ", ".join(parts)
        # The plot object is created once before Main, so every series is a
        # mutating call. matplotlib's implicit current-axes has no Julia
        # equivalent, and a `plot(...)` inside a loop would not escape it.
        head = f"{pad}{s.func}!({s.axes}, {args}"
        self._w(head + (f"; {kw})" if kw else ")"))

    def _print(self, s: PrintCall, depth: int) -> None:
        pad = IND * depth
        if s.fmt:
            self._w(f"{pad}@printf {s.fmt} {s.args[0].code}")
            return
        if not s.args:
            self._w(f"{pad}println()")
            return
        self._w(f"{pad}println(" + ", \" \", ".join(a.code for a in s.args) + ")")

    def _untranslated(self, s: Untranslated, depth: int) -> None:
        pad = IND * depth
        loc = f"line {s.lineno}"
        self._blank()
        self._w(f"{pad}#= UNTRANSLATED  {loc}  severity={s.severity}")
        self._w(f"{pad}   {s.reason}")
        if s.suggestion:
            for line in _wrap(s.suggestion, 68):
                self._w(f"{pad}   -> {line}")
        self._w(f"{pad}   Original Python:")
        for line in s.py_src.splitlines():
            self._w(f"{pad}     {line}")
        self._w(f"{pad}=#")
        if s.severity == "blocking":
            msg = s.reason.replace('"', "'")
            self._w(
                f'{pad}error("echoes2mfh: untranslated construct at '
                f'{loc} — {msg}")'
            )
        self._blank()


def _bound_names(stmts) -> set[str]:
    """Names assigned anywhere in a statement list (one level of nesting)."""
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, Assign):
            out |= set(s.targets)
        elif isinstance(s, If):
            out |= _bound_names(s.body) | _bound_names(s.orelse)
        elif isinstance(s, (For, While)):
            out |= _bound_names(s.body)
        elif isinstance(s, Sweep):
            out |= set(s.accumulators)
    return out


def _tuple(items) -> str:
    """Render a Julia tuple; only a 1-tuple needs the trailing comma."""
    parts = [p for p in items if p]
    if not parts:
        return "()"
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ", ".join(parts) + ")"


def _common_call(elts: list) -> str | None:
    """If every element reads `<call>[k]` for one shared call, return the call.

    This is the `a, b = Chom(phi, sch)` shape after inlining, and recovering
    it is what keeps a swept homogenization from being evaluated twice.
    """
    import re

    pat = re.compile(r"^(.*)\[(\d+)\]$")
    base = None
    seen: list[int] = []
    for e in elts:
        m = pat.match(e.code.strip())
        if not m:
            return None
        if base is None:
            base = m.group(1)
        elif base != m.group(1):
            return None
        seen.append(int(m.group(2)))
    if base is None or seen != list(range(1, len(seen) + 1)):
        return None
    return base


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def emit(script: Script, name: str, literate: bool = False) -> str:
    return Emitter(script, name, literate).emit()
