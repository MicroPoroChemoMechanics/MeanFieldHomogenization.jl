# Two checks, one in each direction, both of which Documenter only performs
# during a full build (~90 min here) and both of which have broken a build:
#
#   forward  — every name listed in an `@docs` block carries a docstring;
#   reverse  — every **exported** name that carries a docstring is listed in
#              some `@docs` block, which is what `checkdocs = :exports` demands.
#
# and a third, on the cross-references inside docstrings:
#
#   qualified — a `@ref Mod.name` written in a docstring must name the module
#               that **defines** `name`.
#
# The reverse one is the subtler failure: a new exported type with a good
# docstring and no `@docs` entry is reported as a missing docstring *and* makes
# every bare `@ref` to it unresolvable, so the build fails on a cross-reference
# far from the actual omission.
using MeanFieldHomogenization
const MFH = MeanFieldHomogenization

"Every module reachable from `MFH`, itself included."
function all_modules()
    seen = Set{Module}([MFH])
    queue = Module[MFH]
    while !isempty(queue)
        m = pop!(queue)
        for n in names(m; all = true)
            isdefined(m, n) || continue
            v = try getfield(m, n) catch; continue end
            if v isa Module && !(v in seen) && startswith(string(v), "MeanFieldHomogenization")
                push!(seen, v); push!(queue, v)
            end
        end
    end
    return collect(seen)
end

const MODS = all_modules()

"Is `path` (bare or dotted) documented in any reachable module?"
function has_doc(path)
    parts = split(path, ".")
    sym = Symbol(parts[end])
    owners = Module[]
    if length(parts) > 1
        m = try
            foldl((mm, s) -> getfield(mm, Symbol(s)), parts[1:(end - 1)]; init = Main)
        catch
            nothing
        end
        m isa Module && push!(owners, m)
    end
    append!(owners, MODS)
    for m in owners
        b = Base.Docs.Binding(m, sym)
        haskey(Base.Docs.meta(m), b) && return true
        # `const B_tensor = cod_tensor`: Documenter follows the alias, so must we.
        a = try Base.Docs.aliasof(b) catch; nothing end
        if a !== nothing && a != b
            for mm in owners
                haskey(Base.Docs.meta(mm), a) && return true
            end
        end
    end
    return false
end

function scan()
    bad, n = String[], 0
    for (root, _, files) in walkdir(joinpath(pkgdir(MFH), "docs", "src"))
        for f in files
            endswith(f, ".md") || continue
            inblock = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs")
                    inblock = true; continue
                elseif inblock && startswith(s, "```")
                    inblock = false; continue
                end
                inblock || continue
                (isempty(s) || startswith(s, "#")) && continue
                name = first(split(s, r"[\s(]"))
                occursin(r"^[A-Za-z_][A-Za-z0-9_.!]*$", name) || continue
                n += 1
                has_doc(name) || push!(bad, "$(relpath(joinpath(root, f), pkgdir(MFH)))  ->  $name")
            end
        end
    end
    return bad, n
end

"""
Every name that appears in any `@docs` block, bare or dotted, as a `Set` of the
final path segments — which is what a `@docs` entry has to match by.
"""
function listed_names()
    out = Set{Symbol}()
    for (root, _, files) in walkdir(joinpath(pkgdir(MFH), "docs", "src"))
        for f in files
            endswith(f, ".md") || continue
            inblock = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs")
                    inblock = true; continue
                elseif inblock && startswith(s, "```")
                    inblock = false; continue
                end
                inblock || continue
                (isempty(s) || startswith(s, "#")) && continue
                name = first(split(s, r"[\s(]"))
                occursin(r"^[A-Za-z_][A-Za-z0-9_.!]*$", name) || continue
                push!(out, Symbol(last(split(name, "."))))
            end
        end
    end
    return out
end

"""
Exported names carrying a docstring that no `@docs` block lists.

Only docstrings **owned** by a reachable `MeanFieldHomogenization` module count.
That restriction is the whole difficulty: `MFH` re-exports a large part of
`TensND`, and those bindings have docstrings that belong to `TensND` and are
documented there. Asking `names(MFH)` instead reports twenty of them and the
list becomes useless.
"""
function unlisted_exports()
    listed = listed_names()
    bad = Tuple{Module, Symbol}[]
    seen = Set{Symbol}()
    for m in MODS, s in names(m)          # exported only: no `all = true`
        s in seen && continue
        isdefined(m, s) || continue
        b = Base.Docs.Binding(m, s)
        haskey(Base.Docs.meta(m), b) || continue   # not owned here
        push!(seen, s)
        s in listed || push!(bad, (m, s))
    end
    return sort(bad; by = t -> (string(t[1]), string(t[2])))
end

"""
    unqualified_module_refs() -> Vector

Cross-references in `src/` docstrings whose module path is not the one that
**defines** the name.

Documenter resolves a docstring's `@ref` starting from that docstring's own
module, and its fallback in `Main` is allowed only for a *fully qualified* name.
So `@ref MeanFieldHomogenization.LayeredSpheroid`, written in a docstring that
lives in `FiniteElements`, fails: the binding is re-exported at the top level but
defined in `LayeredSpheroids`, and Documenter refuses the fallback rather than
guess. The same text in a plain `.md` page resolves in `Main` and is fine.

That asymmetry is why this checks **`src/` only**. Applied to `docs/`, the rule
flags twenty references Documenter accepts, and a check that reports
non-problems is worse than no check.

`check_docrefs.py` covers the *unqualified* refs; this covers the qualified
ones, which is the gap that let a build fail on three of them.
"""
function unqualified_module_refs()
    resolve(path) = begin
        parts = Symbol.(split(path, "."))
        obj = MFH
        parts[1] === :MeanFieldHomogenization || return nothing
        for q in parts[2:end]
            (obj isa Module && isdefined(obj, q)) || return nothing
            obj = getfield(obj, q)
        end
        obj
    end
    # Only what has an unambiguous defining module; a `@ref` to a method
    # signature or to an alias is left to Documenter.
    owner(o) = (o isa Module || o isa Function || o isa Type) ? parentmodule(o) : nothing

    out = Tuple{String, Int, String, String}[]
    for (dir, _, files) in walkdir(joinpath(pkgdir(MFH), "src")), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        for (i, line) in enumerate(eachline(path))
            for m in eachmatch(
                    r"@ref (MeanFieldHomogenization\.[A-Za-z0-9_.!]*[A-Za-z0-9_!])", line
                )
                p = m.captures[1]
                obj = resolve(p)
                obj === nothing && continue
                own = owner(obj)
                own === nothing && continue
                written = join(split(p, ".")[1:(end - 1)], ".")
                expected = replace(string(own), "Main." => "")
                written == expected ||
                    push!(out, (relpath(path, pkgdir(MFH)), i, p, expected))
            end
        end
    end
    return out
end

bad, n = scan()
println("checked $n names in @docs blocks across ", length(MODS), " modules")
isempty(bad) ? println("OK: every one carries a docstring") :
    (println("MISSING DOCSTRING ($(length(bad))):"); foreach(b -> println("  ", b), bad))

missing_entry = unlisted_exports()
println("checked ", length(names(MFH)), " exports of MFH plus its sub-modules'")
isempty(missing_entry) ? println("OK: every documented export is listed in a @docs block") :
    (
    println("NOT IN ANY @docs BLOCK ($(length(missing_entry))):");
        foreach(t -> println("  ", t[1], ".", t[2]), missing_entry)
)

qual = unqualified_module_refs()
isempty(qual) ? println("OK: every qualified @ref in a docstring names its defining module") :
    (
    println("NOT THE DEFINING MODULE ($(length(qual))):");
        foreach(
        t -> println(
            "  ", t[1], ":", t[2], "  ", t[3], "  → ", t[4], ".",
            split(t[3], ".")[end],
        ), qual,
    )
)

exit(isempty(bad) && isempty(missing_entry) && isempty(qual) ? 0 : 1)
