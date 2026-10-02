using Documenter
using DocumenterCitations
# VitePress renders the site from the Markdown that Documenter emits. Math is
# typeset at build time into static SVG (see `docs/src/.vitepress/mathjax-plugin.ts`),
# so no MathJax bundle is ever fetched by the reader's browser; the ```mermaid
# blocks are rendered by `vitepress-plugin-mermaid`, declared in
# `docs/package.json`, which is why DocumenterMermaid is no longer a dependency.
using DocumenterVitepress
using Logging
using MeanFieldHomogenization

# GR needs a headless display driver on CI runners; without this the figures in
# the Applications pages fail to render.
ENV["GKSwstype"] = "100"

# Composite panels crop their outer labels unless the margins are explicit — a
# big enough canvas is not enough on its own. The Literate scripts each set
# their own left/bottom margins; setting the defaults once here extends the same
# treatment to the hand-written `@example` pages, which never picked up that
# idiom. Must come before `literate.jl`, whose notebook pass runs the scripts.
using Plots
Plots.gr()
Plots.default(;
    left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
    right_margin = 4Plots.mm,
    top_margin = 3Plots.mm,
)

# Generates the Gallery pages (+ companion notebooks/scripts) from the
# curated `scripts/` demos before `makedocs` runs, so the generated markdown
# exists when `pages` below references it. Must run after the GKSwstype
# assignment above — the Literate notebook pass actually executes the
# scripts, including their Plots/GR calls.
include("literate.jl")

# The page tree, read by both the draft pre-flight and the real build.
include("pages.jl")
# Reads `pages`, and may replace it. See its header for why a partial build
# needs a pruned SOURCE tree and not just a pruned page tree. After
# `literate.jl`, so that the generated pages exist when the tree is pruned.
include("partial.jl")

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :authoryear,
)

# ── Stopgap: CitationSiteNode in the Markdown writer ─────────────────────────
#
# GUARDED, and the guard is what makes one file serve both versions.
#
# `CitationSiteNode` exists only in DocumenterCitations 1.5, while
# DocumenterVitepress declares a weak dependency `DocumenterCitations = "1 - 1.4"`
# — an upstream cap, so the docs environment resolves to 1.4 and the name is not
# there. Referring to it unconditionally is an `UndefVarError` raised before the
# first page is built, which is exactly how this was found. On 1.4 the citations
# need no help at all: the node is what 1.5 introduced.
#
# When DocumenterVitepress raises its cap, widening the bound in
# `docs/Project.toml` is the only edit needed — this method starts applying again
# on its own.
if isdefined(DocumenterCitations, :CitationSiteNode)
    # DocumenterCitations 1.5 wraps every expanded citation in a `CitationSiteNode`,
    # whose only purpose is to give the citation an HTML anchor so the bibliography
    # can link back to it. Its own docstring calls it "transparent in any output
    # format other than HTML", and both the LaTeX writer and MDFlatten implement it
    # as "render my children".
    #
    # DocumenterVitepress 0.3.5 ships a DocumenterCitations extension, but it covers
    # only `BibliographyNode`. With no method for `CitationSiteNode`, the writer
    # falls through to its generic branch, which prints `Markdown.plain(element)` —
    # so every one of this manual's citations came out as the literal text
    # `DocumenterCitations.CitationSiteNode("kachanov1992-cite-1")`.
    #
    # The same one-line treatment as the other non-HTML writers. Remove this once
    # DocumenterVitepress covers the node upstream.
    function DocumenterVitepress.render(
            io::IO,
            mime::MIME"text/plain",
            node::Documenter.MarkdownAST.Node,
            ::DocumenterCitations.CitationSiteNode,
            page,
            doc;
            kwargs...,
        )
        return DocumenterVitepress.render(
            io, mime, node, node.children, page, doc; kwargs...,
        )
    end
end

DocMeta.setdocmeta!(
    MeanFieldHomogenization,
    :DocTestSetup,
    :(using MeanFieldHomogenization);
    recursive = true,
)

# ── Stopgap: heading anchors that contain LaTeX ──────────────────────────────
# DocumenterVitepress builds each heading as `## <text> {#<slug>}`, where the
# slug is Documenter's anchor label passed through its own
# `sanitized_anchor_label` — whose comment says "vitepress doesn't like special
# markdown characters in the id slug", but which only strips `[ ] ( ) *`.
#
# A heading such as `## The COD tensor ``\boldsymbol{B}``` yields the slug
# `The-COD-tensor-\boldsymbol{B}`. VitePress's `{#...}` parser rejects the
# backslash and the braces, so it treats the whole suffix as *text*: the heading
# renders as "The COD tensor {#The-COD-tensor-\boldsymbol{B}}", the formula is
# dropped, and the same garbage lands in the "On this page" outline. Twenty-three
# headings across seven pages were affected.
#
# Stripping those characters from the slug is safe here: nothing links to those
# anchors (checked against every `](…#…)` destination in the generated Markdown),
# and this narrows to headings only, leaving docstring anchors — which legitimately
# carry braces, are emitted as raw `<a id=…>`, and *are* linked to — untouched.
#
# Remove once `sanitized_anchor_label` covers these characters upstream.
function DocumenterVitepress.render(
        io::IO,
        mime::MIME"text/plain",
        node::Documenter.MarkdownAST.Node,
        header::Documenter.AnchoredHeader,
        page,
        doc;
        kwargs...,
    )
    anchor = header.anchor
    label = DocumenterVitepress.sanitized_anchor_label(anchor)
    id = replace(replace(label, r"[\\{}]" => ""), " " => "-")
    heading = first(node.children)
    println(io)
    print(io, "#"^(heading.element.level), " ")
    heading_iob = IOBuffer()
    DocumenterVitepress.render(heading_iob, mime, node, heading.children, page, doc; kwargs...)
    print(io, rstrip(String(take!(heading_iob))))
    print(io, " {#$(id)}")
    if haskey(kwargs, :inventory)
        item = DocumenterVitepress.InventoryItem(
            name = anchor.id,
            domain = "std",
            role = "label",
            dispname = DocumenterVitepress._get_inventory_dispname(
                anchor.id, Documenter.MDFlatten.mdflatten(anchor.node)
            ),
            priority = -1,
            uri = DocumenterVitepress._get_inventory_uri(doc, page, id),
        )
        push!(kwargs[:inventory], item)
    end
    println(io)
    return nothing
end

# ── Stopgap: ordered lists start at 2, and swallow their first item ──────────
# DocumenterVitepress numbers ordered-list items with `bullet(i) = "$(i+1). "`,
# but `enumerate` is already 1-based: every ordered list in the manual came out
# numbered from 2. It also emits no blank line before the list.
#
# Together those two produce the damage seen on the References page. A list whose
# first marker is `2.` cannot interrupt a paragraph — CommonMark allows that only
# for a list starting at `1.` — so, with no blank line to separate them, the first
# entry was absorbed into the preceding prose as plain text and the list began at
# `3.`. Eighteen ordered lists across the manual were affected; not one of them
# started at 1.
#
# Remove once the numbering is fixed upstream.
function DocumenterVitepress.render(
        io::IO,
        mime::MIME"text/plain",
        node::Documenter.MarkdownAST.Node,
        list::Documenter.MarkdownAST.List,
        page,
        doc;
        kwargs...,
    )
    bullet(i) = list.type === :ordered ? "$(i). " : "- "
    println(io)
    iob = IOBuffer()
    for (i, item) in enumerate(node.children)
        DocumenterVitepress.render(
            iob, mime, item, item.children, page, doc; prenewline = false, kwargs...
        )
        eachline = split(String(take!(iob)), '\n')
        # Continuation lines must line up with the text, i.e. under the marker's
        # full width. Upstream hard-codes two spaces, which fits `- ` but not
        # `1. `: a display equation inside an ordered item fell out of the list,
        # splitting it in two and restarting the numbering.
        pad = " "^length(bullet(i))
        eachline[2:end] .= pad .* eachline[2:end]
        final_string = join(eachline, '\n')
        endswith(final_string, '\n') || (final_string *= "\n")
        print(io, bullet(i))
        print(io, final_string)
    end
    return nothing
end

# ── THREE PRE-FLIGHTS, BECAUSE THE CHECKS THAT MATTER RUN LAST ───────────────
#
# `ExpandBibliography`, `CrossReferences` and `CheckDocument` are among the LAST
# stages of `makedocs`: they run after every `@example` block of the site has
# been executed. So a malformed bibliography entry, an `@ref` naming an anchor
# that does not exist, or an exported name on no curated page does not fail the
# build in seconds — it fails it once the whole computation has been spent, and
# terminates "before rendering", so that time buys nothing. In ChemistryLab,
# whose pages solve chemical equilibria, that cost a 151-minute build and then a
# 70-minute one. The same three guards are ported here.
#
# What is checked here is only what can be checked cheaply. Documenter's own
# checks still run at the end; these move the common failures to the front.

# ── 1. the bibliography is formatted NOW ─────────────────────────────────────
#
# Calls exactly what the late stage calls, so it cannot drift away from what it
# is guarding. A LaTeX escape the TeX parser does not implement is the usual
# cause — `CNASH\_ss` threw `ArgumentError: Invalid command: \_ss` and took a
# three-hour build down at its last stage. Write the character bare inside
# braces instead.
let failures = String[]
    for (key, entry) in bib.entries
        try
            DocumenterCitations.format_bibliography_reference(:authoryear, entry)
        catch err
            push!(failures, "  $key : " * sprint(showerror, err))
        end
    end
    isempty(failures) || error(
        "docs/src/references.bib has $(length(failures)) entry/entries " *
            "DocumenterCitations cannot format. This would otherwise kill the " *
            "build at its LAST stage, after every example has run:\n" *
            join(failures, "\n")
    )
end

# ── 2. the `@ref` anchors written in markdown are resolved NOW ───────────────
#
# This complements `.github/scripts/check_docrefs.py`, which covers the other
# half of the problem: an `@ref` inside a DOCSTRING, resolved in the module that
# docstring lives in. What is checked here is an `@ref` inside a markdown PAGE,
# resolved against the `(@id ...)` anchors and the header slugs.
#
# Both forms. The explicit one, `[text](@ref some-anchor)`, catches the typo in a
# hand-written anchor. The bare one, `[Some Heading](@ref)`, resolves against the
# heading TEXT — slugified and case-sensitively — so a heading renamed or merely
# recapitalized silently breaks every link to it; three were broken that way in
# ChemistryLab. A bare ref whose text is a code span is a docstring name instead
# and is left to Documenter.
let
    srcdir = joinpath(@__DIR__, "src")
    mds = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        endswith(f, ".md") && push!(mds, joinpath(root, f))
    end

    anchors = Set{String}()
    for f in mds
        text = read(f, String)
        for m in eachmatch(r"\(@id\s+([^)]+?)\s*\)", text)
            push!(anchors, m.captures[1])
        end
        # Fenced blocks are removed first: a Julia comment opens with `#` too,
        # and counting those as headers would invent anchors that mask a typo.
        prose = replace(text, r"^```.*?^```"ms => "")
        for m in eachmatch(r"^#+\s+(.+?)\s*$"m, prose)
            title = m.captures[1]
            occursin("(@id", title) && continue
            push!(anchors, replace(strip(title), r"\s+" => "-"))
        end
    end

    unresolved = String[]
    for f in mds
        text = read(f, String)
        for m in eachmatch(r"\]\(@ref\s+([^)]+?)\s*\)", text)
            target = m.captures[1]
            # No hyphen and no space: a docstring name, which only Documenter
            # can resolve.
            occursin('-', target) || continue
            startswith(target, '`') && continue
            target in anchors ||
                push!(unresolved, "  " * relpath(f, srcdir) * " -> @ref " * target)
        end
        for m in eachmatch(r"\[([^]]+)\]\(@ref\)", text)
            label = strip(m.captures[1])
            startswith(label, '`') && continue
            slug = replace(label, r"\s+" => "-")
            slug in anchors ||
                push!(unresolved, "  " * relpath(f, srcdir) * " -> [" * label * "](@ref)")
        end
    end
    isempty(unresolved) || error(
        "$(length(unresolved)) cross-reference(s) name an anchor that does not " *
            "exist. Documenter would report this only at its `CrossReferences` " *
            "stage, after every example on the site has run:\n" *
            join(sort(unique(unresolved)), "\n")
    )
end


# ── Guard: no heading may contain a percent sign ─────────────────────────────
#
# A heading's anchor is its text, so `## Porosity of 30 %` gets the anchor
# `Porosity-of-30-%`. VitePress passes every link destination through
# `decodeURI`, a `%` that does not open an escape makes it throw, and the site
# build stops with "URI malformed" -- at its very last stage, after every example
# has run. It happened once in ChemistryLab. Write "percent" or leave the number
# out of the heading; the prose keeps its `%`.
let
    srcdir = joinpath(@__DIR__, "src")
    offenders = String[]
    for (root, _, files) in walkdir(srcdir), f in filter(endswith(".md"), files)
        path = joinpath(root, f)
        prose = replace(read(path, String), r"^```.*?^```"ms => "")
        for m in eachmatch(r"^#+\s+(.+?)\s*$"m, prose)
            occursin('%', m.captures[1]) &&
                push!(offenders, "  " * relpath(path, srcdir) * ": " * strip(m.match))
        end
    end
    isempty(offenders) || error(
        "a heading contains `%`, which ends up in its anchor and makes VitePress " *
            "stop on \"URI malformed\" at the end of the build:\n" * join(offenders, "\n")
    )
end

# The module list, lifted out for the same reason as `pages`: the draft
# pre-flight must check exactly the modules the real build checks.
const DOC_MODULES = [
    MeanFieldHomogenization,
    MeanFieldHomogenization.Elliptic,
    MeanFieldHomogenization.Core,
    MeanFieldHomogenization.Elasticity,
    MeanFieldHomogenization.Cracks,
    MeanFieldHomogenization.Conductivity,
    MeanFieldHomogenization.LayeredSpheres,
    MeanFieldHomogenization.LayeredSpheroids,
    MeanFieldHomogenization.Interactions,
    MeanFieldHomogenization.Schemes,
    MeanFieldHomogenization.Assemblies,
    MeanFieldHomogenization.Laminates,
    MeanFieldHomogenization.Poromechanics,
    MeanFieldHomogenization.Constitutive,
    MeanFieldHomogenization.Viscoelasticity,
    MeanFieldHomogenization.CustomInclusions,
    MeanFieldHomogenization.FiniteElements,
    MeanFieldHomogenization.NeuralInclusions,
]

# ── 3. a DRAFT build, which is the one that closes the class ────────────────
#
# `checkdocs` and the cross-reference resolution happen in `CheckDocument`, which
# runs AFTER `ExpandTemplates` — so an exported name on no curated page is
# reported once every figure on the site has been drawn, and the build then
# terminates before rendering. A draft build runs the same pipeline with the
# `@example` blocks skipped and reaches the same checks in seconds; measured at
# 24 s on ChemistryLab's site.
#
# Two things are turned off in this pass, and both because draft mode breaks
# them by construction rather than because the source is wrong:
#
#   * `cross_references` — the figures on the Gallery and Applications pages are
#     written by the blocks themselves, so with the blocks skipped every
#     `![](...)` pointing at one is an invalid local link. The static check
#     above covers the anchors instead.
#   * `size_threshold` — an HTML-renderer limit; this site is rendered by
#     DocumenterVitepress, which has none.
#
# `checkdocs` is what this pass exists for, and it stays strict. Its own
# `CitationBibliography`, because the plugin carries state across a build and the
# real pass must start from a fresh one.
#
# SKIPPED IN A PARTIAL BUILD: a partial build prunes the API pages, so every
# docstring on them would be reported missing. Docstring coverage is a property
# of the whole site, and a partial build says up front that it does not check it.

# ── A logger that records while it passes messages through ───────────────────
#
# The draft pre-flight has to demote `cross_references`, since with the
# `@example` blocks skipped every figure they write is a broken local link. But
# an unresolvable `@ref` is reported in that same class and is perfectly
# decidable in a draft, so the messages are kept and sifted afterwards.
const _PREFLIGHT_LOG = IOBuffer()

struct RecordingLogger{L <: AbstractLogger} <: AbstractLogger
    inner::L
    sink::IOBuffer
end

Logging.min_enabled_level(l::RecordingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::RecordingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::RecordingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(
        l::RecordingLogger, level, message, _module, group, id, file, line; kwargs...
    )
    println(l.sink, message)
    return Logging.handle_message(
        l.inner, level, message, _module, group, id, file, line; kwargs...
    )
end

PARTIAL_BUILD || let t0 = time()
    @info "pre-flight: draft build (checks only, no example executed)"
    mktempdir() do draftdir
        Logging.with_logger(RecordingLogger(Logging.current_logger(), _PREFLIGHT_LOG)) do
            makedocs(;
                modules = DOC_MODULES,
                remotes = nothing,
                authors = "Jean-François Barthélémy",
                sitename = "MeanFieldHomogenization.jl",
                format = Documenter.HTML(;
                    edit_link = nothing, repolink = nothing,
                    size_threshold = nothing, size_threshold_warn = nothing,
                ),
                source = DOCS_SOURCE,
                build = draftdir,
                pages = pages,
                plugins = [
                    CitationBibliography(
                        joinpath(@__DIR__, "src", "references.bib"); style = :authoryear
                    ),
                ],
                checkdocs = :exports,
                warnonly = [:docs_block, :cross_references, :example_block, :linkcheck],
                draft = true,
            )
        end
    end
    # The half of `cross_references` a draft build can judge: an `@ref` naming a
    # binding with no docstring does not depend on any example having run, and
    # left in the warning pile it terminates the real build at its last stage.
    let failures = [
            String(m.match) for m in eachmatch(
                    r"Cannot resolve @ref for [^\n]+", String(take!(_PREFLIGHT_LOG))
                )
        ]
        isempty(failures) || error(
            "$(length(failures)) unresolvable `@ref` in a rendered page or docstring. " *
                "The real build reaches this only after every example has run:\n  " *
                join(unique(failures), "\n  ")
        )
    end
    @info "pre-flight: draft build clean" seconds = round(time() - t0; digits = 1)
    # `MFH_DOCS_PREFLIGHT_ONLY=1` stops here. Everything above is decided without
    # running a single `@example`: missing docstrings, unresolvable `@ref`, the
    # page tree, the bibliography and the heading rules.
    if !isempty(strip(get(ENV, "MFH_DOCS_PREFLIGHT_ONLY", "")))
        @info "MFH_DOCS_PREFLIGHT_ONLY is set: stopping before the real build"
        exit(0)
    end
end


makedocs(;
    # `clean = false` was kept here from the first commit, with no stated
    # reason. It let pages deleted from the source survive in `build/` and go
    # on being deployed: four of them were still on the site when this was
    # found. Nothing writes into `build/` before `makedocs`, so wiping it
    # costs nothing.
    modules = DOC_MODULES,
    remotes = nothing,
    authors = "Jean-François Barthélémy",
    sitename = "MeanFieldHomogenization.jl",
    # The favicon and the logo are picked up automatically from `docs/src/assets`,
    # and the sidebar is derived from `pages` below, so neither needs declaring
    # here. There is no page-size ceiling to raise either: the inline data of the
    # interactive 3D figures goes through Vite rather than through Documenter's
    # `size_threshold` guard.
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://MicroPoroChemoMechanics.github.io/MeanFieldHomogenization.jl",
        description = "Mean-field homogenization of heterogeneous materials in Julia",
    ),
    plugins = [bib],
    # `src`, or the pruned tree a partial build stands up. Filtering `pages`
    # alone would leave every other page still executing.
    source = DOCS_SOURCE,
    pages = pages,
    # Only exported names have to appear on a curated page: the internals in
    # the eighteen sub-modules above are documented for the reader of the
    # source, not for the site. Same setting as TensND and DECUHR.
    checkdocs = :exports,
    # NOT `warnonly = true`. A blanket exemption is why a `[`set_amount!`](@ref)
    # pointing at an undocumented internal reached CI as a VitePress "dead
    # link" with a Rollup stack trace instead of a named file and line.
    # `:docs_block` stays exempt because DocumenterCitations' `@bibliography`
    # handling and the re-exported TensND names trip it; everything else —
    # cross-references, doctests, example blocks — is now an error.
    #
    # A partial build can resolve neither a reference into a page it did not
    # build nor a docstring whose API page it pruned, so those two are demoted
    # THERE AND ONLY THERE.
    warnonly = PARTIAL_BUILD ?
        [:docs_block, :cross_references, :missing_docs, :linkcheck] :
        [:docs_block],
)

# DocumenterVitepress writes a real directory per version rather than the
# symlinks Documenter used, so it needs its own `deploydocs`.
#
# A partial build must never reach this: `deploydocs` replaces the deployed tree
# with what is in `build/`, so deploying a three-page site would take every
# other page offline.
if PARTIAL_BUILD
    @info "partial build: deployment skipped"
else
    DocumenterVitepress.deploydocs(;
        repo = "github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl.git",
        target = joinpath(@__DIR__, "build"),
        branch = "gh-pages",
        devbranch = "main",
        push_preview = false,
    )
end
