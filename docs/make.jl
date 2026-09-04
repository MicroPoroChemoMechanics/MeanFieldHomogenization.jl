using Documenter
using DocumenterCitations
# VitePress renders the site from the Markdown that Documenter emits. Math is
# typeset at build time into static SVG (see `docs/src/.vitepress/mathjax-plugin.ts`),
# so no MathJax bundle is ever fetched by the reader's browser; the ```mermaid
# blocks are rendered by `vitepress-plugin-mermaid`, declared in
# `docs/package.json`, which is why DocumenterMermaid is no longer a dependency.
using DocumenterVitepress
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

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style = :numeric,
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

makedocs(;
    # `clean = false` was kept here from the first commit, with no stated
    # reason. It let pages deleted from the source survive in `build/` and go
    # on being deployed: four of them were still on the site when this was
    # found. Nothing writes into `build/` before `makedocs`, so wiping it
    # costs nothing.
    modules = [
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
    ],
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
    pages = [
        "Home" => "index.md",
        # Ordered as a reading path, and grouped so that the standard theory
        # comes before what is built on top of it: conventions, then the
        # Eshelby framework and the tools it produces (Hill tensor,
        # localization, the schemes), then the specializations (cracks,
        # layered inclusions, laminates, viscoelasticity), then the N-body
        # models, and finally the appendices — pages that support the rest but
        # are written in its language rather than the other way round.
        # Grouped by what a chapter *is about*, not by how it is derived. The
        # order follows the dependency chain: the Eshelby problem and the
        # tensors it produces, then the schemes built on them, then the three
        # ways the problem is generalized — a richer pattern in place of the
        # ellipsoid, a different physics, a different time dependence — then
        # periodic homogenization, which is a different construction entirely,
        # and finally the N-body models that drop the one-site picture.
        "Theory" => [
            "theory/index.md",
            "theory/notation.md",
            "Foundations — the Eshelby problem" => [
                "theory/eshelby_problem.md",
                "theory/hill_tensors.md",
                "theory/localization.md",
            ],
            "Homogenization schemes" => [
                "theory/homogenization.md",
                "theory/differential_scheme.md",
            ],
            # A layered sphere or a confocal spheroid is not an inclusion with a
            # Hill tensor: it is a *pattern* whose generalized Eshelby problem is
            # solved for its average concentration tensor. Any pattern admitting
            # that treatment belongs here.
            "The generalized Eshelby problem — morphological patterns" => [
                "theory/layered_sphere.md",
                "theory/layered_spheroid.md",
            ],
            # A crack is a degenerate ellipsoid, so it stays close to the
            # foundations rather than joining the composite patterns.
            "Cracks" => [
                "theory/cod_tensors.md",
                "theory/thermal_cracks.md",
            ],
            "Extension to conductivity" => [
                "theory/conductivity.md",
            ],
            # Two distinct extensions: the correspondence principle, which maps a
            # non-ageing problem onto an elastic one, and the ageing case, where
            # no such map exists and the Eshelby problem itself is generalized.
            "Extension to viscoelasticity" => [
                "theory/laplace_carson.md",
                "theory/viscoelasticity.md",
            ],
            # NOT a morphological pattern: the laminate result comes out of
            # periodic homogenization, a construction of its own.
            "Periodic homogenization" => [
                "theory/laminate.md",
            ],
            "N-body models" => [
                "theory/interaction_tensors.md",
                "theory/cluster_model.md",
                "theory/eim.md",
            ],
            "Appendices" => [
                "theory/corrected_cell.md",
                "theory/elliptic_integrals.md",
            ],
        ],
        # Same principle: the inclusion families first, then the cells and
        # schemes that consume them, then what goes beyond elasticity.
        "Manual" => [
            "manual/installation.md",
            "Inclusions" => [
                "manual/inclusion_gallery.md",
                "manual/ellipsoidal_inclusions.md",
                "manual/cylindrical_inclusions.md",
                "manual/cracks.md",
                "manual/custom_inclusions.md",
                "manual/fe_inclusions.md",
                "manual/neural_inclusions.md",
            ],
            "Cells and schemes" => [
                "manual/schemes.md",
                "manual/particle_assemblies.md",
                "manual/multiscale.md",
            ],
            # Separated from the schemes for the same reason as in Theory: a
            # laminate is the closed form of a periodic problem, not a cell
            # holding inclusions.
            "Periodic homogenization" => [
                "manual/laminates.md",
            ],
            "Beyond elasticity" => [
                "manual/conductivity.md",
                "manual/viscoelasticity.md",
                "manual/rheological_models.md",
                "manual/laplace_inversion.md",
                "manual/poromechanics.md",
            ],
            "Differentiation" => [
                "manual/sensitivities.md",
            ],
            "Appendices" => [
                "manual/elliptic_examples.md",
            ],
        ],
        # One learning path, grouped by theme rather than by how the page
        # happens to be produced. Pages under `tutorials/generated/` are built
        # from `scripts/` by Literate (see `docs/literate.jl`); that is an
        # implementation detail the reader has no reason to care about, so they
        # sit alongside the hand-written ones.
        "Tutorials" => [
            "tutorials/index.md",
            "Fundamentals" => [
                "tutorials/first_estimate.md",
                "tutorials/bounds_and_schemes.md",
                "tutorials/porous_materials.md",
                "tutorials/porous_benchmark.md",
                "tutorials/transport.md",
                "tutorials/differential_paths.md",
                "tutorials/differential_loading_paths.md",
            ],
            "Inclusions, geometries and orientation" => [
                "tutorials/generated/hill_tensors.md",
                "tutorials/cracks.md",
                "tutorials/generated/crack_distributions.md",
                "tutorials/fe_crack.md",
                "tutorials/generated/layered_sphere.md",
                "tutorials/generated/layered_spheroid_effective.md",
                "tutorials/generated/layered_spheroid_interfaces.md",
                "tutorials/generated/layered_spheroid_hc.md",
                "tutorials/generated/nano_spheroids.md",
                "tutorials/generated/laminate.md",
                "tutorials/generated/laminate_interfaces.md",
                "tutorials/generated/symmetrization.md",
                "tutorials/generated/custom_inclusion_contract.md",
                "tutorials/generated/neural_inclusion.md",
                "tutorials/generated/neural_excentered_sphere.md",
            ],
            # After the inclusion families, since an N-body tutorial assumes
            # the reader knows them. `nano_spheroids` is NOT here: it condenses
            # a single particle's interface into an equivalent stiffness and
            # feeds an ordinary Mori-Tanaka — no N-body content at all.
            "Interacting particle assemblies" => [
                "tutorials/generated/cluster_model.md",
                "tutorials/generated/multiscale_assemblies.md",
                "tutorials/generated/eim_assembly.md",
            ],
            "Beyond elasticity" => [
                "tutorials/viscoelasticity.md",
                "tutorials/generated/rheological_models.md",
                "tutorials/generated/kelvin_maxwell.md",
                "tutorials/generated/laplace_inversion.md",
                "tutorials/generated/freq_vs_time.md",
                "tutorials/generated/alv_schemes.md",
                "tutorials/generated/ageing_ages_aspect.md",
                "tutorials/generated/alv_sensitivities.md",
                "tutorials/generated/laminate_alv.md",
            ],
            "Differentiation and solvers" => [
                "tutorials/sensitivities.md",
                "tutorials/strength_criteria.md",
                "tutorials/nonlinear_solvers.md",
                "tutorials/generated/secant_elastoplasticity.md",
            ],
            "Interoperability and tools" => [
                "tutorials/symbolic_spheres.md",
                "tutorials/symbolic_laminate.md",
                "tutorials/symbolic_viscoelasticity.md",
                "tutorials/generated/laminate_multiscale.md",
            ],
        ],
        "Applications" => [
            "applications/cement_paste.md",
            "applications/hydrating_blended_paste.md",
            "applications/ionic_hydrating_paste.md",
            "applications/cement_paste_diffusion.md",
            "applications/strength.md",
            "applications/itz_concrete.md",
            "applications/recycled_aggregate.md",
            "applications/bituminous.md",
            "applications/ageing_creep.md",
            "applications/lamellar_clay.md",
        ],
        # Getting work into and out of MeanFieldHomogenization. These are companions to
        # the library rather than chapters about it, which is why they sit
        # together at the end of the user-facing material instead of
        # interrupting the manual.
        "Tools and migration" => [
            "tools/from_echoes.md",
            "tools/echoes2mfh.md",
            "tools/mfhstudio.md",
        ],
        # MeanFieldHomogenization *inside* a finite-element code — the exact
        # opposite of `manual/fe_inclusions.md`, which is the FE solver inside
        # MeanFieldHomogenization. Kept as its own top-level section so the two
        # can never be read as a continuation of one another.
        # Sorted the way the section is read: the equations first, then how to
        # build a model with them, then worked models.
        "Finite-element coupling" => [
            "fe_coupling/index.md",
            "Theory" => [
                "fe_coupling/scale_transition.md",
                "fe_coupling/poroelastic_coupling.md",
                "fe_coupling/permeability.md",
            ],
            "Manual" => [
                "fe_coupling/materials.md",
                "fe_coupling/fractured_rock.md",
                "fe_coupling/backends.md",
            ],
            "Examples" => [
                "fe_coupling/thick_cylinder.md",
                "fe_coupling/arma2011.md",
            ],
        ],
        "Developer" => [
            "developer/architecture.md",
            "developer/adding_inclusion.md",
            "developer/adding_algorithm.md",
            "developer/adding_scheme.md",
            "developer/testing_conventions.md",
            "developer/validation.md",
            "developer/performance_notes.md",
            "developer/benchmarks.md",
            "developer/roadmap.md",
        ],
        "API" => [
            "api/elliptic.md",
            "api/core.md",
            "api/elasticity.md",
            "api/cracks.md",
            "api/conductivity.md",
            "api/localization.md",
            "api/layered_sphere.md",
            "api/layered_spheroid.md",
            "api/laminate.md",
            "api/interactions.md",
            "api/schemes.md",
            "api/poromechanics.md",
            "api/constitutive.md",
            "api/assemblies.md",
            "api/viscoelasticity.md",
            "api/laplace_carson.md",
            "api/sensitivities.md",
        ],
        "References" => "references.md",
    ],
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
    warnonly = [:docs_block],
)

# DocumenterVitepress writes a real directory per version rather than the
# symlinks Documenter used, so it needs its own `deploydocs`.
DocumenterVitepress.deploydocs(;
    repo = "github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = false,
)
