# ── The full site in shards: pages executed in parallel, the site rendered once ─
#
# Nearly all the time of the documentation goes into the `@example` blocks, and
# one process runs them page after page: well over two hours for the site. The
# shards of `docs/shards.jl` split that work over runners that run side by side,
# and one last job puts the site together and deploys it. Two modes of
# `make.jl`, each set by an environment variable:
#
#     MFH_DOCS_SHARD=theory-manual julia --project=docs docs/make.jl
#     MFH_DOCS_ASSEMBLE=<dir>      julia --project=docs docs/make.jl
#
# A SHARD builds the WHOLE page tree, but executes only its own pages: every
# other page is built as a draft (`Draft = true` in an `@meta` block put in front
# of it, in a copy of the source tree), so its blocks are shown and not run. The
# page tree, the anchors and the docstrings are those of the whole site, so every
# `@ref` of an executed page resolves exactly as in the full build. The shard
# stops before VitePress and keeps, from the Markdown that Documenter emitted,
# its own pages and the files their blocks wrote (figures), in
# `docs/shard_out/<shard>`.
#
# The ASSEMBLY builds the whole tree once more, every page as a draft (a few
# minutes), and before VitePress renders it replaces the Markdown of each page by
# the one the shard that executed it emitted. VitePress then renders the site
# whole, with dead links refused, and `deploydocs` publishes it as the single
# build did. A page that no shard supplied, or that two supplied, stops the
# assembly by name: a draft page must never be deployed in place of the page.
#
# The replacement happens in `vitepress_config_transform`, the hook that
# DocumenterVitepress calls for each plugin after it has written the Markdown of
# every page and before it runs VitePress (`extension_hooks.jl` of
# DocumenterVitepress 0.3.10).
#
# A local link checked against a file that a block writes (a figure) cannot
# resolve in a draft page, so `cross_references` is demoted in both modes. What
# that demotion would hide is checked elsewhere: an `@ref` that resolves nowhere,
# by the draft pre-flight, which the assembly waits for; a link or an image that
# leads nowhere, by VitePress, which refuses dead links when it renders the
# assembled site.

include("shards.jl")

const DOCS_SHARD = strip(get(ENV, "MFH_DOCS_SHARD", ""))
const DOCS_ASSEMBLE = strip(get(ENV, "MFH_DOCS_ASSEMBLE", ""))
const SHARD_BUILD = !isempty(DOCS_SHARD)
const ASSEMBLY_BUILD = !isempty(DOCS_ASSEMBLE)

count((PARTIAL_BUILD, SHARD_BUILD, ASSEMBLY_BUILD)) ≤ 1 || error(
    "MFH_DOCS_ONLY, MFH_DOCS_SHARD and MFH_DOCS_ASSEMBLE select three different " *
        "builds; set one of them at most.",
)

# The Markdown that Documenter emits for a site rendered by DocumenterVitepress,
# under `build/` (its `md_output_path`).
const MD_OUTPUT = ".documenter"

"""
    source_pages(srcdir) -> Vector{String}

Every Markdown page of the source tree, as a path relative to `srcdir` with `/`
separators. Documenter builds every one of them, in the page tree or not.
"""
function source_pages(srcdir)
    found = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        endswith(f, ".md") || continue
        push!(found, replace(relpath(joinpath(root, f), srcdir), '\\' => '/'))
    end
    return sort!(found)
end

"""
    executed_pages(shard, srcdir) -> Set{String}

The pages that shard `shard` executes: those `shard_pages` gives it and, for
`rest`, the Markdown files of the source tree that the page tree does not list,
which no group could claim.
"""
function executed_pages(shard, srcdir)
    leaves = doc_page_leaves(pages)
    own = Set(shard_pages(shard, leaves))
    shard == "rest" && union!(own, setdiff(source_pages(srcdir), leaves))
    return own
end

"""
    drafted_source(srcdir, executed) -> String

A temporary copy of `srcdir` in which every page outside `executed` opens with
`Draft = true`. The executed pages are linked, not copied, as in `partial.jl`;
everything else is copied, for the reason given there.
"""
function drafted_source(srcdir, executed)
    tmp = mktempdir(; prefix = "mfh_docs_")
    for (root, _, files) in walkdir(srcdir)
        rel = relpath(root, srcdir)
        mkpath(normpath(joinpath(tmp, rel)))
        for f in files
            src = abspath(joinpath(root, f))
            dst = normpath(joinpath(tmp, rel, f))
            page = replace(normpath(joinpath(rel, f)), '\\' => '/')
            if !endswith(f, ".md")
                cp(src, dst; force = true)
            elseif page in executed
                # A copy where the platform refuses a link (Windows without
                # Developer Mode); a page is only ever read.
                try
                    symlink(src, dst)
                catch err
                    err isa Base.IOError || rethrow()
                    cp(src, dst; force = true)
                end
            else
                write(dst, "```@meta\nDraft = true\n```\n\n", read(src, String))
            end
        end
    end
    return tmp
end

"""
    export_shard(builddir, srcdir, executed, outdir)

Copy into `outdir`, from the Markdown that Documenter emitted under `builddir`,
the pages in `executed` and every file a block wrote: the files that are neither
a page, nor a copy of a file of the source tree, nor the VitePress configuration
(which the assembly writes for itself).
"""
function export_shard(builddir, srcdir, executed, outdir)
    md = joinpath(builddir, MD_OUTPUT)
    isdir(md) || error("no Markdown emitted under $md: the shard build did not run to its end.")
    rm(outdir; force = true, recursive = true)
    copied = String[]
    for (root, _, files) in walkdir(md), f in files
        rel = replace(relpath(joinpath(root, f), md), '\\' => '/')
        startswith(rel, ".vitepress/") && continue
        keep = endswith(f, ".md") ? rel in executed : !isfile(joinpath(srcdir, rel))
        keep || continue
        mkpath(dirname(joinpath(outdir, rel)))
        cp(joinpath(root, f), joinpath(outdir, rel))
        push!(copied, rel)
    end
    missing_pages = setdiff(executed, copied)
    isempty(missing_pages) || error(
        "the shard emitted no Markdown for: $(join(sort!(collect(missing_pages)), ", "))",
    )
    @info "shard `$DOCS_SHARD`: $(length(executed)) pages and " *
        "$(length(copied) - length(executed)) written files exported to $outdir"
    return nothing
end

"""
    ShardOutputs(root, pages)

A Documenter plugin that carries the outputs of the shards into an assembly
build: `root` holds one directory per shard, each the `outdir` of
`export_shard`, and `pages` lists every page the assembled site must receive.
"""
struct ShardOutputs <: Documenter.Plugin
    root::String
    pages::Set{String}
    merged::Base.RefValue{Bool}
end
ShardOutputs(root, pages) = ShardOutputs(abspath(root), Set(pages), Ref(false))

"""
    merge_shard_outputs!(plugin, mddir)

Replace, under `mddir`, the Markdown of each page by the one a shard emitted, and
add the files the shards' blocks wrote. Throws, naming the pages, when a page
was emitted by two shards or by none.
"""
function merge_shard_outputs!(plugin::ShardOutputs, mddir)
    supplied = Dict{String, String}()
    for shard in readdir(plugin.root)
        dir = joinpath(plugin.root, shard)
        isdir(dir) || continue
        for (root, _, files) in walkdir(dir), f in files
            rel = replace(relpath(joinpath(root, f), dir), '\\' => '/')
            dst = joinpath(mddir, rel)
            if rel in plugin.pages
                haskey(supplied, rel) && error(
                    "page `$rel` emitted by two shards, `$(supplied[rel])` and `$shard`",
                )
                supplied[rel] = shard
                cp(joinpath(root, f), dst; force = true)
            elseif !isfile(dst)
                mkpath(dirname(dst))
                cp(joinpath(root, f), dst)
            end
        end
    end
    unsupplied = setdiff(plugin.pages, keys(supplied))
    isempty(unsupplied) || error(
        "$(length(unsupplied)) pages emitted by no shard, which would be deployed as " *
            "drafts: $(join(sort!(collect(unsupplied)), ", "))",
    )
    @info "assembly: $(length(supplied)) pages taken from " *
        "$(length(unique(values(supplied)))) shards"
    return nothing
end

# Called once per base (`dev`; `stable` and `vX.Y` for a tag), after the Markdown
# of every page is written and before VitePress runs; the pages are merged at the
# first call.
function DocumenterVitepress.vitepress_config_transform(plugin::ShardOutputs, config::String)
    if !plugin.merged[]
        merge_shard_outputs!(plugin, joinpath(@__DIR__, "build", MD_OUTPUT))
        plugin.merged[] = true
    end
    return config
end

const SHARD_EXECUTED = SHARD_BUILD ? executed_pages(DOCS_SHARD, joinpath(@__DIR__, "src")) :
    Set{String}()

# The source tree the build reads: the drafted copy in these two modes, that of
# `partial.jl` otherwise.
const BUILD_SOURCE = if SHARD_BUILD
    @info "documentation shard `$DOCS_SHARD`: $(length(SHARD_EXECUTED)) pages executed, " *
        "the others built as drafts"
    drafted_source(joinpath(@__DIR__, "src"), SHARD_EXECUTED)
elseif ASSEMBLY_BUILD
    isdir(DOCS_ASSEMBLE) || error("MFH_DOCS_ASSEMBLE=\"$DOCS_ASSEMBLE\" is not a directory.")
    drafted_source(joinpath(@__DIR__, "src"), Set{String}())
else
    DOCS_SOURCE
end
