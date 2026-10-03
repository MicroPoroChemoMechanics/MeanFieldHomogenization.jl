# ── Documentation shards: the pages of a pull request, built in parallel ─────
#
# The full site is built in one process, page after page, and a pull request
# used to wait for all of it before learning whether one `@example` block
# failed. A shard is a group of pages built on its own runner with
# `MFH_DOCS_ONLY` (see `partial.jl`), so the pull-request jobs run side by side
# and the answer arrives after the slowest shard rather than after the sum.
#
# What the shards prove together is what a partial build proves, for every page
# at once: that every executed block runs and every page renders. What none of
# them proves is that the cross references resolve, since each prunes the pages
# of the others; the draft pre-flight job checks those on the whole tree, and
# the full build on `main` remains the one that deploys.
#
# The groups follow what the pages execute, not a measured cost: the two
# hydration pages couple a chemical solver to the micromechanics, the generated
# tutorials train surrogates and integrate ageing kernels, and the rest is
# lighter. The shard jobs run with `JULIA_DEBUG=Documenter`, which names each
# page as it is expanded, so their logs give the time per page to rebalance
# from. The pages no group names fall into the last one, `rest`, so a new page
# is built by a pull request without anyone having to assign it. A page
# belongs to the FIRST group with a pattern that matches it, so a specific group
# placed before a broader one takes its pages out of it, and no page is built
# twice. A pattern that matches no page is refused rather than ignored: a
# renamed page would otherwise leave its group silently empty.
#
# Ported from ChemistryLab's `docs/shards.jl`.

const DOC_SHARDS = [
    # Hydration coupled to the chemistry of the pore solution.
    "chemistry" => [
        "applications/hydrating_blended_paste.md",
        "applications/ionic_hydrating_paste.md",
    ],
    # The other cementitious applications, which share the paste models.
    "cement" => [
        "applications/cement_paste.md",
        "applications/cement_paste_diffusion.md",
        "applications/ageing_creep.md",
        "applications/strength.md",
        "applications/itz_concrete.md",
        "applications/itz_elastic_limit.md",
    ],
    # The general concepts, the geomaterials and the bituminous mixture.
    "applications" => [
        "applications/generated/cluster_model.md",
        "applications/generated/eim_assembly.md",
        "applications/recycled_aggregate.md",
        "applications/concave_pores.md",
        "applications/lamellar_clay.md",
        "applications/granular_friction.md",
        "applications/sandstone_strength.md",
        "applications/bituminous.md",
    ],
    # The generated tutorials on viscoelasticity, which integrate ageing kernels
    # and invert transforms; they come before the broader group below.
    "viscoelastic" => [
        "tutorials/generated/alv_",
        "tutorials/generated/ageing_",
        "tutorials/generated/laminate_alv",
        "tutorials/generated/freq_vs_time",
        "tutorials/generated/laplace_inversion",
        "tutorials/generated/kelvin_maxwell",
        "tutorials/generated/rheological_models",
    ],
    # The other tutorials generated from `scripts/` by Literate.
    "generated" => ["tutorials/generated/"],
    "theory-manual" => ["theory/", "manual/"],
    # Everything else: the hand-written tutorials, the finite-element coupling,
    # the tools, the developer pages, the API and the front pages.
    "rest" => String[],
]

"""
    doc_page_leaves(node) -> Vector{String}

Every page path of a `pages` tree, in order. Duplicated from `partial.jl` so that
this file can be read without triggering a partial build.
"""
doc_page_leaves(node::AbstractString) = [node]
doc_page_leaves(node::Pair) = doc_page_leaves(node.second)
doc_page_leaves(node::AbstractVector) =
    reduce(vcat, doc_page_leaves.(node); init = String[])

"""
    shard_pages(name, leaves) -> Vector{String}

The pages of shard `name` among `leaves`. A page belongs to the first shard with
a pattern that matches it; the shard `rest` holds every page no other shard
claims. Throws on an unknown name and on a pattern that matches no page.
"""
function shard_pages(name::AbstractString, leaves::Vector{String})
    named = [s for s in DOC_SHARDS if !isempty(s.second)]
    for (shard, pats) in named, p in pats
        any(pg -> occursin(p, pg), leaves) || error(
            "documentation shard `$shard`: pattern `$p` matches no page of " *
                "docs/pages.jl; rename it with the page it stood for.",
        )
    end
    function owner(pg)
        i = findfirst(s -> any(p -> occursin(p, pg), s.second), named)
        return i === nothing ? "rest" : named[i].first
    end
    any(s -> s.first == name, DOC_SHARDS) || error(
        "unknown documentation shard `$name`; known: " *
            join(first.(DOC_SHARDS), ", "),
    )
    return filter(pg -> owner(pg) == name, leaves)
end
