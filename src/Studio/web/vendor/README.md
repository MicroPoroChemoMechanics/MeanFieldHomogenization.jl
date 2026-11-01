# Vendored assets

`plotly.min.js` is the official **plotly-gl3d** partial bundle (2.35.2), which
carries `surface`, `mesh3d`, `scatter3d` and 2-D `scatter` — everything the
studio draws — at about a third of the size of the full distribution.

It is vendored rather than loaded from a CDN so the interface works offline and
on a machine with no outbound network, which is the common case for a compute
node. Refresh it with:

    curl -sSL -o plotly.min.js https://cdn.plot.ly/plotly-gl3d-2.35.2.min.js
