# -*- coding: utf-8 -*-
"""
Local-κ backends (s–t flow) — NetworkX / scipy / igraph.

Contract: ``local_kappa`` returns int ≥ 0 and respects ``cutoff`` as a cap.
The backend choice comes from StrategyPlan (strategy.py); this module only runs it.
SciPy is the default; NetworkX and igraph only if the CLI asks.
"""

from __future__ import annotations

from networkx.algorithms.connectivity import build_auxiliary_node_connectivity

from maxc.strategy import BACKEND_IGRAPH, BACKEND_NX, BACKEND_SCIPY


def auxiliary_to_csr(aux):
    """Convert the NetworkX auxiliary digraph → (csr_matrix, pos, mapping)."""
    from scipy.sparse import csr_matrix

    mapping = aux.graph["mapping"]
    nodes = list(aux.nodes())
    pos = {n: i for i, n in enumerate(nodes)}
    rows, cols, data = [], [], []
    for a, b, attr in aux.edges(data=True):
        rows.append(pos[a])
        cols.append(pos[b])
        data.append(int(attr.get("capacity", 0)))
    mat = csr_matrix(
        (data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=int
    )
    return mat, pos, mapping


def build_igraph(G):
    """Build ig.Graph + node→index map once (reuse in the worker)."""
    import igraph as ig

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[a], idx[b]) for a, b in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    return g, idx


def _kappa_networkx(G, u, v, auxiliary, residual, flow_fn, cutoff):
    """NetworkX backend (exact). ``flow_fn`` is the callable passed to ``flow_func``."""
    from networkx.algorithms.connectivity import local_node_connectivity

    kwargs = {
        "auxiliary": auxiliary,
        "cutoff": cutoff,
        "flow_func": flow_fn,
    }
    if residual is not None:
        kwargs["residual"] = residual
    return int(local_node_connectivity(G, u, v, **kwargs))


def _kappa_igraph(G, u, v, cutoff, igraph=None, igraph_idx=None):
    """
    igraph backend (vertex_connectivity between two vertices).

    Prefers the worker's pre-built graph; otherwise builds on demand.
    igraph does not take cutoff the same way; the value is exact, then capped.
    """
    if igraph is None or igraph_idx is None:
        igraph, igraph_idx = build_igraph(G)
    k = int(igraph.vertex_connectivity(igraph_idx[u], igraph_idx[v], neighbors="ignore"))
    if cutoff is not None:
        return min(k, int(cutoff))
    return k


def _kappa_scipy(
    G, u, v, cutoff, scipy_mat=None, scipy_pos=None, scipy_map=None
):
    """
    scipy backend: max-flow on the NetworkX auxiliary digraph (same reduction).

    If mat/pos/map are passed (worker), reuse them; otherwise build on demand.
    maximum_flow mutates the matrix — copy so the worker cache stays intact.
    """
    from scipy.sparse.csgraph import maximum_flow

    if scipy_mat is None or scipy_pos is None or scipy_map is None:
        H = build_auxiliary_node_connectivity(G)
        mat, pos, mapping = auxiliary_to_csr(H)
    else:
        mat, pos, mapping = scipy_mat, scipy_pos, scipy_map

    s_name = f"{mapping[u]}B"
    t_name = f"{mapping[v]}A"
    flow = maximum_flow(mat.copy(), pos[s_name], pos[t_name])
    k = int(flow.flow_value)
    if cutoff is not None:
        return min(k, int(cutoff))
    return k


def local_kappa(
    G,
    u,
    v,
    *,
    auxiliary=None,
    residual=None,
    flow_fn=None,
    cutoff=None,
    backend: str = BACKEND_NX,
    scipy_mat=None,
    scipy_pos=None,
    scipy_map=None,
    igraph=None,
    igraph_idx=None,
):
    """
    Exact κ(u,v) via the chosen backend (package default: SciPy in the plan).

    Contract: returns int ≥ 0; respects cutoff as a cap on the useful value.
    ``flow_fn`` is the NetworkX callable (when backend=networkx).
    """
    if backend == BACKEND_SCIPY:
        return _kappa_scipy(
            G,
            u,
            v,
            cutoff,
            scipy_mat=scipy_mat,
            scipy_pos=scipy_pos,
            scipy_map=scipy_map,
        )
    if backend == BACKEND_IGRAPH:
        return _kappa_igraph(
            G, u, v, cutoff, igraph=igraph, igraph_idx=igraph_idx
        )
    return _kappa_networkx(G, u, v, auxiliary, residual, flow_fn, cutoff)
