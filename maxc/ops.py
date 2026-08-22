# -*- coding: utf-8 -*-
"""
Exact κ operations (compute) — do not choose a StrategyPlan.

Map:
  structural_kappa_lb     → structural lower bounds
  extract/propagate cuts  → cut upper bounds
  sort_candidates         → candidate ordering
  k-components            → Phase 2 region

s–t flow (backends) → maxc.backends.local_kappa.
Used by workers.py (tasks) and kappa.py (UB merge / k-comp index).
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from maxc.strategy import PROFILE_KCOMP_N_MAXK_BUDGET

CUT_UB_PAIR_CAP = 4000
KCOMP_N_TIME_SKIP = 80
KCOMP_WALL_LIMIT_S = 10.0
CUT_EXTRACT_K_MAX = 3

# ---------------------------------------------------------------------------
# Structural lower bounds
# ---------------------------------------------------------------------------


def structural_kappa_lb(u, v, G: nx.Graph) -> int:
    """
    Exact/partial structural LB for κ(u,v).

    - deg(u)=1 or deg(v)=1 (G connected) ⇒ κ=1
    - adjacent with a common neighbor ⇒ κ ≥ 2
    Returns 0 if nothing useful is known.
    """
    du = G.degree(u)
    dv = G.degree(v)
    if du == 1 or dv == 1:
        return 1
    if G.has_edge(u, v):
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        if (nu & nv) - {u, v}:
            return 2
    return 0


# ---------------------------------------------------------------------------
# Cut upper bounds (residual)
# ---------------------------------------------------------------------------


def extract_node_cut_from_residual(G, s, t, auxiliary, residual) -> Optional[set]:
    """
    Extract an s–t node-cut reusing an already-warmed auxiliary + residual.

    Avoids recomputing max-flow from scratch: only reads the current residual.
    Contract: set of nodes (excluding s,t), set() if adjacent, None on failure.
    """
    if G.has_edge(s, t) or G.has_edge(t, s):
        return set()
    if residual is None or auxiliary is None:
        return None
    try:
        mapping = auxiliary.graph["mapping"]
        s_name = f"{mapping[s]}B"
        t_name = f"{mapping[t]}A"
    except (KeyError, TypeError):
        return None

    # Vertices reachable from s in the residual with residual capacity > 0.
    reachable = set()
    stack = [s_name]
    reachable.add(s_name)
    while stack:
        u = stack.pop()
        for _, v, attr in residual.out_edges(u, data=True):
            cap = attr.get("capacity", 0)
            flow = attr.get("flow", 0)
            if (cap - flow) > 0 and v not in reachable:
                reachable.add(v)
                stack.append(v)

    # Saturated frontier edges → original nodes (auxiliary id).
    cut = set()
    for u in reachable:
        for _, v, attr in residual.out_edges(u, data=True):
            if v in reachable:
                continue
            cap = attr.get("capacity", 0)
            flow = attr.get("flow", 0)
            if cap > 0 and flow >= cap - 1e-12:
                for x in (u, v):
                    nid = residual.nodes[x].get("id")
                    if nid is None:
                        nid = auxiliary.nodes[x].get("id") if x in auxiliary else None
                    if nid is not None and nid not in (s, t):
                        cut.add(nid)
    return cut if cut else None


def propagate_cut_ubs(G, s, t, cut: set, pair_ub: dict, k: int) -> int:
    """
    If C separates s from t with |C|=k, then κ(u,v)≤k for u on the s-side and v on the t-side.

    Contract: updates pair_ub in-place; returns number of pairs touched.
    No G.copy(): BFS avoiding the cut. Cap |S|·|T| ≤ CUT_UB_PAIR_CAP.
    """
    if not cut or k is None:
        return 0
    blocked = set(cut)

    def component(start):
        if start in blocked:
            return set()
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for w in G.neighbors(u):
                if w in blocked or w in seen:
                    continue
                seen.add(w)
                stack.append(w)
        return seen

    side_s = component(s)
    side_t = component(t)
    if not side_s or not side_t:
        return 0
    side_s -= side_t
    if len(side_s) * len(side_t) > CUT_UB_PAIR_CAP:
        return 0
    updated = 0
    for u in side_s:
        for v in side_t:
            key = (u, v) if u < v else (v, u)
            prev = pair_ub.get(key)
            if prev is None or k < prev:
                pair_ub[key] = int(k)
                updated += 1
    return updated


# ---------------------------------------------------------------------------
# Smart candidate ordering
# ---------------------------------------------------------------------------


def sort_candidates(
    source,
    candidates,
    G,
    degree,
    neighbors,
    pair_cache,
    pair_ub,
    witnesses,
    best_lb,
    use_smart_ordering: bool,
):
    """
    Order candidates to maximize early-stop / cache hits.

    Without smart_order: neighbor → degree (baseline).
    With smart_order: lb/exact cache → neighbor → witness → degree.
    """
    neigh = neighbors if neighbors is not None else set(G.neighbors(source))
    wit_set = set(witnesses or [])

    def key_basic(t):
        return (0 if t in neigh else 1, -degree[t], t)

    if not use_smart_ordering:
        return sorted(candidates, key=key_basic)

    def key_smart(t):
        key = (source, t) if source < t else (t, source)
        exact = pair_cache.get(key)
        # Prefer already-known high pairs and a hopeful ub.
        if exact is not None:
            tier = 0
            score = -int(exact)
        else:
            lb = structural_kappa_lb(source, t, G) if G is not None else 0
            ub = pair_ub.get(key)
            # If ub ≤ best_lb, this candidate cannot improve — send it to the end.
            if ub is not None and ub <= best_lb:
                tier = 3
            elif t in wit_set:
                tier = 1
            elif t in neigh:
                tier = 1
            else:
                tier = 2
            score = -max(lb, 0)
        return (tier, score, 0 if t in neigh else 1, -degree[t], t)

    return sorted(candidates, key=key_smart)


# ---------------------------------------------------------------------------
# k-components regions
# ---------------------------------------------------------------------------


def build_kcomponent_index(G, kappa2: dict, enabled: bool):
    """
    On demand: nx.k_components for k≥3 present in κ₂.

    Experimental: on graphs with n >= KCOMP_N_TIME_SKIP (80) the index is
    skipped — nx.k_components often dominates runtime on Topology Zoo.
    Also respects PROFILE_KCOMP_N_MAXK_BUDGET and KCOMP_WALL_LIMIT_S.

    Contract: returns (index, note). If off / budget / timeout ⇒ {}.
    """
    if not enabled:
        return {}, "kcomponents off"
    ks = sorted({int(k) for k in kappa2.values() if int(k) >= 3})
    if not ks:
        return {}, "kcomponents: no κ₂≥3"
    n = G.number_of_nodes()
    max_k = max(ks)
    if n * max_k > PROFILE_KCOMP_N_MAXK_BUDGET:
        return {}, f"kcomponents skipped: n*max_k={n * max_k} > budget"
    if n >= KCOMP_N_TIME_SKIP:
        return {}, (
            f"kcomponents skipped: n={n} ≥ {KCOMP_N_TIME_SKIP} "
            "(experimental; time budget)"
        )

    import time as _time

    t0 = _time.time()
    try:
        raw = nx.k_components(G)
    except Exception as exc:
        return {}, f"kcomponents skipped: {type(exc).__name__}"
    if _time.time() - t0 > KCOMP_WALL_LIMIT_S:
        return {}, f"kcomponents skipped: took >{KCOMP_WALL_LIMIT_S}s"

    index = {}
    for k in ks:
        comps = raw.get(k)
        if not comps:
            avail = [kk for kk in raw if kk >= k]
            if not avail:
                continue
            comps = raw[min(avail)]
        index[k] = [frozenset(c) for c in comps]
    return index, "kcomponents ok"


def kcomponent_region(j, k, block_region, kcomp_index) -> frozenset:
    """
    Intersection of block-cut ∩ k-connected component containing j.

    If there is no index for k, returns only block_region.
    """
    base = block_region
    comps = kcomp_index.get(int(k)) if kcomp_index else None
    if not comps:
        return base
    for comp in comps:
        if j in comp:
            return frozenset(base & comp)
    return base

