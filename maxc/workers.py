# -*- coding: utf-8 -*-
"""
Process workers for local κ (picklable).

A single global CTX (WorkerCtx), filled in init_worker.
Tasks: phase1_source_round / phase2_client / update_lb_from_pair.

Phase 1: pair_cache goes in the payload (grows between waves; the initializer
runs only once). Phase 2: pair_cache/pair_ub go in initargs.

Contract:
  phase1_source_round
    in:  (s, kappa2_lb0, pair_cache, pair_ub)
    out: Phase1TaskResult
  phase2_client
    in:  (j, kappa2_j, witnesses)
    out: Phase2TaskResult
  cut_events: list (k, u, v, frozenset(cut)) for the parent to propagate UBs.
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from networkx.algorithms.connectivity import build_auxiliary_node_connectivity

from maxc.backends import auxiliary_to_csr, build_igraph, local_kappa
from maxc.ops import (
    CUT_EXTRACT_K_MAX,
    extract_node_cut_from_residual,
    kcomponent_region,
    sort_candidates,
    structural_kappa_lb,
)
from maxc.strategy import BACKEND_IGRAPH, BACKEND_NX, BACKEND_SCIPY, get_nx_flow_func

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def max_safe_workers():
    """
    Process cap: cpu_count()-1 (minimum 1).

    Always reserve 1 core for the OS/UI so the machine stays responsive
    during the parallel κ computation. On 8 cores → 7; on 32 → 31.
    """
    return max(1, (multiprocessing.cpu_count() or 1) - 1)


DEFAULT_WORKERS = max_safe_workers()


@dataclass
class WorkerCtx:
    """Child-process state — graph and structures built once."""

    graph: Any
    aux: Any
    degree: dict
    region: dict
    plan: Any
    flow_func: Any
    residual: Any = None
    kcomp: dict = field(default_factory=dict)
    scipy_mat: Any = None
    scipy_pos: Any = None
    scipy_map: Any = None
    pair_cache: dict = field(default_factory=dict)
    pair_ub: dict = field(default_factory=dict)
    igraph: Any = None
    igraph_idx: Any = None


CTX: Optional[WorkerCtx] = None


@dataclass
class Phase1TaskResult:
    s: Any
    best: int
    witnesses: list
    new_pairs: dict
    cut_events: list
    queries: int
    cache_hits: int
    lb_shortcuts: int
    ub_prunes: int


@dataclass
class Phase2TaskResult:
    j: Any
    adm: list
    new_pairs: dict
    cut_events: list
    queries: int
    cache_hits: int
    lb_shortcuts: int
    ub_prunes: int


def _pair_key(u, v):
    """Undirected key: κ(u,v)=κ(v,u) ⇒ one record per pair."""
    return (u, v) if u < v else (v, u)


def init_worker(
    graph,
    degree,
    region,
    plan,
    kcomp_index=None,
    pair_cache=None,
    pair_ub=None,
):
    """
    Initialize a worker process (one global WorkerCtx).

    Builds the auxiliary digraph once; if reuse_residual, the residual too.
    SciPy/igraph: structures pre-converted once.
    pair_cache/pair_ub in init: Phase 2 only (snapshot). In Phase 1 the cache
    grows between waves and goes in the payload.
    """
    global CTX
    aux = build_auxiliary_node_connectivity(graph)
    scipy_mat = scipy_pos = scipy_map = None
    igraph = igraph_idx = None
    residual = None
    if plan.flow_backend == BACKEND_SCIPY:
        scipy_mat, scipy_pos, scipy_map = auxiliary_to_csr(aux)
    elif plan.flow_backend == BACKEND_IGRAPH:
        igraph, igraph_idx = build_igraph(graph)
    if plan.reuse_residual and plan.flow_backend == BACKEND_NX:
        from networkx.algorithms.flow import build_residual_network

        residual = build_residual_network(aux, "capacity")
    CTX = WorkerCtx(
        graph=graph,
        aux=aux,
        degree=degree,
        region=region,
        plan=plan,
        flow_func=get_nx_flow_func(plan.flow_algorithm),
        residual=residual,
        kcomp=kcomp_index or {},
        scipy_mat=scipy_mat,
        scipy_pos=scipy_pos,
        scipy_map=scipy_map,
        pair_cache=pair_cache if pair_cache is not None else {},
        pair_ub=pair_ub if pair_ub is not None else {},
        igraph=igraph,
        igraph_idx=igraph_idx,
    )


def _query_kappa(u, v, cutoff):
    """One κ(u,v) query via the CTX plan."""
    ctx = CTX
    return local_kappa(
        ctx.graph,
        u,
        v,
        auxiliary=ctx.aux,
        residual=ctx.residual,
        flow_fn=ctx.flow_func,
        cutoff=cutoff,
        backend=ctx.plan.flow_backend,
        scipy_mat=ctx.scipy_mat,
        scipy_pos=ctx.scipy_pos,
        scipy_map=ctx.scipy_map,
        igraph=ctx.igraph,
        igraph_idx=ctx.igraph_idx,
    )


def _lookup_pair_kappa(key, pair_cache, new_pairs):
    """
    Look up κ in the shared cache or this task's new pairs.

    Returns (k, hit) with hit=True if it came from cache/new_pairs; k=None on miss.
    """
    if key in pair_cache:
        return int(pair_cache[key]), True
    if key in new_pairs:
        return int(new_pairs[key]), True
    return None, False


def _apply_structural_shortcut(u, v, ub_cap, best_lb, plan):
    """
    Structural shortcut when the plan allows it.

    Returns:
      None — no useful shortcut (needs a flow)
      ("exact", k) — LB matches the pair cap (exact κ)
      ("raise_lb", lb) — LB > best_lb but does not yet certify the pair
    """
    if not plan.use_structural_lower_bounds:
        return None
    lb = structural_kappa_lb(u, v, CTX.graph)
    if lb >= ub_cap:
        return ("exact", ub_cap)
    if lb > best_lb:
        return ("raise_lb", lb)
    return None


def _maybe_cut_event(u, v, k, plan, cut_events):
    """Extract a residual cut and append to cut_events if flags/backend allow."""
    ctx = CTX
    if not (
        plan.use_cut_upper_bounds
        and plan.flow_backend == BACKEND_NX
        and ctx.residual is not None
        and 0 < k <= CUT_EXTRACT_K_MAX
    ):
        return
    cut = extract_node_cut_from_residual(
        ctx.graph, u, v, ctx.aux, ctx.residual
    )
    if cut:
        cut_events.append((int(k), u, v, frozenset(cut)))


def _update_best_witnesses(best, witnesses, t, k):
    """Update best/witnesses if k improves or ties the current LB."""
    if k > best:
        return k, [t]
    if k == best and k > 0 and t not in witnesses:
        witnesses.append(t)
    return best, witnesses


def phase1_source_round(payload) -> Phase1TaskResult:
    """One Phase 1 pass for source s (each source runs once)."""
    s, kappa2_lb0, pair_cache, pair_ub = payload
    ctx = CTX
    plan = ctx.plan
    deg_cap = ctx.degree[s]
    best = int(kappa2_lb0)
    witnesses = []
    new_pairs = {}
    cut_events = []
    queries = 0
    cache_hits = 0
    lb_shortcuts = 0
    ub_prunes = 0

    def result(**kwargs):
        defaults = dict(
            s=s,
            best=best,
            witnesses=witnesses,
            new_pairs=new_pairs,
            cut_events=cut_events,
            queries=queries,
            cache_hits=cache_hits,
            lb_shortcuts=lb_shortcuts,
            ub_prunes=ub_prunes,
        )
        defaults.update(kwargs)
        return Phase1TaskResult(**defaults)

    if plan.use_structural_lower_bounds and deg_cap == 1:
        nbrs = list(ctx.graph.neighbors(s))
        return result(
            best=1,
            witnesses=[nbrs[0]] if nbrs else [],
            lb_shortcuts=1,
        )

    region = ctx.region[s]
    neighbors = set(ctx.graph.neighbors(s))

    for (a, b), k in pair_cache.items():
        if a != s and b != s:
            continue
        t = b if a == s else a
        best, witnesses = _update_best_witnesses(best, witnesses, t, int(k))

    if best >= deg_cap:
        return result(best=best, witnesses=witnesses)

    candidates = [t for t in region if t != s]
    candidates = sort_candidates(
        s,
        candidates,
        ctx.graph,
        ctx.degree,
        neighbors,
        pair_cache,
        pair_ub,
        witnesses,
        best,
        plan.use_smart_ordering,
    )

    for t in candidates:
        if best >= deg_cap:
            break

        ub_deg = min(deg_cap, ctx.degree[t])
        if ub_deg <= best:
            continue

        key = _pair_key(s, t)

        if plan.use_cut_upper_bounds and key in pair_ub and pair_ub[key] <= best:
            ub_prunes += 1
            continue

        k, hit = _lookup_pair_kappa(key, pair_cache, new_pairs)
        if hit:
            cache_hits += 1
        else:
            shortcut = _apply_structural_shortcut(s, t, ub_deg, best, plan)
            if shortcut is not None:
                kind, val = shortcut
                if kind == "exact":
                    k = val
                    lb_shortcuts += 1
                    new_pairs[key] = k
                    best, witnesses = _update_best_witnesses(best, witnesses, t, k)
                    continue
                best = val
                witnesses = [t]
                lb_shortcuts += 1
                if best >= deg_cap:
                    break

            k = _query_kappa(s, t, cutoff=deg_cap)
            queries += 1
            new_pairs[key] = k
            _maybe_cut_event(s, t, k, plan, cut_events)

        best, witnesses = _update_best_witnesses(best, witnesses, t, k)

    if best == 0 and ctx.graph.number_of_nodes() > 1:
        best = 1
        if neighbors:
            witnesses = [next(iter(neighbors))]

    return result(
        best=int(best),
        witnesses=witnesses,
        queries=queries,
        cache_hits=cache_hits,
        lb_shortcuts=lb_shortcuts,
        ub_prunes=ub_prunes,
    )


def phase2_client(payload) -> Phase2TaskResult:
    """Phase 2 for client j: Adm(j) = {i : κ(j,i) = κ₂(j)}."""
    j, kappa2_j, witnesses = payload
    ctx = CTX
    plan = ctx.plan
    pair_cache = ctx.pair_cache
    pair_ub = ctx.pair_ub
    target = int(kappa2_j)
    adm = set(witnesses)
    adm.add(j)
    queries = 0
    cache_hits = 0
    lb_shortcuts = 0
    ub_prunes = 0
    new_pairs = {}
    cut_events = []

    def result(adm_list):
        return Phase2TaskResult(
            j=j,
            adm=adm_list,
            new_pairs=new_pairs,
            cut_events=cut_events,
            queries=queries,
            cache_hits=cache_hits,
            lb_shortcuts=lb_shortcuts,
            ub_prunes=ub_prunes,
        )

    if target <= 0:
        return result(sorted(adm))

    if target == 1:
        return result(sorted(ctx.graph.nodes()))

    block_reg = ctx.region[j]
    candidates = kcomponent_region(j, target, block_reg, ctx.kcomp)

    ordered = sort_candidates(
        j,
        [i for i in candidates if i not in adm],
        ctx.graph,
        ctx.degree,
        set(ctx.graph.neighbors(j)),
        pair_cache,
        pair_ub,
        witnesses,
        target - 1,
        plan.use_smart_ordering,
    )

    for i in ordered:
        if ctx.degree[i] < target:
            continue
        if min(ctx.degree[j], ctx.degree[i]) < target:
            continue

        key = _pair_key(j, i)

        if plan.use_cut_upper_bounds and key in pair_ub and pair_ub[key] < target:
            ub_prunes += 1
            continue

        k, hit = _lookup_pair_kappa(key, pair_cache, new_pairs)
        if hit:
            cache_hits += 1
        else:
            if plan.use_structural_lower_bounds:
                lb = structural_kappa_lb(j, i, ctx.graph)
                if lb >= target:
                    lb_shortcuts += 1
                    adm.add(i)
                    continue

            k = _query_kappa(j, i, cutoff=target)
            queries += 1
            new_pairs[key] = k
            _maybe_cut_event(j, i, k, plan, cut_events)

        if k >= target:
            adm.add(i)

    return result(sorted(adm))


def update_lb_from_pair(kappa2_lb, witnesses, u, v, k):
    """
    Update κ₂_lb and witnesses at both endpoints after learning κ(u,v)=k.

    Example: if k = deg(t), then κ₂(t) is certified and t can leave later
    Phase 1 waves (bilateral degree certification).
    """
    k = int(k)
    for a, b in ((u, v), (v, u)):
        if k > kappa2_lb[a]:
            kappa2_lb[a] = k
            witnesses[a] = [b]
        elif k == kappa2_lb[a] and k > 0 and b not in witnesses[a]:
            witnesses[a].append(b)
