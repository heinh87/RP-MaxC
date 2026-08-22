# -*- coding: utf-8 -*-
"""
Orchestrates κ₂ and Adm (JSON cache). Does not read graph files.

Map:
  Phase 0  → StrategyPlan (strategy.py)
  Phase 1  → κ₂ waves (workers.phase1_source_round)
  Phase 2  → Adm (workers.phase2_client)
  Compute  → ops.py + backends.py

cli builds Network(G, source_path) after graph.load_graph.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import networkx as nx

from maxc.ops import build_kcomponent_index, propagate_cut_ubs
from maxc.output import log
from maxc.strategy import build_strategy_plan, default_strategy_args, log_strategy_plan
from maxc.workers import (
    DEFAULT_WORKERS,
    init_worker,
    max_safe_workers,
    phase1_source_round,
    phase2_client,
    update_lb_from_pair,
)


@dataclass
class Phase1State:
    kappa2: dict
    witnesses: dict
    pair_cache: dict
    pair_ub: dict
    queries: int
    cache_hits: int
    lb_shortcuts: int
    ub_prunes: int
    rounds: int


@dataclass
class Phase2State:
    admissible: dict
    queries: int
    cache_hits: int
    lb_shortcuts: int
    ub_prunes: int


class Network:
    """
    Graph + κ₂ + admissible sets for minimum MaxC.

    Cycle: graph.load_graph → Network(G, path) → compute_kappa → ILP.

    Does not store a full κ matrix: the ILP only needs, for each client j,
    the list Adm(j) of servers with maximum connectivity.
    """

    def __init__(self, G, source_path: str):
        self.source_path = source_path
        self.G = G
        self.C = []
        self.S = []

        self.kappa2 = {}
        self.admissible = {}
        self.max_kappa2 = None
        self.avg_kappa2 = None

        self.kappa_time = None
        self.phase1_time = None
        self.phase2_time = None
        self.flow_queries = None

        self.degree = {}
        self.region = {}

        self.results = None

        self.strategy_plan = None
        self.graph_profile = None
        self.cache_hits = 0
        self.ub_prunes = 0
        self.lb_shortcuts = 0
        self.kcomp_note = ""

    def compute_kappa(self, workers=None, use_cache=True, args=None):
        """
        Compute κ₂ and Adm in parallel (per source vertex).

        Args:
            workers: number of processes; None ⇒ DEFAULT_WORKERS (= cpu_count()-1).
            use_cache: if True, read/write *_maxc_admissible.json.
            args: CLI namespace (strategies); None ⇒ default plan (SciPy).
        """
        if workers is None:
            workers = DEFAULT_WORKERS

        t0 = time.time()
        cache_file = self._kappa_cache_file()
        self._resolve_strategy(args)
        self._log_graph_stats()

        if use_cache and os.path.exists(cache_file):
            try:
                kappa2, admissible, data = self._load_cache(cache_file)
                self._apply_kappa_cache(kappa2, admissible, data)
                log.print("Loaded MaxC kappa cache: " + cache_file)
                log.print(f"Time to calculate kappa (MaxC): {self.kappa_time} seconds")
                return
            except (KeyError, ValueError, TypeError):
                log.print("Ignoring stale cache: " + cache_file)

        self.degree = dict(self.graph_profile.degree)
        nx.set_node_attributes(self.G, self.degree, "degree")
        region = self._build_block_regions(self.graph_profile.blocks)

        nodes = list(self.G.nodes())
        n = len(nodes)
        n_workers = max(1, min(int(workers), n, max_safe_workers()))

        log.print(
            f"\n Calculating κ₂ + admissible "
            f"(workers={n_workers}, n={n}, pairs_all={n * (n - 1) // 2})...\n"
        )

        t1 = time.time()
        phase1 = self._phase1_kappa2_waves(nodes, region, n_workers)
        phase1_time = round(time.time() - t1, 2)

        t2 = time.time()
        phase2 = self._phase2_admissible(
            nodes,
            phase1.kappa2,
            phase1.witnesses,
            phase1.pair_cache,
            phase1.pair_ub,
            n_workers,
        )
        phase2_time = round(time.time() - t2, 2)

        self.kappa2 = phase1.kappa2
        self.admissible = phase2.admissible
        self._finalize_kappa_stats(
            t0,
            phase1_time,
            phase2_time,
            phase1,
            phase2,
            n,
        )

        if use_cache:
            self._save_cache(cache_file)
            print("Saved MaxC kappa cache: " + cache_file)

        log.print(f"Time to calculate kappa (MaxC): {self.kappa_time} seconds")

    def _log_graph_stats(self):
        profile = self.graph_profile
        log.print(f"nVert {profile.n} nEdges {profile.m}")
        log.print(f"Max degree = {profile.max_deg}")
        log.print(f"Avg degree = {round(profile.avg_deg, 2)}")

    def _resolve_strategy(self, args=None):
        """Pre-analysis + StrategyPlan; print the STRATEGY PLAN block."""
        if args is None:
            args = default_strategy_args()
        profile, plan = build_strategy_plan(self.G, args)
        self.graph_profile = profile
        self.strategy_plan = plan
        log_strategy_plan(plan, log.print)
        return plan

    def _phase1_kappa2_waves(self, nodes, region, n_workers) -> Phase1State:
        """Phase 1: κ₂ in waves with an undirected cache and bilateral certification."""
        n = len(nodes)
        kappa2_lb = {v: 0 for v in nodes}
        witnesses = {v: [] for v in nodes}
        pair_cache = {}
        pair_ub = {}
        finished = set()
        queries = 0
        hits = 0
        lb_s = 0
        ub_p = 0
        rounds = 0
        plan = self.strategy_plan

        if plan and plan.use_structural_lower_bounds:
            for v in nodes:
                if self.degree[v] == 1:
                    kappa2_lb[v] = 1
                    nbrs = list(self.G.neighbors(v))
                    if nbrs:
                        witnesses[v] = [nbrs[0]]
                    finished.add(v)

        order = sorted(nodes, key=lambda v: (-self.degree[v], v))

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(self.G, self.degree, region, plan, None),
        ) as executor:
            while len(finished) < n:
                self._certify_by_degree(nodes, kappa2_lb, finished)

                pending = [v for v in order if v not in finished]
                if not pending:
                    break

                wave = pending[:n_workers]
                rounds += 1
                payloads = [
                    (s, kappa2_lb[s], pair_cache, pair_ub) for s in wave
                ]

                for result in executor.map(
                    phase1_source_round, payloads, chunksize=1
                ):
                    queries += result.queries
                    hits += result.cache_hits
                    lb_s += result.lb_shortcuts
                    ub_p += result.ub_prunes
                    self._merge_phase1_result(
                        result.s,
                        result.best,
                        result.witnesses,
                        result.new_pairs,
                        kappa2_lb,
                        witnesses,
                        pair_cache,
                    )
                    self._merge_cut_events(result.cut_events, pair_ub)
                    finished.add(result.s)

                self._certify_by_degree(nodes, kappa2_lb, finished)

        self._ensure_kappa2_connected(kappa2_lb, witnesses, nodes)
        return Phase1State(
            kappa2={v: int(kappa2_lb[v]) for v in nodes},
            witnesses=witnesses,
            pair_cache=pair_cache,
            pair_ub=pair_ub,
            queries=queries,
            cache_hits=hits,
            lb_shortcuts=lb_s,
            ub_prunes=ub_p,
            rounds=rounds,
        )

    def _phase2_admissible(
        self, nodes, kappa2, witnesses, pair_cache, pair_ub, n_workers
    ) -> Phase2State:
        """Phase 2: Adm(j) in parallel; pair_cache in the initializer."""
        n = len(nodes)
        admissible = {v: set() for v in nodes}
        queries = 0
        hits = 0
        lb_s = 0
        ub_p = 0
        region = self.region
        plan = self.strategy_plan

        kcomp_index, knote = build_kcomponent_index(
            self.G, kappa2, bool(plan and plan.use_k_components)
        )
        self.kcomp_note = knote
        if knote and "skip" in knote:
            log.print(f"k-components note: {knote}")

        payloads = [(j, kappa2[j], witnesses.get(j, [])) for j in nodes]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(
                self.G,
                self.degree,
                region,
                plan,
                kcomp_index,
                pair_cache,
                pair_ub,
            ),
        ) as executor:
            chunksize = max(1, n // (n_workers * 4))
            for result in executor.map(
                phase2_client, payloads, chunksize=chunksize
            ):
                queries += result.queries
                hits += result.cache_hits
                lb_s += result.lb_shortcuts
                ub_p += result.ub_prunes
                for (a, b), k in result.new_pairs.items():
                    if (a, b) not in pair_cache:
                        pair_cache[(a, b)] = int(k)
                self._merge_cut_events(result.cut_events, pair_ub)
                admissible[result.j] = set(result.adm)
                if result.j not in admissible[result.j]:
                    raise RuntimeError(
                        f"Client {result.j} missing self-assignment in Adm"
                    )

        self._seed_symmetric_adm(admissible, kappa2)
        return Phase2State(
            admissible={j: sorted(admissible[j]) for j in nodes},
            queries=queries,
            cache_hits=hits,
            lb_shortcuts=lb_s,
            ub_prunes=ub_p,
        )

    def _finalize_kappa_stats(
        self, t0, phase1_time, phase2_time, phase1: Phase1State, phase2: Phase2State, n
    ):
        """Store times/counters on self and print the summary to the log."""
        self.phase1_time = phase1_time
        self.phase2_time = phase2_time
        self.flow_queries = phase1.queries + phase2.queries
        self.cache_hits = phase1.cache_hits + phase2.cache_hits
        self.lb_shortcuts = phase1.lb_shortcuts + phase2.lb_shortcuts
        self.ub_prunes = phase1.ub_prunes + phase2.ub_prunes
        self.kappa_time = round(time.time() - t0, 2)
        self.max_kappa2 = max(self.kappa2.values())
        self.avg_kappa2 = round(sum(self.kappa2.values()) / n, 2)

        all_pairs = n * (n - 1) // 2
        log.print(
            f"Phase 1 done: time={self.phase1_time}s, flow_queries={phase1.queries}, "
            f"cache_hits={phase1.cache_hits}, rounds={phase1.rounds}, "
            f"max_kappa2={self.max_kappa2}, "
            f"cached_pairs={len(phase1.pair_cache)}"
        )
        log.print(
            f"Phase 2 done: time={self.phase2_time}s, flow_queries={phase2.queries}, "
            f"cache_hits={phase2.cache_hits}"
        )
        log.print(
            f"Total kappa MaxC: time={self.kappa_time}s, "
            f"flow_queries={self.flow_queries}, "
            f"cache_hits={self.cache_hits}, "
            f"ub_prunes={self.ub_prunes}, "
            f"lb_shortcuts={self.lb_shortcuts}, "
            f"vs_all_pairs={all_pairs}, "
            f"ratio={self.flow_queries / max(all_pairs, 1):.3f}"
        )
        log.print(f"Max kappa2 = {self.max_kappa2}")
        log.print(f"Avg kappa2 = {self.avg_kappa2}")
        if self.strategy_plan is not None:
            log.print("---------- STRATEGY PLAN (final) ----------")
            log_strategy_plan(self.strategy_plan, log.print)

    def _merge_phase1_result(
        self, s, best, wit, new_pairs, kappa2_lb, witnesses, pair_cache
    ):
        """Merge a Phase 1 task result into the shared state."""
        for (a, b), k in new_pairs.items():
            if (a, b) not in pair_cache:
                pair_cache[(a, b)] = int(k)
            update_lb_from_pair(kappa2_lb, witnesses, a, b, pair_cache[(a, b)])

        if best > kappa2_lb[s]:
            kappa2_lb[s] = int(best)
            witnesses[s] = list(wit)
        elif best == kappa2_lb[s] and best > 0:
            for t in wit:
                if t not in witnesses[s]:
                    witnesses[s].append(t)

        if best > self.degree[s]:
            raise RuntimeError(
                f"Invariant violated: kappa2[{s}]={best} > deg={self.degree[s]}"
            )

    def _merge_cut_events(self, cut_events, pair_ub):
        """Propagate UBs from cuts returned by the workers."""
        if (
            not cut_events
            or not self.strategy_plan
            or not self.strategy_plan.use_cut_upper_bounds
        ):
            return 0
        n_upd = 0
        for item in cut_events:
            if len(item) != 4:
                continue
            k, s, t, cut = item
            if not cut:
                continue
            n_upd += propagate_cut_ubs(self.G, s, t, set(cut), pair_ub, int(k))
        return n_upd

    def _certify_by_degree(self, nodes, kappa2_lb, finished):
        """Mark finished every v with κ₂_lb(v) ≥ deg(v)."""
        for v in nodes:
            if v not in finished and kappa2_lb[v] >= self.degree[v]:
                finished.add(v)

    def _ensure_kappa2_connected(self, kappa2_lb, witnesses, nodes):
        """Connected graph with n>1 ⇒ κ₂ ≥ 1; guarantee an lb and a minimum witness."""
        if len(nodes) <= 1:
            return
        for v in nodes:
            if kappa2_lb[v] <= 0:
                kappa2_lb[v] = 1
                nbrs = list(self.G.neighbors(v))
                if nbrs and not witnesses[v]:
                    witnesses[v] = [nbrs[0]]

    def _seed_symmetric_adm(self, admissible, kappa2):
        """
        If i ∈ Adm(j) and κ₂(i)=κ₂(j), then j ∈ Adm(i) (undirected graph).

        Adm is not symmetric when the κ₂ values differ.
        """
        for j in list(admissible.keys()):
            kj = kappa2[j]
            for i in list(admissible[j]):
                if i != j and kappa2[i] == kj:
                    admissible[i].add(j)

    def _build_block_regions(self, blocks):
        """
        For each v, R(v) = union of the biconnected components that contain v.

        Invariant: if t ∉ R(s), there is an articulation separating s and t ⇒ κ(s,t) ≤ 1.
        Reuses `blocks` from GraphProfile (does not recompute biconnected_components).
        """
        node_to_blocks = defaultdict(list)
        blocks = list(blocks)
        for bi, block in enumerate(blocks):
            for v in block:
                node_to_blocks[v].append(bi)

        region = {}
        for v in self.G.nodes():
            nodes = set()
            for bi in node_to_blocks[v]:
                nodes |= set(blocks[bi])
            if not nodes:
                nodes = {v}
            region[v] = frozenset(nodes)
        self.region = region
        return region

    def _kappa_cache_file(self):
        """Cache of κ₂+Adm only."""
        base, _ = os.path.splitext(self.source_path)
        return base + "_maxc_admissible.json"

    def _save_cache(self, cache_file):
        """Write κ₂, Adm, and metrics (snake_case keys)."""
        data = {
            "kappa2": {str(k): int(v) for k, v in self.kappa2.items()},
            "admissible": {
                str(j): [int(i) for i in lst] for j, lst in self.admissible.items()
            },
            "kappa_time": float(self.kappa_time),
            "phase1_time": float(self.phase1_time),
            "phase2_time": float(self.phase2_time),
            "flow_queries": int(self.flow_queries),
            "cache_hits": int(self.cache_hits),
            "ub_prunes": int(self.ub_prunes),
            "lb_shortcuts": int(self.lb_shortcuts),
            "max_kappa2": int(self.max_kappa2),
            "avg_kappa2": float(self.avg_kappa2),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_cache(self, cache_file):
        """
        Read *_maxc_admissible.json (snake_case keys).

        No fallback for old camelCase caches.
        """
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "kappa_time" not in data:
            raise ValueError("stale cache (missing kappa_time)")

        node_set = set(self.G.nodes())

        def parse_node(node_str):
            if node_str in node_set:
                return node_str
            try:
                node_int = int(node_str)
                if node_int in node_set:
                    return node_int
            except (TypeError, ValueError):
                pass
            return node_str

        kappa2 = {parse_node(k): int(v) for k, v in data["kappa2"].items()}
        admissible = {
            parse_node(j): [parse_node(i) for i in lst]
            for j, lst in data["admissible"].items()
        }
        return kappa2, admissible, data

    def _apply_kappa_cache(self, kappa2, admissible, data):
        """Fill self from an already-loaded cache."""
        self.kappa2 = kappa2
        self.admissible = admissible
        self.kappa_time = round(float(data["kappa_time"]), 2)
        self.phase1_time = round(float(data["phase1_time"]), 2)
        self.phase2_time = round(float(data["phase2_time"]), 2)
        self.flow_queries = int(data["flow_queries"])
        self.cache_hits = int(data.get("cache_hits", 0))
        self.ub_prunes = int(data.get("ub_prunes", 0))
        self.lb_shortcuts = int(data.get("lb_shortcuts", 0))
        self.max_kappa2 = int(data["max_kappa2"])
        self.avg_kappa2 = float(data["avg_kappa2"])
        self.degree = dict(self.graph_profile.degree)
