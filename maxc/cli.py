# -*- coding: utf-8 -*-
"""CLI and batch: graph.load → compute_kappa → [ILP] → results/table."""

from __future__ import annotations

import argparse
import glob
import multiprocessing
import os

from maxc.graph import load_graph
from maxc.kappa import Network
from maxc.output import DEFAULT_OUTPUT_DIR, Output, log
from maxc.solve import run_min_servers
from maxc.strategy import add_strategy_cli_flags
from maxc.workers import DEFAULT_WORKERS

DEFAULT_INPUT_DIR = "archive"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimize the number of servers with maximum connectivity (RP-MaxC). "
            "Computes only κ₂ and admissible pairs, in parallel."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help="Directory of .gml / .dimacs files (default: archive)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: archive)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel processes for κ₂/Adm. Cap = cpu_count()-1 "
            f"(default on this machine: {DEFAULT_WORKERS}). "
            "Larger values are reduced to the cap so the machine stays responsive."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not write cache *_maxc_admissible.json",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific files (otherwise: all .gml/.dimacs in --input)",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=None,
        help="Only process graphs with at least N vertices (after load)",
    )
    parser.add_argument(
        "--kappa-only",
        action="store_true",
        help="Compute only κ₂/Adm (do not solve the ILP)",
    )
    add_strategy_cli_flags(parser)
    return parser.parse_args()


def _results_common(network, n):
    """Shared fields of network.results (κ + strategies)."""
    strat = (
        network.strategy_plan.compact_line()
        if network.strategy_plan is not None
        else "n/a"
    )
    return {
        "n": (n, "|V|", "Number of vertices"),
        "m": (
            network.G.number_of_edges(),
            "|A|",
            "Number of edges",
        ),
        "max_kappa2": (
            network.max_kappa2,
            "Max κ₂",
            "Maximum κ₂(v) in the graph",
        ),
        "kappa_time": (
            network.kappa_time,
            "t_κ(s)",
            "Wall time to compute κ₂ and admissible sets",
        ),
        "phase1_time": (
            network.phase1_time,
            "t_phase1(s)",
            "Wall time of phase 1 (κ₂)",
        ),
        "phase2_time": (
            network.phase2_time,
            "t_phase2(s)",
            "Wall time of phase 2 (admissible)",
        ),
        "flow_queries": (
            network.flow_queries,
            "#flows",
            "Number of local_node_connectivity calls",
        ),
        "cache_hits": (
            network.cache_hits,
            "cache_hits",
            "Pair lookups served without a new flow",
        ),
        "ub_prunes": (
            network.ub_prunes,
            "ub_prunes",
            "Candidates skipped by cut upper bounds",
        ),
        "lb_shortcuts": (
            network.lb_shortcuts,
            "lb_shortcuts",
            "Structural lower-bound shortcuts",
        ),
        "strategies": (
            strat,
            "strategies",
            "Strategy plan (compact)",
        ),
    }


def main():
    """
    Per-graph batch: load → compute_kappa → [ILP] → results/table.

    --kappa-only: stop after κ₂/Adm (does not call Gurobi).
    Paper model: S = C = V (every vertex is a client and a server candidate).
    """
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    use_cache = not args.no_cache
    workers = args.workers if args.workers is not None else DEFAULT_WORKERS

    if args.files:
        file_list = args.files
    else:
        file_list = sorted(
            glob.glob(os.path.join(input_dir, "*.gml"))
            + glob.glob(os.path.join(input_dir, "*.dimacs"))
        )

    if not file_list:
        raise SystemExit(f"No graphs found in {input_dir}")

    output = Output(output_dir=output_dir)
    log.print(f"Using workers={workers} (cpu_count={multiprocessing.cpu_count()})")

    for name in file_list:
        print(name)
        G = load_graph(name)
        n = G.number_of_nodes()
        if args.min_n is not None and n < args.min_n:
            log.print(f"Skipping {name}: n={n} < min-n={args.min_n}")
            continue

        network = Network(G, source_path=name)
        network.C = sorted(G.nodes())
        network.S = list(network.C)

        network.compute_kappa(workers=workers, use_cache=use_cache, args=args)
        log.print("\n----------------------\n")

        if args.kappa_only:
            network.results = _results_common(network, n)
            log.print_results(network)
            continue

        p, servers, connections, p_time = run_min_servers(network)

        network.results = _results_common(network, n)
        network.results.update(
            {
                "p": (p, "p", "Minimum number of servers for MaxC"),
                "p_time": (
                    round(p_time, 4),
                    "t_p(s)",
                    "ILP time for min servers",
                ),
                "servers": (servers, "Servers", "Chosen server vertices"),
            }
        )
        log.print_results(network)
        output.add(network, p, p_time)

    if not args.kappa_only:
        output.save()


if __name__ == "__main__":
    main()
