# -*- coding: utf-8 -*-
"""Graph load and validation (GML / DIMACS).

Contract: returns a connected, undirected NetworkX Graph with nodes 0..n-1.
Does not compute κ or choose a StrategyPlan.
"""

from __future__ import annotations

import os

import networkx as nx

from maxc.output import log


def read_dimacs(path: str) -> nx.Graph:
    """
    Read edges in a simplified DIMACS format ('a u v' lines).

    Used by OSMnx graphs exported as .dimacs in the project.
    """
    G = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "a" and len(parts) >= 3:
                G.add_edge(parts[1], parts[2])
    return G


def load_graph(path: str) -> nx.Graph:
    """
    Load GML/DIMACS, require connected and undirected, relabel nodes to 0..n-1.

    Why label='id' on GML:
      some Topology Zoo graphs have duplicate '?' labels; id is unique.

    Why ordering='increasing degree':
      stabilizes the 0..n-1 numbering across runs and keeps the original
      label in 'nome' — useful when comparing caches/results.

    Opens the instance log (*_N_maxc_only.txt).
    """
    suffix = path.rsplit(".", 1)[-1].lower()

    if suffix == "gml":
        G = nx.read_gml(path, label="id")
    elif suffix == "dimacs":
        G = read_dimacs(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if not nx.is_connected(G):
        raise SystemExit("Graph not connected")
    if nx.is_directed(G):
        raise SystemExit("Graph is directed")

    if len(nx.get_node_attributes(G, "nome")) == 0:
        try:
            G = nx.convert_node_labels_to_integers(
                G, ordering="increasing degree", label_attribute="nome"
            )
        except TypeError:
            G = nx.convert_node_labels_to_integers(G, label_attribute="nome")

    _start_instance_log(path, G)
    return G


def _start_instance_log(path: str, G: nx.Graph) -> None:
    """Open a per-instance log with suffix _maxc_only."""
    base, ext = os.path.splitext(path)
    log.new_network_log(f"{base}_{G.number_of_nodes()}_maxc_only.txt")
    log.print(" ")
    if ext.lower() == ".gml" and "Network" in G.graph:
        log.print(str(G.graph["Network"]))
