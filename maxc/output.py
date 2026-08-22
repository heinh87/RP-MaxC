# -*- coding: utf-8 -*-
"""Per-instance/session log and aggregated LaTeX table."""

from __future__ import annotations

import os

DEFAULT_OUTPUT_DIR = "archive"


class Log:
    """
    Mirror messages to three destinations:
      - stdout
      - per-instance log (*_N_maxc_only.txt), swapped in new_network_log
      - global session log (log_maxc_only.txt)

    Files stay open; new_network_log closes/reopens the instance file.
    """

    def __init__(self):
        self.network_log_path = None
        self.session_path = "log_maxc_only.txt"
        self._network_fp = None
        self._session_fp = open(self.session_path, "a", encoding="utf-8")

    def new_network_log(self, name):
        """Open (truncate) the instance log file and start mirroring to it."""
        if self._network_fp is not None:
            self._network_fp.close()
            self._network_fp = None
        self.network_log_path = name
        self._network_fp = open(name, "w", encoding="utf-8")

    def print(self, s):
        """Print and append the same line to the active log files."""
        print(s)
        line = str(s) + "\n"
        if self._network_fp is not None:
            self._network_fp.write(line)
            self._network_fp.flush()
        self._session_fp.write(line)
        self._session_fp.flush()

    def print_results(self, network):
        """Format network.results (value/label/description tuples) in the log."""
        if not getattr(network, "results", None):
            self.print("Warning: Network has no results to print.")
            return
        self.print("\n" + "=" * 60)
        self.print("OPTIMIZATION RESULTS")
        self.print("=" * 60)
        for key, data in network.results.items():
            if len(data) == 3:
                value, label, description = data
                self.print(f"\n{label}: {value}")
                self.print(f"  Description: {description}")
            else:
                self.print(f"\n{key}: {data}")
        self.print("\n" + "=" * 60 + "\n")


log = Log()


class Output:
    """
    Aggregated LaTeX table for the batch (one row per graph).

    Columns: Network |V| |A| Maxκ₂ p p/|V| t_κ t_p
    File: tableMaxC_min_servers.tex under --output.
    """

    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.rows = []  # (n, m, name, line) — sorted by n on save
        self.table_file = "tableMaxC_min_servers.tex"

    def add(self, network, p, p_time):
        """
        Append a LaTeX row:
          name & n & m & maxκ₂ & p & p/n & tκ & t_p \\
        """
        n = network.G.number_of_nodes()
        m = network.G.number_of_edges()
        graph_name = os.path.basename(network.source_path).rsplit(".", 1)[0]
        line = (
            f"{graph_name} & {n} & {m} & {network.max_kappa2} & {p} & "
            f"{round(p / n, 2)} & {network.kappa_time} & {round(p_time, 4)} \\"
        )
        self.rows.append((n, m, graph_name, line))
        return line

    def save(self):
        """Write rows sorted by |V| (then |A|, name)."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, self.table_file)
        with open(path, "w", encoding="utf-8") as f:
            for _n, _m, _name, line in sorted(self.rows):
                f.write(line + "\n")
        print("Wrote", path)
