# RP-MaxC and p-median optimization on networks

A Python implementation to analyze network connectivity and solve two optimization problems:

- RP-MaxC (Maximum Connectivity with Server Placement): choose server nodes to maximize client connectivity (node-connectivity, aka kappa) and optimize distance criteria.
- p-median: choose p server nodes to minimize the sum of distances from clients to their assigned server, with additional analyses of connectivity gaps.

The pipeline loads real network topologies from GML (or DIMACS), computes connectivity and distance matrices, solves integer linear programming models for several variants, and produces LaTeX tables and logs.

## Contents

- `RP-MaxC.py` — main script with classes and optimization routines
- `archive/` — input graphs in `.gml` (required)
- Outputs (when you run the script):
  - `tableMaxC.tex` — RP-MaxC summary table
  - `tablePmedian.tex` — p-median summary table
  - `tableRatioMaxCpmedian.tex` — ratios of RP-MaxC vs p-median across p..p+6
  - `log.txt` — global log
  - `<graph>_<|V|>_resul.txt` — per-network log

## Requirements

- Python 3.10+
- Packages: networkx, pymprog
- A linear programming solver supported by pymprog (GLPK). On Linux, GLPK is typically needed at OS level.

### Install

```bash
# Python dependencies
python3 -m pip install --upgrade pip
python3 -m pip install networkx pymprog

# Optional (Linux): GLPK system packages (names can vary by distro)
# sudo apt-get update
# sudo apt-get install -y glpk-utils libglpk-dev
```

## Quick start

Place one or more `.gml` graphs in `archive/` and run:

```bash
python3 RP-MaxC.py
```

The script will:
- Validate each graph (connected, undirected; relabel nodes to integers 0..|V|-1 while preserving original labels in the `nome` attribute)
- Compute node-connectivity between all pairs (kappa, kappa2) and all-pairs shortest path distances (Dijkstra)
- Solve optimization variants and collect results
- Write LaTeX tables to `tableMaxC.tex`, `tablePmedian.tex`, `tableRatioMaxCpmedian.tex`
- Log a readable summary to `log.txt`

## What the script computes

For each graph:
1) RP-MaxC family (maximum connectivity enforced kappa(j,i)=kappa2(j))
   - Minimum number of servers (minimize count of selected servers)
   - Minimum sum of distances with max connectivity (p fixed from step above)
   - Maximum sum of distances with max connectivity (p fixed)
2) p-median family
   - Minimum sum of distances with p servers
   - Maximum connectivity (minimize sum of gaps to kappa2) under the same distance budget as p-median
3) Ratios across p..p+6: for each increment, compare RP-MaxC min Σdist to p-median Σdist
4) Connectivity gap analysis for p-median: how far assigned server connectivity is from each client’s best possible kappa2

## Optimization model (linearProgram)

Decision variables:
- y[i] ∈ {0,1}: whether node i is selected as a server
- x[i,j] ∈ {0,1}: whether client j is assigned to server i

Core constraints:
- Each client is assigned to exactly one server: ∑_i x[i,j] = 1
- Assignment implies openness: x[i,j] ≤ y[i]
- Max-connectivity cases enforce kappa(j,i) = kappa2(j) wherever x[i,j] = 1

Problem types (parameter `problemType`):
- "maxConnect, minimum number of servers"
- "maxConnect, minimum sum of distances" (p fixed)
- "maxConnect, maximum sum of distances" (p fixed)
- "p-median, minimum sum of distances" (p fixed)
- "p-median, maximum connectivity" (minimize gaps under a distance-sum budget)
- "p-median, minimum connectivity" (not used by main, but implemented)

## Classes (high level)

- `Log`
  - `newNetworkLog(name)` — create/clear per-network log
  - `print(s)` — print and append to both logs
  - `printResults(network)` — pretty-print the `network.results` dictionary

- `Network`
  - `loadAndValidateGraph(path)` — from .gml or .dimacs, ensures connected/undirected, relabels nodes
  - `analyzeGraph()` — computes `kappa`, `kappa2`, and all-pairs shortest path `dist`

- `Output`
  - `addMaxCTableLine(network)` — one LaTeX row for RP-MaxC summary
  - `addPmedianTableLine(network)` — one LaTeX row for p-median summary
  - `addRatioMaxCpmedianTableLine(network)` — one LaTeX row for ratios across p..p+6
  - `saveTables()` — writes the LaTeX tables to disk

## Results dictionary (network.results)

After running `main()`, results are stored as:

- `p`: (int, label, description) — minimum number of servers for RP-MaxC
- `pTime`: (float, "$t(s)$", ...) — time in seconds to compute p
- `sumDistMaxConnect`: (int, label, ...) — RP-MaxC minimum Σ distances
- `worstSumDistMaxConnect`: (int, label, ...) — RP-MaxC maximum Σ distances
- `sumDistPmedian`: (int, label, ...) — p-median Σ distances
- `sumGapsPmedian`: (int, label, ...) — minimum sum of connectivity gaps for p-median under budget
- `distRatio`: (float, label, ...) — min ratio RP-MaxC Σdist / p-median Σdist
- `worstDistRatio`: (float, label, ...) — max ratio RP-MaxC Σdist / p-median Σdist
- `numClientGap`: (int, label, ...) — number of p-median clients that don’t reach kappa2 with their server
- `maxGap`: (int, label, ...) — largest gap (kappa2(client) − kappa(server,client))
- `kappaMaxGap`: (int, label, ...) — kappa2 of the client with the largest gap
- `avgGap`: (float, label, ...) — average relative gap
- `ratioMaxCpmedian`: list[float] — ratio sequence for p, p+1, …, p+6 (for the ratio table)
- `listRatioDistMaxC_pmedian`: (list[float], label, description) — the same sequence plus a caption for logging

## Input data

- GML graphs should contain basic metadata like `Network` (name) and `DateYear` (optional). Nodes will be relabeled to 0..|V|-1; the original label is stored in attribute `nome`.
- DIMACS graphs are minimally supported for edges.
- Graph must be connected and undirected.

## Outputs (LaTeX tables)

- `tableMaxC.tex` columns:
  1. Network name and year
  2. |V|
  3. |A|
  4. Max κ₂(v)
  5. p (min servers for RP-MaxC)
  6. p/|V|
  7. t(s)
  8. Max Σ dist (RP-MaxC)
  9. Min Σ dist (RP-MaxC)
  10. p-median Σ dist
  11. Max ratio (RP-MaxC max / p-median)
  12. Min ratio (RP-MaxC min / p-median)

- `tablePmedian.tex` columns:
  1. Network name and year
  2. |V|
  3. |A|
  4. Max κ₂(v)
  5. p
  6. Num clients with diff
  7. Max diff(c)
  8. Max κ₂(c)
  9. Avg diff

- `tableRatioMaxCpmedian.tex` columns:
  1. Network name and year
  2. p, p+1, p+2, p+3, p+4, p+5, p+6 ratios

You can paste these rows into your paper or generate combined tables by post-processing.

## Reproducibility and performance

- The ILP model uses `pymprog` (GLPK). Performance depends on graph size and solver.
- For repeatability of timing, the script averages multiple runs in some steps.

## Troubleshooting

- `NameError: var/minimize/maximize/solve`: ensure `from pymprog import *` is present and GLPK is installed and discoverable.
- `Graph not connected` or `Graph is directed`: the input graph must be connected and undirected.

## Extending

- Add new problem variants by extending `linearProgram()` and the orchestration in `main()`.
- Add new input graphs under `archive/`.
- Customize table formats in `Output` methods.

## License

If you intend to publish or share, consider adding a LICENSE file. For academic work, include citation instructions here.
