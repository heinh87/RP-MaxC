# -*- coding: utf-8 -*-
"""
StrategyPlan for RP-MaxC (no κ arithmetic).

O(n+m) pre-analysis measures degree/blocks for reuse in Network; the default
plan is fixed (SciPy + shortcuts). Compute primitives live in ops.py.

Contract: p, κ₂ and Adm stay exact; this module never approximates the final κ.
ILP (Gurobi) lives in solve.py.
"""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import networkx as nx
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path

# Budget for the k-components index (ops.build_kcomponent_index).
PROFILE_KCOMP_N_MAXK_BUDGET = 50_000  # n * max k acima ⇒ skip k-components

FLOW_FUNC_EK = "edmonds_karp"
FLOW_FUNC_SAP = "shortest_augmenting_path"

BACKEND_NX = "networkx"
BACKEND_SCIPY = "scipy"
BACKEND_IGRAPH = "igraph"

SOURCE_DEFAULT = "default"
SOURCE_CLI = "cli"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraphProfile:
    """O(n+m) graph signals — metrics + degree/blocks for reuse in kappa."""

    n: int
    m: int
    density: float
    avg_deg: float
    max_deg: int
    frac_deg1: float
    n_articulations: int
    n_blocks: int
    avg_block_size: float
    deg_skew: float
    degree: dict
    blocks: list
    reasons: list[str] = field(default_factory=list)


@dataclass
class StrategyPlan:
    """
    Optimization plan for the κ computation.

    sources[key] ∈ {"default", "cli"}.
    """

    flow_algorithm: str  # edmonds_karp | shortest_augmenting_path
    flow_backend: str  # networkx | scipy | igraph
    reuse_residual: bool
    use_cut_upper_bounds: bool
    use_structural_lower_bounds: bool
    use_smart_ordering: bool
    use_k_components: bool
    reasons: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    backend_fallback_note: Optional[str] = None

    def compact_line(self) -> str:
        """One-line summary for network.results / the log."""
        return (
            f"residual_reuse={'on' if self.reuse_residual else 'off'} "
            f"flow_algorithm={self.flow_algorithm} "
            f"cut_upper_bounds={'on' if self.use_cut_upper_bounds else 'off'} "
            f"structural_lower_bounds={'on' if self.use_structural_lower_bounds else 'off'} "
            f"k_components={'on' if self.use_k_components else 'off'} "
            f"smart_ordering={'on' if self.use_smart_ordering else 'off'} "
            f"flow_backend={self.flow_backend}"
        )


# ---------------------------------------------------------------------------
# Pre-analysis
# ---------------------------------------------------------------------------


def analyze_graph_profile(G: nx.Graph) -> GraphProfile:
    """
    Collect structural signals in O(n+m), including degree and biconnected blocks.

    Contract: metrics only (+ data for reuse); does not choose flags.
    Articulations = vertices in two or more blocks (no second NX pass).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    degree = dict(G.degree())
    degrees = list(degree.values())
    avg_deg = (sum(degrees) / n) if n else 0.0
    max_deg = max(degrees) if degrees else 0
    frac_deg1 = (sum(1 for d in degrees if d == 1) / n) if n else 0.0
    deg_skew = (max_deg / avg_deg) if avg_deg > 0 else 0.0

    blocks = [frozenset(b) for b in nx.biconnected_components(G)]
    n_blocks = len(blocks)
    avg_block = (sum(len(b) for b in blocks) / n_blocks) if n_blocks else float(n)
    block_count = Counter()
    for block in blocks:
        for v in block:
            block_count[v] += 1
    n_articulations = sum(1 for c in block_count.values() if c >= 2)

    reasons = [
        f"n={n} m={m}",
        f"density={dens:.4f}",
        f"avg_deg={avg_deg:.2f} max_deg={max_deg}",
        f"frac_deg1={frac_deg1:.3f}",
        f"n_articulations={n_articulations} n_blocks={n_blocks} avg_block={avg_block:.1f}",
        f"deg_skew={deg_skew:.2f}",
    ]
    return GraphProfile(
        n=n,
        m=m,
        density=dens,
        avg_deg=avg_deg,
        max_deg=max_deg,
        frac_deg1=frac_deg1,
        n_articulations=n_articulations,
        n_blocks=n_blocks,
        avg_block_size=avg_block,
        deg_skew=deg_skew,
        degree=degree,
        blocks=blocks,
        reasons=reasons,
    )


def default_base_plan(profile: GraphProfile) -> StrategyPlan:
    """
    Single plan (no profiles): SciPy for κ; residual/cut-UB off.

    Structural LB and smart ordering on; k-components off.
    flow_algorithm only applies if the backend is NetworkX.
    """
    reasons = list(profile.reasons) + [
        "default: flow_backend=scipy (Gurobi for the ILP)",
        "default: residual/cut-UB off (NetworkX only)",
        "default: structural LB on, smart ordering on, k-components off",
    ]
    sources = {
        "flow_algorithm": SOURCE_DEFAULT,
        "flow_backend": SOURCE_DEFAULT,
        "reuse_residual": SOURCE_DEFAULT,
        "use_cut_upper_bounds": SOURCE_DEFAULT,
        "use_structural_lower_bounds": SOURCE_DEFAULT,
        "use_smart_ordering": SOURCE_DEFAULT,
        "use_k_components": SOURCE_DEFAULT,
    }
    return StrategyPlan(
        flow_algorithm=FLOW_FUNC_EK,
        flow_backend=BACKEND_SCIPY,
        reuse_residual=False,
        use_cut_upper_bounds=False,
        use_structural_lower_bounds=True,
        use_smart_ordering=True,
        use_k_components=False,
        reasons=reasons,
        sources=sources,
    )


def resolve_flow_backend(plan: StrategyPlan) -> StrategyPlan:
    """
    Confirm that the requested backend imports.

    SciPy is the default and is required when the plan uses it — no fallback to NetworkX.
    Asking for igraph when it is missing also fails explicitly.
    """
    wanted = plan.flow_backend
    if wanted == BACKEND_NX:
        return plan
    if wanted == BACKEND_SCIPY:
        try:
            import scipy.sparse  # noqa: F401
            from scipy.sparse.csgraph import maximum_flow  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SciPy is required for κ flow "
                "(flow_backend=scipy). Install with: pip install scipy"
            ) from exc
        return plan
    if wanted == BACKEND_IGRAPH:
        try:
            import igraph  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "python-igraph is required with --flow-backend igraph. "
                "Install with: pip install python-igraph"
            ) from exc
        return plan
    raise RuntimeError(f"Unknown flow backend: {wanted}")


def apply_cli_overrides(plan: StrategyPlan, args: Any) -> StrategyPlan:
    """
    Tri-state CLI overrides on top of the default plan.

    Contract: only non-None fields on args change the plan; source becomes "cli".
    """
    p = plan
    sources = dict(p.sources)

    def set_bool(attr: str, value: Optional[bool]):
        nonlocal p
        if value is None:
            return
        p = replace(p, **{attr: value})
        sources[attr] = SOURCE_CLI

    set_bool("reuse_residual", getattr(args, "reuse_residual", None))
    set_bool("use_cut_upper_bounds", getattr(args, "use_cut_upper_bounds", None))
    set_bool("use_structural_lower_bounds", getattr(args, "use_structural_lower_bounds", None))
    set_bool("use_k_components", getattr(args, "use_k_components", None))
    set_bool("use_smart_ordering", getattr(args, "use_smart_ordering", None))

    flow_algorithm = getattr(args, "flow_algorithm", None)
    if flow_algorithm:
        p = replace(p, flow_algorithm=flow_algorithm)
        sources["flow_algorithm"] = SOURCE_CLI

    flow_backend = getattr(args, "flow_backend", None)
    if flow_backend:
        p = replace(p, flow_backend=flow_backend)
        sources["flow_backend"] = SOURCE_CLI

    p = replace(p, sources=sources)
    return resolve_flow_backend(p)


def build_strategy_plan(G: nx.Graph, args: Any) -> tuple[GraphProfile, StrategyPlan]:
    """Pre-analysis → default plan (SciPy) → CLI overrides → confirm backend."""
    profile = analyze_graph_profile(G)
    plan = default_base_plan(profile)
    plan = apply_cli_overrides(plan, args)
    return profile, plan


def format_strategy_plan(plan: StrategyPlan) -> str:
    """Fixed STRATEGY PLAN block (stdout + log)."""

    def src(key: str) -> str:
        return plan.sources.get(key, SOURCE_DEFAULT)

    def onoff(flag: bool) -> str:
        return "ON " if flag else "OFF"

    lines = [
        "========== STRATEGY PLAN ==========",
        f"residual reuse:          {onoff(plan.reuse_residual)}  ({src('reuse_residual')})",
        f"flow algorithm:          {plan.flow_algorithm} ({src('flow_algorithm')})",
        f"cut upper bounds:        {onoff(plan.use_cut_upper_bounds)}  ({src('use_cut_upper_bounds')})",
        f"structural lower bounds: {onoff(plan.use_structural_lower_bounds)}  ({src('use_structural_lower_bounds')})",
        f"k-components regions:    {onoff(plan.use_k_components)}  ({src('use_k_components')})",
        f"smart ordering:          {onoff(plan.use_smart_ordering)}  ({src('use_smart_ordering')})",
        f"flow backend:            {plan.flow_backend} ({src('flow_backend')})",
    ]
    lines.append("reasons:")
    for r in plan.reasons:
        lines.append(f"  - {r}")
    lines.append("===================================")
    return "\n".join(lines)


def log_strategy_plan(plan: StrategyPlan, printer: Callable[[str], None]) -> None:
    """Print the STRATEGY PLAN block via a callback (e.g. log.print)."""
    printer(format_strategy_plan(plan))


def get_nx_flow_func(name: str):
    """Resolve name → NetworkX flow_algorithm callable."""
    if name == FLOW_FUNC_SAP:
        return shortest_augmenting_path
    return edmonds_karp


def default_strategy_args() -> Namespace:
    """SciPy-plan defaults, with no flag overrides."""
    return Namespace(
        flow_algorithm=None,
        flow_backend=None,
        reuse_residual=None,
        use_cut_upper_bounds=None,
        use_structural_lower_bounds=None,
        use_k_components=None,
        use_smart_ordering=None,
    )


def add_strategy_cli_flags(parser) -> None:
    """Register strategy flags (tri-state) on the ArgumentParser."""
    parser.add_argument(
        "--flow-algorithm",
        choices=[FLOW_FUNC_EK, FLOW_FUNC_SAP],
        default=None,
        help=(
            "NetworkX flow algorithm "
            f"(default: {FLOW_FUNC_EK}; only used with --flow-backend networkx)"
        ),
    )
    parser.add_argument(
        "--flow-backend",
        choices=[BACKEND_NX, BACKEND_SCIPY, BACKEND_IGRAPH],
        default=None,
        help="Local-κ backend (default: scipy)",
    )

    # Tri-state: omitted = None (follow the default plan); last flag on the CLI wins.
    parser.set_defaults(
        reuse_residual=None,
        use_cut_upper_bounds=None,
        use_structural_lower_bounds=None,
        use_k_components=None,
        use_smart_ordering=None,
    )
    parser.add_argument(
        "--reuse-residual", dest="reuse_residual", action="store_true",
        help="Reuse the residual network between queries (NetworkX only)",
    )
    parser.add_argument(
        "--no-reuse-residual", dest="reuse_residual", action="store_false",
        help="Do not reuse the residual network",
    )
    parser.add_argument(
        "--cut-upper-bounds", dest="use_cut_upper_bounds", action="store_true",
        help="Propagate upper bounds from cuts (NetworkX only)",
    )
    parser.add_argument(
        "--no-cut-upper-bounds", dest="use_cut_upper_bounds", action="store_false",
        help="Disable cut upper bounds",
    )
    parser.add_argument(
        "--structural-lower-bounds", dest="use_structural_lower_bounds", action="store_true",
        help="Use structural lower bounds (degree/triangle)",
    )
    parser.add_argument(
        "--no-structural-lower-bounds", dest="use_structural_lower_bounds", action="store_false",
        help="Disable structural lower bounds",
    )
    parser.add_argument(
        "--k-components", dest="use_k_components", action="store_true",
        help="Restrict candidates via k-components",
    )
    parser.add_argument(
        "--no-k-components", dest="use_k_components", action="store_false",
        help="Disable k-components pruning",
    )
    parser.add_argument(
        "--smart-ordering", dest="use_smart_ordering", action="store_true",
        help="Order candidates with the smart heuristic",
    )
    parser.add_argument(
        "--no-smart-ordering", dest="use_smart_ordering", action="store_false",
        help="Use basic candidate ordering",
    )
