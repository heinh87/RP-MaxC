# -*- coding: utf-8 -*-
"""MaxC ILP: minimum number of servers with maximum connectivity (Gurobi only)."""

from __future__ import annotations

import os
import time

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from maxc.output import log

# Optional local license file (not shipped). Prefer GRB_LICENSE_FILE.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_GUROBI_LICENSE = os.path.join(_PKG_DIR, "gurobi.lic")

BINARY_X_THRESHOLD = 0.5


def _set_gurobi_license():
    """
    Ensure GRB_LICENSE_FILE.

    If the environment variable already points to a valid file, keep it.
    Otherwise, if a local gurobi.lic exists next to this package, use it.
    No license is shipped with the public repository — set GRB_LICENSE_FILE.
    """
    existing = os.environ.get("GRB_LICENSE_FILE")
    if existing and os.path.isfile(existing):
        return
    if os.path.isfile(LOCAL_GUROBI_LICENSE):
        os.environ["GRB_LICENSE_FILE"] = LOCAL_GUROBI_LICENSE
        return
    raise FileNotFoundError(
        "Gurobi license not found. Set GRB_LICENSE_FILE to your gurobi.lic path "
        "(or place gurobi.lic next to the maxc package locally; it is not in the repo)."
    )


@dataclass
class MinServersResult:
    servers: list
    connections: dict
    obj_val: float
    exec_time: float


def solve_min_servers_max_connectivity(S, C, admissible):
    """
    ILP (Gurobi): min Σ server_var(i)
         s.t. each client j has exactly one i in Adm(j)
              x[i,j] ≤ server_var(i)

    Builds A directly from admissible — the pairs with maximum client
    connectivity — without needing the κ matrix.

    When S == C, server_var(i) = x[i,i] (self-assignment is always admissible
    because κ(i,i):=κ₂(i)). That drops the y variables and the constraints
    y[i] ≥ x[i,j], shrinking the model without changing the minimization optimum.

    Returns: MinServersResult(servers, connections, obj_val, exec_time).
    """
    _set_gurobi_license()

    S = list(S)
    C = list(C)
    S_set = set(S)
    # eliminate_y: with S=C=V (paper), y[i] is redundant — self-assignment
    # x[i,i] ∈ Adm(i) already marks i as an open server (server_var = x[i,i]).
    eliminate_y = S_set == set(C)

    # A = pairs (server i, client j) with the client's maximum connectivity.
    # Only those pairs become x[i,j] variables in the ILP.
    A = []
    for j in C:
        for i in admissible[j]:
            if i in S_set:
                A.append((i, j))

    if not A:
        raise RuntimeError("Empty set A: no admissible pair")

    A_set = set(A)

    model = gp.Model("maxc_min_servers")
    model.Params.OutputFlag = 0

    x = {
        (i, j): model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")
        for i, j in A
    }

    if not eliminate_y:
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y[{i}]") for i in S}

    def server_var(i):
        if eliminate_y:
            # i ∈ Adm(i) is guaranteed; x[i,i] plays the role of y[i].
            return x[i, i]
        return y[i]

    model.setObjective(
        gp.quicksum(server_var(i) for i in S),
        GRB.MINIMIZE,
    )

    for i, j in A:
        model.addConstr(x[i, j] <= server_var(i), name=f"server_connection[{i},{j}]")

    for j in C:
        model.addConstr(
            gp.quicksum(x[i, j] for i in S if (i, j) in A_set) == 1,
            name=f"client_connection[{j}]",
        )

    t0 = time.time()
    model.optimize()
    exec_time = time.time() - t0

    if model.Status == GRB.INFEASIBLE:
        raise RuntimeError("The model is infeasible.")
    if model.SolCount == 0:
        raise RuntimeError(
            f"Gurobi finished with no solution. Status = {model.Status}"
        )

    # Binary variables: .X may be 0.999…; 0.5 is the usual threshold.
    servers = []
    connections = {}
    for i in S:
        if server_var(i).X > BINARY_X_THRESHOLD:
            servers.append(i)
            connections[i] = [
                j for j in C if (i, j) in A_set and x[i, j].X > BINARY_X_THRESHOLD
            ]

    return MinServersResult(
        servers=servers,
        connections=connections,
        obj_val=model.ObjVal,
        exec_time=exec_time,
    )


def run_min_servers(network):
    """
    Run the ILP (Gurobi) and log p, servers, and time.

    Returns: (p, servers, connections, exec_time)
      p            — |servers| = minimum number of MaxC servers
      servers      — chosen vertices
      connections  — client→server assignment
      exec_time    — optimization wall time
    """
    log.print("\n------Minimum number of servers with maximum connectivity------")
    result = solve_min_servers_max_connectivity(
        network.S, network.C, network.admissible
    )
    servers = result.servers
    connections = result.connections
    exec_time = result.exec_time
    p = len(servers)
    log.print("Minimum number of servers = " + str(p))
    log.print("Servers = " + str(servers))
    log.print("Time = " + str(round(exec_time, 4)) + " seconds")
    return p, servers, connections, exec_time
