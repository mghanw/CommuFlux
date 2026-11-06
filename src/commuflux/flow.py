"""Whole-network entropy flow for cell-cell communication.

Ports: each cell's ligands and receptors, with expression budgets (CP10k over the tissue's budget scales). LR arcs: the
retained candidates within the candidate radius, cost (d / radius)^2 plus one dilute offset delta chosen so that the
transport-only flow ships 1% of all ligand. Receiver side (arm "full"): receptor -> its annotated TFs (fixed links,
cost 0) -> the cell's learned CSCN network after them (TF and module nodes, cost -log omega; a learned receptor arc
leaves from the receptor's TFs), its internal arcs made acyclic by support; module -> ligand relay arcs end in sinks
of the cell's ligand ports. Capacities: the receptors' response pull (response_capacity), scaled by the load-matched
intracellular unit. Entropy (temperature 1) acts on every arc and pool residual; each receptor also has a silent
outlet and each internal node an entropic exit (termination cost 0). Reference arms: transport only ("transport") and
the TF layer only ("tf").
"""
from __future__ import annotations

from collections import defaultdict
import heapq
import json
from math import exp, fsum, log, sqrt
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .data import CANDIDATE_RADIUS
from .solver import Arcs, NetworkEntropyProblem, solve_network_entropy

THETA = ETA = 1.
OUTLET_COST = AUTONOMOUS_COST = UNMATCHED_LIGAND_COST = UNMATCHED_RECEPTOR_COST = 1.
UNMATCHED_MODULE_COST = 0.
DILUTE_LIGAND_USE = .01
SCALE = 1.
STRUCTURES = ("transport", "tf", "full")
INTERNAL = ("module", "tf")


def port(identifier: str) -> tuple[str, str, str]:
    cell, role, feature = json.loads(identifier.removeprefix("port:"))
    return str(cell), str(role), str(feature)


def node(cell: str, role: str, feature: str) -> str:
    return "port:" + json.dumps([cell, role, feature], separators=(",", ":"))


def factor_of(identifier: str) -> str:
    _, role, feature = port(identifier)
    return role + ":" + (feature.split("@", 1)[0] if role == "tf" else feature)


def response_capacity(candidates: list[tuple[str, str, float]], receptors: list[str], level: dict[str, float],
                      budget: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Receiver-side pull: a receptor's capacity is its pathway response, the reliability-weighted mean of the
    normalized levels of the nodes it reaches through the cell's network (TF and modules; weight: product of omega
    along the most reliable path, level over the feature's tissue mean), so that more downstream branches do not add
    pull. Per receptor gene, kappa scales the responses so that their tissue sum equals the receptors' budgets; each
    receptor's pull is then shared among its reached nodes in proportion to weight x normalized level (node capacity:
    the sum over the receptors reaching it). A zero-count TF passes a path on in the search but gets no capacity; it
    is cut together with what only it reaches: the network keeps the nodes reached from the receptors
    through nodes with capacity. Returns the capacities and kappa per receptor gene."""
    forward: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for tail, head, omega in candidates:
        if omega > 0 and port(head)[1] in INTERNAL:
            forward[tail].append((head, -log(omega)))
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for name, value in level.items():
        total[factor_of(name)] += value
        count[factor_of(name)] += 1
    mean = {factor: total[factor] / count[factor] for factor in total}

    def normalized(name: str) -> float:
        return level.get(name, 0.) / mean[factor_of(name)] if mean.get(factor_of(name), 0.) > 0 else 0.

    def open_(name: str) -> bool:  # zero-level modules close the path; zero-count TFs pass a signal on
        return normalized(name) > 0 or port(name)[1] == "tf"
    reached: dict[str, dict[str, float]] = {}
    for receptor in receptors:
        costs = {receptor: 0.}
        frontier = [(0., receptor)]
        settled: set[str] = set()
        while frontier:
            cost, name = heapq.heappop(frontier)
            if name in settled:
                continue
            settled.add(name)
            for head, step in forward.get(name, []):
                if open_(head) and cost + step < costs.get(head, np.inf):
                    costs[head] = cost + step
                    heapq.heappush(frontier, (cost + step, head))
        reached[receptor] = {name: exp(-cost) for name, cost in costs.items() if name != receptor}
    response = {r: fsum(w * normalized(j) for j, w in nodes.items()) / fsum(nodes.values())
                for r, nodes in reached.items() if nodes}
    by_gene: dict[str, list[str]] = defaultdict(list)
    for receptor in response:
        by_gene[port(receptor)[2]].append(receptor)
    kappa = {gene: fsum(budget[r] for r in members) / fsum(response[r] for r in members)
             if fsum(response[r] for r in members) > 0 else 0. for gene, members in by_gene.items()}
    capacity: dict[str, float] = defaultdict(float)
    for receptor, value in response.items():
        nodes = reached[receptor]
        weights = {j: w * normalized(j) for j, w in nodes.items()}
        spread = fsum(weights.values())
        for j in nodes:
            capacity[j] += kappa[port(receptor)[2]] * value * weights[j] / spread if spread > 0 else 0.
    positive = {name for name, value in capacity.items() if value > 0}
    reached_alive: set[str] = set()  # reached from a receptor through nodes with capacity only
    queue = list(receptors)
    while queue:
        for head, _ in forward.get(queue.pop(), []):
            if head in positive and head not in reached_alive:
                reached_alive.add(head)
                queue.append(head)
    return {name: value for name, value in capacity.items() if name in reached_alive}, kappa


def acyclic_internal(candidates: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """The arcs among internal nodes (TFs, modules) inserted by decreasing omega, ties by endpoint order, an arc that
    would close a directed cycle left out; every other arc kept. With relay sinks (flow_solver relay_sink) the flow
    network is then acyclic, so no flow circulates without a source."""
    internal = [k for k, (tail, head, _) in enumerate(candidates)
                if port(tail)[1] in INTERNAL and port(head)[1] in INTERNAL and tail != head]
    children: dict[str, set[str]] = defaultdict(set)

    def reaches(start: str, goal: str) -> bool:
        seen, stack = {start}, [start]
        while stack:
            for following in children[stack.pop()]:
                if following == goal:
                    return True
                if following not in seen:
                    seen.add(following)
                    stack.append(following)
        return False

    dropped: set[int] = set()
    for k in sorted(internal, key=lambda i: (-candidates[i][2], candidates[i][0], candidates[i][1])):
        tail, head, _ = candidates[k]
        if head == tail or reaches(head, tail):
            dropped.add(k)
        else:
            children[tail].add(head)
    return [arc for k, arc in enumerate(candidates) if k not in dropped]


def network(data: dict[str, Any], structure: str, delta: float = 0.,
            scale: float = 1.) -> tuple[NetworkEntropyProblem, dict[str, Any]]:
    """Prune ports and internal nodes that cannot carry flow, calibrate capacities and index the problem.

    Structures: transport only ("transport"), receptor -> its annotated TFs ("tf"), and the TFs followed by the cell's
    learned network ("full": a learned receptor arc leaves from each of the receptor's TFs instead; a learned receptor
    -> TF arc is the fixed link itself). The arcs among internal nodes are made acyclic (acyclic_internal) before the
    response pull sets the capacities (response_capacity); relay arcs end in sinks of the ligand ports; every internal
    node terminates through an entropic exit at cost 0."""
    if structure not in STRUCTURES:
        raise ValueError("Unknown flow structure")
    lb, rb = data["ligand_budget"], data["receptor_budget"]
    level = {**data["gem"], **data["tf_level"]}
    lr = [edge for edge in data["lr"] if lb[edge[1]] > 0 and rb[edge[2]] > 0]
    receptors, senders = sorted({edge[2] for edge in lr}), sorted({edge[1] for edge in lr})
    candidates: list[tuple[str, str, float]] = []
    relay_candidates: list[tuple[str, str, float]] = []
    if structure in ("tf", "full"):
        for receptor in receptors:
            cell, _, gene = port(receptor)
            for tf, weight in data["receptor_tfs"].get(gene, []):
                target = node(cell, "tf", tf)
                level[target] = data["tf_level"][node(cell, "tf", tf)]
                candidates.append((receptor, target, weight))
    if structure == "full":
        for tail, head, omega in data["intra"]:
            cell, role, gene = port(tail)
            if port(head)[1] == "ligand":
                relay_candidates.append((tail, head, omega))  # relay arc: module -> the cell's own ligand port (its sink)
                continue
            if role == "receptor":
                for tf, _ in data["receptor_tfs"].get(gene, []):
                    if node(cell, "tf", tf) == head:  # a learned receptor -> TF arc is the fixed link itself
                        continue
                    candidates.append((node(cell, "tf", tf), head, omega))
            else:
                candidates.append((tail, head, omega))
    candidates = acyclic_internal(candidates)
    capacity: dict[str, float] = {}
    kappa: dict[str, float] = {}
    if candidates:
        capacity, kappa = response_capacity(candidates, receptors, level, rb)
    capacity = {internal: value * scale for internal, value in capacity.items()}
    kept, receptor_set = set(capacity), set(receptors)
    rm: list[tuple[str, str, float]] = []
    mm: list[tuple[str, str, float]] = []
    for tail, head, omega in candidates:
        tail_role, head_role = port(tail)[1], port(head)[1]
        if tail_role == "receptor" and tail in receptor_set and head in kept:
            rm.append((tail, head, omega))
        elif tail_role in INTERNAL and head_role in INTERNAL and tail in kept and head in kept:
            mm.append((tail, head, omega))
    internals = sorted(capacity)
    li, ri, mi = ({name: i for i, name in enumerate(items)} for items in (senders, receptors, internals))
    ml = sorted({(tail, head, omega) for tail, head, omega in relay_candidates if tail in kept and head in li})

    def arcs(rows: list[tuple[str, str, float]], tails: dict[str, int], heads: dict[str, int]) -> Arcs:
        return Arcs(np.array([tails[t] for t, _, _ in rows], dtype=np.int64), np.array([heads[h] for _, h, _ in rows], dtype=np.int64),
                    np.array([-ETA * log(w) for _, _, w in rows], dtype=float))

    problem = NetworkEntropyProblem(
        ligand_budget=np.array([lb[name] for name in senders], dtype=float),
        receptor_budget=np.array([rb[name] for name in receptors], dtype=float),
        module_capacity=np.array([capacity[name] for name in internals], dtype=float),
        terminal=np.ones(len(internals), dtype=bool),
        lr=Arcs(np.array([li[edge[1]] for edge in lr], dtype=np.int64), np.array([ri[edge[2]] for edge in lr], dtype=np.int64),
                np.array([edge[3] + delta for edge in lr], dtype=float)),
        rm=arcs(rm, ri, mi), mm=arcs(mm, mi, mi), ml=arcs(ml, mi, li),
        autonomous_cost=AUTONOMOUS_COST, outlet_cost=OUTLET_COST - ETA * log(scale), unmatched_ligand_cost=UNMATCHED_LIGAND_COST,
        unmatched_receptor_cost=UNMATCHED_RECEPTOR_COST, unmatched_module_cost=UNMATCHED_MODULE_COST,
        theta=THETA, eta=ETA, termination_cost=0., relay_sink=True)
    return problem, {"lr": [edge[0] for edge in lr], "receptors": receptors, "modules": internals, "senders": senders,
                     "rm": rm, "mm": mm, "ml": ml, "kappa": kappa}


def dilute_offset(data: dict[str, Any], target: float = DILUTE_LIGAND_USE) -> float:
    """LR cost offset at which the transport-only flow, in its dilute limit, ships `target` of all ligand.

    With free pools near their budgets the entropic outlet balances each receptor port, so its inflow is
    exp(-delta / 2) sqrt(O S_r) (theta = eta = 1), S_r the undiluted push into the port and O = exp(-outlet cost).
    """
    lb, rb = data["ligand_budget"], data["receptor_budget"]
    push: dict[str, float] = defaultdict(float)
    for _, tail, head, cost in data["lr"]:
        if lb[tail] > 0 and rb[head] > 0:
            push[head] += exp((UNMATCHED_LIGAND_COST + UNMATCHED_RECEPTOR_COST - AUTONOMOUS_COST - cost) / THETA) * lb[tail] * rb[head]
    use = fsum(sqrt(exp(-OUTLET_COST / ETA) * value) for value in push.values()) / fsum(lb.values())
    return 2 * THETA * log(use / target)


def relay_reach(problem: NetworkEntropyProblem, solution: Any, index: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Share of each receptor port's inflow that leaves its cell through a relay arc to a ligand port (its sink).

    Proportional path decomposition of the solved flow: from every internal node, a unit of flow continues along each
    outgoing arc (module arcs, relay arcs, termination) in proportion to the arc's flow, so the share reaching ligand
    port l from node i solves h = P h + E (P: module -> module shares, E: module -> ligand port shares); termination
    leaves every node, so I - P is invertible on graphs with cycles too. A receptor port's share is the inflow-weighted
    mean over its receptor -> internal arcs; the silent outlet carries the rest."""
    if not len(problem.ml.tail):
        return {}
    n, senders, receptors = len(index["modules"]), index["senders"], index["receptors"]
    out = (np.bincount(problem.mm.tail, weights=solution.mm, minlength=n)
           + np.bincount(problem.ml.tail, weights=solution.ml, minlength=n) + solution.termination)
    safe = np.where(out > 0, out, 1.)
    step = sparse.csc_matrix((solution.mm / safe[problem.mm.tail], (problem.mm.tail, problem.mm.head)), shape=(n, n))
    exit_ = sparse.csc_matrix((solution.ml / safe[problem.ml.tail], (problem.ml.tail, problem.ml.head)), shape=(n, len(senders)))
    solved = spsolve(sparse.identity(n, format="csc") - step, exit_)
    reach = sparse.csc_matrix(solved if sparse.issparse(solved) else np.reshape(solved, (n, -1)))
    inflow = np.bincount(problem.lr.head, weights=solution.lr, minlength=len(receptors))
    entry_ = sparse.csr_matrix((solution.rm / np.where(inflow > 0, inflow, 1.)[problem.rm.tail], (problem.rm.tail, problem.rm.head)),
                               shape=(len(receptors), n))
    shares = sparse.coo_matrix(entry_ @ reach)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for r, l, value in zip(shares.row.tolist(), shares.col.tolist(), shares.data.tolist()):
        if value > 0:
            result[receptors[r]][senders[l]] = float(value)
    return dict(result)


def solve(data: dict[str, Any], structure: str, delta: float = 0., scale: float = 1.) -> dict[str, Any]:
    problem, index = network(data, structure, delta, scale)
    solution = solve_network_entropy(problem, tolerance=1e-8, max_sweeps=20000, check_every=10)
    if not solution.certificate["converged"]:
        raise RuntimeError("The dual ascent did not reach the residual tolerance")
    into_receptor = np.bincount(problem.lr.head, weights=solution.lr, minlength=len(index["receptors"]))
    relay_in = np.bincount(problem.ml.head, weights=solution.ml, minlength=len(index["senders"]))
    sent = np.bincount(problem.lr.tail, weights=solution.lr, minlength=len(index["senders"]))
    ligand_total, receptor_total = fsum(data["ligand_budget"].values()), fsum(data["receptor_budget"].values())
    return {"flows": dict(zip(index["lr"], solution.lr.tolist())), "certificate": solution.certificate,
            "receptor_inflow": dict(zip(index["receptors"], into_receptor.tolist())),
            "relay_inflow": {port_id: float(value) for port_id, value in zip(index["senders"], relay_in) if value > 0},
            "relay_reach": relay_reach(problem, solution, index),
            "ligand_outflow": dict(zip(index["senders"], sent.tolist())),
            "usage": {"ligand_used_fraction": float(solution.lr.sum()) / ligand_total,
                      "receptor_inflow_fraction": float(into_receptor.sum()) / receptor_total},
            "sizes": {"LR_arcs": len(index["lr"]), "RM_arcs": len(index["rm"]), "MM_arcs": len(index["mm"]),
                      "internal_nodes": len(index["modules"]), "ML_arcs": len(index["ml"])}}


def communication_flow(data: dict[str, Any], structures: tuple[str, ...] = STRUCTURES,
                       scale: float = SCALE) -> dict[str, Any]:
    """The flow arms at spatial scale s (default 1) in the dilute regime with load-matched intracellular units."""
    data = {**data, "lr": [(edge, tail, head, cost * (CANDIDATE_RADIUS / scale) ** 2) for edge, tail, head, cost in data["lr"]]}
    offset = dilute_offset(data)
    saturated = solve(data, "transport")
    unscaled = solve(data, "transport", delta=offset)
    factor = (unscaled["usage"]["receptor_inflow_fraction"] / saturated["usage"]["receptor_inflow_fraction"]) ** 2
    arms = {structure: solve(data, structure, delta=offset, scale=factor) for structure in structures}
    return {"scale": scale, "offset": offset, "intracellular_scale": factor, "arms": arms}
