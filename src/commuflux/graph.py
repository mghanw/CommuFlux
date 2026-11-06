"""From cell-specific B50 networks to one tissue-wide allocation graph (Q additions, then G0).

Q: a boundary arc (receptor->module or module->ligand) absent from a query's original network is added when at least
half of its resampled networks (25 of B50) contain it without a direction conflict and it lies on a stable (omega >= .5)
receptor-to-ligand route. G0: every public LR candidate and then every intracellular arc (by omega) is kept, ties
broken by endpoint and identifier order; G0 breaks no cycle (the flow makes the arcs among a cell's internal nodes
acyclic and ends the relay arcs in sinks). Observed inputs only.
"""
from __future__ import annotations

from collections import Counter
import json
from typing import Any

BOUNDARY = (("receptor", "module"), ("module", "ligand"), ("receptor", "tf"), ("tf", "module"), ("tf", "ligand"))


def port(cell: str, kind: str, feature: str) -> str:
    return "port:" + json.dumps([cell, kind, feature], separators=(",", ":"))


def boundary_counts(replicates: list[dict[str, Any]], roles: dict[str, str]) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    positive: Counter[tuple[str, str]] = Counter()
    conflicts: Counter[tuple[str, str]] = Counter()
    for record in replicates:
        blocked = {frozenset(item["pair"]) for item in record["conflicts"]}
        present: set[tuple[str, str]] = set()
        for tail, head, _basis in record["supported"]:
            if roles[tail] == roles[head] == "module":
                continue
            if frozenset((tail, head)) not in blocked:
                present.add((tail, head))
        positive.update(present)
        for pair in blocked:
            a, b = sorted(pair)
            for tail, head in ((a, b), (b, a)):
                if (roles[tail], roles[head]) in BOUNDARY:
                    conflicts[(tail, head)] += 1
    return positive, conflicts


def _reachable(adjacency: dict[str, set[str]], start: str) -> set[str]:
    found: set[str] = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(adjacency.get(node, ()))
    return found


def _stable(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for edge in edges:
        if edge["omega"] >= .5:
            graph.setdefault(edge["tail"], set()).add(edge["head"])
    return graph


def q_additions(network: dict[str, Any], roles: dict[str, str]) -> list[dict[str, Any]]:
    """Majority boundary arcs missing from the original network that complete a stable receptor-to-ligand route."""
    positive, conflicts = boundary_counts(network["replicates"], roles)
    existing = {frozenset((row["tail"], row["head"])) for row in network["support"]}
    blocked = {frozenset(item["pair"]) for item in network["original"]["conflicts"]}
    planned = len(network["replicates"])
    additions = [{"tail": tail, "head": head, "positive": count, "planned": planned, "omega": count / planned,
                  "direction_conflicts": conflicts[(tail, head)], "basis": "bootstrap_compatible_boundary_consensus"}
                 for (tail, head), count in sorted(positive.items())
                 if 2 * count >= planned and frozenset((tail, head)) not in existing | blocked]
    graph = _stable(network["support"] + additions)
    reverse: dict[str, set[str]] = {}
    for tail, heads in graph.items():
        for head in heads:
            reverse.setdefault(head, set()).add(tail)
    from_receptors = set().union(*(_reachable(graph, node) for node, role in roles.items() if role == "receptor"))
    to_ligands = set().union(*(_reachable(reverse, node) for node, role in roles.items() if role == "ligand"))
    return [edge for edge in additions if edge["tail"] in from_receptors and edge["head"] in to_ligands]


def tissue_graph(cells: list[str], roles: dict[str, str], networks: list[dict[str, Any]],
                 lr_candidates: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """Allocation graph over all cells.

    roles: the CSCN variables (column -> receptor/tf/module/ligand). lr_candidates: (sender, ligand, receiver,
    receptor) in public candidate order. Returns retained intracellular arcs (id, tail port, head port, omega) sorted
    by id, retained LR candidate ids and per-arc decisions. Every candidate is kept (no cycle exclusion), the LR
    candidates inserted before the intracellular arcs; a later arc joining the same two ports is excluded.
    """
    kind_of = {name: ("receptor" if role == "receptor" else role) for name, role in roles.items()}
    feature = {name: json.loads(name)[1] for name in roles}
    nodes: dict[str, tuple[str, str, str, str]] = {}
    for cell in cells:
        for name in roles:
            identifier = port(cell, kind_of[name], feature[name])
            nodes[identifier] = (cell, kind_of[name], feature[name], identifier)
    candidates: list[tuple[str, str, str, float, str]] = []  # id, tail port, head port, priority, kind
    for network in networks:
        cell = network["cell"]
        for edge in network["support"] + network["additions"]:
            if edge["omega"] <= 0:
                continue
            identifier = "cscn:" + json.dumps([cell, edge["tail"], edge["head"]], separators=(",", ":"))
            candidates.append((identifier, port(cell, kind_of[edge["tail"]], feature[edge["tail"]]),
                               port(cell, kind_of[edge["head"]], feature[edge["head"]]), float(edge["omega"]), "intra"))
    for sender, ligand, receiver, receptor in lr_candidates:
        identifier = "lr:" + json.dumps([sender, ligand, receiver, receptor], ensure_ascii=False, separators=(",", ":"))
        candidates.append((identifier, port(sender, "ligand", ligand), port(receiver, "receptor", receptor), 3., "lr"))
    if len({c[0] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate graph candidates")
    ordered = sorted(candidates, key=lambda c: (-c[3], *sorted((nodes[c[1]], nodes[c[2]])), c[0]))
    accepted: set[tuple[str, str]] = set()
    intra: list[tuple[str, str, str, float]] = []
    lr: set[str] = set()
    excluded: Counter[str] = Counter()
    for identifier, tail, head, priority, kind in ordered:
        if (tail, head) in accepted:
            excluded[kind] += 1
            continue
        accepted.add((tail, head))
        if kind == "intra":
            intra.append((identifier, tail, head, priority))
        elif kind == "lr":
            lr.add(identifier)
    return {"intra": sorted(intra), "lr": lr, "excluded": dict(excluded),
            "candidates": {"intra": sum(c[4] == "intra" for c in candidates), "lr": len(lr_candidates)}}
