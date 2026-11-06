"""CommuFlux inference on one run directory (data.py): modules, cell-specific CSCN networks with B50 support, the
tissue graph, and the joint entropic flow with its relay shares.

The stages run in one process (infer) or as separate jobs: prepare, part (B50 networks of a share of the cells),
finish (merge the parts, flow, scores).
"""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import cscn, flow, graph, modules
from .data import CANDIDATE_RADIUS, candidate_axes, read_input, within_pairs

BUDGET_QUANTILE = 75.  # percentile of the positive ligand (receptor) CP10k of the tissue that scales the port budgets
PREFIX = "commuflux"
CONTEXT = "context.pkl"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def module_inputs(observations: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    """WGCNA modules for CSCN and the flow (modules.build_representation): the TF, ligand and receptor genes are left
    out of every eigengene and level (they keep their module index, which pairs a TF variable with the module of its
    gene)."""
    genes = observations["gene_ids"].astype(str).tolist()
    tfs = sorted({str(row["target"]) for row in resources["receptor_tf"]})
    signaling = sorted(set(tfs) | {str(g) for role in ("ligands", "receptors") for g in resources["roles"][role]})
    return modules.build_representation(observations["counts"], genes, exclude=signaling)


def budget_scales(observations: dict[str, Any], resources: dict[str, Any]) -> tuple[float, float]:
    """The tissue's budget scales: the BUDGET_QUANTILE percentile of the positive CP10k of the ligand genes over all
    cells, and of the receptor genes."""
    counts = np.asarray(observations["counts"], dtype=np.float64)
    cp10k = counts / counts.sum(axis=1, keepdims=True) * 10000
    genes = {g: i for i, g in enumerate(observations["gene_ids"].astype(str).tolist())}
    scales = []
    for role in ("ligands", "receptors"):
        values = cp10k[:, [genes[str(g)] for g in resources["roles"][role]]].ravel()
        scales.append(float(np.percentile(values[values > 0], BUDGET_QUANTILE)))
    return scales[0], scales[1]


def flow_inputs(observations: dict[str, Any], resources: dict[str, Any], tissue: dict[str, Any],
                rep: dict[str, Any]) -> dict[str, Any]:
    """Ports, budgets (CP10k over the tissue's budget scales), capacities and arcs of the communication flow."""
    cells = observations["cell_ids"].astype(str).tolist()
    genes = observations["gene_ids"].astype(str).tolist()
    ci, gi = {c: i for i, c in enumerate(cells)}, {g: i for i, g in enumerate(genes)}
    counts = np.asarray(observations["counts"])
    coordinates = np.asarray(observations["coordinates"])
    totals = counts.sum(axis=1)
    divided = counts / totals[:, None] * 10000
    normalized = counts * (10000. / totals)[:, None]
    ligand_scale, receptor_scale = budget_scales(observations, resources)
    ligand_budget = {graph.port(c, "ligand", l): float(1. * (np.float64(divided[ci[c], gi[l]]) / ligand_scale))
                     for c in cells for l in resources["roles"]["ligands"]}
    receptor_budget = {graph.port(c, "receptor", r): float(normalized[ci[c], gi[r]]) / receptor_scale
                       for c in cells for r in resources["roles"]["receptors"]}
    lr = []
    for identifier in sorted(tissue["lr"]):
        sender, ligand, receiver, receptor = json.loads(identifier[3:])
        distance = float(np.linalg.norm(coordinates[ci[sender]] - coordinates[ci[receiver]]))
        lr.append((identifier, graph.port(sender, "ligand", ligand), graph.port(receiver, "receptor", receptor), (distance / CANDIDATE_RADIUS) ** 2))
    receptor_tfs: dict[str, list[tuple[str, float]]] = {}
    for row in resources["receptor_tf"]:
        receptor_tfs.setdefault(str(row["source"]), []).append((str(row["target"]), float(row.get("weight", 1.))))
    tf_level = {flow.node(c, "tf", tf): float(normalized[ci[c], gi[tf]])
                for c in cells for pairs in receptor_tfs.values() for tf, _ in pairs}
    gem = {flow.node(c, "module", m): float(rep["levels"][j, k]) for j, c in enumerate(cells) for k, m in enumerate(rep["module_ids"])}
    return {"cells": cells, "ligand_budget": ligand_budget, "receptor_budget": receptor_budget, "lr": lr,
            "intra": [(tail, head, omega) for _id, tail, head, omega in tissue["intra"]],
            "gem": gem, "tf_level": tf_level, "receptor_tfs": receptor_tfs}


def infer(run: Path, workers: int, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The whole inference in one process; writes the scores of the three arms and the tables of the full model."""
    prepare(run, log)
    networks = cell_networks(run, workers, log)
    write_networks(run / "inference/networks.json.gz", networks)
    return flow_scores(run, networks)


def prepare(run: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """First stage: modules, the CSCN input table and the background knowledge (role tiers, TF/module equivalence,
    receptor -> TF prior), written to RUN/inference with the context of the later stages."""
    output = run / "inference"
    if output.exists():
        raise FileExistsError("Keep earlier inferences")
    output.mkdir(parents=True)
    observations, resources = read_input(run)
    cells = observations["cell_ids"].astype(str).tolist()
    genes = observations["gene_ids"].astype(str).tolist()
    rep = module_inputs(observations, resources)
    np.savez_compressed(output / "representation.npz", eigengenes=rep["eigengenes"], levels=rep["levels"],
                        assignment=rep["assignment"], module_ids=np.array(rep["module_ids"]))
    write_json(output / "representation.json", rep["summary"])
    log(f"Modules: {len(rep['module_ids'])}, soft power {rep['summary'].get('soft_power')}")
    table, roles = cscn.local_table(observations["counts"], genes, cells, resources, rep["eigengenes"], rep["module_ids"])
    table.to_csv(output / "local-table.csv", lineterminator="\n")
    groups = cscn.spatial_partition(np.asarray(observations["coordinates"]))
    knowledge = {**cscn.background_knowledge(genes, resources, rep["assignment"], rep["module_ids"]),
                 "prior": cscn.receptor_tf_prior(resources, roles)}
    log(f"Receptor -> TF prior: {len(knowledge['prior'])} pairs")
    context = {"inputs": {"modules": rep}, "roles": roles, "groups": groups, "knowledge": knowledge}
    with open(output / CONTEXT, "wb") as stream:
        pickle.dump(context, stream)
    return context


def context_of(run: Path) -> dict[str, Any]:
    with open(run / "inference" / CONTEXT, "rb") as stream:
        context: dict[str, Any] = pickle.load(stream)
    return context


def cell_networks(run: Path, workers: int, log: Callable[[str], None] = print, part: int = 1,
                  parts: int = 1) -> list[dict[str, Any]]:
    """B50 networks with their Q additions and arc signs of a prepared run: every cell, or the cells part - 1,
    part - 1 + parts, part - 1 + 2 parts, ... (part 1 .. parts; interleaved, so the parts take about equal time)."""
    if not 1 <= part <= parts:
        raise ValueError("part is 1 .. parts")
    context = context_of(run)
    # CSCN reads its table back from the CSV, so that the inputs do not depend on the normalization formula's last bits.
    table = pd.read_csv(run / "inference/local-table.csv", index_col=0)
    roles, queries = context["roles"], range(part - 1, len(table), parts)
    networks = cscn.bootstrap_networks(table, roles, context["groups"], workers,
                                       progress=lambda n: log(f"B50 {n}/{len(queries)}"),
                                       knowledge=context["knowledge"], queries=queries)
    for network in networks:
        network["additions"] = graph.q_additions(network, roles)
        del network["replicates"]
    return cscn.arc_signs(table, networks, roles)


def write_networks(path: Path, networks: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(networks, stream)


def part(run: Path, workers: int, index: int, parts: int, log: Callable[[str], None] = print) -> Path:
    """Network stage as one of `parts` jobs; writes inference/networks-part-INDEX-of-PARTS.json.gz."""
    path = run / "inference" / f"networks-part-{index}-of-{parts}.json.gz"
    if path.exists():
        raise FileExistsError("Keep earlier network parts")
    write_networks(path, cell_networks(run, workers, log, index, parts))
    return path


def merged_networks(run: Path) -> list[dict[str, Any]]:
    """The networks of every part in query order; each cell of the CSCN table exactly once."""
    networks: list[dict[str, Any]] = []
    for path in sorted((run / "inference").glob("networks-part-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            networks.extend(json.load(stream))
    networks.sort(key=lambda network: network["query"])
    cells = pd.read_csv(run / "inference/local-table.csv", index_col=0).index.astype(str).tolist()
    if [(network["query"], network["cell"]) for network in networks] != list(enumerate(cells)):
        raise ValueError("The network parts do not cover every cell exactly once")
    return networks


def finish(run: Path) -> dict[str, Any]:
    """Last stage after the part jobs: merges the parts into inference/networks.json.gz, then the flow and scores."""
    target = run / "inference/networks.json.gz"
    if target.exists():
        raise FileExistsError("Keep earlier inferences")
    networks = merged_networks(run)
    write_networks(target, networks)
    return flow_scores(run, networks)


def flow_scores(run: Path, networks: list[dict[str, Any]]) -> dict[str, Any]:
    """Tissue graph from the cell networks, the flow arms and their edge scores: scores/commuflux_<arm>.npz (S aligned
    to data.candidate_axes; the full arm also holds relay, cells x receptors x ligands), inference/graph.json and
    inference/flow.json, and the tables of the full model (write_tables)."""
    context = context_of(run)
    roles, rep = context["roles"], context["inputs"]["modules"]
    output = run / "inference"
    observations, resources = read_input(run)
    cells = observations["cell_ids"].astype(str).tolist()
    axes = candidate_axes(observations, resources)
    senders, receivers = within_pairs(observations)
    lr_candidates = [(s, str(x["source"]), r, str(x["target"])) for x in resources["lr"] for s, r in zip(senders, receivers)]
    tissue = graph.tissue_graph(cells, roles, networks, lr_candidates)
    write_json(output / "graph.json", {"intra": tissue["intra"], "lr_retained": len(tissue["lr"]),
                                       "lr_candidates": len(lr_candidates), "excluded": tissue["excluded"]})
    data = flow_inputs(observations, resources, tissue, rep)
    result = flow.communication_flow(data, structures=flow.STRUCTURES)
    scores = run / "scores"
    scores.mkdir(exist_ok=True)
    keys = [graph.port(s, "ligand", l) + "|" + graph.port(r, "receptor", rc)
            for s, l, r, rc in zip(axes["sender"], axes["ligand"], axes["receiver"], axes["receptor"])]
    by_ports = {tail + "|" + head: identifier for identifier, tail, head, _cost in data["lr"]}
    summary: dict[str, Any] = {"offset": result["offset"], "intracellular_scale": result["intracellular_scale"],
                               "budget_scales": list(budget_scales(observations, resources)), "arms": {}}
    for structure, arm in result["arms"].items():
        flows = arm["flows"]
        score = np.array([flows.get(by_ports.get(key, ""), 0.) for key in keys], dtype=np.float64)
        reach = arm["relay_reach"]
        if reach:
            relay = np.array([[[reach.get(graph.port(c, "receptor", r), {}).get(graph.port(c, "ligand", l), 0.)
                                for l in resources["roles"]["ligands"]] for r in resources["roles"]["receptors"]] for c in cells])
            np.savez_compressed(scores / f"{PREFIX}_{structure}.npz", S=score, relay=relay)
        else:
            np.savez_compressed(scores / f"{PREFIX}_{structure}.npz", S=score)
        summary["arms"][structure] = {"usage": arm["usage"], "sizes": arm["sizes"],
                                      "certificate": {k: v for k, v in arm["certificate"].items() if not isinstance(v, (list, dict))}}
    write_json(output / "flow.json", summary)
    write_tables(run)
    return summary


def write_tables(run: Path, arm: str = "full") -> None:
    """communication.csv.gz (sender, receiver, ligand, receptor, score; positive scores) and relay.csv.gz (cell,
    receptor, ligand, routing fraction of the receptor's inflow to the ligand; positive fractions) of one arm."""
    observations, resources = read_input(run)
    axes = candidate_axes(observations, resources)
    stored = load_npz(run / "scores" / f"{PREFIX}_{arm}.npz")
    positive = stored["S"] > 0
    pd.DataFrame({key: axes[key][positive] for key in ("sender", "receiver", "ligand", "receptor")} | {"score": stored["S"][positive]}
                 ).to_csv(run / "communication.csv.gz", index=False)
    # The relay array is stored only when a receptor reaches a ligand in some cell; otherwise the table has no rows.
    relay = stored.get("relay", np.zeros((0, 0, 0)))
    c, r, l = np.nonzero(relay > 0)
    cells = observations["cell_ids"].astype(str)
    pd.DataFrame({"cell": cells[c], "receptor": np.array(resources["roles"]["receptors"])[r],
                  "ligand": np.array(resources["roles"]["ligands"])[l], "fraction": relay[c, r, l]}
                 ).to_csv(run / "relay.csv.gz", index=False)
