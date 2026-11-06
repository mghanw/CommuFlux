"""Input of a CommuFlux run and its candidate cell pairs.

Run layout (one directory per tissue section):
  input/{observations.npz, resources.json}   counts, cell ids, gene ids, coordinates; LR pairs and receptor -> TF links
  inference/                                 modules, CSCN table, context, B50 networks, tissue graph, flow summary
  scores/commuflux_<arm>.npz                 edge scores S aligned to candidate_axes; arm "full" also holds relay shares
  communication.csv.gz, relay.csv.gz         the full model's scores as tables (write_tables)

Coordinates are stored in internal units in which the candidate radius is CANDIDATE_RADIUS (1.5): write_input rescales
the given coordinates so that `radius` (in their unit, e.g. micrometres) maps to it. The distance cost of an LR arc is
then (d / radius)^2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.distance import cdist

CANDIDATE_RADIUS = 1.5  # internal coordinate unit: the candidate radius


def write_input(run: Path, counts: Any, genes: Sequence[str], cells: Sequence[str], coordinates: Any,
                lr_pairs: Sequence[tuple[str, str]], receptor_tf: Sequence[tuple[str, str]], radius: float) -> Path:
    """Writes RUN/input from raw counts (cells x genes, integers), 2-D cell coordinates, the LR pairs (ligand, receptor)
    and the receptor -> TF links (receptor, TF). Every ligand, receptor and TF must be a gene of the panel, and every
    receptor needs at least one TF. radius: the interaction radius in the unit of the coordinates."""
    output = Path(run) / "input"
    if output.exists():
        raise FileExistsError("Use a new run directory")
    counts = np.asarray(counts)
    genes, cells = [str(g) for g in genes], [str(c) for c in cells]
    coordinates = np.asarray(coordinates, dtype=float)
    if counts.shape != (len(cells), len(genes)) or coordinates.shape != (len(cells), 2):
        raise ValueError("counts must be cells x genes and coordinates cells x 2")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("Every cell needs a positive library size")
    lr = [{"source": str(l), "target": str(r)} for l, r in dict.fromkeys((str(l), str(r)) for l, r in lr_pairs)]
    links = [{"source": str(r), "target": str(t), "weight": 1.} for r, t in dict.fromkeys((str(r), str(t)) for r, t in receptor_tf)]
    ligands = sorted({row["source"] for row in lr})
    receptors = sorted({row["target"] for row in lr})
    links = [row for row in links if row["source"] in receptors]
    missing = sorted((set(ligands) | set(receptors) | {row["target"] for row in links}) - set(genes))
    if missing:
        raise ValueError(f"Genes not in the panel: {missing}")
    without = [r for r in receptors if not any(row["source"] == r for row in links)]
    if without:
        raise ValueError(f"Receptors without a TF: {without}")
    output.mkdir(parents=True)
    scale = CANDIDATE_RADIUS / float(radius)
    np.savez_compressed(output / "observations.npz", counts=counts, gene_ids=np.array(genes, dtype=str),
                        cell_ids=np.array(cells, dtype=str), coordinates=coordinates * scale)
    resources = {"lr": lr, "receptor_tf": links, "roles": {"ligands": ligands, "receptors": receptors},
                 "radius": float(radius)}
    (output / "resources.json").write_text(json.dumps(resources, indent=1) + "\n", encoding="utf-8")
    return output


def read_input(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The observations (counts, gene_ids, cell_ids, coordinates in internal units) and resources of RUN/input."""
    with np.load(Path(run) / "input/observations.npz", allow_pickle=False) as archive:
        observations = {key: archive[key] for key in archive.files}
    return observations, json.loads((Path(run) / "input/resources.json").read_text(encoding="utf-8"))


def candidate_axes(observations: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    """Candidate edges: every (sender, receiver) pair of distinct cells within the candidate radius, crossed with the LR
    pairs (sorted); the order of the score vectors S."""
    cells = observations["cell_ids"].astype(str)
    distance = cdist(observations["coordinates"], observations["coordinates"])
    sender, receiver = np.nonzero((distance <= CANDIDATE_RADIUS) & ~np.eye(len(cells), dtype=bool))
    pairs = sorted({(str(item["source"]), str(item["target"])) for item in resources["lr"]})
    return {"pairs": pairs, "sender_index": np.tile(sender, len(pairs)), "receiver_index": np.tile(receiver, len(pairs)),
            "sender": cells[np.tile(sender, len(pairs))], "receiver": cells[np.tile(receiver, len(pairs))],
            "ligand": np.repeat([p[0] for p in pairs], len(sender)), "receptor": np.repeat([p[1] for p in pairs], len(sender)),
            "pair_index": np.repeat(np.arange(len(pairs)), len(sender))}


def within_pairs(observations: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Candidate (sender, receiver) cells within the candidate radius, in the order of the LR candidates of the graph."""
    cells = observations["cell_ids"].astype(str)
    coordinates = np.asarray(observations["coordinates"])
    distances = np.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
    senders, receivers = np.where((distances <= CANDIDATE_RADIUS ** 2) & ~np.eye(len(cells), dtype=bool))
    return cells[senders], cells[receivers]
