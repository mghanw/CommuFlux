"""CommuFlux input from a 10x Xenium output: a square region of cells, an LR panel and receptor -> TF links.

Inputs: the Xenium cell-feature matrix and cell centroids (and the 10x graph-based clusters, used only to choose and
describe the region); LR pairs, e.g. from the LIANA consensus resource (a receptor complex is represented by its
ligand-binding subunit); receptor -> TF links from CellCall (KEGG-derived ligand-receptor-TF triples): the receptor's
TFs in KEGG signalling maps that are TFs in CollecTRI and detected in the region, at most `max_tfs` per receptor (the
TFs linked to the receptor in the most signalling maps first, then the most often detected).

Usage: python examples/xenium/prepare_xenium.py CONFIG.json RUN, then commuflux infer --run RUN
CONFIG keys: dataset (Xenium output directory), radius_um, lr ([[ligand, receptor], ...]), clusters ({id: cell type}),
roi ([x0, y0, x1, y1] um) or auto ({"sizes_um": [...], "cells": [lo, hi], "objective": {type or "gene:G": weight}}),
min_counts (default 20), min_detection (default .01), min_detection_tf (default .02), max_tfs (default 2), cellcall
(CellCall new_ligand_receptor_TFs table), collectri (CollecTRI table with a "source" column).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from commuflux import write_input

# KEGG signalling-pathway maps whose CellCall rows give a receptor's TFs (disease and other maps left out).
SIGNALLING = {"hsa04010", "hsa04012", "hsa04014", "hsa04015", "hsa04020", "hsa04022", "hsa04024", "hsa04064",
              "hsa04066", "hsa04068", "hsa04070", "hsa04071", "hsa04072", "hsa04150", "hsa04151", "hsa04152",
              "hsa04310", "hsa04330", "hsa04340", "hsa04350", "hsa04370", "hsa04371", "hsa04390", "hsa04630",
              "hsa04668", "hsa04062", "hsa04620", "hsa04621", "hsa04622", "hsa04650", "hsa04657", "hsa04658",
              "hsa04659", "hsa04660", "hsa04662", "hsa04664", "hsa04670"}


def load_xenium(directory: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame]:
    """Cells x genes counts (gene-expression features only), gene names, cell ids, and the cell table (centroids, um)."""
    name = directory.name
    with h5py.File(directory / f"{name}_cell_feature_matrix.h5") as handle:
        matrix = handle["matrix"]
        shape = matrix["shape"][:]
        counts = sp.csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape).T.tocsr()
        genes = matrix["features/name"][:].astype(str)
        kinds = matrix["features/feature_type"][:].astype(str)
        barcodes = matrix["barcodes"][:].astype(str)
    keep = kinds == "Gene Expression"
    cells = pd.read_csv(directory / f"{name}_cells.csv.gz").set_index("cell_id").reindex(barcodes)
    return counts[:, keep], genes[keep], barcodes, cells


def cluster_types(directory: Path, barcodes: np.ndarray, clusters: dict[str, str]) -> np.ndarray:
    table = pd.read_csv(directory / "analysis/clustering/gene_expression_graphclust/clusters.csv").set_index("Barcode")
    labels = table.reindex(barcodes)["Cluster"]
    return np.array([clusters.get(str(int(c)), "other") if pd.notna(c) else "other" for c in labels])


def auto_roi(xy: np.ndarray, masks: dict[str, np.ndarray], auto: dict[str, Any], step: float = 20.) -> tuple[list[float], dict[str, Any]]:
    """The square box (side from sizes_um) with lo <= cells <= hi that maximizes min over the objective's keys of
    count / weight (ties: the count closest to the middle of the range). A key is a cell type, or "gene:G" for the
    cells in which gene G is detected. Boxes overlapping a box of auto["exclude"] ([x0, y0, x1, y1] um; e.g. the region
    of an earlier run, for a second, disjoint region) are not considered."""
    lo, hi = auto["cells"]
    origin = xy.min(axis=0)
    bins = np.floor((xy - origin) / step).astype(int)
    shape = bins.max(axis=0) + 1

    def table(mask: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape + 1)
        np.add.at(grid, (bins[mask, 0] + 1, bins[mask, 1] + 1), 1.)
        return grid.cumsum(axis=0).cumsum(axis=1)

    total = table(np.ones(len(xy), dtype=bool))
    tables = {t: table(masks[t]) for t in auto["objective"]}
    best: tuple[float, float] = (-1., 0.)
    choice: dict[str, Any] = {}
    for size in auto["sizes_um"]:
        k = max(1, int(round(size / step)))
        if k > min(shape):
            continue

        def window(s: np.ndarray) -> np.ndarray:
            return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

        n = window(total)
        score = np.full(n.shape, np.inf)
        for t, weight in auto["objective"].items():
            score = np.minimum(score, window(tables[t]) / weight)
        score[(n < lo) | (n > hi)] = -1.
        corner = origin[None, None, :] + np.stack(np.meshgrid(np.arange(n.shape[0]), np.arange(n.shape[1]), indexing="ij"), axis=-1) * step
        for x0, y0, x1, y1 in auto.get("exclude", []):
            overlap = (corner[..., 0] < x1) & (corner[..., 0] + k * step > x0) & (corner[..., 1] < y1) & (corner[..., 1] + k * step > y0)
            score[overlap] = -1.
        i, j = np.unravel_index(np.argmax(score - np.abs(n - (lo + hi) / 2) * 1e-6), score.shape)
        key = (float(score[i, j]), -abs(float(n[i, j]) - (lo + hi) / 2))
        if key > best:
            best = key
            x0, y0 = origin + np.array([i, j]) * step
            choice = {"roi": [float(x0), float(y0), float(x0 + k * step), float(y0 + k * step)], "score": float(score[i, j]),
                      "cells": int(n[i, j]), "size_um": k * step}
    if best[0] < 0:
        raise ValueError("No box satisfies the cell range")
    return choice["roi"], choice


def receptor_tfs(cellcall: Path, collectri: Path, receptors: list[str], detection: dict[str, float],
                 max_tfs: int, min_detection: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tf_set = {row["source"] for row in csv.DictReader(open(collectri, encoding="utf-8"), delimiter="\t")}
    support: dict[str, dict[str, set[str]]] = {r: {} for r in receptors}
    with open(cellcall, encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            receptor = row["Receptor_Symbol"]
            maps = SIGNALLING & set(row["pathway_ID"].split(","))
            if receptor in support and maps:
                support[receptor].setdefault(row["TF_Symbol"], set()).update(maps)
    rows, report = [], {}
    for receptor in receptors:
        # Most supporting KEGG signalling maps first (e.g. TGFBR2 -> SMAD3, IFNGR1 -> STAT1), then most often detected.
        pool = sorted((tf for tf in support[receptor] if tf in tf_set and detection.get(tf, 0.) >= min_detection),
                      key=lambda tf: (-len(support[receptor][tf]), -detection[tf], tf))
        report[receptor] = {"candidates": len(support[receptor]),
                            "chosen": [(tf, len(support[receptor][tf]), round(detection[tf], 3)) for tf in pool[:max_tfs]]}
        rows += [{"source": receptor, "target": tf, "weight": 1.} for tf in pool[:max_tfs]]
    return rows, report


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run = Path(sys.argv[2])
    directory = Path(config["dataset"])
    counts, genes, barcodes, cells = load_xenium(directory)
    types = cluster_types(directory, barcodes, config["clusters"])
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    usable = totals >= config.get("min_counts", 20)
    if "roi" in config:
        roi = config["roi"]
    else:
        gi = {g: i for i, g in enumerate(genes.tolist())}
        masks = {key: (np.asarray((counts[:, gi[key[5:]]] > 0).todense()).ravel() if key.startswith("gene:") else types == key)[usable]
                 for key in config["auto"]["objective"]}
        roi, _ = auto_roi(xy[usable], masks, config["auto"])
    inside = usable & (xy[:, 0] >= roi[0]) & (xy[:, 0] < roi[2]) & (xy[:, 1] >= roi[1]) & (xy[:, 1] < roi[3])
    index = np.flatnonzero(inside)
    sub = counts[index].toarray().astype(np.int32)
    keep_genes = sub.sum(axis=0) > 0
    sub, genes_kept = sub[:, keep_genes], genes[keep_genes]
    detection = dict(zip(genes_kept.tolist(), (sub > 0).mean(axis=0).tolist()))
    min_detection = config.get("min_detection", .01)
    lr = [(l, r) for l, r in config["lr"] if detection.get(l, 0.) >= min_detection and detection.get(r, 0.) >= min_detection]
    receptors = sorted({r for _, r in lr})
    tf_rows, report = receptor_tfs(Path(config["cellcall"]), Path(config["collectri"]), receptors, detection,
                                   config.get("max_tfs", 2), config.get("min_detection_tf", .02))
    write_input(run, sub, genes_kept.astype(str), barcodes[index].astype(str), xy[index], lr,
                [(row["source"], row["target"]) for row in tf_rows], radius=float(config["radius_um"]))
    pd.DataFrame({"cell": barcodes[index].astype(str), "type": types[index], "x_um": xy[index, 0], "y_um": xy[index, 1]}
                 ).to_csv(run / "cells.csv", index=False)
    print(json.dumps({"roi": roi, "cells": int(len(index)), "lr": lr, "tfs": report}))


if __name__ == "__main__":
    main()
