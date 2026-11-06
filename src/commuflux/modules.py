"""Expression modules: deviance-selected genes, standard WGCNA modules and eigengenes, module levels.

1. genes: the 1000 genes with the largest full-library Poisson deviance (public gene ids break ties; every gene when
   the panel has fewer);
2. expression log1p CP10k of those genes; Pearson correlation; signed adjacency ((1 + r) / 2) ** beta with beta
   the smallest power whose scale-free fit (signed R^2) reaches .8, else 12;
3. signed TOM, dissimilarity 1 - TOM, average-linkage tree; hybrid Dynamic Tree Cut (deepSplit 2, minimum module
   size 30, tree-cut stage only; vendored port, see dynamicTreeCut/VENDORED.md); unassigned genes form grey;
4. eigengenes (first principal component of the standardized module genes, sign aligned with the module mean);
   modules whose eigengenes correlate above .75 are merged and eigengenes recomputed; grey is not a module;
5. eigengenes are shifted to be positive (CSCN module nodes); module levels for flow capacities are the mean
   nonlog CP10k of each module's genes.
Observed counts only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import xlogy
from scipy.stats import rankdata

import dynamicTreeCut.dynamicTreeCut as dynamic_tree_cut
from dynamicTreeCut import cutreeHybrid

DEVIANCE_GENES = 1000
POWERS = tuple(range(1, 11)) + tuple(range(12, 31, 2))
SCALE_FREE_R2, FALLBACK_POWER = .8, 12
MIN_MODULE, DEEP_SPLIT, MERGE_CUT = 30, 2, .25


def deviance_genes(counts: NDArray[Any], genes: list[str], number: int = DEVIANCE_GENES) -> NDArray[np.int64]:
    """Column indices (in source order) of the genes with the largest Poisson deviance against a full-library null."""
    raw = np.asarray(counts, dtype=np.float64)
    library = raw.sum(axis=1, keepdims=True)
    if (library <= 0).any() or not 1 <= number <= len(genes):
        raise ValueError("Poisson deviance needs positive libraries and a valid gene budget")
    expected = library * (raw.sum(axis=0, keepdims=True) / library.sum())
    ratio = np.divide(raw, expected, out=np.ones_like(raw), where=expected > 0)
    deviance = np.maximum(0., 2. * np.sum(xlogy(raw, ratio) - raw + expected, axis=0))
    order = np.lexsort((np.asarray(genes), -deviance))
    return np.asarray(np.sort(order[:number]), dtype=np.int64)


def cp10k(counts: NDArray[Any]) -> NDArray[np.float64]:
    counts = np.asarray(counts, dtype=float)
    return np.asarray(counts * (10000. / counts.sum(axis=1))[:, None], dtype=np.float64)


def scale_free_fit(connectivity: NDArray[np.float64], bins: int = 10) -> float:
    """WGCNA scaleFreeFitIndex: signed R^2 of log10 p(k) on log10 mean k over equal-width bins."""
    edges = np.linspace(connectivity.min(), connectivity.max(), bins + 1)
    index = np.clip(np.digitize(connectivity, edges[1:-1]), 0, bins - 1)
    centers, frequency = [], []
    for b in range(bins):
        members = connectivity[index == b]
        if len(members):
            centers.append(members.mean())
            frequency.append(len(members) / len(connectivity))
    x, y = np.log10(np.asarray(centers) + 1e-9), np.log10(np.asarray(frequency) + 1e-9)
    if len(x) < 3:
        return 0.
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r2 = 1. - float(residual @ residual) / float(((y - y.mean()) ** 2).sum())
    return float(-np.sign(slope) * r2)


def eigengenes(expression: NDArray[np.float64], labels: NDArray[np.int64], modules: list[int]) -> NDArray[np.float64]:
    columns = []
    for module in modules:
        block = expression[:, labels == module]
        z = (block - block.mean(axis=0)) / block.std(axis=0)
        u, s, _ = np.linalg.svd(z, full_matrices=False)
        me = u[:, 0] * s[0] / np.sqrt(len(z))
        if np.corrcoef(me, z.mean(axis=1))[0, 1] < 0:
            me = -me
        columns.append(me)
    return np.column_stack(columns)


class _IntegerRepeatNumpy:
    """numpy for the vendored port, with repeat counts cast to integers (the port passes np.ceil floats)."""

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    @staticmethod
    def repeat(values: Any, repeats: Any, axis: Any = None) -> Any:
        return np.repeat(values, np.asarray(repeats).astype(int), axis=axis)


def tree_cut(correlation: NDArray[np.float64], power: int) -> NDArray[np.int64]:
    """Module labels (0 = grey) of the signed TOM network at a soft power: average-linkage tree, cutreeHybrid."""
    adjacency = np.asarray(((1. + correlation) / 2.) ** power, dtype=np.float64)
    np.fill_diagonal(adjacency, 0.)
    connectivity = adjacency.sum(axis=1)
    tom = (adjacency @ adjacency + adjacency) / (np.minimum.outer(connectivity, connectivity) + 1. - adjacency)
    np.fill_diagonal(tom, 1.)
    dissimilarity = np.clip(1. - tom, 0., 1.)
    tree = linkage(squareform(dissimilarity, checks=False), method="average")
    # Call-time configuration of the vendored port (algorithm unchanged): buffers never grow, ordinal ranks are
    # integers, and repeat counts are integers.
    dynamic_tree_cut.chunkSize = max(4096, 2 * len(correlation))
    dynamic_tree_cut.rankdata = lambda values, method="average": np.asarray(rankdata(values, method=method)).astype(int)
    dynamic_tree_cut.np = _IntegerRepeatNumpy()  # type: ignore[assignment]
    cut = cutreeHybrid(tree, dissimilarity, minClusterSize=MIN_MODULE, deepSplit=DEEP_SPLIT, pamStage=False, verbose=0)
    return np.asarray(cut["labels"], dtype=int)


def wgcna(expression: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    """Eigengenes ordered by decreasing module size, per-gene module labels (0 = grey) and a summary.
    Soft power: the smallest with signed scale-free fit >= .8 (else 12). Where weak correlations make that power so
    high that the tree cut leaves fewer than two modules (nearly every gene grey), the power is lowered step by step
    to the largest one that gives at least two modules (tissues that cut at the chosen power are unchanged)."""
    correlation = np.asarray(np.corrcoef(expression, rowvar=False), dtype=np.float64)
    fits = {}
    for power in POWERS:
        adjacency = np.asarray(((1. + correlation) / 2.) ** power, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.)
        fits[power] = scale_free_fit(adjacency.sum(axis=1))
    passing = [p for p in POWERS if fits[p] >= SCALE_FREE_R2]
    chosen = beta = passing[0] if passing else FALLBACK_POWER
    labels = tree_cut(correlation, beta)
    for lower in [p for p in sorted(POWERS, reverse=True) if p < chosen]:
        if len(set(labels.tolist()) - {0}) >= 2:
            break
        beta, labels = lower, tree_cut(correlation, lower)
    modules = sorted(m for m in set(labels.tolist()) if m != 0)
    if len(modules) < 2:
        raise RuntimeError("WGCNA found fewer than two modules")
    merged_from = len(modules)
    me = eigengenes(expression, labels, modules)
    groups = fcluster(linkage(squareform(np.clip(1. - np.corrcoef(me, rowvar=False), 0., 2.), checks=False),
                              method="average"), t=MERGE_CUT, criterion="distance")
    relabel = {module: int(min(m for m, g in zip(modules, groups) if g == group)) for module, group in zip(modules, groups)}
    labels = np.array([relabel.get(int(label), 0) for label in labels])
    modules = sorted(set(relabel.values()))
    me = eigengenes(expression, labels, modules)
    order = sorted(range(len(modules)), key=lambda k: -int((labels == modules[k]).sum()))
    rule = "smallest signed R^2 >= .8" if passing else "fallback 12 (no power reached .8)"
    if beta != chosen:
        rule += f"; lowered from {chosen} (fewer than two modules there)"
    summary = {"soft_power": beta, "scale_free_fit": {str(p): fits[p] for p in POWERS}, "power_rule": rule,
               "modules_before_merge": merged_from, "modules": len(modules),
               "module_sizes": [int((labels == modules[k]).sum()) for k in order], "grey_genes": int((labels == 0).sum())}
    relabelled = np.zeros_like(labels)
    for rank, k in enumerate(order):
        relabelled[labels == modules[k]] = rank + 1
    return me[:, order], relabelled, summary


def build_representation(counts: NDArray[Any], genes: list[str], exclude: Any = ()) -> dict[str, Any]:
    """Deviance genes -> WGCNA. Returns shifted eigengenes (cells x modules), module ids, per-gene module index
    (-1 not selected, 0 grey, k >= 1 module me_{k-1}), kME-weighted module levels and the WGCNA summary.
    exclude: genes clustered as usual (they keep their module index) but left out of every eigengene and level."""
    selected = deviance_genes(counts, genes, min(DEVIANCE_GENES, len(genes)))
    expression = np.log1p(cp10k(counts))[:, selected]
    keep = expression.std(axis=0) > 0
    selected, expression = selected[keep], expression[:, keep]
    me, labels, summary = wgcna(expression)
    excluded = np.isin(np.asarray(genes, dtype=str)[selected], np.asarray(sorted(map(str, exclude)), dtype=str))
    if excluded.any():
        me = eigengenes(expression[:, ~excluded], labels[~excluded], list(range(1, me.shape[1] + 1)))
    summary["excluded_from_eigengenes"] = int(excluded.sum())
    me = me - me.min(axis=0) + 1e-3
    assignment = np.full(len(genes), -1)
    assignment[selected] = labels
    # Module level (flow capacity): members' CP10k averaged with weights max(kME, 0), kME = Pearson correlation of the
    # member's log1p CP10k with the module eigengene.
    normalized = cp10k(counts)[:, selected]
    columns, kme_summary = [], []
    for k in range(me.shape[1]):
        members = np.flatnonzero((labels == k + 1) & ~excluded)
        kme = np.array([np.corrcoef(expression[:, g], me[:, k])[0, 1] for g in members])
        weight = np.clip(kme, 0., None)
        columns.append(normalized[:, members] @ weight / weight.sum())
        kme_summary.append({"min": float(kme.min()), "median": float(np.median(kme)), "max": float(kme.max())})
    levels = np.column_stack(columns)
    # A cell can have no counts in a small module's genes: level 0, and the flow drops the node in that cell (as a
    # zero-count TF). Every module has a positive level somewhere.
    if not (np.isfinite(levels).all() and (levels >= 0).all() and (levels.max(axis=0) > 0).all()):
        raise RuntimeError("Invalid module levels")
    modules = [f"me_{k:02d}" for k in range(me.shape[1])]
    return {"eigengenes": me, "module_ids": modules, "assignment": assignment, "levels": levels,
            "summary": {**summary, "genes": int(len(selected)), "kME": kme_summary}}
