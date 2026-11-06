"""Cell-specific intracellular networks: the CSCN local tests and their B50 bootstrap support.

Variables of the key-node layout, one row per cell (observed counts only):
- ligand: log1p CP10k of each ligand of the LR list;
- receptor: log1p CP10k of the receptor gene;
- TF: log1p CP10k of each annotated receptor TF;
- module: WGCNA eigengenes (modules.py), the coarse-grained nodes.
For each query cell the author's CSCN package (pinned commit d579c03) builds the local PC skeleton and orientation
with three adopted tests: zero-order dependence is a local Spearman test inside the query's X window and inside its
Y window (each at alpha / 2); conditional independence is a Spearman partial correlation inside the query's whole X
window and Y window with Z regressed out (each at alpha / 2; a dependence there is kept) and else inside the query's
Z-window slice; a pair separated only through an information-equivalent TF and module keeps the stronger edge;
collider proposals are checked against the separator plus the centre. The tests are one-sided (CSCN's ci_alternative
"greater"): only a positive dependence counts, so the network holds activation-like arcs only. Pairs with a prior
(the annotated receptor -> TF links) are tested at a raised level (pair_alpha, as CSCN's soft
skeleton prior).
Background knowledge on separating sets (admissible_separator): role tiers receptor -> TF -> module -> ligand (a
separating set holds no variable of a later tier than the earlier endpoint, CSCN's temporal ordering) and
information-equivalent proxies (a TF and the module holding its gene do not separate each other from a third
variable). Native PDAG arrows are kept; role_relations orients the receptor/TF, receptor/module, TF/module,
TF/ligand (a TF directly inducing a ligand) and module/ligand adjacencies by tier, an arc that PC orients
against the tiers kept in the tier direction (role override), and entry modules (equivalent to a TF) take the TF's
role: their arcs to other modules point away from them. B50: 50 spatially stratified resamples of the other cells
with the query appended once; edge support omega = direction-compatible resamples / 50.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
import json
import multiprocessing as mp
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

CSCN_COMMIT = "d579c03d77a6435c570957188de356b72e369f00"
SIGNIFICANCE_LEVEL = .01  # CSCN's significance level of pairs without prior
# ci_alternative two-sided: the engine of arc_signs; the inference engines take engine_parameters (one-sided)
CSCN_PARAMETERS = {
    "sigmoid_score": 0.2, "pc_var": "stable", "significance_level": SIGNIFICANCE_LEVEL,
    "max_cond_vars": 20, "query_engine": "hybrid", "ci_alternative": "two-sided",
    "collider_conflict_policy": "prioritize_existing", "max_bits_cache_entries": 8192,
    "tf_prior_mode": "none", "atac_ci_mode": "none",
}
PRIOR_ALPHA = .2  # CSCN's significance level for prior-supported pairs at strength 1 (external_prior_alpha, tf_skeleton_alpha)
BOOTSTRAP_SEED = 910060
REPETITIONS = 50
TIERS = {"receptor": 0, "tf": 1, "module": 2, "ligand": 3}  # receptors -> TFs -> modules -> ligands


def column(role: str, feature: str) -> str:
    return json.dumps([role, feature], separators=(",", ":"))


def local_table(counts: np.ndarray, genes: list[str], cells: list[str], resources: dict[str, Any],
                eigengenes: np.ndarray, module_ids: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The CSCN input table and variable roles of the key-node layout (column order: ligands, receptors, TFs, modules;
    see the module docstring)."""
    gi = {gene: i for i, gene in enumerate(genes)}
    counts = np.asarray(counts)
    totals = counts.sum(axis=1)
    divided = np.log1p(counts / totals[:, None] * 10000)  # functional-measurement normalization, (x / total) * 1e4
    observed = np.log1p(counts * (10000. / totals)[:, None])
    receptor_tfs: dict[str, list[str]] = {}
    for row in resources["receptor_tf"]:
        receptor_tfs.setdefault(str(row["source"]), []).append(str(row["target"]))
    data: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    for ligand in sorted(map(str, resources["roles"]["ligands"])):
        data[column("ligand", ligand)], roles[column("ligand", ligand)] = divided[:, gi[ligand]], "ligand"
    receptors = sorted(map(str, resources["roles"]["receptors"]))
    for receptor in receptors:
        own = observed[:, gi[receptor]]
        data[column("receptor", receptor)], roles[column("receptor", receptor)] = own, "receptor"
    for tf in dict.fromkeys(tf for receptor in receptors for tf in receptor_tfs[receptor]):
        data[column("tf", tf)], roles[column("tf", tf)] = observed[:, gi[tf]], "tf"
    for k, module in enumerate(module_ids):
        data[column("module", module)], roles[column("module", module)] = eigengenes[:, k], "module"
    keyed = pd.DataFrame(data, index=pd.Index(cells, name="cell"))
    if not np.isfinite(keyed.to_numpy()).all():
        raise ValueError("Nonfinite CSCN input")
    return keyed, roles


def equivalent_proxies(genes: list[str], resources: dict[str, Any], assignment: np.ndarray,
                       module_ids: list[str]) -> list[list[str]]:
    """Pairs [TF variable, module variable] measuring the same regulator: the module that holds the TF's gene.
    assignment: per-gene module index as in representation.build_representation (k >= 1 is module_ids[k - 1])."""
    gi = {gene: i for i, gene in enumerate(genes)}
    receptor_tfs: dict[str, list[str]] = {}
    for row in resources["receptor_tf"]:
        receptor_tfs.setdefault(str(row["source"]), []).append(str(row["target"]))
    variables = {column("tf", tf): tf for tfs in receptor_tfs.values() for tf in tfs}
    pairs = []
    for name, tf in sorted(variables.items()):
        label = int(assignment[gi[tf]])
        if label >= 1:
            pairs.append([name, column("module", module_ids[label - 1])])
    return pairs


def background_knowledge(genes: list[str], resources: dict[str, Any], assignment: np.ndarray,
                         module_ids: list[str]) -> dict[str, Any]:
    """The background knowledge of the adopted tests besides the role tiers: the symmetric equivalence of each TF and
    the module holding its gene (equivalent_proxies; separator barring and entry modules). infer_query also reads its
    "prior" rows (receptor_tf_prior), which infer.prepare adds."""
    pairs: dict[str, set[str]] = {}
    for a, b in equivalent_proxies(genes, resources, assignment, module_ids):
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)
    return {"equivalent": pairs}


def receptor_tf_prior(resources: dict[str, Any], roles: dict[str, str]) -> list[list[Any]]:
    """Prior strengths [receptor, TF, 1] of the annotated receptor -> TF links (key-node layout)."""
    return [[column("receptor", str(row["source"])), column("tf", str(row["target"])), 1.]
            for row in resources["receptor_tf"]
            if column("receptor", str(row["source"])) in roles and column("tf", str(row["target"])) in roles]


def admissible_separator(separator: Any, left: str, right: str, roles: dict[str, str], knowledge: dict[str, Any]) -> bool:
    """Whether a candidate separating set of left and right respects the background knowledge. Tiers: a separator
    holds no variable of a tier later than the earlier endpoint (CSCN's temporal ordering: a variable between the
    endpoints, e.g. a module between a TF and a ligand, cannot separate them). Equivalence: no variable equivalent to
    an endpoint (knowledge "equivalent")."""
    members = [str(member) for member in separator]
    limit = min(TIERS[roles[left]], TIERS[roles[right]])
    if any(TIERS[roles[member]] > limit for member in members):
        return False
    equivalent = knowledge.get("equivalent", {})
    barred = set(equivalent.get(left, ())) | set(equivalent.get(right, ()))
    return not barred.intersection(members)


def spatial_partition(coordinates: np.ndarray, bins_per_axis: int = 4) -> np.ndarray:
    """Fixed equal-width 4 x 4 grid of the coordinates (bootstrap strata)."""
    spans = np.ptp(coordinates, axis=0)
    scaled = (coordinates - coordinates.min(axis=0)) / np.where(spans > 0, spans, 1.)
    bins = np.minimum((scaled * bins_per_axis).astype(np.int64), bins_per_axis - 1)
    return np.asarray(bins[:, 0] * bins_per_axis + bins[:, 1], dtype=np.int64)


def bootstrap_sample(table: pd.DataFrame, groups: np.ndarray, query: int, iteration: int) -> tuple[pd.DataFrame, int]:
    """Resample the other cells within their strata, then append the query once (it is the last row)."""
    eligible = np.flatnonzero(np.arange(len(table)) != query)
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, query, iteration]))
    parts = [rng.choice(members, size=len(members), replace=True)
             for group in sorted(set(groups[eligible].tolist()))
             for members in [eligible[groups[eligible] == group]]]
    sampled = np.concatenate(parts).astype(int).tolist()
    return table.iloc[sampled + [query]].reset_index(drop=True), len(sampled)


def collider_exclusions(engine: Any, skeleton: nx.Graph, separating_sets: Any, query: int,
                        temporal_ordering: Any = None) -> dict[frozenset[Any], tuple[Any, ...]]:
    """Collider centre check: if the separator plus the centre still gives an informative independence decision,
    leave the collider proposal undecided (orientation exclusions only; the skeleton is unchanged)."""
    exclusions = {pair: set(values) for pair, values in separating_sets.items()}
    ordering = dict(temporal_ordering or {})
    for left, right in combinations(engine._sorted_nodes(skeleton.nodes()), 2):
        if skeleton.has_edge(left, right):
            continue
        pair = frozenset((left, right))
        original = set(separating_sets.get(pair, ()))
        shared = set(skeleton.neighbors(left)) & set(skeleton.neighbors(right))
        for center in engine._sorted_nodes(shared):
            if center in original or (ordering and (ordering[center] < ordering[left] or ordering[center] < ordering[right])):
                continue
            condition = tuple(engine._sorted_nodes(original | {center}))
            if len(condition) > engine.max_cond_vars:
                continue
            independent = bool(engine.conditional_independence_test(left, right, condition, data=engine.df,
                independencies=None, key_cell_idx=query, significance_level=engine.significance_level,
                sigmoid_score=engine.sigmoid_score))
            ids = {engine._coerce_gene_id(name) for name in condition}
            x, y = engine._coerce_gene_id(left), engine._coerce_gene_id(right)
            nz, nxz, nyz, nxyz = [int(engine.get_conditional_counts(names, query, engine.sigmoid_score))
                                  for names in (ids, ids | {x}, ids | {y}, ids | {x, y})]
            if not (0 <= nxyz <= min(nxz, nyz) <= max(nxz, nyz) <= nz and nz - nxz - nyz + nxyz >= 0):
                raise ValueError("Invalid native counts in collider qualification")
            if independent and nz > 1 and 0 < nxz < nz and 0 < nyz < nz:
                exclusions.setdefault(pair, set(original)).add(center)
    return {pair: tuple(engine._sorted_nodes(values)) for pair, values in exclusions.items()}


def adopted_engine(captured: dict[str, Any]) -> Any:
    from bitarray import bitarray
    from cscn import CSCN
    from cscn.core import CI_EPSILON

    class AdoptedCSCN(CSCN):
        def _build_deterministic_pc_skeleton(self, *args: Any, **kwargs: Any) -> Any:
            graph, separators = super()._build_deterministic_pc_skeleton(*args, **kwargs)
            return self.keep_information_equivalent(graph, dict(separators))

        def keep_information_equivalent(self, graph: Any, separators: dict[Any, Any]) -> tuple[Any, dict[Any, Any]]:
            """Mutual separation X1-Y | {X2,..} and X2-Y | {X1,..} of a receptor proxy and a module that are themselves
            adjacent presuppose each other; restore the stronger of the two edges."""
            query = captured["query"]
            removed = {pair: set(members) for pair, members in separators.items() if not graph.has_edge(*tuple(pair))}
            restore: set[frozenset[Any]] = set()
            for pair in sorted(removed, key=lambda item: sorted(map(str, item))):
                a, b = sorted(pair, key=str)
                for x1, y in ((a, b), (b, a)):
                    for x2 in sorted(removed[pair], key=str):
                        other = frozenset((x2, y))
                        if {captured["roles"][x1], captured["roles"][x2]} not in ({"receptor", "module"}, {"tf", "module"}):
                            continue
                        if other in removed and x1 in removed[other] and graph.has_edge(x1, x2):
                            strength = {candidate: self._pair_unconditioned_p_value(
                                self._coerce_gene_id(head), self._coerce_gene_id(y), query)
                                for candidate, head in ((pair, x1), (other, x2))}
                            restore.add(min(strength, key=lambda item: (strength[item], sorted(map(str, item)))))
            for pair in restore:
                graph.add_edge(*tuple(pair))
                separators.pop(pair, None)
            return graph, separators

        def _iter_potential_sepsets(self, left: Any, right: Any, neighbors_by_node: Any, temporal_ordering: Any,
                                    lim_neighbors: int) -> Any:
            knowledge = captured["knowledge"]
            for separator in super()._iter_potential_sepsets(left, right, neighbors_by_node, temporal_ordering, lim_neighbors):
                if admissible_separator(separator, str(left), str(right), captured["roles"], knowledge):
                    yield separator

        def _orient_unshielded_colliders(self, skeleton: Any, separating_sets: Any, temporal_ordering: Any = None) -> Any:
            exclusions = collider_exclusions(self, skeleton, separating_sets, captured["query"], temporal_ordering)
            return super()._orient_unshielded_colliders(skeleton, exclusions, temporal_ordering)

        def _build_dag_representation(self, pdag: Any) -> Any:
            try:
                return super()._build_dag_representation(pdag)
            except ValueError as error:
                if str(error) != "Cannot build DAG representation: fixed directed edges contain a cycle":
                    raise
                unavailable = nx.DiGraph()
                unavailable.add_nodes_from(pdag.nodes)
                unavailable.graph.update(available=False, unavailable_reason="native_directed_cycle")
                return unavailable

        def conditional_independence_test(self, X: Any, Y: Any, Z: Any, data: Any, independencies: Any,
                                          key_cell_idx: int = 0, significance_level: Any = None,
                                          sigmoid_score: Any = None) -> bool:
            x, y = sorted((self._coerce_gene_id(X), self._coerce_gene_id(Y)))
            z = {self._coerce_gene_id(gene) for gene in Z} if Z else set()
            cache = captured.setdefault("ci_cache", {})
            key = (int(key_cell_idx), x, y, tuple(sorted(z)))
            if key not in cache:
                alpha = self.pair_alpha(x, y)
                cache[key] = (not self.local_marginal_dependent(x, y, int(key_cell_idx), alpha) if not z
                              else self.partial_independent(x, y, z, int(key_cell_idx), alpha))
            return bool(cache[key])

        def pair_alpha(self, x: int, y: int) -> float:
            """Significance level of a pair: raised towards PRIOR_ALPHA by its prior strength (knowledge "prior"), as
            CSCN's soft skeleton prior (tf_prior): a prior pair needs a larger p-value to be separated. The configured
            level otherwise."""
            base = float(self.significance_level)
            strength = captured.get("prior", {}).get(frozenset((self._node_name_for_id(x), self._node_name_for_id(y))), 0.)
            level = PRIOR_ALPHA
            return base + float(strength) * (level - base) if strength > 0 else base

        def _data(self) -> np.ndarray:
            if "data" not in captured:
                captured["data"] = self.df.to_numpy(dtype=float)
            return np.asarray(captured["data"])

        def window_statistics(self, x: int, y: int, key: int) -> Any:
            """Fisher z of the Spearman of X and Y inside the query's X window, then inside its Y window (lazily;
            windows of fewer than 8 cells or with a constant variable are skipped)."""
            data = self._data()
            for gene in (x, y):
                members = np.flatnonzero(np.array(self._get_count_bits(gene, key, self.sigmoid_score).tolist(), dtype=bool))
                if len(members) < 8:
                    continue
                rx, ry = rankdata(data[members, x]), rankdata(data[members, y])
                if float(np.std(rx)) <= CI_EPSILON or float(np.std(ry)) <= CI_EPSILON:
                    continue
                r = float(np.clip(np.corrcoef(rx, ry)[0, 1], -.999999, .999999))
                yield float(np.arctanh(r) * np.sqrt(len(members) - 3))

        def p_value(self, z: float) -> float:
            """One-sided p-value of a Fisher z (CSCN's ci_alternative "greater", the upper tail: a negative dependence is
            no evidence against independence)."""
            return float(norm.sf(z))

        def local_marginal_dependent(self, x: int, y: int, key: int, alpha: float) -> bool:
            """Spearman of X and Y inside the query's X window and inside its Y window, each at alpha / 2."""
            return any(self.p_value(z) < alpha / 2. for z in self.window_statistics(x, y, key))

        def local_sign(self, x: int, y: int, key: int) -> int:
            """Sign of the zero-order local Spearman of X and Y in the more significant of the query's two windows (0
            when neither can be tested): the direction of effect that the two-sided tests do not keep."""
            statistics = list(self.window_statistics(x, y, key))
            return int(np.sign(max(statistics, key=abs))) if statistics else 0

        def partial_p_value(self, x: int, y: int, z: set[int], members: np.ndarray) -> float | None:
            """P-value (p_value) of the Spearman partial correlation of X and Y given Z among the members (None when a
            residual is constant)."""
            data = self._data()[members]
            ranks = {g: rankdata(data[:, g]) for g in {x, y} | z}
            design = np.column_stack([np.ones(len(members))] + [ranks[g] for g in sorted(z)])
            residuals = [ranks[g] - design @ np.linalg.lstsq(design, ranks[g], rcond=None)[0] for g in (x, y)]
            if min(float(np.std(r)) for r in residuals) <= CI_EPSILON:
                return None
            r = float(np.clip(np.corrcoef(residuals[0], residuals[1])[0, 1], -.999999, .999999))
            return self.p_value(float(np.arctanh(r) * np.sqrt(len(members) - len(z) - 3)))

        def partial_independent(self, x: int, y: int, z: set[int], key: int, alpha: float) -> bool:
            """Spearman partial correlation of X and Y given Z inside the query's whole X window and Y window with Z
            regressed out (each at alpha / 2: a dependence there is kept), then among the cells of the query's Z-window
            slice."""
            bits = bitarray(self.n_cells)
            bits.setall(True)
            for gene in z:
                bits &= self._get_count_bits(gene, key, self.sigmoid_score)
            for gene in (x, y):  # whole X and Y windows, Z regressed out
                window = np.flatnonzero(np.array(self._get_count_bits(gene, key, self.sigmoid_score).tolist(), dtype=bool))
                if len(window) >= len(z) + 8:
                    p = self.partial_p_value(x, y, z, window)
                    if p is not None and p <= alpha / 2.:
                        return False
            members = np.flatnonzero(np.array(bits.tolist(), dtype=bool))
            if len(members) < len(z) + 8:
                return False  # too few cells is not evidence of independence
            p = self.partial_p_value(x, y, z, members)
            return p is not None and p > alpha

    return AdoptedCSCN


def role_relations(pdag: nx.DiGraph, roles: dict[str, str], entry: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Keep native PDAG arrows; orient the receptor/TF, receptor/module, TF/module, TF/ligand (a TF directly inducing a
    ligand) and module/ligand adjacencies by tier (receptor -> TF -> module -> ligand); other pairs of
    different roles are left out. A native arrow against the tiers is kept as an arc in the tier direction (role
    override; no conflicts are reported). Entry modules (equivalent to a TF) are oriented like the TF towards the
    other modules."""
    supported = []
    pairs = sorted({tuple(sorted((str(a), str(b)))) for a, b in pdag.edges if a != b})
    for a, b in pairs:
        unresolved = pdag.has_edge(a, b) and pdag.has_edge(b, a)
        tail, head = (a, b) if pdag.has_edge(a, b) else (b, a)
        kinds = {roles[a], roles[b]}
        if kinds != {"module"}:
            if kinds not in ({"receptor", "tf"}, {"receptor", "module"}, {"tf", "module"}, {"module", "ligand"},
                             {"tf", "ligand"}):
                continue
            expected = [a, b] if TIERS[roles[a]] < TIERS[roles[b]] else [b, a]
        elif (a in entry) != (b in entry):
            expected = [a, b] if a in entry else [b, a]
        else:
            supported.append([a, b, "unresolved"] if unresolved else [tail, head, "PC"])
            continue
        if not unresolved and [tail, head] != expected:
            supported.append([*expected, "role_override"])
        else:
            supported.append([*expected, "role_prior" if unresolved else "PC"])
    return {"supported": sorted(supported), "conflicts": []}


def engine_parameters() -> dict[str, Any]:
    """CSCN's parameters of the inference engines: one-sided tests (ci_alternative "greater", a positive dependence
    only) at SIGNIFICANCE_LEVEL for the pairs without prior. One-sided tests at half the level decide a positive
    dependence exactly as the two-sided tests do and count a negative one as none."""
    return {**CSCN_PARAMETERS, "ci_alternative": "greater"}


def infer_query(table: pd.DataFrame, query: int, roles: dict[str, str], knowledge: dict[str, Any]) -> dict[str, Any]:
    """One cell-specific network with the adopted tests; supported role-compatible arcs (role_relations).
    knowledge: background_knowledge(...) with the "prior" rows [variable, variable, strength] (receptor_tf_prior)
    that raise the pairs' significance level (pair_alpha); see engine_parameters for the tests."""
    captured: dict[str, Any] = {"query": query, "roles": roles, "knowledge": knowledge,
                                "prior": {frozenset((str(a), str(b))): float(w) for a, b, w in knowledge["prior"]}}
    engine = adopted_engine(captured)(**engine_parameters(), variable_names=list(table.columns), enable_query_stats=True)
    messages = StringIO()
    with redirect_stdout(messages):
        engine.run_core(table.to_numpy(dtype=float), use_nmf=False)
        result = engine.infer_cell_pc(query)
    if "ICT ERROR:" in messages.getvalue():
        raise RuntimeError("CSCN reported a CI error")
    return role_relations(result.pdag, roles, entry=entry_modules(knowledge, roles))


def arc_signs(table: pd.DataFrame, networks: list[dict[str, Any]], roles: dict[str, str]) -> list[dict[str, Any]]:
    """Adds "sign" to every arc of the cell networks (support and Q additions, in place): the sign of the zero-order
    local Spearman of its two variables in the more significant of the query's two windows (AdoptedCSCN.local_sign,
    the windows of the zero-order test on the full table; a two-sided engine, CSCN_PARAMETERS). Stored with the
    networks as a record of each arc's direction of effect."""
    engine = adopted_engine({"query": 0, "roles": roles, "knowledge": None, "prior": {}})(
        **CSCN_PARAMETERS, variable_names=list(table.columns), enable_query_stats=True)
    with redirect_stdout(StringIO()):
        engine.run_core(table.to_numpy(dtype=float), use_nmf=False)
    for network in networks:
        for edge in network["support"] + network["additions"]:
            edge["sign"] = engine.local_sign(engine._coerce_gene_id(edge["tail"]), engine._coerce_gene_id(edge["head"]),
                                             int(network["query"]))
    return networks


def entry_modules(knowledge: dict[str, Any], roles: dict[str, str]) -> frozenset[str]:
    """Modules equivalent to a TF variable (knowledge "equivalent"): oriented like the TF (role_relations)."""
    equivalent = knowledge.get("equivalent", {})
    return frozenset(name for name, others in equivalent.items()
                     if roles.get(name) == "module" and any(roles.get(other) in ("receptor", "tf") for other in others))


def edge_support(original: dict[str, Any], replicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Support of every arc of the original graph (both arrows of an unresolved module pair)."""
    rows = []
    for a, b, basis in original["supported"]:
        for tail, head in ([(a, b), (b, a)] if basis == "unresolved" else [(a, b)]):
            pair = {tail, head}
            positive = conflicts = 0
            for record in replicates:
                matching = [edge for edge in record["supported"] if set(edge[:2]) == pair]
                opposite = any(edge[2] != "unresolved" and edge[:2] != [tail, head] for edge in matching)
                conflict = opposite or any(set(item["pair"]) == pair for item in record["conflicts"])
                conflicts += int(conflict)
                positive += int(bool(matching) and not conflict)
            rows.append({"tail": tail, "head": head, "basis": basis, "positive": positive, "planned": REPETITIONS,
                         "omega": positive / REPETITIONS, "direction_conflicts": conflicts})
    return rows


_worker: dict[str, Any] = {}


def _initialize(table: pd.DataFrame, groups: np.ndarray, roles: dict[str, str], knowledge: dict[str, Any]) -> None:
    _worker.update(table=table, groups=groups, roles=roles, knowledge=knowledge)


def _query(query: int) -> dict[str, Any]:
    table, groups, roles, knowledge = _worker["table"], _worker["groups"], _worker["roles"], _worker["knowledge"]
    original = infer_query(table, query, roles, knowledge)
    replicates = []
    for iteration in range(REPETITIONS):
        sample, index = bootstrap_sample(table, groups, query, iteration)
        replicates.append(infer_query(sample, index, roles, knowledge))
    return {"query": query, "cell": str(table.index[query]), "original": original, "replicates": replicates,
            "support": edge_support(original, replicates)}


def bootstrap_networks(table: pd.DataFrame, roles: dict[str, str], groups: np.ndarray, workers: int,
                       progress: Any = None, knowledge: dict[str, Any] | None = None,
                       queries: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Original and REPETITIONS (50) resampled networks for every query cell, or for the given query rows (spawned worker
    processes). A query's resamples depend only on the table, the strata and the query, so parts of the cells can run
    as separate jobs. knowledge: the background knowledge of infer.prepare (required)."""
    if knowledge is None:
        raise ValueError("bootstrap_networks needs the background knowledge (infer.prepare)")
    rows: list[dict[str, Any]] = []
    with mp.get_context("spawn").Pool(workers, initializer=_initialize, initargs=(table, groups, roles, knowledge)) as pool:
        for row in pool.imap_unordered(_query, range(len(table)) if queries is None else queries, chunksize=1):
            rows.append(row)
            if progress is not None and len(rows) % 50 == 0:
                progress(len(rows))
    return sorted(rows, key=lambda row: row["query"])
