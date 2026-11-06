"""Whole-network entropy-regularized communication flow: the dual block-coordinate solver.

Entropy acts on every transport arc and on every pool residual (free ligand m, free receptor n,
unexplained module activity u). Autonomous ligand starts (cost gamma) stay linear; each receptor's silent outlet and
each terminal module's termination are entropic exits, which gives every intracellular step a mass-action form and
makes the problem invariant to a common scale of all pools. The optimum has a product form:
LR flow = affinity x free ligand x free receptor x sender/receiver factors, and every conservation
node balances upstream push and downstream pull.

The solver maximizes the smooth concave dual by exact block coordinate ascent (cyclic Bregman
projections). Every block update has a closed form: pool multipliers rescale a pool and its arcs
(Sinkhorn steps); node potentials solve push-pull balances, clipped by the autonomous-cost cap
at ligand ports. No conic solver is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class Arcs:
    """Arcs of one type: tail index, head index and linear cost."""
    tail: IntArray
    head: IntArray
    cost: FloatArray

    @staticmethod
    def empty() -> Arcs:
        return Arcs(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))

    def __len__(self) -> int:
        return len(self.tail)


@dataclass(frozen=True)
class NetworkEntropyProblem:
    """Ligand ports L, receptor ports R, modules M and four arc types.

    lr: ligand port -> receptor port (extracellular, temperature theta)
    rm: receptor port -> module; mm: module -> module; ml: module -> ligand port (relay)
    (intracellular, temperature eta). Each receptor also has a silent outlet with cost outlet_cost, and each terminal
    module an exit with cost termination_cost.
    relay_sink (default): a relay arc ends in a sink of its ligand port (an entropic exit at potential 0) instead of
    feeding the port, so a ligand port sends only its autonomous start (its own budget) and no flow can circle between
    cells without a source; with relay_sink False a port sends its autonomous start plus its relay inflow. Module arcs
    are acyclic in CommuFlux; on a graph with cycles the modules are updated by colour classes instead of topological
    levels (_module_levels).
    """
    ligand_budget: FloatArray
    receptor_budget: FloatArray
    module_capacity: FloatArray
    terminal: NDArray[np.bool_]
    lr: Arcs
    rm: Arcs = field(default_factory=Arcs.empty)
    mm: Arcs = field(default_factory=Arcs.empty)
    ml: Arcs = field(default_factory=Arcs.empty)
    autonomous_cost: FloatArray | float = 1.
    outlet_cost: FloatArray | float = 1.
    unmatched_ligand_cost: float = 0.
    unmatched_receptor_cost: float = 0.
    unmatched_module_cost: float = 0.
    theta: float = 1.
    eta: float = 1.
    termination_cost: float = 0.
    relay_sink: bool = True

    def __post_init__(self) -> None:
        sizes = (len(self.ligand_budget), len(self.receptor_budget), len(self.module_capacity))
        if len(self.terminal) != sizes[2]:
            raise ValueError("Terminal flags must match the modules")
        for array in (self.ligand_budget, self.receptor_budget):
            if not np.isfinite(array).all() or (array <= 0).any():
                raise ValueError("Pools must be finite and positive; remove empty pools before solving")
        # An infinite module capacity is an uncapacitated conservation node (no residual pool, no capacity price).
        if np.isnan(self.module_capacity).any() or (self.module_capacity <= 0).any():
            raise ValueError("Module capacities must be positive (infinite for an uncapacitated node)")
        bounds = (("lr", self.lr, sizes[0], sizes[1]), ("rm", self.rm, sizes[1], sizes[2]),
                  ("mm", self.mm, sizes[2], sizes[2]), ("ml", self.ml, sizes[2], sizes[0]))
        for name, arcs, tails, heads in bounds:
            if not (len(arcs.tail) == len(arcs.head) == len(arcs.cost)):
                raise ValueError(f"{name} arc arrays differ in length")
            if len(arcs) and ((arcs.tail < 0).any() or (arcs.tail >= tails).any()
                              or (arcs.head < 0).any() or (arcs.head >= heads).any()):
                raise ValueError(f"{name} arcs point outside their node sets")
            if not np.isfinite(arcs.cost).all():
                raise ValueError(f"{name} arc costs must be finite")
        if len(self.mm) and (self.mm.tail == self.mm.head).any():
            raise ValueError("Module self-loops are not allowed")
        # Dead ends make potentials infinite; callers prune them (their flows are exactly zero).
        if (np.bincount(self.lr.head, minlength=sizes[1]) == 0).any():
            raise ValueError("Every receptor port needs at least one LR arc")
        relay_ports = np.bincount(self.ml.head, minlength=sizes[0]) > 0
        if not self.relay_sink and (relay_ports & (np.bincount(self.lr.tail, minlength=sizes[0]) == 0)).any():
            raise ValueError("A relay-fed ligand port needs at least one LR arc")
        inflow = np.bincount(self.rm.head, minlength=sizes[2]) + np.bincount(self.mm.head, minlength=sizes[2])
        outflow = np.bincount(self.mm.tail, minlength=sizes[2]) + np.bincount(self.ml.tail, minlength=sizes[2])
        if (inflow == 0).any() or ((~self.terminal) & (outflow == 0)).any():
            raise ValueError("Every module needs an inflow and every nonterminal module an outflow")
        if not (self.theta > 0 and self.eta > 0):
            raise ValueError("Temperatures must be positive")


@dataclass(frozen=True)
class NetworkEntropySolution:
    lr: FloatArray
    rm: FloatArray
    mm: FloatArray
    ml: FloatArray
    outlet: FloatArray
    autonomous: FloatArray
    termination: FloatArray
    free_ligand: FloatArray
    free_receptor: FloatArray
    free_module: FloatArray
    potentials: dict[str, FloatArray]
    certificate: dict[str, Any]


def _group_logsumexp(values: FloatArray, groups: IntArray, size: int) -> FloatArray:
    """log(sum(exp(values))) per group; -inf for empty groups."""
    result = np.full(size, -np.inf)
    if len(values) == 0:
        return result
    peak = np.full(size, -np.inf)
    np.maximum.at(peak, groups, values)
    safe = np.where(np.isfinite(peak), peak, 0.)
    total = np.bincount(groups, weights=np.exp(values - safe[groups]), minlength=size)
    with np.errstate(divide="ignore"):
        result = np.where(total > 0, np.log(total) + safe, -np.inf)
    return result


def _module_levels(problem: NetworkEntropyProblem) -> list[IntArray]:
    """Topological levels of the module subgraph; modules in one level share no mm arc."""
    count = len(problem.module_capacity)
    indegree = np.bincount(problem.mm.head, minlength=count) if len(problem.mm) else np.zeros(count, np.int64)
    children: list[list[int]] = [[] for _ in range(count)]
    for tail, head in zip(problem.mm.tail.tolist(), problem.mm.head.tolist()):
        children[tail].append(head)
    remaining = indegree.copy()
    frontier = [int(i) for i in np.flatnonzero(remaining == 0)]
    levels: list[IntArray] = []
    seen = 0
    while frontier:
        levels.append(np.asarray(frontier, dtype=np.int64))
        seen += len(frontier)
        following: list[int] = []
        for node in frontier:
            for child in children[node]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    following.append(child)
        frontier = following
    if seen != count:
        # Module arcs with cycles: greedy colouring of the undirected module graph; modules of one colour share no
        # arc, so each colour class is updated exactly as one block (Gauss-Seidel over colours).
        neighbours: list[set[int]] = [set() for _ in range(count)]
        for tail, head in zip(problem.mm.tail.tolist(), problem.mm.head.tolist()):
            neighbours[tail].add(head)
            neighbours[head].add(tail)
        colour = np.full(count, -1, dtype=np.int64)
        for node in sorted(range(count), key=lambda k: -len(neighbours[k])):
            used = {int(colour[other]) for other in neighbours[node] if colour[other] >= 0}
            colour[node] = next(c for c in range(count + 1) if c not in used)
        return [np.flatnonzero(colour == c) for c in range(int(colour.max()) + 1)]
    return levels


def solve_network_entropy(problem: NetworkEntropyProblem, *, tolerance: float = 1e-10,
                          max_sweeps: int = 100000, check_every: int = 10) -> NetworkEntropySolution:
    """Exact block coordinate ascent on the dual; returns flows and a residual/gap certificate."""
    p = problem
    theta, eta = float(p.theta), float(p.eta)
    n_l, n_r, n_m = len(p.ligand_budget), len(p.receptor_budget), len(p.module_capacity)
    gamma = np.broadcast_to(np.asarray(p.autonomous_cost, dtype=float), (n_l,)).copy()
    c0 = np.broadcast_to(np.asarray(p.outlet_cost, dtype=float), (n_r,)).copy()
    alpha, beta, kappa = np.zeros(n_l), np.zeros(n_r), np.zeros(n_m)
    phi_l, phi_r, phi_m = gamma.copy(), np.zeros(n_r), np.zeros(n_m)
    log_a, log_b, log_d = np.log(p.ligand_budget), np.log(p.receptor_budget), np.log(p.module_capacity)
    finite_capacity = np.isfinite(p.module_capacity)
    levels = _module_levels(p)

    def log_flows() -> dict[str, FloatArray]:
        return {
            "lr": -(p.lr.cost + phi_l[p.lr.tail] - phi_r[p.lr.head] + alpha[p.lr.tail] + beta[p.lr.head]) / theta,
            "rm": -(p.rm.cost + phi_r[p.rm.tail] - phi_m[p.rm.head] + kappa[p.rm.head]) / eta,
            "mm": -(p.mm.cost + phi_m[p.mm.tail] - phi_m[p.mm.head] + kappa[p.mm.head]) / eta,
            "ml": -(p.ml.cost + phi_m[p.ml.tail] - (0. if p.relay_sink else phi_l[p.ml.head])) / eta,
            "outlet": -(c0 + phi_r) / eta,
            "t": np.where(p.terminal, -(p.termination_cost + phi_m) / eta, -np.inf),
            "m": -(p.unmatched_ligand_cost + alpha) / theta,
            "n": -(p.unmatched_receptor_cost + beta) / theta,
            "u": np.where(finite_capacity, -(p.unmatched_module_cost + kappa) / eta, -np.inf),
        }

    def balance(log_push: FloatArray, log_pull: FloatArray, scale: float) -> FloatArray:
        """Potential shift solving push*exp(+delta/a) = pull*exp(-delta/b); scale = ab/(a+b)."""
        with np.errstate(invalid="ignore"):
            shift = scale * (log_pull - log_push)
        return shift

    def residuals() -> dict[str, float]:
        f = {key: np.exp(value) for key, value in log_flows().items()}
        out_l = np.bincount(p.lr.tail, weights=f["lr"], minlength=n_l)
        in_r = np.bincount(p.lr.head, weights=f["lr"], minlength=n_r)
        in_m = (np.bincount(p.rm.head, weights=f["rm"], minlength=n_m)
                + np.bincount(p.mm.head, weights=f["mm"], minlength=n_m))
        relay_l = np.zeros(n_l) if p.relay_sink else np.bincount(p.ml.head, weights=f["ml"], minlength=n_l)
        out_r = np.bincount(p.rm.tail, weights=f["rm"], minlength=n_r)
        out_m = (np.bincount(p.mm.tail, weights=f["mm"], minlength=n_m)
                 + np.bincount(p.ml.tail, weights=f["ml"], minlength=n_m))
        ligand_budget = np.abs(out_l + f["m"] - p.ligand_budget) / p.ligand_budget
        receptor_budget = np.abs(in_r + f["n"] - p.receptor_budget) / p.receptor_budget
        module_capacity = np.where(finite_capacity, np.abs(in_m + f["u"] - p.module_capacity) / np.where(finite_capacity, p.module_capacity, 1.), 0.)
        # Receptors: inflow equals outflow (module arcs and the silent outlet).
        receptor = np.abs(in_r - out_r - f["outlet"]) / p.receptor_budget
        # Ligand ports: autonomous start z = out - relay must be >= 0, and 0 when phi < gamma.
        z = out_l - relay_l
        capped = phi_l >= gamma - 1e-12
        port = np.where(capped, np.maximum(-z, 0.), np.abs(z)) / p.ligand_budget
        w = in_m - out_m
        module = np.abs(w - f["t"]) / np.where(finite_capacity, p.module_capacity, np.maximum(in_m, 1e-300))
        return {"ligand_budget": float(ligand_budget.max(initial=0.)),
                "receptor_budget": float(receptor_budget.max(initial=0.)),
                "module_capacity": float(module_capacity.max(initial=0.)),
                "receptor_conservation": float(receptor.max(initial=0.)),
                "ligand_conservation": float(port.max(initial=0.)),
                "module_conservation": float(module.max(initial=0.))}

    sweeps, worst = 0, np.inf
    for sweeps in range(1, max_sweeps + 1):
        # Pool multipliers: each rescales its residual and the arcs that use the pool (Sinkhorn steps).
        lf = log_flows()
        total = np.logaddexp(_group_logsumexp(lf["lr"], p.lr.tail, n_l), lf["m"])
        alpha += theta * (total - log_a)
        lf = log_flows()
        total = np.logaddexp(_group_logsumexp(lf["lr"], p.lr.head, n_r), lf["n"])
        beta += theta * (total - log_b)
        if n_m:
            lf = log_flows()
            inflow = np.logaddexp(_group_logsumexp(lf["rm"], p.rm.head, n_m),
                                  _group_logsumexp(lf["mm"], p.mm.head, n_m))
            kappa = np.where(finite_capacity, kappa + eta * (np.logaddexp(inflow, lf["u"]) - log_d), 0.)
        # Receptor potentials: LR inflow ~ exp(+phi/theta), outflow ~ exp(-phi/eta).
        lf = log_flows()
        push = _group_logsumexp(lf["lr"], p.lr.head, n_r)
        pull = np.logaddexp(_group_logsumexp(lf["rm"], p.rm.tail, n_r), lf["outlet"])
        phi_r += balance(push, pull, theta * eta / (theta + eta))
        # Ligand ports: outflow ~ exp(-phi/theta), relay inflow ~ exp(+phi/eta); cap at gamma.
        lf = log_flows()
        demand = _group_logsumexp(lf["lr"], p.lr.tail, n_l)
        relay = np.full(n_l, -np.inf) if p.relay_sink else _group_logsumexp(lf["ml"], p.ml.head, n_l)
        with np.errstate(invalid="ignore"):
            shift = (theta * eta / (theta + eta)) * (demand - relay)
        candidate = np.where(np.isfinite(shift), phi_l + shift, np.where(relay == -np.inf, np.inf, -np.inf))
        phi_l = np.minimum(candidate, gamma)
        # Modules level by level (colour by colour on cyclic module arcs): inflow ~ exp(+phi/eta), outflow and
        # termination ~ exp(-phi/eta).
        for level in levels:
            lf = log_flows()
            inflow = np.logaddexp(_group_logsumexp(lf["rm"], p.rm.head, n_m),
                                  _group_logsumexp(lf["mm"], p.mm.head, n_m))[level]
            outflow = np.logaddexp(_group_logsumexp(lf["mm"], p.mm.tail, n_m),
                                   _group_logsumexp(lf["ml"], p.ml.tail, n_m))
            outflow = np.logaddexp(outflow, lf["t"])[level]
            with np.errstate(invalid="ignore"):
                shift = .5 * eta * (outflow - inflow)
            candidate = np.where(np.isfinite(shift), phi_m[level] + shift,
                                 np.where(outflow == -np.inf, -np.inf, np.inf))
            updated = candidate
            if not np.isfinite(updated).all():
                raise ValueError("A nonterminal module has no inflow or no outflow; prune it first")
            phi_m[level] = updated
        if sweeps % check_every == 0:
            worst = max(residuals().values())
            if worst <= tolerance:
                break
    report = residuals()
    worst = max(report.values())
    f = {key: np.exp(value) for key, value in log_flows().items()}
    out_l = np.bincount(p.lr.tail, weights=f["lr"], minlength=n_l)
    relay_l = np.zeros(n_l) if p.relay_sink else np.bincount(p.ml.head, weights=f["ml"], minlength=n_l)
    autonomous = np.maximum(out_l - relay_l, 0.)
    termination = f["t"]
    outlet = f["outlet"]
    primal = _primal_objective(p, f, outlet, autonomous, gamma, c0)
    dual = _dual_objective(p, f, alpha, beta, kappa, theta, eta)
    certificate = {"sweeps": sweeps, "max_relative_residual": worst, "residuals": report,
                   "converged": bool(worst <= tolerance), "primal_objective": primal, "dual_objective": dual,
                   "duality_gap": primal - dual, "tolerance": tolerance, "method": "exact dual block coordinate ascent"}
    return NetworkEntropySolution(f["lr"], f["rm"], f["mm"], f["ml"], outlet, autonomous, termination,
                                  f["m"], f["n"], f["u"],
                                  {"ligand": phi_l, "receptor": phi_r, "module": phi_m,
                                   "ligand_budget": alpha, "receptor_budget": beta, "module_capacity": kappa},
                                  certificate)


def _entropy(x: FloatArray) -> float:
    positive = x[x > 0]
    return float(np.sum(positive * np.log(positive) - positive))


def _primal_objective(p: NetworkEntropyProblem, f: dict[str, FloatArray], outlet: FloatArray,
                      autonomous: FloatArray, gamma: FloatArray, c0: FloatArray) -> float:
    linear = (float(p.lr.cost @ f["lr"]) + float(p.rm.cost @ f["rm"]) + float(p.mm.cost @ f["mm"])
              + float(p.ml.cost @ f["ml"]) + float(c0 @ outlet) + float(gamma @ autonomous)
              + p.unmatched_ligand_cost * float(f["m"].sum()) + p.unmatched_receptor_cost * float(f["n"].sum())
              + p.unmatched_module_cost * float(f["u"].sum()))
    linear += p.termination_cost * float(f["t"].sum())
    extracellular = _entropy(f["lr"]) + _entropy(f["m"]) + _entropy(f["n"])
    intracellular = _entropy(f["rm"]) + _entropy(f["mm"]) + _entropy(f["ml"]) + _entropy(f["u"])
    intracellular += _entropy(outlet)
    intracellular += _entropy(f["t"])
    return linear + p.theta * extracellular + p.eta * intracellular


def _dual_objective(p: NetworkEntropyProblem, f: dict[str, FloatArray], alpha: FloatArray, beta: FloatArray,
                    kappa: FloatArray, theta: float, eta: float) -> float:
    """min over entropic variables gives -T*exp(-k/T) = -T*x; the linear autonomous starts z contribute 0 at feasible
    duals."""
    value = -theta * float(f["lr"].sum() + f["m"].sum() + f["n"].sum())
    value -= eta * float(f["rm"].sum() + f["mm"].sum() + f["ml"].sum() + f["u"].sum())
    value -= eta * float(f["outlet"].sum())
    value -= eta * float(f["t"].sum())
    finite = np.isfinite(p.module_capacity)
    value -= float(alpha @ p.ligand_budget + beta @ p.receptor_budget + kappa[finite] @ p.module_capacity[finite])
    return value
