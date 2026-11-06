import numpy as np

from commuflux import flow


def tiny():
    lig = {flow.node("a", "ligand", "L"): 1., flow.node("b", "ligand", "L"): 2.}
    rec = {flow.node("a", "receptor", "R"): 1.5, flow.node("b", "receptor", "R"): .5}
    lr = [("lr:1", flow.node("a", "ligand", "L"), flow.node("b", "receptor", "R"), .3),
          ("lr:2", flow.node("b", "ligand", "L"), flow.node("a", "receptor", "R"), .3)]
    return {"ligand_budget": lig, "receptor_budget": rec, "lr": lr, "intra": [], "gem": {},
            "tf_level": {flow.node(c, "tf", "T"): 2. for c in "ab"}, "receptor_tfs": {"R": [("T", 1.)]}}


def test_dilute_offset_reaches_the_target_ligand_use():
    data = tiny()
    offset = flow.dilute_offset(data, .01)
    used = flow.solve(data, "transport", delta=offset)["usage"]["ligand_used_fraction"]
    assert abs(used - .01) < 2e-3


def test_tf_layer_arms_solve_and_conserve():
    result = flow.solve(tiny(), "tf", delta=2., scale=.5)
    assert result["certificate"]["converged"]
    assert all(value >= 0 for value in result["flows"].values())


def test_entropic_termination_is_invariant_to_a_common_pool_scale():
    from dataclasses import replace
    from commuflux.solver import solve_network_entropy
    data = {**tiny(), "gem": {flow.node(c, "module", "M"): 3. for c in "ab"},
            "intra": [(flow.node(c, "tf", "T"), flow.node(c, "module", "M"), .8) for c in "ab"]}
    problem, _ = flow.network(data, "full", delta=2., scale=.5)
    k = 3.7
    scaled = replace(problem, ligand_budget=problem.ligand_budget * k, receptor_budget=problem.receptor_budget * k,
                     module_capacity=problem.module_capacity * k)
    base, big = (solve_network_entropy(x, tolerance=1e-11) for x in (problem, scaled))
    assert base.certificate["converged"] and big.certificate["converged"]
    assert np.allclose(big.lr / k, base.lr, rtol=1e-6) and np.allclose(big.rm / k, base.rm, rtol=1e-6)


def test_the_transport_arm_has_no_internal_nodes():
    # the calibration solves of communication_flow run this arm: termination, relay sinks and acyclic arcs cannot
    # change them
    problem, index = flow.network(cyclic_case(), "transport")
    assert index["modules"] == [] and not len(problem.rm) and not len(problem.mm) and not len(problem.ml)


def test_relay_arcs_end_in_sinks_of_the_ligand_ports():
    from commuflux.solver import solve_network_entropy
    data = {**tiny(), "gem": {flow.node(c, "module", "M"): 3. for c in "ab"},
            "intra": [arc for c in "ab" for arc in ((flow.node(c, "tf", "T"), flow.node(c, "module", "M"), .8),
                                                     (flow.node(c, "module", "M"), flow.node(c, "ligand", "L"), .5))]}
    problem, index = flow.network(data, "full", delta=2., scale=.5)
    assert len(problem.ml) == 2 and len(problem.mm) == 2 and problem.relay_sink
    solution = solve_network_entropy(problem, tolerance=1e-8, max_sweeps=20000)
    sent = np.bincount(problem.lr.tail, weights=solution.lr, minlength=len(index["senders"]))
    assert np.allclose(solution.autonomous, sent) and (solution.ml > 0).all()  # each port sends its own budget only
    result = flow.solve(data, "full", delta=2., scale=.5)
    assert result["certificate"]["converged"] and len(result["relay_inflow"]) == 2
    assert all(0 < share <= 1 for shares in result["relay_reach"].values() for share in shares.values())


def cyclic_case() -> dict:
    m, n = (flow.node("a", "module", k) for k in "MN")
    return {**tiny(), "gem": {flow.node(c, "module", k): 3. for c in "ab" for k in "MN"},
            "intra": [(flow.node("a", "tf", "T"), m, .8), (m, n, .6), (n, m, .5), (n, flow.node("a", "ligand", "L"), .7)]}


def assert_relay_reach_adds_up(result: dict) -> None:
    assert result["certificate"]["converged"]
    port = flow.node("a", "ligand", "L")
    shares = result["relay_reach"][flow.node("a", "receptor", "R")]
    assert 0 < shares[port] <= 1 and set(shares) == {port}
    # every unit of relay flow into a's ligand port entered through a's receptor
    assert np.isclose(result["receptor_inflow"][flow.node("a", "receptor", "R")] * shares[port], result["relay_inflow"][port])


def test_internal_cycles_are_broken_by_support():
    m, n = (flow.node("a", "module", k) for k in "MN")
    data = cyclic_case()
    problem, index = flow.network(data, "full", delta=2., scale=.5)
    assert len(problem.mm) == 2 and (n, m, .5) not in index["mm"]  # n -> m would close the cycle through m -> n (.6)
    assert_relay_reach_adds_up(flow.solve(data, "full", delta=2., scale=.5))


def response_case(heads: str, tf_a: float) -> dict:
    return {**tiny(), "tf_level": {flow.node("a", "tf", "T"): tf_a, flow.node("b", "tf", "T"): 2.},
            "gem": {flow.node(c, "module", k): 3. for c in "ab" for k in heads},
            "intra": [(flow.node(c, "receptor", "R"), flow.node(c, "tf", "T"), .9) for c in "ab"]
                     + [(flow.node(c, "tf", "T"), flow.node(c, "module", k), .8) for c in "ab" for k in heads]}


def test_response_pull_cuts_a_zero_count_tf_with_what_only_it_reaches():
    data = response_case("M", 0.)
    problem, index = flow.network(data, "full", delta=2., scale=.5)
    assert flow.node("a", "tf", "T") not in index["modules"] and flow.node("a", "module", "M") not in index["modules"]
    assert {flow.node("b", "tf", "T"), flow.node("b", "module", "M")} <= set(index["modules"])
    assert np.isfinite(problem.module_capacity).all() and (problem.module_capacity > 0).all()
    assert flow.solve(data, "full", delta=2., scale=.5)["certificate"]["converged"]


def test_response_pull_does_not_grow_with_downstream_branches():
    def cell_total(heads: str) -> float:
        problem, index = flow.network(response_case(heads, 2.), "full", delta=2., scale=.5)
        return float(sum(c for name, c in zip(index["modules"], problem.module_capacity) if flow.port(name)[0] == "b"))
    assert np.isclose(cell_total("M"), cell_total("MNOP"))
    # equal normalized levels: each receptor's pull equals the mean budget (kappa) times the scale
    assert np.isclose(cell_total("M"), .5 * (1.5 + .5) / 2)


def test_capacities_do_not_depend_on_a_gene_s_expression_unit():
    # two pathways (R -> T -> M, S -> U); kappa is calibrated per factor, so rescaling U and M in every cell changes
    # no capacity and no flow (with one kappa per node type it would shift the TFs' relative capacities)
    lig = {flow.node("a", "ligand", "L"): 1., flow.node("b", "ligand", "L"): 2.}
    rec = {flow.node(c, "receptor", r): v for c, r, v in (("a", "R", 1.5), ("b", "R", .5), ("a", "S", 1.), ("b", "S", 1.))}
    lr = [(f"lr:{k}", flow.node(s, "ligand", "L"), flow.node(r, "receptor", g), cost)
          for k, (s, r, g, cost) in enumerate((("a", "b", "R", .3), ("b", "a", "R", .3), ("a", "b", "S", .2), ("b", "a", "S", .2)))]
    data = {"ligand_budget": lig, "receptor_budget": rec, "lr": lr, "receptor_tfs": {"R": [("T", 1.)], "S": [("U", 1.)]},
            "tf_level": {flow.node("a", "tf", "T"): 3., flow.node("b", "tf", "T"): 1.,
                         flow.node("a", "tf", "U"): .2, flow.node("b", "tf", "U"): .6},
            "gem": {flow.node("a", "module", "M"): 2., flow.node("b", "module", "M"): 5.},
            "intra": [(flow.node(c, "tf", "T"), flow.node(c, "module", "M"), .8) for c in "ab"]}
    scaled = {**data, "tf_level": {k: v * (10. if flow.port(k)[2] == "U" else 1.) for k, v in data["tf_level"].items()},
              "gem": {k: 7. * v for k, v in data["gem"].items()}}
    (problem, index), (other, other_index) = (flow.network(x, "full", delta=2., scale=.5) for x in (data, scaled))
    assert index["modules"] == other_index["modules"] and np.allclose(problem.module_capacity, other.module_capacity)
    base, big = (flow.solve(x, "full", delta=2., scale=.5)["flows"] for x in (data, scaled))
    assert base.keys() == big.keys() and np.allclose([big[k] for k in base], [base[k] for k in base], rtol=1e-6, atol=1e-12)


def test_acyclic_internal_breaks_cycles_by_support() -> None:
    from commuflux.flow import acyclic_internal, node
    a, b, c = (node("c0", "module", m) for m in ("a", "b", "c"))
    receptor = node("c0", "receptor", "R")
    arcs = [(receptor, a, .9), (a, b, .8), (b, a, .5), (b, c, .7), (c, a, .6)]
    kept = acyclic_internal(arcs)
    assert (b, a, .5) not in kept and (c, a, .6) not in kept  # each would close a cycle through the stronger arcs
    assert {(receptor, a, .9), (a, b, .8), (b, c, .7)} <= set(kept)
