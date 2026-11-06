import networkx as nx
import numpy as np
import pytest

from commuflux import cscn as local_network
from commuflux.cscn import column

ROLES = {column("receptor", "R1"): "receptor", column("tf", "TF1"): "tf", column("module", "me_00"): "module",
         column("module", "me_01"): "module", column("ligand", "L1"): "ligand"}


def test_equivalent_proxies_pair_a_tf_variable_with_the_module_of_its_gene():
    genes = ["R1", "TF1", "TF2", "G"]
    resources = {"receptor_tf": [{"source": "R1", "target": "TF1"}, {"source": "R2", "target": "TF2"}]}
    assignment = np.array([-1, 2, 0, 1])  # TF1 in module me_01, TF2 grey
    pairs = local_network.equivalent_proxies(genes, resources, assignment, ["me_00", "me_01"])
    assert pairs == [[column("tf", "TF1"), column("module", "me_01")]]


def test_background_knowledge_pairs_each_tf_with_its_module_both_ways():
    genes, resources = ["R1", "TF1"], {"receptor_tf": [{"source": "R1", "target": "TF1"}]}
    knowledge = local_network.background_knowledge(genes, resources, np.array([-1, 1]), ["me_00"])
    tf, module = column("tf", "TF1"), column("module", "me_00")
    assert knowledge == {"equivalent": {tf: {module}, module: {tf}}}


def test_role_tiers_bar_ligands_from_separating_intracellular_pairs():
    knowledge = {"equivalent": {}}
    receptor, tf = column("receptor", "R1"), column("tf", "TF1")
    m0, m1, ligand = column("module", "me_00"), column("module", "me_01"), column("ligand", "L1")
    assert not local_network.admissible_separator([ligand], tf, m0, ROLES, knowledge)
    assert not local_network.admissible_separator([ligand], m0, m1, ROLES, knowledge)
    assert not local_network.admissible_separator([m1], tf, m0, ROLES, knowledge)  # later than the earlier endpoint
    assert local_network.admissible_separator([receptor], tf, m0, ROLES, knowledge)
    assert local_network.admissible_separator([m1], m0, ligand, ROLES, knowledge)


def test_equivalent_proxies_do_not_separate_each_other():
    tf, m0, m1, ligand = column("tf", "TF1"), column("module", "me_00"), column("module", "me_01"), column("ligand", "L1")
    knowledge = {"equivalent": {tf: {m0}, m0: {tf}}}
    assert not local_network.admissible_separator([tf], m0, m1, ROLES, knowledge)
    assert not local_network.admissible_separator([tf], m0, ligand, ROLES, knowledge)
    assert local_network.admissible_separator([tf], m1, ligand, ROLES, knowledge)


def test_role_override_keeps_conflicting_arcs_in_the_role_direction():
    module, ligand = column("module", "me_00"), column("ligand", "L1")
    pdag = nx.DiGraph([(ligand, module)])  # PC arrow against module -> ligand
    kept = local_network.role_relations(pdag, ROLES)
    assert kept["supported"] == [[module, ligand, "role_override"]] and kept["conflicts"] == []


def test_entry_modules_are_oriented_like_their_tf():
    tf, entry, other = column("tf", "TF1"), column("module", "me_00"), column("module", "me_01")
    knowledge = {"equivalent": {tf: {entry}, entry: {tf}}}
    assert local_network.entry_modules(knowledge, ROLES) == frozenset({entry})
    pdag = nx.DiGraph([(other, entry)])  # PC arrow into the entry module
    kept = local_network.role_relations(pdag, ROLES, entry=frozenset({entry}))
    assert kept["supported"] == [[entry, other, "role_override"]]
    assert local_network.role_relations(pdag, ROLES)["supported"] == [[other, entry, "PC"]]


def test_prior_strength_raises_the_pair_significance_level():
    pytest.importorskip("cscn")
    import pandas as pd
    names = [column("receptor_proxy", "TF1"), column("module", "me_00"), column("module", "me_01")]
    captured = {"prior": {frozenset(names[:2]): .5}}
    engine = local_network.adopted_engine(captured).__new__(local_network.adopted_engine(captured))
    engine.significance_level, engine.df = .01, pd.DataFrame(np.ones((3, 3)), columns=names)
    assert np.isclose(engine.pair_alpha(0, 1), .01 + .5 * (local_network.PRIOR_ALPHA - .01))
    assert engine.pair_alpha(0, 2) == .01


def test_arc_signs_follow_the_local_correlation():
    pytest.importorskip("cscn")
    import pandas as pd
    rng = np.random.default_rng(0)
    a = rng.uniform(size=400)
    names = [column("tf", "T"), column("module", "up"), column("module", "down")]
    table = pd.DataFrame({names[0]: a, names[1]: a + .02 * rng.normal(size=400), names[2]: 1. - a + .02 * rng.normal(size=400)})
    networks = [{"query": q, "support": [{"tail": names[0], "head": names[1]}, {"tail": names[0], "head": names[2]}],
                 "additions": []} for q in (10, 200, 390)]
    signed = local_network.arc_signs(table, networks, {names[0]: "tf", names[1]: "module", names[2]: "module"})
    assert [[edge["sign"] for edge in network["support"]] for network in signed] == [[1, -1]] * 3


def test_one_sided_tests_count_only_a_positive_dependence():
    pytest.importorskip("cscn")
    from contextlib import redirect_stdout
    from io import StringIO
    rng = np.random.default_rng(0)
    a = rng.uniform(size=400)
    data = np.column_stack([a, a + .02 * rng.normal(size=400), 1. - a + .02 * rng.normal(size=400)])
    names = [column("tf", "T"), column("module", "up"), column("module", "down")]
    engine = local_network.adopted_engine({"query": 200, "roles": {}, "knowledge": None, "prior": {}})(
        **local_network.engine_parameters(), variable_names=names, enable_query_stats=True)
    with redirect_stdout(StringIO()):
        engine.run_core(data, use_nmf=False)
    found = ([engine.local_marginal_dependent(0, k, 200, .01) for k in (1, 2)],
             [engine.partial_p_value(0, k, set(), np.arange(400)) < .01 for k in (1, 2)])
    assert found == ([True, False], [True, False])


def test_engine_parameters_are_one_sided_at_the_cscn_level():
    assert local_network.engine_parameters() == {**local_network.CSCN_PARAMETERS, "ci_alternative": "greater",
                                                 "significance_level": .01}
    assert local_network.CSCN_PARAMETERS["ci_alternative"] == "two-sided"  # the engine of arc_signs


def test_keynode_table_tiers_and_receptor_tf_prior():
    genes = ["L1", "R1", "TF1", "G"]
    counts = np.array([[1, 2, 0, 5], [2, 1, 4, 3], [3, 3, 1, 4]])
    resources = {"roles": {"ligands": ["L1"], "receptors": ["R1"]}, "receptor_tf": [{"source": "R1", "target": "TF1"}]}
    cells, eigengenes = ["a", "b", "c"], np.ones((3, 1))
    table, roles = local_network.local_table(counts, genes, cells, resources, eigengenes, ["me_00"])
    receptor, tf = column("receptor", "R1"), column("tf", "TF1")
    module, ligand = column("module", "me_00"), column("ligand", "L1")
    assert roles == {ligand: "ligand", receptor: "receptor", tf: "tf", module: "module"}
    assert list(table.columns) == [ligand, receptor, tf, module]
    knowledge = {"equivalent": {}}
    assert not local_network.admissible_separator([module], tf, ligand, roles, knowledge)  # between the endpoints
    assert not local_network.admissible_separator([ligand], receptor, module, roles, knowledge)
    assert local_network.receptor_tf_prior(resources, roles) == [[receptor, tf, 1.]]


def test_keynode_role_relations_orient_by_tier():
    receptor, tf = column("receptor", "R1"), column("tf", "TF1")
    module, ligand = column("module", "me_00"), column("ligand", "L1")
    roles = {receptor: "receptor", tf: "tf", module: "module", ligand: "ligand"}
    pdag = nx.DiGraph([(tf, receptor), (module, tf), (ligand, module), (receptor, ligand)])  # against the tiers
    kept = local_network.role_relations(pdag, roles)
    assert sorted(kept["supported"]) == sorted([[receptor, tf, "role_override"], [tf, module, "role_override"],
                                                [module, ligand, "role_override"]])  # no receptor -> ligand arc
    assert kept["conflicts"] == []
    plain = local_network.role_relations(nx.DiGraph([(tf, ligand), (ligand, tf)]), roles)
    assert plain["supported"] == [[tf, ligand, "role_prior"]]  # a TF directly inducing a ligand
    reverse = local_network.role_relations(nx.DiGraph([(ligand, tf)]), roles)  # PC arrow against TF -> ligand
    assert reverse["supported"] == [[tf, ligand, "role_override"]]
