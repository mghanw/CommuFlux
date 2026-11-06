from commuflux import graph


def network(cell, support, additions=()):
    return {"cell": cell, "support": list(support), "additions": list(additions)}


def test_g0_keeps_both_arcs_of_a_cycle():
    roles = {'["module","a"]': "module", '["module","b"]': "module"}
    support = [{"tail": '["module","a"]', "head": '["module","b"]', "omega": .9},
               {"tail": '["module","b"]', "head": '["module","a"]', "omega": .4}]
    result = graph.tissue_graph(["c0"], roles, [network("c0", support)], [])
    assert [arc[3] for arc in result["intra"]] == [.9, .4]  # the flow breaks the cycles among a cell's internal nodes
    assert result["excluded"] == {}


def test_g0_keeps_every_lr_arc():
    roles = {'["ligand","L"]': "ligand", '["receptor","R"]': "receptor", '["module","m"]': "module"}
    support = [{"tail": '["receptor","R"]', "head": '["module","m"]', "omega": 1.},
               {"tail": '["module","m"]', "head": '["ligand","L"]', "omega": .6}]
    networks = [network("c0", support), network("c1", support)]
    result = graph.tissue_graph(["c0", "c1"], roles, networks, [("c0", "L", "c1", "R"), ("c1", "L", "c0", "R")])
    # c0 relay -> c1 receptor -> c1 relay -> c0 receptor is a cycle between the cells: G0 keeps it (the flow ends the
    # relay arcs in sinks of the ligand ports).
    assert len(result["intra"]) == 4 and len(result["lr"]) == 2 and result["excluded"] == {}


def test_q_additions_need_a_majority_and_a_complete_route():
    roles = {'["receptor_proxy","R"]': "receptor", '["module","m"]': "module", '["ligand","L"]': "ligand"}
    r, m, l = roles
    replicates = [{"supported": [[r, m, "PC"], [m, l, "PC"]], "conflicts": []}] * 30 + [{"supported": [], "conflicts": []}] * 10
    net = {"original": {"supported": [[r, m, "PC"]], "conflicts": []}, "replicates": replicates,
           "support": [{"tail": r, "head": m, "omega": .75, "planned": 40, "positive": 30}]}
    additions = graph.q_additions(net, roles)
    assert [(a["tail"], a["head"], a["omega"]) for a in additions] == [(m, l, .75)]
    half = [{"supported": [[r, m, "PC"], [m, l, "PC"]], "conflicts": []}] * 25 + [{"supported": [], "conflicts": []}] * 25
    assert [a["planned"] for a in graph.q_additions({**net, "replicates": half}, roles)] == [50]  # B50: 25 of 50 is half
    fewer = half[1:] + [{"supported": [], "conflicts": []}]
    assert graph.q_additions({**net, "replicates": fewer}, roles) == []
