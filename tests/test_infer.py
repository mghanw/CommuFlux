import pickle

import numpy as np
import pandas as pd
import pytest

from commuflux import cscn, graph, infer
from commuflux.data import CANDIDATE_RADIUS, candidate_axes, read_input, write_input


def prepared(tmp_path, cells):
    """A run directory with what the network stage reads: the CSCN table and the stage context."""
    inference = tmp_path / "inference"
    inference.mkdir(exist_ok=True)
    pd.DataFrame({"x": range(len(cells))}, index=cells).to_csv(inference / "local-table.csv", lineterminator="\n")
    with open(inference / infer.CONTEXT, "wb") as stream:
        pickle.dump({"roles": {}, "groups": None, "knowledge": None}, stream)
    return tmp_path


def test_network_parts_are_interleaved_and_merge_in_query_order(tmp_path, monkeypatch):
    cells = [f"c{k}" for k in range(7)]
    run = prepared(tmp_path, cells)
    calls = []

    def bootstrap(table, roles, groups, workers, progress=None, knowledge=None, queries=None):
        calls.append(list(queries))
        return [{"query": q, "cell": str(table.index[q]), "replicates": [], "support": []} for q in queries]

    monkeypatch.setattr(cscn, "bootstrap_networks", bootstrap)
    monkeypatch.setattr(graph, "q_additions", lambda network, roles: [])
    monkeypatch.setattr(cscn, "arc_signs", lambda table, networks, roles: networks)  # needs CSCN
    for part in (2, 1, 3):
        infer.part(run, 1, part, 3, log=lambda message: None)
    assert calls == [[1, 4], [0, 3, 6], [2, 5]]
    networks = infer.merged_networks(run)
    assert [(n["query"], n["cell"]) for n in networks] == list(enumerate(cells))
    assert all(n["additions"] == [] and "replicates" not in n for n in networks)
    with pytest.raises(FileExistsError):
        infer.part(run, 1, 2, 3)
    with pytest.raises(ValueError):
        infer.cell_networks(run, 1, part=4, parts=3)


def test_merged_networks_need_every_cell_exactly_once(tmp_path):
    run = prepared(tmp_path, ["a", "b", "c"])
    path = run / "inference"
    infer.write_networks(path / "networks-part-1-of-2.json.gz", [{"query": 0, "cell": "a"}, {"query": 2, "cell": "c"}])
    with pytest.raises(ValueError):
        infer.merged_networks(run)  # part 2 of 2 is missing
    infer.write_networks(path / "networks-part-2-of-2.json.gz", [{"query": 1, "cell": "b"}])
    assert [n["cell"] for n in infer.merged_networks(run)] == ["a", "b", "c"]


def test_write_input_scales_the_radius_and_checks_the_annotation(tmp_path):
    counts = np.array([[3, 1, 2, 4], [1, 2, 2, 1], [5, 0, 1, 2]])
    genes, cells = ["L", "R", "T", "X"], ["a", "b", "c"]
    xy = np.array([[0., 0.], [30., 0.], [100., 0.]])
    write_input(tmp_path / "run", counts, genes, cells, xy, [("L", "R")], [("R", "T")], radius=50.)
    observations, resources = read_input(tmp_path / "run")
    assert np.allclose(observations["coordinates"][:, 0], [0., 30. * CANDIDATE_RADIUS / 50., 100. * CANDIDATE_RADIUS / 50.])
    assert resources["roles"] == {"ligands": ["L"], "receptors": ["R"]}
    axes = candidate_axes(observations, resources)
    assert sorted(zip(axes["sender"], axes["receiver"])) == [("a", "b"), ("b", "a")]  # c is 70 and 100 away
    with pytest.raises(ValueError, match="without a TF"):
        write_input(tmp_path / "other", counts, genes, cells, xy, [("L", "R")], [], radius=50.)
    with pytest.raises(ValueError, match="not in the panel"):
        write_input(tmp_path / "third", counts, genes, cells, xy, [("L", "Q")], [("Q", "T")], radius=50.)


def test_tables_hold_the_positive_scores_and_relay_fractions(tmp_path):
    counts = np.array([[3, 1, 2, 4], [1, 2, 2, 1], [5, 0, 1, 2]])
    xy = np.array([[0., 0.], [30., 0.], [100., 0.]])
    write_input(tmp_path, counts, ["L", "R", "T", "X"], ["a", "b", "c"], xy, [("L", "R")], [("R", "T")], radius=50.)
    axes = candidate_axes(*read_input(tmp_path))
    scores = tmp_path / "scores"
    scores.mkdir()
    np.savez_compressed(scores / f"{infer.PREFIX}_full.npz", S=np.array([.5, 0.]))
    infer.write_tables(tmp_path)
    table = pd.read_csv(tmp_path / "communication.csv.gz")
    assert table.values.tolist() == [[axes["sender"][0], axes["receiver"][0], "L", "R", .5]]
    relays = pd.read_csv(tmp_path / "relay.csv.gz")  # no receptor reaches a ligand: header only
    assert relays.empty and list(relays.columns) == ["cell", "receptor", "ligand", "fraction"]
    relay = np.zeros((3, 1, 1))
    relay[1, 0, 0] = .25
    np.savez_compressed(scores / f"{infer.PREFIX}_full.npz", S=np.array([.5, 0.]), relay=relay)
    infer.write_tables(tmp_path)
    assert pd.read_csv(tmp_path / "relay.csv.gz").values.tolist() == [["b", "R", "L", .25]]
