import numpy as np

from commuflux.solver import Arcs, NetworkEntropyProblem, solve_network_entropy


def test_relay_sink_removes_sourceless_circulation() -> None:
    # Two cells in a loop: A's ligand -> B's receptor -> B's module -> B's ligand -> A's receptor -> A's module -> A's ligand.
    # Feeding the ports, relay flow lets each port send more than its autonomous start; ending in sinks it cannot.
    lr = Arcs(np.array([0, 1]), np.array([0, 1]), np.array([.1, .1]))
    rm = Arcs(np.array([0, 1]), np.array([0, 1]), np.array([.0, .0]))
    ml = Arcs(np.array([0, 1]), np.array([1, 0]), np.array([.0, .0]))
    common = dict(ligand_budget=np.array([1., 1.]), receptor_budget=np.array([1., 1.]), module_capacity=np.array([1., 1.]),
                  terminal=np.array([True, True]), lr=lr, rm=rm, ml=ml)
    fed = solve_network_entropy(NetworkEntropyProblem(**common, relay_sink=False))
    sink = solve_network_entropy(NetworkEntropyProblem(**common, relay_sink=True))
    assert fed.certificate["converged"] and sink.certificate["converged"]
    sent = lambda s: np.bincount(lr.tail, weights=s.lr, minlength=2)  # noqa: E731
    assert (fed.ml > 0).all() and (fed.autonomous < sent(fed) - 1e-6).all()  # part of each port's flow is relay
    assert np.allclose(sink.autonomous, sent(sink)) and (sink.ml > 0).all()  # every port flow is its own; relay still measured
