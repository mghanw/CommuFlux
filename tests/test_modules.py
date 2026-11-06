import numpy as np

from commuflux import modules as representation


def structured_counts(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Three cell types, each with one of three 60-gene programs switched on, plus unregulated genes."""
    rng = np.random.default_rng(seed)
    cells, programs, size, background = 300, 3, 60, 120
    cell_type = rng.integers(0, programs, cells)
    activity = rng.gamma(2., .5, (cells, programs)) + 3. * np.eye(programs)[cell_type]
    means = np.full((cells, programs * size + background), 2.)
    for p in range(programs):
        means[:, p * size:(p + 1) * size] = 1. + 4. * activity[:, [p]]
    counts = rng.poisson(means * rng.lognormal(0., .2, (cells, 1)))
    return counts, [f"g{g:04d}" for g in range(counts.shape[1])]


def test_a_module_may_have_no_counts_in_a_cell() -> None:
    counts, genes = structured_counts()
    counts[0, :60] = 0  # cell 0 has no counts in the first program's genes
    rep = representation.build_representation(counts, genes)
    assert (rep["levels"] >= 0).all() and (rep["levels"].max(axis=0) > 0).all()
    module = int(rep["assignment"][0])
    if module >= 1 and np.flatnonzero(rep["assignment"] == module).max() < 60:
        assert rep["levels"][0, module - 1] == 0
