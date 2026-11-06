# Vendored: dynamicTreeCut 0.1.1 (PyPI)

Python translation of the hybrid Dynamic Tree Cut (Langfelder, Zhang & Horvath, Bioinformatics 2008) by Kyle S. Smith, https://github.com/kylessmith/dynamicTreeCut, GPL-3 (full text in `LICENSE`). Copies (line endings normalized to LF, otherwise unmodified) of `__init__.py`, `dynamicTreeCut.py`, `R_func.py` and `df_apply.py` from the 0.1.1 source distribution, downloaded 2026-10-17 and read in full before use.

Only the tree-cut stage is used (`pamStage=False`). The PAM stage of this port has defects found on reading: average distances stored in an integer array, an off-by-one loop over cluster labels, and `levels(...)[nearest - 1]` on a zero-based index. 
- `chunkSize` is raised at call time by the caller so that branch buffers never grow (the port grows them with a float repeat count that current numpy rejects).
- `rankdata` is replaced at call time by the caller with the same ranks cast to integers: the port indexes arrays with it and current scipy returns floats. All calls in the port use `method="ordinal"`, whose ranks are integers.
- The port's module-level `np` is replaced at call time by a thin wrapper that forwards everything to numpy except that `repeat` casts its counts to integers (the port passes `np.ceil` floats).
