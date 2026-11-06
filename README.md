# CommuFlux

CommuFlux infers spatial cell–cell communication as one entropy-regularized network flow that runs from ligands through receptors into the cell-specific intracellular network of every receiving cell. The networks are inferred per cell with CSCN, using molecular-role ordering (receptor → TF → module → ligand) and B50 bootstrap support. Solving the joint flow gives:

- **communication strengths** for every sender cell, receiver cell and ligand–receptor pair within the interaction radius;
- **relay routing fractions**: the share of a cell's receptor inflow that its intracellular network routes to each of its ligands, which links incoming and outgoing signals into relays.

## Installation

```bash
pip install git+https://github.com/mghanw/CommuFlux.git
```

Python ≥ 3.10. The CSCN engine is included in `src/cscn` (from github.com/mghanw/CSCN at commit d579c03), so all dependencies install from PyPI.

## Input

One run directory per tissue section. Write it from your own data:

```python
from commuflux import write_input

write_input(
    "runs/section1",
    counts,               # cells x genes raw counts
    genes, cells,         # gene and cell identifiers
    coordinates,          # cells x 2, e.g. micrometres
    lr_pairs=[("IFNG", "IFNGR1"), ("CXCL9", "CXCR3")],
    receptor_tf=[("IFNGR1", "STAT1"), ("CXCR3", "NFKB1")],
    radius=50.0,          # interaction radius, in the unit of the coordinates
)
```

Every receptor needs at least one TF, and every ligand, receptor and TF must be in the gene panel. `examples/xenium/prepare_xenium.py` builds a run from a 10x Xenium output. It selects a region, LR pairs and receptor–TF links from CellCall's KEGG-derived triples, keeping TFs listed in CollecTRI. An example configuration is in `examples/xenium/skin-melanoma.json`.

## Running

```bash
commuflux infer --run runs/section1 --workers 16
```

For large sections, the per-cell networks can be split over jobs:

```bash
commuflux infer --run runs/section1 --stage prepare
commuflux infer --run runs/section1 --stage networks --part 1/8   # ... up to 8/8, in parallel
commuflux infer --run runs/section1 --stage finish
```

## Output

| File | Content |
| --- | --- |
| `communication.csv.gz` | sender, receiver, ligand, receptor, communication score (full model) |
| `relay.csv.gz` | cell, receptor, ligand, fraction of the receptor's inflow routed to the ligand |
| `scores/commuflux_full.npz` | scores of all candidate edges (`S`) and relay fractions (`relay`, cells × receptors × ligands) |
| `scores/commuflux_tf.npz`, `scores/commuflux_transport.npz` | reference arms: TF layer only, and transport without intracellular nodes |
| `inference/` | modules, CSCN input table, cell networks, tissue graph, flow summary |

The order of `S` is given by `commuflux.data.candidate_axes`. A relay A → B → C through receptor R and ligand L′ of cell B has score F(A→B) · π_B(R→L′) · F(B→C) / O_B(L′), where O_B(L′) is B's total outflow of L′.

## License

CommuFlux is released under the GNU General Public License v3.0 (`LICENSE`).

## Included code

- `src/cscn`: the PC engine of CSCN (github.com/mghanw/CSCN at d579c03, MIT); see `src/cscn/VENDORED.md`.
- `src/dynamicTreeCut`: dynamicTreeCut 0.1.1 by Kyle S. Smith (GPL-3, full text in `src/dynamicTreeCut/LICENSE`); see `src/dynamicTreeCut/VENDORED.md`.
