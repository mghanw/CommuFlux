# Vendored: CSCN engine

`core.py` is an unmodified copy of `cscn/core.py` from https://github.com/mghanw/CSCN at commit d579c03d77a6435c570957188de356b72e369f00 (MIT, Menghan Wang; full text in `LICENSE`). It holds the cell-specific PC procedure that `commuflux.cscn` calls.

`__init__.py` exports only `CSCN` and `CellPCResult`. The CSCN preprocessing, runner and command-line modules are not used by CommuFlux and are not included.
