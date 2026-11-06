"""Command line: CommuFlux inference on a run directory written by commuflux.data.write_input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="commuflux", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="infer communication and relays; writes RUN/scores, RUN/communication.csv.gz "
                                              "and RUN/relay.csv.gz")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--workers", type=int, default=16, help="processes for the B50 bootstrap")
    infer.add_argument("--stage", choices=("all", "prepare", "networks", "finish"), default="all",
                       help="all: the whole inference in one process; or as separate jobs: prepare (modules, CSCN table, "
                            "knowledge), networks (B50 networks of one part of the cells, --part), finish (merge the "
                            "parts, flow, scores)")
    infer.add_argument("--part", default="1/1", help="networks stage: part K/N, the cells K-1, K-1+N, K-1+2N, ...")
    args = parser.parse_args()
    from . import infer as inference

    if args.stage == "all":
        print(json.dumps(inference.infer(args.run, args.workers), indent=1))
    elif args.stage == "prepare":
        inference.prepare(args.run)
    elif args.stage == "networks":
        index, parts = (int(value) for value in args.part.split("/"))
        print(inference.part(args.run, args.workers, index, parts))
    else:
        print(json.dumps(inference.finish(args.run), indent=1))


if __name__ == "__main__":
    main()
