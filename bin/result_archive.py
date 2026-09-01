#!/usr/bin/env python3
"""Build one deterministic result archive on a remote stock-Python host."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity.resultsink import build_summary, write_archive  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs-root", required=True)
    parser.add_argument("--verb", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--stages", required=True,
                        help="comma-separated completed/attempted stage names")
    parser.add_argument("--failed-stage")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    stages = [value for value in args.stages.split(",") if value]
    summary = build_summary(args.fs_root, args.verb, args.status, stages,
                            failed_stage=args.failed_stage)
    result = write_archive(args.fs_root, summary, args.out)
    # The controller parses stdout. Keep it to one canonical object and never
    # include job content, environment, credentials, or log bytes.
    print(json.dumps({key: result[key] for key in ("path", "bytes", "sha256")},
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
