#!/usr/bin/env python3
from __future__ import annotations

import json

from _bootstrap import ROOT
from dit_research.utils import atomic_json_dump, environment_manifest


def main() -> None:
    report = environment_manifest()
    destination = ROOT / "environment-report.json"
    atomic_json_dump(report, destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()
