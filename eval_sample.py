from __future__ import annotations

import json

from app.pipeline import run_eval


if __name__ == "__main__":
    report = run_eval(threshold=0.552)
    print(json.dumps(report, indent=2))
