"""Run every experiment and write results/<name>.json. Usage: python run_all.py [names...]"""

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sable.experiments import ALL  # noqa: E402
OUT = ROOT / "results"


def main(names):
    OUT.mkdir(exist_ok=True)
    names = names or list(ALL)
    for name in names:
        t0 = time.time()
        res = ALL[name]()
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf8")
        print(f"{name:24s} {time.time() - t0:7.1f}s -> results/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
