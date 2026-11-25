"""Run and hash-lock frozen external comparisons when every shared input is ready."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time

from growbench_stock import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=["manual_external", "interrupted", "pressure"])
    args = parser.parse_args()
    cohort = args.cohort
    manifest = ROOT / "results/growbench/labels" / cohort / "seeds.json"
    output = ROOT / "results/growbench" / ("stock_" + cohort)
    while not manifest.is_file():
        time.sleep(3)
    sites = json.loads(manifest.read_text())["sites"]
    modes = ([("oriented8", cutoff) for cutoff in (1, 128, 192)] if cohort == "manual_external"
             else [("default20", 1), ("oriented8", 1)])
    while not all((manifest.parent / "inputs" / s["id"] / "m7.json").is_file() for s in sites):
        time.sleep(3)
    def run(mode):
        variant, threshold = mode
        py = [sys.executable, "-u"]
        shared = ["--manifest", str(manifest), "--output", str(output)]
        subprocess.run(py + [str(ROOT/"scripts/growbench_stock.py"), *shared, "--variant", variant,
             "--grid-threshold", str(threshold), "--diagnostics", "--workers", "3",
             "--pitch-um", "7.91" if cohort == "manual_external" else "9.6"], check=True)
        name = variant + (f"_th{threshold}" if threshold != 1 else "") + "_loss"
        subprocess.run(py + [str(ROOT/"scripts/growbench_stock_lock.py"), *shared, "--single", name], check=True)
    with concurrent.futures.ThreadPoolExecutor(len(modes)) as workers:
        list(workers.map(run, modes))


if __name__ == "__main__":
    main()
