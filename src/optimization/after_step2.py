import os
import json
import glob
import argparse

def find_best_run(root_dir, metrics_filename, metric_key):
    best = {"value": float("inf"), "path": None}
    pattern = os.path.join(root_dir, "*", metrics_filename)

    for metrics_file in glob.glob(pattern):
        run_dir = os.path.dirname(metrics_file)
        try:
            with open(metrics_file, "r") as f:
                m = json.load(f)
            # metric must be a list of per-epoch values
            vals = m.get(metric_key)
            if not isinstance(vals, list) or not vals:
                continue
            final = vals[-1]
            if final < best["value"]:
                best["value"] = final
                best["path"] = run_dir
        except Exception:
            # skip unreadable or malformed files
            continue

    return best

def main():
    p = argparse.ArgumentParser(
        description="Find best Step‑2 run by final metric")
    p.add_argument("root",
                   help="Root directory containing all Step‑2 run subfolders")
    p.add_argument("--metric-file", default="eval_metrics.json",
                   help="Filename (in each run dir) containing the metrics JSON")
    p.add_argument("--metric-key", default="fid",
                   help="Which key in the JSON to compare (must be a list)")
    args = p.parse_args()

    best = find_best_run(args.root, args.metric_file, args.metric_key)
    if best["path"] is None:
        print("No valid runs found under", args.root)
        exit(1)

    print(f"Best run → {best['path']}")
    print(f"Final {args.metric_key} = {best['value']:.4f}")

if __name__ == "__main__":
    main()
