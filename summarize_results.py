import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_METHOD_ORDER = ["Random", "ClassMean", "PCA", "KMeans", "Gabor"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarise experiment CSV files into comparison tables with "
            "best accuracy and convergence-speed metrics."
        )
    )
    parser.add_argument("--plot-dir", default="plot_data", help="Directory containing seed=*/experiment/*.csv files.")
    parser.add_argument("--out-dir", default="summary_tables", help="Directory for generated summary CSV files.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Random", "ClassMean", "PCA", "KMeans"],
        help="Methods to include in the main comparison tables.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["MNIST_2layer", "Fashion_2layer", "SIGN_2layer", "CIFAR10_2layer"],
        help=(
            "Experiment folders to include. Use 'all' to scan every experiment "
            "folder under plot_data/seed=*/."
        ),
    )
    parser.add_argument(
        "--include-gabor",
        action="store_true",
        help="Also include Gabor rows when they exist.",
    )
    return parser.parse_args()


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_curve(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(float(row.get("Epoch", len(rows) + 1)))
            loss = as_float(row.get("Training Loss"))
            acc = as_float(row.get("Test Accuracy"))
            if not math.isnan(acc):
                rows.append({"epoch": epoch, "loss": loss, "accuracy": acc})
    return rows


def first_epoch_at(curve, threshold):
    for point in curve:
        if point["accuracy"] >= threshold:
            return point["epoch"]
    return ""


def summarise_curve(csv_path, seed, experiment, method):
    curve = read_curve(csv_path)
    if not curve:
        return None

    best_point = max(curve, key=lambda point: (point["accuracy"], -point["epoch"]))
    best_accuracy = best_point["accuracy"]
    best_epoch = best_point["epoch"]
    final_accuracy = curve[-1]["accuracy"]

    return {
        "experiment": experiment,
        "seed": seed,
        "method": method,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_accuracy": final_accuracy,
        "last_epoch": curve[-1]["epoch"],
        "epoch_to_90pct_best": first_epoch_at(curve, best_accuracy * 0.90),
        "epoch_to_95pct_best": first_epoch_at(curve, best_accuracy * 0.95),
        "epoch_to_97pct_best": first_epoch_at(curve, best_accuracy * 0.97),
        "epoch_to_abs_95": first_epoch_at(curve, 95.0),
        "epoch_to_abs_97": first_epoch_at(curve, 97.0),
        "epoch_to_abs_98": first_epoch_at(curve, 98.0),
        "source_csv": str(csv_path),
    }


def method_sort_key(method):
    try:
        return DEFAULT_METHOD_ORDER.index(method)
    except ValueError:
        return len(DEFAULT_METHOD_ORDER), method


def discover_runs(plot_dir, allowed_methods, allowed_experiments):
    summaries = []
    for csv_path in sorted(plot_dir.glob("seed=*/*/*.csv")):
        method = csv_path.stem
        if method not in allowed_methods:
            continue

        seed = csv_path.parents[1].name.replace("seed=", "")
        experiment = csv_path.parent.name
        if allowed_experiments is not None and experiment not in allowed_experiments:
            continue
        summary = summarise_curve(csv_path, seed, experiment, method)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=lambda row: (row["experiment"], int(row["seed"]), method_sort_key(row["method"])))
    return summaries


def mean(values):
    values = [float(value) for value in values if value != ""]
    return sum(values) / len(values) if values else ""


def std(values):
    values = [float(value) for value in values if value != ""]
    if len(values) < 2:
        return ""
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate_runs(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["experiment"], row["method"])].append(row)

    aggregate_rows = []
    for (experiment, method), rows in grouped.items():
        aggregate_rows.append(
            {
                "experiment": experiment,
                "method": method,
                "runs": len(rows),
                "mean_best_accuracy": mean(row["best_accuracy"] for row in rows),
                "std_best_accuracy": std(row["best_accuracy"] for row in rows),
                "mean_best_epoch": mean(row["best_epoch"] for row in rows),
                "mean_epoch_to_95pct_best": mean(row["epoch_to_95pct_best"] for row in rows),
                "mean_epoch_to_97pct_best": mean(row["epoch_to_97pct_best"] for row in rows),
                "mean_epoch_to_abs_95": mean(row["epoch_to_abs_95"] for row in rows),
                "mean_epoch_to_abs_97": mean(row["epoch_to_abs_97"] for row in rows),
                "mean_epoch_to_abs_98": mean(row["epoch_to_abs_98"] for row in rows),
            }
        )

    aggregate_rows.sort(key=lambda row: (row["experiment"], method_sort_key(row["method"])))
    return aggregate_rows


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def main():
    args = parse_args()
    plot_dir = Path(args.plot_dir)
    out_dir = Path(args.out_dir)

    allowed_methods = list(args.methods)
    if args.include_gabor and "Gabor" not in allowed_methods:
        allowed_methods.append("Gabor")

    allowed_experiments = None if args.experiments == ["all"] else set(args.experiments)
    run_rows = discover_runs(plot_dir, set(allowed_methods), allowed_experiments)
    if not run_rows:
        raise SystemExit(f"No matching CSV files found under {plot_dir.resolve()}")

    run_fields = [
        "experiment",
        "seed",
        "method",
        "best_accuracy",
        "best_epoch",
        "final_accuracy",
        "last_epoch",
        "epoch_to_90pct_best",
        "epoch_to_95pct_best",
        "epoch_to_97pct_best",
        "epoch_to_abs_95",
        "epoch_to_abs_97",
        "epoch_to_abs_98",
        "source_csv",
    ]
    write_csv(out_dir / "all_runs_summary.csv", run_rows, run_fields)

    aggregate_rows = aggregate_runs(run_rows)
    aggregate_fields = [
        "experiment",
        "method",
        "runs",
        "mean_best_accuracy",
        "std_best_accuracy",
        "mean_best_epoch",
        "mean_epoch_to_95pct_best",
        "mean_epoch_to_97pct_best",
        "mean_epoch_to_abs_95",
        "mean_epoch_to_abs_97",
        "mean_epoch_to_abs_98",
    ]
    write_csv(out_dir / "aggregate_summary.csv", aggregate_rows, aggregate_fields)

    for experiment in sorted({row["experiment"] for row in run_rows}):
        experiment_rows = [row for row in run_rows if row["experiment"] == experiment]
        write_csv(out_dir / f"{experiment}_runs.csv", experiment_rows, run_fields)

    print(f"Wrote {len(run_rows)} run summaries to {out_dir.resolve()}")
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {(out_dir / 'aggregate_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
