import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "mnist_1layer": "MNIST_1layer.py",
    "mnist_2layer": "MNIST_2layer.py",
    "fashion_1layer": "FashionMNIST_1layer.py",
    "fashion_2layer": "FashionMNIST_2layer.py",
    "sign_1layer": "Sign_1layer.py",
    "sign_2layer": "Sign_2layer.py",
    "cifar10_2layer": "CIFAR_10_2layer.py",
}

METHODS = {
    "Random": 0,
    "ClassMean": 1,
    "PCA": 2,
    "KMeans": 3,
    "Gabor": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the experiment matrix requested for comparison tables: "
            "dataset x method x seed."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["mnist_2layer", "fashion_2layer", "sign_2layer", "cifar10_2layer"],
        choices=sorted(EXPERIMENTS),
        help="Experiment scripts to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Random", "ClassMean", "PCA", "KMeans"],
        choices=sorted(METHODS),
        help="Initialisation methods to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to run.")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs passed to each training script.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    commands = []
    for experiment in args.experiments:
        script = root / EXPERIMENTS[experiment]
        if not script.exists():
            raise SystemExit(f"Missing script: {script}")
        for seed in args.seeds:
            for method in args.methods:
                commands.append(
                    [
                        sys.executable,
                        str(script),
                        "-s",
                        str(seed),
                        "-t",
                        str(METHODS[method]),
                        "--epochs",
                        str(args.epochs),
                    ]
                )

    for command in commands:
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=root, check=True)

    print(f"Prepared {len(commands)} experiment commands.")


if __name__ == "__main__":
    main()
