"""Run preprocessing, model experiments, and figure generation in order."""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="use the experiment script's smoke-test settings")
    parser.add_argument("--models", nargs="*", help="optional subset of installed model names")
    parser.add_argument("--feature-sets", nargs="*", help="optional subset of feature-set names")
    parser.add_argument("--n-trials", type=int, help="hyperparameter trials per model")
    parser.add_argument("--seeds", type=int, help="random seeds per stochastic model")
    parser.add_argument("--n-boot", type=int, help="bootstrap resamples for confidence intervals")
    parser.add_argument("--skip-risk-map", action="store_true", help="skip the slower spatial risk-map analysis")
    parser.add_argument("--skip-importance", action="store_true", help="skip permutation-importance calculations")
    return parser.parse_args()


def main():
    args = parse_args()
    commands = [
        [sys.executable, "-m", "crash_severity.preprocessing"],
        [sys.executable, "-m", "crash_severity.experiments"],
        [sys.executable, "-m", "crash_severity.figures"],
    ]
    if args.fast:
        commands[1].append("--fast")
    if args.models:
        commands[1].extend(["--models", *args.models])
    if args.feature_sets:
        commands[1].extend(["--feature-sets", *args.feature_sets])
    for option, value in (("--n-trials", args.n_trials), ("--seeds", args.seeds), ("--n-boot", args.n_boot)):
        if value is not None:
            commands[1].extend([option, str(value)])
    if args.skip_risk_map:
        commands[1].append("--skip-risk-map")
    if args.skip_importance:
        commands[1].append("--skip-importance")

    root = Path(__file__).resolve().parents[1]
    for stage, command in zip(("preprocessing and EDA", "model experiments", "figure generation"), commands):
        print(f"\n=== {stage} ===", flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
