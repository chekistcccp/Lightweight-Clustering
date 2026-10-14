"""Fail closed if a Git commit stages research output.

For opt-in local protection:
    git config core.hooksPath .githooks
This check never uploads results and requires no third-party modules.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED_PARTS = {"data", "model", "models", "outputs", "predictions", "cache", "results"}
BLOCKED_EXACT = {"experiment_results.json", "per_case.csv",
                 "torc_diagnostics.json", "run_manifest.json", "summary.json",
                 "split.json", "golden.mat"}
BLOCKED_SUFFIXES = (".nii", ".nii.gz", ".npz", ".npy", ".safetensors",
                    ".pth", ".pt", ".tar", ".tar.gz", ".zip")


def blocked(path: str) -> bool:
    norm=path.replace("\\", "/").lower()
    parts=PurePosixPath(norm).parts
    name=PurePosixPath(norm).name
    if set(parts) & BLOCKED_PARTS or name in BLOCKED_EXACT:
        return True
    return (name.startswith(("experiment_results", "research_results"))
            or name.endswith(BLOCKED_SUFFIXES))


def staged_paths():
    p=subprocess.run(["git","diff","--cached","--name-only","-z"],
                     capture_output=True,check=True)
    return [b.decode("utf-8","replace") for b in p.stdout.split(b"\0") if b]


def main():
    try:
        bad=[p for p in staged_paths() if blocked(p)]
    except (OSError,subprocess.CalledProcessError) as e:
        print("[result-guard] cannot inspect staged Git files:",e,file=sys.stderr)
        return 2
    if bad:
        print("[result-guard] REFUSING COMMIT: experimental results/data/model files staged:",
              *bad,sep="\n  ",file=sys.stderr)
        print("Use git restore --staged <path> and keep results local only.",file=sys.stderr)
        return 1
    print("[result-guard] staged changes contain no restricted research outputs")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
