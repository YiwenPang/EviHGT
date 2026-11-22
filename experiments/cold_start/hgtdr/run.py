"""Run the HGTDR comparison on the prepared strict cold-disease split."""

import argparse
import os
from pathlib import Path
import traceback


def print_summary():
    report = Path(__file__).with_name("results.json")
    print(report.read_text(encoding="utf-8"), end="")


def _local_directory(parser, name, path, parent):
    path = path.resolve()
    if not path.is_relative_to(parent.resolve()) or path == parent.resolve():
        parser.error(f"{name} must be a subdirectory of {parent}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("summary", help="display the recorded aggregate results")
    descriptions = {
        "validate": "check the prepared training and development inputs",
        "train": "fit and select both HGTDR models on development candidates",
        "score": "score frozen candidates with both selected models",
        "run": "train both models and score the common test candidates",
    }
    for mode, description in descriptions.items():
        command = commands.add_parser(mode, help=description)
        command.add_argument("--research-root", type=Path, required=True,
                             help="EviHGT research workspace with the native HGTDR adapter")
        command.add_argument("--corpus", type=Path, required=True,
                             help="prepared cold_disease directory with manifest.json")
        command.add_argument("--cache", type=Path,
                             help="derived graph directory under the research data/cache directory")
        if mode != "validate":
            command.add_argument("--output", type=Path,
                                 help="run directory under the research outputs directory")
        if mode in ("score", "run"):
            command.add_argument("--reference-run", type=Path, required=True,
                                 help="completed EviHGT cold-disease run directory")
    args = parser.parse_args(argv)
    if args.mode == "summary":
        print_summary()
        return 0
    research = args.research_root.resolve()
    corpus = args.corpus.resolve()
    if not (corpus / "manifest.json").is_file():
        parser.error(f"Missing prepared cold-disease corpus at {corpus}")
    cache_root = research / "data/cache"
    cache = _local_directory(parser, "--cache", args.cache or cache_root / "cold_disease_hgtdr", cache_root)
    output_root = research / "outputs"
    output = (
        _local_directory(parser, "--output", args.output or output_root / "cold_disease_hgtdr", output_root)
        if args.mode != "validate" else None
    )
    reference = getattr(args, "reference_run", None)
    if reference is not None:
        reference = reference.resolve()
        if not (reference / "result.json").is_file():
            parser.error(f"Missing EviHGT cold-disease result at {reference}")
        if output == reference or output.is_relative_to(reference) or reference.is_relative_to(output):
            parser.error("Choose a run output separate from the EviHGT reference run")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        from experiments.hgtdr_contra.run import add_research_source

        add_research_source(research)
        from .runtime import execute

        execute(args.mode, corpus, cache, output, reference)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        parser.exit(1, f"{error}\nSee the cold-start guide for the input and native runtime requirements.\n")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
