"""Command-line and IDE entry points for the same local workflow."""
import argparse
import sys
import traceback

from .model import VARIANTS
from .runner import run
from .settings import ROOT, load_settings
from .preprocessing.pipeline import prepare_raw


def startup_failure(exc):
    print(f"[Error] Workflow initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        path = ROOT / "outputs/logs/startup_errors.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as handle:
            traceback.print_exception(exc, file=handle)
        print(f"Full traceback: {path.resolve()}", file=sys.stderr)
    except OSError:
        print("The startup traceback could not be written; check output-directory permissions.", file=sys.stderr)


def main(*, ablation=False):
    parser = argparse.ArgumentParser(description="EviHGT: independent support/contra drug-disease ranking")
    parser.add_argument("--config", help="YAML settings; defaults to configs/default.yaml")
    parser.add_argument("--mode", choices=("train", "prepare", "validate", "evaluate"), default="train")
    parser.add_argument("--raw-config", help="Raw database YAML; defaults to configs/raw.yaml")
    parser.add_argument("--offline", action="store_true", help="Use verified auxiliary caches without downloading")
    parser.add_argument("--checkpoint", help="Trusted local checkpoint for resume or development evaluation")
    if ablation:
        parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=[v for v in VARIANTS if v != "FULL"])
    args = parser.parse_args()
    variants = args.variants if ablation else ["FULL"]
    if args.checkpoint and len(variants) != 1:
        parser.error("A checkpoint requires exactly one variant")
    if args.checkpoint and args.mode == "prepare":
        parser.error("Preparation does not accept a checkpoint")
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        startup_failure(exc)
        return 2
    try:
        if args.mode == "prepare" or (args.mode == "train" and not args.checkpoint
                                      and not settings.resolve(settings.manifest).is_file()):
            prepare_raw(args.raw_config, manifest_path=settings.resolve(settings.manifest),
                        output_dir=settings.resolve(settings.output_dir), offline=args.offline)
        if args.mode == "prepare":
            return 0
    except Exception as exc:
        if not getattr(exc, "evihgt_reported", False):
            startup_failure(exc)
        return 1
    for variant in variants:
        try:
            run(settings, variant=variant, mode=args.mode, checkpoint=args.checkpoint)
        except Exception as exc:
            if not getattr(exc, "evihgt_reported", False):
                startup_failure(exc)
            return 1
    return 0
