"""Names used in the original study manifests, receipts and result directories."""

from pathlib import Path

from ablation_artifacts import FEATURE_DATA, FEATURE_MANIFEST_NAME, FEATURE_OUTPUT


# Feature baseline used to establish the first fold's schedule and initial tensors.
FEATURE_SCHEDULE = Path(FEATURE_DATA) / "schedule_v1/schedule.json"
FEATURE_PAIRING = Path(FEATURE_DATA) / "conditional_v1/plan.json"
FEATURE_REVIEW = "DEV94_REVIEW.json"
FEATURE_LIMITATIONS_FIELD = "DEV94_limitations"

# Single-fold comparison of the two training losses.
LOSS_OUTPUT = Path("outputs/phase9/phase95")
LOSS_SOURCE_DIRS = ("src/evi_hgt_dev95", "scripts/phase95", "tests/phase95")
LOSS_MANIFEST = "DEV95_EXECUTION_MANIFEST.json"
LOSS_PLAN = "DEV95_PLAN.md"
LOSS_JSON = "DEV95_RESULTS.json"
LOSS_MARKDOWN = "DEV95_RESULTS.md"
LOSS_SCHEMA = "DEV95_TWO_ARM_DEVELOPMENT_V1"
LOSS_COMPLETE = "DEV95_COMPLETE_VERIFIED_DEVELOPMENT"
LOSS_INITIAL_TEST_NOTE = "RED: missing DEV95 modules before implementation, no updates"

# Cross-fold comparison with paired training streams.
PAIRED_OUTPUT = Path("outputs/phase9/phase96")
PAIRED_CACHE = Path("data/cache/phase96")
PAIRED_SOURCE_DIRS = ("src/evi_hgt_dev96", "scripts/phase96", "tests/phase96")
PAIRED_MANIFEST = "DEV96_EXECUTION_MANIFEST.json"
PAIRED_PLAN = "DEV96_PLAN.md"
PAIRED_JSON = "DEV96_RESULTS.json"
PAIRED_MARKDOWN = "DEV96_RESULTS.md"
PAIRED_SCHEMA = "DEV96_CROSS_CELL_PAIRED_DEVELOPMENT_V1"
PAIRED_AUTHORIZATION = "DEV96_CROSS_CELL_PAIRED_DEVELOPMENT"
PAIRED_COMPLETE = "DEV96_COMPLETE_VERIFIED_DEVELOPMENT"
PAIRED_PANEL_NOTE = "Hash DEV96_TRAIN_PANEL_V1 current-cell global32/conditional32; fewer than32 all kept; step0 global only"
PAIRED_INITIAL_TEST_NOTE = "RED: DEV96 cell/sampling/runner modules and canonical graph gate absent before implementation."
REFERENCE_LOSS_CHECK_FIELD = "loss_exact_DEV95_on_same_raw_predictions"

# Evidence interventions and their cached full-model references.
EVIDENCE_OUTPUT = Path("outputs/phase9/phase97")
EVIDENCE_SOURCE_DIRS = ("src/evi_hgt_dev97", "scripts/phase97", "tests/phase97")
EVIDENCE_MANIFEST = "DEV97_EXECUTION_MANIFEST.json"
EVIDENCE_PLAN = "DEV97_PLAN.md"
EVIDENCE_CONSOLE_RULES = Path("docs/DEV97_CONSOLE_RULES.md")
EVIDENCE_JSON = "DEV97_RESULTS.json"
EVIDENCE_MARKDOWN = "DEV97_RESULTS.md"
EVIDENCE_SCHEMA = "DEV97_CORE_ABLATION_DEVELOPMENT_V1"
EVIDENCE_AUTHORIZATION = "DEV97_CORE_ABLATION_DEVELOPMENT"
EVIDENCE_COMPLETE = "DEV97_COMPLETE_VERIFIED_DEVELOPMENT"
REFERENCE_PLAN_SOURCE = "READ_ONLY_ACTUAL_DEV96_E0_G_SHARED_PLAN_EXTERNAL_TO_ABLATION_WEIGHTS"
REFERENCE_TRAINING_FIELD = "reference_training_updates_charged_to_DEV97"
REFERENCE_EVALUATION_FIELD = "reference_evaluation_forwards_charged_to_DEV97"
PAIRED_HISTORY_FIELD = "historical_DEV96"
PAIRED_REFERENCE_FIELD = "DEV96"


def display_status(status):
    """Use one completion label in human-readable reports across all three studies."""
    if status in (LOSS_COMPLETE, PAIRED_COMPLETE, EVIDENCE_COMPLETE):
        return "COMPLETE_VERIFIED_DEVELOPMENT"
    return status
