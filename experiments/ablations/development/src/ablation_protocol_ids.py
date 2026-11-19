"""Identifiers in the original sampling plans, feature metadata and state dictionaries."""


FUSION_LAYER_NAME = "dev94_fusion"
FEATURE_SCHEMA = "DEV94_FROZEN_DRG_DIS_FEATURES_V1"
FIRST_FOLD_ORIGIN_USE = "EXOGENOUS_REGISTERED_DEV94_PLAN_NO_WEIGHT_SCORE_OR_TEACHER"
SIDE_INITIALIZATION_NAMESPACE = "PHASE852B_SIDE_INITIALIZATION_V1"
PAIRED_SCHEDULE_SCHEMA = "DEV96_THREE_FULL_CYCLES_V1"
PAIRED_PLAN_SCHEMA = "DEV96_SHARED_CONDITIONAL_V1"
PAIRED_NAMESPACE_FIELD = "dev96_namespace"
PAIRED_NAMESPACE = "DEV96_E0_G_SHARED_CONDITIONAL_V1"
EVIDENCE_INTERVENTION_FIELD = "dev97_ablation"
CONTENT_INTERVENTION_BOUNDARY = "PHASE92_M2"
QUALITY_INTERVENTION_BOUNDARY = "PHASE92_M3"


def paired_sampling_bindings(base, cell, schedule):
    return {
        **base,
        PAIRED_NAMESPACE_FIELD: PAIRED_NAMESPACE,
        "cell": cell,
        "corpus_digest": schedule["corpus_digest"],
        "schedule_digest": schedule["digest"],
        "cycles": [2, 3],
        "record_order": "cycle_then_original_group",
    }


def side_initialization_seed(training_seed, digest):
    return int(digest([SIDE_INITIALIZATION_NAMESPACE, training_seed])[:8], 16)


def conditional_draw_seed(bindings, group_index, disease, digest):
    return int(digest(["PART3A_R2_CONDITIONAL_UNIFORM_V1", bindings, group_index, disease])[:16], 16)
