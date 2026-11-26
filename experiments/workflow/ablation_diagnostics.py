"""Measure fact and bounded-path exposure on fixed training groups."""
import csv
from dataclasses import asdict
import os

import torch

from evi_hgt.io import digest, write_json
from evi_hgt.training.supervision import make_unlabeled_query_supervision


def _selection(plan, cell):
    if cell == "R0F0":
        namespace = "DEV95_TRAIN_PANEL_V1"
        prefix = [namespace]
    elif cell in ("R0F1", "R1F0"):
        namespace = "DEV96_TRAIN_PANEL_V1"
        prefix = [namespace, cell]
    else:
        namespace = "EVIHGT_PORTABLE_TRAIN_PANEL_V1"
        prefix = [namespace, cell]

    def order(step):
        return digest([*prefix, [asdict(query) for query in plan.step(step).queries]])

    return dict(namespace=namespace, order="Original supervision order; first 32 hashes per stage",
        global_steps=sorted(range(1, plan.G + 1), key=order)[:32],
        conditional_steps=sorted(range(plan.G + 1, plan.total_steps + 1), key=order)[:32])


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def fixed_panel_exposure(corpus):
    """Collect the structural exposure without forwarding a model or reading scores."""
    cell = corpus.manifest.get("source_cell")
    selection = _selection(corpus.plan, cell)
    steps = sorted(selection["global_steps"] + selection["conditional_steps"])
    counts = dict(groups=len(steps), query_occurrences=0, facts=0, available_facts=0,
                  query_has_available=0, path_count=0, path_steps=0, available_path_steps=0,
                  queries_without_paths=0)
    witnesses = []
    for step in steps:
        spec = corpus.plan.step(step)
        witnesses.append(dict(step=step, training_member_sha256=spec.identity["member_digest"],
            record_sha256=None if spec.record is None else digest(spec.record)))
        extra = () if spec.record is None else tuple(
            make_unlabeled_query_supervision(*pair) for pair in spec.record["extra_pairs"])
        for query in (*spec.queries, *extra):
            graph = corpus.assembler.assemble(query, domain="training", device="cpu")
            evidence = graph.batch.model_batch.evidence
            paths = graph.batch.path_batch
            available = evidence.fact_support_available & evidence.fact_counter_available
            counts["query_occurrences"] += 1
            counts["facts"] += evidence.num_facts
            counts["available_facts"] += int(available.sum())
            counts["query_has_available"] += int(available.any())
            counts["path_count"] += paths.num_paths
            counts["path_steps"] += int(paths.step_mask.sum())
            counts["available_path_steps"] += int(available[paths.step_fact_index[paths.step_mask]].sum())
            counts["queries_without_paths"] += int(paths.num_paths == 0)
    fractions = dict(
        fact_availability_fraction=_fraction(counts["available_facts"], counts["facts"]),
        query_availability_fraction=_fraction(counts["query_has_available"], counts["query_occurrences"]),
        no_path_query_fraction=_fraction(counts["queries_without_paths"], counts["query_occurrences"]),
        path_step_availability_fraction=_fraction(counts["available_path_steps"], counts["path_steps"]),
    )
    return dict(schema="EVIHGT_FIXED_TRAINING_EXPOSURE_V1", cell=cell, corpus_identity=corpus.identity,
        plan_identity=corpus.plan.identity, counts=counts, fractions=fractions, selection=selection,
        selected_group_bindings=witnesses, model_forwards=0, optimizer_updates=0,
        denominator="Per-query input graph slots, extracted candidate path steps and query occurrences",
        scope="SYNTHETIC_TEST_ONLY" if corpus.manifest.get("purpose") == "SYNTHETIC_TEST_ONLY"
        else "LOCAL_FIXED_TRAINING_DIAGNOSTIC")


def write_exposure(corpus, directory, *, step, variant, loss):
    data = {**fixed_panel_exposure(corpus), "step": step, "variant": variant, "loss": loss,
            "model_path_reasoning_enabled": variant != "NO_PATH"}
    write_json(directory / "exposure.json", data)
    row = {key: data[key] for key in ("cell", "step", "variant", "loss", "scope", "denominator")}
    row.update(data["counts"])
    row.update(data["fractions"])
    temporary = directory / "exposure.csv.partial"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, directory / "exposure.csv")
    return data
