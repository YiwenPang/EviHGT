"""Final candidate rankings and explanations from the completed local model."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from html import escape
import json
import math
import os
from pathlib import Path

import polars as pl
import torch
import yaml

from .data.common import sha256
from .io import digest, write_json
from .metrics import SCORE_FIELDS, metric_pair
from .settings import ROOT
from .training.supervision import make_unlabeled_query_supervision


@dataclass(frozen=True)
class ReportSettings:
    explanation_queries: int = 10
    paths_per_query: int = 5
    ranking_top_k: int = 10

    def __post_init__(self):
        for name, minimum in (("explanation_queries", 0), ("paths_per_query", 1), ("ranking_top_k", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def load_report_settings(path=None):
    path = ROOT / "configs/report.yaml" if path is None else Path(path).expanduser().resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Report configuration must be a YAML mapping")
    return ReportSettings(**values)


def _sigmoid(value):
    if not math.isfinite(value):
        raise ValueError("Nonfinite saved evaluation score")
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1. + exponential)


def _scored(row, domain):
    support = _sigmoid(row["support_logit"])
    contra = _sigmoid(row["unsuitability_logit"])
    return dict(drug_id=row["drug_id"], disease_id=row["disease_id"], role=row["role"],
                source_domain=domain, support_logit=row["support_logit"],
                unsuitability_logit=row["unsuitability_logit"], support_score=support, contra_score=contra,
                m_T=support * (1-contra), m_F=(1-support) * contra,
                m_B=support * contra, m_U=(1-support) * (1-contra))


def _ranking_frame(rows):
    schema = {"disease_id": pl.String, "drug_id": pl.String, "rank": pl.Int64,
              "role": pl.String, "source_domain": pl.String}
    schema.update({name: pl.Float64 for name in ("support_logit", "unsuitability_logit",
                  "support_score", "contra_score", "m_T", "m_F", "m_B", "m_U")})
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def export_rankings(evaluation_dir, output_dir, *, top_k=10):
    """Use the exact P/U pools and score-descending, drug-ID tie rule of metrics."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("ranking top_k must be a positive integer")
    evaluation_dir, output_dir = Path(evaluation_dir), Path(output_dir)
    def read(name):
        path = evaluation_dir / f"{name}.parquet"
        return pl.read_parquet(path).to_dicts() if path.is_file() else []
    support, auxiliary = read("validation"), read("auxiliary_validation")
    saved = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    if any(row.get("role") != "P" for row in auxiliary):
        raise ValueError("Auxiliary evaluation candidates must be verified positives")
    actual = metric_pair(support, auxiliary)
    if actual != saved["metrics"]:
        raise ValueError("Saved evaluation predictions do not match their metrics/candidate identities")
    pools = {"support": [_scored(r, "validation") for r in support],
             "contra": [_scored(r, "auxiliary_validation") for r in auxiliary] +
                       [_scored(r, "validation") for r in support if r["role"] == "U"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings, metadata = {}, {}
    for head, rows in pools.items():
        ordered = sorted(rows, key=lambda r: (r["disease_id"], -r[SCORE_FIELDS[head]], r["drug_id"]))
        counts = {}
        for row in ordered:
            disease = row["disease_id"]
            counts[disease] = counts.get(disease, 0) + 1
            row["rank"] = counts[disease]
        frame = _ranking_frame(ordered)
        frame.write_csv(output_dir / f"{head}.csv")
        frame.filter(pl.col("rank") <= top_k).write_csv(output_dir / f"{head}_top{top_k}.csv")
        rankings[head] = ordered
        metadata[head] = dict(pairs=len(rows), diseases=len(counts),
            candidate_identity_sha256=actual[head].get("candidate_identity_sha256"),
            all_candidates=f"rankings/{head}.csv", top_candidates=f"rankings/{head}_top{top_k}.csv",
            tie_order="raw logit descending, canonical drug_id ascending, within each disease",
            score_semantics="Independent sigmoid scores are not calibrated probabilities")
    return dict(rows=rankings, metadata=metadata, metrics=saved["metrics"])


def select_explanation_queries(rankings, limit):
    """Visit rank first, then disease and head, with each canonical pair once."""
    candidates = []
    for head_index, head in enumerate(("support", "contra")):
        for row in rankings[head]:
            candidates.append((row["rank"], row["disease_id"], head_index, row["drug_id"], head, row))
    selected, seen = [], set()
    for _, _, _, _, head, row in sorted(candidates, key=lambda item: item[:4]):
        pair = row["drug_id"], row["disease_id"]
        if pair in seen:
            continue
        if len(selected) == limit:
            break
        seen.add(pair)
        selected.append(dict(row, selected_head=head))
    return selected


@contextmanager
def diagnostic_inference(built):
    """Expose existing fact logic while preserving parameters, modes and RNG."""
    from .runner import rng_state, restore_rng
    state = rng_state()
    modes = {name: module.training for name, module in built.model.named_modules()}
    original_config = built.model.encoder.config
    original_last = built.last
    try:
        built.model.eval()
        built.model.encoder.config = replace(original_config, return_logic_state=True)
        with torch.no_grad():
            yield
    finally:
        built.model.encoder.config = original_config
        built.last = original_last
        for name, module in built.model.named_modules():
            module.training = modes[name]
        restore_rng(state)


def _relative(path, root):
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return Path(path).resolve().as_uri()


def _text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _report_html(summary, index):
    def number(value):
        return "N/A" if value is None else f"{value:.6g}"
    metric_rows, ranking_rows = [], []
    for head in ("support", "contra"):
        result = summary["metrics"][head]
        metric_rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
            (head, number(result.get("macro_auc")), number(result.get("macro_ap")),
             number(result.get("macro_aupr_trapezoid")), result.get("evaluable_diseases", 0))) + "</tr>")
        for k in (5, 10):
            values = [head, k] + [number(result.get("ranking", {}).get(f"{key}@{k}"))
                                  for key in ("P", "R", "Hits", "NDCG")]
            ranking_rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
    links = []
    for head, item in summary["rankings"].items():
        links.append(f'<li>{head}: <a href="{escape(item["all_candidates"], quote=True)}">all candidates (CSV)</a> · '
                     f'<a href="{escape(item["top_candidates"], quote=True)}">Top-{summary["report_settings"]["ranking_top_k"]} per disease</a></li>')
    queries = []
    for item in index["queries"]:
        label = f'{item["drug_id"]} / {item["disease_id"]}'
        queries.append(f'<li><a href="{escape(item["path"], quote=True)}">{escape(label)}</a> '
                       f'({escape(item["selected_head"])} rank {item["rank"]})</li>')
    if not queries:
        queries.append("<li>No query explanation was requested or no development candidates were available.</li>")
    checkpoint = escape(summary["checkpoint"]["path"], quote=True)
    scope = escape(summary["scope"])
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EviHGT run results</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#17242d;background:#f4f7f8;margin:0}}main{{max-width:1050px;margin:36px auto;padding:32px;background:white;border-radius:12px}}h1{{margin:0;color:#123d4b}}h2{{margin-top:32px;font-size:22px}}.scope{{display:block;padding:12px;background:#e6f1f4;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{padding:10px;border-bottom:1px solid #dce5e8;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#006b85}}code{{overflow-wrap:anywhere}}small{{color:#4c626e}}@media(max-width:700px){{main{{margin:0;padding:18px;border-radius:0}}table{{font-size:13px}}td,th{{padding:6px}}}}</style>
<main><h1>EviHGT run results</h1><p class="scope">{scope}</p>
<p>{escape(summary["variant"])} · update {summary["step"]}/{summary["total_updates"]} · fixed final checkpoint</p>
<p><a href="summary.json">Machine-readable summary</a> · <a href="{checkpoint}">Checkpoint</a> ·
<a href="{escape(summary["evaluation_metrics"], quote=True)}">Full development metrics</a></p>
<h2>Development evaluation</h2><p>Recorded positives are compared with unlabeled candidates. Unlabeled does not mean a confirmed negative.</p>
<table><tr><th>Head</th><th>ROC-AUC</th><th>AP</th><th>AUPR</th><th>Diseases</th></tr>{''.join(metric_rows)}</table>
<table><tr><th>Head</th><th>K</th><th>Precision</th><th>Recall</th><th>Hits</th><th>NDCG</th></tr>{''.join(ranking_rows)}</table>
<small>N/A means unavailable. AP uses threshold steps; AUPR uses trapezoids.</small>
<h2>Drug rankings</h2><p>Each disease is ranked within its development candidate pool. Ties use canonical drug ID order.</p><ul>{''.join(links)}</ul>
<h2>Forward-path explanations</h2><p>Examples visit within-disease rank first, then disease ID and head, with each pair once. The output budget is {summary["report_settings"]["explanation_queries"]} queries and {summary["report_settings"]["paths_per_query"]} paths per query.</p>
<ul>{''.join(queries)}</ul><p><a href="explanations/index.json">Explanation index</a></p>
<p>Paths, their weights and fact logic describe the model forward computation. Independent sigmoid scores are uncalibrated; path weights are not causal effects. Missing paths and unavailable logic remain explicit in the JSON files.</p>
<small>Corpus identity: <code>{escape(summary["corpus_identity"])}</code><br>Checkpoint SHA-256: <code>{summary["checkpoint"]["sha256"]}</code></small></main></html>'''


def create_report(built, corpus, settings, console, root, step, checkpoint, report_settings=None):
    """Finalize rankings, bounded path diagnostics and the local result page."""
    from .explanations.portable import build_query_explanation
    from .runner import precision_context
    config = ReportSettings() if report_settings is None else report_settings
    if not isinstance(config, ReportSettings):
        raise TypeError("report_settings must be ReportSettings")
    root, checkpoint = Path(root), Path(checkpoint).resolve(strict=True)
    evaluation_dir = root / f"evaluation/step_{step}"
    saved = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    if saved["corpus_identity"] != corpus.identity or saved["step"] != step:
        raise ValueError("Final evaluation does not belong to this corpus and update")
    checkpoint_sha256 = sha256(checkpoint)
    ranking = export_rankings(evaluation_dir, root / "rankings", top_k=config.ranking_top_k)
    selected = select_explanation_queries(ranking["rows"], config.explanation_queries)
    target = root / "explanations"
    (target / "queries").mkdir(parents=True, exist_ok=True)
    index = dict(schema="EVIHGT_EXPLANATION_INDEX_V1", corpus_identity=corpus.identity,
                 checkpoint_sha256=checkpoint_sha256, step=step,
                 status="DISABLED" if config.explanation_queries == 0 else "COMPLETE",
                 selection="within-disease rank, disease ID, head (support then contra), drug ID; unique pairs",
                 settings=asdict(config), queries=[])
    with diagnostic_inference(built), precision_context(settings):
        with console.progress(len(selected), "导出路径解释", unit="query") as progress:
            for i, row in enumerate(selected, 1):
                query = make_unlabeled_query_supervision(row["drug_id"], row["disease_id"])
                graph = corpus.assembler.assemble(query, domain=row["source_domain"], device=settings.device)
                prediction = built.forward(graph.batch.model_batch, path_batch=graph.batch.path_batch)
                item = build_query_explanation(corpus, graph, prediction,
                    drug_id=row["drug_id"], disease_id=row["disease_id"],
                    domain=row["source_domain"], top_k=config.paths_per_query)
                item.update(checkpoint_sha256=checkpoint_sha256, step=step, corpus_identity=corpus.identity,
                            ranking_selection={key: row[key] for key in ("selected_head", "rank", "role")},
                            saved_development_logits={key: row[key] for key in SCORE_FIELDS.values()})
                name = f"queries/query_{i:04d}_{digest([row['drug_id'], row['disease_id']])[:12]}.json"
                write_json(target / name, item)
                index["queries"].append(dict(drug_id=row["drug_id"], disease_id=row["disease_id"],
                    source_domain=row["source_domain"], selected_head=row["selected_head"], rank=row["rank"],
                    path="explanations/" + name))
                progress.update()
    if sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Final checkpoint changed while exporting results")
    write_json(target / "index.json", index)
    synthetic = corpus.manifest.get("purpose") == "SYNTHETIC_TEST_ONLY"
    sources = {str(p.relative_to(ROOT).as_posix()): sha256(p)
               for p in sorted((ROOT / "src/evi_hgt").rglob("*.py"))}
    summary = dict(schema="EVIHGT_RUN_SUMMARY_V1", status="COMPLETE", variant=built.variant,
        step=step, total_updates=corpus.plan.total_steps, corpus_identity=corpus.identity,
        scope="SYNTHETIC_TEST_ONLY" if synthetic else "LOCAL_DEVELOPMENT_PANEL",
        checkpoint=dict(path=_relative(checkpoint, root), sha256=checkpoint_sha256),
        source_identity_sha256=digest(sources), source_files=sources,
        report_settings=asdict(config), metrics=ranking["metrics"], rankings=ranking["metadata"],
        evaluation_metrics=f"evaluation/step_{step}/metrics.json", explanation_index="explanations/index.json",
        explained_queries=len(index["queries"]), report="report.html")
    _text(root / "report.html", _report_html(summary, index))
    write_json(root / "summary.json", summary)
    console.say(f"[结果] {root / 'report.html'}")
    return summary
