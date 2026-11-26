"""Original deterministic membership primitives, extracted without execution receipts."""
from __future__ import annotations
from collections import defaultdict
import hashlib, json, math
from typing import Any, Iterable, Mapping
import polars as pl
class CVContractError(RuntimeError):
    """Stop the entire future chain on an invariant violation."""

def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def pair_key(drug: str, disease: str) -> str:
    if any(not isinstance(v, str) or not v.strip() for v in (drug, disease)):
        raise CVContractError("pair endpoints must be nonempty canonical strings")
    return json.dumps([drug, disease], ensure_ascii=True, separators=(",", ":"))

def _pair(row: Mapping[str, Any]) -> tuple[str, str]:
    try:
        pair = row["drug_id"], row["disease_id"]
        pair_key(*pair)
    except (KeyError, TypeError) as exc:
        raise CVContractError("pair row missing valid endpoints") from exc
    return pair

def _panel(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean, seen = [], set()
    for row in rows:
        pair = _pair(row)
        if row.get("role") not in {"P", "U"}:
            raise CVContractError("panel role must be P or U; U is not a verified negative")
        if "label" in row and (row["label"] is not None if row["role"] == "U" else row["label"] != 1):
            raise CVContractError("structured label conflicts with P/U semantics; U label must be absent or null")
        if pair in seen:
            raise CVContractError("duplicate pair or conflicting roles in panel")
        seen.add(pair)
        clean.append(dict(drug_id=pair[0], disease_id=pair[1], role=row["role"]))
    if not clean:
        raise CVContractError("empty development panel")
    return sorted(clean, key=lambda r: _pair(r))

def _bound(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, digest=digest(payload))

def _verify(payload: Mapping[str, Any]) -> None:
    if payload.get("digest") != digest({k: v for k, v in payload.items() if k != "digest"}):
        raise CVContractError("manifest digest mismatch")

def make_repeat_keys(max_repeats: int = 5, *, namespace: str = "phase85-v1") -> tuple[str, ...]:
    if not isinstance(max_repeats, int) or not 1 <= max_repeats <= 5 or not namespace:
        raise CVContractError("Part 1 permits one through five pre-frozen repeat keys")
    return tuple(digest({"namespace": namespace, "repeat": r}) for r in range(max_repeats))

def assign_folds(panel: Iterable[Mapping[str, Any]], y: int, repeat_key: str) -> dict[str, Any]:
    if not isinstance(y, int) or y not in {3, 5, 10} or not repeat_key:
        raise CVContractError("outer folds must use frozen ladder 3, 5, 10")
    rows = _panel(panel)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["disease_id"], row["role"]].append(row)
    assigned = []
    for (disease, role), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: digest([repeat_key, "outer", *_pair(r)]))
        offset = int(digest([repeat_key, disease, role, "offset"]), 16) % y
        assigned.extend(dict(row, fold=(index + offset) % y) for index, row in enumerate(ordered))
    return _bound(dict(schema="PHASE85_PU_FOLD_MANIFEST_V1", y=y, repeat_key=repeat_key,
                       panel_digest=digest(rows), assignments=sorted(assigned, key=_pair),
                       stratification="DISEASE_AND_P_OR_U", unlabeled_is_negative=False))

def _verified_folds(panel: Iterable[Mapping[str, Any]], folds: Mapping[str, Any]) -> list[dict[str, Any]]:
    _verify(folds)
    rows = _panel(panel)
    if folds.get("panel_digest") != digest(rows):
        raise CVContractError("development panel digest mismatch")
    rebuilt = assign_folds(rows, folds["y"], folds["repeat_key"])
    if rebuilt != folds:
        raise CVContractError("fold manifest does not match deterministic construction")
    return rows

def nested_subsets(rows: Iterable[Mapping[str, Any]], fractions: Iterable[float] = (.5, .7, .85, 1.),
                   *, key: str) -> dict[float, tuple[dict[str, Any], ...]]:
    values = tuple(float(value) for value in fractions)
    if not values or any(not math.isfinite(v) or not 0 < v <= 1 for v in values):
        raise CVContractError("data fractions must be finite in (0,1]")
    clean = _panel(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        groups[row["disease_id"], row["role"]].append(row)
    ordered = [sorted(group, key=lambda r: digest([key, "nested", *_pair(r)]))
               for _, group in sorted(groups.items())]
    result = {}
    for value in sorted(set(values)):
        subset = [row for group in ordered for row in group[:max(1, math.floor(len(group) * value))]]
        result[value] = tuple(sorted(subset, key=_pair))
    return result

def _reserve(rows: list[dict[str, Any]], fraction: float, key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained, reserved = [], []
    cutoff = int(fraction * (1 << 256))
    for row in rows:
        # Each legal pair is independently assigned by a deterministic hash;
        # sparse strata can have zero validation or zero training positives.
        # The original Standard semantics do not require another DDA positive.
        selected = int(digest([key, *_pair(row)]), 16) < cutoff
        (reserved if selected else retained).append(row)
    return sorted(retained, key=_pair), sorted(reserved, key=_pair)

def make_cell_manifest(panel: Iterable[Mapping[str, Any]], folds: Mapping[str, Any], fold: int,
                       *, validation_fraction: float = .15, training_fraction: float = 1.,
                       probe_fraction: float = .05, protocol_digest: str,
                       auxiliary_training_positives: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    rows = _verified_folds(panel, folds)
    if not isinstance(fold, int) or not 0 <= fold < folds["y"] or not protocol_digest:
        raise CVContractError("fold index or protocol binding invalid")
    if not 0 < validation_fraction < 1 or not 0 <= probe_fraction < 1 or not 0 < training_fraction <= 1:
        raise CVContractError("invalid validation, training, or probe fraction")
    outer = [{k: r[k] for k in ("drug_id", "disease_id", "role")} for r in folds["assignments"] if r["fold"] == fold]
    remaining = [{k: r[k] for k in ("drug_id", "disease_id", "role")} for r in folds["assignments"] if r["fold"] != fold]
    key = digest([folds["repeat_key"], fold, "inner"])
    domain, validation = _reserve(remaining, validation_fraction, key + "validation")
    domain, probe = _reserve(domain, probe_fraction, key + "probe")
    reserved_keys = {_pair(row) for row in outer + validation + probe}
    aux, excluded_aux, seen_aux = [], 0, set()
    for row in auxiliary_training_positives:
        pair = _pair(row)
        relation = row.get("raw_relation")
        if relation != "CONTRAINDICATION":
            raise CVContractError("auxiliary positives require verified CONTRAINDICATION semantics")
        if pair in seen_aux:
            raise CVContractError("duplicate auxiliary refutation positive")
        seen_aux.add(pair)
        if pair in reserved_keys:
            excluded_aux += 1
        else:
            aux.append(dict(drug_id=pair[0], disease_id=pair[1], raw_relation=relation))
    # Refutation P uses its verified channel semantics and its own fixed inner
    # panel. Pair reservations apply across both channels, including Both pairs.
    aux_roles = [dict(drug_id=row["drug_id"], disease_id=row["disease_id"], role="P") for row in aux]
    if aux_roles:
        aux_domain, aux_validation = _reserve(aux_roles, validation_fraction, key + "aux_validation")
        aux_domain, aux_probe = _reserve(aux_domain, probe_fraction, key + "aux_probe")
    else:
        aux_domain, aux_validation, aux_probe = [], [], []
    aux_reserved = {_pair(row) for row in aux_validation + aux_probe}
    extra_excluded = [row for row in domain if _pair(row) in aux_reserved]
    domain = [row for row in domain if _pair(row) not in aux_reserved]
    if not domain:
        raise CVContractError("all support training pairs were reserved by auxiliary inner panels")
    training = list(nested_subsets(domain, [training_fraction], key=key + "scale")[training_fraction])
    aux_selected = list(nested_subsets(aux_domain, [training_fraction], key=key + "aux_scale")[training_fraction]) if aux_domain else []
    support_unused_keys = {_pair(row) for row in domain} - {_pair(row) for row in training}
    aux_unused_keys = {_pair(row) for row in aux_domain} - {_pair(row) for row in aux_selected}
    training = [row for row in training if _pair(row) not in aux_unused_keys]
    aux_selected = [row for row in aux_selected if _pair(row) not in support_unused_keys]
    training_keys = {_pair(row) for row in training}
    unused = extra_excluded + [row for row in domain if _pair(row) not in training_keys]
    auxiliary_keys = {_pair(row) for row in aux_selected}
    aux_unused = [row for row in aux_domain if _pair(row) not in auxiliary_keys]
    def auxiliary_rows(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(drug_id=row["drug_id"], disease_id=row["disease_id"], raw_relation="CONTRAINDICATION")
                for row in sorted(values, key=_pair)]
    return _bound(dict(schema="PHASE85_MODEL_INDEPENDENT_CELL_V1", protocol_digest=protocol_digest,
                       panel_digest=digest(rows), fold_digest=folds["digest"],
                       repeat_key=folds["repeat_key"], y=folds["y"], fold=fold,
                       validation_fraction=validation_fraction, training_fraction=training_fraction,
                       probe_fraction=probe_fraction, training=training, validation=validation,
                       train_probe=probe, outer_test=sorted(outer, key=_pair), unused_training=unused,
                       auxiliary_training=auxiliary_rows(aux_selected),
                       auxiliary_validation=auxiliary_rows(aux_validation),
                       auxiliary_probe=auxiliary_rows(aux_probe), auxiliary_unused=auxiliary_rows(aux_unused),
                       auxiliary_excluded_count=excluded_aux,
                       train_probe_used_for_optimizer=False, train_probe_used_for_selection=False,
                       train_probe_semantics="TRAIN_DOMAIN_HELDOUT_DIAGNOSTIC_NOT_LITERAL_IN_SAMPLE",
                       observed_roles_are_P_VS_U=True, unlabeled_is_negative=False))

HOLDOUT_KEY=b"EVIHGT_PHASE85_FINAL_HOLDOUT_V1|2026-10-05|PRE_REAL_TRAINING"
import hmac
def order_key(namespace,*values):
    return hmac.new(HOLDOUT_KEY,json.dumps([namespace,*values],ensure_ascii=True,
        separators=(',',':')).encode(),hashlib.sha256).digest()

def select_holdout_positives(rows):
    groups=defaultdict(list); seen=set()
    for row in rows:
        pair=(row['drug_id'],row['disease_id'])
        if pair in seen or row.get('role','P')!='P':
            raise ValueError('holdout source requires unique support P')
        seen.add(pair);groups[pair[1]].append(dict(drug_id=pair[0],disease_id=pair[1],role='P'))
    target=(len(seen)+5)//10
    quotas={d:len(values)//10 for d,values in groups.items()}
    remainder=target-sum(quotas.values())
    diseases=sorted(groups,key=lambda d:(-(len(groups[d])%10),order_key('disease-quota',d)))
    for d in diseases[:remainder]:
        quotas[d]+=1
    chosen=[]
    for d in sorted(groups):
        chosen.extend(sorted(groups[d],key=lambda r:order_key('P',r['drug_id'],d))[:quotas[d]])
    return sorted(chosen,key=lambda r:(r['disease_id'],r['drug_id']))
