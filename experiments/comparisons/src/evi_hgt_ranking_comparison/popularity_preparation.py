"""Prepare actual training-only popularity counts; never score any candidate.

The existing 15-cell plan seal is the membership/mask authority. Only training
role projections and role-free exclusion IDs are opened through TrainingInputs.
No graph, raw DrugBank, model, checkpoint, or evaluation route is called here.
"""
from pathlib import Path
import json
import os
import time
import uuid

from evi_hgt.graph_inputs.common import bound, digest, verify
from .common import ROOT, OUT, read, sha
from .data import TrainingInputs
from .popularity import Popularity, member_digest


CACHE = ROOT/'data/cache/ranking_comparison/popularity'
RUNTIME = OUT/'popularity'
METHOD = 'TRAIN_VISIBLE_POPULARITY'
CELLS = tuple(f'R{r}F{f}' for r in range(3) for f in range(5))


def _publish(path, value):
    """Immutable JSON publication with stable byte serialization."""
    path = Path(path).resolve()
    if not any(path.is_relative_to(p.resolve()) for p in (CACHE,RUNTIME)):
        raise PermissionError('Only RankingComparison derived counts/runtime outputs may be written')
    encoded = (json.dumps(value,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError('Immutable popularity artifact differs: '+str(path))
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    os.rename(temporary,path)
    if path.read_bytes() != encoded:
        raise OSError('Popularity artifact did not persist exactly')
    return path


def prepare_cell(data, seal, *, all_seal_digest, all_seal_sha256, source_sha256):
    seal = verify(seal)
    memberships = {name:digest(data.cell[name]) for name in ('training','auxiliary_training')}
    if (seal['schema'] != 'RANKING_COMPARISON_CELL_PLAN_SEAL_V1' or seal['status'] != 'SEALED'
        or seal['cell'] != data.cell_id or seal['seed'] != data.training_seed
        or seal['training_membership_digests'] != memberships
        or seal['mask_digest'] != data.bindings['fold_mask_digest']
        or seal['mask_digest'] != digest(sorted(data.blocked))):
        raise ValueError('Popularity training membership/seed/mask differs from sealed cell')
    # Bind the existing schedule/conditional artifacts, without opening scores or
    # rebuilding groups. Counting does not depend on U draws or model weights.
    for kind in ('schedule','plan'):
        if sha(seal[kind+'_path']) != seal[kind+'_sha256']:
            raise ValueError('Sealed '+kind+' bytes changed')
    support = set(data.support);contra = set(data.unsuitability)
    support_hash = member_digest(support);contra_hash = member_digest(contra)
    for pair in support|contra:
        data.pairs.assert_optimizer_pair(pair,'P')
    counts = Popularity.build(cell=data.cell_id,support=support,contra=contra,
        expected_support_digest=support_hash,expected_contra_digest=contra_hash,
        forbidden_pairs=data.blocked)
    support_counts = dict(sorted(counts.support.items()))
    contra_counts = dict(sorted(counts.contra.items()))
    if (sum(support_counts.values()) != len(support)
        or sum(contra_counts.values()) != len(contra)
        or len(support) != seal['cycles'][0]['support_P']
        or len(contra) != seal['cycles'][0]['contra_P']
        or any(type(v) is not int or v <= 0 for v in list(support_counts.values())+list(contra_counts.values()))):
        raise ValueError('Actual unique-pair degree totals differ from sealed P counts')
    identity = dict(schema='RANKING_COMPARISON_POPULARITY_IDENTITY_V1',method=METHOD,cell=data.cell_id,
        training_seed=data.training_seed,all_plans_sealed_digest=all_seal_digest,
        all_plans_sealed_sha256=all_seal_sha256,cell_plan_seal_digest=seal['digest'],
        training_membership_digests=memberships,mask_digest=seal['mask_digest'],
        support_member_sha256=support_hash,contra_member_sha256=contra_hash,
        schedule_sha256=seal['schedule_sha256'],plan_sha256=seal['plan_sha256'],
        source_sha256=source_sha256,score_rule=counts.identity['score'],
        unknown_drug_count=0,counting_unit='UNIQUE_TRAINING_P_DRUG_DISEASE_PAIR_PER_HEAD')
    identity_digest = digest(identity)
    payload = bound(dict(schema='RANKING_COMPARISON_POPULARITY_DERIVED_COUNTS_V1',identity=identity,
        identity_digest=identity_digest,support_counts=support_counts,contra_counts=contra_counts,
        support_counts_digest=digest(support_counts),contra_counts_digest=digest(contra_counts),
        support_unique_pairs=len(support),contra_unique_pairs=len(contra),
        support_drugs=len(support_counts),contra_drugs=len(contra_counts),
        support_diseases=len({d for _,d in support}),contra_diseases=len({d for _,d in contra}),
        same_pair_positive_in_both_heads=len(support&contra),query_scores=0,
        model_forwards=0,optimizer_updates=0))
    path = _publish(CACHE/identity_digest/'counts.json',payload)
    receipt = bound(dict(schema='RANKING_COMPARISON_POPULARITY_CELL_PREPARED_V1',status='COUNTS_PREPARED_NO_SCORES',
        method=METHOD,cell=data.cell_id,identity=identity,identity_digest=identity_digest,
        counts_path=str(path),counts_sha256=sha(path),counts_digest=payload['digest'],
        support_member_sha256=support_hash,contra_member_sha256=contra_hash,
        support_counts_digest=payload['support_counts_digest'],contra_counts_digest=payload['contra_counts_digest'],
        support_unique_pairs=len(support),contra_unique_pairs=len(contra),
        support_drugs=len(support_counts),contra_drugs=len(contra_counts),
        support_diseases=payload['support_diseases'],contra_diseases=payload['contra_diseases'],
        same_pair_positive_in_both_heads=payload['same_pair_positive_in_both_heads'],
        forbidden_pair_intersection=0,projection_audit=data.projection_audit,
        query_scores=0,model_instances=0,model_forwards=0,optimizer_updates=0,
        graph_loaded=False,raw_drugbank_opened=False,evaluation_candidates_opened=False,
        protected_labels_read=False,role_free_exclusion_ids_used=True,
        formal_training_authorized=False,outer_eval_authorized=False,final_eval_authorized=False))
    _publish(RUNTIME/f'{data.cell_id}.json',receipt)
    return receipt


def prepare_all_popularity():
    started = time.monotonic()
    path = OUT/'sampling/ALL_PLANS_SEALED.json'
    initial_seal_sha = sha(path);all_seals = verify(read(path))
    if (all_seals['schema'] != 'RANKING_COMPARISON_ALL_CELL_PLANS_SEALED_V1'
        or all_seals['status'] != 'SEALED_15_CELLS'
        or len(all_seals['cells']) != 15
        or {r['cell'] for r in all_seals['cells']} != set(CELLS)):
        raise ValueError('Exactly fifteen completed original cell seals are required')
    for source,expected in all_seals['source_sha256'].items():
        if sha(source) != expected:
            raise ValueError('Training-only projection/plan implementation changed: '+source)
    source_paths = [ROOT/'src/evi_hgt_ranking_comparison'/name for name in ('data.py','popularity.py','popularity_preparation.py')]
    source_paths.append(ROOT/'scripts/ranking_comparison/prepare_popularity.py')
    source_hashes = {str(p):sha(p) for p in source_paths}
    inputs = TrainingInputs()
    seals = {r['cell']:r for r in all_seals['cells']}
    rows = []
    for cell in CELLS:
        data = inputs.cell(cell)
        row = prepare_cell(data,seals[cell],all_seal_digest=all_seals['digest'],
            all_seal_sha256=initial_seal_sha,source_sha256=source_hashes)
        rows.append(row)
        print(f"训练计数完成 {cell}：support P={row['support_unique_pairs']}，contra P={row['contra_unique_pairs']}；query 评分 0。",flush=True)
    if sha(path) != initial_seal_sha or any(sha(p) != h for p,h in source_hashes.items()):
        raise ValueError('Bound input/source changed during popularity preparation')
    summary = bound(dict(schema='RANKING_COMPARISON_ALL_POPULARITY_COUNTS_PREPARED_V1',status='PREPARED_15_CELLS_NO_SCORES',
        method=METHOD,cells=rows,cell_count=15,all_plans_sealed_digest=all_seals['digest'],
        all_plans_sealed_sha256=initial_seal_sha,source_sha256=source_hashes,
        support_unique_pairs_sum_across_cells=sum(r['support_unique_pairs'] for r in rows),
        contra_unique_pairs_sum_across_cells=sum(r['contra_unique_pairs'] for r in rows),
        totals_semantics='sum of fold-local training memberships; not globally deduplicated subjects',
        all_cell_identity_digest=digest([r['identity_digest'] for r in rows]),
        all_count_payloads_digest=digest([r['counts_sha256'] for r in rows]),
        query_scores=0,model_instances=0,model_forwards=0,optimizer_updates=0,
        graph_loaded=False,raw_drugbank_opened=False,evaluation_candidates_opened=False,
        protected_labels_read=False,role_free_exclusion_ids_used=True,
        formal_training_authorized=False,outer_eval_authorized=False,final_eval_authorized=False))
    result_path = _publish(RUNTIME/'ALL_POPULARITY_PREPARED.json',summary)
    # The deterministic summary has no timing fields. Runtime remains separate.
    runtime = dict(schema='RANKING_COMPARISON_POPULARITY_PREPARATION_RUNTIME_V1',elapsed_seconds=time.monotonic()-started,
        summary_path=str(result_path),summary_sha256=sha(result_path),cells_completed=15,
        query_scores=0,model_forwards=0,optimizer_updates=0)
    _publish(RUNTIME/'runs'/f'{uuid.uuid4().hex}.json',runtime)
    print(f'15 个 cell 训练计数已完成：{result_path}',flush=True)
    return summary
