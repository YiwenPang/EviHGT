"""Frozen post-access/pre-results feasibility amendment. No scoring in this module.

Only endpoint identities, verified positive membership, source hashes and counts
enter candidate selection. U retains the existing evaluator's benchmark meaning:
not a held-out Policy B support positive; never a verified negative.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import uuid

from evi_hgt.evaluation.formal_runtime import FROZEN_FORMAL_MATRIX, digest, file_sha, read_json

PROTOCOL_ID = 'STANDARD_SAMPLED_PU_COMPARISON_HMAC_SHA256_U1024_V1'
PLAN_FORMAT = 'EVIHGT_THREE_LAYER_FORMAL_EXECUTION_V1'
FORMAL_RUN_ID = 'EVIHGT-FORMAL-ad110867a0e6f6db-a6fdc76814484f4bb228673646a03e82'
COLD_POLICY = '22772a9fb7484a6f23e5073800a5a1221b396eac4c6e8da97b573cc2afd65de7'
LEDGER_SHA = 'edf38465d76e638ce03b51c020d6ea0a4aac5170f40be1248a26b4b48bb67cb2'
REPRESENTATION_SHA = '8cdcae5648e8fab4d8213e00d384302424df625c9ee3856172b9c9437a527013'
MATRIX = FROZEN_FORMAL_MATRIX
OUT = Path('outputs/formal_redesign')
MANIFEST_DIR = Path('data/processed/formal_evaluation_redesign_v1')
FREEZE_PATH = OUT/'freeze.json'
PLAN_PATH = OUT/'execution_plan.json'
BLINDNESS = dict(formal_metrics_computed=False, formal_results_viewed=False,
    formal_top_k_viewed=False, formal_scoring_executed=False)


class RedesignInvariantError(RuntimeError):
    """Abort before model loading, scoring, metrics or state mutation."""


def require(condition, message):
    if not condition:
        raise RedesignInvariantError(message)


def exclusive_json(path, value):
    """Fsync + exclusive hard-link commit; never replaces a frozen artifact."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison_protocol():
    return dict(protocol_version=PROTOCOL_ID, split='standard', partition='test',
        all_diseases=True, u_limit=1024, positive_retention='ALL_ELIGIBLE_FORMAL_POLICY_B_SUPPORT_POSITIVES',
        eligibility='EXISTING_RANKING_CANDIDATE_PROTOCOL_TRAINING_KNOWN_EXCLUSIONS',
        u_definition='ELIGIBLE_CANDIDATES_MINUS_FORMAL_POLICY_B_SUPPORT_POSITIVES',
        unlabeled_is_negative=False, selection_key='UTF8_PROTOCOL_VERSION',
        selection_message='UTF8_COMPACT_JSON_ARRAY_CANONICAL_DISEASE_ID_CANONICAL_DRUG_ID',
        selection_digest='HMAC_SHA256', selection_order=['DIGEST_ASCENDING','CANONICAL_DRUG_ID_ASCENDING'],
        frozen_candidate_order='CANONICAL_DRUG_ID_ASCENDING', model_independent=True,
        result_independent=True, score='support_logit', comparison_channel='support',
        primary=['DISEASE_WISE_P_VS_U_ROC_AUC','DISEASE_WISE_P_VS_U_NON_INTERPOLATED_AP'],
        aggregation='UNWEIGHTED_MACRO; EXISTING_NO_P_OR_NO_U_SKIP_SEMANTICS',
        ranking_k=[5,10,20], internal_ranking_k=[5,10,20,50],
        ranking_label='SAMPLED COMPARISON BENCHMARK',
        internal_at50_is_full_retrieval=False,
        unsuitability_head='UNCHANGED_DUAL_FORWARD; NO_NEW_COMPARISON_GROUND_TRUTH')


def u_digest(disease_id, drug_id):
    message=json.dumps([disease_id,drug_id],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hmac.digest(PROTOCOL_ID.encode('utf-8'), message, 'sha256')


def select_u(disease_id, eligible_drugs, positives):
    eligible=set(eligible_drugs); positives=set(positives)
    require(positives<=eligible, 'positive outside eligible candidate scope')
    selected=sorted(eligible-positives,key=lambda drug:(u_digest(disease_id,drug),drug))[:1024]
    # Selection is by digest; scoring order is canonical, independently frozen.
    return sorted(selected)


def _seal(payload, field='global_manifest_digest'):
    return {**payload, field:digest(payload)}


def build_comparison_manifest(sources, *, source_binding, progress=False):
    from tqdm import tqdm
    protocol=comparison_protocol(); rows=[]; seen=set()
    for source in tqdm(sources, desc='[MANIFEST][RUNNING]', unit='disease',
            disable=not progress, dynamic_ncols=True, leave=True, mininterval=2):
        require((source.split_kind,source.partition,source.channel)==('standard','test','support'),
            'comparison source must use Formal Standard Policy B support')
        disease=source.disease_id
        require(disease not in seen, 'duplicate disease'); seen.add(disease)
        eligible=set(source.candidate_drug_ids); positives=sorted(set(source.held_out_relevant_drug_ids))
        require(len(eligible)==len(source.candidate_drug_ids),'duplicate source candidate')
        require(len(positives)==len(source.held_out_relevant_drug_ids),'duplicate source positive')
        u=select_u(disease,eligible,positives); ids=sorted(set(positives)|set(u))
        row=dict(protocol_version=PROTOCOL_ID, disease_id=disease, positive_drug_ids=positives,
            u_drug_ids=u, candidate_drug_ids=ids, positive_count=len(positives), u_count=len(u),
            total_candidate_count=len(ids), eligible_candidate_count=len(eligible),
            eligible_membership_digest=digest(sorted(eligible)),
            source_candidate_membership_digest=source.candidate_membership_digest,
            source_definition_digest=source.definition_digest,
            positive_digest=digest(positives), u_digest=digest(u), candidate_order_digest=digest(ids))
        rows.append(_seal(row,'candidate_digest'))
    rows.sort(key=lambda row:row['disease_id'])
    require(bool(rows),'empty comparison disease set')
    return _seal(dict(format='EVIHGT_COMPARISON_MANIFEST_V1', protocol=protocol,
        protocol_digest=digest(protocol), source_binding=source_binding,
        disease_set_digest=digest([r['disease_id'] for r in rows]), diseases=rows,
        totals=dict(diseases=len(rows),positives=sum(r['positive_count'] for r in rows),
            u=sum(r['u_count'] for r in rows),pairs=sum(r['total_candidate_count'] for r in rows))))


def validate_comparison_manifest(manifest, *, expected_digest, sources=None, source_binding=None):
    require(manifest.get('global_manifest_digest')==expected_digest,'comparison manifest external hash changed')
    require(digest({k:v for k,v in manifest.items() if k!='global_manifest_digest'})==expected_digest,
        'comparison manifest payload changed')
    require(manifest['protocol']==comparison_protocol() and manifest['protocol_digest']==digest(comparison_protocol()),
        'comparison protocol digest changed')
    ids=[r['disease_id'] for r in manifest['diseases']]
    require(ids==sorted(set(ids)) and manifest['disease_set_digest']==digest(ids),'disease set/order changed')
    for row in manifest['diseases']:
        require(row['candidate_digest']==digest({k:v for k,v in row.items() if k!='candidate_digest'}),'candidate digest changed')
        p,u,c=(row[k] for k in ('positive_drug_ids','u_drug_ids','candidate_drug_ids'))
        require(p==sorted(set(p)) and u==sorted(set(u)) and not set(p)&set(u),'positive/U membership invalid')
        require(c==sorted(set(p)|set(u)),'candidate order changed')
        require((len(p),len(u),len(c))==(row['positive_count'],row['u_count'],row['total_candidate_count']),
            'candidate counts changed')
        require(len(u)==min(1024,row['eligible_candidate_count']-len(p)),'U quota changed')
        require((digest(p),digest(u),digest(c))==(row['positive_digest'],row['u_digest'],row['candidate_order_digest']),
            'membership/order hash changed')
    if sources is not None:
        rebuilt=build_comparison_manifest(sources,source_binding=source_binding)
        require(rebuilt==manifest,'disease/positive/U/eligible source membership changed')


def full_candidate_manifest(streams, *, split, source_binding, progress=False):
    """Compact full-retrieval commitments: one disease at a time, no Cartesian table."""
    from tqdm import tqdm
    rows=[]
    for support,counter in tqdm(zip(*streams,strict=True),desc=f'[FULL_MANIFEST:{split}][RUNNING]',
            unit='disease',disable=not progress,dynamic_ncols=True,leave=True,mininterval=2):
        require(support.disease_id==counter.disease_id and support.candidate_drug_ids==counter.candidate_drug_ids,
            'full channel candidate mismatch')
        row=dict(disease_id=support.disease_id,candidate_count=support.candidate_count,
            candidate_order_digest=digest(support.candidate_drug_ids),
            candidate_membership_digest=support.candidate_membership_digest,
            definitions=[support.definition_digest,counter.definition_digest],
            positive_counts=[support.relevance_count,counter.relevance_count])
        rows.append(row)
    return _seal(dict(format='EVIHGT_FULL_CANDIDATE_COMMITMENT_V1',split=split,source_binding=source_binding,
        diseases=rows,disease_set_digest=digest([r['disease_id'] for r in rows]),
        queries=len(rows),pairs=sum(r['candidate_count'] for r in rows),
        cold_policy_digest=COLD_POLICY if split!='standard' else None))


def canonical_cells(cells):
    mapped={c['label']:c for c in cells}
    require(len(mapped)==len(cells)==8,'checkpoint cell matrix changed')
    ordered=[]
    for label,model,split in MATRIX:
        require(label in mapped,'checkpoint label missing')
        row=mapped[label]
        require((row['model_variant'],row['split'])==(model,split),'checkpoint model/split binding changed')
        if split!='standard': require(row['cold_visibility_policy_digest']==COLD_POLICY,'Cold protocol changed')
        ordered.append(row)
    return ordered


def validate_checkpoint_context(cell,binding,execution,identity):
    require(binding['label']==cell['label'] and execution.run['run_id']==cell['run_id']
        and execution.run['model_variant']==cell['model_variant'] and execution.run['split']==cell['split']
        and identity.identity_digest==cell['checkpoint_identity'],'checkpoint mapping/identity digest changed')


def build_execution_plan(manifest, cells, *, full_manifests, formal_run_id):
    cells=canonical_cells(cells); units=[]
    for cell in cells[:6]:
        units.append(dict(unit_id='A_'+cell['label'],layer='A',split='standard',
            label=cell['label'],model_variant=cell['model_variant'],
            comparison_manifest_digest=manifest['global_manifest_digest'],
            benchmark='SAMPLED COMPARISON BENCHMARK',pairs=manifest['totals']['pairs'],
            diseases=manifest['totals']['diseases'],ranking_k=[5,10,20],primary_metrics=['ROC_AUC_P_VS_U','AP_P_VS_U']))
    for layer,cell,queries,pairs in [('B',cells[0],1354,13525600),('C',cells[6],1313,1294618),('D',cells[7],232,2326496)]:
        units.append(dict(unit_id=layer+'_'+cell['label'],layer=layer,split=cell['split'],
            label=cell['label'],model_variant='FULL_EVIHGT',benchmark='FULL CANDIDATE RETRIEVAL',
            full_manifest_digest=full_manifests.get(cell['split'],{}).get('global_manifest_digest'),
            pairs=pairs,diseases=queries,ranking_k=[5,10,20,50]))
    protocol=dict(layer1=comparison_protocol(),layer2='FULL_ONLY_ALL_1354_STANDARD_DISEASES_10028_DRUG_UNIVERSE',
        layer3='UNCHANGED_FULL_ONLY_COLD_QUERY_SIDE_INFORMATION_V1',
        candidate_eligibility='EXISTING_TRAINING_KNOWN_EXCLUSIONS; FROZEN_CENSUS',
        no_base_full_retrieval=True,no_ablation_full_retrieval=True,no_full_base_subset=True)
    return dict(format=PLAN_FORMAT,formal_run_id=formal_run_id,formal_test_count=1,
        readiness='READY FOR FORMAL EXECUTION',execution_authorized=False,graph_batch_size=1,
        precision='UNCHANGED_CUDA_BF16_CPU_FP32',deterministic_algorithms=True,
        cublas_workspace_config=':4096:8',training_protocol='PRACTICAL_FORMAL_V1',
        training_allowed=False,validation_replay_allowed=False,automatic_retry_allowed=False,
        exactness='FROZEN_INDEXED_REPRESENTATION; NO_APPROXIMATION; NO_GRAPH_BATCH_CHANGE',
        representation='STANDARD_INDEXED_FACT_COLUMNS_V1',representation_sha256=REPRESENTATION_SHA,
        cold_policy_digest=COLD_POLICY,blindness=BLINDNESS.copy(),protocol=protocol,
        protocol_digest=digest(protocol),cells=cells,checkpoint_binding_digest=digest(cells),units=units,
        full_manifests=full_manifests,output_root='outputs/phase8_formal/evaluation/three_layer_formal_v1')


def validate_execution_plan(plan, manifest, cells):
    expected=build_execution_plan(manifest,cells,full_manifests=plan['full_manifests'],formal_run_id=FORMAL_RUN_ID)
    # Other freeze fields are checked against their external file commitment.
    require(all(plan.get(key)==value for key,value in expected.items()),'execution plan/protocol/checkpoint/Formal state changed')


def verify_history(root):
    directory=Path(root)/'outputs/phase8_4_1/amendment_history'
    files=sorted(directory.glob('*.json')); previous=None; hashes={}
    for index,path in enumerate(files,1):
        require(path.name.startswith(f'{index:04d}_'),'non-monotonic amendment history')
        row=read_json(path)
        require(row['formal_run_id']==FORMAL_RUN_ID and row['formal_test_count']==1,'amendment Formal state mismatch')
        require(row['previous_event_sha256']==previous,'amendment chain hash mismatch')
        previous=file_sha(path); hashes[path.relative_to(root).as_posix()]=previous
    return hashes


def next_amendment(root):
    found=[]
    for path in (Path(root)/'outputs').glob('phase*/FORMAL_*AMENDMENT_*.json'):
        match=re.fullmatch(r'FORMAL_.*AMENDMENT_(\d+)\.json',path.name)
        if match: found.append((int(match[1]),path))
    require(bool(found),'prior amendment missing')
    found.sort()
    return f'FORMAL_EVALUATION_AMENDMENT_{found[-1][0]+1:03d}', found
