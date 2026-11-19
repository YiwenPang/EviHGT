"""Final master-paper scope: pure candidate identities, commitments and budgets.

No scoring, model loading, ranking or metric calculation occurs here. Candidate
eligibility is supplied by the unchanged original candidate builder. Cold query
units remain its existing disease-wise units, not a new endpoint orientation.
"""
from copy import deepcopy
import hmac
import json
import statistics

from evi_hgt.evaluation.formal_redesign import (
    COLD_POLICY, FORMAL_RUN_ID, REPRESENTATION_SHA, canonical_cells, digest, require,
)
from evi_hgt.evaluation.emergency_u32 import (
    validate_manifest as validate_u32, engineering_budget, seal,
)

FULL_PROTOCOL_ID = 'STANDARD_FULL_RETRIEVAL_STRATIFIED_6_V1'
COLD_PROTOCOL_IDS = {'cold_drug':'COLD_DRUG_SAMPLED_PU_U64_V1',
    'cold_disease':'COLD_DISEASE_SAMPLED_PU_U64_V1'}
PLAN_FORMAT = 'EVIHGT_FINAL_FORMAL_FOUR_STAGE_V1'
FULL_LABEL = 'DETERMINISTIC STRATIFIED FULL-CANDIDATE RETRIEVAL SUBSET'
COLD_LABEL = 'SAMPLED COLD GENERALIZATION BENCHMARK'


def hmac_key(version, *ids):
    message=json.dumps(list(ids),ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hmac.digest(version.encode('utf-8'),message,'sha256'),ids[-1]


def stratify(rows):
    counts=[(r['disease_id'],r['positive_count']) for r in rows]
    require(len(counts)>=6 and len({d for d,_ in counts})==len(counts),'six distinct eligible diseases required')
    require(all(isinstance(n,int) and n>=0 for _,n in counts),'invalid positive counts')
    ordered=[list(v) for v in sorted(counts,key=lambda r:(r[1],r[0]))]
    quotient,remainder=divmod(len(ordered),3);start=0;strata=[]
    for i,name in enumerate(('LOW','MID','HIGH')):
        end=start+quotient+int(i<remainder);ids=[r[0] for r in ordered[start:end]]
        selected=sorted(ids,key=lambda d:hmac_key(FULL_PROTOCOL_ID,d))[:2]
        strata.append(dict(name=name,start_inclusive=start,end_exclusive=end,size=len(ids),disease_ids=ids,
            positive_count_min=ordered[start][1],positive_count_max=ordered[end-1][1],selected_disease_ids=selected))
        start=end
    return dict(protocol_version=FULL_PROTOCOL_ID,sorting_rule=['POSITIVE_COUNT_ASCENDING','CANONICAL_DISEASE_ID_ASCENDING'],
        partition_rule='DIVMOD_N_3; EXTRA_ONE_TO_EARLIEST_STRATA_LOW_THEN_MID_THEN_HIGH; CONTIGUOUS_HALF_OPEN_BOUNDARIES',
        sorted_positive_count_vector=ordered,positive_count_vector_digest=digest(ordered),
        eligible_disease_set_digest=digest(sorted(d for d,_ in counts)),strata=strata,
        hmac_key='UTF8_PROTOCOL_VERSION',hmac_message='UTF8_COMPACT_JSON_ARRAY_CANONICAL_DISEASE_ID',
        selection_order=['HMAC_SHA256_DIGEST_ASCENDING','CANONICAL_DISEASE_ID_ASCENDING'],selected_per_stratum=2,
        selected_disease_ids=sorted(d for s in strata for d in s['selected_disease_ids']))


def specification(split):
    full=split=='standard'
    require(full or split in COLD_PROTOCOL_IDS,'unsupported final split')
    return dict(protocol_version=FULL_PROTOCOL_ID if full else COLD_PROTOCOL_IDS[split],split=split,partition='test',
        benchmark=FULL_LABEL if full else COLD_LABEL,score='support_logit',comparison_channel='support',
        eligible_universe='ORIGINAL_RANKING_CANDIDATE_PROTOCOL; TRAINING_KNOWN_EXCLUSIONS_UNCHANGED',
        candidate_order='CANONICAL_DRUG_ID_ASCENDING',tie_break='CANONICAL_DRUG_ID_ASCENDING',
        positive_retention='ALL_ELIGIBLE_FORMAL_POLICY_B_SUPPORT_POSITIVES',unlabeled_is_negative=False,
        u_definition='ELIGIBLE_CANDIDATES_MINUS_FORMAL_POLICY_B_SUPPORT_POSITIVES',
        u_quota=None if full else 64,full_candidate_universe=full,all_queries=not full,
        primary=['DISEASE_WISE_P_VS_U_ROC_AUC','DISEASE_WISE_P_VS_U_NON_INTERPOLATED_AP'],
        existing_statistical_unit='DISEASE_WISE_QUERY_FROM_ORIGINAL_CANDIDATE_BUILDER',
        ranking_k=[5,10,20,50] if full else [5,10],excluded_ranking_k=[] if full else [20,50],
        ranking_emphasis='FULL_CANDIDATE_RETRIEVAL' if full else 'SECONDARY',
        aggregation='SIX_DISEASE_DETERMINISTIC_STRATIFIED_SUBSET_MACRO; NOT_POPULATION_ESTIMATE' if full
            else 'ALL_QUERY_UNWEIGHTED_MACRO; EXISTING_NO_P_OR_NO_U_SKIP_SEMANTICS',
        population_wide_estimate=False,model_independent=True,result_independent=True,checkpoint_independent=True,
        selection_digest='HMAC_SHA256',selection_key='UTF8_PROTOCOL_VERSION',
        selection_message='UTF8_COMPACT_JSON_ARRAY_CANONICAL_QUERY_ID_CANONICAL_CANDIDATE_ID' if not full
            else 'UTF8_COMPACT_JSON_ARRAY_CANONICAL_DISEASE_ID_WITHIN_FROZEN_STRATUM',
        selection_order=['DIGEST_ASCENDING','CANONICAL_ID_ASCENDING'],
        unsuitability_head='UNCHANGED_DUAL_FORWARD; NO_NEW_NEGATIVE_GROUND_TRUTH',
        cold_side_information_policy=None if full else 'COLD_QUERY_SIDE_INFORMATION_V1',
        cold_policy_digest=None if full else COLD_POLICY)


def source_row(source):
    require((source.partition,source.channel)==('test','support'),'Formal Policy B support source required')
    ids=list(source.candidate_drug_ids);p=list(source.held_out_relevant_drug_ids)
    require(ids==sorted(set(ids)) and p==sorted(set(p)) and set(p)<=set(ids),'invalid original candidate/positive membership')
    return dict(disease_id=source.disease_id,positive_count=len(p),positive_digest=digest(p),
        eligible_candidate_count=len(ids),eligible_membership_digest=digest(ids),
        source_candidate_membership_digest=source.candidate_membership_digest,source_definition_digest=source.definition_digest)


def candidate_row(source,version,u_quota,stratum=None):
    base=source_row(source);p=list(source.held_out_relevant_drug_ids)
    u_universe=set(source.candidate_drug_ids)-set(p)
    u=sorted(u_universe) if u_quota is None else sorted(sorted(u_universe,
        key=lambda c:hmac_key(version,source.disease_id,c))[:u_quota])
    ids=sorted(set(p)|set(u))
    return seal(dict(**base,query_id=source.disease_id,protocol_version=version,stratum=stratum,
        positive_drug_ids=p,u_drug_ids=u,candidate_drug_ids=ids,u_count=len(u),total_candidate_count=len(ids),
        u_digest=digest(u),candidate_order_digest=digest(ids)),'candidate_digest')


def distribution(rows):
    values=sorted(r['total_candidate_count'] for r in rows)
    return dict(min=min(values),median=statistics.median(values),mean=statistics.fmean(values),max=max(values))


def checkpoint_binding(cell):
    return {k:deepcopy(cell[k]) for k in ('label','model_variant','split','checkpoint',
        'checkpoint_sha256','checkpoint_identity')}


def commitment(rows,split,selection,source_binding,checkpoint,amendment_id,superseded_digest):
    require(checkpoint['model_variant']=='FULL_EVIHGT' and checkpoint['split']==split,'FULL checkpoint split binding differs')
    rows=sorted(rows,key=lambda r:r['disease_id']);require(bool(rows),'empty final candidate manifest')
    spec=specification(split)
    return seal(dict(format='EVIHGT_FINAL_CANDIDATE_MANIFEST_V1',amendment_id=amendment_id,
        protocol=spec,protocol_digest=digest(spec),source_binding=deepcopy(source_binding),checkpoint=checkpoint_binding(checkpoint),
        representation='STANDARD_INDEXED_FACT_COLUMNS_V1',representation_sha256=REPRESENTATION_SHA,graph_batch_size=1,
        cold_policy_digest=None if split=='standard' else COLD_POLICY,
        cold_policy=None if split=='standard' else 'COLD_QUERY_SIDE_INFORMATION_V1',
        selection=selection,query_set_digest=digest([r['disease_id'] for r in rows]),queries=rows,
        totals=dict(queries=len(rows),positives=sum(r['positive_count'] for r in rows),u=sum(r['u_count'] for r in rows),
            pairs=sum(r['total_candidate_count'] for r in rows)),candidate_distribution=distribution(rows),
        supersession=dict(status='SUPERSEDED FOR COMPUTATIONAL FEASIBILITY',previous_manifest_digest=superseded_digest,
            previous_artifacts_preserved=True,previous_partial_results_reusable=False)), 'global_manifest_digest')


def build_full_subset(layer1,sources,*,source_binding,checkpoint,amendment_id,superseded_digest,progress=False):
    validate_u32(layer1);require(layer1['source_binding']==source_binding,'Standard source binding changed')
    selection=stratify(layer1['diseases']);selected=set(selection['selected_disease_ids'])
    strata={d:s['name'] for s in selection['strata'] for d in s['selected_disease_ids']}
    frozen={r['disease_id']:r for r in layer1['diseases']};seen=set();rows=[]
    from tqdm import tqdm
    for source in tqdm(sources,total=len(frozen),desc='[LAYER2_MANIFEST][RUNNING]',unit='query',
            disable=not progress,dynamic_ncols=True,leave=True,mininterval=2):
        require(source.split_kind=='standard','Standard split changed')
        d=source.disease_id;require(d in frozen and d not in seen,'Standard disease source set changed');seen.add(d)
        identity=source_row(source)
        require(all(identity[k]==frozen[d][k] for k in identity),'Standard positive/eligible source identity changed')
        if d in selected:rows.append(candidate_row(source,FULL_PROTOCOL_ID,None,strata[d]))
    require(seen==set(frozen),'Standard disease source set incomplete')
    return commitment(rows,'standard',selection,source_binding,checkpoint,amendment_id,superseded_digest)


def build_cold_sample(sources,*,split,source_binding,checkpoint,amendment_id,superseded_digest,progress=False):
    require(split in COLD_PROTOCOL_IDS,'Cold split required');seen=set();rows=[]
    from tqdm import tqdm
    for source in tqdm(sources,desc=f'[{split.upper()}_MANIFEST][RUNNING]',unit='query',
            disable=not progress,dynamic_ncols=True,leave=True,mininterval=2):
        require(source.split_kind==split and source.disease_id not in seen,'Cold split/query set changed')
        seen.add(source.disease_id);rows.append(candidate_row(source,COLD_PROTOCOL_IDS[split],64))
    return commitment(rows,split,dict(all_eligible_queries_retained=True,
        eligible_query_set_digest=digest(sorted(seen)),query_statistical_unit='UNCHANGED_DISEASE_WISE_CANDIDATE_BUILDER'),
        source_binding,checkpoint,amendment_id,superseded_digest)


def validate_manifest(value,*,expected_digest):
    require(value['global_manifest_digest']==expected_digest and digest({k:v for k,v in value.items()
        if k!='global_manifest_digest'})==expected_digest,'manifest external/payload digest changed')
    split=value['protocol']['split'];spec=specification(split)
    require(value['protocol']==spec and value['protocol_digest']==digest(spec),'final protocol changed')
    require(value['graph_batch_size']==1 and value['representation_sha256']==REPRESENTATION_SHA and
        value['representation']=='STANDARD_INDEXED_FACT_COLUMNS_V1','indexed/exact batch binding changed')
    require(value['checkpoint']['model_variant']=='FULL_EVIHGT' and value['checkpoint']['split']==split,'checkpoint split changed')
    require((value['cold_policy_digest'],value['cold_policy'])==
        ((None,None) if split=='standard' else (COLD_POLICY,'COLD_QUERY_SIDE_INFORMATION_V1')),'Cold policy changed')
    rows=value['queries'];ids=[r['disease_id'] for r in rows]
    require(ids==sorted(set(ids)) and value['query_set_digest']==digest(ids),'query set/order changed')
    if split=='standard':
        selection=value['selection'];recomputed=stratify([dict(disease_id=d,positive_count=n)
            for d,n in selection['sorted_positive_count_vector']])
        require(selection==recomputed and ids==selection['selected_disease_ids'],'strata/selected disease identity changed')
        strata={d:s['name'] for s in selection['strata'] for d in s['selected_disease_ids']}
    else:require(value['selection']['eligible_query_set_digest']==digest(ids) and
        value['selection']['all_eligible_queries_retained'] is True,'Cold all-query scope changed')
    for row in rows:
        require(row['candidate_digest']==digest({k:v for k,v in row.items() if k!='candidate_digest'}),'per-query digest changed')
        p,u,c=(row[k] for k in ('positive_drug_ids','u_drug_ids','candidate_drug_ids'))
        require(p==sorted(set(p)) and u==sorted(set(u)) and not set(p)&set(u) and
            c==sorted(set(p)|set(u)),'positive/U/order changed')
        require((len(p),len(u),len(c))==(row['positive_count'],row['u_count'],row['total_candidate_count']),'per-query counts changed')
        require((digest(p),digest(u),digest(c))==(row['positive_digest'],row['u_digest'],row['candidate_order_digest']),
            'per-query membership commitments changed')
        require(row['query_id']==row['disease_id'] and row['protocol_version']==spec['protocol_version'],'query protocol changed')
        require(row['u_count']==(row['eligible_candidate_count']-len(p) if split=='standard' else
            min(64,row['eligible_candidate_count']-len(p))),'full universe/U64 quota changed')
        if split=='standard':require(digest(c)==row['eligible_membership_digest'] and row['stratum']==strata[row['disease_id']],
            'full candidate universe/stratum changed')
    require(value['totals']==dict(queries=len(rows),positives=sum(r['positive_count'] for r in rows),
        u=sum(r['u_count'] for r in rows),pairs=sum(r['total_candidate_count'] for r in rows)) and
        value['candidate_distribution']==distribution(rows),'candidate totals/distribution changed')


def build_final_plan(layer1_plan,layer1,full,cold,*,amendment_id):
    validate_u32(layer1);p=deepcopy(layer1_plan);cells=canonical_cells(p['cells'])
    require((p['formal_run_id'],p['formal_test_count'])==(FORMAL_RUN_ID,1),'Formal state changed')
    manifests={'standard':full,**cold}
    for split,value in manifests.items():
        validate_manifest(value,expected_digest=value['global_manifest_digest'])
        require(value['amendment_id']==amendment_id and value['source_binding']==p['data_binding'],'manifest source/amendment differs')
        require(value['checkpoint']==checkpoint_binding(next(c for c in cells if c['split']==split and c['model_variant']=='FULL_EVIHGT')),
            'manifest checkpoint differs from plan')
    require(p['comparison_manifest_digest']==layer1['global_manifest_digest'],'Layer1 binding changed')
    units=deepcopy(p['units']);require(len(units)==6 and all(u['layer']=='A' and u['ranking_k']==[5,10] for u in units),
        'frozen Layer1 units changed')
    for stage,split in [('B','standard'),('C','cold_drug'),('D','cold_disease')]:
        manifest=manifests[split];cell=next(c for c in cells if c['split']==split and c['model_variant']=='FULL_EVIHGT')
        units.append(dict(unit_id=stage+'_'+cell['label'],layer=stage,split=split,label=cell['label'],model_variant='FULL_EVIHGT',
            benchmark=manifest['protocol']['benchmark'],candidate_manifest_digest=manifest['global_manifest_digest'],
            pairs=manifest['totals']['pairs'],diseases=manifest['totals']['queries'],ranking_k=manifest['protocol']['ranking_k'],
            primary_metrics=['ROC_AUC_P_VS_U','AP_P_VS_U'],protocol_digest=manifest['protocol_digest']))
    old_roots=list(dict.fromkeys([p['output_root'],p.get('old_output_root',p['output_root'])]))
    for key in ('full_manifests','deferred_units','deferred_status','old_output_root'):p.pop(key,None)
    spec=dict(layer1=deepcopy(layer1['protocol']),layer2=full['protocol'],
        cold_drug=cold['cold_drug']['protocol'],cold_disease=cold['cold_disease']['protocol'])
    p.update(format=PLAN_FORMAT,amendment_id=amendment_id,graph_batch_size=1,cells=cells,checkpoint_binding_digest=digest(cells),
        units=units,protocol=spec,protocol_digest=digest(spec),execution_authorized=False,
        readiness='READY FOR FINAL FORMAL EXECUTION',output_root='outputs/phase8_formal/evaluation/'+amendment_id.lower()+'_final_v1',
        superseded_output_roots=old_roots,old_partial_score_reuse=False,scientific_results_visibility_during_execution=False,
        scientific_protocol_selection_basis='USER_FROZEN_COMPUTATIONAL_FEASIBILITY; NO_MODEL_RESULTS',
        manifest_bindings={'A':layer1['global_manifest_digest'],'B':full['global_manifest_digest'],
            'C':cold['cold_drug']['global_manifest_digest'],'D':cold['cold_disease']['global_manifest_digest']},
        total_model_forwards=sum(u['pairs'] for u in units),execution_order=['A','B','C','D'],
        stage_execution='SERIAL; GRAPH_BATCH_SIZE_1; NO_APPROXIMATION',
        candidate_stream_adapter='evi_hgt.evaluation.final_protocol.candidate_stream',
        result_schema_contract='QUERY_COMMIT_BINDING_PLUS_COMPLETE_STAGE_COUNTS; ONLY_FROZEN_RANKING_K; NO_TERMINAL_SCIENTIFIC_VALUES')
    return p


def validate_plan(plan,*,expected_digest):
    require(digest(plan)==expected_digest and plan['format']==PLAN_FORMAT,'superseded/mutated execution plan')
    require(plan['formal_run_id']==FORMAL_RUN_ID and plan['formal_test_count']==1 and plan['graph_batch_size']==1 and
        plan['execution_authorized'] is False,'Formal state/authorization/batch changed')
    require(plan['checkpoint_binding_digest']==digest(canonical_cells(plan['cells'])) and
        plan['protocol_digest']==digest(plan['protocol']),'checkpoint/protocol binding changed')
    require(plan['output_root'] not in plan['superseded_output_roots'] and plan['old_partial_score_reuse'] is False,
        'superseded partial output reuse forbidden')
    units=plan['units'];require(len(units)==9 and [u['layer'] for u in units]==['A']*6+['B','C','D'], 'stage matrix changed')
    require(plan['total_model_forwards']==sum(u['pairs'] for u in units),'forward census changed')
    for u in units:
        stage=u['layer'];binding=u.get('comparison_manifest_digest',u.get('candidate_manifest_digest'))
        require(binding==plan['manifest_bindings'][stage],'unit manifest changed')
        require(u['ranking_k']==([5,10,20,50] if stage=='B' else [5,10]),'reporting K changed')
        if stage!='A':require(u['model_variant']=='FULL_EVIHGT','BASE/ablation retrieval forbidden')
        if stage=='B':require(u['diseases']==6 and u['benchmark']==FULL_LABEL,'population-wide full retrieval forbidden')
        if stage in ('C','D'):require(u['benchmark']==COLD_LABEL,'full Cold retrieval forbidden')
    require(plan['representation_sha256']==REPRESENTATION_SHA and plan['cold_policy_digest']==COLD_POLICY,'Cold/indexed binding changed')


def validate_resume(current,previous,*,expected_digest):
    validate_plan(current,expected_digest=expected_digest)
    require(digest(previous)==expected_digest,'old protocol/manifest/plan/checkpoint/source cannot resume into new protocol')


def candidate_stream(manifest):
    """Frozen identities only. Graph assembly and scoring remain the original routes."""
    from evi_hgt.evaluation.candidates import DiseaseCandidateSet
    validate_manifest(manifest,expected_digest=manifest['global_manifest_digest'])
    for row in manifest['queries']:
        yield DiseaseCandidateSet(split_kind=manifest['protocol']['split'],partition='test',channel='support',
            disease_id=row['disease_id'],candidate_drug_ids=tuple(row['candidate_drug_ids']),
            held_out_relevant_drug_ids=tuple(row['positive_drug_ids']),training_known_excluded_drug_ids=(),
            candidate_protocol_digest=manifest['protocol_digest'],candidate_membership_digest=row['candidate_order_digest'],
            definition_digest=row['candidate_digest'])


def budget(pairs,rate,startup):
    result=engineering_budget(pairs,rate,startup,margin=1.20)
    result['gate']='PASS' if result['estimated_seconds']<10800 else 'FAIL'
    return result
