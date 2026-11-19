"""V2 Layer2 evaluability amendment: labels/identities only, no model scoring."""
from copy import deepcopy

from evi_hgt.evaluation.formal_redesign import digest,require
from evi_hgt.evaluation.emergency_u32 import validate_manifest as validate_u32,seal
from evi_hgt.evaluation import final_protocol as v1

PROTOCOL_ID='STANDARD_FULL_RETRIEVAL_EVALUABLE_STRATIFIED_6_V2'
PLAN_FORMAT='EVIHGT_FINAL_FORMAL_EVALUABLE_FOUR_STAGE_V2'
LABEL='DETERMINISTIC RETRIEVAL-EVALUABLE STRATIFIED FULL-CANDIDATE RETRIEVAL SUBSET'


def selection(rows):
    counts=sorted([[r['disease_id'],r['positive_count']] for r in rows])
    require(bool(counts) and len({d for d,n in counts})==len(counts),'disease source census invalid')
    require(all(isinstance(n,int) and not isinstance(n,bool) and n>=0 for d,n in counts),'positive count invalid')
    ordered=sorted([r for r in counts if r[1]>=1],key=lambda r:(r[1],r[0]))
    require(len(ordered)>=6,'fewer than six retrieval-evaluable diseases; block without changing protocol')
    q,r=divmod(len(ordered),3);start=0;strata=[]
    for i,name in enumerate(('LOW','MID','HIGH')):
        end=start+q+int(i<r);ids=[row[0] for row in ordered[start:end]]
        chosen=sorted(ids,key=lambda d:v1.hmac_key(PROTOCOL_ID,d))[:2]
        strata.append(dict(name=name,start_inclusive=start,end_exclusive=end,size=len(ids),disease_ids=ids,
            positive_count_min=ordered[start][1],positive_count_max=ordered[end-1][1],selected_disease_ids=chosen))
        start=end
    return dict(protocol_version=PROTOCOL_ID,N_all=len(counts),N_zero_positive=sum(n==0 for d,n in counts),
        N_retrieval_evaluable=len(ordered),all_disease_set_digest=digest([d for d,n in counts]),
        all_positive_count_vector=counts,all_positive_count_vector_digest=digest(counts),
        retrieval_evaluable_disease_set_digest=digest(sorted(d for d,n in ordered)),
        sorted_positive_count_vector=ordered,positive_count_vector_digest=digest(ordered),
        eligibility='ELIGIBLE_STANDARD_FORMAL_DISEASE_AND_ELIGIBLE_SUPPORT_POSITIVE_COUNT_AT_LEAST_1',
        sorting_rule=['POSITIVE_COUNT_ASCENDING','CANONICAL_DISEASE_ID_ASCENDING'],
        partition_rule='DIVMOD_N_EVALUABLE_3; EXTRA_ONE_TO_EARLIEST_STRATA; CONTIGUOUS_HALF_OPEN_BOUNDARIES',
        stratum_interpretation='POSITIVE_COUNT_RANK_STRATA_ONLY; NOT_BIOLOGICAL_CATEGORIES',strata=strata,
        tied_counts_crossing_boundary=[strata[i]['positive_count_max']==strata[i+1]['positive_count_min'] for i in range(2)],
        hmac_key='UTF8_PROTOCOL_VERSION',hmac_message='UTF8_COMPACT_JSON_ARRAY_CANONICAL_DISEASE_ID',
        selection_order=['HMAC_SHA256_DIGEST_ASCENDING','CANONICAL_DISEASE_ID_ASCENDING'],selected_per_stratum=2,
        selected_disease_ids=sorted(d for s in strata for d in s['selected_disease_ids']))


def specification():
    spec=v1.specification('standard')
    spec.update(protocol_version=PROTOCOL_ID,benchmark=LABEL,
        retrieval_evaluable_requirement='ELIGIBLE_SUPPORT_POSITIVE_COUNT_AT_LEAST_1; METRIC_DEFINABILITY_NOT_PERFORMANCE',
        aggregation='SIX_DISEASE_DETERMINISTIC_EVALUABLE_STRATIFIED_SUBSET_MACRO; NOT_POPULATION_ESTIMATE')
    return spec


def build_manifest(layer1,sources,*,source_binding,checkpoint,amendment_id,superseded_digest,progress=False):
    validate_u32(layer1);require(layer1['source_binding']==source_binding,'Standard source data identity changed')
    selected=selection(layer1['diseases']);chosen=set(selected['selected_disease_ids'])
    strata={d:s['name'] for s in selected['strata'] for d in s['selected_disease_ids']}
    frozen={r['disease_id']:r for r in layer1['diseases']};seen=set();rows=[]
    from tqdm import tqdm
    for source in tqdm(sources,total=len(frozen),desc='[EVALUABLE_V2_MANIFEST][RUNNING]',unit='disease',
            disable=not progress,dynamic_ncols=True,leave=True,mininterval=2):
        d=source.disease_id;require(source.split_kind=='standard' and d in frozen and d not in seen,'Standard disease source changed')
        seen.add(d);identity=v1.source_row(source)
        require(all(identity[k]==frozen[d][k] for k in identity),'positive/full eligible source identity changed')
        if d in chosen:rows.append(v1.candidate_row(source,PROTOCOL_ID,None,strata[d]))
    require(seen==set(frozen),'Standard census missing diseases')
    value=v1.commitment(rows,'standard',selected,source_binding,checkpoint,amendment_id,superseded_digest)
    value.pop('global_manifest_digest');spec=specification()
    value.update(format='EVIHGT_EVALUABLE_FULL_CANDIDATE_MANIFEST_V2',protocol=spec,protocol_digest=digest(spec),
        supersession=dict(status='SUPERSEDED BEFORE SCIENTIFIC EXECUTION',previous_manifest_digest=superseded_digest,
            previous_artifacts_preserved=True,previous_scientific_scoring_executed=False,previous_partial_results_reusable=False))
    value=seal(value,'global_manifest_digest');validate_manifest(value,expected_digest=value['global_manifest_digest'])
    return value


def validate_manifest(value,*,expected_digest):
    require(value['global_manifest_digest']==expected_digest and digest({k:v for k,v in value.items()
        if k!='global_manifest_digest'})==expected_digest,'V2 external/payload manifest changed')
    require(value['format']=='EVIHGT_EVALUABLE_FULL_CANDIDATE_MANIFEST_V2' and value['protocol']==specification()
        and value['protocol_digest']==digest(specification()),'V1/V2 protocol mismatch')
    selected=value['selection'];recomputed=selection([dict(disease_id=d,positive_count=n) for d,n in selected['all_positive_count_vector']])
    require(selected==recomputed,'evaluability census/strata/disease selection changed')
    require(selected['N_all']==selected['N_zero_positive']+selected['N_retrieval_evaluable'],'evaluability census inconsistent')
    rows=value['queries'];ids=[r['disease_id'] for r in rows]
    require(ids==selected['selected_disease_ids'] and len(ids)==6 and value['query_set_digest']==digest(ids),'six-disease set/order changed')
    strata={d:s['name'] for s in selected['strata'] for d in s['selected_disease_ids']}
    counts=dict(selected['all_positive_count_vector'])
    for row in rows:
        require(row['candidate_digest']==digest({k:v for k,v in row.items() if k!='candidate_digest'}),'query digest changed')
        p,u,c=(row[k] for k in ('positive_drug_ids','u_drug_ids','candidate_drug_ids'))
        require(p==sorted(set(p)) and u==sorted(set(u)) and not set(p)&set(u) and c==sorted(set(p)|set(u)),'P/U/order changed')
        require((len(p),len(u),len(c))==(row['positive_count'],row['u_count'],row['total_candidate_count']) and
            len(c)==row['eligible_candidate_count'],'full candidate universe was sampled/truncated')
        require(row['positive_count']==counts[row['disease_id']] and row['positive_count']>=1 and row['u_count']>=1,'selected metrics not evaluable')
        require((digest(p),digest(u),digest(c))==(row['positive_digest'],row['u_digest'],row['candidate_order_digest']) and
            digest(c)==row['eligible_membership_digest'],'full membership/ordering changed')
        require(row['protocol_version']==PROTOCOL_ID and row['query_id']==row['disease_id'] and row['stratum']==strata[row['disease_id']],
            'query protocol/stratum changed')
    require(value['totals']==dict(queries=6,positives=sum(r['positive_count'] for r in rows),u=sum(r['u_count'] for r in rows),
        pairs=sum(r['total_candidate_count'] for r in rows)) and value['candidate_distribution']==v1.distribution(rows),'V2 totals changed')
    require(value['checkpoint']['split']=='standard' and value['checkpoint']['model_variant']=='FULL_EVIHGT' and
        value['graph_batch_size']==1 and value['representation_sha256']==v1.REPRESENTATION_SHA,'checkpoint/indexed/batch binding changed')
    require(value['cold_policy_digest'] is None and value['cold_policy'] is None,'Standard/Cold protocol mix')
    require(value['supersession']==dict(status='SUPERSEDED BEFORE SCIENTIFIC EXECUTION',
        previous_manifest_digest=value['supersession']['previous_manifest_digest'],previous_artifacts_preserved=True,
        previous_scientific_scoring_executed=False,previous_partial_results_reusable=False),'V1 disposition changed')


def build_plan(previous,manifest,*,amendment_id):
    v1.validate_plan(previous,expected_digest=digest(previous))
    validate_manifest(manifest,expected_digest=manifest['global_manifest_digest'])
    require(manifest['source_binding']==previous['data_binding'] and manifest['amendment_id']==amendment_id,'source/amendment mismatch')
    require(manifest['checkpoint']==v1.checkpoint_binding(previous['cells'][0]),'FULL checkpoint changed')
    plan=deepcopy(previous);spec=deepcopy(previous['protocol']);spec['layer2']=deepcopy(manifest['protocol'])
    for unit in plan['units']:
        if unit['layer']=='B':unit.update(benchmark=LABEL,candidate_manifest_digest=manifest['global_manifest_digest'],
            pairs=manifest['totals']['pairs'],diseases=6,protocol_digest=manifest['protocol_digest'])
    plan.update(format=PLAN_FORMAT,amendment_id=amendment_id,protocol=spec,protocol_digest=digest(spec),
        output_root='outputs/phase8_formal/evaluation/'+amendment_id.lower()+'_evaluable_v2',
        superseded_output_roots=list(dict.fromkeys(previous['superseded_output_roots']+[previous['output_root']])),
        scientific_protocol_selection_basis='LAYER2_METRIC_EVALUABILITY_ONLY; NO_PERFORMANCE_FILTERING',
        candidate_stream_adapter='evi_hgt.evaluation.evaluable_protocol.candidate_stream',
        overnight_orchestrator='A_TO_B_TO_C_TO_D; VERIFY_COMPLETE_RECEIPT_BEFORE_NEXT; STOP_FIRST_FAILURE; NO_AUTO_RETRY')
    plan['manifest_bindings']['B']=manifest['global_manifest_digest']
    plan['total_model_forwards']=sum(u['pairs'] for u in plan['units'])
    validate_plan(plan,expected_digest=digest(plan))
    return plan


def validate_plan(plan,*,expected_digest):
    require(digest(plan)==expected_digest and plan['format']==PLAN_FORMAT,'superseded/mutated V2 plan')
    require(plan['protocol']['layer2']==specification(),'V2 Layer2 protocol changed')
    projected=deepcopy(plan);projected['format']=v1.PLAN_FORMAT
    for u in projected['units']:
        if u['layer']=='B':
            require(u['protocol_digest']==digest(specification()) and u['benchmark']==LABEL,'V2 unit protocol changed')
            u['benchmark']=v1.FULL_LABEL
    v1.validate_plan(projected,expected_digest=digest(projected))


def validate_resume(current,previous,*,expected_digest):
    validate_plan(current,expected_digest=expected_digest)
    require(digest(previous)==expected_digest,'V1/old U1024/full Cold or changed binding cannot resume into V2')


def candidate_stream(manifest):
    from evi_hgt.evaluation.candidates import DiseaseCandidateSet
    validate_manifest(manifest,expected_digest=manifest['global_manifest_digest'])
    for row in manifest['queries']:
        yield DiseaseCandidateSet(split_kind='standard',partition='test',channel='support',disease_id=row['disease_id'],
            candidate_drug_ids=tuple(row['candidate_drug_ids']),held_out_relevant_drug_ids=tuple(row['positive_drug_ids']),
            training_known_excluded_drug_ids=(),candidate_protocol_digest=manifest['protocol_digest'],
            candidate_membership_digest=row['candidate_order_digest'],definition_digest=row['candidate_digest'])
