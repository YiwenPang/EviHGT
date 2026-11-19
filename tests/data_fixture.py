"""Synthetic corpus fixtures for software tests."""
from pathlib import Path
import json
import numpy as np
import polars as pl

from evi_hgt.data.common import sha256
from evi_hgt.data.pretrained import ENCODER_SPECS
from evi_hgt.data.raw_features import DRUG_DIM, DISEASE_DIM


def make_corpus(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    def js(name, value):
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        (root/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
        return name
    def pq(name, frame):
        frame.write_parquet(root/name)
        return name
    def npy(name, values):
        np.save(root/name, values, allow_pickle=False)
        return name
    node_defs = [('A0','ANA'),('B0','BPO'),('C0','CCO'),('S0','DIS'),('S1','DIS'),('S2','DIS'),
                 ('D0','DRG'),('D1','DRG'),('D2','DRG'),('D3','DRG'),('E0','EXP'),('G0','GEN'),
                 ('M0','MFN'),('P0','PHE'),('W0','PWY')]
    ids = {name:i for i,(name,_) in enumerate(node_defs)}
    kinds = dict(node_defs)
    nodes = pl.DataFrame({'global_node_id':list(range(len(node_defs))),
                         'node_id':[v[0] for v in node_defs], 'node_type':[v[1] for v in node_defs]})
    catalog = [('ANA','EXPRESSION_ABSENT','GEN'),('DRG','INTERACTS','GEN'),
               ('GEN','ASSOCIATED','DIS'),('DRG','INDICATION','DIS'),('DRG','CONTRAINDICATION','DIS')]
    specs = [('A0','G0','EXPRESSION_ABSENT')] + [(d,'G0','INTERACTS') for d in ('D0','D1','D2','D3')]
    specs += [('G0',d,'ASSOCIATED') for d in ('S0','S1','S2')]
    specs += [('D0','S0','INDICATION'),('D1','S1','CONTRAINDICATION')]
    fact_rows, evidence_rows, traversals = [], [], []
    target_rows = []
    for fid,(src,dst,relation) in enumerate(specs):
        fact_rows.append(dict(canonical_fact_id=fid, source_global_id=ids[src],target_global_id=ids[dst],
            source_node_id=src,target_node_id=dst,source_node_type=kinds[src],target_node_type=kinds[dst],
            relation=relation,undirected=True))
        evidence_rows.append(dict(evidence_record_id=fid,canonical_fact_id=fid,properties=json.dumps({'evidence_count':1}),
            raw_direct_sources=['SYNTHETIC_TEST_ONLY'],raw_indirect_sources=[],source_row_number=fid,
            source_component_path='synthetic_test_only',overlap_category=None,
            edge_support_observed=False,edge_counter_evidence_observed=relation=='EXPRESSION_ABSENT'))
        traversals += [(ids[src],ids[dst],fid),(ids[dst],ids[src],fid)]
        if kinds[src] == 'DRG' and kinds[dst] == 'DIS':
            target_rows.append(dict(drug_id=src,disease_id=dst,canonical_fact_id=fid))
    facts = pl.DataFrame(fact_rows)
    evidence = pl.DataFrame(evidence_rows, schema_overrides={'raw_direct_sources':pl.List(pl.String),
        'raw_indirect_sources':pl.List(pl.String),'overlap_category':pl.String})
    traversals.sort(key=lambda v:v[0])
    ptr = np.zeros(len(nodes)+1,dtype=np.int64)
    for src,_,_ in traversals:
        ptr[src+1] += 1
    ptr = np.cumsum(ptr)
    manifest = dict(schema='EVIHGT_PORTABLE_CORPUS_V1',purpose='SYNTHETIC_TEST_ONLY',training_seed=42,token_buckets=8192,
        semantics={key:True for key in ('canonical_entities_verified','canonical_facts_deduplicated','evidence_records_deduplicated',
        'clinical_labels_verified','target_aliases_resolved','disease_gene_label_leakage_removed','split_mask_applied','features_label_independent')})
    manifest['semantics']['unknown_is_negative'] = False
    manifest['nodes'] = pq('nodes.parquet', nodes)
    manifest['edge_catalog'] = js('edge_catalog.json',[dict(source_node_type=s,relation=r,target_node_type=t) for s,r,t in catalog])
    manifest['fact_files'] = [dict(path=pq('facts.parquet',facts),component='synthetic',rows=len(facts))]
    manifest['evidence_files'] = [dict(path=pq('evidence.parquet',evidence),component='synthetic',rows=len(evidence))]
    manifest['csr'] = {name:npy(name+'.npy',value) for name,value in
        {'indptr':ptr,'indices':np.array([v[1] for v in traversals],dtype=np.int64),
         'fact_ids':np.array([v[2] for v in traversals],dtype=np.int64)}.items()}
    manifest['target_map'] = pq('target_map.parquet',pl.DataFrame(target_rows))
    manifest['forbidden_facts'] = pq('forbidden.parquet',pl.DataFrame({'canonical_fact_id':[]},schema={'canonical_fact_id':pl.Int64}))
    manifest['blocked_pairs'] = pq('blocked.parquet',pl.DataFrame({'drug_id':['D0','D1','D2'], 'disease_id':['S2']*3}))
    manifest['observed_pairs'] = pq('observed.parquet',pl.DataFrame({'drug_id':['D0','D1','D0','D1'], 'disease_id':['S0','S1','S2','S2']}))
    manifest['eligibility'] = js('eligibility.json',dict(eligible_drugs=['D0','D1','D2','D3'],eligible_diseases=['S0','S1','S2']))
    manifest['training_pairs'] = pq('training.parquet',pl.DataFrame(dict(drug_id=['D0','D1'],disease_id=['S0','S1'],
        channel=['support','unsuitability'],raw_relation=['INDICATION','CONTRAINDICATION'])))
    manifest['panels'] = {
        'validation':pq('validation.parquet',pl.DataFrame(dict(drug_id=['D0','D2'],disease_id=['S2','S2'],
            role=['P','U'],raw_relation=['INDICATION',None]))),
        'auxiliary_validation':pq('auxiliary_validation.parquet',pl.DataFrame(dict(drug_id=['D1'],disease_id=['S2'],
            role=['P'],raw_relation=['CONTRAINDICATION'])))}
    raw, pretrained = {}, {}
    for prefix,kind,width in (('drug','DRG',DRUG_DIM),('disease','DIS',DISEASE_DIM)):
        entities = nodes.filter(pl.col('node_type') == kind)['global_node_id'].to_numpy()
        values = np.zeros((len(entities),width),dtype=np.uint8)
        values[:,-2:] = 1
        values[:,0] = 1
        raw[prefix+'_ids'] = npy(prefix+'_ids.npy',entities)
        raw[prefix+'_values'] = npy(prefix+'_raw.npy',values)
        pretrained[prefix+'_ids'] = raw[prefix+'_ids']
        pretrained[prefix+'_values'] = npy(prefix+'_d1.npy',np.full((len(entities),768),0.01,dtype=np.float32))
        pretrained[prefix+'_available'] = npy(prefix+'_available.npy',np.ones(len(entities),dtype=np.bool_))
    pretrained['metadata'] = js('d1_metadata.json',dict(schema='EVIHGT_D1_FEATURES_V1',purpose='SYNTHETIC_TEST_ONLY',
        clinical_text_encoded=False,supervised_dda_features_present=False,trained_node_embeddings_loaded=False,
        prediction_scores_loaded=False,optimizer_steps=0,width=768,fixed_encoder_specs=ENCODER_SPECS))
    manifest['raw_features'], manifest['pretrained'] = raw, pretrained
    rows = []
    for step in range(1,4):
        for position,(drug,disease,role) in enumerate([('D0','S0','support'),('D1','S1','unsuitability'),('D2','S0','U')]):
            rows.append(dict(step=step,cycle=step,position=position,drug_id=drug,disease_id=disease,role=role))
    records = [dict(logical_step=step,group_index=step-2,support_positive_pairs=[['D0','S0']],
        contra_positive_pairs=[['D1','S1']],original_global_u=['D2','S0'],original_support_u_count=1,
        disease_unlabeled_pairs={'S0':[['D3','S0']]},disease_status={'S0':'DISEASE_CONDITIONAL_UNIFORM'},
        draw_seed_by_disease={'S0':step},extra_pairs=[['D3','S0']]) for step in (2,3)]
    manifest['schedule'] = dict(G=1,groups=pq('global_groups.parquet',pl.DataFrame(rows)),
        conditional_records=js('conditional_records.json',records),sampling_provenance='SYNTHETIC_TEST_ONLY')
    manifest['files'] = {p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file()}
    js('manifest.json',manifest)
    return root/'manifest.json'
