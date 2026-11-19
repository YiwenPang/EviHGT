"""Build new, score-blind cold-start inputs for the frozen portable runner."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import shutil
import sys
import time

import polars as pl

sys.dont_write_bytecode = True
from locations import CACHE, D1_FEATURES, FROZEN, WORKSPACE, check_frozen
CASES = ('cold_drug', 'cold_disease', 'double_cold', 'first_approval_year_cohort')
PARTS = ('train', 'validation', 'test')
SEED = 42
PAIR_COLS = ('drug_id', 'disease_id')
PAIR_SCHEMA = {key: pl.String for key in PAIR_COLS}


def file_sha(path):
    with Path(path).open('rb') as handle:
        import hashlib
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def key(*parts):
    return int(sha256(json.dumps(parts, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()[:16], 16)


def entity_partition(entity, axis):
    n = key('EVIHGT_FROZEN_DEVELOPMENT_COLD_V1', SEED, axis, entity)
    return 'test' if 10 * n < 2**64 else 'validation' if 5 * n < 2**64 else 'train'


def exclusions_by_disease(pairs, drugs, diseases):
    out = defaultdict(set)
    for drug, disease in pairs:
        if drug in drugs and disease in diseases:
            out[disease].add(drug)
    return out


def panel_frame(positives, us, relation):
    return pl.DataFrame([(a,b,'P',relation) for a,b in sorted(positives)] +
                        [(a,b,'U',None) for a,b in sorted(us)], orient='row',
                        schema={**PAIR_SCHEMA, 'role':pl.String, 'raw_relation':pl.String})


def make_panels(support, contra, drugs, excluded, partition):
    diseases = {d for _,d in support | contra}
    deny = exclusions_by_disease(excluded, drugs, diseases)
    drug_order = sorted(drugs)
    us = set()
    for disease in sorted(diseases):
        candidates = [drug for drug in drug_order if drug not in deny[disease]]
        # One shared set per disease for both channels, U64 without replacement.
        draw = random.Random(key('PANEL_U64_V1', SEED, partition, disease))
        us.update((drug,disease) for drug in draw.sample(candidates, min(64, len(candidates))))
    return panel_frame(support, us, 'INDICATION'), panel_frame(contra, us, 'CONTRAINDICATION'), us


def make_schedule(support, contra, drugs, diseases, excluded, case):
    G = min(len(support),len(contra))
    if G == 0 or support & contra:
        raise ValueError('Both disjoint positive channel streams are required by this input adapter')
    if any(a not in drugs or b not in diseases for a,b in support | contra):
        raise ValueError('Training positive outside its assigned endpoint cohort')
    deny = exclusions_by_disease(excluded, drugs, diseases)
    drug_order, disease_order = sorted(drugs), sorted(diseases)
    if not drugs or not diseases or len(drugs)*len(diseases) <= sum(map(len,deny.values())):
        raise ValueError('No eligible unlabeled training pair')
    local_legal = {d: [a for a in drug_order if a not in deny[d]] for d in {b for _,b in support}}
    rows, records = [], []
    for cycle in (1,2,3):
        seed = key('TRAIN_3G_V1', SEED, case, cycle)
        draw = random.Random(seed)
        global_u = []
        while len(global_u) < G:
            drug, disease = draw.choice(drug_order), draw.choice(disease_order)
            if drug not in deny[disease]:
                global_u.append((drug,disease))
        stream = [(p,'support') for p in sorted(support)] + [(p,'unsuitability') for p in sorted(contra)] + [(p,'U') for p in global_u]
        random.Random(seed).shuffle(stream)
        sp, cp, up = ([p for p,r in stream if r == role] for role in ('support','unsuitability','U'))
        for i in range(G):
            step = (cycle-1)*G+i+1
            ss = sp[i*len(sp)//G:(i+1)*len(sp)//G]
            cc = cp[i*len(cp)//G:(i+1)*len(cp)//G]
            upair = up[i]
            group = ss + cc + [upair]
            rows.extend((step,cycle,pos,*pair,role) for pos,(pair,role) in enumerate(zip(
                group, ['support']*len(ss)+['unsuitability']*len(cc)+['U'], strict=True)))
            if cycle == 1:
                continue
            local, statuses, seeds, extras = {}, {}, {}, []
            ix = step-G-1
            for disease in dict.fromkeys(d for _,d in ss):
                sub_seed = key('CONDITIONAL_SUPPORT_U_V1', SEED, case, ix, disease)
                seeds[disease] = sub_seed
                if not local_legal[disease]:
                    statuses[disease], local[disease] = 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK', []
                    continue
                pair = (random.Random(sub_seed).choice(local_legal[disease]),disease)
                statuses[disease], local[disease] = 'DISEASE_CONDITIONAL_UNIFORM', [list(pair)]
                if pair not in group and list(pair) not in extras:
                    extras.append(list(pair))
            records.append(dict(logical_step=step, group_index=ix, support_positive_pairs=list(map(list,ss)),
                contra_positive_pairs=list(map(list,cc)), original_global_u=list(upair), original_support_u_count=1,
                disease_unlabeled_pairs=local, disease_status=statuses, draw_seed_by_disease=seeds, extra_pairs=extras))
    frame = pl.DataFrame(rows, orient='row', schema={'step':pl.Int64,'cycle':pl.Int64,'position':pl.Int64,
                          **PAIR_SCHEMA, 'role':pl.String})
    return G, frame, records


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf8')


def prepare(case):
    from approval_cohort import approval_partitions
    from graph_builder import build_graph
    start = time.monotonic()
    check_frozen()
    if case not in CASES:
        raise ValueError(case)
    CACHE.mkdir(parents=True,exist_ok=True)
    final = CACHE / case
    if final.exists():
        raise FileExistsError(final)
    base = WORKSPACE / 'data/processed/phase85_part2'
    node_path = WORKSPACE / 'data/processed/phase3/v1/nodes/node_index.parquet'
    inventory = {}
    def read_source(path, expected=None):
        digest = file_sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Source digest changed: '+str(path))
        inventory[str(path)] = digest
        return path
    def source_json(path, expected=None):
        return json.loads(read_source(path,expected).read_text(encoding='utf8'))
    dev = source_json(base/'development_manifest.json')
    universe = source_json(base/'training_u_manifest.json')
    raw = source_json(base/'features_v2/manifest.json')
    def pairs(name):
        return set(pl.read_parquet(read_source(base/name,dev['files'][name]),columns=PAIR_COLS).iter_rows())
    support, contra = pairs('development_positives.parquet'), pairs('auxiliary_positives.parquet')
    protected = pairs('all_forbidden_pairs.parquet')
    observed = pairs('observed_pair_exclusions.parquet')
    if (support | contra) & protected or not (support | contra) <= observed:
        raise ValueError('Original score-blind development/observed/protected contracts differ')
    nodes = pl.read_parquet(read_source(node_path)).sort('global_node_id')
    node_drugs = set(nodes.filter(node_type='DRG')['node_id'])
    node_diseases = set(nodes.filter(node_type='DIS')['node_id'])
    drugs, diseases = set(universe['eligible_drugs']), set(universe['eligible_diseases'])
    if not drugs <= node_drugs or not diseases <= node_diseases:
        raise ValueError('Candidate IDs are not canonical')
    split_audit = dict(protocol='EVIHGT_FROZEN_DEVELOPMENT_COLD_V1', seed=SEED, train_fraction_nominal=.8,
                       stable_sha256_entity_grouping=True, existing_protected_pair_roles_read=False)
    drug_part, disease_part = {}, {}
    if case == 'first_approval_year_cohort':
        drug_part, split_audit = approval_partitions(WORKSPACE,drugs,node_drugs)
        read_source(WORKSPACE/split_audit['source'],split_audit['source_sha256'])
    if case in ('cold_drug','double_cold'):
        drug_part = {d:entity_partition(d,'drug_id') for d in node_drugs}
    if case in ('cold_disease','double_cold'):
        disease_part = {d:entity_partition(d,'disease_id') for d in node_diseases}
    def assigned(positive,part):
        return {(a,b) for a,b in positive if (not drug_part or drug_part[a]==part) and
                                           (not disease_part or disease_part[b]==part)}
    ds = {p:{d for d in drugs if not drug_part or drug_part[d] == p} for p in PARTS}
    ss = {p:{d for d in diseases if not disease_part or disease_part[d] == p} for p in PARTS}
    positives = {p:(assigned(support,p),assigned(contra,p)) for p in PARTS}
    if any(not a or not b for a,b in positives.values()):
        raise ValueError('A required support or contra cohort is empty')
    dev_support, dev_contra, dev_u = make_panels(*positives['validation'],ds['validation'], observed | protected, 'validation')
    test_support, test_contra, test_u = make_panels(*positives['test'],ds['test'], observed | protected | dev_u, 'test')
    if not dev_u or not test_u or dev_u & test_u:
        raise ValueError('Disjoint development and test U panels are required')
    blocked = protected | ((support | contra) - set.union(*positives['train'])) | dev_u | test_u
    G, groups, conditional = make_schedule(*positives['train'],ds['train'],ss['train'], observed | blocked,case)
    d_hold = {k for k,v in drug_part.items() if v!='train'}
    s_hold = {k for k,v in disease_part.items() if v!='train'}
    print(f'[{case}] 训练/验证/测试 P（支持、禁忌）：{[(len(a),len(b)) for a,b in positives.values()]}；计划更新 {3*G:,} 次。',flush=True)
    stage = CACHE / f'{case}.prepare_{os.getpid()}'
    stage.mkdir(exist_ok=False)
    graph = build_graph(WORKSPACE,stage,d_hold,s_hold,blocked)
    print(f'[{case}] 图谱隔离完成，写入采样计划和特征。',flush=True)
    files = graph.pop('files')
    graph_audit = graph.pop('graph_audit')
    def put(name, writer):
        target = stage/name
        target.parent.mkdir(parents=True,exist_ok=True)
        writer(target)
        files[name] = file_sha(target)
        return name
    def js(name,value): return put(name,lambda p:write_json(p,value))
    def pq(name,frame): return put(name,frame.write_parquet)
    def copy(name,src,expected):
        read_source(src,expected)
        return put(name,lambda p:shutil.copyfile(src,p))
    sys.path.insert(0,str(FROZEN/'src'))
    from evi_hgt.data.research_export import _export_d1
    import numpy as np
    def array(name,values): return put(name,lambda p:np.save(p,values,allow_pickle=False))
    d1root = D1_FEATURES
    pretrained = _export_d1(d1root,nodes,node_path,copy,js,array)
    raw_features = {k:copy('raw_features/'+k+'.npy',base/'features_v2'/f'{k}.npy',raw['files'][k+'.npy'])
                    for k in ('drug_ids','drug_values','disease_ids','disease_values')}
    train = [(a,b,role,relation) for (role,relation,ps) in (
        ('support','INDICATION',positives['train'][0]),('unsuitability','CONTRAINDICATION',positives['train'][1]))
        for a,b in sorted(ps)]
    projection = dev_contra.filter(role='P')
    assert test_support.schema == test_contra.schema
    test_support.write_parquet(stage/'test_support.parquet')
    test_contra.write_parquet(stage/'test_contra.parquet')
    # Test label files remain outside the input inventory loaded by the trainer.
    test_files = {name:file_sha(stage/name) for name in ('test_support.parquet','test_contra.parquet')}
    audit = dict(case=case,split=split_audit,graph=graph_audit,graph_drugs_excluded=len(d_hold),
        graph_diseases_excluded=len(s_hold),retained_P={p:dict(support=len(a),contra=len(b)) for p,(a,b) in positives.items()},
        excluded_P=dict(support=len(support)-sum(len(a) for a,b in positives.values()),
                       contra=len(contra)-sum(len(b) for a,b in positives.values())),
        dev_U=len(dev_u),test_U=len(test_u),test_files=test_files,train_updates=3*G,
        candidates='Disease-wise shared U64; up to 64 distinct unobserved drugs per disease and partition; later cohorts use frozen static nonclinical features.',
        test_labels_used_by_optimizer=False, original_protected_membership_roles_read=False,
        ready_after_seconds=time.monotonic()-start,source_sha256=inventory)
    # The frozen development scorer gets its shared U from validation.
    manifest = dict(schema='EVIHGT_PORTABLE_CORPUS_V1',training_seed=SEED,token_buckets=8192,**graph,
        blocked_pairs=pq('blocked_pairs.parquet',pl.DataFrame(sorted(blocked),schema=PAIR_SCHEMA,orient='row')),
        observed_pairs=pq('observed_pairs.parquet',pl.DataFrame(sorted(observed),schema=PAIR_SCHEMA,orient='row')),
        eligibility=js('eligibility.json',dict(eligible_drugs=sorted(ds['train']),eligible_diseases=sorted(ss['train']))),
        training_pairs=pq('training_pairs.parquet',pl.DataFrame(train,orient='row',schema={**PAIR_SCHEMA,'channel':pl.String,'raw_relation':pl.String})),
        panels=dict(validation=pq('validation.parquet',dev_support),auxiliary_validation=pq('auxiliary_validation.parquet',projection)),
        raw_features=raw_features,pretrained=pretrained,
        schedule=dict(G=G,groups=pq('global_groups.parquet',groups),conditional_records=js('conditional_records.json',conditional),
                      sampling_provenance=dict(namespace='COLD_START_3G_V1',case=case,seed=SEED,full_positive_coverage=True)),
        semantics={k:True for k in ('canonical_entities_verified','canonical_facts_deduplicated','evidence_records_deduplicated',
                    'clinical_labels_verified','target_aliases_resolved','disease_gene_label_leakage_removed',
                    'split_mask_applied','features_label_independent')},files=files)
    manifest['semantics']['unknown_is_negative'] = False
    manifest['cold_case_receipt'] = js('preparation_audit.json',audit)
    write_json(stage/'manifest.json',manifest)
    os.rename(stage,final)
    print(f'[{case}] 输入已封存：{final}',flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=CASES)
    prepare(parser.parse_args().case)
