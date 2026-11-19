"""Export a corpus from the research workspace.

Read the canonical inputs, training records, role-free exclusions and two
development panels required by the corpus loader.
"""
from pathlib import Path, PureWindowsPath
import gc
import json
import os
import re
import shutil
import uuid
import numpy as np
import polars as pl

from .common import read_json, sha256
from .projection import objects_in_array, field, reserved, project_training_assignments, digest as old_digest
from .corpus import load_corpus
from .pretrained import ENCODER_SPECS
from .prepare import destination_path


def verified(path):
    value = read_json(path)
    if value.get('digest') != old_digest({k:v for k,v in value.items() if k != 'digest'}):
        raise ValueError('Research artifact content identity changed: ' + str(path))
    return value


def export_research(research_root, cell, d1_root, output, *, recorded_root=None, progress=None):
    if not isinstance(cell,str) or re.fullmatch(r'R[0-2]F[0-4]',cell) is None:
        raise ValueError('Expected one original cell R0F0 through R2F4')
    root, d1_root, output = Path(research_root).resolve(), Path(d1_root).resolve(), Path(output).resolve()
    if not root.is_dir() or not d1_root.is_dir():
        raise FileNotFoundError('Explicit research and D1 feature directories must exist')
    if output.exists() or output == root or root.is_relative_to(output):
        raise FileExistsError('Choose a new derived output directory outside the original inputs')
    if progress:
        progress('research artifact verification',0,0,0)
    def source_path(value):
        # Absolute paths in old manifests are accepted only under the explicitly
        # supplied research root, or through an explicit recorded-root mapping.
        text = str(value)
        if recorded_root:
            original = PureWindowsPath(str(recorded_root))
            candidate = PureWindowsPath(text)
            if candidate.is_relative_to(original):
                text = candidate.relative_to(original).as_posix()
        p = Path(text)
        p = p.resolve() if p.is_absolute() else (root/p).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Unresolved original input; use --recorded-root for relocated manifests: '+text)
        return p
    repeat, fold = int(cell[1]), int(cell[3])
    sealed_path = root/'outputs/phase9/phase98/part2/sampling/ALL_PLANS_SEALED.json'
    sealed = verified(sealed_path)
    if sealed.get('status') != 'SEALED_15_CELLS':
        raise ValueError('The original three-cycle sampling plans are not sealed')
    selected = [v for v in sealed['cells'] if v['cell'] == cell]
    if len(selected) != 1:
        raise ValueError('Cell is absent or duplicated in the original plan seal')
    seal = selected[0]
    schedule_path, conditional_path = (source_path(seal[k+'_path']) for k in ('schedule','plan'))
    if sha256(schedule_path) != seal['schedule_sha256'] or sha256(conditional_path) != seal['plan_sha256']:
        raise ValueError('Original sealed schedule/conditional plan bytes changed')
    schedule, conditional = verified(schedule_path), verified(conditional_path)
    if (schedule['digest'] != seal['schedule_digest'] or conditional['digest'] != seal['plan_digest']
        or schedule['G'] != seal['G'] or schedule['total'] != 3*seal['G']
        or conditional['schedule_digest'] != schedule['digest'] or len(conditional['records']) != 2*seal['G']):
        raise ValueError('Original three-cycle schedule/plan identity differs')
    groups_path = source_path(schedule['groups_path'])
    if sha256(groups_path) != schedule['groups_sha256']:
        raise ValueError('Original ordered global group bytes changed')
    base = root/'data/processed/phase85_part2'
    development = verified(base/'development_manifest.json')
    features = verified(base/'features_v2/manifest.json')
    universe = verified(base/'training_u_manifest.json')
    keys = verified(root/'outputs/phase85_part3a_r2/cv_three_key_binding.json')
    census = read_json(root/'outputs/phase85_part3b1/cv_15_cell_static_census.json')
    frozen = next(v for v in census['cells'] if (v['repeat_index'],v['fold_index']) == (repeat,fold))
    def dev_path(name):
        p = base/name
        if sha256(p) != development['files'][name]:
            raise ValueError('Original role-free source changed: '+name)
        return p
    def pair_set(path):
        return set(pl.read_parquet(path,columns=['drug_id','disease_id']).iter_rows())
    panel_ids = pair_set(dev_path('development_panel.parquet'))
    auxiliary = pair_set(dev_path('auxiliary_positives.parquet'))
    forbidden_pairs = pair_set(dev_path('all_forbidden_pairs.parquet'))
    observed_path = dev_path('observed_pair_exclusions.parquet')
    direct = root/f'data/processed/phase85_part3b1/repeat_{repeat}_fold_{fold}/cell_manifest.json'
    domains = ('training','auxiliary_training','validation','auxiliary_validation')
    if direct.is_file():
        # Streaming sections avoid decoding outer_test/final/auxiliary_probe.
        parts = {name:[json.loads(v) for v in objects_in_array(direct,name)] for name in domains}
    else:
        entry = next(v for v in keys['selected_repeat_bindings'] if v['original_repeat_index'] == repeat)
        assignment = source_path(entry['fold_assignment_path'])
        if sha256(assignment) != entry['fold_assignment_sha256']:
            raise ValueError('Original fold assignment bytes changed')
        training,_ = project_training_assignments(assignment,fold=fold,repeat_key=frozen['repeat_key'])
        train_ids = {(v['drug_id'],v['disease_id']) for v in training}
        excluded = panel_ids-train_ids
        key = old_digest([frozen['repeat_key'],fold,'inner'])
        aux = auxiliary-excluded
        aux_val = {p for p in aux if reserved(p,key+'aux_validation',.2)}
        aux_probe = {p for p in aux-aux_val if reserved(p,key+'aux_probe',.05)}
        parts = dict(training=[v for v in training if (v['drug_id'],v['disease_id']) not in aux_val|aux_probe],
            auxiliary_training=[dict(drug_id=a,disease_id=b,raw_relation='CONTRAINDICATION') for a,b in sorted(aux-aux_val-aux_probe)])
        validation = []
        for raw in objects_in_array(assignment,'assignments'):
            if field(raw,'fold',True) == fold:
                continue
            pair = field(raw,'drug_id'),field(raw,'disease_id')
            if reserved(pair,key+'validation',.2):
                validation.append(dict(drug_id=pair[0],disease_id=pair[1],role=field(raw,'role')))
        parts.update(validation=sorted(validation,key=lambda v:(v['drug_id'],v['disease_id'])),
            auxiliary_validation=[dict(drug_id=a,disease_id=b,raw_relation='CONTRAINDICATION') for a,b in sorted(aux_val)])
    for name in domains:
        if old_digest(parts[name]) != frozen['membership_digests'][name]:
            raise ValueError('Original projected membership/order differs: '+name)
    support = {(v['drug_id'],v['disease_id']) for v in parts['training'] if v['role'] == 'P'}
    contra = {(v['drug_id'],v['disease_id']) for v in parts['auxiliary_training']}
    train_ids = {(v['drug_id'],v['disease_id']) for v in parts['training']}
    blocked = forbidden_pairs | (panel_ids-train_ids) | (auxiliary-contra)
    if old_digest(sorted(blocked)) != seal['mask_digest'] or frozen['training_seed'] != seal['seed']:
        raise ValueError('Original split mask/seed differs from the sealed schedule')
    roots = [root/f'data/processed/phase85_part3b1/repeat_{repeat}_fold_{fold}/corpus_v1',
             root/f'outputs/phase9/phase92b/test_only/r{repeat}_f{fold}/corpus',
             root/f'outputs/phase9/formal/PHASE9_G3_75_V1/data/r{repeat}_f{fold}/corpus',
             root/f'data/cache/phase98/corpora/{cell}']
    corpus_path = next((p/'manifest.json' for p in roots if (p/'manifest.json').is_file()),None)
    if corpus_path is None:
        raise FileNotFoundError('No completed masked corpus exists for '+cell+'; this exporter does not build a full graph')
    corpus = verified(corpus_path)
    if (corpus.get('schema') != 'PART3B1_FOLD_LOCAL_CORPUS_V1'
        or corpus.get('contaminated_DG_evidence_imported') is not False
        or corpus.get('heldout_DDA_properties_imported') is not False):
        raise ValueError('Expected the audited fold-local corpus without contaminated disease-gene/held-out properties')
    acceptance = verified(root/'outputs/phase85_part2b/graph_acceptance.json')
    if (acceptance.get('graph_provenance') != 'PASS' or acceptance.get('topology') != 'PASS'
        or acceptance.get('clean_graph_digest') != corpus['bindings'].get('clean_graph_digest')):
        raise ValueError('The original clean-graph acceptance does not match the exported corpus')
    if corpus['bindings']['fold_mask_digest'] != seal['mask_digest']:
        raise ValueError('Original graph corpus belongs to another split mask')
    source_hashes = {source_path(p):h for p,h in {**corpus['inputs'],**corpus['outputs']}.items()}
    node_path = root/'data/processed/phase3/v1/nodes/node_index.parquet'
    catalog_path = root/'data/processed/phase3/v1/graph/csr/edge_type_catalog.json'
    nodes = pl.read_parquet(node_path).sort('global_node_id')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = output.with_name(output.name+'.attempt_'+uuid.uuid4().hex)
    staging.mkdir()
    files = {}
    file_total = 27 + len(corpus['fact_files']) + len(corpus['evidence_files'])
    def store(name,writer):
        if name != 'manifest.json':
            destination_path(name)
        target = staging/name
        if not target.resolve().is_relative_to(staging.resolve()):
            raise ValueError('Export destination escaped the new corpus directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        writer(target)
        files[name] = sha256(target)
        if progress:
            progress('export files',len(files),file_total,0)
        return name
    def copy(name,path,expected):
        if sha256(path) != expected:
            raise ValueError('Certified original source changed: '+str(path))
        result = store(name,lambda p:shutil.copyfile(path,p))
        if files[name] != expected:
            raise ValueError('Original source changed during export: '+str(path))
        return result
    def js(name,value):
        return store(name,lambda p:p.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False),encoding='utf-8'))
    def parquet(name,frame):
        return store(name,frame.write_parquet)
    def array(name,values):
        return store(name,lambda p:np.save(p,values,allow_pickle=False))
    try:
        manifest = dict(schema='EVIHGT_PORTABLE_CORPUS_V1',training_seed=seal['seed'],token_buckets=8192,
            source_cell=cell,source_corpus_digest=corpus['digest'],semantics={k:True for k in
            ('canonical_entities_verified','canonical_facts_deduplicated','evidence_records_deduplicated','clinical_labels_verified',
             'target_aliases_resolved','disease_gene_label_leakage_removed','split_mask_applied','features_label_independent')})
        manifest['semantics']['unknown_is_negative'] = False
        manifest['nodes'] = copy('nodes.parquet',node_path,source_hashes[node_path])
        manifest['edge_catalog'] = copy('edge_catalog.json',catalog_path,source_hashes[catalog_path])
        for kind in ('fact','evidence'):
            manifest[kind+'_files'] = []
            for i,entry in enumerate(corpus[kind+'_files']):
                p = source_path(entry['path'])
                name = copy(f'{kind}/{i:02d}_{entry["component"]}.parquet',p,source_hashes[p])
                manifest[kind+'_files'].append(dict(path=name,component=entry['component'],rows=entry['rows']))
        manifest['csr'] = {name:copy(f'csr/{name}.npy',corpus_path.parent/'csr'/f'{name}.npy',
            source_hashes[corpus_path.parent/'csr'/f'{name}.npy']) for name in ('indptr','indices','fact_ids')}
        for key,name in (('target_map','legal_target_map.parquet'),('forbidden_facts','forbidden_fact_ids.parquet')):
            p = corpus_path.parent/name
            manifest[key] = copy(name,p,source_hashes[p])
        manifest['blocked_pairs'] = parquet('blocked_pairs.parquet',pl.DataFrame(sorted(blocked),schema=['drug_id','disease_id'],orient='row'))
        manifest['observed_pairs'] = copy('observed_pairs.parquet',observed_path,development['files']['observed_pair_exclusions.parquet'])
        manifest['eligibility'] = js('eligibility.json',{k:universe[k] for k in ('eligible_drugs','eligible_diseases')})
        positives = [dict(drug_id=a,disease_id=b,channel=channel,raw_relation=relation)
            for channel,relation,pairs in (('support','INDICATION',support),('unsuitability','CONTRAINDICATION',contra)) for a,b in sorted(pairs)]
        manifest['training_pairs'] = parquet('training_pairs.parquet',pl.DataFrame(positives))
        manifest['panels'] = {}
        for name in ('validation','auxiliary_validation'):
            rows = [dict(row,role=row.get('role','P'),raw_relation=('INDICATION' if row['role']=='P' else None)
                if name == 'validation' else 'CONTRAINDICATION') for row in parts[name]]
            manifest['panels'][name] = parquet(name+'.parquet',pl.DataFrame(rows))
        manifest['schedule'] = dict(G=seal['G'],groups=copy('global_groups.parquet',groups_path,schedule['groups_sha256']),
            conditional_records=js('conditional_records.json',conditional['records']),sampling_provenance=dict(
                original_schedule_sha256=seal['schedule_sha256'],original_plan_sha256=seal['plan_sha256'],
                original_schedule_digest=seal['schedule_digest'],original_plan_digest=seal['plan_digest'],
                original_mask_digest=seal['mask_digest'],original_member_stream_digest=old_digest(schedule['member_stream'])))
        manifest['raw_features'] = {name:copy('raw_features/'+name+'.npy',base/'features_v2'/f'{name}.npy',features['files'][name+'.npy'])
            for name in ('drug_ids','drug_values','disease_ids','disease_values')}
        manifest['pretrained'] = _export_d1(d1_root,nodes,node_path,copy,js,array)
        manifest['files'] = files
        js('manifest.json',manifest)
        loaded = load_corpus(staging/'manifest.json',progress=progress)
        identity = loaded.identity
        # Compare the exported ordered supervision to the original member stream.
        from dataclasses import asdict
        for step in range(1,loaded.plan.total_steps+1):
            member = schedule['member_stream'][step-1]
            if old_digest([asdict(q) for q in loaded.plan.step(step).queries]) != member['member_sha256']:
                raise ValueError('Export changed the original ordered supervision records')
            if progress:
                progress('original supervision comparison',step,loaded.plan.total_steps,0)
        del loaded
        gc.collect()
        os.rename(staging,output)
        return dict(manifest=str(output/'manifest.json'),identity=identity,cell=cell,
            files=len(files)-1,source_files_modified=False,outer_final_roles_read=False,
            original_3G_members_and_conditional_plan_preserved=True)
    except BaseException:
        raise


def _export_d1(root,nodes,node_path,copy,js,array):
    source = read_json(root/'manifest.json')
    provenance_path = root.parent/'input_manifest.json'
    inputs_path = root.parent/'feature_inputs.parquet'
    provenance = read_json(provenance_path)
    if (source.get('status') != 'COMPLETE' or source.get('training') is not False
        or source.get('optimizer_steps') != 0 or source.get('sealed_final_labels_read') is not False
        or provenance.get('clinical_text_encoded') is not False or provenance.get('sealed_final_labels_read') is not False
        or provenance.get('optimizer_steps') != 0 or source.get('width') != 768 or source.get('dtype') != 'float32'):
        raise ValueError('D1 source must be complete frozen inference without clinical labels/text')
    if (sha256(provenance_path) != source['input_manifest_sha256'] or sha256(node_path) != provenance['node_index_sha256']
        or sha256(inputs_path) != source['feature_inputs_sha256'] or sha256(inputs_path) != provenance['feature_inputs_sha256']):
        raise ValueError('D1 canonical input/provenance identity changed')
    for name,expected in ENCODER_SPECS.items():
        if any(source.get('fixed_encoder_specs',{}).get(name,{}).get(k) != v for k,v in expected.items()):
            raise ValueError('D1 pinned encoder specification changed')
    inputs = pl.read_parquet(inputs_path).sort('global_node_id')
    for column in ('global_node_id','type_local_id','node_type','node_id'):
        if nodes[column].to_list() != inputs[column].to_list():
            raise ValueError('D1 canonical entity mapping differs: '+column)
    result = {}
    for kind,prefix,encoder in (('DRG','drug','chemberta'),('DIS','disease','biobert')):
        typed = inputs.filter(pl.col('node_type') == kind).sort('type_local_id')
        records = [r for r in source['types'] if r['node_type'] == kind]
        if len(records) != 1:
            raise ValueError('D1 typed array descriptor missing or duplicated')
        record = records[0]
        allowed = {'optimus_canonical_smiles','drugbank_structured_smiles'} if kind == 'DRG' else {'canonical_name'}
        import hashlib
        input_digest = hashlib.sha256()
        for row in typed.select('node_id','encoder_text','available','source_field','encoder').iter_rows(named=True):
            present,text = row['available'],row['encoder_text']
            if (type(present) is not bool or row['encoder'] != encoder
                or (present and (row['source_field'] not in allowed or not isinstance(text,str) or not text.strip()))
                or (not present and (row['source_field'] is not None or text is not None))):
                raise ValueError('D1 typed source/availability is not label-independent')
            input_digest.update(json.dumps([row['node_id'],text if present else None,present],ensure_ascii=False).encode()+b'\n')
        if input_digest.hexdigest() != record['input_sha256']:
            raise ValueError('D1 typed feature input digest changed')
        values_path,mask_path = root/f'type_{record["node_type_id"]:02d}.npy',root/f'type_{record["node_type_id"]:02d}_available.npy'
        if sha256(values_path) != record['sha256'] or sha256(mask_path) != record['mask_sha256']:
            raise ValueError('D1 frozen vector/mask bytes changed')
        values = np.load(values_path,mmap_mode='r',allow_pickle=False)
        masks = np.load(mask_path,mmap_mode='r',allow_pickle=False)
        if not np.array_equal(masks,typed['available'].to_numpy()):
            raise ValueError('D1 array mask differs from verified source availability')
        ordered = nodes.filter(pl.col('node_type') == kind).sort('global_node_id')
        local = ordered['type_local_id'].to_numpy()
        result[prefix+'_ids'] = array('d1/'+prefix+'_ids.npy',ordered['global_node_id'].to_numpy())
        result[prefix+'_values'] = array('d1/'+prefix+'_values.npy',values[local])
        result[prefix+'_available'] = array('d1/'+prefix+'_available.npy',masks[local])
    result['metadata'] = js('d1/metadata.json',dict(schema='EVIHGT_D1_FEATURES_V1',width=768,optimizer_steps=0,
        clinical_text_encoded=False,supervised_dda_features_present=False,trained_node_embeddings_loaded=False,
        prediction_scores_loaded=False,fixed_encoder_specs=source['fixed_encoder_specs'],
        original_feature_manifest_sha256=sha256(root/'manifest.json'),original_input_manifest_sha256=sha256(provenance_path),
        upstream_pretraining_target_exposure='NOT_VERIFIED',
        pooling='LAST_HIDDEN_STATE_MEAN_OVER_NONPADDING_NONSPECIAL_TOKENS'))
    return result
