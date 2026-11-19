"""Check export from a synthetic research workspace."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import polars as pl

from evi_hgt.data.common import sha256
from evi_hgt.data.corpus import load_corpus
from evi_hgt.data.projection import digest
from evi_hgt.data.pretrained import ENCODER_SPECS
from evi_hgt.data.research_export import export_research
from data_fixture import make_corpus


def make_research(tmp_path):
    portable = make_corpus(tmp_path/'portable_seed')
    source = load_corpus(portable)
    manifest = source.manifest
    root = tmp_path/'research'
    d1 = tmp_path/'d1/features'
    root.mkdir()
    d1.mkdir(parents=True)
    def write(path,value,bound=False):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if bound:
            value = dict(value,digest=digest(value))
        path.write_text(json.dumps(value,indent=2),encoding='utf-8')
        return value
    def copy(name,path):
        target = root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        return target
    nodes = source.nodes
    local = []
    counts = {}
    for kind in nodes['node_type']:
        local.append(counts.get(kind,0))
        counts[kind] = counts.get(kind,0)+1
    nodes = nodes.with_columns(pl.Series('type_local_id',local,dtype=pl.Int64))
    node_path = root/'data/processed/phase3/v1/nodes/node_index.parquet'
    node_path.parent.mkdir(parents=True)
    nodes.write_parquet(node_path)
    cat_path = copy('data/processed/phase3/v1/graph/csr/edge_type_catalog.json',portable.parent/manifest['edge_catalog'])
    corpus_root = root/'data/processed/phase85_part3b1/repeat_0_fold_1/corpus_v1'
    corpus_root.mkdir(parents=True)
    files = {str(node_path):sha256(node_path),str(cat_path):sha256(cat_path)}
    corpus = dict(fact_files=[],evidence_files=[])
    for kind in ('fact','evidence'):
        for entry in manifest[kind+'_files']:
            p = corpus_root/(kind+'.parquet')
            shutil.copyfile(portable.parent/entry['path'],p)
            files[str(p)] = sha256(p)
            corpus[kind+'_files'].append(dict(entry,path=str(p)))
    for key,relative in {**manifest['csr'],'legal_target_map':manifest['target_map'],'forbidden_fact_ids':manifest['forbidden_facts']}.items():
        name = 'csr/'+key+'.npy' if key in manifest['csr'] else key+'.parquet'
        p = corpus_root/name
        p.parent.mkdir(exist_ok=True)
        shutil.copyfile(portable.parent/relative,p)
        files[str(p)] = sha256(p)
    base = root/'data/processed/phase85_part2'
    base.mkdir(parents=True)
    training = [dict(drug_id='D0',disease_id='S0',role='P'),dict(drug_id='D3',disease_id='S1',role='U')]
    aux = [dict(drug_id='D1',disease_id='S1',raw_relation='CONTRAINDICATION')]
    validation = [dict(drug_id='D0',disease_id='S2',role='P'),dict(drug_id='D2',disease_id='S2',role='U')]
    aux_val = [dict(drug_id='D1',disease_id='S2',raw_relation='CONTRAINDICATION')]
    parts = dict(training=training,auxiliary_training=aux,validation=validation,auxiliary_validation=aux_val)
    # These unused fields would violate the public corpus role schema if decoded.
    write(corpus_root.parent/'cell_manifest.json',dict(parts,outer_test=[{'do_not_decode':'outer role'}],final_test=[{'do_not_decode':'final role'}]))
    panel = pl.DataFrame(training+validation)
    panel.write_parquet(base/'development_panel.parquet')
    pl.DataFrame(aux+aux_val).write_parquet(base/'auxiliary_positives.parquet')
    pl.DataFrame({'drug_id':[],'disease_id':[]},schema={'drug_id':pl.String,'disease_id':pl.String}).write_parquet(base/'all_forbidden_pairs.parquet')
    shutil.copyfile(portable.parent/manifest['observed_pairs'],base/'observed_pair_exclusions.parquet')
    devfiles = {p.name:sha256(p) for p in base.glob('*.parquet')}
    write(base/'development_manifest.json',dict(files=devfiles),bound=True)
    write(base/'training_u_manifest.json',json.loads((portable.parent/manifest['eligibility']).read_text()),bound=True)
    feature_root = base/'features_v2'
    feature_root.mkdir()
    for key,relative in manifest['raw_features'].items():
        shutil.copyfile(portable.parent/relative,feature_root/(key+'.npy'))
    write(feature_root/'manifest.json',dict(files={p.name:sha256(p) for p in feature_root.glob('*.npy')}),bound=True)
    write(root/'outputs/phase85_part3a_r2/cv_three_key_binding.json',dict(selected_repeat_bindings=[]),bound=True)
    mask = {('D0','S2'),('D2','S2'),('D1','S2')}
    membership = {k:digest(v) for k,v in parts.items()}
    write(root/'outputs/phase85_part3b1/cv_15_cell_static_census.json',dict(cells=[dict(repeat_index=0,fold_index=1,
        training_seed=42,membership_digests=membership,repeat_key='SYNTHETIC_TEST_ONLY')]))
    corpus.update(schema='PART3B1_FOLD_LOCAL_CORPUS_V1',contaminated_DG_evidence_imported=False,
        heldout_DDA_properties_imported=False,bindings={'fold_mask_digest':digest(sorted(mask)),
        'clean_graph_digest':digest('SYNTHETIC_TEST_ONLY')},inputs=files,outputs={})
    write(root/'outputs/phase85_part2b/graph_acceptance.json',dict(graph_provenance='PASS',topology='PASS',
        clean_graph_digest=digest('SYNTHETIC_TEST_ONLY')),bound=True)
    write(corpus_root/'manifest.json',corpus,bound=True)
    schedule_root = root/'data/cache/phase98/R0F1/synthetic'
    schedule_root.mkdir(parents=True)
    groups = schedule_root/'global_groups.parquet'
    shutil.copyfile(portable.parent/manifest['schedule']['groups'],groups)
    members = [dict(step=i,member_sha256=digest([asdict(q) for q in source.plan.step(i).queries])) for i in (1,2,3)]
    schedule = write(schedule_root/'schedule.json',dict(G=1,total=3,groups_path=str(groups),groups_sha256=sha256(groups),member_stream=members),bound=True)
    conditional = write(schedule_root/'conditional_plan.json',dict(schedule_digest=schedule['digest'],records=source.plan._records),bound=True)
    seal = dict(cell='R0F1',seed=42,G=1,schedule_path=str(schedule_root/'schedule.json'),schedule_sha256=sha256(schedule_root/'schedule.json'),
        schedule_digest=schedule['digest'],plan_path=str(schedule_root/'conditional_plan.json'),plan_sha256=sha256(schedule_root/'conditional_plan.json'),
        plan_digest=conditional['digest'],mask_digest=digest(sorted(mask)))
    write(root/'outputs/phase9/phase98/part2/sampling/ALL_PLANS_SEALED.json',dict(status='SEALED_15_CELLS',cells=[seal]),bound=True)
    input_rows = []
    for row in nodes.iter_rows(named=True):
        kind = row['node_type']
        available = kind in {'DRG','DIS'}
        input_rows.append(dict(row,available=available,encoder_text=row['node_id'] if available else None,
            source_field='optimus_canonical_smiles' if kind=='DRG' else 'canonical_name' if kind=='DIS' else None,
            encoder='chemberta' if kind=='DRG' else 'biobert' if kind=='DIS' else None))
    inputs = pl.DataFrame(input_rows)
    inputs.write_parquet(d1.parent/'feature_inputs.parquet')
    provenance = dict(clinical_text_encoded=False,sealed_final_labels_read=False,optimizer_steps=0,
        node_index_sha256=sha256(node_path),feature_inputs_sha256=sha256(d1.parent/'feature_inputs.parquet'))
    write(d1.parent/'input_manifest.json',provenance)
    records = []
    for kind,prefix,tid in (('DRG','drug',4),('DIS','disease',3)):
        shutil.copyfile(portable.parent/manifest['pretrained'][prefix+'_values'],d1/f'type_{tid:02d}.npy')
        shutil.copyfile(portable.parent/manifest['pretrained'][prefix+'_available'],d1/f'type_{tid:02d}_available.npy')
        h = hashlib.sha256()
        for row in inputs.filter(pl.col('node_type')==kind).sort('type_local_id').iter_rows(named=True):
            h.update(json.dumps([row['node_id'],row['encoder_text'],True],ensure_ascii=False).encode()+b'\n')
        records.append(dict(node_type=kind,node_type_id=tid,input_sha256=h.hexdigest(),sha256=sha256(d1/f'type_{tid:02d}.npy'),
                            mask_sha256=sha256(d1/f'type_{tid:02d}_available.npy')))
    write(d1/'manifest.json',dict(status='COMPLETE',training=False,optimizer_steps=0,sealed_final_labels_read=False,width=768,dtype='float32',
        input_manifest_sha256=sha256(d1.parent/'input_manifest.json'),feature_inputs_sha256=sha256(d1.parent/'feature_inputs.parquet'),
        fixed_encoder_specs=ENCODER_SPECS,types=records))
    return root,d1


def test_export_one_prepared_cell_without_original_runtime(tmp_path):
    root,d1 = make_research(tmp_path)
    original = {p:sha256(p) for p in root.rglob('*') if p.is_file()}
    output = tmp_path/'exported'
    result = export_research(root,'R0F1',d1,output)
    assert result['outer_final_roles_read'] is False
    assert result['original_3G_members_and_conditional_plan_preserved'] is True
    corpus = load_corpus(output/'manifest.json')
    assert corpus.plan.total_steps == 3
    assert set(corpus.panels) == {'validation','auxiliary_validation'}
    assert not any('outer' in p.name or 'final' in p.name for p in output.rglob('*'))
    assert all(sha256(p) == value for p,value in original.items())


def test_export_relocated_research_paths_are_explicit(tmp_path):
    root,d1 = make_research(tmp_path)
    relocated = tmp_path/'relocated_research'
    shutil.copytree(root,relocated)
    result = export_research(relocated,'R0F1',d1,tmp_path/'exported',recorded_root=str(root))
    assert result['original_3G_members_and_conditional_plan_preserved']
