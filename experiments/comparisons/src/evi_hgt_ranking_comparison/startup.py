"""Verify exact launch input identities before constructing a model."""
from pathlib import Path
from .common import ROOT,digest,sha

def verify_task_inputs(task,data,corpus,*,seal=None,feature_metadata=None):
    expected=task['input_identity']
    if task['data_digest']!=digest(expected):raise ValueError('Task data digest differs')
    if task['cell']!=data.cell_id or task['training_seed']!=data.training_seed:raise ValueError('Task cell/seed differs')
    if expected['data_bindings']!=data.bindings:raise ValueError('Task training bindings differ')
    if expected['mask_digest']!=data.bindings['fold_mask_digest']:raise ValueError('Task mask differs')
    bound=task['corpus'];manifest=Path(corpus['corpus_root'])/'manifest.json'
    if Path(bound['root']).resolve()!=Path(corpus['corpus_root']).resolve():raise ValueError('Task corpus root differs')
    if bound['digest']!=corpus['digest'] or bound['manifest_sha256']!=sha(manifest):raise ValueError('Task corpus identity differs')
    if expected['corpus_digest']!=corpus['digest']:raise ValueError('Input corpus digest differs')
    sampling=task['sampling']
    method=task.get('method')
    if method in ('FULL','PAIR_MLP_D1_MATCHED','HGT_D1_MATCHED'):
        for name in ('schedule','plan'):
            if sha(sampling[name+'_path'])!=sampling[name+'_sha256']:raise ValueError('Sampling bytes differ: '+name)
        if expected['sampling_plan_digest']!=sampling['plan_digest']:raise ValueError('Sampling plan binding differs')
        if seal is not None:
            if seal['plan_digest']!=sampling['plan_digest'] or seal['mask_digest']!=expected['mask_digest']:raise ValueError('Runtime sampling differs')
    elif method in ('HGTDR_ADAPTED','TXGNN_ADAPTED','TRAIN_VISIBLE_POPULARITY'):
        actual=({'kind':'NO_TRAINING_SAMPLER'} if method=='TRAIN_VISIBLE_POPULARITY' else
                {'kind':'ORIGINAL_NATIVE_TRAINING_SAMPLER','cell_seed':data.training_seed})
        if sampling!=actual or expected['sampling_binding_digest']!=digest(actual):raise ValueError('Native/no-training sampling identity differs')
        if seal is not None:raise ValueError('Common sampling plan is not a native/no-training execution recipe')
    else:raise ValueError('Unrecognized method sampling contract')
    if feature_metadata is not None:
        if feature_metadata['feature_manifest_sha256']!=expected['D1_feature_manifest_sha256']:raise ValueError('D1 feature bytes differ')
    return dict(status='PASS',cell=data.cell_id,data_digest=task['data_digest'],model_constructed=False)

def verify_launch_manifest(manifest):
    closure=manifest['source_sha256']
    for path,expected in closure.items():
        if sha(path)!=expected:raise ValueError('Source identity changed: '+path)
    source_digest=digest(closure)
    for task in manifest['training_tasks']+manifest.get('development_tasks',[]):
        if task['source_digest']!=source_digest:raise ValueError('Task source closure differs')
        if digest(task['input_identity'])!=task['data_digest']:raise ValueError('Task input contract digest differs')
    return source_digest
