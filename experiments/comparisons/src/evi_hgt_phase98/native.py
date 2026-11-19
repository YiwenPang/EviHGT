"""Native comparator static wiring. No training or model factory is invoked."""
import ast
from evi_hgt.phase9.txgnn_protocol import fixed_recipe
from evi_hgt.phase85_v2.common import bound
from .common import ROOT,OUT,read,sha,write_new


def txgnn_static_contract(seed):
    paths=['src/evi_hgt/phase9/txgnn_protocol.py','src/evi_hgt/phase9/txgnn_runner.py',
           'src/evi_hgt/phase9/txgnn_data.py','src/evi_hgt/phase9/txgnn_recovery.py',
           'src/evi_hgt/phase9/txgnn_adapter.py']
    sources={p:sha(ROOT/p) for p in paths}
    interfaces={}
    for path in paths:
        tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
        interfaces[path]=sorted(n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
    expected={'src/evi_hgt/phase9/txgnn_runner.py':{'run_fixed_control'},
        'src/evi_hgt/phase9/txgnn_recovery.py':{'TxTwoStageTrainer','pretrain_step','transition','finetune_step','snapshot','restore'},
        'src/evi_hgt/phase9/txgnn_adapter.py':{'prepare_fold','TrainingOnlyU','verify_native_sources'}}
    if any(not names<=set(interfaces[path]) for path,names in expected.items()):
        raise ValueError('Native two-stage/recovery interface missing')
    recipe=fixed_recipe(seed)
    assert recipe['device']=='cpu' and recipe['pretrain_epochs']==1 and recipe['finetune_epochs']==500
    return bound(dict(schema='PHASE98_TXGNN_NATIVE_STATIC_WIRING_V1',status='STATIC_SOURCE_WIRING_ONLY',
        recipe=recipe,sources=sources,static_ast_interfaces=interfaces,stages=[
            dict(name='ALL_RELATION_PRETRAIN',epochs=1,batch_size=65536,lr=.001,loss='native all-relation objective'),
            dict(name='DDA_FINETUNE',epochs=500,lr=.003,decoder_reset_count=1,selection_opportunities=20)],
        graph_policy='same-cell masked legal all-relation graph; no cross-fold fallback',
        recovery='pretrain batch cursor + finetune epoch + sampler/control + optimizer + scheduler + complete RNG + selected state',
        evaluation='inner support native selection unchanged; selected support/contra evaluation separately gated',
        model_forwards=0,optimizer_updates=0,model_constructed=False,training_authorized=False,
        static_wiring_does_not_claim_training_or_runtime_qualification=True))


def seal_native_contracts():
    part=read(ROOT/'outputs/phase9/phase98/part1/PHASE98_EVALUATION_PLAN.json')
    result=bound(dict(schema='PHASE98_NATIVE_STATIC_CONTRACTS_V2',supersedes='NATIVE_STATIC_WIRING.json',
        txgnn_cells={x['cell']:txgnn_static_contract(x['seed']) for x in part['cells']},
        hgtdr=dict(epochs=300,initial_lr=.003,selection='support inner macro ROC-AUC then AP then earlier epoch',
            selected_weight_source='sampler.native_control.best_model',contra='N/A_NO_HEAD',
            completed_R0F0_selected_epoch=222,completed_R0F0_recovery_epoch=300),
        model_forwards=0,optimizer_updates=0,enabled=False))
    path=OUT/'reuse/NATIVE_STATIC_WIRING_V2.json'
    if path.exists():
        if read(path)!=result: raise ValueError('Immutable native contract differs')
    else: write_new(path,result)
    return result
