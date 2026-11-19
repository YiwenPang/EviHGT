"""Complete RankingComparison Tx predictor caches, unregistered modes and gradients.

The native predictor extends disease profiles during finetuning. These values
must be restored, not asserted immutable. Encoding retains NumPy dictionary
key types and uses only weights-only-safe payload types.
"""
import copy
import torch
from .observation_io import encode,decode

KEY='ranking_comparison_tx_auxiliary'
SCHEMA='RANKING_COMPARISON_TX_AUXILIARY_V1'


def capture(model):
    pred=model.pred
    value=dict(rel2idx=pred.rel2idx,etypes=model.etypes,sim=pred.sim_all_etypes,
        diseaseid2id=pred.diseaseid2id_etypes,profiles=pred.diseases_profile_etypes,
        gate_modes={k:{n:m.training for n,m in gate.named_modules()} for k,gate in pred.W_gate.items()},
        gate_gradients={k:{n:p.grad for n,p in gate.named_parameters()} for k,gate in pred.W_gate.items()},
        model_gradients={n:p.grad for n,p in model.named_parameters()})
    return dict(schema=SCHEMA,payload=encode(value))


def validate(value):
    if not isinstance(value,dict) or set(value)!= {'schema','payload'} or value['schema']!=SCHEMA:
        raise ValueError('Complete Tx auxiliary state required')


def restore(model,value):
    validate(value)
    raw=decode(copy.deepcopy(value['payload']))
    if set(raw)!= {'rel2idx','etypes','sim','diseaseid2id','profiles','gate_modes','gate_gradients','model_gradients'}:
        raise ValueError('Tx auxiliary fields differ')
    pred=model.pred
    if raw['rel2idx']!=pred.rel2idx or raw['etypes']!=model.etypes:raise ValueError('Tx relation mapping differs')
    if set(raw['gate_modes'])!=set(pred.W_gate) or set(raw['gate_gradients'])!=set(pred.W_gate):raise ValueError('Tx gate layout differs')
    def check_gradients(parameters,values):
        if parameters.keys()!=values.keys():raise ValueError('Tx gradient names differ')
        for name,p in parameters.items():
            v=values[name]
            if v is not None and (not isinstance(v,torch.Tensor) or v.shape!=p.shape or v.dtype!=p.dtype):
                raise ValueError('Tx gradient schema differs: '+name)
    parameters=dict(model.named_parameters());check_gradients(parameters,raw['model_gradients'])
    for key,gate in pred.W_gate.items():
        check_gradients(dict(gate.named_parameters()),raw['gate_gradients'][key])
        if dict(gate.named_modules()).keys()!=raw['gate_modes'][key].keys() or any(type(v) is not bool for v in raw['gate_modes'][key].values()):raise ValueError('Tx gate mode schema differs')
    pred.sim_all_etypes=raw['sim'];pred.diseaseid2id_etypes=raw['diseaseid2id'];pred.diseases_profile_etypes=raw['profiles']
    for name,p in parameters.items():p.grad=raw['model_gradients'][name]
    for key,gate in pred.W_gate.items():
        for name,p in gate.named_parameters():p.grad=raw['gate_gradients'][key][name]
        for name,module in gate.named_modules():module.training=raw['gate_modes'][key][name]


def snapshot_receiver(trainer):
    original=trainer.external_control
    trainer.external_control=dict(original,**{KEY:capture(trainer.model)})
    try:return trainer._module.TxTwoStageTrainer.snapshot(trainer)
    finally:trainer.external_control=original


def restore_receiver(trainer,state):
    value=state['external_control'].get(KEY)
    validate(value)
    previous=capture(trainer.model)
    try:
        restore(trainer.model,value)
        trainer._module.TxTwoStageTrainer.restore(trainer,state)
    except BaseException:
        restore(trainer.model,previous)
        raise
    finally:
        trainer.external_control.pop(KEY,None)
