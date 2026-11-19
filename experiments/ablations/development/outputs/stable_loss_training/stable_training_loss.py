"""Branch-local PU risk. Production modules and model sigmoids are untouched."""
from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'outputs/phase85_part3b10'))
from stable_core import stable_sigmoid_negative,StableNegativeSigmoid
from evi_hgt.training.formal_objective import SymmetricPUPairwiseRankingLoss,PairwiseRankingRisk,DualChannelPairwiseRankingObjective
VERSION='B11_PU_RISK_WITH_B10_STABLE_BACKWARD_V1'

class StablePURisk(SymmetricPUPairwiseRankingLoss):
    def __init__(self,temperature=1.,pairwise_chunk_size=None,enabled=True):
        super().__init__(temperature,pairwise_chunk_size);self.enabled=enabled;self.trace=None
    def begin_trace(self):self.trace=[]
    def end_trace(self):
        result=self.trace;self.trace=None;return result
    def forward(self,positive_logits,unlabeled_logits):
        if not self.enabled:return super().forward(positive_logits,unlabeled_logits)
        if positive_logits.numel()==0 or unlabeled_logits.numel()==0:raise ValueError('Ranking sets must be nonempty')
        positive=positive_logits.float().reshape(-1);unlabeled=unlabeled_logits.float().reshape(-1)
        if not torch.isfinite(positive).all() or not torch.isfinite(unlabeled).all():raise ValueError('Ranking logits must be finite')
        np_=int(positive.numel());nu=int(unlabeled.numel());n=np_*nu;chunk=self.pairwise_chunk_size
        record=None
        if self.trace is not None:
            record=dict(P=np_,U=nu,pairs=n,chunks=0,backward_calls=0,dtype='torch.float32',candidate=VERSION)
            self.trace.append(record)
        def target(z):
            y=stable_sigmoid_negative(z)
            if record is not None:
                record['chunks']+=1
                # Local scalar-output witness only for smoke/first update; returns the
                # original gradient unchanged. No model/global sigmoid/module hook.
                if y.requires_grad:
                    def witness(gradient):
                        record['backward_calls']+=1
                        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite PU upstream seed')
                        return gradient
                    y.register_hook(witness)
            return y
        if chunk is None or chunk>=n:
            margins=(positive[:,None]-unlabeled[None,:])/self.temperature
            loss_sum=target(margins).sum(dtype=torch.float32);margin_sum=margins.sum(dtype=torch.float32)
        else:
            losses=[];means=[]
            for start in range(0,n,chunk):
                flat=torch.arange(start,min(start+chunk,n),device=positive.device)
                pi=torch.div(flat,nu,rounding_mode='floor');ui=torch.remainder(flat,nu)
                margins=(positive[pi]-unlabeled[ui])/self.temperature
                losses.append(target(margins).sum(dtype=torch.float32));means.append(margins.sum(dtype=torch.float32))
            loss_sum=torch.stack(losses).sum(dtype=torch.float32);margin_sum=torch.stack(means).sum(dtype=torch.float32)
        total=loss_sum/n;mean_margin=margin_sum/n;finite=bool(torch.isfinite(total) and torch.isfinite(mean_margin))
        if not finite:raise FloatingPointError('Pairwise reduction nonfinite')
        return PairwiseRankingRisk(total=total,positive_count=np_,unlabeled_count=nu,pair_count=n,mean_margin=mean_margin,finite=finite,temperature=self.temperature,pairwise_chunk_size=chunk)

def make_objective(enabled=True):
    obj=DualChannelPairwiseRankingObjective(temperature=1.,lambda_support=1.,lambda_unsuitability=1.,pairwise_chunk_size=4096)
    obj.channel_loss=StablePURisk(temperature=1.,pairwise_chunk_size=4096,enabled=enabled)
    return obj

def qualify_trace(calls,record):
    diseases=list(dict.fromkeys(p[1] for p in record['support_positive_pairs']))
    expected_support_calls=len(diseases) if any(record['disease_unlabeled_pairs'].values()) else 1
    if len(calls)!=expected_support_calls+1:raise ValueError('Both-channel call mapping differs')
    if any(c['backward_calls']!=c['chunks'] or c['backward_calls']<1 for c in calls):raise ValueError('Candidate backward did not execute for every PU call')
    if sum(c['P'] for c in calls[:-1])!=len(record['support_positive_pairs']) or calls[-1]['P']!=len(record['contra_positive_pairs']):raise ValueError('PU role mapping differs')
    return dict(support_calls=calls[:-1],contra_call=calls[-1],both_channels_stable_backward_executed=True,local_identity_hooks_gradient_unchanged=True)
