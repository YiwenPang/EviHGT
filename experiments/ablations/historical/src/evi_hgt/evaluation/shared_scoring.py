"""One bounded graph/path assembly per pair, reused across frozen models.

Each model still receives exactly one graph with the original dtypes. Only raw
scores are spooled per disease; model states, candidate graphs and path search
are never shared across different pairs. No fused or approximate forward.
"""
from __future__ import annotations
from pathlib import Path
import math
from evi_hgt.evaluation.formal_runtime import (
    atomic_json,read_json,digest,SharedInvariantError,ModelEvaluationError,
)


class SharedScoreStore:
    def __init__(self,output,identity,labels):
        self.output=Path(output);self.identity=dict(identity,labels=list(labels))
        self.labels=tuple(labels);self.failures={};self._loaded=None;self._scores=None

    def compute(self,candidate_streams,*,assemble,score_batch):
        """Both channel streams must match before ANY query graph is assembled."""
        for support,counter in zip(*candidate_streams,strict=True):
            if (support.disease_id!=counter.disease_id or
                    support.candidate_membership_digest!=counter.candidate_membership_digest):
                raise SharedInvariantError('shared candidate channel mismatch')
            binding={'identity':self.identity,'disease':support.disease_id,
                     'candidate_membership':support.candidate_membership_digest}
            path=self.output/(digest(support.disease_id)+'.json')
            if path.exists():
                saved=read_json(path)
                if saved['sha256']!=digest(saved['payload']) or saved['payload']['binding']!=binding:
                    raise SharedInvariantError('shared raw-score commit corruption/binding mismatch')
                if tuple(saved['payload']['candidate_ids'])!=tuple(support.candidate_drug_ids):
                    raise SharedInvariantError('shared raw-score candidate order mismatch')
                self.failures.update(saved['payload']['failures']);continue
            scores={label:[] for label in self.labels}
            for drug in support.candidate_drug_ids:
                if len(self.failures)==len(self.labels):break
                batch=assemble(drug,support.disease_id)
                for label in self.labels:
                    if label in self.failures:continue
                    try:
                        values=score_batch(label,batch)
                        if len(values)!=2 or not all(math.isfinite(float(v)) for v in values):
                            raise ModelEvaluationError('nonfinite dual-head prediction')
                        scores[label].append([float(v) for v in values])
                    except ModelEvaluationError as exc:
                        import traceback
                        reason=f'{type(exc).__name__}: {exc}'
                        log=self.output/'failures'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                        log.write_text(traceback.format_exc(),encoding='utf-8')
                        self.failures[label]={'reason':reason,'log':str(log)}
            scores={label:rows for label,rows in scores.items() if label not in self.failures}
            if any(len(rows)!=len(support.candidate_drug_ids) for rows in scores.values()):
                raise SharedInvariantError('incomplete shared score count')
            payload={'binding':binding,'candidate_ids':support.candidate_drug_ids,
                     'scores':scores,'failures':self.failures.copy()}
            atomic_json(path,{'payload':payload,'sha256':digest(payload)})
            print(f'SHARED_RAW_SCORES_COMMITTED SPLIT={self.identity.get("split")} ACTIVE_MODELS={len(scores)}',flush=True)

    def score(self,label,drug,disease):
        if label in self.failures:raise ModelEvaluationError(self.failures[label]['reason'])
        if self._loaded!=disease:
            saved=read_json(self.output/(digest(disease)+'.json'));payload=saved['payload']
            if (saved['sha256']!=digest(payload) or payload['binding']['identity']!=self.identity
                    or payload['binding']['disease']!=disease):
                raise SharedInvariantError('shared raw-score commit corruption')
            ids=payload['candidate_ids']
            if len(ids)!=len(set(ids)):raise SharedInvariantError('duplicate spooled candidate')
            if any(len(v)!=len(ids) for v in payload['scores'].values()):raise SharedInvariantError('spool row mismatch')
            self._scores={k:dict(zip(ids,values,strict=True)) for k,values in payload['scores'].items()}
            self._loaded=disease
        try:return self._scores[label][drug]
        except KeyError as exc:raise SharedInvariantError('missing frozen candidate/model score') from exc
