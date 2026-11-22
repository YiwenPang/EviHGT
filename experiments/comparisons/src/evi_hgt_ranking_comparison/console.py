"""RankingComparison versioned event wrapper; inherited EVIDENCE_ABLATION mechanics stay untouched."""
from evi_hgt_evidence_ablation.console import Console as AcceptedConsole
from evi_hgt_console import _table,_count,_number

def number(value):return 'Unavailable' if value is None else _number(value,6)

class Console(AcceptedConsole):
    def __init__(self,*args,context='TEST_ONLY',queue_total=71,**kwargs):
        if context not in ('TEST_ONLY','REPLAY','AUTHORIZED_RUNTIME'):raise ValueError('Explicit display context required')
        if type(queue_total) is not int or queue_total<0:raise ValueError('Invalid queue size')
        self.context=context;self.queue_total=queue_total
        super().__init__(*args,**kwargs)

    def stage(self,title,detail=''):
        super().stage(f'[{self.context}] {title}',detail)

    def banner(self,queue,budget,resumed_step=0):
        self.stage('EviHGT · Ranking comparison',f'{self.queue_total} jobs, {budget:,} planned updates, mode {self.context}')
        self._write(' → '.join('/'.join(q) if isinstance(q,(tuple,list)) else str(q) for q in queue))

    def frame(self,width=None):
        return [line.replace(f'\u961f\u5217\uff1a\u5df2\u5b8c\u6210{self.completed}/4',f'Queue: {self.completed}/{self.queue_total} complete') for line in super().frame(width)]

    def _render_evaluation(self,*,force=False):
        # Accepted renderer hardcoded EVIDENCE_ABLATION queue size; temporarily formatting
        # is not a global patch and does not touch predictions/counters/RNG.
        now=self.clock()
        if not force and now-self._last_render<self.refresh_seconds:return
        from evi_hgt_evidence_ablation.console import percent,wrap
        e=self._evaluation
        self._draw(wrap(f'[{self.context}] Scored {e["completed"]:,}/{e["total"]:,} queries ({percent(e["completed"],e["total"])}). Queue: {self.completed}/{self.queue_total} complete.',self._columns()))
        self._last_render=now

    def evaluation_result(self,arm,step,metrics):
        self.stage('Evaluation · '+str(arm),f'Update {step:,}')
        metrics=metrics.get('metrics',metrics);headline=[];ranking=[]
        for head,label in (('support','Support'),('contra','Contraindication')):
            m=metrics.get(head,{}) or {}
            headline.append([label,*[number(m.get(k)) for k in ('macro_auc','macro_ap','macro_aupr_trapezoid')],f"{_count(m.get('evaluable_diseases'))}/{_count(m.get('skipped_diseases'))}",_count(m.get('P')),_count(m.get('U'))])
            for k in (5,10):
                eff=[r.get(f'effective_k_{k}',min(k,r['P']+r['U'])) for r in m.get('disease_metrics',[]) if 'P' in r and 'U' in r]
                ranking.append([label,k,*[number(m.get('ranking',{}).get(f'{name}@{k}')) for name in ('P','R','Hits','NDCG')],f'{min(eff)}–{max(eff)}' if eff else 'Unavailable'])
        self._write(_table(['Task','ROC-AUC','AP','AUPR (trapezoidal)','Evaluable/skipped','P','U'],headline))
        self._write(_table(['Task','k','Precision','Recall','Hits','NDCG','Effective k'],ranking))
        self._write('P = recorded positives. U = unlabeled candidates. AP and trapezoidal AUPR use separate calculations. Hits can be guaranteed when the panel is smaller than k.')
        self._evaluation=None
