"""RankingComparison versioned event wrapper; inherited EVIDENCE_ABLATION mechanics stay untouched."""
from evi_hgt_evidence_ablation.console import Console as AcceptedConsole
from evi_hgt_console import _table,_count,_number

def number(value):return '未计算/不可用' if value is None else _number(value,6)

class Console(AcceptedConsole):
    def __init__(self,*args,context='TEST_ONLY',queue_total=71,**kwargs):
        if context not in ('TEST_ONLY','回放','AUTHORIZED_RUNTIME'):raise ValueError('Explicit display context required')
        if type(queue_total) is not int or queue_total<0:raise ValueError('Invalid queue size')
        self.context=context;self.queue_total=queue_total
        super().__init__(*args,**kwargs)

    def stage(self,title,detail=''):
        super().stage(f'[{self.context}] {title}',detail)

    def banner(self,queue,budget,resumed_step=0):
        self.stage('EviHGT · RankingComparison',f'队列{self.queue_total}条；固定逻辑预算{budget:,}；当前模式{self.context}')
        self._write(' → '.join('/'.join(q) if isinstance(q,(tuple,list)) else str(q) for q in queue))

    def frame(self,width=None):
        return [line.replace(f'队列：已完成{self.completed}/4',f'队列：已完成{self.completed}/{self.queue_total}') for line in super().frame(width)]

    def _render_evaluation(self,*,force=False):
        # Accepted renderer hardcoded EVIDENCE_ABLATION queue size; temporarily formatting
        # is not a global patch and does not touch predictions/counters/RNG.
        now=self.clock()
        if not force and now-self._last_render<self.refresh_seconds:return
        from evi_hgt_evidence_ablation.console import percent,wrap
        e=self._evaluation
        self._draw(wrap(f'[{self.context}] 评价查询：{e["completed"]:,} / {e["total"]:,}（{percent(e["completed"],e["total"])}）；队列已完成{self.completed}/{self.queue_total}',self._columns()))
        self._last_render=now

    def evaluation_result(self,arm,step,metrics):
        self.stage('评价指标 · '+str(arm),f'第{step:,}步 · 仅显示已计算结果；回放不代表新评分')
        metrics=metrics.get('metrics',metrics);headline=[];ranking=[]
        for head,label in (('support','支持'),('contra','反证')):
            m=metrics.get(head,{}) or {}
            headline.append([label,*[number(m.get(k)) for k in ('macro_auc','macro_ap','macro_aupr_trapezoid')],f"{_count(m.get('evaluable_diseases'))}/{_count(m.get('skipped_diseases'))}",_count(m.get('P')),_count(m.get('U'))])
            for k in (5,10):
                eff=[r.get(f'effective_k_{k}',min(k,r['P']+r['U'])) for r in m.get('disease_metrics',[]) if 'P' in r and 'U' in r]
                ranking.append([label,k,*[number(m.get('ranking',{}).get(f'{name}@{k}')) for name in ('P','R','Hits','NDCG')],f'{min(eff)}–{max(eff)}' if eff else '未计算/不可用'])
        self._write(_table(['通道','ROC-AUC','AP','AUPR（梯形）','可评/跳过','P','U'],headline))
        self._write(_table(['通道','k','Precision','Recall','Hits','NDCG','有效k'],ranking))
        self._write('P=已知正例；U=未知候选。AP与梯形AUPR独立；N≤K时Hits可能由候选构造保证。')
        self._evaluation=None
