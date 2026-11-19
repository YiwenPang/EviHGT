"""Part2.1 independent, durable engineering authority and single-worker ledger."""
import os,time,threading,uuid,shutil
from contextlib import contextmanager
from pathlib import Path
from .common import ROOT,read,sha,digest,write_new,append

OUT21=ROOT/'outputs/ranking_protocol/ranking_comparison/part2_1'
LIMITS=dict(updates=192,forwards=4096,queries=4096,seconds=2700.,HGTDR=4,TX_PRETRAIN=4,TX_FINETUNE=4)

def disk_guard(path,required_bytes=0,*,reserve_bytes=70_000_000_000):
    free=shutil.disk_usage(Path(path).anchor).free
    if free < reserve_bytes+required_bytes:raise OSError('磁盘水位不足；保留有效恢复锚')
    from .resource_guard import check_growth
    quotas=check_growth(path,required_bytes)
    return dict(free_bytes=free,required_bytes=required_bytes,reserve_bytes=reserve_bytes,directory_limits=quotas)

class RuntimeBudget:
    def __init__(self):
        self.root=OUT21/'TEST_ONLY';self.events=self.root/'BUDGET_EVENTS.jsonl';self.started=None
    def totals(self):
        rows=[__import__('json').loads(s) for s in self.events.read_text().splitlines()] if self.events.exists() else []
        return {k:sum(r.get(k,0) for r in rows if r['event']==('END' if k=='seconds' else 'RESERVE')) for k in LIMITS}
    def check(self,**cost):
        total=self.totals()
        for key,limit in LIMITS.items():
            value=total[key]+cost.get(key,0)+(time.monotonic()-self.started if key=='seconds' and self.started else 0)
            if value>limit:raise RuntimeError('Part2.1累计硬上限: '+key)
        if self.started:
            for k,v in self.spec['limits'].items():
                if k=='seconds':used=time.monotonic()-self.started
                else:used=total[k]-self.initial[k]+cost.get(k,0)
                if used>v:raise RuntimeError('登记run上限: '+k)
        return total
    @contextmanager
    def session(self,spec):
        self.root.mkdir(parents=True,exist_ok=True)
        if spec.get('mode')!='TEST_ONLY' or spec.get('authority')!='USER_RANKING_COMPARISON_PART21_2026_10_29':raise PermissionError('Part2.1 engineering authority required')
        if any(spec.get(k) is not False for k in ('formal_training_authorized','outer_eval_authorized','final_eval_authorized','diagnostics_authorized')):raise PermissionError('TEST_ONLY scope differs')
        if spec.get('digest')!=digest({k:v for k,v in spec.items() if k!='digest'}):raise ValueError('Registration digest differs')
        for p,h in spec['source_sha256'].items():
            if sha(p)!=h:raise ValueError('Registered source changed '+p)
        if not spec.get('fixtures') or not spec.get('purpose'):raise ValueError('Missing registered fixtures/purpose')
        from .safety import snapshot
        snapshot();disk_guard(self.root,2_000_000_000)
        rows=[__import__('json').loads(s) for s in self.events.read_text().splitlines()] if self.events.exists() else []
        closed={r['session_id'] for r in rows if r['event']=='END'}
        if any(r['event']=='START' and r['session_id'] not in closed for r in rows):raise RuntimeError('Previous session time unresolved')
        if any(r.get('run_id')==spec['run_id'] for r in rows):raise ValueError('Run ID already used; new registered evidence-driven attempt required')
        self.check();lock=self.root/'active_worker.lock';token=uuid.uuid4().hex
        write_new(lock,dict(pid=os.getpid(),token=token))
        self.initial=self.totals();self.started=time.monotonic();self.spec=spec;self.sid=uuid.uuid4().hex
        import psutil
        process=psutil.Process();peak=[process.memory_info().rss];monitor_stop=threading.Event()
        def monitor():
            while not monitor_stop.wait(.2):
                peak[0]=max(peak[0],process.memory_info().rss)
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        append(self.events,dict(event='START',session_id=self.sid,run_id=spec['run_id'],pid=os.getpid()))
        def timeout():
            append(self.events,dict(event='END',session_id=self.sid,run_id=spec['run_id'],seconds=time.monotonic()-self.started,status='HARD_TIME_LIMIT'))
            os._exit(124)
        timer=threading.Timer(min(2700-self.initial['seconds'],spec['limits']['seconds']),timeout);timer.daemon=True;timer.start()
        try:yield self
        finally:
            timer.cancel();monitor_stop.set();watcher.join()
            append(self.events,dict(event='END',session_id=self.sid,run_id=spec['run_id'],seconds=time.monotonic()-self.started,status='RETURNED',peak_cpu_rss_bytes=peak[0],rss_sampling_seconds=.2))
            self.started=None
            if lock.exists() and read(lock).get('token')==token:lock.unlink()
    def charge(self,**cost):
        if self.started is None:raise PermissionError('No active engineering session')
        if any(k not in LIMITS or type(v) is not int or v<0 for k,v in cost.items()):raise ValueError('Invalid cost')
        self.check(**cost);append(self.events,dict(event='RESERVE',session_id=self.sid,run_id=self.spec['run_id'],**cost))
    def forward(self,call,queries=1,*,label=''):
        self.charge(forwards=1,queries=queries);value=call()
        append(self.events,dict(event='FORWARD_DONE',session_id=self.sid,label=label,queries=queries));return value
    def step(self,optimizer,*,label='',replay=False,native=None):
        cost=dict(updates=1)
        if native:cost[native]=1
        self.charge(**cost);optimizer.step()
        append(self.events,dict(event='UPDATE_DONE',session_id=self.sid,label=label,replay=replay,native=native))
    def native_call(self,kind,call,*,optimizer_updates,forward_calls,positive_cap,negative_cap,relation_edge_cap,replay=False):
        key={'HGTDR':'HGTDR','HGTDR_ADAPTED':'HGTDR','TX_PRETRAIN':'TX_PRETRAIN','TX_FINETUNE':'TX_FINETUNE'}[kind]
        declared=self.spec['native_caps'][key]
        if any(v<0 or v>declared[k] for k,v in dict(positive_cap=positive_cap,negative_cap=negative_cap,relation_edge_cap=relation_edge_cap).items()):raise ValueError('Native entry cap differs from registration')
        self.charge(updates=optimizer_updates,forwards=forward_calls,**{key:optimizer_updates})
        append(self.events,dict(event='NATIVE_BEGIN',session_id=self.sid,kind=key,positive_cap=positive_cap,negative_cap=negative_cap,relation_edge_cap=relation_edge_cap,replay=replay))
        value=call()
        append(self.events,dict(event='NATIVE_DONE',session_id=self.sid,kind=key,optimizer_updates=optimizer_updates,forward_calls=forward_calls,replay=replay))
        return value

class EngineeringCapability:
    def __init__(self,budget):
        if not isinstance(budget,RuntimeBudget) or budget.started is None:raise PermissionError('Active Part2.1 session required')
        self.budget=budget;self.mode='TEST_ONLY'
    def check_native(self,method,cell):
        self.budget.check()
        if method not in self.budget.spec['methods'] or cell!=self.budget.spec['cell']:raise PermissionError('Native scope not registered')
