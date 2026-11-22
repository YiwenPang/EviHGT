"""Presentation-only progress: one tqdm line in TTY; periodic text otherwise."""
import sys,time,threading
from contextlib import contextmanager
from tqdm import tqdm

class RuntimeConsole:
    def __init__(self,method,cell,mode,*,stream=None,refresh=5.):
        self.method=method;self.cell=cell;self.mode=mode;self.stream=stream or sys.stdout
        self.refresh=refresh;self.tty=bool(getattr(self.stream,'isatty',lambda:False)())
        self.bar=None;self.last=0.;self.started=time.monotonic();self.updates=0
    def write(self,text):
        print(text,file=self.stream,flush=True)
    @contextmanager
    def preparing(self):
        stop=threading.Event();started=time.monotonic()
        self.write(f'━━ [{self.mode}] {self.method}/{self.cell} · Preparing data and features ━━')
        def heartbeat():
            while not stop.wait(self.refresh):
                self.write(f'[{self.mode}] Preparing inputs for {time.monotonic()-started:.0f} s. Optimizer updates: 0.')
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:yield
        finally:stop.set();thread.join()
    def update(self,row,G):
        now=time.monotonic();self.updates+=1
        epoch=min(3,(row['source_step']-1)//G+1)
        duration=sum(row[k] for k in ('prepare_seconds','forward_seconds','backward_seconds','optimizer_seconds'))
        speed=1/duration if duration>0 else 0
        remaining=(3*G-row['source_step'])/speed if speed else None
        eta='n/a (test fixture)' if self.mode=='TEST_ONLY' else f'{remaining/3600:.2f} h' if remaining is not None else 'unknown'
        info=(f"Epoch {epoch}/3 loss={row['loss']:.6g} {speed:.3f} updates/s ETA={eta} "
              f"GPU allocated={row['allocated_peak']/2**30:.3f}GiB reserved={row['reserved_peak']/2**30:.3f}GiB")
        prefix=f'[{self.mode}] {self.method}/{self.cell}'
        if self.mode=='TEST_ONLY':prefix+=' fixture'+(' · replay' if row['replay'] else '')
        if self.tty:
            if self.bar is None:
                self.bar=tqdm(total=None if self.mode=='TEST_ONLY' else 3*G,desc=prefix,file=self.stream,mininterval=self.refresh,dynamic_ncols=True)
            self.bar.set_postfix_str(info,refresh=False);self.bar.update(1)
        elif now-self.last>=self.refresh or self.updates==1:
            self.write(prefix+f" source_step={row['source_step']} · "+info);self.last=now
    def close(self):
        if self.bar is not None:self.bar.close();self.bar=None

class NativeConsole:
    def __init__(self,method,mode):
        self.method=method;self.mode=mode;self.last=0.;self.started=time.monotonic();self.bar=None
    def update(self,row,*,gpu_bytes=None,replay=False):
        now=time.monotonic();cursor=row.get('cursor',{})
        stage=row.get('stage','HGTDR')
        epoch=row.get('epoch',cursor.get('finetune_epoch',0)) if stage!='PRETRAIN' else 1
        total=300 if stage=='HGTDR' else 1 if stage=='PRETRAIN' else 500
        seconds=row.get('elapsed_seconds',row.get('seconds',0));speed=1/seconds if seconds else 0.
        memory='CPU' if gpu_bytes is None else f'{gpu_bytes/2**30:.3f}GiB'
        prefix=f'[{self.mode}] {self.method} {stage}'+(' · replay' if replay else '')
        detail=f"Epoch {epoch}/{total} loss={row['loss']:.6g} {speed:.3f} updates/s ETA="+('n/a (test fixture)' if self.mode=='TEST_ONLY' else 'native stage')+f' GPU={memory}'
        if sys.stdout.isatty():
            if self.bar is None:self.bar=tqdm(total=None,desc=prefix,mininterval=5,dynamic_ncols=True)
            self.bar.set_postfix_str(detail,refresh=False);self.bar.update()
        elif now-self.last>=5:
            print(prefix+' · '+detail,flush=True);self.last=now
    def close(self):
        if self.bar is not None:self.bar.close()
