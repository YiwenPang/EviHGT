"""Read-only certified R0F0 graph with an independent narrow development gate."""
from contextlib import contextmanager
from pathlib import Path
import time

from evi_hgt.phase9.real_data import RealInputs,load_index,load_canonical_assembler
from evi_hgt.phase85_part3b1.data import assert_corpus_identity
from evi_hgt.phase85_v2.common import verify
from .common import ROOT,read,file_sha

class DevelopmentData:
    def __init__(self,original):
        self.original=original
        self._evaluation_step=None
        self.allowed={n:{(r['drug_id'],r['disease_id']) for r in original.cell[n]}
                      for n in ('validation','auxiliary_validation')}

    def __getattr__(self,name):return getattr(self.original,name)

    @contextmanager
    def evaluation_scope(self,step):
        if step not in (10284,20568,41136) or self._evaluation_step is not None:
            raise PermissionError('Only predeclared nonnested DEV94 milestone evaluation')
        self._evaluation_step=step
        try:yield
        finally:self._evaluation_step=None

    def assert_score_domain(self,pair,domain):
        if (self._evaluation_step is None or domain not in self.allowed
            or tuple(pair) not in self.allowed[domain] or tuple(pair) in self.frozen):
            raise PermissionError('No probe/outer/final or nonmember DEV94 scoring route')

    def enable_development(self,*args,**kwargs):raise PermissionError('Old G3 grant cannot enable DEV94')

def load_data():
    d=DevelopmentData(RealInputs().cell(0,0))
    if d.training_seed!=1289779183 or d.budget()['G_rf']!=6856:raise ValueError('Wrong frozen cell')
    roots=[d.original_corpus_root,ROOT/'outputs/phase9/formal/PHASE9_G3_75_V1/data/r0_f0/corpus']
    existing=[p for p in roots if (p/'manifest.json').is_file()]
    if not existing:raise ValueError('Certified R0F0 corpus unavailable; no automatic rebuild')
    d.corpus_root=existing[0]
    corpus=verify(read(d.corpus_root/'manifest.json'))
    assert_corpus_identity(d,corpus)
    for p,h in {**corpus['inputs'],**corpus['outputs']}.items():
        if file_sha(p)!=h:raise ValueError('Certified corpus identity changed: '+p)
    return d,corpus

def prepare(d,corpus):
    started=time.monotonic();last=[0.]
    def progress(*values):
        if time.monotonic()-last[0]>20:
            print('INDEX',values,flush=True);last[0]=time.monotonic()
    index=load_index(corpus,progress=progress)
    assembler=load_canonical_assembler(d,corpus,index)
    return index,assembler,time.monotonic()-started
