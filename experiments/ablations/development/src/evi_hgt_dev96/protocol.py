import json
import os
from pathlib import Path
import uuid

from evi_hgt.phase9.identity import ROOT, digest, file_sha, CellContract
from evi_hgt_dev94.common import read
from evi_hgt_dev95.protocol import ARMS, LOSSES, DOMAINS

OUT=ROOT/'outputs/phase9/phase96'
CACHE=ROOT/'data/cache/phase96'
MANIFEST=OUT/'DEV96_EXECUTION_MANIFEST.json'
CELLS={
    'R0F1':dict(repeat=0,fold=1,seed=342565966,G=6847,milestones=[10270,20541],total=20541),
    'R1F0':dict(repeat=1,fold=0,seed=625224385,G=6900,milestones=[10350,20700],total=20700),
}
QUEUE=tuple((c,a) for c in CELLS for a in ARMS)


def cell_spec(cell):
    if cell not in CELLS:raise PermissionError('Only registered R0F1 and R1F0 development cells')
    return dict(CELLS[cell])


def frozen_cell(cell):
    s=cell_spec(cell)
    c=CellContract.load('M0',s['repeat'],s['fold']).bindings
    census=read(ROOT/'outputs/phase85_part3b1/cv_15_cell_static_census.json')
    rows=[r for r in census['cells'] if (r['repeat_index'],r['fold_index'])==(s['repeat'],s['fold'])]
    if len(rows)!=1:raise ValueError('Original census cell missing')
    r=rows[0]
    for k in ('repeat_key','cell_digest','training_seed','membership_digests','mask_digest'):
        if r[k]!=c[k]:raise ValueError('Frozen census identity differs: '+k)
    if c['training_seed']!=s['seed'] or c['budget']['G_rf']!=s['G'] or c['budget']['B_rf']!=s['milestones'][0]:
        raise ValueError('Seed/G/budget disagree with frozen census')
    return c


def _path(path,derived=False):
    p=Path(path).resolve()
    if not p.is_relative_to((CACHE if derived else OUT).resolve()):raise PermissionError('DEV96 namespace required')
    return p


def write(path,value,*,immutable=True,derived=False):
    p=_path(path,derived);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and immutable:
        if read(p)!=value:raise FileExistsError('Immutable DEV96 artifact differs: '+str(p))
        return
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
    with t.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def journal(path,row):
    p=_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def checkpoint_due(cell,step):
    s=cell_spec(cell)
    return 0<step<=s['total'] and (step%1024==0 or step==s['G'] or step in s['milestones'])


def check_evaluation(cell,step,domain):
    if step not in cell_spec(cell)['milestones'] or domain not in DOMAINS:raise PermissionError('DEV96 fixed cell development milestones only')


def identity(m,cell,arm,mode='DEVELOPMENT'):
    if arm not in ARMS or mode!='DEVELOPMENT':raise PermissionError('DEV96 development arms only')
    c=m['cells'][cell]
    return dict(cell=cell,arm=arm,loss_id=LOSSES[arm],mode=mode,manifest_sha256=file_sha(MANIFEST),
        source_sha256=digest(m['source_sha256']),data_bindings=c['data_bindings'],
        schedule_sha256=c['schedule_digest'],model_initial_sha256=c['initial_tensor_sha256'])


def source_hashes():
    sources=dict(read(ROOT/'outputs/phase9/phase95/DEV95_EXECUTION_MANIFEST.json')['source_sha256'])
    for d in ('src/evi_hgt_dev96','scripts/phase96','tests/phase96'):
        for p in (ROOT/d).glob('*'):
            if p.suffix in ('.py','.ps1'):sources[str(p.resolve())]=file_sha(p)
    return sources


def validate_manifest():
    m=read(MANIFEST)
    fixed=dict(schema='DEV96_CROSS_CELL_PAIRED_DEVELOPMENT_V1',mode='DEVELOPMENT',authorization='DEV96_CROSS_CELL_PAIRED_DEVELOPMENT',
        cells_order=list(CELLS),queue=[list(q) for q in QUEUE],arms=list(ARMS),losses=LOSSES,total_logical_updates=82482,
        old_G3='HELD',outer='UNSCORED',new_final='SEALED',G4=False,G5=False,old_Formal_access=1,new_Final_access=0)
    if any(m.get(k)!=v for k,v in fixed.items()):raise PermissionError('DEV96 authorization/queue changed')
    for cell,s in CELLS.items():
        if m['cells'][cell]['spec']!=s:raise ValueError('Cell budget changed')
    for k in ('source_sha256','artifact_sha256'):
        for p,h in m[k].items():
            if file_sha(p)!=h:raise ValueError('Bound DEV96 source/artifact changed: '+p)
    return m
