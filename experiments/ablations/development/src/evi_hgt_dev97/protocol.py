"""DEV97-only identities and immutable runtime scope."""
import json
import os
from pathlib import Path
import uuid
from evi_hgt_dev96.protocol import ROOT,CELLS,cell_spec,checkpoint_due,check_evaluation,read,digest,file_sha
from .model import ARMS

OUT=ROOT/'outputs/phase9/phase97'
MANIFEST=OUT/'DEV97_EXECUTION_MANIFEST.json'
QUEUE=tuple((c,a) for c in CELLS for a in ARMS)
LOSSES={a:'EMPIRICAL_PU_HALF_SOFTPLUS_V1' for a in ARMS}
DOMAINS=('validation','auxiliary_validation')
HISTORY=ROOT/'outputs/phase9/phase96'
REFERENCE_ARM='E1_D1_SOFTPLUS_RANKING'


def write(path,value,*,immutable=True):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT.resolve()):raise PermissionError('Only DEV97 output namespace')
    p.parent.mkdir(parents=True,exist_ok=True)
    if immutable and p.exists():
        if read(p)!=value:raise FileExistsError('Immutable DEV97 artifact differs: '+str(p))
        return
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
    with t.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def journal(path,row):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT.resolve()):raise PermissionError('Only DEV97 logs')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def identity(m,cell,arm,mode='DEVELOPMENT'):
    cell_spec(cell)
    if arm not in ARMS or mode not in ('DEVELOPMENT','TEST_ONLY'):raise PermissionError('DEV97 only')
    c=m['cells'][cell]
    return dict(cell=cell,arm=arm,ablation=arm,loss_id=LOSSES[arm],mode=mode,manifest_sha256=file_sha(MANIFEST),
        source_sha256=digest(m['source_sha256']),data_bindings=c['data_bindings'],schedule_sha256=c['schedule_digest'],
        model_initial_sha256=c['initial_tensor_sha256'])


def sources():
    from evi_hgt_dev96.protocol import validate_manifest
    result=dict(validate_manifest()['source_sha256'])
    for folder in ('src/evi_hgt_dev97','scripts/phase97','tests/phase97'):
        for p in (ROOT/folder).glob('*'):
            if p.suffix in ('.py','.ps1'):result[str(p.resolve())]=file_sha(p)
    return result


def validate_manifest():
    m=read(MANIFEST)
    if m['authorization']!='DEV97_CORE_ABLATION_DEVELOPMENT' or m['queue']!=[list(q) for q in QUEUE] or m['total_logical_updates']!=82482:
        raise PermissionError('DEV97 authorization/queue/budget differs')
    if (m['old_G3'],m['outer'],m['new_final'],m['G4'],m['G5'])!=('HELD','UNSCORED','SEALED',False,False):raise PermissionError('Protected scope changed')
    for kind in ('source_sha256','artifact_sha256'):
        for p,h in m[kind].items():
            if file_sha(p)!=h:raise ValueError('DEV97 bound identity changed: '+p)
    for cell in CELLS:
        if m['cells'][cell]['spec']!=cell_spec(cell):raise ValueError('Cell spec differs')
    return m
