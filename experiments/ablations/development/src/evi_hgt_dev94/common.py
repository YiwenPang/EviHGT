"""Small DEV94-only immutable JSON and source-identity helpers."""
import json
import os
from pathlib import Path
import uuid
from evi_hgt.phase9.identity import ROOT,file_sha,digest

OUT=ROOT/'outputs/phase9/phase94'
DATA=ROOT/'data/processed/phase94'
MANIFEST=OUT/'DEV94_EXECUTION_MANIFEST.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value,*,immutable=True):
    path=Path(path).resolve()
    if not (path.is_relative_to(OUT) or path.is_relative_to(DATA)):raise PermissionError('DEV94 output namespace required')
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists() and immutable:
        if read(path)!=value:raise FileExistsError('Immutable artifact differs: '+str(path))
        return
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('x',encoding='utf-8',newline='\n') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def source_hashes():
    old=read(ROOT/'outputs/phase9/phase93/PHASE9_G3_75_V1.execution.json')['source_sha256']
    # M4 environment changes are separately reported, not an assertion that the
    # old package closure describes the repaired environment.
    paths=list(old)
    paths += [str(p) for p in (ROOT/'src/evi_hgt_dev94').glob('*.py')]
    paths += [str(p) for p in (ROOT/'scripts/phase94').glob('*') if p.suffix in ('.py','.ps1')]
    paths += [str(p) for p in (ROOT/'tests/phase94').glob('*.py')]
    return {str(Path(p).resolve()):file_sha(p) for p in sorted(set(paths))}

def verify_sources(expected):
    for path,wanted in expected.items():
        if file_sha(path)!=wanted:raise ValueError('Bound source changed: '+path)

def validate_manifest(path=MANIFEST):
    p=Path(path).resolve()
    if p!=MANIFEST:raise PermissionError('Only the isolated DEV94 manifest is permitted')
    m=read(p)
    from .model import ARMS
    if (m.get('schema')!='DEV94_THREE_ARM_V1' or m.get('mode')!='DEVELOPMENT'
        or m.get('arms')!=list(ARMS) or m.get('updates_per_arm')!=41136
        or m.get('milestones')!=[10284,20568,41136]
        or m.get('seed')!=1289779183 or m.get('repeat')!=0 or m.get('fold')!=0
        or m.get('G3') is not False or m.get('G4') is not False or m.get('G5') is not False
        or m.get('evaluation_domains')!=['validation','auxiliary_validation']):
        raise PermissionError('DEV94 fixed authorization/recipe mismatch')
    verify_sources(m['source_sha256'])
    for name,h in m['artifact_sha256'].items():
        if file_sha(name)!=h:raise ValueError('Bound DEV94 artifact changed: '+name)
    return m
