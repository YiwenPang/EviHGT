"""Small immutable-publication and identity helpers, no dataset imports."""
import hashlib,json,os,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/ranking_protocol/ranking_comparison/part2'
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Immutable publication already exists: '+str(path))
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    from .resource_guard import guard_json_publication
    guard_json_publication(path,len(encoded.encode('utf-8'))+encoded.count('\n')*(len(os.linesep)-1))
    partial=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with partial.open('x',encoding='utf-8') as f:
        f.write(encoded);f.flush();os.fsync(f.fileno())
    os.rename(partial,path)
    return path
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
