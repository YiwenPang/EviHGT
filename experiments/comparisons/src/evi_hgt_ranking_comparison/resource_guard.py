"""Directory growth limits for new artifacts; never removes files."""
from pathlib import Path
import os,json,shutil
from .common import ROOT

DEFAULT_LIMITS={
    'data/cache/ranking_comparison/corpora':12_000_000_000,
    'data/cache/ranking_comparison/native':25_000_000_000,
    'outputs/ranking_protocol/ranking_comparison/part2_1/TEST_ONLY':25_000_000_000,
    'outputs/ranking_protocol/ranking_comparison/runs':400_000_000_000,
}
_limits=dict(DEFAULT_LIMITS)
DEFAULT_CATEGORIES={'evaluation':42_000_000_000,'logs_metadata':5_000_000_000,
                    'atomic':5_000_000_000,'uncommitted_orphans':20_000_000_000}
_categories=dict(DEFAULT_CATEGORIES)

def checkpoint_required_bytes(state):
    """Whole-state transaction allowance, including native permutations/best.

    Snapshot writers clone tensor views before serialization. Counting every
    tensor occurrence is intentionally conservative even for shared storage.
    A second tensor-sized allowance and 10MB cover serialization and metadata.
    """
    import torch
    def count(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(count(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(count(v) for v in value)
        return 0
    return 2*count(state)+10_000_000

def configure(limits):
    global _limits,_categories
    requested=limits.get('directory_limits_bytes',DEFAULT_LIMITS)
    checked={}
    for relative,cap in requested.items():
        path=(ROOT/relative).resolve()
        if relative not in DEFAULT_LIMITS or not path.is_relative_to(ROOT) or type(cap) is not int or cap<=0:raise ValueError('Unrecognized resource quota')
        checked[relative]=cap
    if set(checked)!=set(DEFAULT_LIMITS):raise ValueError('Complete resource quota set required')
    categories=limits.get('category_limits_bytes',DEFAULT_CATEGORIES)
    if set(categories)!=set(DEFAULT_CATEGORIES) or any(type(v) is not int or v<=0 for v in categories.values()):raise ValueError('Complete positive category quota set required')
    _limits=checked
    _categories=dict(categories)

def bytes_under(path):
    if not Path(path).exists():return 0
    total=0
    for directory,_,files in os.walk(path):
        for name in files:total+=(Path(directory)/name).stat().st_size
    return total

def check_growth(target,required_bytes=0):
    target=Path(target).resolve();rows=[]
    part24=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_4/TEST_ONLY').resolve()
    if target==part24 or target.is_relative_to(part24):
        from .runtime_budget24 import disk_guard24
        rows.append(disk_guard24(target,required_bytes))
    part23=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_3/TEST_ONLY').resolve()
    if target==part23 or target.is_relative_to(part23):
        actual=bytes_under(part23);cap=25_000_000_000
        if actual+required_bytes>cap:raise OSError('Part2.3 TEST_ONLY 25 GB cap exceeded')
        rows.append(dict(path=str(part23),actual_bytes=actual,required_bytes=required_bytes,limit_bytes=cap))
    # Independent Part2.2 authority; the historical Part2.1 quota is untouched.
    part22=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_2/TEST_ONLY').resolve()
    if target==part22 or target.is_relative_to(part22):
        actual=bytes_under(part22);cap=25_000_000_000
        if actual+required_bytes>cap:raise OSError('Part2.2 TEST_ONLY 25 GB cap exceeded')
        rows.append(dict(path=str(part22),actual_bytes=actual,required_bytes=required_bytes,limit_bytes=cap))
    for relative,cap in _limits.items():
        path=(ROOT/relative).resolve()
        if target==path or target.is_relative_to(path):
            actual=bytes_under(path)
            if actual+required_bytes>cap:raise OSError(f'Artifact directory limit at {path}: {actual} bytes used, {required_bytes} requested, {cap} allowed. Existing checkpoints retained.')
            rows.append(dict(path=str(path),actual_bytes=actual,required_bytes=required_bytes,limit_bytes=cap))
    base=(ROOT/'outputs/ranking_protocol/ranking_comparison/runs').resolve()
    if target==base or target.is_relative_to(base):
        usage=category_usage(base)
        relative=target.relative_to(base)
        category=('evaluation' if 'evaluation' in relative.parts else
                  'logs_metadata' if target.suffix in ('.json','.jsonl','.log') else 'atomic')
        for key,cap in _categories.items():
            # Every commit reserves metadata/log growth between durable anchors.
            extra=(required_bytes if key==category or (key=='uncommitted_orphans' and category=='atomic')
                   else 1_000_000 if key=='logs_metadata' else 0)
            if usage[key]+extra>cap:raise OSError('Artifact category limit exceeded: '+key+'. Existing checkpoints retained.')
            rows.append(dict(category=key,actual_bytes=usage[key],required_bytes=extra,limit_bytes=cap))
    return rows

def category_usage(base):
    base=Path(base);result={k:0 for k in DEFAULT_CATEGORIES}
    if not base.exists():return result
    paths=[Path(d)/n for d,_,names in os.walk(base) for n in names]
    linked=set()
    for path in paths:
        if path.name.endswith('.COMMIT.json'):
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                linked.add(Path(record.get('payload_path',record.get('checkpoint_path',''))).resolve())
            except (ValueError,OSError,TypeError):pass
    for path in paths:
        size=path.stat().st_size
        if 'evaluation' in path.relative_to(base).parts:result['evaluation']+=size
        elif path.suffix=='.partial':result['atomic']+=size
        elif path.name.endswith('.state.pt') and path.resolve() not in linked:result['uncommitted_orphans']+=size
        elif path.suffix in ('.json','.jsonl','.log'):result['logs_metadata']+=size
    return result

def guard_json_publication(path,encoded_bytes):
    path=Path(path).resolve();base=(ROOT/'outputs/ranking_protocol/ranking_comparison/runs').resolve()
    if path.is_relative_to(base):
        from .runtime_budget import disk_guard
        disk_guard(path,encoded_bytes)

def check_queue_capacity():
    """Reserve remaining new-run/native-cache caps plus the 70GB system floor.

    Existing new artifacts already consume physical disk and are subtracted
    from their remaining growth, so resume does not double-charge them.
    """
    remaining={}
    for relative in ('outputs/ranking_protocol/ranking_comparison/runs','data/cache/ranking_comparison/native'):
        path=ROOT/relative;actual=bytes_under(path);cap=_limits[relative]
        if actual>cap:raise OSError('Existing output exceeds the queue storage limit: '+relative)
        remaining[relative]=cap-actual
    required=sum(remaining.values())+70_000_000_000
    free=shutil.disk_usage(ROOT.anchor).free
    if free<required:raise OSError(f'The queue needs {required-free} more bytes. Existing checkpoints retained.')
    return dict(free_bytes=free,required_free_bytes=required,remaining_growth_bytes=remaining,system_reserve_bytes=70_000_000_000)
