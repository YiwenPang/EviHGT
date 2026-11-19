"""Exact original streaming projections; other role arrays are never decoded."""
import json
import re
import hashlib
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def objects_in_array(path, field):
    """Bounded streaming slices. Other arrays are scanned, never JSON-decoded."""
    with Path(path).open(encoding='utf-8-sig') as handle:
        prefix = ''
        while True:
            c = handle.read(1)
            if not c:
                raise ValueError('Missing projected array: ' + field)
            prefix = (prefix + c)[-128:]
            if re.search(r'"' + re.escape(field) + r'"\s*:\s*\[$', prefix):
                break
        depth = 0; quoted = False; escaped = False; buf = []
        while True:
            c = handle.read(1)
            if not c:
                raise ValueError('Truncated projected array')
            if not depth:
                if c == ']':
                    return
                if c in ' \t\r\n,':
                    continue
                if c != '{':
                    raise ValueError('Expected object in projected array')
            buf.append(c)
            if quoted:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': quoted = False
            elif c == '"': quoted = True
            elif c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if not depth:
                    yield ''.join(buf)
                    buf = []

def field(raw, key, integer=False):
    pattern = r'(-?\d+)' if integer else r'("(?:[^"\\]|\\.)*")'
    values = re.findall(r'"'+re.escape(key)+r'"\s*:\s*'+pattern, raw)
    if len(values) != 1:
        raise ValueError('Missing/duplicate projected field ' + key)
    return int(values[0]) if integer else json.loads(values[0])

def reserved(pair, key, fraction):
    return int(digest([key, *pair]), 16) < int(fraction * (1 << 256))

def project_training_assignments(path, *, fold, repeat_key):
    key = digest([repeat_key, fold, 'inner'])
    rows = []; outer_skipped = 0; reserved_skipped = 0
    for raw in objects_in_array(path, 'assignments'):
        if field(raw, 'fold', True) == fold:
            outer_skipped += 1
            continue
        pair = field(raw, 'drug_id'), field(raw, 'disease_id')
        if reserved(pair, key+'validation', .2) or reserved(pair, key+'probe', .05):
            reserved_skipped += 1
            continue
        role = field(raw, 'role')
        if role not in ('P', 'U'):
            raise ValueError('Invalid training role')
        rows.append(dict(drug_id=pair[0], disease_id=pair[1], role=role))
    rows.sort(key=lambda r: (r['drug_id'], r['disease_id']))
    return rows, dict(mode='STREAMING_TRAINING_PREDICATE_BEFORE_ROLE_DECODE',
        outer_objects_skipped_before_id_decode=outer_skipped, outer_ids_decoded=0,
        outer_roles_decoded=0, reserved_roles_decoded=0,
        reserved_objects_skipped_before_role_decode=reserved_skipped)
