"""Content identities and confined, portable input paths."""
from pathlib import Path
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_input(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Input paths must be nonempty relative POSIX paths')
    path = Path(relative)
    if path.is_absolute() or ':' in relative or '..' in path.parts:
        raise ValueError('Input paths must remain inside the portable corpus')
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise FileNotFoundError('Missing corpus input: ' + relative)
    return result
