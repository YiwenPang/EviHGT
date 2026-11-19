"""Copy and validate explicitly supplied normalized artifacts without changing sources."""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import gc
import json
import os
import shutil
import uuid

from .common import read_json, sha256
from .corpus import load_corpus, SCHEMA


def destination_path(relative):
    value = PurePosixPath(relative)
    if not relative or value.is_absolute() or '..' in value.parts or '\\' in relative or ':' in relative or relative == 'manifest.json':
        raise ValueError('Invalid portable destination path: ' + str(relative))
    return value


def prepare_from_spec(spec_path, output, *, progress=None):
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    if spec.get('schema') != 'EVIHGT_SOURCE_MAP_V1':
        raise ValueError('Expected EVIHGT_SOURCE_MAP_V1; see docs/data.md')
    manifest = deepcopy(spec['manifest'])
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Source map must declare the portable corpus schema')
    sources = spec['sources']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Source map requires explicit local source files')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new derived corpus directory: ' + str(output))
    # Validate every supplied file before creating any destination file.
    resolved = {}
    for i, (relative, descriptor) in enumerate(sources.items(), 1):
        destination_path(relative)
        if not isinstance(descriptor, dict) or set(descriptor) != {'path','sha256'}:
            raise ValueError('Each source requires exactly path and sha256')
        source = Path(descriptor['path'])
        source = (spec_path.parent/source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file() or source.is_relative_to(output):
            raise FileNotFoundError('Missing or invalid explicit source file: ' + str(source))
        if sha256(source) != descriptor['sha256']:
            raise ValueError('Source hash mismatch: ' + str(source))
        resolved[relative] = source
        if progress:
            progress('source verification',i,len(sources),0)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.attempt_' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        manifest['files'] = {}
        for i, (relative, source) in enumerate(sorted(resolved.items()), 1):
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            actual = sha256(target)
            if actual != sources[relative]['sha256']:
                raise ValueError('Input changed while being copied: ' + str(source))
            manifest['files'][relative] = actual
            if progress:
                progress('copy', i, len(resolved), 0)
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False),encoding='utf-8')
        corpus = load_corpus(staging/'manifest.json',progress=progress)
        identity = corpus.identity
        del corpus
        gc.collect()  # Close Windows mmap handles before atomic directory publication.
        os.rename(staging,output)
        return dict(manifest=str(output/'manifest.json'),identity=identity,files=len(resolved),
                    validation='PASSED',source_files_modified=False)
    except BaseException:
        # Incomplete artifacts are retained under the explicitly named attempt.
        raise
