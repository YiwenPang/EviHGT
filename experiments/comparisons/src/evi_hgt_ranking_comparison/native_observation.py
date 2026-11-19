"""Observe one already-delivered native batch without importing DGL.

This module never requests a batch, calls a model, seeds a generator or changes
runtime flags. Callers pass the exact (input_nodes, positive, negative, blocks)
tuple they already received. Graph access uses DGL's public duck-typed views.
Supply nid_key=dgl.NID/eid_key=dgl.EID at the integration boundary, or expose
NID/EID (or nid_key/eid_key) attributes on the graph interface.

All tensors, including every node/edge data field, are independent CPU copies.
The local-node row IDs and corresponding NID vectors retain the full local to
global mapping for positive/negative graphs and each block side. Missing IDs
remain explicitly unavailable; local IDs are never relabeled as global IDs.
V2 includes features and ordered data-field inventories; V1 manifests remain
readable. Neither version replaces the historical batch digest by itself.
"""
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re

import torch


LEGACY_SCHEMA = 'RANKING_COMPARISON_NATIVE_BATCH_OBSERVATION_V1'
LEGACY_MANIFEST_SCHEMA = 'RANKING_COMPARISON_NATIVE_BATCH_OBSERVATION_MANIFEST_V1'
SCHEMA = 'RANKING_COMPARISON_NATIVE_BATCH_OBSERVATION_V2'
MANIFEST_SCHEMA = 'RANKING_COMPARISON_NATIVE_BATCH_OBSERVATION_MANIFEST_V2'
TEST_ONLY = Path(__file__).resolve().parents[2] / 'outputs/ranking_protocol/ranking_comparison/part2_1/TEST_ONLY'
TEST_ONLY22 = Path(__file__).resolve().parents[2] / 'outputs/ranking_protocol/ranking_comparison/part2_2/TEST_ONLY'


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.is_quantized or value.device.type == 'meta'):
        raise ValueError(name + ' must be a materialized dense tensor')
    return value.detach().resolve_conj().resolve_neg().cpu().clone(memory_format=torch.contiguous_format)


def _ids(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64)):
        raise ValueError(name + ' must be a one-dimensional integer ID tensor')
    return _tensor(value, name)


def _key(graph, supplied, name):
    value = supplied
    if value is None:
        value = getattr(graph, name, None)
    if value is None:
        value = getattr(graph, name.lower() + '_key', None)
    if value is not None and not isinstance(value, str):
        raise ValueError(name + ' field key must be a string')
    return value


def _field(data, key, length, name):
    if key is None:
        return dict(status='KEY_UNSPECIFIED', field_key=None, ids=None)
    if key not in data:
        return dict(status='ABSENT', field_key=key, ids=None)
    value = _ids(data[key], name)
    if value.numel() != length:
        raise ValueError(name + ' length differs from its local graph domain')
    return dict(status='PRESENT', field_key=key, ids=value)


def _types(values, name):
    result = list(values)
    if any(not isinstance(x, str) for x in result) or len(result) != len(set(result)):
        raise ValueError(name + ' must contain unique type names in interface order')
    return result


def _data_fields(data, length, name):
    """Copy all delivered fields, including IDs, without allowlists or sorting.

    An unresolved lazy-feature object is an incomplete observation, never a
    field to silently omit. Public data access may materialize a graph library's
    read cache, but no graph assignment, sampler, or generator call occurs here.
    """
    fields = _types(data.keys(), name + ' data fields')
    result = {}
    for field in fields:
        value = _tensor(data[field], name + ' data field ' + field)
        if value.ndim == 0 or value.shape[0] != length:
            raise ValueError(name + ' data field ' + field + ' length differs from its local graph domain')
        result[field] = value
    return dict(data_field_order=fields, data=result)


def _nodes(view, types, nid_key):
    result = []
    for ntype in types:
        local = _ids(view(ntype=ntype), 'local node IDs for ' + ntype)
        data = view[ntype].data
        result.append(dict(type=ntype, local_ids=local,
                           NID=_field(data, nid_key, local.numel(), 'NID for ' + ntype),
                           **_data_fields(data, local.numel(), 'node ' + ntype)))
    return result


def _graph(graph, *, block, nid_key, eid_key):
    if bool(getattr(graph, 'is_block', False)) != block:
        raise ValueError('positive/negative graphs and message blocks have different interfaces')
    nid_key = _key(graph, nid_key, 'NID')
    eid_key = _key(graph, eid_key, 'EID')
    result = dict(is_block=block)
    if block:
        source = _types(graph.srctypes, 'source node types')
        destination = _types(graph.dsttypes, 'destination node types')
        result.update(source_node_type_order=source, source_nodes=_nodes(graph.srcnodes, source, nid_key),
                      destination_node_type_order=destination,
                      destination_nodes=_nodes(graph.dstnodes, destination, nid_key))
    else:
        types = _types(graph.ntypes, 'node types')
        result.update(node_type_order=types, nodes=_nodes(graph.nodes, types, nid_key))
    etypes = [tuple(x) for x in graph.canonical_etypes]
    if (any(len(x) != 3 or any(not isinstance(y, str) for y in x) for x in etypes)
            or len(etypes) != len(set(etypes))):
        raise ValueError('canonical edge types must be unique source/relation/target triples')
    result['canonical_etype_order'] = [list(x) for x in etypes]
    result['edges'] = []
    for etype in etypes:
        # order='eid' observes stored local edge order; never sort endpoints or
        # original EIDs to manufacture agreement between two actual batches.
        source, target = graph.edges(etype=etype, order='eid')
        source, target = _ids(source, 'edge source'), _ids(target, 'edge destination')
        if source.numel() != target.numel():
            raise ValueError('edge source/destination length differs')
        data = graph.edges[etype].data
        result['edges'].append(dict(canonical_etype=list(etype), src=source, dst=target,
            EID=_field(data, eid_key, source.numel(), 'EID for ' + repr(etype)),
            **_data_fields(data, source.numel(), 'edge ' + repr(etype))))
    return result


def _context(value):
    """Clone already-fetched control/RNG values, never inspect runtime objects."""
    if isinstance(value, torch.Tensor):
        return _tensor(value, 'context')
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('context mapping keys must be strings')
        return {key: _context(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_context(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('context must contain already-fetched tensors or plain values')


def snapshot_native_batch(batch, *, nid_key=None, eid_key=None, context=None):
    """Copy a supplied ordered batch and all data; never advance an iterator.

    ``blocks`` must already be a list/tuple, so an accidentally supplied loader
    cannot be consumed by this diagnostic. Node, edge, type and block order is
    retained exactly as exposed by the supplied interfaces. Optional ``context``
    contains caller-fetched stage, cursor, permutation, boundaries, sampler or
    generator states. It is cloned, not fetched implicitly, and contributes to
    the digest. Do not include run labels or file paths in comparable context.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise ValueError('native observation requires a four-component batch')
    input_nodes, positive, negative, blocks = batch
    if not isinstance(blocks, (tuple, list)):
        raise ValueError('message blocks must be an already-delivered list or tuple')
    if isinstance(input_nodes, Mapping):
        types = _types(input_nodes, 'input node types')
        inputs = dict(kind='typed', type_order=types,
                      ids={name: _ids(input_nodes[name], 'input nodes for ' + name) for name in types})
    else:
        inputs = dict(kind='homogeneous', ids=_ids(input_nodes, 'input nodes'))
    return dict(schema=SCHEMA, context=_context(context), input_nodes=inputs,
                positive=_graph(positive, block=False, nid_key=nid_key, eid_key=eid_key),
                negative=_graph(negative, block=False, nid_key=nid_key, eid_key=eid_key),
                blocks=[_graph(graph, block=True, nid_key=nid_key, eid_key=eid_key) for graph in blocks])


def native_batch_manifest(payload):
    """Hash each ordered tensor component and the full observation structure."""
    if not isinstance(payload, dict) or payload.get('schema') not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError('unknown native observation payload schema')
    schema = payload['schema']
    components = []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu' or value.layout != torch.strided or value.is_quantized:
                raise ValueError('manifest requires an independent CPU observation payload')
            if schema == LEGACY_SCHEMA and (value.ndim != 1 or value.dtype not in (torch.int32, torch.int64)):
                raise ValueError('manifest identity components must be integer ID tensors')
            descriptor = dict(dtype=str(value.dtype), shape=list(value.shape))
            hasher = hashlib.sha256(_canonical(descriptor))
            # Byte views preserve all dense dtypes, including BF16; converting
            # to float32 or NumPy's unsupported BF16 dtype would alter evidence.
            raw = value.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
            hasher.update(raw.numpy().tobytes(order='C'))
            component = dict(path=list(path), **descriptor, numel=value.numel(),
                             nbytes=value.numel() * value.element_size(), sha256=hasher.hexdigest())
            components.append(component)
            return dict(tensor_component=len(components) - 1, sha256=component['sha256'])
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError('observation structure keys must be strings')
            return {key: visit(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, path + (index,)) for index, item in enumerate(value)]
        if value is None or isinstance(value, (str, int, bool)) or (schema == SCHEMA and type(value) is float):
            return value
        raise ValueError('unsupported value in native observation structure')

    structure = visit(payload, ())
    identity = dict(payload_schema=schema, components=components, structure=structure)
    return dict(schema=LEGACY_MANIFEST_SCHEMA if schema == LEGACY_SCHEMA else MANIFEST_SCHEMA, payload_schema=schema,
                payload_digest=hashlib.sha256(_canonical(identity)).hexdigest(),
                component_count=len(components), components=components, structure=structure)


def _output_paths(destination, label):
    if (not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', label)
            or label.endswith('.') or label.split('.')[0].upper() in
            {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('observation label must be a safe filename stem')
    directory = Path(destination).resolve()
    allowed = (TEST_ONLY.resolve(), TEST_ONLY22.resolve(), (TEST_ONLY.parent.parent/'part2_3/TEST_ONLY').resolve(), (TEST_ONLY.parent.parent/'part2_4/TEST_ONLY').resolve())
    if not any(directory != root and directory.is_relative_to(root) for root in allowed):
        raise PermissionError('native observations must stay within a RankingComparison Part2.1/Part2.2 TEST_ONLY subdirectory')
    paths = directory / (label + '.pt'), directory / (label + '.manifest.json')
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('native observation artifacts are immutable: ' + str(directory / label))
    return paths


def _file_sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def persist_native_batch(batch, destination, *, nid_key=None, eid_key=None, label='batch', context=None):
    """Publish a complete payload and JSON inventory inside this phase TEST_ONLY.

    Files use exclusive creation. The JSON manifest is written last and is the
    completion marker; an interrupted payload is retained, never overwritten.
    The caller remains responsible for existing runtime disk/budget guards.
    """
    payload_path, manifest_path = _output_paths(destination, label)
    payload = snapshot_native_batch(batch, nid_key=nid_key, eid_key=eid_key, context=context)
    manifest = native_batch_manifest(payload)
    from .runtime_budget import disk_guard
    disk_guard(payload_path,2*sum(row['nbytes'] for row in manifest['components'])+10_000_000)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest.update(status='COMPLETE', mode='TEST_ONLY', payload_path=str(payload_path),
                    payload_sha256=_file_sha(payload_path), payload_bytes=payload_path.stat().st_size)
    with manifest_path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return dict(payload_path=str(payload_path), manifest_path=str(manifest_path),
                payload_sha256=manifest['payload_sha256'], manifest_sha256=_file_sha(manifest_path),
                payload_digest=manifest['payload_digest'], component_count=manifest['component_count'])


__all__ = ['snapshot_native_batch', 'native_batch_manifest', 'persist_native_batch']
