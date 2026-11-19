"""Immutable columnar canonical inputs with bounded query gathers."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
GiB = 1024 ** 3
def _immutable(array: np.ndarray) -> np.ndarray:
    """Bytes backing prevents callers re-enabling NumPy WRITEABLE."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

def _integers(values: Any) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        values = list(values)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
            raise ValueError('IDs must be integers, excluding Boolean values')
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError('IDs must be a one-dimensional vector')
    if not array.size:
        return np.empty(0, dtype=np.int64)
    if array.ndim != 1 or array.dtype.kind not in 'iu':
        raise ValueError('IDs must be an integer vector; no lossy coercion')
    if array.min() < 0 or array.max() > np.iinfo(np.int64).max:
        raise ValueError('IDs must fit nonnegative int64 model input')
    return array.astype(np.int64, copy=False)

@dataclass(frozen=True, slots=True)
class OrderedIdMap:
    _ids: np.ndarray
    _positions: np.ndarray
    dense_zero_based: bool

    @classmethod
    def build(cls, ids: np.ndarray, *, expected_count: int) -> 'OrderedIdMap':
        ids = _integers(ids)
        if len(ids) != expected_count:
            raise ValueError('ID row count mismatch')
        order = np.argsort(ids, kind='stable')
        sorted_ids = ids[order]
        if len(ids) > 1 and np.any(sorted_ids[1:] <= sorted_ids[:-1]):
            raise ValueError('duplicate IDs violate uniqueness')
        dense = not len(ids) or (sorted_ids[0] == 0 and sorted_ids[-1] == len(ids) - 1)
        return cls(_immutable(sorted_ids), _immutable(order), bool(dense))

    def positions(self, ids: Any) -> np.ndarray:
        values = _integers(ids)
        positions = np.searchsorted(self._ids, values)
        if np.any(positions >= len(self._ids)):
            raise KeyError('unknown required ID')
        if np.any(self._ids[positions] != values):
            raise KeyError('unknown required ID')
        return self._positions[positions].copy()

    def ordered_ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def nbytes(self) -> int:
        return self._ids.nbytes + self._positions.nbytes

@dataclass(slots=True)
class BuildBudget:
    """Build-only accounting. No truncation/eviction at either hard boundary."""
    static_limit: int = 8 * GiB
    scratch_limit: int = 2 * GiB
    persistent: int = 0
    scratch_peak: int = 0
    categories: dict[str, int] = field(default_factory=dict)

    def retain(self, category: str, size: int) -> None:
        if size < 0 or self.persistent + size > self.static_limit:
            raise MemoryError('static immutable input hard budget exceeded')
        self.persistent += size
        self.categories[category] = self.categories.get(category, 0) + size

    def scratch(self, size: int) -> None:
        self.scratch_peak = max(self.scratch_peak, size)
        if size > self.scratch_limit:
            raise MemoryError('startup scratch hard budget exceeded')

def _pack_raw(frame: pl.DataFrame, budget: BuildBudget) -> bytes:
    # Native lossless representation: JSON property strings remain strings.
    # IPC compression affects storage only; it is not a semantic token transform.
    budget.scratch(frame.estimated_size() * 6 + 65536)
    table = frame.to_arrow()
    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(compression='zstd', use_threads=False)
    with ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    payload = sink.getvalue().to_pybytes()
    budget.retain('raw_native_blocks', len(payload))
    return payload

@dataclass(frozen=True, slots=True)
class RawRowStore:
    _blocks: tuple[bytes, ...]
    _offsets: np.ndarray
    _schema: tuple[tuple[str, pl.DataType], ...]

    @classmethod
    def build(cls, frames: Iterable[pl.DataFrame], *, expected_count: int,
              budget: BuildBudget) -> 'RawRowStore':
        blocks: list[bytes] = []
        offsets = [0]
        schema = None
        for frame in frames:
            if schema is None:
                schema = tuple(frame.schema.items())
            if tuple(frame.schema.items()) != schema:
                raise ValueError('raw block schema mismatch')
            offsets.append(offsets[-1] + frame.height)
            if offsets[-1] > expected_count:
                raise ValueError('raw row count mismatch')
            blocks.append(_pack_raw(frame, budget))
        if offsets[-1] != expected_count or schema is None:
            raise ValueError('raw row count/schema missing')
        offset_array = _immutable(np.array(offsets, dtype=np.int64))
        budget.retain('catalog_offsets', offset_array.nbytes)
        return cls(tuple(blocks), offset_array, schema)

    def gather(self, positions: Any) -> pl.DataFrame:
        positions = _integers(positions)
        if np.any(positions >= self._offsets[-1]):
            raise KeyError('raw row offset out of range')
        if not len(positions):
            return pl.DataFrame(schema=dict(self._schema))
        chunk_ids = np.searchsorted(self._offsets[1:], positions, side='right')
        order_name = '__input_order'
        while order_name in dict(self._schema):
            order_name += '_'
        parts = []
        for chunk in np.unique(chunk_ids):
            locations = np.flatnonzero(chunk_ids == chunk)
            table = ipc.open_stream(self._blocks[int(chunk)]).read_all()
            frame = pl.from_arrow(table.take(pa.array(positions[locations] - self._offsets[chunk])))
            parts.append(frame.with_columns(pl.Series(order_name, locations)))
        return pl.concat(parts).sort(order_name).drop(order_name)

@dataclass(frozen=True, slots=True)
class InputBinding:
    phase3_digest: str
    files: tuple
    helper_digest: str
    token_buckets: int
    def assert_matches(self, other):
        if self != other:
            raise ValueError('Wrong data/schema/catalog/encoding binding')

@dataclass(frozen=True, slots=True)
class StaticInputIndex:
    binding: InputBinding

    _node_map: OrderedIdMap

    _node_raw: RawRowStore

    _node_names: np.ndarray

    _node_name_order: np.ndarray

    _node_globals: np.ndarray

    _node_types: np.ndarray

    node_type_catalog: tuple[str, ...]

    edge_catalog: tuple[tuple[str, str, str], ...]

    _fact_map: OrderedIdMap

    _fact_raw: RawRowStore

    _fact_types: np.ndarray

    _fact_logic: np.ndarray

    _evidence_map: OrderedIdMap

    _evidence_raw: RawRowStore

    _evidence_ids: np.ndarray

    _record_facts: np.ndarray

    _fact_evidence_offsets: np.ndarray

    _fact_evidence_rows: np.ndarray

    _encoded: tuple[Any, ...]

    _encoded_offsets: np.ndarray

    _record_production_visible: np.ndarray

    _lookups: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    _audit: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def node_type_ids(self, global_ids: Any) -> np.ndarray:
        return self._node_types[self._node_map.positions(global_ids)].copy()

    @classmethod
    def _from_streams(cls, *, nodes: pl.DataFrame, fact_frames: Iterable[pl.DataFrame],
                      evidence_frames: Iterable[pl.DataFrame], fact_count: int, evidence_count: int,
                      edge_catalog: tuple[tuple[str, str, str], ...], binding: InputBinding,
                      budget: BuildBudget, progress=None) -> 'StaticInputIndex':
        from .input_encoding import encode_records
        started = time.perf_counter()
        if fact_count < 0 or evidence_count < 0:
            raise ValueError('negative expected row count')
        if not edge_catalog or len(edge_catalog) > 65536 or len(set(edge_catalog)) != len(edge_catalog):
            raise ValueError('invalid/duplicate edge type catalog')
        node_globals = _integers(nodes['global_node_id'].to_numpy())
        node_map = OrderedIdMap.build(node_globals, expected_count=nodes.height)
        names = np.asarray(nodes['node_id'].to_list())
        if names.dtype.kind != 'U' or nodes['node_id'].null_count():
            raise ValueError('missing/nonstring node identifier')
        name_order = np.argsort(names, kind='stable')
        sorted_names = names[name_order]
        if len(names) > 1 and np.any(sorted_names[1:] == sorted_names[:-1]):
            raise ValueError('duplicate canonical node identifiers')
        if nodes['node_type'].null_count():
            raise ValueError('missing node type')
        node_types = tuple(sorted(nodes['node_type'].unique().to_list()))
        if not set(x for s, _, t in edge_catalog for x in (s, t)) <= set(node_types):
            raise ValueError('unknown node type in edge catalog')
        types = np.array([node_types.index(x) for x in nodes['node_type']], dtype=np.int64)
        node_raw = RawRowStore.build(iter((nodes,)), expected_count=nodes.height, budget=budget)
        budget.retain('node_indexes', node_map.nbytes + names.nbytes + name_order.nbytes + node_globals.nbytes + types.nbytes)
        names, name_order, node_globals, types = map(_immutable, (sorted_names, name_order, node_globals, types))
        budget.scratch(fact_count * 48 + evidence_count * 8)
        # Full-size native buffers, never full-size row object lists.
        fact_ids = np.empty(fact_count, dtype=np.int64)
        fact_types = np.empty(fact_count, dtype=np.uint16)
        fact_logic = np.empty(fact_count, dtype=np.bool_)
        fact_components = np.empty(fact_count, dtype=np.uint16)
        component_ids: dict[str, int] = {}
        fact_cursor = 0
        type_lookup = {key: i for i, key in enumerate(edge_catalog)}

        def consume_facts():
            nonlocal fact_cursor
            for frame in fact_frames:
                end = fact_cursor + frame.height
                if end > fact_count:
                    raise ValueError('fact row count exceeds binding')
                fact_ids[fact_cursor:end] = _integers(frame['canonical_fact_id'].to_numpy())
                components = frame['adapter_component'].to_list()
                for component in set(components):
                    if not isinstance(component, str) or not component:
                        raise ValueError('unknown required component value')
                    if component not in component_ids:
                        if len(component_ids) >= 65536:
                            raise ValueError('component catalog cannot be represented losslessly')
                        component_ids[component] = len(component_ids)
                fact_components[fact_cursor:end] = [component_ids[v] for v in components]
                source_positions = node_map.positions(frame['source_global_id'].to_numpy())
                target_positions = node_map.positions(frame['target_global_id'].to_numpy())
                keys = list(zip(frame['source_node_type'], frame['relation'], frame['target_node_type'], strict=True))
                try:
                    fact_types[fact_cursor:end] = [type_lookup[key] for key in keys]
                except KeyError as exc:
                    raise ValueError('unknown required edge catalog value') from exc
                if any(node_types[types[p]] != key[0] or node_types[types[q]] != key[2]
                       for p, q, key in zip(source_positions, target_positions, keys, strict=True)):
                    raise ValueError('fact endpoint node type mismatch')
                fact_logic[fact_cursor:end] = [s == 'ANA' and t == 'GEN' and r in
                    {'EXPRESSION_PRESENT', 'EXPRESSION_ABSENT'} for s, r, t in keys]
                fact_cursor = end
                if progress is not None:
                    progress('facts', fact_cursor, fact_count, budget.persistent)
                yield frame

        fact_raw = RawRowStore.build(consume_facts(), expected_count=fact_count, budget=budget)
        fact_map = OrderedIdMap.build(fact_ids, expected_count=fact_count)
        budget.retain('fact_indexes', fact_map.nbytes + fact_types.nbytes + fact_logic.nbytes)
        fact_types, fact_logic = _immutable(fact_types), _immutable(fact_logic)
        del fact_ids
        metadata_seconds = time.perf_counter() - started
        evidence_ids = np.empty(evidence_count, dtype=np.int64)
        record_facts = np.empty(evidence_count, dtype=np.int64)
        production_visible = np.empty(evidence_count, dtype=np.bool_)
        encoded = []
        encoded_offsets = [0]
        encoding_seconds = 0.0
        evidence_started = time.perf_counter()

        def consume_evidence():
            nonlocal encoding_seconds
            for frame in evidence_frames:
                start = encoded_offsets[-1]
                end = start + frame.height
                if end > evidence_count:
                    raise ValueError('evidence row count exceeds binding')
                evidence_ids[start:end] = _integers(frame['evidence_record_id'].to_numpy())
                record_facts[start:end] = fact_map.positions(frame['canonical_fact_id'].to_numpy())
                try:
                    record_components = np.array([component_ids[v] for v in frame['adapter_component']], dtype=np.uint16)
                except KeyError as exc:
                    raise ValueError('unknown required evidence component') from exc
                production_visible[start:end] = record_components == fact_components[record_facts[start:end]]
                before_encoding = time.perf_counter()
                block = encode_records(frame, token_buckets=binding.token_buckets)
                budget.scratch(block.build_scratch_upper_bound_bytes + evidence_count * 17 + fact_components.nbytes)
                encoding_seconds += time.perf_counter() - before_encoding
                token_offsets_bytes = block.property_offsets.nbytes + block.evidence_offsets.nbytes + block.provenance_offsets.nbytes
                budget.retain('preencoded_fields', block.nbytes - token_offsets_bytes)
                budget.retain('catalog_offsets', token_offsets_bytes)
                encoded.append(block)
                encoded_offsets.append(end)
                if progress is not None:
                    progress('evidence', end, evidence_count, budget.persistent)
                yield frame

        evidence_raw = RawRowStore.build(consume_evidence(), expected_count=evidence_count, budget=budget)
        del fact_components
        streaming_seconds = time.perf_counter() - evidence_started - encoding_seconds
        final_started = time.perf_counter()
        budget.scratch(evidence_count * 64 + fact_count * 16)
        evidence_map = OrderedIdMap.build(evidence_ids, expected_count=evidence_count)
        # Stable physical positions within each fact; query gather later sorts evidence IDs globally.
        record_order = np.argsort(record_facts, kind='stable')
        counts = np.bincount(record_facts, minlength=fact_count)
        offsets = np.empty(fact_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        if offsets[-1] != evidence_count:
            raise ValueError('corrupt fact-to-evidence offsets')
        del counts
        budget.retain('evidence_indexes', evidence_map.nbytes + evidence_ids.nbytes + record_facts.nbytes + record_order.nbytes + production_visible.nbytes)
        budget.retain('catalog_offsets', offsets.nbytes + (len(encoded_offsets) * 8))
        evidence_ids, record_facts, record_order, offsets = map(_immutable, (evidence_ids, record_facts, record_order, offsets))
        audit = {'metadata_index_seconds': metadata_seconds, 'evidence_streaming_seconds': streaming_seconds,
                 'property_provenance_encoding_seconds': encoding_seconds,
                 'final_validation_freeze_seconds': time.perf_counter() - final_started,
                 'core_build_seconds': time.perf_counter() - started,
                 'node_count': nodes.height, 'fact_count': fact_count, 'evidence_count': evidence_count,
                 'dense_node_ids_verified': node_map.dense_zero_based,
                 'dense_fact_ids_verified': fact_map.dense_zero_based,
                 'dense_evidence_ids_verified': evidence_map.dense_zero_based,
                 'id_lookup_mode': 'explicit sorted-ID mapping, including when dense',
                 'cross_component_records_retained_outside_production_selection': int(np.count_nonzero(~production_visible)),
                 'production_evidence_selection': 'same-component canonical fact filter exactly as frozen _load_fact_evidence',
                 'all_uniqueness_row_counts_verified': True}
        return cls(binding, node_map, node_raw, names, name_order, node_globals, types, node_types,
                   edge_catalog, fact_map, fact_raw, fact_types, fact_logic, evidence_map, evidence_raw,
                   evidence_ids, record_facts, offsets, record_order, tuple(encoded),
                   _immutable(np.array(encoded_offsets, dtype=np.int64)), _immutable(production_visible), _audit=MappingProxyType(audit))

    def node_global_ids(self, canonical_ids: Iterable[str]) -> np.ndarray:
        canonical_ids = list(canonical_ids)
        if any(not isinstance(value, str) for value in canonical_ids):
            raise ValueError('canonical node IDs must be strings')
        values = np.asarray(canonical_ids)
        if not values.size:
            return np.empty(0, dtype=np.int64)
        if values.dtype.kind != 'U':
            raise ValueError('canonical node IDs must be strings')
        positions = np.searchsorted(self._node_names, values)
        if np.any(positions >= len(self._node_names)) or np.any(self._node_names[positions] != values):
            raise KeyError('unknown required canonical node ID')
        return self._node_globals[self._node_name_order[positions]].copy()

    def node_metadata(self, global_ids: Any) -> pl.DataFrame:
        return self._node_raw.gather(self._node_map.positions(global_ids))

    def fact_metadata(self, fact_ids: Any) -> pl.DataFrame:
        ids = np.unique(_integers(fact_ids))
        return self._fact_raw.gather(self._fact_map.positions(ids))

    def evidence_rows(self, evidence_ids: Any) -> pl.DataFrame:
        return self._evidence_raw.gather(self._evidence_map.positions(evidence_ids))

    def _selection(self, fact_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.unique(_integers(fact_ids))
        facts = self._fact_map.positions(ids)
        parts = [self._fact_evidence_rows[self._fact_evidence_offsets[p]:self._fact_evidence_offsets[p + 1]] for p in facts]
        records = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if records.size:
            records = records[self._record_production_visible[records]]
            records = records[np.argsort(self._evidence_ids[records], kind='stable')]
        return ids, facts, records

    def evidence_for_facts(self, fact_ids: Any) -> pl.DataFrame:
        return self._evidence_raw.gather(self._selection(fact_ids)[2])

    def evidence_static(self, evidence_ids: Any) -> dict[str, torch.Tensor]:
        """Direct raw-field encoding lookup, including cross-component records.

        Preserves requested IDs/order/duplicates. Query selection is not applied.
        """
        return self._static_rows(self._evidence_map.positions(evidence_ids))

    def _static_rows(self, records: np.ndarray) -> dict[str, torch.Tensor]:
        # Per-block static gathers are reordered by the caller's global evidence order.
        chunk_ids = np.searchsorted(self._encoded_offsets[1:], records, side='right')
        pieces = []
        for chunk in np.unique(chunk_ids):
            locations = np.flatnonzero(chunk_ids == chunk)
            values = self._encoded[int(chunk)].gather(records[locations] - self._encoded_offsets[chunk])
            pieces.append((locations, values))
        fixed = ('numeric_values', 'numeric_mask', 'boolean_values', 'boolean_mask',
                 'overlap_category_id', 'support_observed', 'counter_observed')
        from evi_hgt.data.encoding_fields import NUMERIC_FIELDS, BOOLEAN_FIELDS
        widths = {'numeric_values': len(NUMERIC_FIELDS), 'numeric_mask': len(NUMERIC_FIELDS),
                  'boolean_values': len(BOOLEAN_FIELDS), 'boolean_mask': len(BOOLEAN_FIELDS)}
        result = {}
        for name in fixed:
            dtype = torch.float32 if name == 'numeric_values' else (torch.long if name == 'overlap_category_id' else torch.bool)
            shape = (len(records), widths[name]) if name in widths else (len(records),)
            result[name] = torch.empty(shape, dtype=dtype)
            for locations, values in pieces:
                result[name][torch.tensor(locations, dtype=torch.long)] = values[name]
        for kind in ('property', 'evidence', 'provenance'):
            index_name = 'provenance_record_index' if kind == 'provenance' else kind + '_token_record_index'
            names = [kind + '_token_ids']
            if kind == 'provenance':
                names += ['provenance_role_ids', 'provenance_kind_ids']
            record_indices = []
            data = {name: [] for name in names}
            for locations, values in pieces:
                record_indices.append(torch.tensor(locations, dtype=torch.long)[values[index_name]])
                for name in names:
                    data[name].append(values[name])
            indices = torch.cat(record_indices) if record_indices else torch.empty(0, dtype=torch.long)
            order = torch.argsort(indices, stable=True)
            result[index_name] = indices[order]
            for name in names:
                result[name] = torch.cat(data[name])[order] if data[name] else torch.empty(0, dtype=torch.long)
        return result

    def gather(self, fact_ids: Any, *, binding: InputBinding | None = None) -> dict[str, Any]:
        """Restore original model dtypes and order into private bounded buffers."""
        from evi_hgt.models.input_contract import PackedEvidenceBatch
        if binding is not None:
            self.binding.assert_matches(binding)
        ids, facts, records = self._selection(fact_ids)
        result = self._static_rows(records)
        local_by_physical = {int(p): i for i, p in enumerate(facts)}
        local_records = torch.tensor([local_by_physical[int(p)] for p in self._record_facts[records]], dtype=torch.long)
        fact_types = torch.tensor(self._fact_types[facts].astype(np.int64), dtype=torch.long)
        result.update(num_facts=len(facts), fact_edge_type_id=fact_types,
                      fact_graph_id=torch.zeros(len(facts), dtype=torch.long), record_fact_index=local_records,
                      record_edge_type_id=fact_types[local_records],
                      fact_support_available=torch.tensor(self._fact_logic[facts], dtype=torch.bool),
                      fact_counter_available=torch.tensor(self._fact_logic[facts], dtype=torch.bool))
        return {'fact_ids': ids.copy(), 'evidence_ids': self._evidence_ids[records].copy(),
                'evidence': PackedEvidenceBatch(**result)}
