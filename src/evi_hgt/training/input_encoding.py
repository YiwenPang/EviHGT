"""Lossless static evidence encoding, independent of training execution.

The builder consumes bounded record frames. Only already existing raw input
features are encoded; query membership, graph extraction and model execution
are deliberately outside this module. Arrays have immutable ``bytes`` backing,
and every gather returns fresh tensors so callers cannot alter later lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import json
import math
import sys
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from evi_hgt.data.encoding_fields import (
    BOOLEAN_FIELDS,
    EVIDENCE_FIELD_MARKERS,
    EXPECTED_TOKEN_FIELDS,
    NUMERIC_FIELDS,
    OVERLAP_IDS,
    PROVENANCE_KIND_IDS,
    PROVENANCE_ROLE_IDS,
    RECORD_IDENTIFIER_FIELDS,
    _as_values,
    _flatten_properties,
    _provenance_kind,
    _stable_bucket,
)


ENCODING_VERSION = "phase3-static-raw-v1"
TOKEN_NAMESPACE_VERSION = "phase4-token-v1"
REQUIRED_COLUMNS = (
    "properties", "adapter_component", "raw_direct_sources",
    "raw_indirect_sources", "source_row_number", "source_component_path",
    "overlap_category", "edge_support_observed", "edge_counter_evidence_observed",
)


class _FrozenArray(np.ndarray):
    """Protect public array descriptors as well as the immutable data buffer.

    NumPy's WRITEABLE flag protects elements but permits in-place shape/dtype
    changes. Such descriptor changes would otherwise corrupt later gathers.
    Ordinary slicing/reshaping into a separate view remains available.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"shape", "dtype", "strides", "data"}:
            raise AttributeError("immutable encoded array metadata cannot be changed")
        super().__setattr__(name, value)

    def resize(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("immutable encoded array cannot be resized")


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape).view(_FrozenArray)


def _uint_dtype(maximum: int) -> np.dtype:
    if maximum < 0 or maximum > np.iinfo(np.uint64).max:
        raise ValueError("integer range cannot be represented losslessly")
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _positions(values: Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        values = list(values)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in values):
            raise TypeError("record positions must be integers, excluding Boolean values")
    positions = np.asarray(values)
    if positions.ndim != 1:
        raise ValueError("record positions must be one-dimensional")
    if positions.size == 0:
        return np.empty(0, dtype=np.int64)
    if positions.dtype.kind not in "iu":
        raise TypeError("record positions must be integers")
    if np.any(positions < 0) or np.any(positions >= count):
        raise IndexError("record position is outside the immutable block")
    return positions.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class EncodedRecords:
    """One immutable ordered block; offsets preserve token order and repeats."""

    numeric_values: np.ndarray
    numeric_mask: np.ndarray
    boolean_values: np.ndarray
    boolean_mask: np.ndarray
    property_token_ids: np.ndarray
    property_offsets: np.ndarray
    evidence_token_ids: np.ndarray
    evidence_offsets: np.ndarray
    provenance_token_ids: np.ndarray
    provenance_role_ids: np.ndarray
    provenance_kind_ids: np.ndarray
    provenance_offsets: np.ndarray
    overlap_category_id: np.ndarray
    support_observed: np.ndarray
    counter_observed: np.ndarray
    token_buckets: int
    build_scratch_upper_bound_bytes: int = 0

    def __post_init__(self) -> None:
        _validate_buckets(self.token_buckets)
        count = self.numeric_values.shape[0]
        shapes = {
            "numeric_values": ((count, len(NUMERIC_FIELDS)), np.dtype(np.float32)),
            "numeric_mask": ((count, len(NUMERIC_FIELDS)), np.dtype(np.bool_)),
            "boolean_values": ((count, len(BOOLEAN_FIELDS)), np.dtype(np.bool_)),
            "boolean_mask": ((count, len(BOOLEAN_FIELDS)), np.dtype(np.bool_)),
            "support_observed": ((count,), np.dtype(np.bool_)),
            "counter_observed": ((count,), np.dtype(np.bool_)),
        }
        for name, (shape, dtype) in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"corrupt encoded {name}: shape/dtype mismatch")
        for prefix in ("property", "evidence", "provenance"):
            tokens = getattr(self, f"{prefix}_token_ids")
            offsets = getattr(self, f"{prefix}_offsets")
            if tokens.ndim != 1 or tokens.dtype.kind != "u":
                raise ValueError(f"corrupt {prefix} token array")
            if tokens.size and np.max(tokens) >= self.token_buckets:
                raise ValueError(f"{prefix} token outside bound namespace")
            if offsets.shape != (count + 1,) or offsets.dtype.kind != "u":
                raise ValueError(f"corrupt {prefix} offsets")
            if offsets[0] != 0 or offsets[-1] != tokens.size or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"corrupt {prefix} token span")
        for name, count_expected, catalog in (
            ("provenance_role_ids", self.provenance_token_ids.size, PROVENANCE_ROLE_IDS),
            ("provenance_kind_ids", self.provenance_token_ids.size, PROVENANCE_KIND_IDS),
            ("overlap_category_id", count, OVERLAP_IDS),
        ):
            value = getattr(self, name)
            if value.shape != (count_expected,) or value.dtype.kind != "u" or not np.isin(value, list(catalog.values())).all():
                raise ValueError(f"unknown or corrupt {name}")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _immutable(value))

    @property
    def record_count(self) -> int:
        return self.numeric_values.shape[0]

    @property
    def nbytes(self) -> int:
        """Actual immutable array payload bytes (no estimate)."""
        return sum(getattr(self, field.name).nbytes for field in fields(self)
                   if isinstance(getattr(self, field.name), np.ndarray))

    def byte_accounting(self) -> dict[str, int | bool]:
        """Actual payload and Python holder overhead, counted without aliases."""
        seen: set[int] = set()
        holder = (sys.getsizeof(self) + sys.getsizeof(self.token_buckets)
                  + sys.getsizeof(self.build_scratch_upper_bound_bytes))
        for field in fields(self):
            value = getattr(self, field.name)
            while isinstance(value, (np.ndarray, bytes)) and id(value) not in seen:
                seen.add(id(value))
                holder += sys.getsizeof(value) - (len(value) if isinstance(value, bytes) else 0)
                value = value.base if isinstance(value, np.ndarray) else None
        offsets = self.property_offsets.nbytes + self.evidence_offsets.nbytes + self.provenance_offsets.nbytes
        return {"array_payload_bytes": self.nbytes, "offset_bytes": offsets,
                "python_holder_bytes": holder, "total_owned_bytes": self.nbytes + holder,
                "build_scratch_upper_bound_bytes": self.build_scratch_upper_bound_bytes,
                "build_scratch_is_measured_peak": False}

    def gather(self, row_positions: Sequence[int] | np.ndarray) -> dict[str, torch.Tensor]:
        """Return production dtypes in requested order, retaining duplicates."""
        positions = _positions(row_positions, self.record_count)
        result: dict[str, torch.Tensor] = {}
        for name, dtype in (
            ("numeric_values", torch.float32), ("numeric_mask", torch.bool),
            ("boolean_values", torch.bool), ("boolean_mask", torch.bool),
            ("overlap_category_id", torch.long), ("support_observed", torch.bool),
            ("counter_observed", torch.bool),
        ):
            result[name] = torch.tensor(getattr(self, name)[positions], dtype=dtype)
        for prefix in ("property", "evidence", "provenance"):
            offsets = getattr(self, f"{prefix}_offsets")
            starts = offsets[positions].astype(np.int64)
            lengths = offsets[positions + 1].astype(np.int64) - starts
            record_indices = np.repeat(np.arange(positions.size, dtype=np.int64), lengths)
            total = int(lengths.sum())
            if total:
                output_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
                token_positions = np.repeat(starts, lengths) + np.arange(total, dtype=np.int64) - output_starts
            else:
                token_positions = np.empty(0, dtype=np.int64)
            names = [f"{prefix}_token_ids"]
            if prefix == "provenance":
                names += ["provenance_role_ids", "provenance_kind_ids"]
            for name in names:
                result[name] = torch.tensor(getattr(self, name)[token_positions].astype(np.int64), dtype=torch.long)
            record_name = "provenance_record_index" if prefix == "provenance" else f"{prefix}_token_record_index"
            result[record_name] = torch.tensor(record_indices, dtype=torch.long)
        return result


def _validate_buckets(token_buckets: int) -> None:
    if isinstance(token_buckets, (bool, np.bool_)) or not isinstance(token_buckets, (int, np.integer)):
        raise TypeError("token bucket count must be an integer")
    if not 1 <= token_buckets <= 2**63:
        raise ValueError("token IDs must be losslessly representable as torch.long")


def encode_records(frame: pl.DataFrame, *, token_buckets: int) -> EncodedRecords:
    """Pre-encode a bounded frame using the existing production raw semantics.

    The local token memo tables have fixed bounds and are discarded on return.
    JSON decoding, traversal, classification and hashing occur only here.
    """
    _validate_buckets(token_buckets)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"required evidence columns are missing: {sorted(missing)}")
    count = frame.height
    numeric = np.zeros((count, len(NUMERIC_FIELDS)), dtype=np.float32)
    numeric_mask = np.zeros(numeric.shape, dtype=np.bool_)
    boolean = np.zeros((count, len(BOOLEAN_FIELDS)), dtype=np.bool_)
    boolean_mask = np.zeros(boolean.shape, dtype=np.bool_)
    overlap = np.empty(count, dtype=np.uint8)
    support = np.empty(count, dtype=np.bool_)
    counter = np.empty(count, dtype=np.bool_)
    property_tokens: list[int] = []
    evidence_tokens: list[int] = []
    provenance_tokens: list[int] = []
    provenance_roles: list[int] = []
    provenance_kinds: list[int] = []
    property_offsets = [0]
    evidence_offsets = [0]
    provenance_offsets = [0]
    largest_token_key_bytes = 0
    largest_kind_key_bytes = 0

    @lru_cache(maxsize=65_536)
    def token(namespace: str, text: str) -> int:
        nonlocal largest_token_key_bytes
        largest_token_key_bytes = max(largest_token_key_bytes, sys.getsizeof(namespace) + sys.getsizeof(text))
        return _stable_bucket(namespace, text, token_buckets)

    @lru_cache(maxsize=16_384)
    def kind(text: str) -> int:
        nonlocal largest_kind_key_bytes
        largest_kind_key_bytes = max(largest_kind_key_bytes, sys.getsizeof(text))
        return _provenance_kind(text)

    def provenance(role: str, value: Any) -> None:
        text = str(value)
        provenance_tokens.append(token(f"provenance:{role}", text))
        provenance_roles.append(PROVENANCE_ROLE_IDS[role])
        provenance_kinds.append(kind(text))

    for index, row in enumerate(frame.select(REQUIRED_COLUMNS).iter_rows(named=True)):
        raw = row["properties"]
        properties = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(properties, dict):
            raise RuntimeError("formal evidence properties must decode to an object")
        ordered_fields = tuple(_flatten_properties(properties))
        flattened = dict(ordered_fields)
        for column, name in enumerate(NUMERIC_FIELDS):
            value = flattened.get(name)
            present = isinstance(value, (int, float)) and not isinstance(value, bool)
            present = bool(present and math.isfinite(float(value)))
            if present:
                # torch.tensor(..., float32) also overflows finite out-of-range
                # values to infinity. Nonfinite raw values remain absent.
                with np.errstate(over="ignore"):
                    numeric[index, column] = float(value)
                numeric_mask[index, column] = True
        for column, name in enumerate(BOOLEAN_FIELDS):
            value = flattened.get(name)
            if isinstance(value, bool):
                boolean[index, column] = value
                boolean_mask[index, column] = True
        component = str(row["adapter_component"])
        observed_paths: set[str] = set()
        for path, value in ordered_fields:
            if path.startswith("sources."):
                continue
            leaf = path.rsplit(".", 1)[-1]
            observed_paths.add(leaf)
            if leaf in NUMERIC_FIELDS or leaf in BOOLEAN_FIELDS:
                continue
            if leaf in RECORD_IDENTIFIER_FIELDS:
                for item in _as_values(value):
                    provenance("RECORD_IDENTIFIER", item)
                continue
            is_evidence = any(marker in leaf.lower() for marker in EVIDENCE_FIELD_MARKERS)
            output = evidence_tokens if is_evidence else property_tokens
            namespace = "evidence" if is_evidence else "property"
            for item in _as_values(value):
                output.append(token(f"{namespace}:{component}:{path}", str(item)))
                if "reference" in leaf.lower():
                    provenance("REFERENCE", item)
        for expected in EXPECTED_TOKEN_FIELDS.get(component, ()):
            if expected not in observed_paths or not _as_values(flattened.get(expected)):
                marker = token(f"missing:{component}:{expected}", "<MISSING>")
                output = evidence_tokens if any(part in expected for part in EVIDENCE_FIELD_MARKERS) else property_tokens
                output.append(marker)
        for value in row["raw_direct_sources"] or []:
            provenance("DIRECT", value)
        for value in row["raw_indirect_sources"] or []:
            provenance("INDIRECT", value)
        provenance("RECORD_IDENTIFIER", f"source_row:{row['source_row_number']}")
        provenance("SOURCE_RELEASE", "OptimusKG:2.0")
        provenance("SOURCE_RELEASE", row["source_component_path"])
        overlap[index] = OVERLAP_IDS[row["overlap_category"]]
        support[index] = bool(row["edge_support_observed"])
        counter[index] = bool(row["edge_counter_evidence_observed"])
        property_offsets.append(len(property_tokens))
        evidence_offsets.append(len(evidence_tokens))
        provenance_offsets.append(len(provenance_tokens))

    token_dtype = _uint_dtype(int(token_buckets) - 1)
    def offsets(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.uint32 if values[-1] <= np.iinfo(np.uint32).max else np.uint64)
    # Count every list entry, including integer aliases, for the allocation bound.
    # The raw-frame estimate covers decoded strings, containers and Polars row
    # conversion. Memo estimates include keys, values, slots and LRU links.
    # The process audit records peak RSS separately.
    lists = (property_tokens, evidence_tokens, provenance_tokens, provenance_roles,
             provenance_kinds, property_offsets, evidence_offsets, provenance_offsets)
    list_bytes = sum(sys.getsizeof(values) + sum(map(sys.getsizeof, values)) for values in lists)
    array_bytes = (numeric.nbytes + numeric_mask.nbytes + boolean.nbytes + boolean_mask.nbytes
                   + overlap.nbytes + support.nbytes + counter.nbytes
                   + (len(property_tokens) + len(evidence_tokens) + len(provenance_tokens)) * token_dtype.itemsize
                   + len(provenance_roles) + len(provenance_kinds)
                   + sum(len(values) * (4 if values[-1] <= np.iinfo(np.uint32).max else 8)
                         for values in (property_offsets, evidence_offsets, provenance_offsets)))
    scratch_bound = (
        128 * frame.estimated_size() + 4096 * count + list_bytes + 3 * array_bytes
        + token.cache_info().currsize * (largest_token_key_bytes + 512)
        + kind.cache_info().currsize * (largest_kind_key_bytes + 256)
    )
    return EncodedRecords(
        numeric_values=numeric, numeric_mask=numeric_mask,
        boolean_values=boolean, boolean_mask=boolean_mask,
        property_token_ids=np.asarray(property_tokens, dtype=token_dtype),
        property_offsets=offsets(property_offsets),
        evidence_token_ids=np.asarray(evidence_tokens, dtype=token_dtype),
        evidence_offsets=offsets(evidence_offsets),
        provenance_token_ids=np.asarray(provenance_tokens, dtype=token_dtype),
        provenance_role_ids=np.asarray(provenance_roles, dtype=np.uint8),
        provenance_kind_ids=np.asarray(provenance_kinds, dtype=np.uint8),
        provenance_offsets=offsets(provenance_offsets),
        overlap_category_id=overlap, support_observed=support, counter_observed=counter,
        token_buckets=int(token_buckets),
        build_scratch_upper_bound_bytes=int(scratch_bound),
    )
