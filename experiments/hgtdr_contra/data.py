"""Load the frozen R0 folds and build inputs for HGTDR contraindication ranking."""

from dataclasses import dataclass
import json
from pathlib import Path

from evi_hgt.phase85_v2.common import digest as historical_digest
from evi_hgt_phase98.common import ROOT, digest, read, sha
from evi_hgt_phase98.data import TrainingCell, TrainingInputs, field, objects_in_array, reserved


CELLS = tuple(f'R0F{fold}' for fold in range(5))


@dataclass(frozen=True)
class ContraFold:
    data: TrainingCell
    native: object
    evaluation_rows: tuple[dict, ...]
    binding: dict


def _development_rows(data):
    direct = data.original_corpus_root.parent / 'cell_manifest.json'
    if direct.is_file():
        return tuple([json.loads(raw) for raw in objects_in_array(direct, name)]
                     for name in ('validation', 'auxiliary_validation'))

    entry = next(x for x in data.shared.keys['selected_repeat_bindings']
                 if x['original_repeat_index'] == data.repeat_index)
    path = ROOT / entry['fold_assignment_path']
    if sha(path) != entry['fold_assignment_sha256']:
        raise ValueError('Frozen assignment changed')
    fold = data.cell['fold']
    key = historical_digest([data.cell['repeat_key'], fold, 'inner'])
    validation, eligible_aux_ids = [], set()
    for raw in objects_in_array(path, 'assignments'):
        if field(raw, 'fold', True) == fold:
            continue
        pair = field(raw, 'drug_id'), field(raw, 'disease_id')
        if reserved(pair, key + 'validation', .2):
            validation.append(dict(drug_id=pair[0], disease_id=pair[1], role=field(raw, 'role')))
        elif not reserved(pair, key + 'probe', .05):
            eligible_aux_ids.add(pair)

    validation.sort(key=lambda row: (row['drug_id'], row['disease_id']))
    excluded = data.shared.panel_ids_for_exclusion - eligible_aux_ids
    contra = [dict(drug_id=drug, disease_id=disease, raw_relation='CONTRAINDICATION')
              for drug, disease in sorted(data.shared.aux_pairs - excluded)
              if reserved((drug, disease), key + 'aux_validation', .2)]
    return validation, contra


def project_contra_candidates(data):
    """Join the frozen contraindication positives with the shared unlabeled pairs."""
    if data.cell_id not in CELLS or data.repeat_index != 0:
        raise ValueError('HGTDR-contra development is limited to R0')
    validation, contra = _development_rows(data)
    digests = data.contract.bindings['membership_digests']
    for key, rows in (('validation', validation), ('auxiliary_validation', contra)):
        if historical_digest(rows) != digests[key]:
            raise ValueError(f'Frozen {key} membership changed')
    if any(row['role'] not in ('P', 'U') for row in validation):
        raise ValueError('Invalid inner validation role')
    if any(row['raw_relation'] != 'CONTRAINDICATION' for row in contra):
        raise ValueError('Contra evaluation positives require confirmed contraindication')
    rows = [dict(row, role='P') for row in contra]
    rows.extend(dict(row) for row in validation if row['role'] == 'U')
    rows.sort(key=lambda row: (row['drug_id'], row['disease_id']))
    pairs = {(row['drug_id'], row['disease_id']) for row in rows}
    if not contra or len(rows) == len(contra) or len(rows) != len(pairs):
        raise ValueError('Contra panel must contain distinct P and U candidates')
    if pairs & data.frozen:
        raise ValueError('Contra panel contains a protected pair')
    if not pairs <= data.blocked:
        raise ValueError('Contra panel contains a pair outside the fold mask')
    return tuple(rows)


def _native_cache(cell):
    if cell == 'R0F0':
        return (ROOT / 'outputs/phase9/formal/PHASE9_G3_75_V1/r0_f0/M1/native_graph',
                'FORMAL_native_graph.json', 'FORMAL_DATA_READY')
    return (ROOT / 'data/cache/phase98/native/HGTDR' / cell,
            'PHASE98_native_graph.json', 'PHASE98_NATIVE_DATA_READY')


def _cache_artifact(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Bound native graph artifact is outside its cache')
    return path


def _load_native_graph(data, corpus):
    import numpy as np
    import polars as pl
    from evi_hgt.competition.data import UnlabeledPolicy, normalize_nodes
    from evi_hgt.phase9 import hgtdr_adapter as adapter
    from evi_hgt.phase9.identity import G1_SHA256, digest as graph_digest, file_sha
    from evi_hgt_phase98.native_runtime import assert_native_global_fact_mask

    root, receipt_name, status = _native_cache(data.cell_id)
    receipt_path = root / receipt_name
    receipt = read(receipt_path)
    if receipt['status'] != status:
        raise ValueError('Original native graph is not certified ready')
    assert_native_global_fact_mask(corpus)
    facts = [Path(row['path']) for row in corpus['fact_files']]
    certified = {**corpus['inputs'], **corpus['outputs']}
    source_hashes = {str(path): file_sha(path) for path in facts}
    if any(certified.get(path) != value for path, value in source_hashes.items()):
        raise ValueError('Native cache source is outside the certified fold corpus')
    nodes = normalize_nodes(pl.read_parquet(ROOT / 'data/processed/phase3/v1/nodes/node_index.parquet'))
    features, available, counts, feature_binding = adapter._read_features(nodes)
    expected = dict(g1_sha256=G1_SHA256, method='M1', cell=data.bindings, corpus_digest=corpus['digest'],
                    feature_binding=feature_binding, source_hashes=source_hashes,
                    native_sources=adapter.verify_native_identity(), support_digest=graph_digest(sorted(data.support)),
                    blocked_digest=graph_digest(sorted(data.blocked)), fixed_config=adapter.fixed_config(data.training_seed),
                    graph_policy='NATIVE_ALL_CANONICAL_FORWARD_PLUS_COMPUTED_INVERSE_CURRENT_FOLD_LEGAL_FACTS',
                    U_is_confirmed_negative=False)
    if any(receipt['bindings'].get(key) != value for key, value in expected.items()):
        raise ValueError('Cached HGTDR graph does not match the certified current fold')
    artifacts = {relative: _cache_artifact(root, relative) for relative in receipt['artifacts']}
    for relative, path in artifacts.items():
        if sha(path) != receipt['artifacts'][relative]:
            raise ValueError(f'Cached HGTDR graph artifact changed: {path}')
    relations = pl.read_parquet(artifacts['relations.parquet']).to_dicts()
    arrays = [np.load(artifacts[row['edge_index_path']], mmap_mode='r', allow_pickle=False)
              for row in relations]
    if any(row['fact_ids_path'] not in artifacts for row in relations):
        raise ValueError('Relation fact identity was not included in the native cache receipt')

    native = adapter.HGTDRFoldData()
    native.training_seed, native.bindings, native.cell_source = data.training_seed, dict(data.bindings), data
    native.features, native.available, native.counts, native.feature_binding = features, available, counts, feature_binding
    native.lookup = {kind: dict(nodes.filter(pl.col('human_node_type') == kind)
                                .select('node_id', 'type_local_id').iter_rows()) for kind in ('drug', 'disease')}
    positive_rows = data.cell['auxiliary_training']
    positive_pairs = sorted(data.unsuitability)
    if (not positive_pairs or set(positive_pairs) & data.blocked
            or len(positive_rows) != len(positive_pairs)
            or {r['raw_relation'] for r in positive_rows} != {'CONTRAINDICATION'}):
        raise ValueError('Native contra positives differ from the legal training contraindications')
    for pair in positive_pairs:
        data.pairs.assert_optimizer_pair(pair, 'P')
    # HGTDR reads its training positives from native.supports.
    native.support_ids = tuple(positive_pairs)
    native.supports = np.asarray([[native.lookup['drug'][d], native.lookup['disease'][s]]
                                  for d, s in positive_pairs], dtype=np.int64)
    universe, gate = data.pairs.universe, data.pairs.gate
    native.u_policy = UnlabeledPolicy(tuple(universe.eligible_drugs), tuple(universe.eligible_diseases),
                                      frozenset(universe.observed_pairs),
                                      frozenset(gate.formal | gate.blocked | gate.positive))
    native.graph_receipt, native.relations = receipt, relations
    native.sampler = adapter.native_module().QueryMaskedSampler(counts, relations, arrays)
    native.metadata = (list(counts), native.sampler.edge_types)
    return native, receipt_path


def load_contra_fold(cell: str, shared: TrainingInputs | None = None) -> ContraFold:
    if cell not in CELLS:
        raise ValueError('HGTDR-contra is limited to the five frozen R0 cells')
    from evi_hgt_phase98.corpus import verify_manifest

    data = (shared if shared is not None else TrainingInputs()).cell(cell)
    corpus = verify_manifest(data)
    rows = project_contra_candidates(data)
    native, receipt = _load_native_graph(data, corpus)
    return ContraFold(data, native, rows, dict(
        method='HGTDR_ADAPTED_CONTRA', cell=cell, head='contra',
        fold_mask_digest=data.bindings['fold_mask_digest'],
        training_positive_digest=historical_digest(data.cell['auxiliary_training']),
        training_u_manifest_digest=data.bindings['training_u_manifest_digest'],
        corpus_digest=corpus['digest'], native_graph_receipt=str(receipt),
        native_graph_sha256=sha(receipt), candidate_order_sha256=digest(rows),
        source_sha256=sha(__file__),
    ))
