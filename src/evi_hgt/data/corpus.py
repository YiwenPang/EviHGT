"""Verified portable canonical corpus, fixed training plan, and development panels.

This reader consumes normalized artifacts; it does not infer clinical labels
from free text, download a graph, or open a research workspace implicitly.
"""
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.training.input_index import StaticInputIndex, InputBinding, BuildBudget
from evi_hgt.training.supervision import derive_observed_query_supervision
from .common import digest, read_json, resolve_input, sha256
from .raw_features import FeatureStore, DRUG_DIM, DISEASE_DIM
from .pretrained import PretrainedStore, NODE_TYPES
from .plan import TrainingPlan
from .graph import QueryAssembler

SCHEMA = 'EVIHGT_PORTABLE_CORPUS_V1'


def _pairs(frame, name):
    required = {'drug_id', 'disease_id'}
    if not required <= set(frame.columns):
        raise ValueError(name + ' requires drug_id and disease_id')
    selected = frame.select('drug_id', 'disease_id')
    if selected.null_count().sum_horizontal().item() or selected.n_unique() != selected.height:
        raise ValueError(name + ' has null or duplicate canonical pairs')
    rows = set(selected.iter_rows())
    if any(not isinstance(v, str) or not v for pair in rows for v in pair):
        raise ValueError(name + ' requires nonempty canonical string IDs')
    return rows


class PortableCorpus:
    def __init__(self, manifest_path, progress=None):
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError('Prepared corpus is missing: ' + str(path)
                + '. Run main.py --mode prepare with the raw files listed in docs/data.md, or import verified normalized inputs with prepare_data.py.')
        self.root = path.parent
        self.manifest = m = read_json(path)
        if m.get('schema') != SCHEMA:
            raise ValueError('Unsupported corpus schema; expected ' + SCHEMA)
        if type(m.get('training_seed')) is not int or not 0 <= m['training_seed'] < 2**31:
            raise ValueError('Corpus training_seed must be an integer in [0, 2**31)')
        self.training_seed = m['training_seed']
        files = m.get('files', {})
        if not files or not isinstance(files, dict):
            raise ValueError('A content-hashed input inventory is required')
        self._paths = {}
        for i, (relative, expected) in enumerate(sorted(files.items()), 1):
            resolved = resolve_input(self.root, relative)
            if not isinstance(expected, str) or len(expected) != 64 or sha256(resolved) != expected:
                raise ValueError('Corpus input hash mismatch: ' + relative)
            self._paths[relative] = resolved
            if progress:
                progress('verify', i, len(files), 0)
        self.identity = digest(m)
        def stage(name):
            if progress:
                progress(name,0,0,0)
        stage('canonical metadata')
        provenance = m.get('semantics', {})
        for key in ('canonical_entities_verified', 'canonical_facts_deduplicated', 'evidence_records_deduplicated',
                    'clinical_labels_verified', 'target_aliases_resolved', 'disease_gene_label_leakage_removed',
                    'split_mask_applied', 'features_label_independent'):
            if provenance.get(key) is not True:
                raise ValueError('Normalized source verification must be explicit: ' + key)
        if provenance.get('unknown_is_negative') is not False:
            raise ValueError('Unlabeled pairs must remain unknown, not confirmed negatives')
        nodes = pl.read_parquet(self.resolve(m['nodes'])).sort('global_node_id')
        if tuple(sorted(nodes['node_type'].unique())) != NODE_TYPES:
            raise ValueError('Canonical node-type catalog must preserve the ten implemented types')
        if not np.array_equal(nodes['global_node_id'].to_numpy(), np.arange(nodes.height)):
            raise ValueError('Global node IDs must be unique, contiguous, and zero based')
        self.nodes = nodes
        catalog = read_json(self.resolve(m['edge_catalog']))
        edge_types = tuple((r['source_node_type'], r['relation'], r['target_node_type']) for r in catalog)
        def stream(entries):
            for entry in entries:
                p = self.resolve(entry['path'])
                parquet = pq.ParquetFile(p)
                if entry['rows'] != parquet.metadata.num_rows:
                    raise ValueError('Canonical/evidence row count changed: ' + entry['path'])
                for chunk in parquet.iter_batches(batch_size=8192, use_threads=False):
                    yield pl.from_arrow(chunk).with_columns(pl.lit(entry['component']).alias('adapter_component'))
        if not m.get('fact_files') or not m.get('evidence_files'):
            raise ValueError('Canonical fact and evidence files are required')
        records = tuple((rel, h, self.resolve(rel).stat().st_size, '', None) for rel, h in sorted(files.items()))
        token_buckets = m.get('token_buckets', 8192)
        if token_buckets != 8192:
            raise ValueError('The implemented token namespace uses 8192 buckets')
        binding = InputBinding(self.identity, records, 'portable-raw-v1', token_buckets)
        self.index = StaticInputIndex._from_streams(nodes=nodes, fact_frames=stream(m['fact_files']),
            evidence_frames=stream(m['evidence_files']), fact_count=sum(e['rows'] for e in m['fact_files']),
            evidence_count=sum(e['rows'] for e in m['evidence_files']), edge_catalog=edge_types,
            binding=binding, budget=BuildBudget(), progress=progress)
        self.model_config = EviHGTConfig(num_node_types=len(NODE_TYPES), num_edge_types=len(edge_types),
            num_nodes=nodes.height, num_numeric_features=6, num_boolean_features=1,
            return_edge_context=True)
        self.csr = {key: np.load(self.resolve(m['csr'][key]), mmap_mode='r', allow_pickle=False)
                    for key in ('indptr', 'indices', 'fact_ids')}
        stage('CSR validation')
        self._validate_csr()
        stage('training and development memberships')
        self.forbidden = set(map(int, pl.read_parquet(self.resolve(m['forbidden_facts']))['canonical_fact_id']))
        self.blocked = _pairs(pl.read_parquet(self.resolve(m['blocked_pairs'])), 'blocked_pairs')
        self.observed = _pairs(pl.read_parquet(self.resolve(m['observed_pairs'])), 'observed_pairs')
        eligible = read_json(self.resolve(m['eligibility']))
        self.eligible_drugs = frozenset(eligible['eligible_drugs'])
        self.eligible_diseases = frozenset(eligible['eligible_diseases'])
        node_names = {kind: set(nodes.filter(pl.col('node_type') == kind)['node_id']) for kind in ('DRG','DIS')}
        if not self.eligible_drugs <= node_names['DRG'] or not self.eligible_diseases <= node_names['DIS']:
            raise ValueError('Eligible entities must match canonical drug/disease identities')
        training = pl.read_parquet(self.resolve(m['training_pairs']))
        if 'channel' not in training.columns or 'raw_relation' not in training.columns:
            raise ValueError('Training positives require verified channel and raw_relation columns')
        self.support, self.unsuitability = set(), set()
        for channel, relation in (('support','INDICATION'), ('unsuitability','CONTRAINDICATION')):
            frame = training.filter(pl.col('channel') == channel)
            if frame.height == 0 or set(frame['raw_relation']) != {relation}:
                raise ValueError('Verified positive relation semantics missing for ' + channel)
            setattr(self, channel, _pairs(frame, channel))
        if training.height != len(self.support)+len(self.unsuitability):
            raise ValueError('Training table has an unknown channel')
        if not (self.support | self.unsuitability) <= self.observed:
            raise ValueError('Positive labels are missing from observed-pair U exclusions')
        if (self.support | self.unsuitability) & self.blocked:
            raise ValueError('Blocked evaluation/holdout pair entered training positives')
        self.panels, self.panel_pairs = {}, {}
        for domain, relative in m.get('panels', {}).items():
            if domain not in ('validation', 'train_probe', 'auxiliary_validation'):
                raise ValueError('Unsupported development panel: ' + domain)
            frame = pl.read_parquet(self.resolve(relative))
            pairs = _pairs(frame, domain)
            if 'role' not in frame.columns or not set(frame['role']) <= {'P','U'}:
                raise ValueError('Development panel roles must be P or U')
            if domain == 'auxiliary_validation' and set(frame['role']) != {'P'}:
                raise ValueError('Contra development panel contains only verified contraindication positives')
            if not pairs <= self.blocked:
                raise ValueError('Development panel is missing from the graph/training exclusions')
            if domain in ('validation', 'train_probe'):
                unlabeled = set(frame.filter(pl.col('role') == 'U').select('drug_id','disease_id').iter_rows())
                if unlabeled & self.observed:
                    raise ValueError('Observed clinical pair cannot be a development U')
            positive = set(frame.filter(pl.col('role') == 'P').select('drug_id','disease_id').iter_rows())
            if not positive <= self.observed:
                raise ValueError('Development positives must remain in observed-pair exclusions')
            expected_relation = 'CONTRAINDICATION' if domain == 'auxiliary_validation' else 'INDICATION'
            if positive and ('raw_relation' not in frame.columns or
                set(frame.filter(pl.col('role') == 'P')['raw_relation']) != {expected_relation}):
                raise ValueError('Development P requires its verified raw clinical relation')
            self.panels[domain], self.panel_pairs[domain] = frame, pairs
        stage('target leakage validation')
        self._validate_targets()
        stage('raw entity feature validation')
        raw = m['raw_features']
        arrays = [np.load(self.resolve(raw[name]), mmap_mode='r', allow_pickle=False)
                  for name in ('drug_ids','drug_values','disease_ids','disease_values')]
        for ids, values, kind, width in ((arrays[0],arrays[1],'DRG',DRUG_DIM), (arrays[2],arrays[3],'DIS',DISEASE_DIM)):
            expected = nodes.filter(pl.col('node_type') == kind)['global_node_id'].to_numpy()
            if not np.array_equal(ids, expected) or values.dtype != np.uint8 or values.shape != (len(ids),width):
                raise ValueError('Raw SideInfo identity/dtype/dimension mismatch: ' + kind)
            for start in range(0, len(ids), 4096):
                chunk = values[start:start+4096]
                if not np.isin(chunk, [0,1]).all():
                    raise ValueError('Raw SideInfo must contain binary values')
                split = 2048 if kind == 'DRG' else 512
                if np.any(chunk[chunk[:,-2] == 0,:split]) or np.any(chunk[chunk[:,-1] == 0,split:-2]):
                    raise ValueError('Unavailable SideInfo family contains nonzero values')
        self.features = FeatureStore(*arrays)
        stage('frozen D1 feature validation')
        self.pretrained = PretrainedStore(nodes, m['pretrained'], self.resolve)
        stage('three-cycle schedule validation')
        self.plan = TrainingPlan(self, m['schedule'], self.resolve)
        self.assembler = QueryAssembler(self)

    def resolve(self, relative):
        if relative not in self._paths:
            raise ValueError('Unbound corpus input path: ' + str(relative))
        return self._paths[relative]

    def _validate_csr(self):
        ptr, neighbors, facts = (self.csr[n] for n in ('indptr','indices','fact_ids'))
        if any(v.ndim != 1 or v.dtype.kind not in 'iu' for v in (ptr,neighbors,facts)):
            raise ValueError('CSR requires one-dimensional integer arrays')
        if len(ptr) != self.nodes.height+1 or ptr[0] != 0 or ptr[-1] != len(neighbors) or len(facts) != len(neighbors):
            raise ValueError('CSR dimensions or terminal offset differ from canonical nodes')
        for start in range(0, len(ptr)-1, 1_000_000):
            chunk = ptr[start:start+1_000_001]
            if np.any(chunk[1:] < chunk[:-1]):
                raise ValueError('CSR row offsets must be monotone')
        for start in range(0, len(facts), 1_000_000):
            chunk = neighbors[start:start+1_000_000]
            if len(chunk) and (chunk.min() < 0 or chunk.max() >= self.nodes.height):
                raise ValueError('CSR neighbor ID is outside the canonical node index')
            self.index._fact_map.positions(facts[start:start+1_000_000])

    def _validate_targets(self):
        m = self.manifest
        mapping = pl.read_parquet(self.resolve(m['target_map']))
        self.target_ids = {}
        for drug, disease, fact in mapping.select('drug_id','disease_id','canonical_fact_id').iter_rows():
            self.target_ids.setdefault((drug,disease),set()).add(int(fact))
        known = self.index._fact_map._ids
        def any_present(values):
            candidates = np.fromiter(values, dtype=np.int64)
            positions = np.searchsorted(known, candidates)
            valid = positions < len(known)
            return bool(np.any(known[positions[valid]] == candidates[valid]))
        if any_present(self.forbidden):
            raise ValueError('Forbidden split fact remains in canonical inputs')
        # Project identity columns only; no graph-sized Python property objects.
        for entry in m['fact_files']:
            frame = pl.read_parquet(self.resolve(entry['path']), columns=['canonical_fact_id',
                'source_node_id','target_node_id','source_node_type','target_node_type'])
            direct = frame.filter(((pl.col('source_node_type') == 'DRG') & (pl.col('target_node_type') == 'DIS')) |
                                  ((pl.col('source_node_type') == 'DIS') & (pl.col('target_node_type') == 'DRG')))
            for fact, src, dst, src_kind, _ in direct.iter_rows():
                pair = (src,dst) if src_kind == 'DRG' else (dst,src)
                if pair in self.blocked:
                    raise ValueError('Held-out clinical endpoints remain in the canonical graph')
                if int(fact) not in self.target_ids.get(pair,set()):
                    raise ValueError('Target map must cover every direct/reverse drug-disease fact, regardless of relation')
        for pair in self.blocked:
            if any_present(self.target_ids.get(pair,set())):
                raise ValueError('Held-out equivalent target fact remains in the canonical graph')

    def assert_training_pair(self, pair, role):
        pair = tuple(pair)
        if len(pair) != 2 or pair in self.blocked:
            raise ValueError('Blocked or malformed training pair')
        if role == 'U':
            if pair in self.observed or pair[0] not in self.eligible_drugs or pair[1] not in self.eligible_diseases:
                raise ValueError('Unlabeled pair is observed or outside the eligible universe')
        elif role not in ('support','unsuitability') or pair not in getattr(self,role):
            raise ValueError('Positive absent from its verified clinical channel')

    def positive(self, pair, channel):
        self.assert_training_pair(pair, channel)
        return derive_observed_query_supervision(dict(drug_id=pair[0], disease_id=pair[1],
            raw_relation='INDICATION' if channel == 'support' else 'CONTRAINDICATION', policy_b_member=True,
            support_target=1.0 if channel == 'support' else None, support_mask=channel == 'support',
            unsuitability_target=1.0 if channel == 'unsuitability' else None, unsuitability_mask=channel == 'unsuitability'))


def load_corpus(manifest_path, progress=None):
    return PortableCorpus(manifest_path, progress=progress)
