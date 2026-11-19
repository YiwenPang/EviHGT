"""Training-only projections; no outer/final member or label decoder exists here.

Legacy cell containers are streamed by field. Missing cell containers use the
original pair-hash reservations, decoding roles only after training eligibility.
Role-free development pair IDs serve only to construct exclusion complements.
"""
from .protocol_ids import EXCLUSION_UNIVERSE_VERSION
from pathlib import Path
from types import SimpleNamespace
import json
import re

import polars as pl
from evi_hgt.graph_inputs.common import DATA, digest, read, verify
from evi_hgt.cross_validation import TrainingGate
from evi_hgt.graph_inputs.adapter import DevelopmentPairs
from evi_hgt.training.candidate_universe import CandidateUniverse
from evi_hgt.query_model.data import CellDataset
from evi_hgt.inner_validation.production import HOLDOUT
from .common import ROOT, sha


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


def pair_ids(path):
    frame = pl.read_parquet(path, columns=['drug_id', 'disease_id'])
    return set(frame.iter_rows())


class TrainingInputs:
    def __init__(self):
        self.protocol = verify(read(ROOT/'outputs/development_protocol/protocol_v2_r2.json'))
        self.keys = verify(read(ROOT/'outputs/conditional_sampling/cv_three_key_binding.json'))
        self.dev = verify(read(DATA/'development_manifest.json'))
        self.u = verify(read(DATA/'training_u_manifest.json'))
        self.features = verify(read(DATA/'features_v2/manifest.json'))
        self.census = read(ROOT/'outputs/query_model/cv_15_cell_static_census.json')
        self.part1 = read(ROOT/'outputs/ranking_protocol/ranking_comparison/part1/RANKING_COMPARISON_EVALUATION_PLAN.json')
        self.input_hashes = {}
        for name in ('all_forbidden_pairs.parquet', 'observed_pair_exclusions.parquet',
                     'auxiliary_positives.parquet', 'development_panel.parquet',
                     'final_holdout_exclusion_pairs.parquet'):
            path = DATA/name; actual = sha(path)
            if actual != self.dev['files'][name]:
                raise ValueError('Frozen role-free source changed: '+name)
            self.input_hashes[str(path)] = actual
        self.final = pair_ids(DATA/'final_holdout_exclusion_pairs.parquet')
        hp = ROOT/'data/processed/pair_exclusions/formal_pair_exclusions.parquet'
        self.historical = pair_ids(hp)
        self.frozen = pair_ids(DATA/'all_forbidden_pairs.parquet')
        if self.frozen != self.final | self.historical:
            raise ValueError('Role-free exclusion union differs')
        self.observed = pair_ids(DATA/'observed_pair_exclusions.parquet')
        self.aux_pairs = pair_ids(DATA/'auxiliary_positives.parquet')
        # Only IDs; never the role column, split roles, or protected member files.
        self.panel_ids_for_exclusion = pair_ids(DATA/'development_panel.parquet')
        self.exclusion_shas = dict(final=sha(DATA/'final_holdout_exclusion_pairs.parquet'), historical=sha(hp))

    def cell(self, cell):
        r, f = int(cell[1]), int(cell[3])
        frozen = next(x for x in self.census['cells'] if (x['repeat_index'],x['fold_index']) == (r,f))
        public = next(x for x in self.part1['cells'] if x['cell'] == cell)
        if any(public[k] != frozen[k] for k in ('repeat_key','cell_digest','membership_digests','mask_digest')):
            raise ValueError('Part1/census identity differs')
        entry = next(x for x in self.keys['selected_repeat_bindings'] if x['original_repeat_index'] == r)
        assignment = ROOT/entry['fold_assignment_path']
        if sha(assignment) != entry['fold_assignment_sha256']:
            raise ValueError('Assignment bytes changed')
        # The old pilot has a different repeat key; it is never a CV-cell input.
        direct = ROOT/f'data/processed/query_model/repeat_{r}_fold_{f}/cell_manifest.json'
        if direct.is_file():
            parts = {n: [json.loads(x) for x in objects_in_array(direct, n)] for n in ('training','auxiliary_training')}
            audit = dict(mode='STREAMING_TWO_TRAINING_FIELDS_ONLY', source=str(direct), source_sha256=sha(direct),
                         outer_ids_decoded=0, outer_roles_decoded=0, reserved_roles_decoded=0)
        else:
            training, audit = project_training_assignments(assignment, fold=f, repeat_key=frozen['repeat_key'])
            train_ids = {(x['drug_id'],x['disease_id']) for x in training}
            excluded = self.panel_ids_for_exclusion - train_ids
            key = digest([frozen['repeat_key'],f,'inner'])
            aux = self.aux_pairs - excluded
            av = {p for p in aux if reserved(p,key+'aux_validation',.2)}
            ap = {p for p in aux-av if reserved(p,key+'aux_probe',.05)}
            parts = dict(training=[x for x in training if (x['drug_id'],x['disease_id']) not in av|ap],
                auxiliary_training=[dict(drug_id=a,disease_id=b,raw_relation='CONTRAINDICATION') for a,b in sorted(aux-av-ap)])
            audit.update(source=str(assignment),source_sha256=sha(assignment))
        for n in ('training','auxiliary_training'):
            if digest(parts[n]) != frozen['membership_digests'][n]:
                raise ValueError('Training projection membership differs: '+cell+' '+n)
        return TrainingCell(self,cell,frozen,parts,entry['fold_assignment_digest'],audit)


class TrainingCell(CellDataset):
    def __init__(self, shared, name, frozen, parts, fold_digest, audit):
        self.shared=shared; self.cell_id=name; self.projection_audit=audit
        self.protocol=shared.protocol; self.u=shared.u; self.features=shared.features
        self.repeat_index=frozen['repeat_index']; self.training_seed=frozen['training_seed']
        self.final=shared.final; self.historical=shared.historical; self.frozen=shared.frozen
        self.cell=dict(parts,repeat_key=frozen['repeat_key'],fold=frozen['fold_index'],digest=frozen['cell_digest'],fold_digest=fold_digest)
        self.contract=SimpleNamespace(bindings=frozen,seed=self.training_seed)
        self.support={(x['drug_id'],x['disease_id']) for x in parts['training'] if x['role']=='P'}
        self.unsuitability={(x['drug_id'],x['disease_id']) for x in parts['auxiliary_training']}
        train_ids={(x['drug_id'],x['disease_id']) for x in parts['training']}
        self.blocked=self.frozen | (shared.panel_ids_for_exclusion-train_ids) | (shared.aux_pairs-self.unsuitability)
        if digest(sorted(self.blocked)) != frozen['mask_digest']:
            raise ValueError('Role-free training-complement mask differs: '+name)
        gate=TrainingGate.__new__(TrainingGate)
        gate.formal=set(self.frozen); gate.blocked=self.blocked-self.frozen; gate.positive=self.support|self.unsuitability
        pairs=DevelopmentPairs.__new__(DevelopmentPairs)
        pairs.drugs=frozenset(self.u['eligible_drugs']); pairs.diseases=frozenset(self.u['eligible_diseases'])
        pairs.observed=frozenset(shared.observed); pairs.final=frozenset(self.final); pairs.gate=gate
        pairs.universe=CandidateUniverse(tuple(sorted(pairs.drugs)),tuple(sorted(pairs.diseases)),pairs.observed,
            {},frozenset(),frozenset(),'standard',version=EXCLUSION_UNIVERSE_VERSION)
        self.pairs=pairs; self.domains={}
        self.bindings=dict(protocol_digest=self.protocol['digest'],fold_manifest_digest=frozen['cell_digest'],
            fold_assignment_digest=fold_digest,development_pool_digest=shared.dev['digest'],feature_digest=self.features['digest'],
            holdout_exclusion_digest=shared.exclusion_shas['final'],old_formal_exclusion_digest=shared.exclusion_shas['historical'],
            final_holdout_digest=HOLDOUT,cv_three_key_binding_digest=shared.keys['digest'],original_repeat_index=self.repeat_index,
            repeat_key=frozen['repeat_key'],fold_index=frozen['fold_index'],training_seed=self.training_seed,
            training_seed_rule='PART3B1_SHARED_CELL_SEED_V1',training_u_manifest_digest=self.u['digest'],fold_mask_digest=frozen['mask_digest'])
        self.original_corpus_root=ROOT/f"data/processed/query_model/repeat_{self.repeat_index}_fold_{self.cell['fold']}/corpus_v1"
        self.corpus_root=self.original_corpus_root
        if self.budget()!=frozen['budget'] or self.legal_u_domain()!=frozen['legal_training_u_domain']:
            raise ValueError('Training budget/U universe differs')
        for p in self.support|self.unsuitability:
            pairs.assert_optimizer_pair(p,'P')

    def assert_score_domain(self, *args, **kwargs):
        raise PermissionError('Training-only RankingComparison data cannot score evaluation domains')

    def enable_development(self, *args, **kwargs):
        raise PermissionError('Training-only RankingComparison data has no evaluation grant route')


def load_training_graph(data):
    """Load a certified historical or atomic RankingComparison cell corpus, no construction."""
    from evi_hgt.ranking_protocol.real_data import load_index, load_canonical_assembler
    from .corpus import verify_manifest
    corpus=verify_manifest(data)
    index=load_index(corpus)
    return corpus,index,load_canonical_assembler(data,corpus,index)
