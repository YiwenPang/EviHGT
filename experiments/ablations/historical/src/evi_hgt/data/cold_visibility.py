"""Opt-in, query-local cold evaluation visibility. Never used for training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

POLICY_PATH = 'outputs/phase8_2_2/cold_visibility_policy.json'
POLICY_VERSION = 'COLD_QUERY_SIDE_INFORMATION_V1'
A_BLOCKED = [19, 24, 31]
C_BLOCKED = [8, 11, 15, 20, 23, 25, 32, 34, 41, 54]
# A source is admitted only for this component and relation family. Aggregate
# Open Targets disease/gene evidence without a named genetic source is denied.
SOURCE_RULES = {
    'anatomy_anatomy': ('direct', ['UBERON']),
    'anatomy_gene': ('direct', ['BGEE']),
    'biological_process_biological_process': ('direct', ['GO']),
    'cellular_component_cellular_component': ('direct', ['GO']),
    'molecular_function_molecular_function': ('direct', ['GO']),
    'drug_gene': ('direct', ['DRUG_BANK']),
    'disease_gene': ('indirect', ['CLINGEN', 'GENOMICS_ENGLAND', 'ORPHANET']),
    'disease_phenotype': ('indirect', ['HPO']),
    'pathway_gene': ('indirect', ['REACTOME']),
    'pathway_pathway': ('direct', ['REACTOME']),
    'gene_gene': ('indirect', ['BIOPLEX', 'INTACT', 'BIOGRID']),
}
CERTIFIED_SOURCES = {s for _, sources in SOURCE_RULES.values() for s in sources}
DERIVED_MARKERS = r'(?i)"(?:target_dda|dda_label|supervision_label|target_label|highest_clinical_trial_phase|indication|contraindication|off_label_use)"\s*:'


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class ColdFactMask:
    """Canonical membership: dropping incident isolation never drops other reasons."""
    admitted: np.ndarray
    source: np.ndarray
    target: np.ndarray
    permanent: np.ndarray
    hidden_nodes: set[int] | frozenset[int]
    target_facts: set[int] | frozenset[int]

    def __contains__(self, value):
        i = int(value)
        return (i < 0 or i >= self.admitted.size or not self.admitted[i]
                or self.permanent[i] or i in self.target_facts
                or int(self.source[i]) in self.hidden_nodes
                or int(self.target[i]) in self.hidden_nodes)

    def __bool__(self):
        return True


def sanitize_evidence(evidence, fact_ids, certified_sources):
    """Facts own evidence visibility. Unverified fields never reach tokenization."""
    result = evidence.filter(pl.col('canonical_fact_id').is_in(sorted(fact_ids))).with_columns(
        pl.lit('{}').alias('properties'),
        pl.col('raw_direct_sources').list.eval(pl.element().filter(pl.element().is_in(sorted(certified_sources)))),
        pl.col('raw_indirect_sources').list.eval(pl.element().filter(pl.element().is_in(sorted(certified_sources)))),
        pl.lit(0).cast(evidence.schema['source_row_number']).alias('source_row_number'),
    )
    # These optional annotations are not independent cold input features.
    for name in ('edge_evidence_proposition', 'overlap_category'):
        if name in result.columns:
            result = result.with_columns(pl.lit(None).cast(result.schema[name]).alias(name))
    return result


def _safe_record(component, role, sources):
    expression = (pl.col(f'raw_{role}_sources').list.eval(pl.element().is_in(sources)).list.any()
        & (pl.col('source_component_path') == f'edges/{component}.parquet')
        & ~pl.col('properties').fill_null('{}').str.contains(DERIVED_MARKERS))
    if component != 'anatomy_gene':
        expression = expression & (~pl.col('edge_support_observed').fill_null(True)) & (~pl.col('edge_counter_evidence_observed').fill_null(True))
    return expression


def build_admission(root):
    """Read-only bounded column scans; no KG Python object graph or model."""
    root = Path(root)
    base = root/'data/processed/phase3/v1'
    catalog_path = base/'graph/csr/edge_type_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    if [x['edge_type_id'] for x in catalog] != list(range(57)):
        raise ValueError('cold visibility requires the audited 57-family catalog')
    catalog_df = pl.DataFrame(catalog)
    count = json.loads((base/'manifest/completion.json').read_text())['canonical_fact_count']
    admitted = np.zeros(count, dtype=np.bool_)
    source = np.zeros(count, dtype=np.uint32)
    target = np.zeros(count, dtype=np.uint32)
    bindings = {}
    for p in [catalog_path, base/'manifest/completion.json', base/'manifest/edge_partitions.json',
              root/'src/evi_hgt/data/canonical_facts.py', root/'src/evi_hgt/data/dda_supervision.py',
              root/'outputs/optimuskg_schema/provenance_summary.csv', root/'docs/data_dictionary.md']:
        bindings[p.relative_to(root).as_posix()] = file_digest(p)
    summary = []
    for component, (role, sources) in SOURCE_RULES.items():
        fp = base/'canonical_facts'/f'{component}.parquet'
        ep = base/'evidence_records'/fp.name
        if not fp.exists() or not ep.exists():
            summary.append({'component': component, 'status': 'MISSING_DENIED'})
            continue
        for p in (fp, ep):
            bindings[p.relative_to(root).as_posix()] = file_digest(p)
        facts = pl.read_parquet(fp, columns=['canonical_fact_id', 'source_global_id', 'target_global_id',
            'source_node_type', 'target_node_type', 'relation', 'source_component_path'])
        facts = facts.join(catalog_df, on=['source_node_type', 'relation', 'target_node_type'], how='inner').filter(
            (~pl.col('edge_type_id').is_in(A_BLOCKED+C_BLOCKED)) &
            (pl.col('source_component_path') == f'edges/{component}.parquet'))
        # At least one independently sourced evidence record must support the
        # exact canonical relation/component. Other evidence is not exposed.
        safe = pl.scan_parquet(ep).filter(_safe_record(component, role, sources)).select('canonical_fact_id', 'relation').unique().collect()
        safe_facts = facts.join(safe, on=['canonical_fact_id', 'relation'], how='semi')
        ids = safe_facts['canonical_fact_id'].to_numpy()
        admitted[ids] = True
        source[ids] = safe_facts['source_global_id'].to_numpy()
        target[ids] = safe_facts['target_global_id'].to_numpy()
        summary.append({'component': component, 'canonical_facts': facts.height,
                        'admitted_facts': safe_facts.height,
                        'admitted_edge_type_ids': sorted(safe_facts['edge_type_id'].unique().to_list()),
                        'evidence_role': role, 'certified_sources': sources})
    passed = sorted({i for s in summary for i in s.get('admitted_edge_type_ids', [])})
    policy = {
        'policy_version': POLICY_VERSION, 'a_blocked_ids': A_BLOCKED,
        'c_default_deny_ids': C_BLOCKED, 'b_admitted_ids': passed,
        'b_rejected_or_uncertain_ids': sorted(set(range(57))-set(A_BLOCKED+C_BLOCKED+passed)),
        'rejected_basis': {'3,5,51': 'GO annotation evidence not individually certified in this patch',
                          '44,45,46,47,48,49': 'Exposure associations/hierarchy origin not independently certified',
                          '53': 'Aggregate phenotype-gene association lacks certified nonclinical lineage'},
        'source_rules': SOURCE_RULES, 'admission_summary': summary, 'source_sha256': bindings,
        'field_policy': {'properties': 'ALL_EXCLUDED_UNVERIFIED_FREE_TEXT_NUMERIC_AND_RECORD_IDENTIFIERS',
                         'source_row_number': 'EXCLUDED_CONSTANT_ZERO',
                         'raw_sources': 'COMPONENT_CERTIFIED_DATASET_NAMES_ONLY',
                         'evidence': 'ONLY_CERTIFIED_RECORDS_OF_ADMITTED_FACTS',
                         'overlap_category_and_proposition': 'EXCLUDED',
                         'observed_polarity': 'BGEE_EXPRESSION_ONLY_OTHER_NONFALSE_FLAGS_DENIED'},
        'precedence': 'ONLY_HELDOUT_DRUG_INCIDENT_OR_HELDOUT_DISEASE_INCIDENT_MAY_BE_RELEASED; ALL_OTHER_REASONS_WIN',
        'endpoint_rule': 'CURRENT_QUERY_COLD_ENDPOINT_IN_REQUESTED_PARTITION_ONLY',
        'canonical_rule': 'ONE_CANONICAL_MASK_FOR_BOTH_CSR_DIRECTIONS',
        'path_rule': 'ADMITTED_FACTS_ONLY; OTHER_HELDOUT_NODES_FORBIDDEN; CANONICAL_FACT_SIMPLE; EXISTING_EXACT_FALLBACK',
        'training': 'UNCHANGED_STRICT_ISOLATION', 'standard': 'UNCHANGED',
        'source_protocols': ['PRACTICAL_FORMAL_V1', 'phase3-dda-target-mask-v1',
                             'docs/superpowers/specs/2026-08-22-phase3-leakage-safe-data-design.md'],
        'admission_basis': 'Canonicalization preserves component endpoints and observed relation; no DDA label join creates these facts. DrugBank molecular action, HPO phenotype, named curated genetic associations and explicit ontology/interaction sources only. Aggregate/unknown sources denied.',
    }
    policy['digest'] = digest(policy)
    return policy, admitted, source, target


class ColdVisibility:
    def __init__(self, root, expected_digest):
        self.root = Path(root)
        stored = json.loads((self.root/POLICY_PATH).read_text())
        policy, self.admitted, self.source, self.target = build_admission(self.root)
        if policy['digest'] != expected_digest or stored != json.loads(json.dumps(policy)):
            raise ValueError('cold visibility policy/source admission drift')
        self.policy = policy
        self._permanent = {}

    def mask(self, assembler, split, partition, pair, endpoints, target_facts, universe):
        if split not in {'cold_drug', 'cold_disease'} or partition not in {'validation', 'test'}:
            raise ValueError('cold visibility is evaluation-only for two frozen splits')
        axis = 'drug' if split == 'cold_drug' else 'disease'
        index = 0 if axis == 'drug' else 1
        rows = pl.read_parquet(assembler.formal_root/'splits'/f'{split}.parquet').filter(pl.col(f'{axis}_id') == pair[index])
        assignments = set(rows[f'{axis}_partition'].drop_nulls().to_list())
        if assignments != {partition}:
            raise ValueError('current cold endpoint is not in requested partition')
        if split not in self._permanent:
            permanent = np.zeros(self.admitted.size, dtype=np.bool_)
            for path in sorted((assembler.formal_root/'views'/split/'training_forbidden_facts').glob('*.parquet')):
                frame = pl.read_parquet(path)
                # Empty/missing reasons fail closed, as do future unknown reasons.
                releasable = pl.col('exclusion_reasons').list.eval(
                    (pl.element() == f'HELDOUT_{axis.upper()}_INCIDENT').fill_null(False)
                ).list.all().fill_null(False) & (pl.col('exclusion_reasons').list.len() > 0)
                blocked = frame.filter(~releasable)['canonical_fact_id'].to_numpy()
                permanent[blocked] = True
            self._permanent[split] = permanent
        hidden = assembler._heldout_globals(split, universe) - {endpoints[index]}
        return ColdFactMask(self.admitted, self.source, self.target, self._permanent[split], hidden, target_facts)

    def evidence(self, evidence, fact_ids):
        parts = []
        for component, (role, sources) in SOURCE_RULES.items():
            part = evidence.filter((pl.col('adapter_component') == component) & _safe_record(component, role, sources))
            parts.append(sanitize_evidence(part, fact_ids, set(sources)))
        return pl.concat(parts).sort('evidence_record_id')
