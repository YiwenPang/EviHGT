"""Local-only extraction of structured DrugBank SMILES for reconstructed graph drugs.

This module never uploads data, infers therapeutic labels, or reads evaluation files.
It streams the original XML read-only and retains only requested graph drug records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('POLARS_MAX_THREADS', '4')
DB = '{http://www.drugbank.ca}'
XS = '{http://www.w3.org/2001/XMLSchema}'


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_schema_contract(xsd_path):
    """Verify only the referenced field contract, not full XML schema validity."""
    schema = ET.parse(xsd_path).getroot()
    if schema.get('targetNamespace') != 'http://www.drugbank.ca':
        raise ValueError('Unexpected DrugBank schema namespace')
    drug = schema.find(f"{XS}complexType[@name='drug-type']")
    calculated = schema.find(f"{XS}complexType[@name='calculated-property-type']")
    kind = schema.find(f"{XS}simpleType[@name='calculated-property-kind-type']")
    identifier = schema.find(f"{XS}complexType[@name='drugbank-drug-salt-id-type']")
    checks = {
        'drug_has_calculated_properties': drug is not None and drug.find(f".//{XS}element[@name='calculated-properties']") is not None,
        'drug_has_drugbank_id': drug is not None and drug.find(f".//{XS}element[@name='drugbank-id']") is not None,
        'primary_boolean': identifier is not None and identifier.find(f".//{XS}attribute[@name='primary'][@type='xs:boolean']") is not None,
        'smiles_enumerated': kind is not None and kind.find(f".//{XS}enumeration[@value='SMILES']") is not None,
        'calculated_value_string': calculated is not None and calculated.find(f".//{XS}element[@name='value'][@type='xs:string']") is not None,
        'calculated_source': calculated is not None and calculated.find(f".//{XS}element[@name='source']") is not None,
    }
    if not all(checks.values()):
        raise ValueError(f'DrugBank schema contract mismatch: {checks}')
    return {'checks': checks, 'xsd_sha256': sha256_file(xsd_path),
            'full_xml_schema_validation': 'NOT_PERFORMED',
            'field_path': "drug/calculated-properties/property[kind='SMILES']/value"}


def missing_record(reason='drugbank_id_not_found'):
    return {'drugbank_primary_id': None, 'id_match_type': None, 'smiles': None,
            'smiles_property_source': None, 'structure_available': False,
            'availability_reason': reason, 'distinct_smiles_values': 0}


def extract_candidate_smiles(xml_path, candidate_ids):
    """Return one record per requested ID with primary matches preferred over aliases."""
    xml_path = Path(xml_path)
    requested = set(candidate_ids)
    if not requested or any(not isinstance(value, str) or not value for value in requested):
        raise ValueError('A nonempty set of graph DrugBank IDs is required')
    before = xml_path.stat()
    matches = defaultdict(list)
    meta = {'drug_records_scanned': 0, 'candidate_ids': len(requested)}
    depth = 0
    root = None
    with xml_path.open('rb') as stream:
        reader = HashingReader(stream)
        for event, element in ET.iterparse(reader, events=('start', 'end')):
            if event == 'start':
                depth += 1
                if depth == 1:
                    if element.tag != DB + 'drugbank':
                        raise ValueError('Unexpected DrugBank XML root/namespace')
                    root = element
                    meta.update({'drugbank_version': element.get('version'),
                                 'exported_on': element.get('exported-on')})
                continue
            if depth == 2 and element.tag == DB + 'drug':
                meta['drug_records_scanned'] += 1
                identifiers = element.findall(DB + 'drugbank-id')
                all_ids = {(item.text or '').strip() for item in identifiers}
                requested_here = requested & all_ids
                if requested_here:
                    primary_ids = {(item.text or '').strip() for item in identifiers
                                   if item.get('primary', 'false') in ('true', '1')}
                    primary_ids.discard('')
                    primary = next(iter(primary_ids)) if len(primary_ids) == 1 else None
                    values = defaultdict(set)
                    for prop in element.findall(DB + 'calculated-properties/' + DB + 'property'):
                        if prop.findtext(DB + 'kind') != 'SMILES':
                            continue
                        value = (prop.findtext(DB + 'value') or '').strip()
                        if value:
                            values[value].add((prop.findtext(DB + 'source') or '').strip())
                    record = missing_record('structured_smiles_missing')
                    record['drugbank_primary_id'] = primary
                    record['distinct_smiles_values'] = len(values)
                    if primary is None:
                        record['availability_reason'] = 'invalid_or_ambiguous_primary_id'
                    elif len(values) > 1:
                        record['availability_reason'] = 'conflicting_structured_smiles'
                    elif len(values) == 1:
                        smiles = next(iter(values))
                        record.update({'smiles': smiles, 'smiles_property_source': '|'.join(sorted(values[smiles])),
                                       'structure_available': True, 'availability_reason': 'structured_smiles_present'})
                    for requested_id in requested_here:
                        matches[requested_id].append(dict(record, id_match_type='primary' if requested_id == primary else 'secondary'))
                element.clear()
                # Remove each completed top-level record; nested <drug> elements
                # in pathway tables are intentionally never treated as entries.
                root.remove(element)
            depth -= 1
        meta['xml_sha256'] = reader.digest.hexdigest()
    after = xml_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError('DrugBank XML changed during read-only extraction')
    meta['xml_size_bytes'] = before.st_size
    meta['source_unchanged_during_read'] = True
    resolved = {}
    for requested_id in sorted(requested):
        candidates = matches.get(requested_id, [])
        primary_matches = [row for row in candidates if row['id_match_type'] == 'primary']
        candidates = primary_matches or candidates
        if not candidates:
            resolved[requested_id] = missing_record()
            continue
        if len({row['drugbank_primary_id'] for row in candidates}) > 1:
            resolved[requested_id] = missing_record('ambiguous_drugbank_id')
            continue
        variants = {json.dumps(row, sort_keys=True) for row in candidates}
        if len(variants) > 1:
            resolved[requested_id] = missing_record('conflicting_source_records')
        else:
            resolved[requested_id] = candidates[0]
    meta['matched_primary'] = sum(row['id_match_type'] == 'primary' for row in resolved.values())
    meta['matched_secondary'] = sum(row['id_match_type'] == 'secondary' for row in resolved.values())
    return resolved, meta


def validate_smiles_records(records):
    """Validate locally with RDKit; preserve original SMILES and suppress its text logs."""
    from rdkit import Chem, rdBase
    counts = Counter()
    for record in records.values():
        record['smiles_syntax_valid'] = None
        if not record['structure_available']:
            counts['not_available_before_chemical_parsing'] += 1
            continue
        with rdBase.BlockLogs():
            try:
                molecule = Chem.MolFromSmiles(record['smiles'])
            except Exception:
                # Parser exceptions may contain licensed source strings; keep only
                # an explicit aggregate count and never echo the exception message.
                molecule = None
                counts['parser_exception'] += 1
        valid = molecule is not None and molecule.GetNumAtoms() > 0
        record['smiles_syntax_valid'] = valid
        if valid:
            counts['syntax_valid'] += 1
            record['availability_reason'] = 'structured_smiles_valid'
        else:
            counts['syntax_invalid'] += 1
            record['structure_available'] = False
            record['availability_reason'] = 'smiles_parse_invalid'
    return {'parser': 'RDKit.Chem.MolFromSmiles', 'rdkit_version': rdBase.rdkitVersion,
            'original_strings_preserved': True, 'canonicalization_performed': False,
            'specific_smiles_logs_suppressed': True, **dict(counts)}


def reconstruct_smiles(nodes_path, xml_path, xsd_path, output_path, report_path):
    import polars as pl
    start = time.perf_counter()
    inputs = {Path(path).resolve() for path in (nodes_path, xml_path, xsd_path)}
    if Path(output_path).resolve() in inputs or Path(report_path).resolve() in inputs:
        raise ValueError('Output paths must never overwrite input files')
    schema = verify_schema_contract(xsd_path)
    drugs = (pl.read_parquet(nodes_path).filter(pl.col('node_type') == 'drug')
             .select('node_id', 'type_node_id', 'source_index', 'source_id').sort('type_node_id'))
    if drugs['source_id'].null_count() or drugs['source_id'].n_unique() != drugs.height:
        raise ValueError('Graph drug source IDs must be nonnull and unique')
    resolved, provenance = extract_candidate_smiles(xml_path, set(drugs['source_id'].to_list()))
    chemical_validation = validate_smiles_records(resolved)
    rows = [dict(node, **resolved[node['source_id']]) for node in drugs.iter_rows(named=True)]
    result = pl.DataFrame(rows, schema={
        'node_id': pl.Int64, 'type_node_id': pl.Int64, 'source_index': pl.Int64, 'source_id': pl.String,
        'drugbank_primary_id': pl.String, 'id_match_type': pl.String, 'smiles': pl.String,
        'smiles_property_source': pl.String, 'structure_available': pl.Boolean,
        'availability_reason': pl.String, 'distinct_smiles_values': pl.Int64,
        'smiles_syntax_valid': pl.Boolean})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    report = {'status': 'EXTRACTED_FROM_VERIFIED_STRUCTURED_FIELD',
              'source_xml': str(Path(xml_path).resolve()), 'source_xsd': str(Path(xsd_path).resolve()),
              'graph_nodes': str(Path(nodes_path).resolve()), 'graph_nodes_sha256': sha256_file(nodes_path),
              'output': str(output_path.resolve()), 'output_sha256': sha256_file(output_path),
              'schema_contract': schema, 'provenance': provenance,
              'drug_nodes': drugs.height, 'available_structured_smiles': result['structure_available'].sum(),
              'coverage_fraction': result['structure_available'].sum() / drugs.height,
              'availability_counts': dict(Counter(result['availability_reason'].to_list())),
              'chemistry_parser_validation': chemical_validation,
              'missing_policy': 'Absent structures use null SMILES. Parser-invalid source strings are retained verbatim for local audit with structure_available=false. No fabricated or inferred structures.',
              'privacy': 'Local licensed DrugBank input only; no network/upload, no SMILES listing in report',
              'labels_accessed_or_inferred': False, 'evaluation_data_accessed': False,
              'elapsed_seconds': time.perf_counter() - start}
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=Path, default=root / 'data/processed/reconstructed_v1/nodes.parquet')
    parser.add_argument('--xml', type=Path, default=root.parent / 'EviHGT/data/drugbank_all_full_database.xml')
    parser.add_argument('--xsd', type=Path, default=root.parent / 'EviHGT/data/drugbank.xsd')
    parser.add_argument('--output', type=Path, default=root / 'data/processed/reconstructed_v1/drug_smiles.parquet')
    parser.add_argument('--report', type=Path, default=root / 'outputs/reconstruction/smiles_manifest.json')
    args = parser.parse_args()
    reconstruct_smiles(args.nodes, args.xml, args.xsd, args.output, args.report)


if __name__ == '__main__':
    main()
