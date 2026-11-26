"""Local structured SMILES extraction, preserved from the research generator.

No clinical label inference and no network access occur in this module.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

DB = '{http://www.drugbank.ca}'
XS = '{http://www.w3.org/2001/XMLSchema}'

RESEARCH_SOURCE_SHA256 = {'rebuild_smiles.py': 'a37930deaeb57f442ae8a5363e8f88e5c0b08f1d3b65da46f8cbc511f8aed39a', 'competition_features.py': 'dd4787d15900db5ef5c98c1762b0495709040d20301f5bbb700c95b6500ccca0'}

class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
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

def extract_candidate_smiles(xml_path, candidate_ids, *, progress=None):
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
    last_progress = time.monotonic()
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
                if progress and time.monotonic() - last_progress >= 5:
                    progress('流式提取 DrugBank 结构', reader.bytes_read, before.st_size, reader.bytes_read)
                    last_progress = time.monotonic()
            depth -= 1
        meta['xml_sha256'] = reader.digest.hexdigest()
    after = xml_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError('DrugBank XML changed during read-only extraction')
    meta['xml_size_bytes'] = before.st_size
    meta['source_unchanged_during_read'] = True
    if progress:
        progress('流式提取 DrugBank 结构', before.st_size, before.st_size, before.st_size)
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

def resolved_smiles(original, fallback_records):
    """Existing validated canonical structure wins; ambiguous fallback is absent."""
    if original and original.get('structure_available'):
        return original['smiles'], 'optimus_canonical_smiles', 'available'
    candidates = {r['smiles'] for r in fallback_records if r['structure_available']}
    if len(candidates) == 1:
        return next(iter(candidates)), 'drugbank_structured_smiles', 'available'
    if len(candidates) > 1:
        return None, None, 'conflicting_mapped_structures'
    return None, None, 'structured_smiles_unavailable'
