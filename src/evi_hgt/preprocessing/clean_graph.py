"""Physically remove target aliases and contaminated disease-gene evidence."""
from pathlib import Path
import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .clean_graph_source import clean_source_rows, aggregate_clean_edges, mask_endpoint_pairs


def build_clean_graph(canonical_root, splits_root, official_parquet_paths, policy, destination,
                      *, audit_dir, progress=None):
    canonical_root, splits_root, destination, audit_dir = map(Path,
        (canonical_root, splits_root, destination, audit_dir))
    if destination.exists():
        raise FileExistsError('Choose a new clean graph destination')
    if not official_parquet_paths:
        raise ValueError('Verified Open Targets 25.06 datasource files are required')
    if progress:
        progress('按来源清理疾病—基因证据', 0, 1, 0)
    source_rows = clean_source_rows(pl.read_parquet(official_parquet_paths), policy).filter(
        ~pl.col('diseaseId').str.contains('HP'), ~pl.col('diseaseId').str.starts_with('GO_'),
        ~pl.col('diseaseId').str.starts_with('UBERON_'))
    edges = aggregate_clean_edges(source_rows)
    original = pl.read_parquet(canonical_root / 'canonical_facts/disease_gene.parquet')
    clean = original.join(edges.select(pl.col('disease_id').alias('source_node_id'),
        pl.col('gene_id').alias('target_node_id')), on=['source_node_id', 'target_node_id'], how='semi')
    if clean.height != edges.height:
        raise ValueError('Clean endpoints do not map one-to-one to canonical disease-gene facts')
    edges = edges.join(clean.select('canonical_fact_id', pl.col('source_node_id').alias('disease_id'),
        pl.col('target_node_id').alias('gene_id')), on=['disease_id', 'gene_id'], validate='1:1').sort('disease_id', 'gene_id')
    clean = clean.with_columns(
        pl.lit('phase85_part2b/LEAKAGE_SAFE_OT_DISEASE_GENE_V1').alias('source_component_path'),
        pl.lit(1, dtype=pl.UInt32).alias('aggregated_observation_count')).sort('canonical_fact_id')
    properties = edges.select('canonical_fact_id', pl.struct(
        pl.col('source_count'), pl.col('evidence_count'), pl.col('source_scores').alias('clean_source_scores'),
        pl.col('allowed_datasource_ids').alias('clean_datasources'), pl.lit('25.06').alias('source_release'),
        pl.col('association_semantics')).struct.json_encode().alias('properties'),
        pl.col('allowed_datasource_ids').alias('raw_indirect_sources'))
    evidence = clean.join(properties, on='canonical_fact_id', validate='1:1').select(
        (pl.col('canonical_fact_id') + 1000000000).cast(pl.UInt64).alias('evidence_record_id'),
        'canonical_fact_id', 'canonical_fact_key', 'source_component_path',
        pl.int_range(0, pl.len(), dtype=pl.UInt32).alias('source_row_number'), 'relation', 'properties',
        pl.lit(['OPEN_TARGETS']).alias('raw_direct_sources'), 'raw_indirect_sources',
        pl.lit(False).alias('edge_support_observed'), pl.lit(False).alias('edge_counter_evidence_observed'),
        pl.lit(None, dtype=pl.String).alias('edge_evidence_proposition'),
        pl.lit(None, dtype=pl.String).alias('overlap_category'), pl.lit(False).alias('independent_vote_eligible'))
    targets = pl.read_parquet(canonical_root / 'views/query_target_fact_map.parquet')
    blocked_frame = pl.read_parquet(splits_root / 'blocked_pairs.parquet')
    blocked = set(blocked_frame.iter_rows())
    forbidden = targets.join(blocked_frame, on=['drug_id', 'disease_id'], how='semi')['canonical_fact_id'].unique()
    legal = pl.read_parquet(splits_root / 'legal_source_rows.parquet').join(blocked_frame,
        on=['drug_id', 'disease_id'], how='anti')
    destination.mkdir(parents=True)
    def write(name, table):
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)
        return path
    forbidden_path = write('forbidden_fact_ids.parquet', pl.DataFrame({'canonical_fact_id': forbidden}))
    fact_files, evidence_files = [], []
    files = sorted((canonical_root / 'canonical_facts').glob('*.parquet'))
    for i, path in enumerate(files):
        ep = canonical_root / 'evidence_records' / path.name
        if path.stem == 'disease_gene':
            fp = write('canonical_facts/' + path.name, clean)
            ep = write('evidence_records/' + path.name, evidence)
        else:
            identity = pl.read_parquet(path, columns=['canonical_fact_id', 'source_node_id', 'target_node_id',
                'source_node_type', 'target_node_type'])
            safe = mask_endpoint_pairs(identity, blocked).filter(~pl.col('canonical_fact_id').is_in(forbidden.implode()))
            if path.stem == 'drug_disease':
                safe = safe.join(legal.rename({'drug_id': 'source_node_id', 'disease_id': 'target_node_id'}),
                    on=['source_node_id', 'target_node_id'], how='semi')
            fp = path
            if safe.height != identity.height or path.stem == 'drug_disease':
                selected = safe.lazy().select('canonical_fact_id')
                fp = write('canonical_facts/' + path.name,
                    pl.scan_parquet(path).join(selected, on='canonical_fact_id', how='semi').collect())
                ep = write('evidence_records/' + path.name,
                    pl.scan_parquet(ep).join(selected, on='canonical_fact_id', how='semi').collect())
        fact_files.append(dict(path=str(fp.resolve()), component=path.stem, rows=pq.ParquetFile(fp).metadata.num_rows))
        evidence_files.append(dict(path=str(ep.resolve()), component=path.stem, rows=pq.ParquetFile(ep).metadata.num_rows))
        if progress:
            progress('按划分屏蔽目标事实', i + 1, len(files), 0)
    admitted = np.concatenate([pl.read_parquet(v['path'], columns=['canonical_fact_id']).to_series().to_numpy() for v in fact_files])
    if len(np.unique(admitted)) != len(admitted) or np.isin(admitted, forbidden.to_numpy()).any():
        raise ValueError('Duplicate or forbidden fact survived physical graph masking')
    original_csr = {name: np.load(canonical_root / 'graph/csr' / (name + '.npy'), mmap_mode='r')
                    for name in ('indptr', 'indices', 'fact_ids', 'edge_type_ids')}
    keep = np.isin(original_csr['fact_ids'], admitted)
    prefix = np.empty(len(keep) + 1, dtype=np.int64); prefix[0] = 0
    np.cumsum(keep, dtype=np.int64, out=prefix[1:])
    csr = {'indptr': prefix[original_csr['indptr']]}
    csr.update({name: original_csr[name][keep] for name in ('indices', 'fact_ids', 'edge_type_ids')})
    paths = {}
    for name, value in csr.items():
        path = destination / 'csr' / (name + '.npy'); path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False); paths[name] = str(path.resolve())
    target_path = write('legal_target_map.parquet', targets.filter(
        pl.col('canonical_fact_id').is_in(pl.Series(admitted).implode())
    ).select('drug_id', 'disease_id', 'canonical_fact_id'))
    result = dict(schema='EVIHGT_REBUILT_CLEAN_GRAPH_V1', fact_files=fact_files, evidence_files=evidence_files,
        csr=paths, target_map=str(target_path.resolve()), forbidden_facts=str(forbidden_path.resolve()),
        contaminated_DG_evidence_imported=False, heldout_DDA_properties_imported=False,
        clean_DG_facts=clean.height, traversal_count=len(csr['fact_ids']),
        nodes=str((canonical_root / 'nodes/node_index.parquet').resolve()),
        edge_catalog=str((canonical_root / 'graph/csr/edge_type_catalog.json').resolve()))
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'graph_reconstruction.json').write_text(json.dumps(dict(status='PASS',
        clean_DG_facts=clean.height, clean_DG_evidence=evidence.height,
        target_facts_removed=len(forbidden), traversal_count=result['traversal_count'],
        datasource_policy_validated=True, original_DG_properties_reused=False,
        target_reverse_equivalent_endpoint_mask_applied=True), indent=2), encoding='utf-8')
    return result
