"""Verified Open Targets policy and endpoint masking primitives from the research implementation."""
import polars as pl
def clean_source_rows(frame: pl.DataFrame, policy: dict) -> pl.DataFrame:
    observed = set(frame['datasourceId'].unique())
    if observed - policy.keys():
        raise ValueError('unclassified datasource; FAIL CLOSED')
    for source, datatype in frame.select('datasourceId', 'datatypeId').unique().iter_rows():
        if policy[source]['datatype_id'] != datatype:
            raise ValueError('datasource datatype mismatch; FAIL CLOSED')
    if frame.filter(~pl.col('score').is_finite() | pl.col('score').is_null()
                    | (pl.col('score') < 0) | (pl.col('score') > 1)).height:
        raise ValueError('invalid datasource score; FAIL CLOSED')
    if frame.select('datasourceId', 'diseaseId', 'targetId').n_unique() != frame.height:
        raise ValueError('duplicate datasource pair; FAIL CLOSED')
    if frame.filter(pl.any_horizontal(pl.all().is_null()) | (pl.col('evidenceCount') <= 0)).height:
        raise ValueError('missing provenance or invalid evidence count; FAIL CLOSED')
    return frame.filter(pl.col('datasourceId').is_in([s for s, p in policy.items() if p['eligible']]))

def aggregate_clean_edges(frame: pl.DataFrame) -> pl.DataFrame:
    # Presence means a source reports the association; it is not a probability
    # or an explicit support/refutation label. Never import overall/DisGeNET props.
    return (frame.sort('diseaseId', 'targetId', 'datasourceId')
        .group_by('diseaseId', 'targetId').agg(
            pl.col('datasourceId').alias('allowed_datasource_ids'),
            pl.col('datatypeId').alias('allowed_datatype_ids'),
            pl.col('score').alias('source_scores'),
            pl.col('evidenceCount').alias('source_evidence_counts'),
            pl.col('evidenceCount').sum().alias('evidence_count'),
            pl.len().alias('source_count'))
        .rename({'diseaseId': 'disease_id', 'targetId': 'gene_id'})
        .with_columns(pl.lit(1.0).alias('clean_presence'),
            pl.lit('INDIRECT_DISEASE_ONTOLOGY_ANCESTOR_PROPAGATION').alias('association_semantics'),
            pl.lit('Open Targets 25.06 association_by_datasource_indirect').alias('source_provenance'))
        .sort('disease_id', 'gene_id'))
def mask_endpoint_pairs(frame, forbidden):
    if not forbidden:
        return frame
    blocked = pl.DataFrame(sorted(forbidden), schema=['_drug', '_disease'], orient='row')
    typed = frame.with_columns(
        pl.when(pl.col('source_node_type') == 'DRG').then(pl.col('source_node_id'))
        .when(pl.col('target_node_type') == 'DRG').then(pl.col('target_node_id')).otherwise(None).alias('_drug'),
        pl.when(pl.col('source_node_type') == 'DIS').then(pl.col('source_node_id'))
        .when(pl.col('target_node_type') == 'DIS').then(pl.col('target_node_id')).otherwise(None).alias('_disease'))
    return typed.join(blocked, on=['_drug', '_disease'], how='anti').drop('_drug', '_disease')
