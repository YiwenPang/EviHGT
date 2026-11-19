"""Only current-cell role-restricted inputs and hash-certified fold corpora."""
from contextlib import contextmanager
from pathlib import Path
import polars as pl
from evi_hgt.ranking_protocol.real_data import RealInputs, load_index, load_canonical_assembler
from evi_hgt.query_model.data import assert_corpus_identity
from evi_hgt.graph_inputs.common import verify
from evi_hgt_feature_baseline.data import DevelopmentData
from .protocol import *


class Data(DevelopmentData):
    def __init__(self,original,cell):super().__init__(original);self.cell_id=cell

    @contextmanager
    def evaluation_scope(self,step):
        if step not in cell_spec(self.cell_id)['milestones'] or self._evaluation_step is not None:
            raise PermissionError('Only current-cell PAIRED_LOSSES evaluation milestones')
        self._evaluation_step=step
        try:yield
        finally:self._evaluation_step=None


def assert_query_graph_safe(graph,assembler,pair):
    source=graph.batch.path_source
    admitted={int(f.canonical_fact_id) for f in source.facts}
    forbidden=set(source.forbidden_fact_ids)|assembler.base.base.forbidden|assembler.base.base.target_ids.get(pair,set())
    if admitted&forbidden:raise ValueError('Query/reverse/equivalent/protected fact entered actual graph')
    return dict(facts=len(admitted),forbidden_fact_intersection=0)


def load_data(cell,shared=None):
    s=cell_spec(cell);c=frozen_cell(cell);d=Data((shared or RealInputs()).cell(s['repeat'],s['fold']),cell)
    if (d.training_seed!=s['seed'] or d.budget()!=c['budget'] or d.legal_u_domain()!=c['legal_training_u_domain']
        or digest(sorted(d.blocked))!=c['mask_digest']):raise ValueError('Current-cell seed/budget/U/mask differs')
    roots=[d.original_corpus_root,ROOT/f"outputs/phase9/phase92b/test_only/r{s['repeat']}_f{s['fold']}/corpus"]
    found=[p for p in roots if (p/'manifest.json').is_file()]
    if not found:raise ValueError('Certified current-cell corpus unavailable; no cross-fold fallback')
    d.corpus_root=found[0];corpus=verify(read(d.corpus_root/'manifest.json'));assert_corpus_identity(d,corpus)
    for p,h in {**corpus['inputs'],**corpus['outputs']}.items():
        if file_sha(p)!=h:raise ValueError('Certified cell corpus bytes changed: '+p)
    # Actual DDA identities and evidence must remain within this cell boundary.
    dda=next(x for x in corpus['fact_files'] if x['component']=='drug_disease')
    frame=pl.read_parquet(dda['path'],columns=['source_node_id','target_node_id','canonical_fact_id'])
    pairs=set(zip(frame['source_node_id'],frame['target_node_id']))
    if pairs & d.blocked:raise ValueError('Protected/current-fold forbidden DDA in graph')
    ev=next(x for x in corpus['evidence_files'] if x['component']=='drug_disease')
    if not set(pl.read_parquet(ev['path'],columns=['canonical_fact_id']).to_series()).issubset(set(frame['canonical_fact_id'])):
        raise ValueError('Evidence includes a fact absent from current-cell DDA graph')
    return d,corpus
