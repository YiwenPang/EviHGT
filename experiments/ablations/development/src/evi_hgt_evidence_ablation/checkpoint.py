"""Private instance of the accepted atomic full-state backend."""
import ablation_artifacts as artifacts
import importlib.util
from pathlib import Path
from evi_hgt_feature_baseline import checkpoint as original
from .protocol import OUT,ARMS,LOSSES,cell_spec

spec=importlib.util.spec_from_file_location(f"{original.__package__}._evidence_checkpoint",original.__file__)
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
backend.DEV_ROOT=OUT.resolve();backend.TEST_ROOT=(OUT/'TEST_ONLY').resolve();backend.ARMS=set(ARMS)
backend.MAX_UPDATES=20700;backend.SCHEMA=artifacts.EVIDENCE_STATE_SCHEMA;backend.COMMIT_SCHEMA=artifacts.EVIDENCE_COMMIT_SCHEMA
original_identity=backend._identity;original_basic=backend._basic_integrity;original_sources=backend._implementation_sources


def _identity(v):
    original_identity(v);cell_spec(v.get('cell'))
    if v.get('ablation')!=v['arm'] or v.get('loss_id')!=LOSSES[v['arm']]:raise ValueError('EVIDENCE_ABLATION ablation/loss/cell identity changed')


def _basic(s):
    original_basic(s)
    if s['step']>cell_spec(s['identity']['cell'])['total']:raise ValueError('Cell budget exceeded')


def _sources():
    return {**original_sources(),str(Path(__file__).resolve().relative_to(backend.ROOT)).replace('\\','/'):backend.file_sha(__file__)}


backend._identity=_identity;backend._basic_integrity=_basic;backend._implementation_sources=_sources
Store,capture,restore,verify_commit=backend.Store,backend.capture,backend.restore,backend.verify_commit
