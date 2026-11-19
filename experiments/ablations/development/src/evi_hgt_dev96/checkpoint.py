"""Private complete-state backend with DEV96 cell/loss and path boundaries."""
import importlib.util
from pathlib import Path
from evi_hgt_dev94 import checkpoint as original
from .protocol import OUT, ARMS, LOSSES, cell_spec

spec=importlib.util.spec_from_file_location('_dev96_complete_state',original.__file__)
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
backend.DEV_ROOT=OUT.resolve();backend.TEST_ROOT=(OUT/'TEST_ONLY').resolve();backend.ARMS=set(ARMS)
backend.MAX_UPDATES=20700;backend.SCHEMA='DEV96_COMPLETE_TRAINING_STATE_V1';backend.COMMIT_SCHEMA='DEV96_ATOMIC_CHECKPOINT_COMMIT_V1'
original_identity=backend._identity;original_sources=backend._implementation_sources;original_basic=backend._basic_integrity


def _identity(v):
    original_identity(v);cell_spec(v.get('cell'))
    if v.get('loss_id')!=LOSSES[v['arm']]:raise ValueError('DEV96 loss identity differs')


def _basic(s):
    original_basic(s)
    if s['step']>cell_spec(s['identity']['cell'])['total']:raise ValueError('Cell checkpoint budget exceeded')


def _sources():
    return {**original_sources(),str(Path(__file__).resolve().relative_to(backend.ROOT)).replace('\\','/'):backend.file_sha(__file__)}


backend._identity=_identity;backend._basic_integrity=_basic;backend._implementation_sources=_sources
Store,capture,restore,verify_commit=backend.Store,backend.capture,backend.restore,backend.verify_commit
