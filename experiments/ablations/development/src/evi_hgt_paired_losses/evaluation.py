"""Instance-local LOSS_COMPARISON cache/evaluation with current-cell namespace and gates."""
import importlib.util
from evi_hgt_loss_comparison import evaluation as original
from . import protocol as p
from . import checkpoint as cp


def facility(cell,*,test_only=False):
    p.cell_spec(cell)
    spec=importlib.util.spec_from_file_location(f"{original.__package__}._paired_evaluation_{cell}",original.__file__)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=p.OUT/('TEST_ONLY' if test_only else '')/cell
    m.write=p.write;m.journal=p.journal;m.Store=cp.Store;m.verify_commit=cp.verify_commit
    m.check_evaluation=lambda step,domain:p.check_evaluation(cell,step,domain)
    m.validate_manifest=lambda:dict(p.validate_manifest(),**p.read(p.MANIFEST)['cells'][cell])
    m.identity=lambda manifest,arm:p.identity(manifest,cell,arm)
    return m


def evaluate(cell,*args,**kwargs):return facility(cell).evaluate(*args,**kwargs)
def verify_result(cell,*args,**kwargs):return facility(cell).verify_result(*args,**kwargs)
