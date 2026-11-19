"""Reuse DEV95 measurement, with fixed hash-selected current-cell panels."""
import importlib.util
from dataclasses import asdict
from evi_hgt_dev95 import diagnostics as original
from .protocol import cell_spec,digest

isolated=original.isolated


def panel_indices(cell,groups):
    s=cell_spec(cell);g=s['G']
    key=lambda i:digest(['DEV96_TRAIN_PANEL_V1',cell,[asdict(q) for q in groups[i]]])
    return dict(global32=sorted(range(g),key=key)[:32],conditional32=sorted(range(g,s['total']),key=key)[:32])


def diagnose(cell,assembler,built,optimizer,groups,records,step,*,arm,panels=None,ui=None):
    s=cell_spec(cell)
    spec=importlib.util.spec_from_file_location('evi_hgt_dev95._dev96_diag_'+cell,original.__file__)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.G=s['G'];m.TOTAL=s['total']
    selected=panels or panel_indices(cell,groups)
    if step==0:selected={'global32':selected['global32']}
    r=m.diagnose(assembler,built,optimizer,groups,records,step,arm=arm,panels=selected,ui=ui)
    r['cell']=cell;return r
