"""D1 unchanged construction, then validated Phase9 instance hooks."""
from evi_hgt.phase9.model import _ZeroContentInput as ZeroContentInput, _zero_provenance as zero_provenance, _constant_quality as constant_quality
from evi_hgt_dev96.model import CellD1, make_optimizer

ARMS=('A_EP_ZERO','A_QUALITY_CONST')


def content_layout(cfg):
    first=2*cfg.num_numeric_features+2*cfg.num_boolean_features+cfg.evidence_dim
    return first,first+cfg.evidence_dim+cfg.provenance_dim


class AblationD1(CellD1):
    def __init__(self,cell,pretrained,ablation=None):
        if ablation not in (*ARMS,None):raise PermissionError('Only two core ablations or TEST_ONLY reference')
        super().__init__(cell,pretrained)
        self.ablation=ablation;self.interventions=[]
        enc=self.model.encoder;cfg=enc.config;first,last=content_layout(cfg)
        if cfg.evidence_dim!=64 or cfg.provenance_dim!=16 or enc.evidence_encoder.encoder[0].in_features!=last+cfg.relation_dim:
            raise ValueError('Actual evidence concat schema changed')
        if ablation==ARMS[0]:
            module=enc.evidence_encoder.encoder;callback=ZeroContentInput(first,last)
            self.interventions.append((module,'_forward_pre_hooks',module.register_forward_pre_hook(callback),callback))
            module=enc.evidence_encoder.provenance_encoder;callback=zero_provenance
            self.interventions.append((module,'_forward_hooks',module.register_forward_hook(callback),callback))
        elif ablation==ARMS[1]:
            module=enc.logic;callback=constant_quality
            self.interventions.append((module,'_forward_hooks',module.register_forward_hook(callback),callback))

    def metadata(self):
        m=super().metadata()
        for module,kind,handle,callback in self.interventions:
            if getattr(module,kind).get(handle.id) is not callback:raise ValueError('Ablation hook missing')
        if self.ablation is not None:m=dict(m,dev97_ablation=self.ablation,intervention_boundary='PHASE92_M2' if self.ablation==ARMS[0] else 'PHASE92_M3')
        return m


def build_arm(cell,pretrained,ablation):return AblationD1(cell,pretrained,ablation)
