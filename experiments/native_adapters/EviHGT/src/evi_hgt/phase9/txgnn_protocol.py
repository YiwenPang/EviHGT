"""Fixed M4 scientific/control contract. This module grants no execution."""
import copy
import math

SELECTION_EPOCHS = tuple(range(25, 501, 25))

def fixed_recipe(seed):
    if type(seed) is not int or not 0 <= seed < 2**32 - 1001:
        raise ValueError('Original cell seed required')
    return dict(dimension=100, pretrain_epochs=1, pretrain_lr=.001,
                pretrain_batch_size=65536, finetune_lr=.003, finetune_epochs=500,
                optimizer='AdamW', weight_decay=.01, betas=[.9,.999], eps=1e-8,
                scheduler=dict(name='ReduceLROnPlateau',mode='min',factor=.8,patience=10,
                               threshold=1e-4,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-8,
                               stepped_on='training_loss_each_finetune_epoch'),
                seed=seed, pretrain_seed=seed+11, pretrain_U_seed=seed+102,
                decoder_seed=seed+1001, finetune_U_seed=seed+103,
                selection_epochs=list(SELECTION_EPOCHS),
                selection_key=['support_disease_macro_ROC_AUC','support_AP','earlier_epoch'],
                pretrain_commit_every_batches=32,
                finetune_commit='PRE_INNER_AND_INNER_COMPLETE_AT_25_EPOCHS',
                encoder='native_HeteroRGCN_2_layer', decoder='native_DistMult',
                prototype=dict(k=3,similarity='all_nodes_profile',aggregation='rarity',exp_lambda=.7),
                device='cpu', threads=8, scope='FIXED_NATIVE_SINGLE_LR_RECIPE_NO_EXECUTION_AUTHORITY')

def planned_updates(pretrain_edge_count):
    if type(pretrain_edge_count) is not int or pretrain_edge_count <= 0:
        raise ValueError('Actual legal all-relation edge count required')
    return math.ceil(pretrain_edge_count/65536)+500

def pretrain_commit_due(batch, planned_batches):
    if type(batch) is not int or type(planned_batches) is not int or not 1<=batch<=planned_batches:
        raise ValueError('Invalid pretrain batch cursor')
    return batch % 32 == 0 or batch == planned_batches

class InnerSelection:
    def __init__(self):
        self.history=[]
        self.best=None

    def observe(self, epoch, metrics):
        if len(self.history)>=len(SELECTION_EPOCHS) or epoch!=SELECTION_EPOCHS[len(self.history)]:
            raise ValueError('Exactly the next fixed inner opportunity is required')
        values=[metrics.get('macro_auc'),metrics.get('macro_ap')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Finite disease-macro AUC and AP required')
        row=dict(epoch=epoch,macro_auc=float(values[0]),macro_ap=float(values[1]))
        key=lambda x:(x['macro_auc'],x['macro_ap'],-x['epoch'])
        improved=self.best is None or key(row)>key(self.best)
        self.history.append(row)
        if improved:self.best=copy.deepcopy(row)
        return improved

    def state(self):
        return copy.deepcopy(dict(history=self.history,best=self.best))

    @classmethod
    def from_state(cls,state):
        if not isinstance(state,dict) or set(state)!={'history','best'}:
            raise ValueError('Invalid selection state')
        obj=cls()
        for row in state['history']:
            if set(row)!={'epoch','macro_auc','macro_ap'}:raise ValueError('Invalid selection history')
            obj.observe(row['epoch'],row)
        if obj.state()!=state:raise ValueError('Stored selected checkpoint contradicts fixed history')
        return obj

def require_endpoint(*,pretrain_batches,planned_batches,finetune_epoch,selection_epochs,decoder_resets):
    if (type(planned_batches) is not int or planned_batches<=0 or pretrain_batches!=planned_batches
            or finetune_epoch!=500 or selection_epochs!=list(SELECTION_EPOCHS) or decoder_resets!=1):
        raise ValueError('M4 exact two-stage budget, decoder reset or selection endpoint incomplete')
    return planned_batches+500
