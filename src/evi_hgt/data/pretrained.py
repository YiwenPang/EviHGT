"""Frozen, label-independent D1 vectors; missing vectors have a false mask."""
import numpy as np
import torch

from .common import read_json

NODE_TYPES = ('ANA', 'BPO', 'CCO', 'DIS', 'DRG', 'EXP', 'GEN', 'MFN', 'PHE', 'PWY')
ENCODER_SPECS = {
    'chemberta': {'repo': 'seyonec/ChemBERTa-zinc-base-v1',
        'revision': '761d6a18cf99db371e0b43baf3e2d21b3e865a20', 'max_length': 512},
    'biobert': {'repo': 'dmis-lab/biobert-v1.1',
        'revision': '551ca18efd7f052c8dfa0b01c94c2a8e68bc5488', 'max_length': 128},
}


class PretrainedStore:
    def __init__(self, nodes, paths, resolve):
        self.metadata = read_json(resolve(paths['metadata']))
        metadata = self.metadata
        if metadata.get('schema') != 'EVIHGT_D1_FEATURES_V1':
            raise ValueError('D1 metadata must use EVIHGT_D1_FEATURES_V1')
        for name in ('clinical_text_encoded', 'supervised_dda_features_present',
                     'trained_node_embeddings_loaded', 'prediction_scores_loaded'):
            if metadata.get(name) is not False:
                raise ValueError('D1 features require an explicit false declaration: ' + name)
        if metadata.get('optimizer_steps') != 0 or metadata.get('width') != 768:
            raise ValueError('D1 requires frozen 768-dimensional inference vectors')
        for key, expected in ENCODER_SPECS.items():
            actual = metadata.get('fixed_encoder_specs', {}).get(key, {})
            if any(actual.get(k) != value for k, value in expected.items()):
                raise ValueError('D1 encoder source/revision differs: ' + key)
        self.kind_by_global = np.array([NODE_TYPES.index(k) for k in nodes['node_type']], dtype=np.int16)
        self.arrays = {}
        for kind, prefix in (('DRG', 'drug'), ('DIS', 'disease')):
            ids = np.load(resolve(paths[prefix + '_ids']), mmap_mode='r', allow_pickle=False)
            values = np.load(resolve(paths[prefix + '_values']), mmap_mode='r', allow_pickle=False)
            mask = np.load(resolve(paths[prefix + '_available']), mmap_mode='r', allow_pickle=False)
            expected_ids = nodes.filter(nodes['node_type'] == kind)['global_node_id'].to_numpy()
            if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected_ids):
                raise ValueError('D1 IDs must match sorted canonical entity IDs: ' + kind)
            if values.shape != (len(ids), 768) or values.dtype != np.float32:
                raise ValueError('D1 values require float32 [entity_count,768]')
            if mask.shape != (len(ids),) or mask.dtype != np.bool_:
                raise ValueError('D1 availability requires one explicit Boolean per entity')
            for start in range(0, len(ids), 4096):
                chunk = values[start:start+4096]
                if not np.isfinite(chunk).all() or np.any(chunk[~mask[start:start+4096]]):
                    raise ValueError('Nonfinite D1 vector or nonzero unavailable vector')
            self.arrays[kind] = ids, values, mask

    def gather(self, global_ids, node_type_ids, drug_type=4, disease_type=3, device='cpu'):
        if (drug_type, disease_type) != (4, 3):
            raise ValueError('D1 node type order differs from the canonical catalog')
        ids = global_ids.detach().cpu().numpy() if torch.is_tensor(global_ids) else np.asarray(global_ids)
        types = node_type_ids.detach().cpu().numpy() if torch.is_tensor(node_type_ids) else np.asarray(node_type_ids)
        if ids.ndim != 1 or types.shape != ids.shape or ids.dtype.kind not in 'iu' or types.dtype.kind not in 'iu':
            raise ValueError('D1 lookup requires equally sized integer ID/type vectors')
        if len(ids) and (ids.min() < 0 or ids.max() >= len(self.kind_by_global)):
            raise ValueError('Unknown D1 global entity ID')
        if not np.array_equal(types, self.kind_by_global[ids]):
            raise ValueError('Global entity/model node-type mismatch')
        result = np.zeros((len(ids), 768), dtype=np.float32)
        available = np.zeros(len(ids), dtype=np.bool_)
        for kind, type_id in (('DRG', drug_type), ('DIS', disease_type)):
            positions = np.flatnonzero(types == type_id)
            known, values, mask = self.arrays[kind]
            rows = np.searchsorted(known, ids[positions])
            if np.any(rows >= len(known)) or not np.array_equal(known[rows], ids[positions]):
                raise ValueError('D1 entity row is missing')
            result[positions], available[positions] = values[rows], mask[rows]
        return torch.from_numpy(result).to(device), torch.from_numpy(available).to(device)
