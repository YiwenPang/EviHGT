"""Deterministic, label-independent entity features. No fitted transforms."""
from __future__ import annotations

import hashlib
import numpy as np

DRUG_DIM = 2048 + 512 + 2
DISEASE_DIM = 512 + 512 + 2
PACKED_DIM = DRUG_DIM + DISEASE_DIM
FEATURE_VERSION = 'PHASE852_ENTITY_RAW_V1'
ALLOWED_SOURCES = frozenset({'canonical_smiles', 'drug_gene', 'drugbank_drug_gene',
                            'ontology_ancestors', 'disease_phenotype_HPO'})


def require_feature_source(source):
    if source not in ALLOWED_SOURCES:
        raise ValueError('unapproved entity feature source: ' + str(source))


def ontology_input(*, parents, ancestors):
    """A present empty source field is observed zero, not missing data."""
    available = parents is not None or ancestors is not None
    return sorted(set((parents or []) + (ancestors or []))), available


def fingerprint(smiles):
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    result = np.zeros(2048, dtype=np.uint8)
    if not isinstance(smiles, str) or not smiles.strip():
        return result, False
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return result, False
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result, True


def profile(tokens, namespace):
    """512 fixed binary buckets; duplicates/order do not change the vector."""
    result = np.zeros(512, dtype=np.uint8)
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError('profile requires nonempty canonical tokens')
        raw = (FEATURE_VERSION + '\x1f' + namespace + '\x1f' + token).encode()
        result[int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % 512] = 1
    return result


def _pack(a, b, available_a, available_b, width_a):
    if np.shape(a) != (width_a,) or np.shape(b) != (512,):
        raise ValueError('feature dimension mismatch')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('raw features must be finite binary values')
    if (not available_a and np.any(a)) or (not available_b and np.any(b)):
        raise ValueError('unavailable family cannot carry feature values')
    return np.concatenate([a, b, [bool(available_a), bool(available_b)]]).astype(np.uint8)


def pack_drug(morgan, biology, structure_available, biology_available):
    return _pack(morgan, biology, structure_available, biology_available, 2048)


def pack_disease(ontology, phenotype, ontology_available, phenotype_available):
    return _pack(ontology, phenotype, ontology_available, phenotype_available, 512)


class FeatureStore:
    """Separate mmap-able typed arrays; only a bounded query batch is expanded."""
    def __init__(self, drug_ids, drugs, disease_ids, diseases):
        self.drugs, self.diseases = drugs, diseases
        self.drug_rows = {int(v): i for i, v in enumerate(drug_ids)}
        self.disease_rows = {int(v): i for i, v in enumerate(disease_ids)}
        if len(self.drug_rows) != len(drug_ids) or len(self.disease_rows) != len(disease_ids):
            raise ValueError('duplicate feature entity')
        if self.drug_rows.keys() & self.disease_rows.keys():
            raise ValueError('feature entity type overlap')
        if drugs.shape != (len(drug_ids), DRUG_DIM) or diseases.shape != (len(disease_ids), DISEASE_DIM):
            raise ValueError('feature store dimension mismatch')

    def gather(self, global_ids, node_types):
        if len(global_ids) != len(node_types):
            raise ValueError('feature lookup shape mismatch')
        result = np.zeros((len(global_ids), PACKED_DIM), dtype=np.float32)
        for i, (gid, kind) in enumerate(zip(global_ids, node_types, strict=True)):
            if gid in self.drug_rows:
                if kind != 'DRG':
                    raise ValueError('drug feature node type mismatch')
                result[i, :DRUG_DIM] = self.drugs[self.drug_rows[gid]]
            elif gid in self.disease_rows:
                if kind != 'DIS':
                    raise ValueError('disease feature node type mismatch')
                result[i, DRUG_DIM:] = self.diseases[self.disease_rows[gid]]
            elif kind in {'DRG', 'DIS'}:
                raise ValueError('missing entity in feature index; source binding mismatch')
        return result
