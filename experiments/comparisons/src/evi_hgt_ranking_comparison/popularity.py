"""Training-only per-channel degree control. No implicit data loading."""
from .protocol_ids import POPULARITY_SCHEMA
from collections import Counter
from dataclasses import dataclass
import hashlib,json,math

def _pairs(values):
    out=set()
    for row in values:
        if isinstance(row,dict):
            if row.get('role','P')!='P':raise ValueError('Only verified training positives may contribute')
            p=(row['drug_id'],row['disease_id'])
        else:p=tuple(row)
        if len(p)!=2 or not all(isinstance(x,str) and x for x in p):raise ValueError('Canonical string pair required')
        out.add(p)
    return out

def member_digest(values):
    raw=json.dumps(sorted(_pairs(values)),ensure_ascii=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class Popularity:
    cell:str
    support:dict
    contra:dict
    identity:dict

    @classmethod
    def build(cls,*,cell,support,contra,expected_support_digest,expected_contra_digest,forbidden_pairs):
        if cell not in {f'R{r}F{f}' for r in range(3) for f in range(5)}:raise ValueError('Unregistered cell')
        s,c=_pairs(support),_pairs(contra);forbidden=_pairs(forbidden_pairs)
        if member_digest(s)!=expected_support_digest or member_digest(c)!=expected_contra_digest:
            raise ValueError('Training-P membership identity mismatch')
        if (s|c)&forbidden:raise ValueError('Forbidden/held-out pair cannot contribute degree')
        sd,cd=dict(Counter(d for d,_ in s)),dict(Counter(d for d,_ in c))
        ident=dict(schema=POPULARITY_SCHEMA,cell=cell,support_members=expected_support_digest,
            contra_members=expected_contra_digest,support_unique_pairs=len(s),contra_unique_pairs=len(c),
            score='log1p(per-head unique training-P drug degree)',unknown_drug=0,optimizer_steps=0,
            ranking_tie='canonical drug ID; threshold metrics keep whole score groups')
        return cls(cell,sd,cd,ident)

    def score(self,drug_id,head):
        if head not in ('support','contra','unsuitability'):raise ValueError('Independent head required')
        return math.log1p((self.support if head=='support' else self.contra).get(drug_id,0))

    def score_rows(self,members,*,cell):
        if cell!=self.cell:raise ValueError('Cross-cell popularity use')
        rows=[];seen=set()
        for r in members:
            pair=(r['drug_id'],r['disease_id'])
            if pair in seen:raise ValueError('Duplicate candidate pair')
            seen.add(pair)
            rows.append(dict(r,support_logit=self.score(pair[0],'support'),unsuitability_logit=self.score(pair[0],'contra')))
        return rows
