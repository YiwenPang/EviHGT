"""Read only the frozen science document and original metadata, never labels."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
G1_PATH = ROOT / "outputs/phase9/phase92/G1_FROZEN_SCIENCE_PROTOCOL.md"
G1_SHA256 = "28685ba42122fd216c8a706d8cc0ddb96919ca9006bf33f86f82fc234845b8aa"
MODES = {"M0": "FULL_B", "M2": "CONTENT_ZERO", "M3": "FACT_QUALITY_CONSTANT"}


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def frozen_science():
    if file_sha(G1_PATH) != G1_SHA256:
        raise ValueError("Frozen G1 document changed")
    text = G1_PATH.read_text(encoding="utf-8")
    block = text.split("<!-- PHASE9_BINDINGS_BEGIN -->", 1)[1].split("<!-- PHASE9_BINDINGS_END -->", 1)[0]
    result = json.loads(block.split("```json", 1)[1].split("```", 1)[0])
    if result["g1_status"] != "G1_FROZEN" or any(result["gates"][g] for g in ("G3", "G4", "G5")):
        raise ValueError("Unexpected frozen scientific scope")
    return result


def verify_parent_sources():
    for row in frozen_science()["sources"]:
        if file_sha(row["path"]) != row["sha256"]:
            raise ValueError("Parent source/metadata identity changed: " + row["path"])


def require_access(domain):
    if domain not in {"synthetic_test", "static_metadata"}:
        raise PermissionError("G2 only; no real task training/scoring access: " + str(domain))


@dataclass(frozen=True)
class CellContract:
    method: str
    entry_json: str

    def __post_init__(self):
        if self.method not in MODES:
            raise ValueError("G2 model implementation only permits M0/M2/M3")
        entry = json.loads(self.entry_json)
        matches = [x for x in frozen_science()["cells"] if x["repeat_index"] == entry["repeat_index"]
                   and x["fold_index"] == entry["fold_index"]]
        if len(matches) != 1 or canonical(matches[0]) != canonical(entry):
            raise ValueError("Exact original cell bindings differ")

    @classmethod
    def load(cls, method, repeat, fold, *, seed=None, exact_bindings=None,
             graph_policy=None, backend=None, budget=None, flags=None):
        if type(repeat) is not int or type(fold) is not int:
            raise ValueError("Original integer repeat/fold required")
        science = frozen_science()
        candidates = [x for x in science["cells"] if x["repeat_index"] == repeat and x["fold_index"] == fold]
        if len(candidates) != 1:
            raise ValueError("Unknown original cell")
        entry = candidates[0]
        expected_flags = {"intervention": MODES.get(method), "direct_mass_flags": [False, False]}
        checks = [(seed, entry["training_seed"]), (exact_bindings, entry),
                  (graph_policy, science["graph_policy"]), (backend, science["backend"]),
                  (budget, entry["budget"]), (flags, expected_flags)]
        if any(value is not None and canonical(value) != canonical(expected) for value, expected in checks):
            raise ValueError("Explicit entry identity conflicts with frozen cell")
        # Recheck original census and source identities before a new receiver is constructed.
        verify_parent_sources()
        return cls(method, canonical(entry))

    @property
    def seed(self):
        return self.bindings["training_seed"]

    @property
    def bindings(self):
        return json.loads(self.entry_json)

    @property
    def budget(self):
        return self.bindings["budget"]

    def payload(self):
        science = frozen_science()
        return {"method": self.method, "cell": self.bindings, "g1_sha256": G1_SHA256,
                "graph_policy": science["graph_policy"], "backend": science["backend"],
                "intervention": MODES[self.method], "direct_mass_flags": [False, False]}
