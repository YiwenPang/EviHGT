"""Check raw database filenames and identities without modifying their contents."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
AGGREGATE = Path("data/cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0/recovery/aggregate")
OPTIMUS = {
    "nodes.parquet": (163857672, "0601aaf9a659ca0f89aa9ab8b8baa2d3318b2d21e1d93d678d53e800a86eddad"),
    "edges.parquet": (177526798, "2a551d832c7d10758d2d5a6a636fb8eef863beeb65967b74546385ba3e7f6940"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_raw_inputs(root: Path) -> list[tuple[bool, str]]:
    results = []
    for name, (size, digest) in OPTIMUS.items():
        path = root / AGGREGATE / name
        if not path.is_file():
            results.append((False, f"Missing file: {path}"))
        elif path.stat().st_size != size or sha256(path) != digest:
            results.append((False, f"Does not match OptimusKG release 2.0: {path}"))
        else:
            results.append((True, f"OptimusKG release 2.0 verified: {path}"))
    for name, expected_tag in (
        ("drugbank_all_full_database.xml", "{http://www.drugbank.ca}drugbank"),
        ("drugbank.xsd", "{http://www.w3.org/2001/XMLSchema}schema"),
    ):
        path = root / "data" / name
        if not path.is_file():
            results.append((False, f"Missing file: {path}"))
            continue
        try:
            with path.open("rb") as source:
                _, element = next(ET.iterparse(source, events=("start",)))
                if element.tag != expected_tag:
                    raise ValueError("unexpected XML root; check the extracted file type")
                metadata = ""
                if name.endswith(".xml"):
                    metadata = (f" (version={element.get('version', 'not recorded')}, "
                                f"exported-on={element.get('exported-on', 'not recorded')})")
                results.append((True, f"XML header and filename verified: {path}{metadata}"))
        except (OSError, ET.ParseError, StopIteration, ValueError) as error:
            results.append((False, f"Cannot read the expected XML header: {path}: {error}"))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Repository or research-workspace root (default: this repository)")
    args = parser.parse_args(argv)
    results = check_raw_inputs(args.root.resolve())
    for passed, message in results:
        print(("PASS " if passed else "FAIL ") + message)
    print("This check verifies raw input placement, OptimusKG hashes, and DrugBank XML headers.")
    print("Training requires data/processed/corpus/manifest.json; see docs/data.md.")
    return 0 if all(passed for passed, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
