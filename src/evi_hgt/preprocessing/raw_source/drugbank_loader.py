"""Streaming extraction of verified structured fields from DrugBank XML."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree


DRUGBANK_NAMESPACE = "http://www.drugbank.ca"
_Q = f"{{{DRUGBANK_NAMESPACE}}}"
_PARTNER_COLLECTIONS = (
    ("targets", "target"),
    ("enzymes", "enzyme"),
    ("carriers", "carrier"),
    ("transporters", "transporter"),
)


@dataclass(frozen=True, slots=True)
class ExternalIdentifier:
    resource: str
    identifier: str


@dataclass(frozen=True, slots=True)
class ATCLevel:
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class ATCCode:
    code: str
    levels: tuple[ATCLevel, ...]


@dataclass(frozen=True, slots=True)
class DrugInteraction:
    drugbank_id: str
    name: str | None
    description: str | None


@dataclass(frozen=True, slots=True)
class PartnerRecord:
    category: str
    partner_id: str | None
    name: str | None
    organism: str | None
    actions: tuple[str, ...]
    known_action: str | None
    reference_ids: tuple[str, ...]
    polypeptide_id: str | None
    polypeptide_source: str | None
    polypeptide_name: str | None
    gene_name: str | None
    polypeptide_organism: str | None
    uniprot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DrugBankRecord:
    primary_drugbank_id: str
    all_drugbank_ids: tuple[str, ...]
    name: str | None
    drug_type: str | None
    groups: tuple[str, ...]
    atc_codes: tuple[ATCCode, ...]
    external_identifiers: tuple[ExternalIdentifier, ...]
    drug_interactions: tuple[DrugInteraction, ...]
    partners: tuple[PartnerRecord, ...]


@dataclass(frozen=True, slots=True)
class DrugBankConversionSummary:
    source_xml: Path
    output_dir: Path
    drug_count: int
    atc_row_count: int
    external_identifier_row_count: int
    drug_interaction_row_count: int
    partner_row_count: int


def _text(element: etree._Element, child_name: str) -> str | None:
    child = element.find(f"{_Q}{child_name}")
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _reference_ids(partner: etree._Element) -> tuple[str, ...]:
    values: list[str] = []
    references = partner.find(f"{_Q}references")
    if references is None:
        return ()
    articles = references.find(f"{_Q}articles")
    if articles is None:
        return ()
    for article in articles.findall(f"{_Q}article"):
        pubmed_id = _text(article, "pubmed-id")
        ref_id = _text(article, "ref-id")
        if pubmed_id:
            values.append(f"PUBMED:{pubmed_id}")
        if ref_id:
            values.append(f"DRUGBANK_REF:{ref_id}")
    return _unique(values)


def _uniprot_ids(polypeptide: etree._Element) -> tuple[str, ...]:
    values: list[str] = []
    source = (polypeptide.get("source") or "").casefold()
    polypeptide_id = (polypeptide.get("id") or "").strip()
    if source in {"swiss-prot", "uniprot", "uniprotkb"} and polypeptide_id:
        values.append(polypeptide_id)

    external = polypeptide.find(f"{_Q}external-identifiers")
    if external is not None:
        for item in external.findall(f"{_Q}external-identifier"):
            resource = _text(item, "resource")
            identifier = _text(item, "identifier")
            if resource and "uniprot" in resource.casefold() and identifier:
                values.append(identifier)
    return _unique(values)


def _parse_partners(drug: etree._Element) -> tuple[PartnerRecord, ...]:
    records: list[PartnerRecord] = []
    for collection_name, item_name in _PARTNER_COLLECTIONS:
        collection = drug.find(f"{_Q}{collection_name}")
        if collection is None:
            continue
        for partner in collection.findall(f"{_Q}{item_name}"):
            actions_element = partner.find(f"{_Q}actions")
            actions = (
                _unique(
                    [
                        (action.text or "").strip()
                        for action in actions_element.findall(f"{_Q}action")
                    ]
                )
                if actions_element is not None
                else ()
            )
            shared = {
                "category": item_name,
                "partner_id": _text(partner, "id"),
                "name": _text(partner, "name"),
                "organism": _text(partner, "organism"),
                "actions": actions,
                "known_action": _text(partner, "known-action"),
                "reference_ids": _reference_ids(partner),
            }
            polypeptides = partner.findall(f"{_Q}polypeptide")
            if not polypeptides:
                records.append(
                    PartnerRecord(
                        **shared,
                        polypeptide_id=None,
                        polypeptide_source=None,
                        polypeptide_name=None,
                        gene_name=None,
                        polypeptide_organism=None,
                        uniprot_ids=(),
                    )
                )
                continue
            for polypeptide in polypeptides:
                records.append(
                    PartnerRecord(
                        **shared,
                        polypeptide_id=polypeptide.get("id"),
                        polypeptide_source=polypeptide.get("source"),
                        polypeptide_name=_text(polypeptide, "name"),
                        gene_name=_text(polypeptide, "gene-name"),
                        polypeptide_organism=_text(polypeptide, "organism"),
                        uniprot_ids=_uniprot_ids(polypeptide),
                    )
                )
    return tuple(records)


def _parse_drug_interactions(
    drug: etree._Element,
) -> tuple[DrugInteraction, ...]:
    collection = drug.find(f"{_Q}drug-interactions")
    if collection is None:
        return ()
    records: list[DrugInteraction] = []
    for item in collection.findall(f"{_Q}drug-interaction"):
        drugbank_id = _text(item, "drugbank-id")
        if drugbank_id:
            records.append(
                DrugInteraction(
                    drugbank_id=drugbank_id,
                    name=_text(item, "name"),
                    description=_text(item, "description"),
                )
            )
    return tuple(records)


def _parse_drug(drug: etree._Element) -> DrugBankRecord:
    id_elements = drug.findall(f"{_Q}drugbank-id")
    all_ids = _unique(
        [(element.text or "").strip() for element in id_elements]
    )
    primary = next(
        (
            (element.text or "").strip()
            for element in id_elements
            if element.get("primary") == "true" and (element.text or "").strip()
        ),
        all_ids[0] if all_ids else None,
    )
    if primary is None:
        raise ValueError("DrugBank record has no DrugBank identifier")

    groups_element = drug.find(f"{_Q}groups")
    groups = (
        _unique(
            [(element.text or "").strip() for element in groups_element]
        )
        if groups_element is not None
        else ()
    )

    atc_codes: list[ATCCode] = []
    atc_element = drug.find(f"{_Q}atc-codes")
    if atc_element is not None:
        for code_element in atc_element.findall(f"{_Q}atc-code"):
            code = (code_element.get("code") or "").strip()
            if not code:
                continue
            levels = tuple(
                ATCLevel(
                    code=(level.get("code") or "").strip(),
                    name=(level.text or "").strip(),
                )
                for level in code_element.findall(f"{_Q}level")
            )
            atc_codes.append(ATCCode(code=code, levels=levels))

    external_identifiers: list[ExternalIdentifier] = []
    external_element = drug.find(f"{_Q}external-identifiers")
    if external_element is not None:
        for item in external_element.findall(f"{_Q}external-identifier"):
            resource = _text(item, "resource")
            identifier = _text(item, "identifier")
            if resource and identifier:
                external_identifiers.append(
                    ExternalIdentifier(resource=resource, identifier=identifier)
                )

    return DrugBankRecord(
        primary_drugbank_id=primary,
        all_drugbank_ids=all_ids,
        name=_text(drug, "name"),
        drug_type=drug.get("type"),
        groups=groups,
        atc_codes=tuple(atc_codes),
        external_identifiers=tuple(external_identifiers),
        drug_interactions=_parse_drug_interactions(drug),
        partners=_parse_partners(drug),
    )


def iter_drugbank_records(
    xml_path: str | Path, *, limit: int | None = None
) -> Iterator[DrugBankRecord]:
    """Yield DrugBank records with bounded memory using ``lxml.iterparse``."""

    source = Path(xml_path)
    if not source.is_file():
        raise FileNotFoundError(f"DrugBank XML not found: {source}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return

    count = 0
    context = etree.iterparse(
        str(source),
        events=("end",),
        tag=f"{_Q}drug",
        huge_tree=True,
        recover=False,
    )
    for _, element in context:
        parent = element.getparent()
        if parent is None or parent.tag != f"{_Q}drugbank":
            continue
        record = _parse_drug(element)
        element.clear(keep_tail=True)
        while element.getprevious() is not None:
            del parent[0]
        yield record
        count += 1
        if limit is not None and count >= limit:
            return


_DRUG_SCHEMA = pa.schema(
    [
        ("drugbank_id", pa.string()),
        ("all_drugbank_ids", pa.list_(pa.string())),
        ("name", pa.string()),
        ("drug_type", pa.string()),
        ("groups", pa.list_(pa.string())),
    ]
)
_ATC_SCHEMA = pa.schema(
    [
        ("drugbank_id", pa.string()),
        ("atc_code", pa.string()),
        ("level_codes", pa.list_(pa.string())),
        ("level_names", pa.list_(pa.string())),
    ]
)
_EXTERNAL_ID_SCHEMA = pa.schema(
    [
        ("drugbank_id", pa.string()),
        ("resource", pa.string()),
        ("identifier", pa.string()),
    ]
)
_DRUG_INTERACTION_SCHEMA = pa.schema(
    [
        ("source_drugbank_id", pa.string()),
        ("target_drugbank_id", pa.string()),
        ("target_name", pa.string()),
        ("description", pa.string()),
    ]
)
_PARTNER_SCHEMA = pa.schema(
    [
        ("drugbank_id", pa.string()),
        ("category", pa.string()),
        ("partner_id", pa.string()),
        ("name", pa.string()),
        ("organism", pa.string()),
        ("actions", pa.list_(pa.string())),
        ("known_action", pa.string()),
        ("reference_ids", pa.list_(pa.string())),
        ("polypeptide_id", pa.string()),
        ("polypeptide_source", pa.string()),
        ("polypeptide_name", pa.string()),
        ("gene_name", pa.string()),
        ("polypeptide_organism", pa.string()),
        ("uniprot_ids", pa.list_(pa.string())),
    ]
)


def _write_rows(
    writer: pq.ParquetWriter, rows: list[dict[str, object]], schema: pa.Schema
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def convert_drugbank_xml(
    xml_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 2_000,
    limit: int | None = None,
) -> DrugBankConversionSummary:
    """Stream DrugBank XML into normalized Parquet tables under interim data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(xml_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    specifications = {
        "drugs": ("drugbank_drugs.parquet", _DRUG_SCHEMA),
        "atc": ("drugbank_atc.parquet", _ATC_SCHEMA),
        "external": (
            "drugbank_external_identifiers.parquet",
            _EXTERNAL_ID_SCHEMA,
        ),
        "interactions": (
            "drugbank_drug_interactions.parquet",
            _DRUG_INTERACTION_SCHEMA,
        ),
        "partners": ("drugbank_partners.parquet", _PARTNER_SCHEMA),
    }
    temporary_paths = {
        key: destination / f"{filename}.part"
        for key, (filename, _) in specifications.items()
    }
    writers = {
        key: pq.ParquetWriter(temporary_paths[key], schema, compression="zstd")
        for key, (_, schema) in specifications.items()
    }
    rows: dict[str, list[dict[str, object]]] = {key: [] for key in specifications}
    counts = {
        "drugs": 0,
        "atc": 0,
        "external": 0,
        "interactions": 0,
        "partners": 0,
    }

    try:
        for record in iter_drugbank_records(source, limit=limit):
            rows["drugs"].append(
                {
                    "drugbank_id": record.primary_drugbank_id,
                    "all_drugbank_ids": list(record.all_drugbank_ids),
                    "name": record.name,
                    "drug_type": record.drug_type,
                    "groups": list(record.groups),
                }
            )
            counts["drugs"] += 1
            for atc in record.atc_codes:
                rows["atc"].append(
                    {
                        "drugbank_id": record.primary_drugbank_id,
                        "atc_code": atc.code,
                        "level_codes": [level.code for level in atc.levels],
                        "level_names": [level.name for level in atc.levels],
                    }
                )
                counts["atc"] += 1
            for external in record.external_identifiers:
                rows["external"].append(
                    {
                        "drugbank_id": record.primary_drugbank_id,
                        "resource": external.resource,
                        "identifier": external.identifier,
                    }
                )
                counts["external"] += 1
            for interaction in record.drug_interactions:
                rows["interactions"].append(
                    {
                        "source_drugbank_id": record.primary_drugbank_id,
                        "target_drugbank_id": interaction.drugbank_id,
                        "target_name": interaction.name,
                        "description": interaction.description,
                    }
                )
                counts["interactions"] += 1
            for partner in record.partners:
                partner_row = asdict(partner)
                partner_row["drugbank_id"] = record.primary_drugbank_id
                partner_row["actions"] = list(partner.actions)
                partner_row["reference_ids"] = list(partner.reference_ids)
                partner_row["uniprot_ids"] = list(partner.uniprot_ids)
                rows["partners"].append(partner_row)
                counts["partners"] += 1

            if counts["drugs"] % batch_size == 0:
                for key, (_, schema) in specifications.items():
                    _write_rows(writers[key], rows[key], schema)

        for key, (_, schema) in specifications.items():
            _write_rows(writers[key], rows[key], schema)
    finally:
        for writer in writers.values():
            writer.close()

    for key, (filename, _) in specifications.items():
        temporary_paths[key].replace(destination / filename)

    summary = DrugBankConversionSummary(
        source_xml=source,
        output_dir=destination,
        drug_count=counts["drugs"],
        atc_row_count=counts["atc"],
        external_identifier_row_count=counts["external"],
        drug_interaction_row_count=counts["interactions"],
        partner_row_count=counts["partners"],
    )
    (destination / "drugbank_conversion_summary.json").write_text(
        json.dumps(
            {
                "source_xml": str(summary.source_xml),
                "source_size_bytes": source.stat().st_size,
                "output_dir": str(summary.output_dir),
                "drug_count": summary.drug_count,
                "atc_row_count": summary.atc_row_count,
                "external_identifier_row_count": summary.external_identifier_row_count,
                "drug_interaction_row_count": summary.drug_interaction_row_count,
                "partner_row_count": summary.partner_row_count,
                "semantic_warning": (
                    "Free text and known-action values are preserved only as source "
                    "fields; no DDA labels or refutation labels are inferred."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
