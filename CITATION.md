# Citing EviHGT

Identify the software as **EviHGT: Evidence-aware Heterogeneous Graph Transformer for Drug–Disease Association Ranking** and include the repository URL and commit SHA. For a new experiment, record the model variant, configuration, corpus identifier, training seed, software versions, hardware and precision. Formal paper citation metadata and a DOI are pending.

Cite the database releases used in the experiment. This study uses [OptimusKG release 2.0, DOI: 10.7910/DVN/IYNGEV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IYNGEV&version=2.0). Record the release information accompanying your authorized [DrugBank](https://go.drugbank.com/releases/latest) download; the study XML header gives version 5.1 and an export date of 2026-04-23, without identifying a patch release. The [data guide](docs/data.md) gives the source files and checksums.

For runs with D1 features, also cite [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) and [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1). Encoder revisions and input fields are listed in the [data guide](docs/data.md).
