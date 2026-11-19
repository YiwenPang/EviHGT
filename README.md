# EviHGT

EviHGT is an evidence-aware heterogeneous graph Transformer for drug–disease association ranking. Given a drug–disease query, it builds a subgraph from OptimusKG and structured DrugBank records, encodes the evidence and its provenance, and scores treatment support and contraindication-related associations in separate heads.

[Try the public checks on CPU](#try-the-public-checks) · [Inputs for the paper experiments](experiments/README.md#inputs-for-the-paper-experiments)

## Architecture

![EviHGT architecture](docs/figures/architecture.png)

[Overview (SVG)](docs/figures/architecture.svg) · [Layer and path details (SVG)](docs/figures/architecture_details.svg) · [Editable Visio source](docs/figures/architecture.vsdx)

The full model combines provenance-aware evidence pooling, four-valued fact representations (True, False, Both and Unknown), query-conditioned graph layers, and bounded path encoding. It can use frozen chemical and disease-name embeddings as node features. When fact logic is available, the four direct logic transforms are averaged with equal weights. Learned fact masses enter the edge context. [Model equations and ablations](docs/model.md)

## Try the public checks

With Git and Python 3.11 or later, you can check the published five-fold comparison immediately. Run these commands in a terminal.

```sh
git clone https://github.com/YiwenPang/EviHGT.git
cd EviHGT
python experiments/audit/verify_r0.py
```

This uses only the Python standard library. Expect `Verified 55 method/fold/head entries across 10 common panels.` The check reads the published ledger and verifies that the methods share the recorded candidate counts and panel identity within each fold and output.

To try the model and the experiment code, create a virtual environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `. .venv/bin/activate` in a POSIX shell. For a CPU installation on Windows or Linux, run the following from the repository root.

```sh
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python experiments/run_checks.py
```

Once the dependencies are installed, the last command runs the public checks offline on CPU. It exercises the comparison metrics, Popularity baseline, ablations, HGTDR-contra scoring and checkpoints, and cold-start splits. It also trains EviHGT for three updates on a tiny synthetic graph and scores held-out candidates with both prediction heads. A successful run ends with `All public experiment checks passed.`

The paper's reported scores come from the original research inputs. Reproducing them requires the matching data, splits and study-specific artifacts. The [experiment guide](experiments/README.md#inputs-for-the-paper-experiments) lists what each study needs.

## Installation

Use Python 3.11 or later in a virtual environment. Install a [PyTorch build](https://pytorch.org/get-started/locally/) for your platform and device, then install the remaining dependencies.

```sh
python -m pip install -r requirements.txt
python check_environment.py --device cuda
```

The default configuration uses CUDA with BF16. For a CPU/FP32 run, use `configs/cpu.yaml` and check the environment with `--device cpu`. The observed dependency versions are listed in [requirements-observed.txt](requirements-observed.txt).

## Data and training

Training requires a local corpus with normalized facts and evidence, leakage-filtered graph indexes, node features and availability masks, a three-cycle sample schedule, and development evaluation panels. Use [the data guide](docs/data.md) to import normalized artifacts with `prepare_data.py` or export them from an audited EviHGT workspace with `export_data.py`. The data guide identifies the OptimusKG, DrugBank, ChemBERTa and BioBERT sources and the inputs each step needs. Keep source records and derived data in the ignored `data/` directory and use your own authorized DrugBank copy.

### Train the full model

```sh
python main.py
```

Training checks the inputs first, then runs the three-cycle plan in the prepared corpus. The default settings are in `configs/default.yaml`. Training writes logs, development metrics and checkpoints to `outputs/`. See the [reproducibility guide](docs/reproducibility.md) for output files, checkpoints and evaluation scope.

### Check the inputs before training (optional)

```sh
python main.py --mode validate
```

Use this command after preparing or changing the corpus. It checks the configuration and loads the prepared inputs, writes a validation report to `outputs/`, and exits before initializing the model.

For CPU, add `--config configs/cpu.yaml` to either command. In PyCharm, select the same interpreter and set the desired arguments on the `main.py` run configuration.

Run specific ablations by name, for example:

```sh
python ablation.py --variants NO_D1 NO_PATH
```

## Development results

Browse the [baseline comparisons](experiments/comparisons/), [ablations](experiments/ablations/) and [cold-start experiments](experiments/cold_start/). The [experiment index](experiments/) also links to the HGTDR contraindication study and the fold-by-fold audit.

The table summarizes repeat R0 across five overlapping folds. Metrics were averaged across eligible folds for each disease, then across 1,697 support diseases or 314 contraindication-related diseases. HGTDR supplies only support scores in this comparison.

| Method | Support AUC | Support AP | Support NDCG@10 | Contra AUC | Contra AP | Contra NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EviHGT | 0.898440 | 0.873292 | 0.911893 | 0.922813 | 0.895974 | 0.927292 |
| HGT-D1 | 0.864699 | 0.835220 | 0.883860 | 0.927014 | 0.897898 | 0.928273 |
| PairMLP-D1 | 0.858382 | 0.823047 | 0.885701 | 0.901586 | 0.878157 | 0.913876 |
| HGTDR adapted | 0.922917 | 0.901453 | 0.930877 | — | — | — |
| TxGNN adapted | 0.813216 | 0.778118 | 0.848928 | 0.876748 | 0.837941 | 0.884425 |
| Popularity | 0.828239 | 0.789221 | 0.864763 | 0.932036 | 0.893040 | 0.934880 |

EviHGT scores above the input- and budget-matched HGT-D1 and PairMLP-D1 on all three support metrics. Adapted HGTDR leads those metrics. HGT-D1 leads contra AP, and Popularity leads contra AUC and NDCG@10. [Experiment figures and evaluation protocol](docs/experiments.md)

## Tests

To run just the portable model and data tests after installing the test dependencies, use `python -m pytest tests -q --basetemp outputs/pytest`. The [public checks](#try-the-public-checks) also exercise the study-specific code.

[CI workflow](.github/workflows/tests.yml) · [Python API](docs/python_api.md) · [Troubleshooting](docs/troubleshooting.md) · [Model card](docs/model_card.md) · [Contributing](CONTRIBUTING.md) · [Citation](CITATION.md)
