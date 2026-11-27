# Native environments

[中文指南](README.zh-CN.md)

Install the [experiment dependencies](../../../requirements-experiments.txt) in the main EviHGT environment for the matched controls and HGTDR. HGTDR uses the native CPU torch_sparse.hgt_sample operator and CUDA for the recorded training runs. Install torch-sparse 0.6.18 and torch-scatter builds that match your Python, PyTorch and platform. The [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) lists the compatible wheels and source build procedure. The reference workstation used PyTorch 2.13.0 with CUDA 13.2 and PyG 2.8.0.post1.

The portable HGTDR implementation follows the [published architecture](https://doi.org/10.1093/bioinformatics/btae349) and uses the existing query-masked CPU sampler and [MIT-licensed PyG 2.3.1 layer](../../native_adapters/HGTDR/src/vendor/PYG_LICENSE). The adapter infers BioBERT name features for eight additional node types from the local canonical node aggregate. It exports the legal native graph to the experiment data cache with a file-hash manifest.

The comparison command runs TxGNN in a separate CPython 3.11 CPU interpreter with PyTorch 2.2.2+cpu and DGL 1.1.2. On Windows, run the following from the EviHGT repository root.

    py -3.11 -m venv .venv-txgnn
    .venv-txgnn/Scripts/python.exe -m pip install -r requirements-txgnn.txt

On Linux, use the installed CPython 3.11 and the virtual environment's bin/python executable. The recorded CPU environment is Windows. The EVIHGT_TXGNN_PYTHON environment variable can identify a different interpreter.

The first online production run retrieves the [pinned TxGNN revision](https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023), applies the bundled pandas compatibility patch and verifies the model and utility hashes. Prepare the source cache in advance for offline and smoke runs. To use an existing checkout, supply the source directory with the same revision and the [compatibility patch](../../native_adapters/TxGNN/utils-compat.patch) applied. TxGNN loads the pinned HeteroRGCN and uses the local EviHGT graph and candidates.

Synthetic runs keep the native models and samplers and use short training budgets. They require the corresponding optional libraries and an already-verified local TxGNN source checkout or cache.
