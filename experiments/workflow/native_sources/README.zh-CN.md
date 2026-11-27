<a id="native-environments"></a>

# 原生基线环境

[English](README.md)

请在 EviHGT 主环境中安装匹配对照和 HGTDR 所需的[实验依赖](../../../requirements-experiments.txt)。已记录的 HGTDR 实验用原生 CPU 算子 torch_sparse.hgt_sample 采样，在 CUDA 上训练。安装 torch-sparse 0.6.18 和 torch-scatter 时，请选择适配 Python、PyTorch 和操作系统的构建版本。参考工作站采用 PyTorch 2.13.0、CUDA 13.2 和 PyG 2.8.0.post1。 [PyG 安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

可移植的 HGTDR 实现沿用已发表的架构、现有的查询屏蔽 CPU 采样器，以及 MIT 许可的 PyG 2.3.1 层。适配器以本地的规范节点汇总数据为输入，对另外八种节点的名称运行 BioBERT 编码，并将允许使用的原生图连同文件哈希清单一并导出到实验数据缓存。 [模型论文](https://doi.org/10.1093/bioinformatics/btae349) · [PyG 层的 MIT 许可证](../../native_adapters/HGTDR/src/vendor/PYG_LICENSE)

对比命令在独立的 CPython 3.11 CPU 环境中运行 TxGNN，使用 PyTorch 2.2.2+cpu 和 DGL 1.1.2。在 Windows 上，从 EviHGT 仓库根目录运行以下命令。

    py -3.11 -m venv .venv-txgnn
    .venv-txgnn/Scripts/python.exe -m pip install -r requirements-txgnn.txt

在 Linux 上，使用已安装的 CPython 3.11 和虚拟环境中的 bin/python。项目记录的 CPU 运行环境为 Windows。如需指定其他解释器，可设置 EVIHGT_TXGNN_PYTHON 环境变量。

首次联网执行正式实验时，程序会检出 TxGNN 的固定版本，应用附带的 pandas 兼容补丁，并校验模型和工具文件的哈希。离线运行和冒烟测试需要事先准备源码缓存。也可以指定已有的检出目录，前提是它采用同一提交并已应用兼容补丁。TxGNN 加载固定版本的 HeteroRGCN，使用本地 EviHGT 图和候选集。 [TxGNN 固定版本](https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023) · [兼容补丁](../../native_adapters/TxGNN/utils-compat.patch)

合成数据测试沿用原生模型和采样器，缩短训练预算。运行这类测试仍需要相应的可选依赖和已校验的本地 TxGNN 检出目录或缓存。
