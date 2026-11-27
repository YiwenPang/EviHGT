<a id="native-baseline-adaptations"></a>

# 原生基线的适配实现

[English](README.md)

五折对比和早期试验采用两个已发表模型的本地实现。源码按 EviHGT 实验协议及 HGTDR、TxGNN 的集成代码组织。五折实验统一了各折的数据划分、允许使用的图和评分规则。 [五折对比](../comparisons/README.zh-CN.md)

| 模型 | 上游源码 | 本地适配 |
| --- | --- | --- |
| HGTDR | [作者仓库的固定版本](https://github.com/bcb-sut/HGTDR/tree/a0adec64b08c9d70835e1e7c04a513b9be6a87db) | [折适配器](EviHGT/src/evi_hgt/ranking_protocol/hgtdr_adapter.py)、[采样、训练与评分](HGTDR/src/competition.py)、[输入特征](HGTDR/src/competition_features.py) |
| TxGNN | [作者仓库的固定版本](https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023) | [只使用训练数据的采样](EviHGT/src/evi_hgt/ranking_protocol/txgnn_adapter.py)、[训练配置与选择规则](EviHGT/src/evi_hgt/ranking_protocol/txgnn_protocol.py)、[模型集成](TxGNN/txgnn/competition.py) |

HGTDR 重建版使用冻结的 BioBERT 名称向量和 ChemBERTa 分子特征，并处理输入缺失的情况。模型包含三层 HGT，每层宽度 64，使用八个注意力头。模型按关系做均值聚合，将各层输出拼接后输入带批归一化的 MLP。预测头的 dropout 为 0.2。损失函数保留上游模型按样本设定正例权重的做法。本地的 `HGTDR/src/runtime.py` 由原 notebook 整理而来，是额外依赖。记录的 SHA-256 为 `583c40f389a8395881703b8661bc1097150f4f36ca1cdbaac159b85d9f2b2fe8`。仓库附有 PyG 兼容层及其许可证。HGTDR-contra 实验沿用相同的单输出结构，将训练目标改为禁忌证。 [PyG 兼容层许可证](HGTDR/src/vendor/PYG_LICENSE)

TxGNN 采用上游的 HeteroRGCN 和 DistMult。疾病原型索引和度数根据该折可见的训练正例确定。原始 `model.py` 可在固定提交中获取。兼容补丁将上游工具中 pandas DataFrame 的追加调用改为拼接。在固定版本的 TxGNN 仓库根目录执行 `git apply /path/to/utils-compat.patch`。 [兼容补丁](TxGNN/utils-compat.patch) · [TxGNN 许可证](TxGNN/LICENSE)

源码校验值用于核验归档的项目适配器。源码还原工具会恢复原始训练命令所需的目录结构。运行这些命令需要相邻的研究检出目录、原生依赖、图和特征缓存，以及封存的折划分和模型选择记录。 [源码校验值](source_sha256.txt) · [源码还原](../replay.py) · [早期试验协议](../pilot/README.zh-CN.md) · [禁忌证适配器](../hgtdr_contra/README.zh-CN.md)
