<a id="hgtdr-adapted-contra"></a>

# HGTDR-adapted-contra 禁忌证模型

[English](README.md)

这个适配器使用已记录的禁忌证训练单输出 HGTDR 模型。R0 每折采用原有的屏蔽图、训练禁忌证正例和抽样未标注对，并在共同的内层开发候选集上评估。损失函数将 P 和抽样 U 分别编码为 1 和 0。U 的未标注状态保持不变。

训练使用 AdamW，初始学习率为 0.003，在 300 个 epoch 内进行余弦退火。每个批次的 P 和 U 数量相同。选择检查点时，先看按疾病宏平均的 ROC-AUC，再看 AP，仍同分时取较早的 epoch。R0 汇总先对每个疾病的可评估折取均值，再对疾病等权取均值。

五折内层开发候选集共覆盖 314 个可评估疾病。HGTDR-adapted-contra 的 ROC-AUC 为 0.964060，AP 为 0.942237，NDCG@10 为 0.959349。 [主对比结果中的禁忌证行](../../README.zh-CN.md#development-results)

<a id="training"></a>

## 训练

归档入口需要已审计的 EviHGT 研究工作区，其中包含排序与采样模块、原始 R0 折清单，以及经过哈希校验的原生图缓存和特征。它还使用该工作区对应的本地 HGTDR 重建版和 CUDA 环境。 [HGTDR 原论文](https://doi.org/10.1093/bioinformatics/btae349)

在本仓库中使用研究工作区配置的解释器运行。

```sh
python -m experiments.hgtdr_contra.run run --research-root /path/to/research-workspace
```

用 `--output` 在研究工作区的 `outputs` 下指定单独的输出目录。运行器在每个 epoch 完成后保存训练状态，并记录各折的选定权重和禁忌证指标。检查点记录适配器源码哈希和所在折的输入身份。恢复该适配器发起的任务时，请使用此检出版本。2026 年 11 月的实验调用研究检出目录中的 `src/evi_hgt_hgtdr_contra/` 软件包，其 `protocol.json` 保存五个原始适配器文件的 SHA-256。请将原始和派生数据、逐对预测、日志及权重保存在本地工作区。

<a id="tests"></a>

## 测试

公开 CPU 测试用合成候选和检查点检查封存的候选顺序、评分、检查点选择与恢复，以及按疾病优先汇总的规则。从仓库根目录运行。

```sh
python -m pytest -q tests/test_hgtdr_contra.py experiments/hgtdr_contra/tests/test_run.py
```

另有九项测试核验数据绑定和原生训练，需要研究工作区的 Python 模块。运行时指定该工作区的源码路径。

```sh
python -m pytest -q -o pythonpath=/path/to/research-workspace/src experiments/hgtdr_contra/tests/test_data.py experiments/hgtdr_contra/tests/test_trainer.py
```
