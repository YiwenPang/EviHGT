<a id="hgtdr-cold-disease-comparison"></a>

# HGTDR 疾病冷启动对比

[English](README.md)

本实验使用冷启动脚本生成的严格疾病留出划分。两个单输出 HGTDR 模型分别以已记录的适应证和禁忌证为正例。每个模型沿用原生的 300 个训练周期，使用 AdamW，初始学习率为 0.003，按余弦调度衰减。每批使用等量的已记录正例和抽样未标注实体对。检查点先按开发集上逐疾病宏平均 ROC-AUC 选择，随后比较 AP，同分时取较早的周期。原实验的 support 和 contra 检查点分别在第 149 和第 53 个周期选定。[冷启动脚本](../README.zh-CN.md)

选定的模型与 EviHGT 对同一组留出候选排序。留出疾病未参与训练，图中移除了与这些疾病相连的事实。两种方法共用冻结的药物和疾病向量。EviHGT 还使用其静态侧信息，并使用固定三个训练周期后的检查点进行评估。[对比结果与配对区间](../../../docs/experiments.md#cold-disease-comparison)

<a id="read-the-results"></a>

## 查看结果

在仓库根目录使用 Python 3.11 或更高版本。汇总结果可直接查看，无需模型或数据依赖。

```sh
python -B -m experiments.cold_start.hgtdr.run summary
```

<a id="repeat-training-and-scoring"></a>

## 复跑训练和评分

论文实验使用已审计的 EviHGT 研究工作区，以及同级目录下的 HGTDR 重建工程。请使用包含 HGTDR 原生实现、CUDA 及兼容的 PyTorch、PyG 和 torch-sparse 软件包的解释器。复跑还需冻结的 HGTDR 特征包及其身份校验记录。原生运行时沿用对比适配器的工作区和特征绑定。[原生适配器指南](../../native_adapters/README.zh-CN.md) · [R0 禁忌证适配器](../../hgtdr_contra/README.zh-CN.md)

使用本地授权数据，按照冷启动指南准备并运行 `cold_disease` 场景。传入准备好的 `manifest.json` 所在目录，以及已完成的 EviHGT `result.json` 所在目录。以下命令检查输入和原生图，运行到校验阶段即结束。[冷启动指南](../README.zh-CN.md#running-with-licensed-inputs)

```sh
python -B -m experiments.cold_start.hgtdr.run validate --research-root /path/to/research --corpus /path/to/prepared/cold_disease
```

以下命令训练两个模型并对测试候选评分。

```sh
python -B -m experiments.cold_start.hgtdr.run run --research-root /path/to/research --corpus /path/to/prepared/cold_disease --reference-run /path/to/evihgt/runs/cold_disease
```

使用 `train` 时，流程在训练和检查点选择完成后结束。两个训练回执写入后，使用 `score`、相同的语料和运行目录，并传入 `--reference-run` 对测试候选评分。默认图缓存和运行输出分别位于研究工作区下的 `data/cache/cold_disease_hgtdr` 和 `outputs/cold_disease_hgtdr`。可用 `--cache` 和 `--output` 在对应父目录下另选位置。训练中断后，按相同设置重新运行会从最近的兼容周期检查点恢复。两个模型均完成后，`result.json` 给出配对比较结果。

[原实验汇总结果](results.json) · [源码校验和](source_sha256.json) · [数据来源与获取方式](../../../docs/data.md)
