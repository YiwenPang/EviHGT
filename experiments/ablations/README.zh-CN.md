<a id="training-objective-and-ablation-experiments"></a>

# 训练目标与消融实验

[English](README.md)

项目主入口可运行损失与证据干预的配对实验。本目录也收录了早期的六变体留出实验。该实验采用了当时的模型、数据划分与检查点选择规则。

<a id="run-from-the-four-source-files"></a>

## 从四个原始文件运行

按照项目主指南准备原始文件和数据准备配置，然后在仓库根目录执行命令。[下载与放置](../../README.zh-CN.md#download-and-place-the-four-source-files)

```sh
python main.py --experiment ablations
python main.py --experiment losses
```

`ablations` 在 `R0F1` 和 `R1F0` 上运行 `FULL`、`CONTENT_ZERO` 和 `FACT_QUALITY_CONSTANT`。`losses` 在 `R0F0`、`R0F1` 和 `R1F0` 上分别用 `softplus` 和 `sigmoid` 训练 FULL。其中 `R0F0` 采用专用的 `DEV95` 损失对照配方。

```sh
python main.py --experiment ablations --cells R0F1 --variants FULL FACT_QUALITY_CONSTANT
python main.py --experiment losses --cells R0F0 --mode prepare
python main.py --experiment losses --smoke
```

`--mode prepare` 在输入准备完成后退出。快速自检在 CPU 上使用合成数据，默认运行一个数据折。实验运行目录中的 `report.html` 提供汇总，`disease_metrics.csv` 保存逐疾病指标。[运行目录与检查点恢复](../README.zh-CN.md#run-from-the-source-databases)

<a id="paired-loss-and-evidence-experiments"></a>

## 损失与证据干预的配对实验

损失选择使用三个开发折，证据干预使用两个开发折。同一折的两个损失实验臂采用相同的模型初始化，各自建立优化器状态。它们共用节点特征、图掩码、正例与未标注样本、更新计划及开发集候选。每条更新日志都保存样本标识及两个输出通道的配对数。条件采样从训练集的合格药物中均匀抽取未标注候选，固定检查点的摘要参与确定抽样种子。抽样过程与模型分数无关。

| 数据折 | 每个实验臂的更新数 | 对照 |
| --- | ---: | --- |
| R0F0 | 20,568 | 有界 sigmoid 与乘以二分之一的 softplus 损失 |
| R0F1 | 20,541 | 比较两种损失，以及 FULL 分别与 Content zero、Constant quality 的对照 |
| R1F0 | 20,700 | 比较两种损失，以及 FULL 分别与 Content zero、Constant quality 的对照 |

源码包括[损失定义](development/src/evi_hgt_loss_comparison/losses.py)、[首折执行程序](development/src/evi_hgt_loss_comparison/runner.py)、[两折执行程序](development/src/evi_hgt_paired_losses/runner.py)和[配对采样计划](development/src/evi_hgt_paired_losses/sampling.py)。[消融执行程序](development/src/evi_hgt_evidence_ablation/runner.py)复用乘以二分之一的 softplus 更新，并在每个逻辑步核对与 FULL 的九个采样字段。[模型钩子](development/src/evi_hgt_evidence_ablation/model.py)调用[原始干预代码](development/src/evi_hgt/ranking_protocol/model.py)。

- **Content zero** 将池化后的证据内容向量和来源向量置零。其他输入字段和模型结构保持原样，实际梯度传播发生变化。AP 差异随数据折及输出通道而异。
- **Constant quality** 将适用位置的四个证据分量固定为 0.25，并保留可用性掩码。相较于该变体，FULL 的禁忌证 AP 在 R0F1 和 R1F0 分别提高 0.92 和 0.09 个百分点，治疗支持 AP 在两折各低约 0.11 个百分点。

配对比较采用固定的三周期终点，在内层开发面板上评估。R1F0 的 sigmoid 实验从检查点恢复时重放了 116 次有日志记录的更新。对齐依据是逻辑更新编号，重放记录的样本标识也会核对。实际计算量有差别。[原始重放计数与执行程序](development/scripts/paired_losses/recovery/recovery_excerpts.py)和[后续执行及汇总代码](development/scripts/evidence_ablation/recovery/recovery_excerpts.py)保留了恢复过程的相关代码。后者复用一个已完成的证据干预，并完成其余三个实验臂。

<a id="historical-six-variant-study"></a>

## 历史六变体实验

[训练执行程序](historical/src/evi_hgt/training/formal_driver.py)、[变体目录](historical/src/evi_hgt/models/variants.py)和[补充变体](historical/src/evi_hgt/protocols/confirmatory_ablation.py)说明了 FULL、项目内仅用关系的 BASE_HGT 和四个消融变体。四个评估阶段的构建与评分见[候选协议](historical/src/evi_hgt/evaluation/evaluable_protocol.py)、[评分程序](historical/src/evi_hgt/evaluation/overnight_runtime.py)和[运行后分析](historical/scripts/analyze_results.py)。

阶段 A 在同一组冻结的 47,769 个候选对上比较六种方法的治疗支持分数。1,354 个疾病中，1,109 个同时有已记录的正例和未标注候选。ROC-AUC 对并列分数计半分，AP 将相同分数归入同一阈值组。Top-K 排名出现并列时按规范化药物 ID 升序排列。FULL 在验证集上比较五个学习率，选出 0.00005。随后所有实验臂使用该学习率，因此 FULL 有额外的调参机会。各臂遵循同一规则，分别选择自己的最佳验证检查点。BASE 选中第 9,380 步，其余五臂选中第 18,760 步。各模型的架构和参数量有差别。

FULL 的宏平均 AP 为 0.475181，BASE_HGT 为 0.412357。该实验中四个消融变体的 AP 均高于 FULL。结果适用于单个训练种子下、使用抽样候选的治疗支持任务。阶段 B 只评估 FULL，使用预先选定的六种疾病及其完整候选空间。阶段 C 和 D 也只评估 FULL，分别使用历史药物冷启动与疾病冷启动的治疗支持面板。两者的宏平均 ROC-AUC 分别为 0.4950 和 0.4992。评分时允许使用经过审查的查询端点辅助信息。[历史冷启动可见性策略](historical/src/evi_hgt/data/cold_visibility.py)

<a id="verification"></a>

## 校验

在仓库根目录运行以下命令。校验公开源码的文件内容只需 Python 标准库。

```sh
python experiments/ablations/verify.py sources
```

该命令也接受 `--research-root PATH`，用于在已有研究工作区中核对原始文件校验值和运行凭证。如果本地保留了原始更新日志，以下命令还能检查全部三组损失对照与四次干预运行，包括逻辑步是否完整、九个配对字段是否一致，以及重放时是否使用相同样本。

```sh
python experiments/ablations/verify.py pairing --research-root PATH
```

历史分析的合成用例涵盖并列排名、跳过查询、配对 bootstrap 重采样、精确指标复算和候选成员核对。安装 NumPy、Polars、tqdm 和 pytest 后，从本目录运行测试。

```sh
cd historical
python -m pytest -q -o pythonpath=src tests/test_results_analysis.py
```

在仓库根目录运行以下协议回归测试。测试使用一个三周期合成实例，将计划摘要、条件采样绑定、随机种子和选中的候选与原始实现的结果逐项核对。它也核对三个数据折各自的初始化种子，以及融合层在检查点中的键名。

```sh
python -m pytest -q experiments/ablations/tests/test_protocol_ids.py
```

[SHA256.json](SHA256.json) 记录各文件在当前公开版本、早期发布版本和原研究工作区中的校验值。两个恢复模块保留了代码节选。[回放归档](../README.zh-CN.md)提供历史实验的原始文件布局、凭证映射和确切的源码绑定。

[协议标识](development/src/ablation_protocol_ids.py)保留计算原始抽样种子与计划摘要，以及记录特征元数据、命名模型参数时使用的字符串。[产物标识](development/src/ablation_artifacts.py)和[记录标识](development/src/ablation_records.py)给出相应诊断面板、检查点、清单及运行凭证的名称。公开目录和脚本按所实现的实验命名。

上述模块公开了供核查的实验实现。历史记录绑定了文件路径、字段名及确切的源码摘要。若要在原研究工作区运行训练、注册或评估脚本，可在原有参数之外传入 `--research-root PATH`。启动器在固定版本的归档中定位对应的研究脚本，校验其 SHA-256，再使用当前 Python 解释器在研究工作区执行。原有依赖、选模和授权检查仍适用。公开的合成分析测试与源码校验可在本地运行。
