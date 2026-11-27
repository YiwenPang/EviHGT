<a id="comparisons-on-the-r0-development-folds"></a>

# R0 开发折对比实验

[English](README.md)

<a id="run-the-r0-comparison-from-the-four-source-files"></a>

## 从四个原始文件运行 R0 对比实验

```sh
python main.py --experiment comparisons
```

命令会准备 R0 的五折数据（`R0F0` 至 `R0F4`），运行 FULL 和六个基线。各方法在适用的开发候选集上评分。用 `--cells` 指定折，用 `--methods` 和下表的 ID 指定方法。

| 命令行 ID | 模型 |
| --- | --- |
| `FULL` | EviHGT |
| `HGT_D1_MATCHED` | HGT-D1 |
| `PAIR_MLP_D1_MATCHED` | PairMLP-D1 |
| `HGTDR_ADAPTED` | 用于治疗支持的 HGTDR-adapted |
| `HGTDR_ADAPTED_CONTRA` | 用于禁忌证的 HGTDR-adapted-contra |
| `TXGNN_ADAPTED` | TxGNN-adapted |
| `TRAIN_VISIBLE_POPULARITY` | 按训练正例计数评分的 Popularity |

```sh
python main.py --experiment comparisons --cells R0F1 --methods FULL HGT_D1_MATCHED
python main.py --experiment comparisons --smoke --methods FULL TRAIN_VISIBLE_POPULARITY
```

HGT-D1 和 PairMLP-D1 需要匹配对照的依赖。HGTDR 方法还需要原生 `torch-sparse` 和 `torch-scatter`。TxGNN 使用独立的 CPython 3.11 CPU 解释器和固定源码版本，按需通过 `--txgnn-python` 和 `--txgnn-source` 配置。上面的冒烟测试使用合成数据。如果选择原生方法，也要装好相应依赖。选择 TxGNN 时，还需要经过校验的本地源码。 [原生依赖安装](../workflow/native_sources/README.zh-CN.md)

运行目录中的 `report.html` 展示汇总结果。`disease_metrics.csv` 保存逐疾病指标。汇总先对每个疾病的可评估折取均值，再对疾病取均值。 [筛选、准备和恢复参数](../README.zh-CN.md#run-from-the-source-databases)

<a id="original-six-method-comparison"></a>

## 原始六方法对比实验

本目录保存共同药物–疾病排序实验的研究代码。R0 的五折分别设有治疗支持和禁忌证两套内层开发候选集。同一折内，同一任务的各方法使用相同的药物–疾病对和 P/U 标签。P 表示该任务中已记录的正例，两套候选集共用同一批抽样未标注对（U）。五折存在数据重叠，属于同一次重复实验。

<a id="what-each-method-received"></a>

## 各方法的输入、训练和检查点

| 方法 | 输入与训练 | 开发集评分所用检查点 |
| --- | --- | --- |
| EviHGT（FULL） | 查询图、候选路径、节点辅助信息、冻结的 D1 特征和事实级证据 | 三个训练周期的终点 |
| HGT-D1 | 在查询图上运行 PyG HGT，节点辅助信息和 D1 特征与 FULL 相同 | 训练终点、采样计划、损失和优化器均与 FULL 相同 |
| PairMLP-D1 | 药物和疾病节点特征，节点编码器和 D1 特征来源与 FULL 相同 | 训练终点、采样计划、损失和优化器均与 FULL 相同 |
| HGTDR-adapted | HGTDR 使用固定节点特征和屏蔽查询关系的采样器，训练治疗支持任务 | 在 300 个 epoch 内按内层治疗支持 ROC-AUC、AP、较早 epoch 的顺序选择 |
| TxGNN-adapted | 预训练关系 1 个 epoch，微调 DDA 500 个 epoch | 在内层治疗支持任务上评估 20 次，选出的同一模型为两个任务评分 |
| Popularity | 对每种药物和每项任务，分别统计与该药物构成训练正例的不同疾病数 | 该药物对所有候选疾病的分数固定为 `log1p(count)`。该任务中没有训练正例的药物得 0 分 |

三个匹配模型每处理一组正例就更新一次，共训练三个周期，各折预算为 20,421–20,568 次更新。两个原生适配模型与匹配模型在输入特征、训练预算和检查点选择机会方面有所不同，衡量的是各系统在相同开发候选集上的整体表现。原生适配模型采用报告结果所用的内层治疗支持候选集来选择检查点。FULL 和匹配对照使用的损失函数也在开发阶段选定，较早的损失对比包含 R0F0 和 R0F1。 [消融与损失对比](../ablations/README.zh-CN.md)

实验覆盖 30 个方法与折的组合。运行队列新训练了 19 个模型实例，核验了 FULL 五折和 HGTDR 治疗支持一折的已有终点，并对每折运行了 Popularity 评分。此次对比中，HGTDR 为每个候选生成一个治疗支持分数。后来开展的禁忌证专项实验有单独的实现和检查点选择记录。 [HGTDR 禁忌证实验](../hgtdr_contra/README.zh-CN.md)

<a id="from-training-membership-to-the-published-table"></a>

## 从训练数据到发表结果

1. `TrainingInputs` 构造各折的训练视图。语料构建器移除屏蔽的临床事实及其证据，将允许使用的图绑定到该折。两个原生适配模型在导出图之前也会核验允许使用的事实 ID。 [训练样本](src/evi_hgt_ranking_comparison/data.py) · [图构建](src/evi_hgt_ranking_comparison/corpus.py) · [原生模型输入处理](src/evi_hgt_ranking_comparison/native_runtime.py)
2. 匹配模型采用同一套有序 P/U 采样计划、优化器调度和损失。HGT-D1 和 PairMLP-D1 的端点读出层共享初始参数。FULL 按自身结构进行图和路径计算。训练频次基线直接用训练正例构建两个任务的计数字典。 [采样](src/evi_hgt_ranking_comparison/sampling.py) · [共享训练流程](src/evi_hgt_ranking_comparison/training.py) · [匹配模型](src/evi_hgt_ranking_comparison/models.py) · [FULL](src/evi_hgt_ranking_comparison/full.py) · [Popularity 准备](src/evi_hgt_ranking_comparison/popularity_preparation.py)
3. 调度器读取对应的运行清单，每次启动一个方法在一折上的任务。原生训练进程将选定权重与训练结束时的恢复状态分开保存。HGTDR 复用的 R0F0 使用留存的第 222 个 epoch 权重评分。 [训练入口](scripts/ranking_comparison/runtime.py) · [调度器](src/evi_hgt_ranking_comparison/runtime.py) · [原生检查点处理](src/evi_hgt_ranking_comparison/native_checkpoint.py) · [复用权重](src/evi_hgt_ranking_comparison/reuse.py)
4. 评分基于开发候选集，并核验候选与检查点身份。ROC-AUC 和 AP 将得分完全相同的候选归为一组。Top-K 指标按规范药物 ID 打破同分，并将 K 限定在可用候选数以内。每个可评估疾病至少含一个 P 和一个 U。R0 表先对每个疾病的可评估折取均值，再对疾病等权取均值。 [开发集评分](src/evi_hgt_ranking_comparison/runtime_evaluation.py) · [指标定义](src/evi_hgt_ranking_comparison/metrics.py) · [R0 表计算](summarize_r0.py)

按这一汇总口径，FULL、匹配 HGT 和 HGTDR-adapted 的治疗支持 AP 分别为 0.873292、0.835220 和 0.901453。在原始六方法对比中，HGT-D1 的禁忌证 AP 最高（0.897898），Popularity 的禁忌证 ROC-AUC（0.932036）和 NDCG@10（0.934880）最高。若改用各折等权平均，Popularity 在这组方法中排名禁忌证 AP 第一。主结果表还收录了后来训练的 HGTDR-adapted-contra，其禁忌证 AP 为 0.942237。这些指标衡量的是已记录正例与抽样未标注候选的排序表现。引用结果时应注明检查点选择和汇总规则。 [主结果表](../../README.zh-CN.md#development-results) · [实验图与结果](../../docs/experiments.md)

<a id="source-identity-and-running-the-checks"></a>

## 核验源码与运行测试

源码清单为 45 个文件分别记录了当前发布版的 SHA-256、11 月 9 日执行清单中的原始 SHA-256 及对应的 Git blob。用于采样、权重初始化以及保存指标和计数的协议标识符沿用原值。 [源码哈希](SOURCE_SHA256.json) · [协议标识符](src/evi_hgt_ranking_comparison/protocol_ids.py)

在仓库根目录运行以下合成 CPU 测试，检查指标对同分和 P/U 的处理、候选成员错误以及只用训练正例计算的频次。测试需要 Python 和 pytest。

```sh
python -m pytest -o pythonpath=experiments/comparisons/src experiments/comparisons/tests/ranking_comparison
```

有原始研究结果时，可以直接重新计算汇总，省去加载模型。传入 R0 审计 JSON 和 HGTDR R0F0 选定检查点的评分缓存。历史评分缓存逐行记录该折的候选身份。工具会校验缓存，再从这些行重算候选集摘要。

```sh
python experiments/comparisons/summarize_r0.py /path/to/metrics_audit.json /path/to/epoch_0222/validation.ROWS.json
```

具名归档的训练源码涵盖研究实验的图构建、训练和检查点选择。归档中给研究包、存储路径、回执字段和检查点参数使用了描述性别名。历史清单和检查点仍保存原名和校验值。各具名训练入口对照登记摘要核验原始研究脚本，再在指定的研究检出目录中运行。调度器按原始规则核验清单、权限和源码。以下命令可查看调度器的参数。

```sh
python experiments/comparisons/scripts/ranking_comparison/runtime.py --research-root /path/to/research --help
```

要从 Git 检出目录还原对应源码和入口，在仓库根目录运行以下命令，并选用一个新的输出目录。

```sh
python experiments/replay.py --output NEW_DIRECTORY
```

还原后的对比实验 README 列出原始入口。重放某项原始任务需要对应的研究检出目录、登记的折清单、图与特征缓存、采样计划、原生运行环境以及任务权限。部分输入受再分发限制。D1 上游预训练是否接触过目标关系，尚未核实。 [发布版数据约定](../../docs/data.md) · [原生模型适配](../native_adapters/README.zh-CN.md) · [早期单折试验](../pilot/README.zh-CN.md)
