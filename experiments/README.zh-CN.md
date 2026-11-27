<a id="experiment-code"></a>

# 实验代码

[English](README.md)

<a id="run-from-the-source-databases"></a>

## 从原始数据库运行

下载四个原始文件后，在仓库根目录运行以下命令。准备阶段会构建规范化图谱、数据划分、候选文件、特征和训练计划。首次联网运行时，程序会下载缓存中缺少的指定版本 Open Targets 文件和编码器资源。[数据下载与放置](../docs/data.md#download-and-place-the-source-databases) · [安装环境](../README.zh-CN.md#installation)

用 `requirements-preprocess.txt` 安装的依赖包含实验协调程序所需的 `filelock`。匹配对照和 HGTDR 使用 `requirements-experiments.txt` 中的依赖。HGTDR 还需要与平台匹配的 `torch-sparse` 和 `torch-scatter`。TxGNN 使用独立的 CPython 3.11 环境和固定版本的源码检出。[原生环境安装与源码缓存](workflow/native_sources/README.zh-CN.md)

| 命令 | 默认任务 |
| --- | --- |
| `python main.py --experiment ablations` | 在 `R0F1` 和 `R1F0` 上运行 `FULL`、`CONTENT_ZERO` 和 `FACT_QUALITY_CONSTANT`。[证据消融](ablations/README.zh-CN.md) |
| `python main.py --experiment losses` | 在 `R0F0`、`R0F1` 和 `R1F0` 上分别使用 `softplus` 和 `sigmoid` 训练 FULL。其中 `R0F0` 使用 `DEV95` 配方。[损失对照](ablations/README.zh-CN.md) |
| `python main.py --experiment comparisons` | 在 `R0F0` 至 `R0F4` 上运行 FULL 和六种基线。[基线方法与 ID](comparisons/README.zh-CN.md) |
| `python main.py --experiment cold` | 运行药物冷启动、疾病冷启动、双冷启动和按首次获批年份划分的实验。疾病冷启动还在共同的测试候选上比较 FULL 与 HGTDR 的治疗支持和禁忌证模型。[冷启动协议](cold_start/README.zh-CN.md) |
| `python main.py --experiment all` | 依次运行四组实验。消融和损失对照中配置及数据折相同的实验臂只运行一次。 |

消融、损失对照和基线对比都可用 `--cells` 选择数据折。`ablations` 用 `--variants` 选变体，`comparisons` 用 `--methods` 选方法，`cold` 用 `--cases` 选场景。运行 `all` 时，每个参数只筛选对应的实验组。多个 ID 用空格分隔。在冷启动实验命令中加上 `--skip-cold-hgtdr`，即可单独运行 EviHGT 的冷启动场景。

```sh
python main.py --experiment ablations --cells R0F1 --variants FULL CONTENT_ZERO
python main.py --experiment losses --cells R0F0
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease
```

使用 `--mode prepare` 构建并校验所选任务的输入。默认的 `train` 模式接着执行训练、评估和报告生成。派生语料存放在 `data/processed/experiments/`，可共用的准备阶段产物存放在 `data/interim/experiments/`。通过 `--config` 指定运行配置，通过 `--raw-config` 指定原始数据路径或特征准备选项。指定版本的资源已缓存时可使用 `--offline`。TxGNN 可用 `--txgnn-python` 指定解释器，用 `--txgnn-source` 指定通过校验的本地源码检出。[原始数据配置](../docs/data.md) · [TxGNN 环境](workflow/native_sources/README.zh-CN.md)

默认运行目录为 `outputs/experiments/<timestamp>_<experiment>/`。通过 `--run-dir outputs/NAME` 可指定固定目录。以相同选项重复执行命令时，程序会校验并复用已完成的任务。要从最新检查点接续未完成的任务，请加上 `--resume` 并使用同一目录。

```sh
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease --resume
```

任务结束后打开 `report.html`。`summary.json` 记录运行信息与汇总指标，`metrics.csv` 保存指标汇总，`disease_metrics.csv` 保存逐疾病结果。对于按折运行的实验，先对每种疾病在可评估折中的指标取平均，再对疾病做宏平均。冷启动实验分别报告各自评估面板的指标。

<a id="cpu-smoke-runs"></a>

## CPU 快速自检

快速自检在 CPU 上使用小规模合成数据，每组实验默认运行一个数据折或场景，并使用所选方法或变体。输出标记为 `SYNTHETIC_TEST_ONLY`。

```sh
python main.py --experiment comparisons --smoke --methods FULL TRAIN_VISIBLE_POPULARITY
python main.py --experiment cold --smoke --cases cold_disease --skip-cold-hgtdr
```

快速自检若包含匹配对照、HGTDR 或 TxGNN，仍需安装相应的库。TxGNN 还需要通过校验的本地源码检出或缓存。[原生环境安装](workflow/native_sources/README.zh-CN.md)

<a id="run-the-public-checks"></a>

## 运行公开检查

完成 [CPU 环境安装](../README.zh-CN.md#try-the-public-checks) 后，在仓库根目录执行以下命令。

```sh
python experiments/run_checks.py
```

该命令运行公开测试套件，检查五折台账与消融源码校验值，再训练并评估一个小型合成冷启动实例。测试数据来自本地或在运行时生成。要单独查看冷启动实例的输出，可运行 `python -B experiments/cold_start/synthetic/test_frozen_test_score.py`。程序会打印含检查点和 `test/metrics.json` 的临时目录。

只安装 Python 也可以运行 `python experiments/audit/verify_r0.py` 检查公开的分折台账，或运行 `python experiments/ablations/verify.py sources` 对照记录的校验值验证公开消融源码。

<a id="inputs-for-the-paper-experiments"></a>

## 论文实验的输入

归档运行流程使用下列输入，并保留原始产物标识、检查点和分析记录。

| 实验 | 评估内容 | 归档流程所需输入 |
| --- | --- | --- |
| [基线对比](comparisons/README.zh-CN.md) | 在 R0 五折上比较六种方法，包括按训练频次评分的 Popularity 基线。[分折台账](audit/README.zh-CN.md) 记录各方法的输出。 | 对应版本的研究仓库、分折清单、符合协议的训练图、特征及采样计划。HGTDR 和 TxGNN 还需要各自的原生环境。复用的 HGTDR 数据折需要所选检查点。 |
| [消融实验](ablations/README.zh-CN.md) | 训练损失与证据干预的配对对照，以及早期的六变体留出评估。 | 对应版本的研究仓库、冻结的训练计划、图、特征与评估面板。配对审计读取原始更新日志，留出分析读取已保存的预测和运行记录。 |
| [冷启动](cold_start/README.zh-CN.md) | 药物冷启动、疾病冷启动、双冷启动和按首次获批年份划分的队列。 | 已准备的研究图谱和证据、划分与排除清单，以及冻结的药物和疾病特征。按获批年份划分时还需要药物首次获批年份字段。 |
| [疾病冷启动与 HGTDR 对比](cold_start/hgtdr/README.zh-CN.md) | 在同一批留出疾病和测试候选上评估 EviHGT 和分别训练的两个 HGTDR 变体。 | 已准备的疾病冷启动语料、已完成的 EviHGT 疾病冷启动运行，以及绑定到对应源码版本的 HGTDR CUDA 运行时和特征包。汇总结果可直接查阅。 |
| [HGTDR-adapted-contra](hgtdr_contra/README.zh-CN.md) | 单独训练的单输出模型，使用已记录的禁忌证拟合，在 R0 五个共同的内层开发面板上评估。 | 原始 R0 数据折、特征与图缓存、对应的研究模块，以及本地 CUDA 版 HGTDR 重建实现。 |

原始授权记录、已准备的数据折、逐药病对预测和训练权重均保存在仓库之外。若要用自己的规范化数据训练可移植版 EviHGT，请按[数据指南](../docs/data.md)准备输入。

若不重新训练，复算原始 R0 对比需要原始逐疾病结果、审计索引和已记录的 HGTDR-adapted 分数缓存。HGTDR-adapted-contra 使用单独运行得到的五折输出进行汇总。[对比汇总脚本](comparisons/summarize_r0.py) · [HGTDR 与 TxGNN 原生集成](native_adapters/README.zh-CN.md)

[单折先导实验](pilot/README.zh-CN.md) 记录早期使用共同候选的治疗支持对比。[学习率实验](tuning/README.zh-CN.md) 记录 FULL 完成的五种学习率设置，以及如何依据验证目标选择学习率。

[实验图与结果](../docs/experiments.md) · [论文中英双语图](../docs/manuscript_figures.md) · [可移植训练与评估](../docs/reproducibility.md)

<a id="restore-the-archived-source-layout"></a>

## 恢复归档源码布局

历史运行凭证绑定当时使用的确切源码路径及校验值。在包含归档提交的 Git 克隆中运行以下命令，可将早期公开的源码树恢复到新目录。

```sh
python experiments/replay.py --output ../evihgt-replay-source
```

该命令恢复公开源码。若要按原来的产物标识回放某个任务，请使用该任务的归档流程和配套输入。
