<a id="cold-start-experiments"></a>

# 冷启动实验

[English](README.md)

本目录按 2026 年 11 月 18 日的协议构建并评估四种实体留出场景，每个场景单独初始化 EviHGT FULL。[实验图集](../../docs/experiments.md#historical-holdout-evaluation)中较早的 support-only cold-drug 和 cold-disease 图使用另一套评估协议。

<a id="run-from-the-four-source-files"></a>

## 从四个原始文件运行

按照仓库首页放置四个原始文件后，在仓库根目录运行下列命令。入口会重建各场景所需的图、固定特征、训练计划和留出面板。批准年份读取 OptimusKG 2.0 聚合节点文件 `nodes.parquet` 中药物的 `properties.year_of_first_approval` 字段。[原始文件下载与放置](../../README.zh-CN.md#下载并放置四个原始文件)

```sh
python main.py --experiment cold
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease
python main.py --experiment cold --smoke --cases cold_disease --skip-cold-hgtdr
```

默认运行四个场景，并在 `cold_disease` 的同一测试面板上加跑 HGTDR 的 support 与 contra 模型。HGTDR 使用与 PyTorch 版本匹配的 `torch-sparse`、`torch-scatter` 及 CUDA 训练环境。`--skip-cold-hgtdr` 可直接运行 EviHGT 的冷启动任务。第三条命令使用合成数据检查 CPU 流程。[HGTDR 安装](../workflow/native_sources/README.zh-CN.md)

用 `--mode prepare` 只准备所选场景。实验目录中的 `report.html` 和 `summary.json` 汇总结果，`metrics.csv` 和 `disease_metrics.csv` 提供指标明细。中断后用原有选项、同一个 `--run-dir` 和 `--resume` 继续。[运行目录与恢复参数](../README.zh-CN.md#run-from-the-source-databases)

<a id="splits-and-scoring"></a>

## 划分与评分

| 场景 | 留出单位 |
| --- | --- |
| cold_drug | 规范化药物 ID |
| cold_disease | 规范化疾病 ID |
| double_cold | 药物和疾病 ID；一对实体只有在两端均属于同一分区时才进入该分区。 |
| first_approval_year_cohort | 按首次批准年份划分药物，完整年份保持在一起；未记录年份的药物不参与该场景。 |

前三种场景对相应的规范化实体 ID 做哈希划分，训练、验证和测试的名义比例为 80%、10% 和 10%。正例来自已审计的开发池和辅助池，原先受保护的实体对继续排除。批准年份场景使用静态图和特征快照，衡量不同批准年份队列间的迁移。

图构建阶段删除与留出端点关联的事实、证据记录和 CSR 遍历边，并检查两个方向及已知的目标等价事实。端点保留允许使用的静态特征，包括冻结的 D1 特征，以及非临床的药物–基因、疾病–表型摘要。上游编码器预训练是否接触目标信息仍为 NOT_VERIFIED。

每个场景使用 seed 42、冻结的 FULL 配置和三个训练周期。训练器接收不含角色标签的排除列表；单独的完整性检查读取留出面板，测试角色文件不进入训练器输入清单。测试评分使用固定的最后一步 checkpoint。每个预测头在每种疾病内，将其记录正例与同一组至多 64 个抽样未标注药物比较。输出包含逐疾病排名指标、恒定得分与并列计数、图覆盖率。每个场景运行一次 FULL；[HGTDR 比较](hgtdr/README.zh-CN.md)使用相同的 cold-disease 划分和测试面板，各模型分别训练一次。

<a id="source-and-runtime-identity"></a>

## 源码与运行时身份

- [prepare_case.py](prepare_case.py)构建划分、合法未标注候选和固定训练计划；[approval_cohort.py](approval_cohort.py)定义年份边界。
- [graph_builder.py](graph_builder.py)物化屏蔽后的事实、证据和邻接结构。
- [run_case.py](run_case.py)检查成员关系、训练、校验 checkpoint 并评分留出查询；[run_queue.py](run_queue.py)依次运行四个场景。
- [source_sha256.json](source_sha256.json)保留研究脚本、测试及其最初公开副本按 LF 换行规范化后的 SHA-256。

公开适配器通过 [locations.py](locations.py)替换工作站路径，将源代码与运行产物分开，冷启动合成测试使用系统临时目录。原始队列校验包含文档和图片的 130 个冻结文件。运行时检查另有以下三份完整清单：

| 返回身份 | 清单 | 对应版本 |
| --- | --- | --- |
| historical-release | [runtime_sha256.json](runtime_sha256.json) | 51 个 Python 源文件、两个 YAML 配置和原合成数据 fixture。 |
| raw-preparation-release | [runtime_sha256_raw_preparation.json](runtime_sha256_raw_preparation.json) | 84 个 Python 源文件、三个 YAML 配置和同一 fixture，共 88 个文件；增加原始数据库准备流程。 |
| full-workflow-release | [runtime_sha256_full_workflow.json](runtime_sha256_full_workflow.json) | 90 个 Python 源文件、四个 YAML 配置和同一 fixture，共 95 个文件；增加最终排名、路径解释、结果报告和独立 tiny 自检入口。 |

check_frozen() 要求实际文件名集合与某一份清单完全一致，且每个文件的 LF 规范化哈希都匹配该清单。额外模块、缺失文件、内容改动和不同版本的混合均无法通过。回执保留 runtime_variant 和 runtime_manifest；后两种身份的 historical_runtime_identical 为 False。旧两份清单保持原有字节。

raw-preparation-release 在历史文件中修改了 CLI 的准备分派和语料缺失提示，其数值模型与训练源文件仍匹配历史接受哈希。full-workflow-release 的 runner 在最终开发评估后导出 rankings/、explanations/index.json、summary.json 和 report.html，并提供 main.py --smoke。模型结构、损失、优化器更新、三个训练周期及评分公式沿用既有实现；路径解释读取模型前向计算中的权重和事实逻辑。

历史比较结果及先前的 Python 语法树比较（去除 docstring）和 YAML 内容比较，仍对应原实验使用的历史文件身份。复现这些比较时，应使用原收据绑定的源码、输入和 checkpoint。[源码恢复命令](../README.zh-CN.md#restore-the-archived-source-layout)继续恢复其固定 Git revision，source_sha256.json 中的历史脚本哈希保持不变。

run_case.py 在目录包含判断前解析 case 根路径，使 Windows 短路径别名和等价相对路径指向同一目录。输入哈希及拒绝 case 外部路径的检查继续生效。[source_sha256_windows_paths.json](source_sha256_windows_paths.json)单独记录该适配器及回归测试；这项路径处理变更不改变模型、成员关系、训练预算或评分公式。

<a id="software-checks"></a>

## 软件自检

安装仓库依赖和 pytest，使用 Python 3.11 或更高版本，在仓库根目录执行：

    python -B -m pytest experiments/cold_start -q
    python -B experiments/cold_start/synthetic/test_frozen_test_score.py
    python main.py --smoke

第一条命令在合成输入上检查实体分配、抽样排除、图屏蔽、成员关系、并列得分和三种运行时身份。第二条执行冷启动的三步 CPU 训练及独立留出评估。

第三条通过主入口执行普通工作流的 tiny 衔接测试：固定 CPU/FP32、15 个合成节点和三次真实参数更新，随后生成开发评估、checkpoint、排名、路径解释及报告。该入口离线运行，使用自带的合成输入和配置。语料写入 data/processed/smoke/&lt;uuid&gt;/，运行产物写入 outputs/smoke/&lt;uuid&gt;/&lt;run_id&gt;/，输入和完成记录明确标为 SYNTHETIC_TEST_ONLY。这两种 tiny 测试各自验证对应的软件流程，合成结果不属于论文实验结果。

<a id="running-with-licensed-inputs"></a>
<a id="使用授权数据运行"></a>

<a id="archived-research-workspace"></a>

## 归档研究工作区入口

下列归档命令使用原研究工作区的派生输入和哈希清单。清单中的路径须能在工作区内解析。归档准备阶段读取以下文件。

- 规范化节点索引、事实和证据表、目标映射及图 CSR 文件。
- 开发集与 U manifest、正例成员、已观测和禁止实体对，以及静态特征。
- 已核验的疾病–基因图、证据适配器及图验收记录。
- 冻结的 D1 特征包；批准年份场景还需要 OptimusKG 2.0 中按节点类型划分的药物节点 Parquet 中的结构化首次批准年份字段。详见[输入约定](../../docs/data.md)。

在仓库内设置本地研究工作区和 D1 特征路径。PowerShell 示例：

    $env:EVIHGT_RESEARCH_ROOT = (Resolve-Path '../EviHGT-research').Path
    $env:EVIHGT_D1_FEATURES = (Resolve-Path '../HGTDR/data/processed/competition_pilot_v1/features').Path
    python -B experiments/cold_start/run_queue.py

当前公开仓库默认匹配 full-workflow-release。设置 EVIHGT_FROZEN_ROOT 可使用另行归档、与其他注册清单完全匹配的运行时。也可以在新的运行目录中单独准备、校验并训练一个场景。HGTDR 比较使用 cold_disease 场景，并将准备好的语料及已完成的运行交给[比较入口](hgtdr/README.zh-CN.md)。

    python -B experiments/cold_start/prepare_case.py --case cold_drug
    python -B experiments/cold_start/run_case.py --case cold_drug --mode validate
    python -B experiments/cold_start/run_case.py --case cold_drug --mode train

[locations.py](locations.py)中的工作区布局对应原 manifest 记录的路径。派生输入按场景分别放在 CACHE 下，训练和评估记录以相同场景名写入 EXPERIMENT/runs。授权图记录、成员文件、特征、预测和 checkpoint 均保留在本地。
