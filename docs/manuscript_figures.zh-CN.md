<a id="manuscript-figures"></a>

# 论文图件

<sub>[English](manuscript_figures.md)</sub>

本图集对应 2026 年 11 月 28 日修订稿的图 1–11。各 SVG 是相应语言稿件中实际嵌入的矢量图，图中文字转为轮廓以保留原有字形。各图另有 PNG 预览和单页 PDF。 [文件校验值](figures/manuscript/manifest.json)

图 1 和图 2 沿用原图的算子标注。图 2 中的 `src` 和 `dst` 分别表示源节点和目标节点，`Drop.` 是 `Dropout` 的缩写。两种语言版本采用相同的数据和评估口径，方法标识、指标名称和数学符号保持一致。

分析图集使用独立的 `fig01_...` 至 `fig16_...` 文件名，另收录其他分析视图和早期实验。 [分析图集](experiments.md)

| 图 | 内容 | 英文 | 中文 | 相关说明 |
| --- | --- | --- | --- | --- |
| 1 | 总体架构 | [SVG](figures/manuscript/figure01-en.svg) · [PNG](figures/manuscript/figure01-en.png) · [PDF](figures/manuscript/figure01-en.pdf) | [SVG](figures/manuscript/figure01-zh.svg) · [PNG](figures/manuscript/figure01-zh.png) · [PDF](figures/manuscript/figure01-zh.pdf) | [模型](model.md) |
| 2 | 证据注意力与路径推理 | [SVG](figures/manuscript/figure02-en.svg) · [PNG](figures/manuscript/figure02-en.png) · [PDF](figures/manuscript/figure02-en.pdf) | [SVG](figures/manuscript/figure02-zh.svg) · [PNG](figures/manuscript/figure02-zh.png) · [PDF](figures/manuscript/figure02-zh.pdf) | [模型](model.md) |
| 3 | R0 开发集评估协议 | [SVG](figures/manuscript/figure03-en.svg) · [PNG](figures/manuscript/figure03-en.png) · [PDF](figures/manuscript/figure03-en.pdf) | [SVG](figures/manuscript/figure03-zh.svg) · [PNG](figures/manuscript/figure03-zh.png) · [PDF](figures/manuscript/figure03-zh.pdf) | [评估协议](experiments.md#implemented-development-protocol) |
| 4 | 按疾病宏平均的 ROC 与 PR 曲线 | [SVG](figures/manuscript/figure04-en.svg) · [PNG](figures/manuscript/figure04-en.png) · [PDF](figures/manuscript/figure04-en.pdf) | [SVG](figures/manuscript/figure04-zh.svg) · [PNG](figures/manuscript/figure04-zh.png) · [PDF](figures/manuscript/figure04-zh.pdf) | [曲线](experiments.md#saved-prediction-roc-and-pr-curves) |
| 5 | 折内排序指标 | [SVG](figures/manuscript/figure05-en.svg) · [PNG](figures/manuscript/figure05-en.png) · [PDF](figures/manuscript/figure05-en.pdf) | [SVG](figures/manuscript/figure05-zh.svg) · [PNG](figures/manuscript/figure05-zh.png) · [PDF](figures/manuscript/figure05-zh.pdf) | [各折结果](experiments.md#fold-profiles) |
| 6 | 证据与路径覆盖率 | [SVG](figures/manuscript/figure06-en.svg) · [PNG](figures/manuscript/figure06-en.png) · [PDF](figures/manuscript/figure06-en.pdf) | [SVG](figures/manuscript/figure06-zh.svg) · [PNG](figures/manuscript/figure06-zh.png) · [PDF](figures/manuscript/figure06-zh.pdf) | [固定诊断实验](experiments.md#historical-diagnostic-exposure) |
| 7 | 各采样阶段的训练损失 | [SVG](figures/manuscript/figure07-en.svg) · [PNG](figures/manuscript/figure07-en.png) · [PDF](figures/manuscript/figure07-en.pdf) | [SVG](figures/manuscript/figure07-zh.svg) · [PNG](figures/manuscript/figure07-zh.png) · [PDF](figures/manuscript/figure07-zh.pdf) | [训练损失](experiments.md#training-loss-by-sampling-stage) |
| 8 | 逐疾病 AP 差值 | [SVG](figures/manuscript/figure08-en.svg) · [PNG](figures/manuscript/figure08-en.png) · [PDF](figures/manuscript/figure08-en.pdf) | [SVG](figures/manuscript/figure08-zh.svg) · [PNG](figures/manuscript/figure08-zh.png) · [PDF](figures/manuscript/figure08-zh.pdf) | [累计分布曲线](experiments.md#paired-disease-differences) |
| 9 | 按候选集大小分组的排序结果 | [SVG](figures/manuscript/figure09-en.svg) · [PNG](figures/manuscript/figure09-en.png) · [PDF](figures/manuscript/figure09-en.pdf) | [SVG](figures/manuscript/figure09-zh.svg) · [PNG](figures/manuscript/figure09-zh.png) · [PDF](figures/manuscript/figure09-zh.pdf) | [候选集分层](experiments.md#candidate-strata-and-disease-counts) |
| 10 | 禁忌证 AP 与汇总方式 | [SVG](figures/manuscript/figure10-en.svg) · [PNG](figures/manuscript/figure10-en.png) · [PDF](figures/manuscript/figure10-en.pdf) | [SVG](figures/manuscript/figure10-zh.svg) · [PNG](figures/manuscript/figure10-zh.png) · [PDF](figures/manuscript/figure10-zh.pdf) | [汇总口径](experiments.md#contra-ap-aggregation-across-methods) |
| 11 | 三个疾病的全候选集排名 | [SVG](figures/manuscript/figure11-en.svg) · [PNG](figures/manuscript/figure11-en.png) · [PDF](figures/manuscript/figure11-en.pdf) | [SVG](figures/manuscript/figure11-zh.svg) · [PNG](figures/manuscript/figure11-zh.png) · [PDF](figures/manuscript/figure11-zh.pdf) | [案例排名](experiments.md#case-rankings) |

<a id="editable-diagram-sources"></a>

## 可编辑图源

每个架构文件各有两页，对应图 1 和图 2。图 3 的评估协议有单独的图源。

- 图 1 和图 2。 [英文 Visio](figures/manuscript/architecture-en.vsdx) · [中文 Visio](figures/manuscript/architecture-zh.vsdx)
- 图 3。 [英文 Visio](figures/manuscript/protocol-en.vsdx) · [中文 Visio](figures/manuscript/protocol-zh.vsdx)

开发集结果表包含 HGTDR-adapted-contra。图 4、5、8、9 和 10 沿用原有对比方法，其中 HGTDR-adapted 提供治疗支持结果。单独训练的禁忌证模型在结果表中报告。 [开发集结果](../README.zh-CN.md#development-results)
