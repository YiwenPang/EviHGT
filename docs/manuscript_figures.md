# Manuscript figures

<sub>[简体中文](manuscript_figures.zh-CN.md)</sub>

This set follows Figures 1–11 in the 28 November 2026 manuscript revision. Each SVG is the vector artwork embedded in the corresponding language edition, with text converted to outlines to preserve typography. PNG files provide previews, and each PDF contains one figure on one page. [File checksums](figures/manuscript/manifest.json)

Figures 1 and 2 retain the original operator labels. In Figure 2, `src` and `dst` mark the source and destination nodes, and `Drop.` stands for `Dropout`. Method identifiers, metric names and mathematical notation are shared across the language versions. Corresponding figures use the same data and evaluation definitions.

The analysis gallery includes additional views and historical studies under its own `fig01_...` to `fig16_...` filenames. [Analysis gallery](experiments.md)

| Figure | Subject | English | Chinese | Related explanation |
| --- | --- | --- | --- | --- |
| 1 | Overall architecture | [SVG](figures/manuscript/figure01-en.svg) · [PNG](figures/manuscript/figure01-en.png) · [PDF](figures/manuscript/figure01-en.pdf) | [SVG](figures/manuscript/figure01-zh.svg) · [PNG](figures/manuscript/figure01-zh.png) · [PDF](figures/manuscript/figure01-zh.pdf) | [Model](model.md) |
| 2 | Evidence attention and path reasoning | [SVG](figures/manuscript/figure02-en.svg) · [PNG](figures/manuscript/figure02-en.png) · [PDF](figures/manuscript/figure02-en.pdf) | [SVG](figures/manuscript/figure02-zh.svg) · [PNG](figures/manuscript/figure02-zh.png) · [PDF](figures/manuscript/figure02-zh.pdf) | [Model](model.md) |
| 3 | R0 development evaluation protocol | [SVG](figures/manuscript/figure03-en.svg) · [PNG](figures/manuscript/figure03-en.png) · [PDF](figures/manuscript/figure03-en.pdf) | [SVG](figures/manuscript/figure03-zh.svg) · [PNG](figures/manuscript/figure03-zh.png) · [PDF](figures/manuscript/figure03-zh.pdf) | [Protocol](experiments.md#implemented-development-protocol) |
| 4 | Disease-macro ROC and PR curves | [SVG](figures/manuscript/figure04-en.svg) · [PNG](figures/manuscript/figure04-en.png) · [PDF](figures/manuscript/figure04-en.pdf) | [SVG](figures/manuscript/figure04-zh.svg) · [PNG](figures/manuscript/figure04-zh.png) · [PDF](figures/manuscript/figure04-zh.pdf) | [Curves](experiments.md#saved-prediction-roc-and-pr-curves) |
| 5 | Within-fold ranking metrics | [SVG](figures/manuscript/figure05-en.svg) · [PNG](figures/manuscript/figure05-en.png) · [PDF](figures/manuscript/figure05-en.pdf) | [SVG](figures/manuscript/figure05-zh.svg) · [PNG](figures/manuscript/figure05-zh.png) · [PDF](figures/manuscript/figure05-zh.pdf) | [Fold profiles](experiments.md#fold-profiles) |
| 6 | Evidence and path coverage | [SVG](figures/manuscript/figure06-en.svg) · [PNG](figures/manuscript/figure06-en.png) · [PDF](figures/manuscript/figure06-en.pdf) | [SVG](figures/manuscript/figure06-zh.svg) · [PNG](figures/manuscript/figure06-zh.png) · [PDF](figures/manuscript/figure06-zh.pdf) | [Fixed diagnostics](experiments.md#historical-diagnostic-exposure) |
| 7 | Training loss by sampling stage | [SVG](figures/manuscript/figure07-en.svg) · [PNG](figures/manuscript/figure07-en.png) · [PDF](figures/manuscript/figure07-en.pdf) | [SVG](figures/manuscript/figure07-zh.svg) · [PNG](figures/manuscript/figure07-zh.png) · [PDF](figures/manuscript/figure07-zh.pdf) | [Training loss](experiments.md#training-loss-by-sampling-stage) |
| 8 | Per-disease AP differences | [SVG](figures/manuscript/figure08-en.svg) · [PNG](figures/manuscript/figure08-en.png) · [PDF](figures/manuscript/figure08-en.pdf) | [SVG](figures/manuscript/figure08-zh.svg) · [PNG](figures/manuscript/figure08-zh.png) · [PDF](figures/manuscript/figure08-zh.pdf) | [Cumulative curves](experiments.md#paired-disease-differences) |
| 9 | Ranking by candidate-set size | [SVG](figures/manuscript/figure09-en.svg) · [PNG](figures/manuscript/figure09-en.png) · [PDF](figures/manuscript/figure09-en.pdf) | [SVG](figures/manuscript/figure09-zh.svg) · [PNG](figures/manuscript/figure09-zh.png) · [PDF](figures/manuscript/figure09-zh.pdf) | [Candidate strata](experiments.md#candidate-strata-and-disease-counts) |
| 10 | Contra AP and aggregation | [SVG](figures/manuscript/figure10-en.svg) · [PNG](figures/manuscript/figure10-en.png) · [PDF](figures/manuscript/figure10-en.pdf) | [SVG](figures/manuscript/figure10-zh.svg) · [PNG](figures/manuscript/figure10-zh.png) · [PDF](figures/manuscript/figure10-zh.pdf) | [Aggregation](experiments.md#contra-ap-aggregation-across-methods) |
| 11 | Full candidate rankings for three diseases | [SVG](figures/manuscript/figure11-en.svg) · [PNG](figures/manuscript/figure11-en.png) · [PDF](figures/manuscript/figure11-en.pdf) | [SVG](figures/manuscript/figure11-zh.svg) · [PNG](figures/manuscript/figure11-zh.png) · [PDF](figures/manuscript/figure11-zh.pdf) | [Case rankings](experiments.md#case-rankings) |

## Editable diagram sources

Each architecture file contains two pages for Figures 1 and 2. Figure 3 has its own evaluation protocol diagram.

- Figures 1 and 2. [English Visio](figures/manuscript/architecture-en.vsdx) · [Chinese Visio](figures/manuscript/architecture-zh.vsdx)
- Figure 3. [English Visio](figures/manuscript/protocol-en.vsdx) · [Chinese Visio](figures/manuscript/protocol-zh.vsdx)

The development result table includes HGTDR-adapted-contra. Figures 4, 5, 8, 9 and 10 retain the original comparison group, with HGTDR-adapted contributing support results. The table reports the separately trained contraindication model. [Development results](../README.md#development-results)
