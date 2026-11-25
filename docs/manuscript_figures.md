# Manuscript figures

Figures 1–11 below follow the bilingual manuscript revised on 25 November 2026. The SVG files contain the exact vector artwork embedded in each language version. Glyphs are outlined to preserve the typography, and PNG files provide previews. [File checksums](figures/manuscript/manifest.json) identify this set.

The language columns identify the manuscript edition. Both editions retain the same English plot labels in Figures 4–11.

The [analysis gallery](experiments.md) retains its separate `fig01_...` to `fig16_...` filename series and includes additional views and historical studies. The last column links each manuscript figure to the relevant method or analysis.

| Figure | Subject | English | Chinese | Related explanation |
| --- | --- | --- | --- | --- |
| 1 | Overall architecture | [SVG](figures/manuscript/figure01-en.svg) · [PNG](figures/manuscript/figure01-en.png) | [SVG](figures/manuscript/figure01-zh.svg) · [PNG](figures/manuscript/figure01-zh.png) | [Model](model.md) |
| 2 | Evidence attention and path reasoning | [SVG](figures/manuscript/figure02-en.svg) · [PNG](figures/manuscript/figure02-en.png) | [SVG](figures/manuscript/figure02-zh.svg) · [PNG](figures/manuscript/figure02-zh.png) | [Model](model.md) |
| 3 | R0 development evaluation protocol | [SVG](figures/manuscript/figure03-en.svg) · [PNG](figures/manuscript/figure03-en.png) | [SVG](figures/manuscript/figure03-zh.svg) · [PNG](figures/manuscript/figure03-zh.png) | [Protocol](experiments.md#implemented-development-protocol) |
| 4 | Disease-macro ROC and PR curves | [SVG](figures/manuscript/figure04-en.svg) · [PNG](figures/manuscript/figure04-en.png) | [SVG](figures/manuscript/figure04-zh.svg) · [PNG](figures/manuscript/figure04-zh.png) | [Curves](experiments.md#saved-prediction-roc-and-pr-curves) |
| 5 | Within-fold ranking metrics | [SVG](figures/manuscript/figure05-en.svg) · [PNG](figures/manuscript/figure05-en.png) | [SVG](figures/manuscript/figure05-zh.svg) · [PNG](figures/manuscript/figure05-zh.png) | [Fold profiles](experiments.md#fold-profiles) |
| 6 | Evidence and path coverage | [SVG](figures/manuscript/figure06-en.svg) · [PNG](figures/manuscript/figure06-en.png) | [SVG](figures/manuscript/figure06-zh.svg) · [PNG](figures/manuscript/figure06-zh.png) | [Fixed diagnostics](experiments.md#historical-diagnostic-exposure) |
| 7 | Training loss by sampling stage | [SVG](figures/manuscript/figure07-en.svg) · [PNG](figures/manuscript/figure07-en.png) | [SVG](figures/manuscript/figure07-zh.svg) · [PNG](figures/manuscript/figure07-zh.png) | [Training loss](experiments.md#training-loss-by-sampling-stage) |
| 8 | Per-disease AP differences | [SVG](figures/manuscript/figure08-en.svg) · [PNG](figures/manuscript/figure08-en.png) | [SVG](figures/manuscript/figure08-zh.svg) · [PNG](figures/manuscript/figure08-zh.png) | [Cumulative curves](experiments.md#paired-disease-differences) |
| 9 | Ranking by candidate-set size | [SVG](figures/manuscript/figure09-en.svg) · [PNG](figures/manuscript/figure09-en.png) | [SVG](figures/manuscript/figure09-zh.svg) · [PNG](figures/manuscript/figure09-zh.png) | [Candidate strata](experiments.md#candidate-strata-and-disease-counts) |
| 10 | Contra AP and aggregation | [SVG](figures/manuscript/figure10-en.svg) · [PNG](figures/manuscript/figure10-en.png) | [SVG](figures/manuscript/figure10-zh.svg) · [PNG](figures/manuscript/figure10-zh.png) | [Aggregation](experiments.md#contra-ap-aggregation-across-methods) |
| 11 | Full candidate rankings for three diseases | [SVG](figures/manuscript/figure11-en.svg) · [PNG](figures/manuscript/figure11-en.png) | [SVG](figures/manuscript/figure11-zh.svg) · [PNG](figures/manuscript/figure11-zh.png) | [Case rankings](experiments.md#case-rankings) |

## Editable diagram sources

Each architecture file contains two pages for Figures 1 and 2: [English Visio](figures/manuscript/architecture-en.vsdx) and [Chinese Visio](figures/manuscript/architecture-zh.vsdx). Figure 3 has its own [English Visio](figures/experiments/fig01_evaluation_protocol.vsdx) and [Chinese Visio](figures/experiments/fig01_evaluation_protocol_zh.vsdx) files.

The [development result table](../README.md#development-results) includes HGTDR-adapted-contra. Figures 4, 5, 8, 9 and 10 retain the original comparison group; its HGTDR-adapted model contributes support results. The separately trained contraindication model is reported in the table.
