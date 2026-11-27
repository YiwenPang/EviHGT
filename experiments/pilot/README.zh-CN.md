<a id="october-common-development-pilot"></a>

# 十月共用开发集预实验

[English](README.md)

预实验让 HGTDR、TxGNN 和一个已有的 EviHGT 检查点使用同一张屏蔽图和同一组 support 候选。实验采用一个折和一个随机种子，两种原生模型各试三个学习率。HGTDR 使用 CUDA，共训练 300 个周期，每个周期结束后在开发集评分。TxGNN 使用 CPU，先预训练一个周期，再微调 500 个周期，每隔 25 个周期在开发集评估一次，用于模型选择。缓存的 EviHGT 结果来自更早的一次开发集选模。各方法沿用各自的损失、输入和选模预算。该实验的 support ROC-AUC 从高到低依次为 HGTDR、EviHGT 和 TxGNN。

预实验设置保留已记录的科研参数，省去了工作站路径。EviHGT 源码按数据导出、评分与执行组织。数据导出构建共用图、候选与抽样规则。协议检查成员关系并选择候选。运行脚本检查各基线的回执、检查点身份、保存的分数和指标。各基线的代码位于原生适配器目录。[预实验设置](settings.json) · [EviHGT 源码](EviHGT/) · [数据导出](EviHGT/src/evi_hgt/competition/data.py) · [实验协议](EviHGT/src/evi_hgt/competition/protocol.py) · [运行脚本](EviHGT/src/evi_hgt/competition/runner.py) · [原生适配器](../native_adapters/README.zh-CN.md)

两个基线任务均已完成。研究工作区入口中一处与实验无关的分派分支改动，导致上层运行脚本的最终源码校验失败。公开入口提供预实验命令，源码重放恢复原先启动时使用的完整分派表。后续审计重新核对了六个学习率候选和缓存对照，上层运行记录仍为失败状态。[公开入口](EviHGT/main.py) · [源码重放](../replay.py)

校验和清单标识公开文件。公开副本调整过文件布局和表述，原始校验和用于标识源码重放中的版本。完整重放原 `--competition` 命令需要原研究软件包、准备好的共用图、本地模型源码、运行环境、实验协议，以及用作对照的检查点。[公开文件校验和](source_sha256.txt) · [原始校验和](archived_sha256.json)
