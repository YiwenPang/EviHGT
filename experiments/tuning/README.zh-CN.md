<a id="historical-learning-rate-selection"></a>

# 历史学习率选择

[English](README.md)

五档学习率实验使用随机种子 42。每个候选学习率下，FULL 将训练正例完整遍历两次。每个候选学习率取两个周期中较低的验证目标值，再从候选中选出目标值最小的一档。绝对差不超过 1e-8 时选较小的学习率。完成的候选为 2e-5、5e-5、1e-4、3e-4 和 1e-3。得到第一批结果后，搜索范围向较低学习率延伸。存储故障使 2e-4 的尝试在完成前失效，计划中的 5e-4 未执行。之后的消融实验对所有变体统一使用为 FULL 选出的 5e-5。该设计比较对参考配置的改动，FULL 获得的超参数选择机会更多。

五组配对验证读数可重现记录中的选择。报告程序校验运行记录并使用本地归档产物生成历史扫描图。公开副本默认写入研究工作区的 outputs 下的报告目录。源码清单记录报告程序的公开版和原版两份校验值。通用选模代码和验证驱动存放在历史消融归档中。[五组验证读数](validation_objectives.csv) · [报告程序](tools/lr_tuning_report.py) · [源码清单](source_sha256.json) · [通用选模代码](../ablations/historical/src/evi_hgt/training/model_selection.py) · [验证驱动](../ablations/historical/src/evi_hgt/training/formal_driver.py)

独立的内部开发集选模程序先按 support AUC、再按 AP 排列候选检查点，同分时取较早的训练步。该程序属于后续开发集协议，也为单折预实验提供了缓存对照。生成历史报告需要对应的研究软件包、运行清单和检查点。[内部开发集选模程序](src/evi_hgt/checkpoint_selection/selection.py) · [单折预实验](../pilot/README.zh-CN.md) · [历史验证图](../../docs/experiments.md#historical-validation-and-learning-rate-tuning)
