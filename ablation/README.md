# HMoE-SAM3 科学消融实验

本目录提供一个独立、可复现且不会修改原始数据路径的消融框架。训练入口仍是
`MedSAM3-main/train_moe_sam3.py`；这里不复制训练逻辑，只负责：

1. 从真实 Stage-5 配置继承并生成每个方法 × seed 的完整 YAML；
2. 强制保护 `dataset/model/svanet` 下所有 root、JSON、threshold file 与 checkpoint；
3. 顺序或多 GPU 并行运行，保存独立日志和可恢复的运行状态；
4. 汇总相同 seed 的配对差值、bootstrap 置信区间、sign-flip 检验和 Holm 多重比较校正。

`run.sh` 是远端正式启动入口，环境和绝对路径与顶层 `train.sh` 保持一致；Python CLI 仍可用于
配置审查、生成及结果汇总。

## 快速使用

服务器上直接启动默认 `primary` suite（每个实验使用 GPU 0/1 做双卡 DDP）：

```bash
bash ablation/run.sh
```

常用环境变量示例：

```bash
# 仅打印 smoke suite 将执行的命令，不启动训练
SUITE=smoke DEVICE_SETS="0,1" DRY_RUN=1 bash ablation/run.sh

# GPU 0、1 各运行一个独立单卡实验，使用 study.yaml 的默认 seeds
SUITE=primary DEVICE_SETS="0 1" bash ablation/run.sh

# 两组双卡实验并行，并将 seeds 显式扩展为 5 个
DEVICE_SETS="0,1 2,3" SEEDS="13 29 42 73 101" bash ablation/run.sh

# 只运行 experiment id 匹配 confidence 的实验
SUITE=all INCLUDE=confidence bash ablation/run.sh
```

支持的环境变量为 `STUDY`、`SUITE`、`DEVICE_SETS`、`SEEDS`、`INCLUDE`、`MAX_PARALLEL`、
`MASTER_PORT`、`DRY_RUN`、`RERUN_COMPLETED` 和 `FAIL_FAST`。默认 `FAIL_FAST=1`；状态文件支持
再次执行同一命令时自动跳过已经完成的实验。

以下命令均在 `sam3-finetune/` 仓库根目录下执行：

```bash
python ablation/run_ablation.py validate --suite primary
python ablation/run_ablation.py list --suite primary
python ablation/run_ablation.py generate --suite primary --seeds 13 42 73
```

先做不启动训练的命令审查：

```bash
python ablation/run_ablation.py run \
  --suite smoke \
  --device-sets 0 \
  --dry-run
```

单卡依次完成 primary suite：

```bash
python ablation/run_ablation.py run \
  --suite primary \
  --device-sets 0
```

四张卡并发四个独立的单卡实验：

```bash
python ablation/run_ablation.py run \
  --suite primary \
  --device-sets 0 1 2 3
```

两组双卡 DDP 实验并发运行：

```bash
python ablation/run_ablation.py run \
  --suite primary \
  --device-sets 0,1 2,3
```

`state/primary.json` 记录完成状态；相同命令会跳过已完成的 cell。只有明确传入
`--rerun-completed` 才会重复运行。训练输出根目录默认沿用基础配置中 output 的父目录，
其下新增 `ablation_hmoe_v2/<suite>/<experiment>/seed_<seed>/`。如果服务器输出盘不同，只能用
`--output-root` 改训练输出，不能在消融 override 中改数据或权重路径。

完成后汇总：

```bash
python ablation/summarize_ablation.py \
  --suite primary \
  --epoch best
```

汇总器直接兼容训练器的原生写盘格式：`router_statistics.json` 为 epoch 记录组成的 JSON
数组，`val_stats.json` 为逐行 JSON。它会在离线环境从原始计数派生 SvANet trigger、skip、
locator/box/full-image fallback 比例和 prompt 比例，无需依赖 W&B。

输出包括：

- `reports/primary/per_seed_metrics.csv`：每个 seed、每个指标；
- `reports/primary/aggregate_metrics.csv`：均值、标准差、seed bootstrap 95% CI；
- `reports/primary/paired_comparisons.csv`：相对 `full_v2` 的相同-seed配对差值；
- `reports/primary/report.md`：主指标表与缺失运行审计；
- `reports/primary/missing_runs.json`：未完成 cell。

## 实验设计

`study.yaml` 的 `full_v2` 是唯一参照；其他 primary 实验均继承它并尽量只改变一个
方法因素。这样可将差异归因于单一设计，而不是不同数据、epoch、优化器或初始化集合。

| 问题 | 主比较 | 结论边界 |
|---|---|---|
| 是否真正需要条件式层级路由 | `parallel_router` vs `full_v2` | 只改变 child 是否依赖 modality parent |
| 图像-only推理是否仍能正确路由 | `prompt_conditioned_no_locator`, `prompt_rich_training`, `image_only_training` | 分开研究 locator、提示课程和纯无框训练 |
| 路由是否真正与样本提示解耦 | `decoder_memory_routing_source` vs `full_v2` | `backbone_fpn` 位于提示融合前；`decoder_memory` 是旧路径及潜在泄漏诊断 |
| 不确定样本怎样执行专家 | `fixed_top1`, `confidence_top2_all` | 区分 adaptive top1/top2/fallback 与固定 top-k |
| teacher route 是否仅为优化工具 | `no_teacher_routing`, `permanent_teacher_routing` | permanent 诊断训练/部署 exposure bias；验证仍使用预测路由 |
| ROI细化是否真实有效且安全 | `no_refinement`, `legacy_roi_fusion` | 同时报告 Dice 与 fallback/触发统计 |
| 路由正则应匹配什么分布 | `uniform_balance`, `no_router_regularizer` | 医学类别不均衡时 prior 比 uniform 更合理 |
| 专家容量放在哪里 | `fixed_expert_scale`, `layer6_qv_only`, `layers456_all_attention` | 控制 rank 不变，只改变深度、投影范围或缩放 |

`routing` 与 `refinement` suite 用于机制专项分析；其中 routing suite 包含严格的
`backbone_fpn`/`decoder_memory` 路由源对照。`capacity` 检查 expert rank、locator
宽度和注入范围；`sensitivity` 检查置信阈值、teacher 衰减和融合权重。
`smoke` 只用于检查流程，不能用于论文结论。`all` 包含所有诊断与敏感性变体。

## 推荐的两阶段执行顺序

1. **流程验证**：对 `smoke` 临时覆盖很少 epoch（建议另建本地 study 副本，不修改主
   study），检查 image-only batch、locator loss、专家计数、ROI skip 和输出统计均正常。
2. **筛选实验**：primary 全部运行 3 seeds，检查效应方向、失败率与方差。
3. **确认实验**：只对预先声明的主比较运行至少 5 seeds；不能依据 3-seed 结果临时选择
   有利 seed。
4. **外部/跨域确认**：固定训练 checkpoint，在相同 image-only inference 协议下分别报告
   in-domain 与 cross-domain，不混合挑选阈值。

例如扩展到五个 seed：

```bash
python ablation/run_ablation.py run \
  --suite primary \
  --seeds 13 29 42 73 101 \
  --device-sets 0 1 2 3
```

## 公平性约束

- 所有 cell 继承同一基础配置，训练/验证患者列表、area/boundary threshold、SAM3 与
  ResNet checkpoint 完全不变；生成器发现这些路径变化会立即报错。
- `training.seed` 与 `training.data_order_seed` 随重复实验改变，但 patient sampling seed 和
  已选患者 JSON 固定，使方法间能够按 seed 配对。
- 训练器在构建 SAM3/MoE 前用 `training.seed` 同步设置 Python、NumPy、PyTorch 与 CUDA
  RNG；生成器同时设置 prompt seed 和数据顺序 seed，并默认启用 deterministic 模式。
- 所有正式结果使用 `dataset.prompt_curriculum.evaluation_mode=image_only`。内部固定文本
  `prostate` 只是 SAM3 接口所需的任务 token，不包含 GT bbox、GT mask 或 slice 标签。
- 主方法的 `moe.routing_feature_source=backbone_fpn`，路由特征在任何 sample-specific
  text/bbox 融合前提取。`decoder_memory_routing_source` 只用于诊断旧的提示依赖路径，不可
  作为“image-only routing”的证据。
- `router_source_layer` 在 `backbone_fpn` 模式下只控制 query-derived `coarse_mask_p3`
  辅助头的生成时机，不改变分类路由输入，因此不把该字段包装为“路由特征深度”消融。
- 主选择准则统一为验证损失最低 epoch；若改为 last epoch，应对所有 cell 统一改，不能
  按方法选择。
- 保留 batch size、epoch、学习率、rank 和增强不变。宽模块实验参数量增大，表格中还应
  额外报告 trainable parameters、FLOPs/吞吐和显存。
- `permanent_teacher_routing` 仅在训练期永久使用 GT route，验证仍强制预测路由；它用于诊断
  exposure bias，不是 oracle 推理上界，也不能作为候选部署方法。
- 如果任一方法/seed 失败，不应直接删除；`missing_runs.json` 必须随结果保留并解释原因。

## 最低报告集合

论文主表至少报告 patient-macro Dice/IoU、slice Dice/IoU、MR/US Dice、小/中/大面积 Dice、
clear/fuzzy/complex 边界 Dice。机制表同时报告 modality/area/boundary routing accuracy、entropy、
各专家使用率、低置信度 fallback 比例、SvANet trigger/skip/full-image fallback 比例。效率表
报告参数量、单 slice 延迟（含 P50/P95）、峰值显存。

统计单位首先是患者而不是 slice。当前汇总器的 seed-level CI 用于模型重复实验；正式论文
还应在每个 seed 内对患者做 stratified bootstrap（按 MR/US 分层），再报告跨 seed 综合区间。
对所有相对 `full_v2` 的比较使用相同 seed，脚本会在每个指标族内做 Holm 校正。

## 扩展矩阵

新增实验只需编辑 `study.yaml`：

```yaml
experiments:
  my_single_factor:
    description: 只改变一个设计因素
    comparison: 说明与 full_v2 的可证伪假设
    overrides:
      moe:
        confidence_high_threshold: 0.80
```

随后把 id 加入某个 suite。禁止在 `overrides` 内设置数据或权重路径；输出路径由生成器统一
注入。生成后的 YAML 含 `ablation` 元数据和哈希，manifest 同时记录 Git commit 与 dirty
状态，可用于核对远端实际运行配置；正式实验应在 `git_dirty=false` 的固定提交上执行。
