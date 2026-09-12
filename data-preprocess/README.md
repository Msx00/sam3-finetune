# data-preprocess

一次性生成 `train.sh` 训练所需的数据产物，训练启动时只读、不重算。

`MedSAM3-main/data/patient_dataset.py` 在构建 `PatientDataset` 时会为每个 slice 现算
area / boundary 标签，单 slice 约 0.4 s（线程池较小）到 3 s（默认线程池）。当前配置选了
train 210+210 个 patient、val 35+35 个 patient，共 23104 个 slice —— 每次启动训练都要重新
测量就是几小时。把测量结果缓存到 `dataset.label_cache_dir`
（= `data-preprocess/labels`）后，训练日志会打印
`area+boundary labels: N/N from cache`，启动阶段只剩读 JSON。

## 快速开始

```bash
cd /mnt/afs/zhemin/ShixingMa/Project/sam3-finetune/data-preprocess

./run_preprocess.sh                 # 全流程（默认配置）
./run_preprocess.sh verify          # 只做完整性检查（不写任何文件）
./run_preprocess.sh labels --workers 64
./run_preprocess.sh coco --force    # 强制重建 sam3/boxes JSON
```

`run_preprocess.sh` 与 `train.sh` 使用同一套约定：conda 环境 `sam3`，路径全部来自 YAML 配置，
可用环境变量覆盖 `CONFIG` / `CONDA_ENV` / `CONDA_SH` / `WORKERS`。日志写到
`data-preprocess/logs/preprocess_<时间戳>.log`。

## 阶段

`./run_preprocess.sh [stage]`，stage 取 `patients | coco | labels | thresholds | verify | all`
（默认 `all`，按 `patients -> coco -> labels -> thresholds -> verify` 顺序执行）。

| 阶段 | 产物 | 说明 |
|---|---|---|
| `patients` | `datasetConfig/selected_patients.json`、`selected_val_patients.json` | 按配置里的 `num_{mr,us}_patients`、`patient_sampling`、`num_*_val_patients`、`val_patient_sampling` 复现训练用的 patient 选择，格式与 `save_selected_patients()` 完全一致 |
| `coco` | `<modality>-2d/{train,val,test}_sam3.json`、`*_boxes_xyxy.json` | 调用仓库根目录的 `trans_format.py`，把上游导出的初始 COCO（`<split>.json`）转成训练用的最终 COCO + SAM3 XYXY box prompt。已存在且比输入新时自动跳过 |
| `labels` | `data-preprocess/labels/<modality>/<split>/<patient_id>.json` | 调用根目录 `prepare_slice_labels.py --scope selected`，并行缓存每 slice 的 `area_ratio` / `boundary_contrast` / `boundary_complexity`。**这是省掉每次训练预处理的关键** |
| `thresholds` | `datasetConfig/area_thresholds.json`、`boundary_thresholds.json` | 默认只检查文件是否存在；`--force` 时才用 train masks 重新计算（boundary 阈值较慢） |
| `verify` | 无 | 校验最终 COCO / box JSON、阈值文件、patient manifest、label cache 覆盖率（含 shard 版本、`boundary_band_width`、slice 完整性）。全部 OK 退出码 0，否则打印问题并返回 1 |

产物路径优先取配置里已有的键（`mr_train_coco_json`、`mr_train_boxes_json`、
`label_cache_dir`、`selected_patients_json` 等），因此产物一定落在训练实际读取的位置；没有配置
的键才按 `<modality>-2d/<split>_sam3.json` 之类的默认名推导。

## 常用参数

| 参数 | 说明 |
|---|---|
| `--config PATH` | 训练配置，默认 `MedSAM3-main/configs/moe_sam3_train_from_scratch.yaml` |
| `--splits train val test` | `coco`/`labels` 处理的 split，默认 `train val`。需要缓存 test 时加 `test` |
| `--workers N` | label cache 并行进程数，默认 `min(32, CPU 数)` |
| `--limit N` | 只处理前 N 个 shard，用于小规模试跑 |
| `--force` | 重算已存在的产物（含阈值） |
| `--dry-run` | 只打印将要执行的命令 |
| `--category-name NAME` | 传给 `trans_format.py` 的前景类别名；默认沿用输入 JSON 里的名字 |

## 验证与测试

```bash
PYTHONPATH=../MedSAM3-main python preprocess.py verify      # 覆盖率报告
python -m unittest -v test_preprocess.py                    # 单元测试（不需要 GPU）
```

`verify` 会在以下情况报错：`*_sam3.json` / `*_boxes_xyxy.json` 缺失或为空；阈值文件缺字段；
manifest 与当前配置选出的 patient 不一致（说明配置改动后没重跑 `patients`）；label cache
缺 shard、shard 版本或 `boundary_band_width` 过期、shard 未覆盖该 patient 的全部 slice。

`test_preprocess.py` 里包含与 `PatientDataset._select_patient_ids` 的选择结果对比测试，保证
manifest 和训练实际选中的 patient 不会漂移；其余测试在临时目录里构造迷你数据集，覆盖
`verify` 的各类失败分支。

## 注意事项

- label cache 只保存原始测量值，`area_label` / `boundary_label` 仍在加载时按
  `datasetConfig/*.json` 推导，所以改阈值文件不需要重建缓存；但改
  `boundary_band_width` 会让缓存失效（`verify` 会提示）。
- `thresholds` 阶段默认不动现有阈值文件。`--force` 会用 train masks 重新计算，可能得到与
  现有文件不同的分位点（现有文件是按当时那份 manifest/样本量算的），从而改变伪标签分布，
  需要在训练前确认。
- 上游 3D → 2D 导出（`unified_preprocess_and_coco_20260410.py`）不在本流程内：它需要
  `/mnt/afs/zhemin/ShixingMa/dataset-tcia-shixingma-v2` 这类原始 3D 数据，服务器上只有导出的
  2D 数据。本流程以 `<modality>-2d/<split>.json` 为输入起点。
