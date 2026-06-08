# Pipeline 脚本说明

本目录包含 VoxForge 各阶段脚本。通常直接运行 `pipeline/run_full.py` 即可；单独运行某个脚本主要用于调试或从中间阶段恢复。

## 阶段顺序

默认有标签评估流程：

```text
0 -> 0.5 -> 1 -> s1_nosafety -> s2_c -> 1_metrics -> 2_roi -> 2_eval -> gate_table -> gate_apply -> eval_final
```

无标签推理流程：

```text
0 -> 0.5 -> 1 -> s1_nosafety -> s2_c -> 2_roi -> 2_eval -> gate_table -> gate_apply -> export_final
```

无标签推理命令：

```bash
python pipeline/run_full.py --config configs/pipeline.yaml --inference-only
```

## 脚本列表

| 脚本 | 阶段 | 作用 |
|---|---|---|
| `run_full.py` | orchestrator | 按顺序调度各阶段，传递 config/manifest/stage 参数，写 `pipeline_summary.json` |
| `stage0_crop.py` | `0` | 加载 Qwen/router，预测解剖和类别，生成 crop groups |
| `stage0_5_proposals.py` | `0.5` | 在 Stage0 crop 内运行多专家 proposal 生成 |
| `stage1_verify.py` | `1` | 运行 VoxTell verifier，输出 verified/rejected proposals 和 coarse masks |
| `build_s1_vp.py` | `s1_nosafety` | 应用 fallback selector，生成 S1 选中的 proposals |
| `build_s2_vp.py` | `s2_c` | 构建 S2 proposal 输入，用 HU/diffuse 补充空的 S1 选择 |
| `stage1_metrics.py` | `1_metrics` | 计算 S1 Dice/Recall/Precision，需要 GT |
| `stage2_rois.py` | `2_roi` | 裁剪 Stage2 ROI image；`--no-labels` 下不裁 GT mask |
| `stage2_eval.py` | `2_eval` | 运行 STU-Net ROI 推理并写 S2 特征；`--no-gt` 下不计算指标 |
| `build_gate_table.py` | `gate_table` | 从预测结果构建逐 finding 的 gate feature table |
| `apply_gate.py` | `gate_apply` | 应用 gate 模型，写出 S1/S2 决策 |
| `eval_final.py` | `eval_final` | 计算最终验证指标，需要 GT |
| `export_final.py` | `export_final` | 无标签推理时按 gate 决策导出最终掩膜 |
| `_verifier.py` | helper | Stage1 使用的 VoxTell verifier 实现 |

## 验证模式和推理模式

评估模式会读取 GT label 并计算指标：

- `stage1_metrics.py` 读取 label。
- `stage2_rois.py` 会裁剪 ROI mask。
- `stage2_eval.py` 会计算 S2 Dice/Recall/Precision。
- `eval_final.py` 会计算最终 Dice/Recall/Precision。

推理模式不依赖 GT：

- `run_full.py --inference-only` 会跳过 `1_metrics` 和 `eval_final`。
- 自动调用 `stage2_rois.py --no-labels`。
- 自动调用 `stage2_eval.py --no-gt`。
- 最后调用 `export_final.py` 导出最终掩膜。

## 主要中间文件

| 文件 | 生成脚本 | 后续用途 |
|---|---|---|
| `outputs/stage0/stage0_router_predictions.jsonl` | `stage0_crop.py` | Stage0.5、gate table |
| `outputs/stage0/stage0_crop_groups.jsonl` | `stage0_crop.py` | Stage0.5、Stage1、gate table |
| `outputs/stage0_5/stage0_5_proposals.jsonl` | `stage0_5_proposals.py` | Stage1、gate table |
| `outputs/stage1/verified_proposals.jsonl` | `stage1_verify.py` | S1/S2 VP 构建 |
| `outputs/stage1/verified_proposals_s1_nosafety.jsonl` | `build_s1_vp.py` | S1 metrics、gate table、最终 S1 导出 |
| `outputs/stage1/verified_proposals_s2_c.jsonl` | `build_s2_vp.py` | Stage2 ROI/eval |
| `outputs/stage2/roi_manifest.jsonl` | `stage2_rois.py` | Stage2 eval |
| `outputs/stage2/eval_finding_level.json` | `stage2_eval.py` | gate table、final metrics |
| `outputs/stage2/finding_preds/*.nii.gz` | `stage2_eval.py` | final export |
| `outputs/gate_feature_table.csv` | `build_gate_table.py` | gate apply |
| `outputs/final/gate_decisions_raw.json` | `apply_gate.py` | final eval/export |

## 常用命令

只跑 1 个 CT 做 smoke test：

```bash
python pipeline/run_full.py --config configs/pipeline.local.yaml --limit-cases 1 --no-clean
```

只跑 1 个 CT 的无标签推理 smoke test：

```bash
python pipeline/run_full.py --config configs/pipeline.local.yaml --limit-cases 1 --inference-only --no-clean
```

如果 Stage2 输出已经存在，只重跑 gate 和最终导出：

```bash
python pipeline/run_full.py --config configs/pipeline.local.yaml \
  --limit-cases 1 --inference-only \
  --stages gate_table,gate_apply,export_final --no-clean
```
