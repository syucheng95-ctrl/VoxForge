# VoxForge：基于自由文本的 CT 病灶分割流水线

VoxForge 是一个面向胸部 CT 的自由文本病灶分割流水线。它会根据每条影像描述预测解剖区域并裁剪 CT，使用多个专家生成候选框，再用 VoxTell 验证候选区域，用 STU-Net 做 ROI 级细化，最后通过学习得到的 gate 在 S1 粗分割和 S2 细化结果之间逐 finding 选择最终掩膜。

本仓库同时支持复现实验和无标签正式推理。模型权重、CT 数据和运行输出体积较大，不纳入 git 追踪。

## 流水线结构

```text
自由文本 prompt
  -> Stage 0：文本路由 + 解剖裁剪
  -> Stage 0.5：多专家候选框生成
  -> Stage 1：VoxTell 验证 + S1 proposal 选择
  -> Stage 2：STU-Net ROI 推理/评估 + gate 特征
  -> Gate：逐 finding 选择 S1 或 S2
  -> 最终掩膜导出或指标评估
```

| 阶段 | 主要文件 | 作用 |
|---|---|---|
| Stage 0 | `pipeline/stage0_crop.py` | 根据文本预测类别和解剖区域，裁剪 CT |
| Stage 0.5 | `pipeline/stage0_5_proposals.py` | 使用结节检测、HU 阈值、弥漫病灶专家生成候选框 |
| Stage 1 | `pipeline/stage1_verify.py`, `pipeline/build_s1_vp.py` | 用 VoxTell 验证候选框，并选择 S1 proposal 分支 |
| Stage 2 | `pipeline/stage2_rois.py`, `pipeline/stage2_eval.py`, `pipeline/build_s2_vp.py` | 构建 ROI，运行 STU-Net，并提取 S2 预测特征 |
| Gate | `pipeline/build_gate_table.py`, `pipeline/apply_gate.py` | 构建推理特征表，并预测每个 finding 使用 S1 还是 S2 |
| 输出 | `pipeline/eval_final.py`, `pipeline/export_final.py` | 有标签时计算指标；无标签推理时导出最终掩膜 |

## 评估结果

ReXGroundingCT 评估数据集上的结果：

| 方法 | Dice | Recall | Precision |
|---|---:|---:|---:|
| VoxTell 单阶段 baseline | 0.4132 | 0.4944 | 0.3550 |
| Stage0-1 Pipeline (S1) | 0.4933 | 0.4840 | 0.5029 |
| VoxForge full pipeline + ML gate | 0.7138 | 0.6254 | 0.8313 |

## 环境准备

依赖环境：

- Python 3.10+
- PyTorch 2.0+ 和 CUDA
- 当前顺序推理流程建议约 10 GB+ GPU 显存
- 完整模型权重约需要 15 GB 磁盘空间

安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

提交用的可移植配置文件是 `configs/pipeline.yaml`：

```yaml
data:
  ct_images: ./data/images
  labels: ./data/labels
manifests:
  default: ./data/manifest.jsonl
python: python
```

相对路径都按仓库根目录解析。如果需要写本机绝对路径，请新建 `configs/pipeline.local.yaml`；该文件已被 `.gitignore` 忽略，不会提交。

manifest 是 JSONL 格式，每行对应一个 finding。评估模式需要 label 路径；无标签正式推理使用 `--inference-only`，不需要 GT label。

## 提交说明

由于 CT 数据和模型权重体积较大，本仓库提交版本不包含以下内容：

- `data/images/`：CT NIfTI 图像
- `data/labels/`：GT label，仅评估时需要
- `models/`：Qwen、VoxTell、STU-Net、nodule detector、gate 等模型权重
- `outputs/`：运行中间结果和最终输出

运行前请按照 `configs/pipeline.yaml` 中的路径准备数据和模型文件。

## 模型文件

请按 `configs/pipeline.yaml` 中的路径把模型放到 `./models/` 下。

| 配置项 | 期望路径 | 说明 |
|---|---|---|
| `models.qwen_embedding` | `models/Qwen3-Embedding-4B` | Qwen3 文本 embedding 模型 |
| `models.stage0_router` | `models/stage0_router/artifacts/models/router_head_best.pt` | Stage0 router head |
| `models.voxtell` | `models/voxtell_v1.1` | VoxTell checkpoint 目录 |
| `models.nodule_detector` | `models/lung_nodule_ct_detection/lung_nodule_ct_detection` | MONAI lung nodule detector bundle |
| `models.stunet` | `models/stunet/eval_current.pt` | STU-Net checkpoint |
| `models.medim_ckpt` | `models/medim_ckpt` | MedIM/STU-Net 依赖缓存 |
| `gate.model` | `models/gate_model.joblib` | Random Forest gate bundle |
| `gate.metadata` | `models/gate_metadata.json` | gate 特征元数据 |
| `models.fallback_selector` | `models/fallback_selector_xgb_4feat.pkl` | S1 expert selector |

`models/` 已被 git 忽略。

## 运行

有标签评估，计算最终 Dice/Recall/Precision：

```bash
python pipeline/run_full.py --config configs/pipeline.yaml
```

无标签正式推理，导出最终掩膜和 gate decision：

```bash
python pipeline/run_full.py --config configs/pipeline.yaml --inference-only
```

只跑 1 个 CT 做 smoke test：

```bash
python pipeline/run_full.py --config configs/pipeline.local.yaml --limit-cases 1 --no-clean
```

只跑 1 个 CT 的无标签推理 smoke test：

```bash
python pipeline/run_full.py --config configs/pipeline.local.yaml --limit-cases 1 --inference-only --no-clean
```

如果前面阶段已经跑完，可以从后续阶段恢复：

```bash
python pipeline/run_full.py --config configs/pipeline.yaml \
  --stages s1_nosafety,s2_c,1_metrics,2_roi,2_eval,gate_table,gate_apply,eval_final
```

## 输出

有标签评估输出：

| 路径 | 说明 |
|---|---|
| `outputs/final/final_metrics.json` | 最终 Dice/Recall/Precision 和 baseline 对比 |
| `outputs/final/gate_decisions.csv` | 每个 finding 的 S1/S2 gate 决策和指标 |
| `outputs/stage2/eval_finding_level.json` | S2 每个 finding 的指标和 gate 特征 |
| `outputs/pipeline_summary.json` | 各阶段耗时和摘要 |

无标签推理输出：

| 路径 | 说明 |
|---|---|
| `outputs/final/gate_decisions_raw.json` | gate 概率和 S1/S2 决策 |
| `outputs/final/gate_decisions.csv` | 每个 finding 的 gate 决策表 |
| `outputs/final/final_prediction_manifest.jsonl` | 最终掩膜清单 |
| `outputs/final/final_preds/{finding_id}_pred.nii.gz` | 每个 finding 的最终分割掩膜 |
| `outputs/final/export_summary.json` | 最终导出摘要 |

## 项目结构

```text
VoxForge/
├── configs/
│   └── pipeline.yaml
├── pipeline/
│   ├── run_full.py
│   ├── stage0_crop.py
│   ├── stage0_5_proposals.py
│   ├── stage1_verify.py
│   ├── build_s1_vp.py
│   ├── build_s2_vp.py
│   ├── stage1_metrics.py
│   ├── stage2_rois.py
│   ├── stage2_eval.py
│   ├── build_gate_table.py
│   ├── apply_gate.py
│   ├── eval_final.py
│   └── export_final.py
├── src/
│   ├── stage0/
│   ├── proposal_generator/
│   ├── stage1/
│   ├── stage2/
│   ├── gate/
│   └── voxtell/
├── data/
├── models/
└── outputs/
```

## 注意事项

- `run_full.py` 不包含训练过程；它只调度已有模型推理、特征表构建、gate 推理，以及可选的评估或导出。
- `stage1_metrics.py` 和 `eval_final.py` 需要 GT label，在 `--inference-only` 下会被跳过。
- `--inference-only` 会自动调用 `stage2_rois.py --no-labels` 和 `stage2_eval.py --no-gt`。
