import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.stage0.build_qwen_embeddings import last_token_pool, pick_device, pick_dtype
from src.stage0.router_model import RouterHead
from src.stage0.router_policy import apply_fail_open
from src.stage0.router_utils import read_jsonl, write_jsonl


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parents[1]):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (Path(__file__).resolve().parent / p).resolve()


def load_head(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]["model"]
    model = RouterHead(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        num_blocks=int(cfg.get("num_blocks", 2)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def embed_texts(texts, tokenizer, embedder, device, max_length, normalize):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    out = embedder(**encoded)
    pooled = last_token_pool(out.last_hidden_state, encoded["attention_mask"])
    if normalize:
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.float()


@torch.inference_mode()
def predict_rows(rows, tokenizer, embedder, head, cfg, device, embedding_cache=None):
    qwen_cfg = cfg["qwen"]
    thresholds = cfg["fail_open"]
    batch_size = int(qwen_cfg.get("batch_size", 8))
    out_rows = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        if embedding_cache is not None:
            missing = [r["id"] for r in batch_rows if r["id"] not in embedding_cache]
            if missing:
                raise KeyError(f"Missing precomputed embeddings: {missing[:5]} (total missing: {len(missing)})")
            embs = [embedding_cache[r["id"]] for r in batch_rows]
            emb = torch.stack(embs).to(device=device, dtype=torch.float32)
        else:
            prompts = [r["prompt"] for r in batch_rows]
            emb = embed_texts(
                prompts,
                tokenizer,
                embedder,
                device,
                int(qwen_cfg.get("max_length", 256)),
                bool(qwen_cfg.get("normalize", True)),
            )
        logits = head(emb)
        cat_probs = F.softmax(logits["category_logits"], dim=1).cpu().numpy()
        tight_probs = F.softmax(logits["tightness_logits"], dim=1).cpu().numpy()
        for row, cp, tp in zip(batch_rows, cat_probs, tight_probs):
            d = apply_fail_open(
                cp.tolist(),
                tp.tolist(),
                row["prompt"],
                category_threshold=float(thresholds["category_threshold"]),
                tightness_threshold=float(thresholds["tightness_threshold"]),
            )
            out = dict(row)
            out.update(
                {
                    "pred_category": d.pred_category,
                    "raw_pred_category": d.raw_pred_category,
                    "category_postprocess_reason": d.category_postprocess_reason,
                    "category_confidence": d.category_confidence,
                    "category_tightness": d.category_tightness,
                    "pred_tightness": d.pred_tightness,
                    "tightness_confidence": d.tightness_confidence,
                    "laterality": d.laterality,
                    "final_tightness": d.final_tightness,
                    "final_policy": d.final_policy,
                    "fail_open_reason": d.fail_open_reason,
                }
            )
            out_rows.append(out)
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--text")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl", default="artifacts/predictions/predictions.jsonl")
    args = parser.parse_args()

    if not args.text and not args.input_jsonl:
        raise SystemExit("Provide --text or --input-jsonl")

    device = pick_device("auto")
    head, ckpt = load_head(resolve_path(args.checkpoint), device)
    cfg = ckpt["config"]
    qwen_cfg = cfg["qwen"]
    model_path = resolve_path(qwen_cfg["model_path"])
    dtype = pick_dtype(qwen_cfg.get("dtype", "auto"), device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    embedder = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    embedder.eval()

    if args.text:
        rows = [{"id": "manual", "prompt": args.text}]
    else:
        rows = read_jsonl(resolve_path(args.input_jsonl))

    preds = predict_rows(rows, tokenizer, embedder, head, cfg, device)
    if args.text:
        print(json.dumps(preds[0], ensure_ascii=False, indent=2))
    else:
        write_jsonl(preds, resolve_path(args.output_jsonl))
        print(f"saved {len(preds)} predictions to {resolve_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
