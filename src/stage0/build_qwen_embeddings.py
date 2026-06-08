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

from src.stage0.router_policy import labels_for_category
from src.stage0.router_utils import read_jsonl


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parents[1]):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (Path(__file__).resolve().parent / p).resolve()


def pick_device(config_device: str) -> torch.device:
    if config_device and config_device != "auto":
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(config_dtype: str, device: torch.device):
    if config_dtype == "float16":
        return torch.float16
    if config_dtype == "bfloat16":
        return torch.bfloat16
    if config_dtype == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lens]


@torch.inference_mode()
def encode_prompts(prompts, tokenizer, model, device, batch_size, max_length, normalize):
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        chunks.append(pooled.float().cpu())
        print(f"encoded {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(chunks, dim=0)


def build_embeddings(name: str, rows, tokenizer, model, qwen_cfg, artifact_dir: Path, device):
    prompts = [r["prompt"] for r in rows]
    category_ids = []
    tightness_ids = []
    for row in rows:
        category_id, tightness_id = labels_for_category(row["category"])
        category_ids.append(category_id)
        tightness_ids.append(tightness_id)

    embeddings = encode_prompts(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=int(qwen_cfg["batch_size"]),
        max_length=int(qwen_cfg["max_length"]),
        normalize=bool(qwen_cfg.get("normalize", True)),
    )
    out = {
        "ids": [r["id"] for r in rows],
        "case_names": [r.get("case_name") for r in rows],
        "prompts": prompts,
        "categories": [r["category"] for r in rows],
        "category_labels": torch.tensor(category_ids, dtype=torch.long),
        "tightness_labels": torch.tensor(tightness_ids, dtype=torch.long),
        "embeddings": embeddings,
    }
    out_path = artifact_dir / "embeddings" / f"{name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved {name}: {out_path} shape={tuple(embeddings.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/router_qwen.json")
    parser.add_argument("--splits", nargs="+", default=["all"])
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    data_dir = resolve_path(config["data_dir"])
    artifact_dir = resolve_path(config["artifact_dir"])
    qwen_cfg = config["qwen"]
    model_path = resolve_path(qwen_cfg["model_path"])
    device = pick_device(qwen_cfg.get("device", "auto"))
    dtype = pick_dtype(qwen_cfg.get("dtype", "auto"), device)

    print(f"loading tokenizer/model from {model_path}")
    print(f"device={device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    for name in args.splits:
        rows = read_jsonl(data_dir / f"router_{name}.jsonl")
        build_embeddings(name, rows, tokenizer, model, qwen_cfg, artifact_dir, device)


if __name__ == "__main__":
    main()
