"""Export final inference masks from gate decisions.

For each finding, the gate chooses either:
  - S2 prediction from outputs.stage2/finding_preds/{finding_id}_pred.nii.gz
  - S1 coarse mask reconstructed from verified proposals

This script is for inference-only runs and does not read GT labels.
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from src.utils import load_config, load_jsonl, resolve, write_jsonl


def reader_mask_to_raw_hwd(mask_reader: np.ndarray) -> np.ndarray:
    """Convert saved reader-space DWH mask crop back to raw HWD crop."""
    return mask_reader.transpose(2, 1, 0)[::-1, ::-1, :]


def paste_verified_mask(full_mask: np.ndarray, entry: dict, gate_bbox: list[int]) -> None:
    path = entry.get("coarse_mask_path")
    if not path:
        return
    try:
        mask_reader = np.asanyarray(nib.load(path).dataobj) > 0
    except (FileNotFoundError, OSError):
        return
    if mask_reader.sum() == 0:
        return

    gh0, gh1, gw0, gw1, gd0, gd1 = gate_bbox
    raw_h = gh1 - gh0
    raw_w = gw1 - gw0
    raw_d = gd1 - gd0

    ph0, ph1, pw0, pw1, pd0, pd1 = entry["proposal_bbox_hwd"]
    ch0 = max(0, ph0 - gh0)
    ch1 = min(raw_h, ph1 - gh0)
    cw0 = max(0, pw0 - gw0)
    cw1 = min(raw_w, pw1 - gw0)
    cd0 = max(0, pd0 - gd0)
    cd1 = min(raw_d, pd1 - gd0)
    if ch0 >= ch1 or cw0 >= cw1 or cd0 >= cd1:
        return

    mask_raw = reader_mask_to_raw_hwd(mask_reader)
    expected = (ch1 - ch0, cw1 - cw0, cd1 - cd0)
    if mask_raw.shape != expected:
        h = min(mask_raw.shape[0], expected[0])
        w = min(mask_raw.shape[1], expected[1])
        d = min(mask_raw.shape[2], expected[2])
        mask_raw = mask_raw[:h, :w, :d]
        ch1, cw1, cd1 = ch0 + h, cw0 + w, cd0 + d

    full_mask[gh0 + ch0:gh0 + ch1, gw0 + cw0:gw0 + cw1, gd0 + cd0:gd0 + cd1] |= mask_raw


def resolve_ct_path(mrow: dict, ct_images_dir: Path) -> Path | None:
    candidates = []
    if mrow.get("image"):
        image_rel = mrow["image"].replace("\\", "/")
        candidates.extend([
            ct_images_dir.parent / image_rel,
            ct_images_dir / image_rel,
        ])
    if mrow.get("case_name"):
        candidates.append(ct_images_dir / mrow["case_name"])
    for p in candidates:
        if p.exists():
            return p
    return None


def build_s1_mask(row: dict, shape: tuple[int, int, int]) -> np.ndarray:
    pred = np.zeros(shape, dtype=bool)
    gate_bbox = row.get("gate_bbox_hwd")
    for entry in row.get("verified", []):
        mask_path = entry.get("coarse_mask_path")
        if entry.get("source_expert") == "full_ct_voxtell":
            if mask_path and Path(mask_path).exists():
                mask_reader = np.asanyarray(nib.load(mask_path).dataobj) > 0
                mask_raw = reader_mask_to_raw_hwd(mask_reader)
                h = min(mask_raw.shape[0], pred.shape[0])
                w = min(mask_raw.shape[1], pred.shape[1])
                d = min(mask_raw.shape[2], pred.shape[2])
                pred[:h, :w, :d] |= mask_raw[:h, :w, :d]
        elif gate_bbox is not None:
            paste_verified_mask(pred, entry, gate_bbox)
    return pred.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final inference masks")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--manifest", default=None,
                        help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument("--verified-proposals", default="verified_proposals_s1_nosafety.jsonl",
                        help="S1 VP filename inside stage1 dir")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or resolve(config, "manifests.default")
    manifest_rows = load_jsonl(manifest_path)
    manifest = {r["id"]: r for r in manifest_rows}

    final_dir = Path(resolve(config, "outputs.final"))
    stage1_dir = Path(resolve(config, "outputs.stage1"))
    stage2_dir = Path(resolve(config, "outputs.stage2"))
    ct_images_dir = Path(resolve(config, "data.ct_images"))
    out_dir = final_dir / "final_preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = final_dir / "gate_decisions_raw.json"
    if not decisions_path.exists():
        raise SystemExit(f"[ERROR] Gate decisions not found: {decisions_path}")
    with open(decisions_path, encoding="utf-8") as f:
        decisions = json.load(f)

    vp_rows = load_jsonl(stage1_dir / args.verified_proposals)
    vp = {r["finding_id"]: r for r in vp_rows}

    output_rows = []
    n_s1 = 0
    n_s2 = 0
    for decision in decisions.get("decisions", []):
        fid = decision["finding_id"]
        mrow = manifest.get(fid)
        if mrow is None:
            print(f"[SKIP] {fid}: not in manifest")
            continue
        ct_path = resolve_ct_path(mrow, ct_images_dir)
        if ct_path is None:
            print(f"[SKIP] {fid}: CT not found")
            continue
        ct_nii = nib.load(str(ct_path))
        shape = ct_nii.shape[:3]
        out_path = out_dir / f"{fid}_pred.nii.gz"

        use_s2 = int(decision.get("use_s2", 0))
        source = "s1"
        if use_s2:
            s2_path = stage2_dir / "finding_preds" / f"{fid}_pred.nii.gz"
            if s2_path.exists():
                shutil.copyfile(s2_path, out_path)
                source = "s2"
                n_s2 += 1
            else:
                print(f"[WARN] {fid}: S2 mask missing, falling back to empty mask")
                source = "empty"
                nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), ct_nii.affine), str(out_path))
        else:
            s1_row = vp.get(fid)
            if s1_row is None:
                print(f"[WARN] {fid}: S1 VP missing, exporting empty mask")
                pred = np.zeros(shape, dtype=np.uint8)
            else:
                pred = build_s1_mask(s1_row, shape)
            nib.save(nib.Nifti1Image(pred, ct_nii.affine), str(out_path))
            n_s1 += 1

        output_rows.append({
            "finding_id": fid,
            "case_name": mrow.get("case_name", ""),
            "category": mrow.get("category", decision.get("category", "")),
            "use_s2": use_s2,
            "source": source,
            "mask_path": str(out_path),
        })

    manifest_out = final_dir / "final_prediction_manifest.jsonl"
    write_jsonl(manifest_out, output_rows)
    summary = {
        "n_predictions": len(output_rows),
        "n_s1": n_s1,
        "n_s2": n_s2,
        "manifest": str(manifest_out),
        "pred_dir": str(out_dir),
    }
    with open(final_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(output_rows)} final masks -> {out_dir}")
    print(f"Summary -> {final_dir / 'export_summary.json'}")


if __name__ == "__main__":
    main()
