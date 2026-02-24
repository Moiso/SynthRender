#!/usr/bin/env python3
import os
import shutil
import argparse


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def merge_split(root_dir: str, out_root: str, split: str, img_exts=(".png", ".jpg", ".jpeg")):
    """
    Merge all images + labels for a given split ('train' or 'val')
    from all batch subfolders under root_dir into out_root/yolo/{images,labels}/{split}.

    Images will be renamed to:
        {index}.ext
    Labels will be renamed to:
        {index:06d}.txt
    """
    out_img_dir = os.path.join(out_root, "yolo", "images", split)
    out_lbl_dir = os.path.join(out_root, "yolo", "labels", split)
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    idx = 0  # global index per split

    batch_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    print(f"\n[Merging {split}] Found batch dirs:", batch_dirs)

    for batch_name in batch_dirs:
        batch_path = os.path.join(root_dir, batch_name)
        yolo_dir = os.path.join(batch_path, "yolo")
        if not os.path.isdir(yolo_dir):
            print(f"[{split}] Skipping {batch_name}: no 'yolo' dir")
            continue

        src_img_dir = os.path.join(yolo_dir, "images", split)
        src_lbl_dir = os.path.join(yolo_dir, "labels", split)

        if not os.path.isdir(src_img_dir) or not os.path.isdir(src_lbl_dir):
            print(f"[{split}] Skipping {batch_name}: missing images/{split} or labels/{split}")
            continue

        img_files = sorted(
            f for f in os.listdir(src_img_dir)
            if os.path.splitext(f)[1].lower() in img_exts
        )

        print(f"[{split}] {batch_name}: {len(img_files)} images")

        for img_file in img_files:
            img_src_path = os.path.join(src_img_dir, img_file)
            base, ext = os.path.splitext(img_file)

            # Try to locate matching label:
            # 1) same basename
            # 2) zero-padded basename (length 6)
            label_candidates = [
                os.path.join(src_lbl_dir, f"{base}.txt"),
                os.path.join(src_lbl_dir, f"{int(base):06d}.txt") if base.isdigit() else None,
            ]
            label_candidates = [p for p in label_candidates if p is not None]

            label_src_path = None
            for cand in label_candidates:
                if os.path.exists(cand):
                    label_src_path = cand
                    break

            if label_src_path is None:
                print(f"[WARN] No label found for image {img_src_path}, skipping this image.")
                continue

            # Destination names
            new_base = f"{idx}"
            img_dst_path = os.path.join(out_img_dir, new_base + ext.lower())
            lbl_dst_path = os.path.join(out_lbl_dir, f"{idx:06d}.txt")

            # Copy
            shutil.copy2(img_src_path, img_dst_path)
            shutil.copy2(label_src_path, lbl_dst_path)

            idx += 1

    print(f"[{split}] Total merged images: {idx}")


def main():
    ap = argparse.ArgumentParser(description="Merge YOLO batches into a single dataset")
    ap.add_argument(
        "--root-dir",
        required=True,
        help="Root folder containing batch subfolders (e.g. BATCHES_100)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for merged dataset (will create yolo/images/{train,val}, yolo/labels/{train,val})",
    )
    args = ap.parse_args()

    # Make sure out_dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Merge train and val separately
    merge_split(args.root_dir, args.out_dir, "train")
    merge_split(args.root_dir, args.out_dir, "val")


if __name__ == "__main__":
    main()



# python merge_batches.py \
#   --root-dir /media/vrt/D/Datasets/Outputs/BATCHES_100_b7 \
#   --out-dir  /media/vrt/D/Datasets/Outputs/BATCHES_100_MERGED_7
