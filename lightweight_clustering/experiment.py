"""Full reproducible baseline experiment; no lesion label use before evaluation.

Usage: python -m lightweight_clustering.experiment --help
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import time
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .torc_approx import cluster_patient_support, torque_approx

METHODS = ("patch_knn", "torc_approx", "torc_consensus")
MODEL_ID = "timm/tf_mobilenetv3_small_minimal_100.in1k"
MODEL_NAME = "tf_mobilenetv3_small_minimal_100.in1k"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--output", default="outputs")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--offline", action="store_true", help="do not download model weights")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--max-cases", type=int, default=0, help="debug subset only")
    p.add_argument("--slice-stride", type=int, default=4, help="1 for all slices")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--patch-grid", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ref-per-case", type=int, default=64)
    p.add_argument("--torc-max-points", type=int, default=512)
    p.add_argument("--torc-max-clusters", type=int, default=12)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--force", action="store_true", help="recompute predictions/caches")
    return p


def safe_extract(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    root = dst.resolve()
    def check(name):
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            raise RuntimeError("Unsafe archive path: " + name)
    if tarfile.is_tarfile(src):
        with tarfile.open(src) as tf:
            for m in tf.getmembers():
                check(m.name)
                if m.issym() or m.islnk() or m.isdev():
                    raise RuntimeError("Archive links/devices are forbidden: " + m.name)
            tf.extractall(dst)
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            for m in zf.infolist():
                check(m.filename)
                if (m.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Archive symbolic links forbidden")
            zf.extractall(dst)
    else:
        raise RuntimeError("Unsupported archive: " + str(src))


def locate_data(root: Path):
    def find():
        for folder in sorted(root.rglob("imagesTr")):
            if not folder.is_dir() or folder.parent.name.startswith("."):
                continue
            labels = folder.parent / "labelsTr"
            if labels.is_dir():
                images = {f.name: f for f in folder.glob("*.nii.gz")}
                pairs = [(f, labels / name) for name, f in images.items()
                         if (labels / name).is_file()]
                if len(pairs) >= 3:
                    return sorted(pairs, key=lambda a: a[0].name)
        return None
    pairs = find()
    if pairs:
        return pairs
    archive_list = sorted(f for f in root.rglob("*") if f.is_file() and (
        f.suffix.lower() in (".tar", ".zip", ".tgz") or f.name.endswith(".tar.gz")
    ) and "extracted" not in f.parts)
    extracted = root / "extracted"
    for archive in archive_list:
        marker = extracted / (hashlib.sha256(str(archive.resolve()).encode()).hexdigest()[:16] + ".done")
        if not marker.exists():
            print("[extract]", archive)
            safe_extract(archive, extracted)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(archive.stat().st_size), encoding="utf-8")
        pairs = find()
        if pairs:
            return pairs
    raise FileNotFoundError("No MSD imagesTr/labelsTr pairs found in " + str(root) +
                            "; put original Task10_Colon archive into data/archives/")


def split_cases(pairs, out: Path, seed: int, test_fraction: float):
    if not (0.05 <= test_fraction <= 0.5):
        raise ValueError("--test-fraction must be within [0.05,0.5]")
    identities = [f.name[:-7] for f, _ in pairs]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        info = json.loads(out.read_text(encoding="utf-8"))
        if info["all_cases"] != identities or info["seed"] != seed or info["test_fraction"] != test_fraction:
            raise RuntimeError("Patient list/split args changed: remove experiment output directory explicitly")
        return set(info["train"]), set(info["test"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(identities))
    n_test = max(1, int(round(len(idx) * test_fraction)))
    if n_test == len(idx):
        raise ValueError("Insufficient cases for train/test split")
    train = sorted(identities[j] for j in idx[n_test:])
    test = sorted(identities[j] for j in idx[:n_test])
    out.write_text(json.dumps(dict(seed=seed, test_fraction=test_fraction,
                 all_cases=identities, train=train, test=test), indent=2), encoding="utf-8")
    assert not (set(train) & set(test))
    return set(train), set(test)


def read_slices(path: Path, indices):
    import nibabel as nib
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected 3D CT: " + str(path))
    h, w, z = arr.shape
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) == 0 or max(indices) >= z:
        raise ValueError("Slice index out of range")
    return arr, (h, w, z)


def preprocess_window(img, z, size):
    import torch
    import torch.nn.functional as F
    depth = img.shape[2]
    channels = [np.clip(img[:, :, min(depth-1,max(0,z+d))], -150, 250)
                for d in (-1, 0, 1)]
    arr = np.stack([(v + 150.) / 400. for v in channels]).astype(np.float32)
    x = torch.from_numpy(arr[None])
    x = F.interpolate(x, size=(size,size), mode="bilinear", align_corners=False)[0]
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std = torch.tensor([0.229,0.224,0.225])[:,None,None]
    return (x - mean) / std


def load_backbone(args):
    import torch
    import timm
    from modelscope import snapshot_download
    from safetensors.torch import load_file

    model_root = Path(args.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / args.model_id.replace("/", "__")
    weights = list(destination.glob("**/*.safetensors")) + list(destination.glob("**/pytorch_model.bin"))
    if not weights:
        if args.offline:
            raise FileNotFoundError("Offline weights missing under " + str(destination))
        print("[ModelScope]", args.model_id, "=>", destination)
        snapshot_download(args.model_id, local_dir=str(destination))
        weights = list(destination.glob("**/*.safetensors")) + list(destination.glob("**/pytorch_model.bin"))
    if not weights:
        raise FileNotFoundError("ModelScope snapshot has no recognized weights at "+str(destination))
    path = sorted(weights, key=lambda x:(x.name != "model.safetensors", x.name))[0]
    net = timm.create_model(MODEL_NAME, pretrained=False, features_only=True, out_indices=(1,2)).eval()
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module.").removeprefix("model."): v for k,v in state.items()}
    keys = set(net.state_dict()) & set(state)
    if len(keys) < 0.65 * len(net.state_dict()):
        raise RuntimeError("Incompatible checkpoint: only "+str(len(keys))+" matching keys; refuse random backbone")
    net.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    print("[backbone]", MODEL_NAME, "device", device, "loaded keys", len(keys))
    return net, device, (path.stat().st_size, path.name)


def extract_case(imgpath: Path, model, device, args, cache_dir: Path, fingerprint: str):
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (imgpath.name[:-7] + "_" + fingerprint + ".npz")
    if cache.exists() and not args.force:
        with np.load(cache) as data:
            return data["features"].astype(np.float32), data["zs"]
    img, (_, _, depth) = read_slices(imgpath, [0])
    zs = np.arange(0, depth, args.slice_stride, dtype=np.int32)
    # No GT mask is loaded here and no slice is selected by lesion presence.
    all_feats = []
    for begin in range(0, len(zs), args.batch_size):
        batch_z = zs[begin:begin + args.batch_size]
        batch = torch.stack([preprocess_window(img, int(z), args.input_size)
                             for z in batch_z]).to(device)
        with torch.inference_mode():
            maps = model(batch)
            maps = [F.adaptive_avg_pool2d(m.float(), (args.patch_grid,args.patch_grid))
                    for m in maps]
            joined = torch.cat(maps, dim=1)
            joined = F.normalize(joined, p=2, dim=1)
            feat = joined.permute(0,2,3,1).contiguous().cpu().numpy()
        all_feats.append(feat.astype(np.float16))
    features = np.concatenate(all_feats,axis=0)
    np.savez_compressed(cache, features=features, zs=zs)
    return features.astype(np.float32), zs


def balanced_reference(cases: dict, train: set, seed: int, per_case: int, maximum: int):
    rng = np.random.default_rng(seed)
    pool = {}
    for name in sorted(train):
        x = cases[name][0].reshape(-1,cases[name][0].shape[-1])
        count = min(len(x), per_case)
        indices = rng.choice(len(x), size=count, replace=False)
        pool[name] = x[indices]
    # Round-robin interleaving prevents patients with more slices dominating.
    selected = []
    owners = []
    for i in range(max(map(len,pool.values()))):
        for person in sorted(pool):
            if i < len(pool[person]) and len(selected) < maximum:
                selected.append(pool[person][i])
                owners.append(person)
    if len(selected) < 3:
        raise ValueError("Too few reference patches")
    return np.asarray(selected,np.float32), np.asarray(owners)


def prepare_metric_mask(path: Path, zs, shape):
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    img = nib.load(str(path))
    label = img.get_fdata(dtype=np.float32)
    if label.shape[2] <= int(max(zs)):
        raise RuntimeError("Label z axis mismatch")
    # MSD Colon foreground ID 1 is lesion; use any >0 only for evaluation.
    gt = []
    for z in zs:
        t = torch.from_numpy(np.asarray(label[:,:,int(z)] > 0,np.float32))[None,None]
        gt.append(F.interpolate(t, size=shape, mode="nearest")[0,0].numpy().astype(bool))
    return np.stack(gt)


def metric_one(score, mask):
    s = np.asarray(score,dtype=np.float32).ravel()
    y = np.asarray(mask,dtype=bool).ravel()
    if len(s) != len(y) or not np.all(np.isfinite(s)):
        raise ValueError("Invalid score/label shape or NaN")
    pos = int(y.sum())
    neg = len(y) - pos
    result = dict(positive_pixels=pos,negative_pixels=neg,
                  pixel_auroc=None,pixel_ap=None,dice_at_top5=None)
    if pos and neg:
        result["pixel_auroc"] = float(roc_auc_score(y,s))
        result["pixel_ap"] = float(average_precision_score(y,s))
    if pos:
        # PREDECLARED per-case top-5% area selection, NOT a trained or tuned threshold.
        n_pick = max(1, int(np.ceil(0.05 * len(s))))
        idx = np.argpartition(s, -n_pick)[-n_pick:]
        hit = int(y[idx].sum())
        result["dice_at_top5"] = float(2*hit / (n_pick+pos))
    return result


def evaluate(args, out, pairs, case_data, train, test, ref, ref_owners, clustering):
    import torch
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    k = min(args.knn,len(ref))
    finder = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute",
                              n_jobs=args.num_workers).fit(ref)
    # Reference intrinsic scale is train-only.
    dist0, _ = NearestNeighbors(n_neighbors=min(k+1,len(ref)), metric="cosine",
                  algorithm="brute", n_jobs=args.num_workers).fit(ref).kneighbors(ref)
    norm = float(np.median(dist0[:,1:].mean(axis=1)))
    norm = max(norm,1e-5)
    sizes, coverage = cluster_patient_support(clustering.labels,ref_owners)
    max_size = max(float(sizes.max()),1e-6)
    summary_rows = []
    paths = {f.name[:-7]: (f,l) for f,l in pairs}
    for name in sorted(test):
        features,zs = case_data[name]
        b,h,w,dim = features.shape
        query = features.reshape(-1,dim)
        ds, neighbor_ids = finder.kneighbors(query)
        base = ds.mean(axis=1).astype(np.float32) / norm
        rarity = 1 - sizes[neighbor_ids].mean(axis=1) / max_size
        consensus = 1 - coverage[neighbor_ids].mean(axis=1)
        scores = {
            "patch_knn": base,
            "torc_approx": base + 0.25*rarity,
            "torc_consensus": base + 0.25*rarity + 0.25*consensus,
        }
        gt = prepare_metric_mask(paths[name][1],zs,(args.input_size,args.input_size))
        for method, flat in scores.items():
            grid = flat.reshape(b,1,h,w)
            full = F.interpolate(torch.from_numpy(grid), size=(args.input_size,args.input_size),
                                 mode="bilinear",align_corners=False)[:,0].numpy()
            pred_path = out/"predictions"/method/(name+".npz")
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pred_path, score_grid=grid[:,0].astype(np.float16), zs=zs)
            row = dict(method=method,patient=name,slices=len(zs),**metric_one(full,gt))
            summary_rows.append(row)
        print("[evaluated]", name, "slices",len(zs))
    out_metrics = out/"metrics"
    out_metrics.mkdir(parents=True,exist_ok=True)
    fields = ["method","patient","slices","positive_pixels","negative_pixels",
              "pixel_auroc","pixel_ap","dice_at_top5"]
    with (out_metrics/"per_case.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader()
        wr.writerows(summary_rows)
    overall={}
    for method in METHODS:
        rows=[r for r in summary_rows if r["method"]==method]
        overall[method]={"patients":len(rows),"evaluated_slices":int(sum(r["slices"] for r in rows)),
                          "positive_pixels":int(sum(r["positive_pixels"] for r in rows))}
        for metric in ("pixel_auroc","pixel_ap","dice_at_top5"):
            vals=[r[metric] for r in rows if r[metric] is not None]
            overall[method][metric+"_macro"]=float(np.mean(vals)) if vals else None
            overall[method][metric+"_valid_patients"]=len(vals)
    (out_metrics/"summary.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    return overall


def main():
    args = parser().parse_args()
    if args.slice_stride < 1 or args.max_cases < 0 or args.torc_max_points < 3:
        raise ValueError("Positive stride, nonnegative max-cases, >=3 TORC points required")
    np.random.seed(args.seed); random.seed(args.seed)
    t0=time.perf_counter()
    root=Path(args.data)
    if not root.exists():
        raise FileNotFoundError("Missing data/; place original MSD archive in data/archives/")
    pairs=locate_data(root)
    if args.max_cases:
        pairs=pairs[:args.max_cases]
        if len(pairs)<3: raise ValueError("debug subset requires at least three cases")
    if len(pairs)<3:
        raise ValueError("Need at least 3 image/label pairs")
    out=Path(args.output)/("debug_"+str(args.max_cases) if args.max_cases else "full")
    out.mkdir(parents=True,exist_ok=True)
    train,test=split_cases(pairs,out/"split.json",args.seed,args.test_fraction)
    print("[cases]",len(pairs),"train",len(train),"test",len(test))
    model,device,weights_id=load_backbone(args)
    fingerprint=hashlib.sha256(json.dumps([args.model_id,weights_id,args.input_size,
                    args.patch_grid,args.slice_stride],sort_keys=True).encode()).hexdigest()[:12]
    cases={}
    for image,_ in pairs:
        name=image.name[:-7]
        cases[name]=extract_case(image,model,device,args,out/"cache",fingerprint)
        print("[features]",name,cases[name][0].shape)
    ref,owners=balanced_reference(cases,train,args.seed,args.ref_per_case,args.torc_max_points)
    print("[reference]",ref.shape,"patients",len(np.unique(owners)))
    clustering=torque_approx(ref,max_clusters=args.torc_max_clusters)
    size,coverage=cluster_patient_support(clustering.labels, owners)
    stats=dict(algorithm="independent_TORC_inspired_approx_NOT_official_parity",
               n_reference=len(ref),n_clusters=clustering.requested_clusters,
               cluster_counts=clustering.cluster_sizes.tolist(),
               cluster_coverage_by_patch=coverage.tolist(),
               merge_torques=clustering.merge_torques.tolist())
    (out/"torc_diagnostics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    overall=evaluate(args,out,pairs,cases,train,test,ref,owners,clustering)
    import torch
    import importlib.metadata
    ver={}
    for name in ("torch","timm","numpy","scikit-learn","nibabel","modelscope"):
        try: ver[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: ver[name]="unknown"
    meta=dict(args=vars(args),weights=weights_id,backbone=MODEL_NAME,
              versions=ver,seed=args.seed,seconds=time.perf_counter()-t0,
              total_train=len(train),total_test=len(test),
              peak_cuda_memory_mb=(float(torch.cuda.max_memory_allocated())/1048576
              if device=="cuda" else None),
              metrics=overall,warning="Prototype results; TORC Matlab parity not established")
    (out/"run_manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(overall,indent=2))
    print("[done]",out.resolve())


if __name__=="__main__":
    main()
