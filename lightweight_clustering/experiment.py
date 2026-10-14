"""Full reproducible baseline experiment; no lesion label use before evaluation.

Usage: python -m lightweight_clustering.experiment --help
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import random
import tarfile
import time
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .torc_approx import cluster_patient_support, torque_approx
from .reporting import ensure_not_git_trackable, export_report, utc_timestamp

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
    p.add_argument("--slice-stride", type=int, default=1, help="1 (default) evaluates every slice; >1 for quick exploratory runs")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--patch-grid", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ref-per-case", type=int, default=64)
    p.add_argument("--torc-max-points", type=int, default=512)
    p.add_argument("--torc-max-clusters", type=int, default=12)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--force", action="store_true", help="recompute predictions/caches")
    p.add_argument("--save-predictions", action="store_true",
                   help="opt-in: save local score grids as NPZ (not included in report; Git ignored)")
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
                images = {f.name: f for f in folder.glob("*.nii.gz") if not f.name.startswith("._")}
                pairs = [(f, labels / name) for name, f in images.items()
                         if (labels / name).is_file()]
                if len(pairs) >= 3:
                    return sorted(pairs, key=lambda a: a[0].name)
        return None
    pairs = find()
    if pairs:
        return pairs
    # Up to 3 extraction passes handle archives nested inside the original archive.
    extracted = root / "extracted"
    for _pass in range(3):
        archive_list = sorted(f for f in root.rglob("*") if f.is_file() and (
            f.suffix.lower() in (".tar", ".zip", ".tgz") or f.name.endswith(".tar.gz")
        ))
        changed = False
        for archive in archive_list:
            marker_id = hashlib.sha256(
                (str(archive.resolve()) + ":" + str(archive.stat().st_size)).encode()
            ).hexdigest()[:16]
            marker = extracted / (marker_id + ".done")
            if marker.exists():
                continue
            print("[extract]", archive)
            safe_extract(archive, extracted)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(archive.stat().st_size), encoding="utf-8")
            changed = True
            pairs = find()
            if pairs:
                return pairs
        if not changed:
            break
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
    return net, device, (path.stat().st_size, path.stat().st_mtime_ns, path.name)


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
    """Pure pixel-level metrics; never return images, arrays or patient masks."""
    s = np.asarray(score,dtype=np.float32).ravel()
    y = np.asarray(mask,dtype=bool).ravel()
    if len(s) != len(y) or not np.all(np.isfinite(s)):
        raise ValueError("Invalid score/label shape or NaN")
    pos = int(y.sum())
    neg = len(y) - pos
    result = dict(positive_pixels=pos,negative_pixels=neg,
                  pixel_auroc=None,pixel_ap=None,dice_at_top5=None,
                  iou_at_top5=None, precision_at_top5=None,
                  recall_at_top5=None, fpr_at_top5=None,
                  top5_tp=None, top5_fp=None, top5_fn=None, top5_tn=None)
    if pos and neg:
        result["pixel_auroc"] = float(roc_auc_score(y,s))
        result["pixel_ap"] = float(average_precision_score(y,s))
    if len(y):
        # Per-patient, fixed top-5% threshold; no test-label tuning.
        n_pick = max(1, int(np.ceil(0.05 * len(s))))
        idx = np.argpartition(s, -n_pick)[-n_pick:]
        tp = int(y[idx].sum())
        fp = n_pick - tp
        fn = pos - tp
        tn = neg - fp
        result.update(top5_tp=tp,top5_fp=fp,top5_fn=fn,top5_tn=tn,
                      precision_at_top5=float(tp / n_pick),
                      fpr_at_top5=(float(fp / neg) if neg else None))
        if pos:
            result.update(dice_at_top5=float(2*tp / (n_pick+pos)),
                          iou_at_top5=float(tp / (n_pick+pos-tp)),
                          recall_at_top5=float(tp / pos))
    return result

def evaluate(args, out, pairs, case_data, train, test, ref, ref_owners, clustering):
    import torch
    import torch.nn.functional as F
    elapsed = {}
    t_ref = time.perf_counter()
    k = min(args.knn,len(ref))
    finder = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute",
                              n_jobs=args.num_workers).fit(ref)
    # Reference intrinsic scale is train-only, and self-matches are excluded.
    dist0, _ = NearestNeighbors(n_neighbors=min(k+1,len(ref)), metric="cosine",
                  algorithm="brute", n_jobs=args.num_workers).fit(ref).kneighbors(ref)
    norm = float(np.median(dist0[:,1:].mean(axis=1)))
    norm = max(norm,1e-5)
    sizes, coverage = cluster_patient_support(clustering.labels,ref_owners)
    max_size = max(float(sizes.max()),1e-6)
    elapsed["reference_knn_setup"] = time.perf_counter() - t_ref
    elapsed["per_patient"] = {}
    elapsed["method_scoring_and_metric_seconds"] = {m:0.0 for m in METHODS}
    summary_rows = []
    paths = {f.name[:-7]: (f,l) for f,l in pairs}
    for name in sorted(test):
        t_case = time.perf_counter()
        features,zs = case_data[name]
        b,h,w,dim = features.shape
        query = features.reshape(-1,dim)
        t_knn=time.perf_counter()
        ds, neighbor_ids = finder.kneighbors(query)
        knn_seconds=time.perf_counter()-t_knn
        base = ds.mean(axis=1).astype(np.float32) / norm
        rarity = 1 - sizes[neighbor_ids].mean(axis=1) / max_size
        consensus = 1 - coverage[neighbor_ids].mean(axis=1)
        scores = {
            "patch_knn": base,
            "torc_approx": base + 0.25*rarity,
            "torc_consensus": base + 0.25*rarity + 0.25*consensus,
        }
        gt = prepare_metric_mask(paths[name][1],zs,(args.input_size,args.input_size))
        elapsed["per_patient"][name] = {"shared_knn_seconds":knn_seconds,
                                       "methods_seconds":{}}
        for method, flat in scores.items():
            t_method=time.perf_counter()
            grid = flat.reshape(b,1,h,w)
            full = F.interpolate(torch.from_numpy(grid), size=(args.input_size,args.input_size),
                                 mode="bilinear",align_corners=False)[:,0].numpy()
            if args.save_predictions:
                pred_path = out/"predictions"/method/(name+".npz")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pred_path, score_grid=grid[:,0].astype(np.float16), zs=zs)
            metric = metric_one(full,gt)
            secs=time.perf_counter()-t_method
            elapsed["method_scoring_and_metric_seconds"][method] += secs
            elapsed["per_patient"][name]["methods_seconds"][method] = secs
            row = dict(method=method,patient=name,slices=len(zs),**metric)
            summary_rows.append(row)
        elapsed["per_patient"][name]["total_seconds"] = time.perf_counter()-t_case
        print("[evaluated]", name, "slices",len(zs))
    return summary_rows, elapsed

def git_version():
    """Report only a commit hash, never remote URLs or credentials."""
    try:
        p = subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,
                           timeout=4,check=True)
        return p.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return None


def main():
    args = parser().parse_args()
    if (args.slice_stride < 1 or args.max_cases < 0 or args.torc_max_points < 3
            or args.batch_size < 1 or args.knn < 1 or args.ref_per_case < 1
            or args.input_size < 1 or args.patch_grid < 1):
        raise ValueError("Invalid experiment size / reference / kNN arguments")
    np.random.seed(args.seed); random.seed(args.seed)
    started_at = utc_timestamp()
    t0 = time.perf_counter()
    root=Path(args.data)
    if not root.exists():
        raise FileNotFoundError("Missing data/; place original MSD archive in data/archives/")
    # Patient-list discovery first, run type segregates debug from full outputs.
    pairs=locate_data(root)
    if args.max_cases:
        pairs=pairs[:args.max_cases]
        if len(pairs)<3: raise ValueError("debug subset requires at least three cases")
    if len(pairs)<3:
        raise ValueError("Need at least 3 image/label pairs")
    out=Path(args.output)/("debug_"+str(args.max_cases) if args.max_cases else "full")
    out.mkdir(parents=True,exist_ok=True)
    target=out/"experiment_results.json"
    # Ensure the result path is Git-ignored (or entirely outside the repo).
    # Refuse an unsafe destination before deleting or writing any prior report.
    ensure_not_git_trackable(target)
    # Prevent stale 'completed' output from an earlier run being mistaken for the current run.
    target.unlink(missing_ok=True)
    timings={}
    train,test=split_cases(pairs,out/"split.json",args.seed,args.test_fraction)
    print("[cases]",len(pairs),"train",len(train),"test",len(test))
    t=time.perf_counter()
    model,device,weights_id=load_backbone(args)
    timings["model_initialization_seconds"]=time.perf_counter()-t
    fingerprint=hashlib.sha256(json.dumps([args.model_id,weights_id,args.input_size,
                    args.patch_grid,args.slice_stride],sort_keys=True).encode()).hexdigest()[:12]
    cases={}
    timings["feature_extraction_per_case_seconds"]={}
    t=time.perf_counter()
    for image,_ in pairs:
        name=image.name[:-7]
        t_case=time.perf_counter()
        cases[name]=extract_case(image,model,device,args,out/"cache",fingerprint)
        timings["feature_extraction_per_case_seconds"][name]=time.perf_counter()-t_case
        print("[features]",name,cases[name][0].shape)
    timings["feature_extraction_total_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    ref,owners=balanced_reference(cases,train,args.seed,args.ref_per_case,args.torc_max_points)
    timings["reference_sampling_seconds"]=time.perf_counter()-t
    print("[reference]",ref.shape,"patients",len(np.unique(owners)))
    t=time.perf_counter()
    clustering=torque_approx(ref,max_clusters=args.torc_max_clusters)
    timings["torc_clustering_seconds"]=time.perf_counter()-t
    sizes,coverage=cluster_patient_support(clustering.labels, owners)
    # Numeric diagnostics only: no reference patch features or image arrays.
    stats=dict(algorithm="independent_TORC_inspired_approx_NOT_official_parity",
               n_reference=len(ref),n_reference_patients=len(np.unique(owners)),
               reference_dim=int(ref.shape[1]),
               n_clusters=clustering.requested_clusters,
               cluster_counts=clustering.cluster_sizes.tolist(),
               reference_cluster_coverage_by_patch=coverage.tolist(),
               reference_cluster_relative_size_by_patch=sizes.tolist(),
               merge_torques=clustering.merge_torques.tolist())
    t=time.perf_counter()
    rows,eval_timings=evaluate(args,out,pairs,cases,train,test,ref,owners,clustering)
    timings["evaluation_seconds"]=time.perf_counter()-t
    timings["evaluation_breakdown"]=eval_timings
    timings["end_to_end_total_seconds"]=time.perf_counter()-t0
    import torch
    import importlib.metadata
    ver={}
    for name in ("torch","timm","numpy","scikit-learn","nibabel","modelscope"):
        try: ver[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: ver[name]="unknown"
    metadata=dict(
        experiment_mode=("debug_subset_DO_NOT_REPORT_AS_FINAL" if args.max_cases else "full"),
        is_full_slice_coverage=(args.slice_stride==1 and args.max_cases==0),
        args=vars(args),
        weight_snapshot=dict(filename=weights_id[-1],size_bytes=weights_id[0],
                             modified_ns=weights_id[1]),
        backbone=MODEL_NAME, feature_cache_fingerprint=fingerprint,
        versions=ver, git_commit=git_version(), seed=args.seed,
        train_cases=len(train), test_cases=len(test),
        cuda_device=(torch.cuda.get_device_name(0) if device=="cuda" else None),
        peak_cuda_memory_allocated_mb=(float(torch.cuda.max_memory_allocated())/1048576
                                       if device=="cuda" else None),
        results_were_uploaded_to_github=False,
        methodological_warning="TORC-inspired approximation; NOT a validated MATLAB port",
    )
    split_info=dict(seed=args.seed,train=sorted(train),test=sorted(test))
    filepath=export_report(target,methods=METHODS, rows=rows, metadata=metadata,
                           diagnostics=stats,timings=timings,splits=split_info,
                           started_at=started_at)
    print("[complete] all numeric metrics and metadata (no images):",filepath.resolve())
    print("[privacy] report is local only, ignored by Git; attach this JSON directly to ChatGPT.")
    print("[summary] total patients:",len(test),"methods:",len(METHODS))


if __name__=="__main__":
    main()
