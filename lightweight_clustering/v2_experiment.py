"""V2 end-to-end experiments: CT body ROI + balanced/diverse reference bank.

No model training, no health/control patients, no lesion labels during feature
selection or scoring. Results only exported to a Git-ignored local JSON.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from .experiment import (parser as old_parser, locate_data, split_cases, load_backbone,
                         extract_case, read_slices, prepare_metric_mask,
                         balanced_reference, MODEL_NAME, METHODS as OLD_METHODS)
from .reporting import export_report, ensure_not_git_trackable, utc_timestamp
from .torc_approx import torque_approx
from .body import ct_body_mask, body_patch_candidate_mask
from .sampling import (collect_body_references, representative_torc_bank,
                       propagate_cluster_labels, score_support, normalize)
from .metrics_v2 import patient_metrics

METHODS=("patch_knn","body_patch_knn","diverse_patch_knn",
         "torc_approx","torc_consensus")


def parser():
    p=old_parser()
    p.description=__doc__
    p.add_argument("--legacy-ref-points",type=int,default=512,
                   help="match former original unmasked Patch KNN reference budget")
    p.add_argument("--body-ref-points",type=int,default=512,
                   help="body-filtered baseline, same size as legacy")
    p.add_argument("--diverse-ref-points",type=int,default=8192,
                   help="larger main reference bank for final three methods")
    p.add_argument("--diverse-per-patient",type=int,default=128)
    p.add_argument("--body-hu-threshold",type=float,default=-500)
    p.add_argument("--body-min-occupancy",type=float,default=0.35,
                   help="minimum fraction of body inside reference patch")
    p.add_argument("--knn-query-chunk",type=int,default=2048,
                   help="CUDA/CPU query matmul chunk to constrain memory")
    return p


def get_revision():
    try:
        p=subprocess.run(["git","rev-parse","HEAD"],check=True,timeout=4,
                         capture_output=True,text=True)
        return p.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return None


def knn_cosine(query, reference, *, k, device, chunk=2048):
    """Exact cosine top-k for small/medium banks using chunked CUDA matmul."""
    import torch
    if len(reference)<k or k<1 or chunk<1:
        raise ValueError("Invalid k/bank/chunk for exact KNN")
    ref=torch.as_tensor(normalize(reference),dtype=torch.float32,device=device)
    distances=[]
    neighbor_ids=[]
    for start in range(0,len(query),chunk):
        block=torch.as_tensor(normalize(query[start:start+chunk]),
                              dtype=torch.float32,device=device)
        sim=torch.matmul(block,ref.T)
        vals,idx=torch.topk(sim,k=k,dim=1,largest=True,sorted=True)
        distances.append((1-vals.clamp(-1,1)).clamp_min(0).cpu().numpy())
        neighbor_ids.append(idx.cpu().numpy())
    return np.concatenate(distances).astype(np.float32),np.concatenate(neighbor_ids)


def reference_scale(ref, *, k, device, chunk, seed):
    """Training-only intrinsic distance scale; prevent self-match domination."""
    rng=np.random.default_rng(seed)
    subset=rng.choice(len(ref),size=min(len(ref),1024),replace=False)
    d,ids=knn_cosine(ref[subset],ref,k=min(k+1,len(ref)),device=device,chunk=chunk)
    # Remove the own reference index wherever it appears, not just first by
    # numerical sorting if identical references create ties.
    vals=[]
    for ix,ds,nn in zip(subset,d,ids):
        good=ds[nn!=ix]
        if len(good)>=k:
            vals.append(float(np.mean(good[:k])))
        else:
            vals.append(float(np.mean(good)) if len(good) else 0.)
    return max(1e-5,float(np.median(vals)))


def body_training_candidates(pairs, cases, train, args):
    """Only read image HU data, never the labelsTr masks."""
    paths={f.name[:-7]:f for f,_ in pairs}
    train_masks={}
    all_counts={}
    empty_slices=0
    for name in sorted(train):
        img,_ = read_slices(paths[name], [0])
        zs=cases[name][1]
        candidates=body_patch_candidate_mask(
            img,zs,patch_grid=args.patch_grid,input_size=args.input_size,
            threshold=args.body_hu_threshold,
            minimum_occupancy=args.body_min_occupancy)
        train_masks[name]=candidates
        all_counts[name]=int(candidates.sum())
        empty_slices+=int((~candidates.reshape(len(zs),-1).any(axis=1)).sum())
    return train_masks,all_counts,empty_slices


def evaluate(args, out, pairs, cases, test, old_ref, body_ref, diverse_ref,
             big_cluster_size, big_cluster_coverage, *, device, scales):
    import torch
    import torch.nn.functional as F
    files={f.name[:-7]:(f,l) for f,l in pairs}
    timings={"per_patient":{},"knn_seconds_by_bank":dict(legacy=0.,body=0.,diverse=0.),
             "scoring_seconds_by_method":{m:0. for m in METHODS}}
    rows=[]
    for patient in sorted(test):
        t_patient=time.perf_counter()
        features,zs=cases[patient]
        n_slices,h,w,dim=features.shape
        query=features.reshape(-1,dim)
        # The same CT-only body mask is used for all body-aware methods.
        image,_=read_slices(files[patient][0],[0])
        body=np.stack([ct_body_mask(image[:,:,int(z)],size=args.input_size,
                                    hu_threshold=args.body_hu_threshold) for z in zs])
        if not body.any():
            raise RuntimeError("Empty body region for evaluation patient "+patient)
        # Gold tumor mask is first opened *after* all reference sets, TORC
        # clusters, predictions and body ROIs have been defined.
        gt=prepare_metric_mask(files[patient][1],zs,(args.input_size,args.input_size))
        k=min(args.knn,len(old_ref),len(body_ref),len(diverse_ref))
        knn={}
        for bank,ref in [("legacy",old_ref),("body",body_ref),("diverse",diverse_ref)]:
            t=time.perf_counter()
            knn[bank]=knn_cosine(query,ref,k=k,device=device,chunk=args.knn_query_chunk)
            timings["knn_seconds_by_bank"][bank]+=time.perf_counter()-t
        base=knn["legacy"][0].mean(axis=1)/scales["legacy"]
        body_base=knn["body"][0].mean(axis=1)/scales["body"]
        diverse_base=knn["diverse"][0].mean(axis=1)/scales["diverse"]
        near=knn["diverse"][1]
        rarity=1-big_cluster_size[near].mean(axis=1)/max(float(big_cluster_size.max()),1e-8)
        coverage_penalty=1-big_cluster_coverage[near].mean(axis=1)
        score_flat={
            "patch_knn":base,
            "body_patch_knn":body_base,
            "diverse_patch_knn":diverse_base,
            "torc_approx":diverse_base+0.25*rarity,
            "torc_consensus":diverse_base+0.25*rarity+0.25*coverage_penalty,
        }
        per_method={}
        for method,flat in score_flat.items():
            start=time.perf_counter()
            grid=np.asarray(flat,dtype=np.float32).reshape(n_slices,1,h,w)
            full=F.interpolate(torch.from_numpy(grid),size=(args.input_size,args.input_size),
                               mode="bilinear",align_corners=False)[:,0].numpy()
            if method!="patch_knn":
                for j in range(n_slices):
                    in_body=body[j]
                    if in_body.any():
                        # Outside anatomy must score below inside anatomy.
                        # This never uses labels/GT for scoring.
                        full[j,~in_body]=float(full[j,in_body].min())-1e-5
                    else:
                        full[j,:]=float(full[j].min())-1e-5
            values,curve=patient_metrics(full,gt,body)
            if args.save_predictions:
                pred_path=out/"predictions"/method/(patient+".npz")
                pred_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(pred_path,score_grid=grid[:,0].astype(np.float16),zs=zs)
            row=dict(method=method,patient=patient,slices=len(zs),**values,
                     pr_curve=curve)
            rows.append(row)
            secs=time.perf_counter()-start
            per_method[method]=secs
            timings["scoring_seconds_by_method"][method]+=secs
        timings["per_patient"][patient]=dict(total_seconds=time.perf_counter()-t_patient,
                                             method_eval_seconds=per_method,
                                             axial_slices=len(zs))
        print("[evaluated]",patient,len(zs),"slices",flush=True)
    return rows,timings


def run():
    args=parser().parse_args()
    if not (1<=args.ref_per_case and 3<=args.legacy_ref_points
            and 3<=args.body_ref_points and 3<=args.diverse_ref_points
            and 1<=args.diverse_per_patient and 3<=args.torc_max_points<=1024
            and args.knn_query_chunk>=1 and 0<args.body_min_occupancy<=1
            and args.slice_stride>=1 and args.batch_size>=1 and args.knn>=1
            and args.max_cases>=0):
        raise ValueError("Invalid bank, sampling, body or slice budget")
    started_at=utc_timestamp()
    clock=time.perf_counter()
    rng=np.random.default_rng(args.seed)
    random.seed(args.seed);np.random.seed(args.seed)
    data_root=Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError("Missing data directory, place official raw archive at data/archives/")
    pairs=locate_data(data_root)
    if args.max_cases:
        pairs=pairs[:args.max_cases]
    if len(pairs)<3:
        raise ValueError("Need at least 3 cases")
    out=Path(args.output)/("debug_"+str(args.max_cases) if args.max_cases else "full")
    out.mkdir(parents=True,exist_ok=True)
    target=out/"experiment_results.json"
    ensure_not_git_trackable(target)
    target.unlink(missing_ok=True)
    train,test=split_cases(pairs,out/"split.json",args.seed,args.test_fraction)
    timings={}
    t=time.perf_counter()
    net,device,weights_id=load_backbone(args)
    timings["model_load_seconds"]=time.perf_counter()-t
    fingerprint=hashlib.sha256(json.dumps([args.model_id,weights_id,args.input_size,
                    args.patch_grid,args.slice_stride],sort_keys=True).encode()).hexdigest()[:12]
    cases={}
    timings["features_by_patient_seconds"]={}
    t=time.perf_counter()
    for image,_ in pairs:
        name=image.name[:-7]
        t_case=time.perf_counter()
        cases[name]=extract_case(image,net,device,args,out/"cache",fingerprint)
        timings["features_by_patient_seconds"][name]=time.perf_counter()-t_case
    timings["feature_extraction_seconds"]=time.perf_counter()-t
    # Baseline reproduces old, unmasked and randomly sampled 512-point coreset.
    t=time.perf_counter()
    old_ref,old_owners=balanced_reference(
        cases,train,args.seed,args.ref_per_case,args.legacy_ref_points)
    timings["legacy_reference_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    body_candidates,eligible_per_patient,empty_train_slices=body_training_candidates(
        pairs,cases,train,args)
    timings["body_mask_train_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    body_ref,body_owners,_=collect_body_references(
        cases,body_candidates,train,args.seed,
        per_case=args.ref_per_case,maximum=args.body_ref_points,diverse=False)
    diverse_ref,diverse_owners,_=collect_body_references(
        cases,body_candidates,train,args.seed,
        per_case=args.diverse_per_patient,maximum=args.diverse_ref_points,diverse=True)
    timings["reference_collection_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    proto,proto_owners,proto_idx=representative_torc_bank(
        diverse_ref,diverse_owners,maxpoints=args.torc_max_points,seed=args.seed)
    result=torque_approx(proto,max_clusters=args.torc_max_clusters)
    propagated=propagate_cluster_labels(diverse_ref,proto,result.labels)
    csize,coverage=score_support(propagated,diverse_owners)
    timings["torc_and_propagation_seconds"]=time.perf_counter()-t
    scales={}
    t=time.perf_counter()
    for key,ref in [("legacy",old_ref),("body",body_ref),("diverse",diverse_ref)]:
        scales[key]=reference_scale(ref,k=min(args.knn,len(ref)-1),
                                    device=device,chunk=args.knn_query_chunk,
                                    seed=args.seed)
    timings["reference_scale_seconds"]=time.perf_counter()-t
    print("[reference sizes] legacy",len(old_ref),"body",len(body_ref),
          "diverse",len(diverse_ref),"TORC subset",len(proto),flush=True)
    t=time.perf_counter()
    rows,detail=evaluate(args,out,pairs,cases,test,old_ref,body_ref,diverse_ref,
                         csize,coverage,device=device,scales=scales)
    timings["evaluation_seconds"]=time.perf_counter()-t
    timings["evaluation_breakdown"]=detail
    timings["total_seconds"]=time.perf_counter()-clock
    import torch
    versions={}
    for name in ("torch","timm","numpy","scipy","scikit-learn","nibabel","modelscope"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]="unavailable"
    unique,counts=np.unique(propagated,return_counts=True)
    patient_coverage_by_cluster={
        str(int(cluster)):int(len(np.unique(diverse_owners[propagated==cluster])))
        for cluster in unique
    }
    diagnostics=dict(
        algorithm="independent_TORC_inspired_approx_NOT_official_parity",
        legacy_bank_points=len(old_ref),body_bank_points=len(body_ref),
        diverse_bank_points=len(diverse_ref),torc_representative_points=len(proto),
        legacy_reference_patients=len(np.unique(old_owners)),
        body_reference_patients=len(np.unique(body_owners)),
        diverse_reference_patients=len(np.unique(diverse_owners)),
        number_of_torc_clusters=result.requested_clusters,
        torc_subset_cluster_counts=result.cluster_sizes.tolist(),
        full_reference_cluster_counts={str(int(x)):int(y) for x,y in zip(unique,counts)},
        patient_coverage_by_cluster=patient_coverage_by_cluster,
        merge_torques=result.merge_torques.tolist(),
        body_eligible_patches_train_per_patient=eligible_per_patient,
        body_empty_training_slices=empty_train_slices,
        reference_scales=scales,
        explain="Cluster IDs propagated from small representative TORC bank to large KNN bank using train-only features.",
    )
    metadata=dict(experiment_mode=("debug_subset_DO_NOT_REPORT_AS_FINAL"
                                   if args.max_cases else "full"),
                  is_full_slice_coverage=(args.slice_stride==1 and args.max_cases==0),
                  args=vars(args),
                  backbone=MODEL_NAME,
                  weight_snapshot=dict(filename=weights_id[-1],size_bytes=weights_id[0],
                                       modified_ns=weights_id[1]),
                  feature_cache_fingerprint=fingerprint, versions=versions,
                  git_commit=get_revision(),device=device,
                  cuda_device=(torch.cuda.get_device_name(0) if device=="cuda" else None),
                  peak_cuda_memory_allocated_mb=(
                      float(torch.cuda.max_memory_allocated()/1048576)
                      if device=="cuda" else None),
                  results_were_uploaded_to_github=False,
                  methodological_warning="TORC-inspired only; no official MATLAB parity",
                  ablation_protocol={
                     "patch_knn":"legacy unmasked reference; original whole-frame scoring",
                     "body_patch_knn":"CT-only body sampling 512; scores suppressed outside body",
                     "diverse_patch_knn":"body-filtered patient-balanced diverse larger bank; no TORC",
                     "torc_approx":"same diverse bank + representative TORC-based cluster rarity",
                     "torc_consensus":"same diverse bank + cluster rarity + cross-patient support"
                  },
                  evaluation_rois=["full_image","body_only"],
                  prespecified_top_area_percent=[0.1,0.5,1.,5.],
                  pr_curve_kind="21 sampled numeric precision-recall operating points per patient and ROI",
                  no_health_control_images=True)
    split=dict(seed=args.seed,train=sorted(train),test=sorted(test))
    report=export_report(target,methods=METHODS,rows=rows,metadata=metadata,
                         diagnostics=diagnostics,timings=timings,splits=split,
                         started_at=started_at)
    print("[complete] numeric-only report:",report.resolve(),flush=True)
    print("[privacy] never upload experiment results to GitHub; attach JSON directly to ChatGPT.")


if __name__=="__main__":
    run()
