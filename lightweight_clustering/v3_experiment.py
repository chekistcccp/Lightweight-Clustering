"""V3: factorial bank-size/diversity experiment and *local* patient support.

A single end-to-end run uses the identical patient split across several
predeclared sampling seeds, keeping the previous CT feature cache. No normal
patients, no training or GT-guided reference selection, no remote result upload.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np

from .v2_experiment import (parser as old_parser,get_revision,knn_cosine,
                            reference_scale,body_training_candidates)
from .experiment import (locate_data,split_cases,load_backbone,extract_case,
                         balanced_reference,read_slices,prepare_metric_mask,MODEL_NAME)
from .body import ct_body_mask
from .sampling import (collect_body_references,representative_torc_bank,
                       propagate_cluster_labels,score_support)
from .local_support import local_patient_penalty,support_diagnostics
from .metrics_v2 import patient_metrics
from .torc_approx import torque_approx
from .reporting import ensure_not_git_trackable,utc_timestamp
from .v3_reporting import write_multiseed_report

METHODS=("patch_knn","body_patch_knn","body_diverse_512",
         "body_random_8192","diverse_patch_knn","torc_approx",
         "local_consensus_knn","torc_consensus")


def parser():
    p=old_parser()
    p.description=__doc__
    p.add_argument("--sampling-seeds",nargs="+",type=int,default=[42,123,2026],
                   help="Repeat bank sampling with FIXED original patient split; not independent patient folds")
    p.add_argument("--local-support-k",type=int,default=16,
                   help="Number of local reference neighbors for cross-patient support")
    p.add_argument("--local-support-weight",type=float,default=0.25)
    p.add_argument("--torc-rarity-weight",type=float,default=0.25)
    return p


def _validate(args):
    if (args.max_cases<0 or args.slice_stride<1 or args.knn<1 or
        args.legacy_ref_points<3 or args.body_ref_points<3 or
        args.diverse_ref_points<3 or args.diverse_per_patient<1 or
        args.torc_max_points<3 or args.torc_max_points>1024 or
        args.knn_query_chunk<1 or args.batch_size<1 or
        args.local_support_k<2 or args.local_support_k<args.knn or
        args.local_support_weight<0 or args.torc_rarity_weight<0 or
        not np.isfinite(args.local_support_weight) or
        not np.isfinite(args.torc_rarity_weight) or
        not args.sampling_seeds or len(set(args.sampling_seeds))!=len(args.sampling_seeds) or
        not 0<args.body_min_occupancy<=1):
        raise ValueError("Invalid sizes, local k, coefficients or sampling seeds")
    if args.body_ref_points>args.diverse_ref_points:
        raise ValueError("Small bank cannot exceed large bank")
    if args.local_support_k>args.diverse_ref_points:
        raise ValueError("Support-k exceeds configured large reference bank")


def archive_previous_report(target,root):
    target=Path(target)
    ensure_not_git_trackable(target)
    if target.exists():
        folder=Path(root)/"history"/(target.parent.name+"_"+time.strftime("%Y%m%d_%H%M%S",time.gmtime()))
        previous=folder/"experiment_results.json"
        ensure_not_git_trackable(previous)
        folder.mkdir(parents=True,exist_ok=True)
        if previous.exists():raise FileExistsError("Local result archive conflict: "+str(previous))
        target.replace(previous)
        print("[previous local report archived]",previous,flush=True)


def build_seed_banks(cases,train,candidates,args,sampling_seed,device):
    """Fixed 2x2 factorial, with 512-prefix nested in same 8192 candidate bank."""
    t=time.perf_counter()
    old_ref,old_owners=balanced_reference(
        cases,train,sampling_seed,args.ref_per_case,args.legacy_ref_points)
    # Note: 512 in V3 is the nested prefix of the same 8192 sampled bank;
    # this is intentional for size-only factorial comparisons. Consequently
    # body_patch_knn in V3 need not match the old V2 512-point bank bitwise.
    random_large,owners_random,_=collect_body_references(
        cases,candidates,train,sampling_seed,per_case=args.diverse_per_patient,
        maximum=args.diverse_ref_points,diverse=False)
    diverse_large,owners_diverse,_=collect_body_references(
        cases,candidates,train,sampling_seed,per_case=args.diverse_per_patient,
        maximum=args.diverse_ref_points,diverse=True)
    small_random=random_large[:args.body_ref_points]
    small_diverse=diverse_large[:args.body_ref_points]
    banks=dict(legacy=old_ref,random_small=small_random,diverse_small=small_diverse,
               random_large=random_large,diverse_large=diverse_large)
    owners=dict(legacy=old_owners,random_small=owners_random[:args.body_ref_points],
                diverse_small=owners_diverse[:args.body_ref_points],
                random_large=owners_random,diverse_large=owners_diverse)
    if min(map(len,banks.values()))<max(args.knn,3):
        raise ValueError("Not enough eligible reference Patch samples for chosen knn")
    times=dict(collection_seconds=time.perf_counter()-t)
    t=time.perf_counter()
    proto,proto_owners,proto_ids=representative_torc_bank(
        diverse_large,owners_diverse,maxpoints=args.torc_max_points,seed=sampling_seed)
    result=torque_approx(proto,max_clusters=args.torc_max_clusters)
    propagated=propagate_cluster_labels(diverse_large,proto,result.labels)
    sizes,global_coverage=score_support(propagated,owners_diverse)
    times["torc_and_propagation_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    scales={key:reference_scale(ref,k=args.knn,device=device,
                                chunk=args.knn_query_chunk,seed=sampling_seed)
            for key,ref in banks.items()}
    times["reference_scale_seconds"]=time.perf_counter()-t
    owner_ids=np.unique(owners_diverse,return_inverse=True)[1].astype(np.int32)
    bank_stats={name:dict(n_patches=int(len(ref)),n_distinct_train_patients=
                           int(len(np.unique(owners[name]))))
                for name,ref in banks.items()}
    uniq,ct=np.unique(propagated,return_counts=True)
    coverage={str(int(u)):int(len(np.unique(owners_diverse[propagated==u]))) for u in uniq}
    diagnostic=dict(
        bank_design="same per-patient sampled large bank, first-N interleaved prefix for small bank",
        bank_stats=bank_stats,
        reference_scales=scales,
        torc_prototype_count=int(len(proto)),torc_clusters=int(result.requested_clusters),
        torc_prototype_cluster_sizes=result.cluster_sizes.tolist(),
        large_bank_torc_cluster_counts={str(int(k)):int(n) for k,n in zip(uniq,ct)},
        global_cluster_patient_coverage=coverage,
        torc_merge_torques=result.merge_torques.tolist(),
        local_support_definition="distance-weighted Simpson same-patient concentration over query-local top-k",
        local_support_k=args.local_support_k,
        local_support_temperature=float(scales["diverse_large"]),
        locality_vs_global="Unlike V2, this support does NOT use entire-cluster patient coverage",
    )
    # No GT, tumor labels or test-case ROI used in bank construction.
    return dict(banks=banks,owners=owners,cluster_sizes=sizes,
                owner_codes=owner_ids,timings=times,diagnostics=diagnostic)


def evaluate_seed(args,out,pairs,cases,test,built,*,sampling_seed,device):
    import torch
    import torch.nn.functional as F
    paths={f.name[:-7]:(f,l) for f,l in pairs}
    banks=built["banks"]
    scales=built["diagnostics"]["reference_scales"]
    size=built["cluster_sizes"]
    local_temp=built["diagnostics"]["local_support_temperature"]
    owner_codes=built["owner_codes"]
    rows=[]
    details={"by_patient":{},"bank_knn_seconds":{b:0. for b in banks},
             "method_scoring_seconds":{m:0. for m in METHODS}}
    penalties=[]
    rarity_samples=[]
    for patient in sorted(test):
        t_patient=time.perf_counter()
        features,zs=cases[patient]
        slices,h,w,dim=features.shape
        query=features.reshape(-1,dim)
        img,_=read_slices(paths[patient][0],[0])
        body=np.stack([ct_body_mask(img[:,:,int(z)],size=args.input_size,
                                    hu_threshold=args.body_hu_threshold) for z in zs])
        if not body.any():
            raise RuntimeError("No CT body pixels found in evaluation: "+patient)
        distances={}
        for key,ref in banks.items():
            t=time.perf_counter()
            requested=max(args.knn,args.local_support_k) if key=="diverse_large" else args.knn
            requested=min(requested,len(ref))
            distances[key]=knn_cosine(query,ref,k=requested,device=device,
                                      chunk=args.knn_query_chunk)
            details["bank_knn_seconds"][key]+=time.perf_counter()-t
        def base(key):
            return distances[key][0][:,:args.knn].mean(axis=1)/scales[key]
        standard=base("diverse_large")
        near=distances["diverse_large"][1][:,:args.knn]
        rarity=1.-size[near].mean(axis=1)/max(float(size.max()),1e-8)
        local=local_patient_penalty(distances["diverse_large"][0][:,:args.local_support_k],
                                    distances["diverse_large"][1][:,:args.local_support_k],
                                    owner_codes,temperature=local_temp)
        penalties.append(support_diagnostics(local))
        rarity_samples.append(float(rarity.mean()))
        scores=dict(
            patch_knn=base("legacy"),
            body_patch_knn=base("random_small"),
            body_diverse_512=base("diverse_small"),
            body_random_8192=base("random_large"),
            diverse_patch_knn=standard,
            torc_approx=standard+args.torc_rarity_weight*rarity,
            local_consensus_knn=standard+args.local_support_weight*local,
            torc_consensus=standard+args.torc_rarity_weight*rarity+
                           args.local_support_weight*local,
        )
        # Ground truth is accessed only AFTER bank construction and all score
        # vectors are finalized. Never use GT to choose reference or thresholds.
        gt=prepare_metric_mask(paths[patient][1],zs,(args.input_size,args.input_size))
        timings={}
        for method in METHODS:
            t=time.perf_counter()
            grid=np.asarray(scores[method],dtype=np.float32).reshape(slices,1,h,w)
            full=F.interpolate(torch.from_numpy(grid),size=(args.input_size,args.input_size),
                               mode="bilinear",align_corners=False)[:,0].numpy()
            if method!="patch_knn":
                for k in range(slices):
                    inside=body[k]
                    if inside.any():
                        full[k,~inside]=float(full[k,inside].min())-1e-5
                    else:
                        full[k,:]=float(full[k].min())-1e-5
            metrics,pr_curve=patient_metrics(full,gt,body)
            if args.save_predictions:
                dst=out/"predictions"/("seed_"+str(sampling_seed))/method/(patient+".npz")
                dst.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dst,score_grid=grid[:,0].astype(np.float16),zs=zs)
            rows.append(dict(patient=patient,method=method,slices=len(zs),
                             pr_curve=pr_curve,**metrics))
            dt=time.perf_counter()-t
            timings[method]=dt
            details["method_scoring_seconds"][method]+=dt
        details["by_patient"][patient]=dict(slices=len(zs),
                                           total_seconds=time.perf_counter()-t_patient,
                                           per_method_seconds=timings,
                                           local_patient_support=penalties[-1],
                                           mean_cluster_rarity=rarity_samples[-1])
        print("[sample seed",sampling_seed,"] [evaluated]",patient,flush=True)
    details["local_support_per_patient"]=penalties
    details["mean_cluster_rarity_per_patient"]=rarity_samples
    return rows,details


def run():
    args=parser().parse_args()
    _validate(args)
    started_at=utc_timestamp()
    start=time.perf_counter()
    random.seed(args.seed);np.random.seed(args.seed)
    if not Path(args.data).exists():
        raise FileNotFoundError("Data missing; put official raw Task10_Colon archive under data/archives")
    pairs=locate_data(Path(args.data))
    if args.max_cases:
        pairs=pairs[:args.max_cases]
    if len(pairs)<3:raise ValueError("Need >=3 patient CTs")
    out=Path(args.output)/("debug_"+str(args.max_cases) if args.max_cases else "full")
    out.mkdir(parents=True,exist_ok=True)
    target=out/"experiment_results.json"
    archive_previous_report(target,args.output)
    train,test=split_cases(pairs,out/"split.json",args.seed,args.test_fraction)
    total_timings={}
    t=time.perf_counter()
    model,device,weights_id=load_backbone(args)
    total_timings["model_load_seconds"]=time.perf_counter()-t
    fingerprint=hashlib.sha256(json.dumps([args.model_id,weights_id,args.input_size,
                    args.patch_grid,args.slice_stride],sort_keys=True).encode()).hexdigest()[:12]
    cases={}
    t=time.perf_counter()
    for image,_ in pairs:
        name=image.name[:-7]
        cases[name]=extract_case(image,model,device,args,out/"cache",fingerprint)
    total_timings["cache_feature_extraction_seconds"]=time.perf_counter()-t
    t=time.perf_counter()
    candidates,counts,empty_slices=body_training_candidates(pairs,cases,train,args)
    total_timings["body_training_preprocess_seconds"]=time.perf_counter()-t
    records=[]
    for sample_seed in args.sampling_seeds:
        print("[start sampling seed]",sample_seed,flush=True)
        t=time.perf_counter()
        built=build_seed_banks(cases,train,candidates,args,sample_seed,device)
        eval_t=time.perf_counter()
        rows,details=evaluate_seed(args,out,pairs,cases,test,built,
                                   sampling_seed=sample_seed,device=device)
        built["timings"]["evaluation_seconds"]=time.perf_counter()-eval_t
        built["timings"]["per_patient_and_bank_breakdown"]=details
        built["timings"]["sampling_seed_total_seconds"]=time.perf_counter()-t
        built["diagnostics"]["local_support_diagnostics_by_patient"]=details["local_support_per_patient"]
        built["diagnostics"]["cluster_rarity_mean_by_patient"]=details["mean_cluster_rarity_per_patient"]
        records.append(dict(sampling_seed=sample_seed,rows=rows,
                            diagnostics=built["diagnostics"],timings=built["timings"]))
        print("[finished sampling seed]",sample_seed,flush=True)
    total_timings["overall_seconds"]=time.perf_counter()-start
    import torch
    versions={}
    for lib in ("torch","timm","numpy","scipy","scikit-learn","nibabel","modelscope"):
        try:versions[lib]=importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:versions[lib]="not-found"
    metadata=dict(
        experiment_mode=("debug_subset_DO_NOT_REPORT_AS_FINAL" if args.max_cases else "full"),
        is_full_slice_coverage=(not args.max_cases and args.slice_stride==1),
        args=vars(args),weight_snapshot=dict(filename=weights_id[-1],
             size_bytes=weights_id[0],modified_ns=weights_id[1]),
        backbone=MODEL_NAME,feature_cache_fingerprint=fingerprint,
        versions=versions,git_commit=get_revision(),device=device,
        cuda_device=(torch.cuda.get_device_name(0) if device=="cuda" else None),
        peak_cuda_memory_allocated_mb=(float(torch.cuda.max_memory_allocated())/1048576
                                       if device=="cuda" else None),
        train_cases=len(train),test_cases=len(test),
        body_eligible_training_patches=counts,empty_body_training_slices=empty_slices,
        key_design="2x2 reference size vs farthest diversity, local query-patient support independent of global clusters",
        sampling_seeds_share_same_patient_split=True,
        patient_test_split_was_seen_in_prior_exploratory_runs=True,
        results_were_uploaded_to_github=False,
        methodology_warning="TORC-inspired approximation, not official MATLAB equivalent",
        expected_method_order=list(METHODS),
        no_normal_control_patients=True,
    )
    report=write_multiseed_report(target,methods=METHODS,records=records,
                                  split=dict(seed=args.seed,train=sorted(train),test=sorted(test)),
                                  metadata=metadata,common_timings=total_timings,
                                  started_at=started_at)
    print("[complete] all seeds and metrics in ONE local JSON:",report.resolve(),flush=True)
    print("[privacy] Never synchronize results to GitHub; manually upload JSON in this chat.")


if __name__=="__main__":
    run()
