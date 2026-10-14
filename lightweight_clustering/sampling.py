"""Balanced training-patient sampling and cosine farthest-first coverage.

No masks/labels are used; the body candidate mask is derived solely from CT HU.
The enlarged KNN bank and TORC prototype subset are deliberately separate.
"""
import numpy as np
from .torc_approx import cluster_patient_support


def normalize(x):
    v=np.asarray(x,dtype=np.float32)
    return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)


def farthest_indices(x, take, rng, initial=()):
    """Streaming farthest-first selection; O(N*K*D) and no NxN matrix."""
    v=normalize(x)
    n=len(v)
    if take<1 or n<take:
        raise ValueError("Invalid farthest-first budget")
    chosen=list(dict.fromkeys(int(k) for k in initial))
    if len(chosen)>take:
        raise ValueError("More required anchors than selection budget")
    if not chosen:
        chosen=[int(rng.integers(n))]
    picked=np.zeros(n,dtype=bool)
    picked[chosen]=True
    min_distance=np.full(n,np.inf,dtype=np.float32)
    for anchor in chosen:
        min_distance=np.minimum(min_distance,1-v@v[anchor])
    while len(chosen)<take:
        eligible=np.where(~picked,min_distance,-np.inf)
        nxt=int(np.argmax(eligible))
        picked[nxt]=True
        chosen.append(nxt)
        min_distance=np.minimum(min_distance,1-v@v[nxt])
    return np.asarray(chosen,dtype=np.int64)


def collect_body_references(cases, body_candidates, train, seed, *,
                            per_case, maximum, diverse, candidate_pool=384):
    """Each patient contributes up to 'per_case' body patches; round-robin cap."""
    rng=np.random.default_rng(seed)
    pool={}
    counts={}
    for name in sorted(train):
        features=np.asarray(cases[name][0],dtype=np.float32)
        if name not in body_candidates or body_candidates[name].shape!=features.shape[:3]:
            raise ValueError("Body candidate mask mismatches cached features: "+name)
        candidates=features.reshape(-1,features.shape[-1])[
            body_candidates[name].reshape(-1)]
        counts[name]=int(len(candidates))
        if not len(candidates):
            raise RuntimeError("No body-positive patches for training case "+name)
        if diverse:
            subset=rng.choice(len(candidates),size=min(len(candidates),candidate_pool),replace=False)
            candidates=candidates[subset]
            indices=farthest_indices(candidates,min(per_case,len(candidates)),rng)
            pool[name]=normalize(candidates[indices])
        else:
            idx=rng.choice(len(candidates),size=min(per_case,len(candidates)),replace=False)
            pool[name]=normalize(candidates[idx])
    selected=[];owners=[]
    for i in range(max(map(len,pool.values()))):
        for name in sorted(pool):
            if i<len(pool[name]) and len(selected)<maximum:
                selected.append(pool[name][i]);owners.append(name)
    if len(selected)<3:
        raise RuntimeError("Too few eligible patient-balanced reference patches")
    return np.asarray(selected,dtype=np.float32),np.asarray(owners),counts


def representative_torc_bank(ref, owners, *, maxpoints, seed, pool_limit=2048):
    """Select train-only representative subset; start with one per patient."""
    ref=normalize(ref)
    owners=np.asarray(owners)
    if len(ref)!=len(owners) or maxpoints<3:
        raise ValueError("TORC reference or budget invalid")
    if len(ref)<=maxpoints:
        return ref,owners,np.arange(len(ref),dtype=np.int64)
    rng=np.random.default_rng(seed)
    anchors=[int(rng.choice(np.flatnonzero(owners==p))) for p in np.unique(owners)]
    if len(anchors)>maxpoints:
        # Preserve balanced patient coverage to the extent supported by the budget.
        anchors=list(rng.choice(np.asarray(anchors),size=maxpoints,replace=False))
    anchor_set=set(anchors)
    other=np.asarray([i for i in range(len(ref)) if i not in anchor_set],dtype=np.int64)
    need=max(0,min(pool_limit,len(ref))-len(anchors))
    sampled=rng.choice(other,size=min(need,len(other)),replace=False)
    candidates=np.asarray(anchors+list(sampled),dtype=np.int64)
    # Anchors are the first indices in candidate array; the remaining are diverse.
    selected_local=farthest_indices(ref[candidates],maxpoints,rng,
                                     initial=range(len(anchors)))
    idx=candidates[selected_local]
    return ref[idx],owners[idx],idx


def propagate_cluster_labels(ref, torc_features, cluster_labels):
    """Assign all main-bank patches to nearest TORC prototype, not GT class."""
    ref=normalize(ref)
    sub=normalize(torc_features)
    labels=np.asarray(cluster_labels,dtype=np.int32)
    if len(sub)!=len(labels):
        raise ValueError("TORC prototype and cluster label mismatch")
    out=[]
    for offset in range(0,len(ref),2048):
        sim=ref[offset:offset+2048]@sub.T
        out.extend(labels[np.argmax(sim,axis=1)].tolist())
    return np.asarray(out,dtype=np.int32)


def score_support(full_cluster_labels, full_owners):
    size,coverage=cluster_patient_support(full_cluster_labels,full_owners)
    # Standard support uses ALL main reference points and ALL train patients.
    return np.asarray(size,dtype=np.float32),np.asarray(coverage,dtype=np.float32)
