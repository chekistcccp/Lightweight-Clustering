"""Query-local cross-patient support, replacing ineffective global cluster coverage.

Uses only frozen feature similarities, reference patient identities, and fixed
train-only calibration. It never reads tumor masks or patient clinical labels.
"""
import numpy as np


def local_patient_penalty(distances, neighbor_ids, owner_codes, *, temperature):
    """Weighted Simpson concentration of support among local kNN.

    A value near zero means nearest reference patches come from distinct
    patients; a value near one means they concentrate on a single patient.
    Identical patient occurrence in every *global* TORC cluster has no impact.
    Shape: (n_queries, k); output (n_queries,), no image arrays retained.
    """
    d=np.asarray(distances,dtype=np.float32)
    ids=np.asarray(neighbor_ids,dtype=np.int64)
    owners=np.asarray(owner_codes)
    if (d.ndim!=2 or ids.shape!=d.shape or d.shape[1]<2 or
            not np.isfinite(d).all() or not np.isfinite(temperature) or
            temperature<=0 or ids.min()<0 or ids.max()>=len(owners)):
        raise ValueError("Invalid local distances/indices/temperature")
    k=d.shape[1]
    result=np.empty(len(d),dtype=np.float32)
    # Bound temporary k*k patient-id comparisons, regardless of input size.
    for start in range(0,len(d),4096):
        dist=d[start:start+4096]
        patient=owners[ids[start:start+4096]]
        logits=-(dist-dist.min(axis=1,keepdims=True))/temperature
        weights=np.exp(np.maximum(logits,-60)).astype(np.float32)
        weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
        same=patient[:,:,None]==patient[:,None,:]
        probability=(weights[:,:,None]*weights[:,None,:]*same).sum(axis=(1,2))
        result[start:start+len(dist)]=np.clip((probability-1./k)/(1.-1./k),0.,1.)
    return result


def support_diagnostics(values):
    x=np.asarray(values,dtype=np.float64).ravel()
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size),mean=float(x.mean()),std=float(x.std()),
                q05=float(np.quantile(x,.05)),median=float(np.median(x)),
                q95=float(np.quantile(x,.95)),nonzero_fraction=float(np.mean(x>1e-5)),
                variation_detected=bool(np.ptp(x)>1e-5))


def local_support_variance(patients, k):
    """Simple unused independent indicator for tests/diagnostics."""
    p=np.asarray(patients)
    if p.ndim!=2 or p.shape[1]!=k:raise ValueError("Wrong neighbor patient array")
    unique=np.asarray([len(np.unique(row)) for row in p],dtype=np.float32)
    return 1.-(unique-1.)/max(k-1,1)
