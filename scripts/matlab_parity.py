"""Compare *labels* from user-supplied official MATLAB TORC on identical distance matrices.

Generate gold file in MATLAB (users run official code on their own MATLAB machine):
  rng(42); X=rand(40,8);
  DM=pdist2(X,X,'cosine');
  [Idx,Idx_with_noise]=TorqueClustering(DM,0,1,0);
  save('golden.mat','DM','Idx','Idx_with_noise');

This Python prototype is intentionally NOT official TORC and parity may fail.
Usage: python scripts/matlab_parity.py --golden /path/to/golden.mat
"""
import argparse
import json
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import MDS
from lightweight_clustering.torc_approx import torque_approx


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden",required=True)
    ap.add_argument("--output",default="outputs/parity_report.json")
    args=ap.parse_args()
    data=loadmat(args.golden)
    if "DM" not in data or "Idx" not in data:
        raise KeyError("MATLAB export must contain DM and Idx")
    dm=np.asarray(data["DM"],np.float64)
    gt=np.asarray(data["Idx"]).ravel()
    if dm.shape != (len(gt),len(gt)):
        raise ValueError("Invalid MATLAB distance/label shape")
    # Build a faithful Euclidean embedding of the supplied distances as an
    # approximate comparison only; cosine inputs cannot be recovered exactly.
    eigenvals,eigenvectors=np.linalg.eigh(-0.5*(np.eye(len(gt))-
                 np.ones_like(dm)/len(gt))@(dm**2)@(np.eye(len(gt))-
                 np.ones_like(dm)/len(gt)))
    pos=np.maximum(eigenvals,0)
    x=eigenvectors*np.sqrt(pos)[None,:]
    pred=torque_approx(x,fixed_k=len(np.unique(gt))).labels
    ari=float(adjusted_rand_score(gt,pred))
    result={"ARI_vs_MATLAB_labels":ari,
            "same_partition":bool(abs(ari-1.0)<1e-12),
            "warning":"NOT an exact parity test of original DM: metric embedding followed by cosine distance changes geometry. Strict parity requires a direct distance-matrix implementation."}
    from pathlib import Path
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
