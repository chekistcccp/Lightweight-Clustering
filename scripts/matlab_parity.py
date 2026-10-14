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
from lightweight_clustering.torc_approx import torque_from_distance


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
    automatic = torque_from_distance(dm).labels
    forced = torque_from_distance(dm,fixed_k=len(np.unique(gt))).labels
    ari_auto=float(adjusted_rand_score(gt,automatic))
    ari_fixed=float(adjusted_rand_score(gt,forced))
    result={"ARI_auto_vs_MATLAB":ari_auto,
            "ARI_fixed_K_vs_MATLAB":ari_fixed,
            "auto_clusters":int(len(np.unique(automatic))),
            "matlab_clusters":int(len(np.unique(gt))),
            "same_partition_auto":bool(abs(ari_auto-1.0)<1e-12),
            "warning":"Uses identical input DM but DIFFERENT independent TORC-inspired algorithm; ARI=1 on one case does not establish general MATLAB parity."}
    from pathlib import Path
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
