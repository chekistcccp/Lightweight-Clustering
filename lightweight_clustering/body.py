"""Training-free HU-based CT body region and patch filtering.

It is a conventional heuristic, NOT a learned abdominal or bowel segmentation.
No tumor masks are accessed; masking errors must be reported numerically.
"""
import numpy as np
from scipy import ndimage as ndi


def ct_body_mask(ct_slice: np.ndarray, size: int = 224, hu_threshold: float = -500) -> np.ndarray:
    img = np.asarray(ct_slice,dtype=np.float32)
    if img.ndim != 2 or size < 16:
        raise ValueError("Expected 2D HU slice and output size >=16")
    # Nearest-neighbor, deterministic; image orientation same as CT backbone.
    ys=np.rint(np.linspace(0,img.shape[0]-1,size)).astype(np.int64)
    xs=np.rint(np.linspace(0,img.shape[1]-1,size)).astype(np.int64)
    hu=img[np.ix_(ys,xs)]
    foreground=np.isfinite(hu) & (hu>hu_threshold) & (hu<3500)
    foreground=ndi.binary_closing(foreground,structure=np.ones((3,3),dtype=bool),iterations=2)
    components,n=ndi.label(foreground)
    if n==0:
        return np.zeros((size,size),dtype=bool)
    counts=np.bincount(components.ravel())
    counts[0]=0
    label=int(np.argmax(counts))
    # Exclude tiny disjoint fragments when the actual patient body is absent.
    if counts[label] < max(64,int(0.01*size*size)):
        return np.zeros((size,size),dtype=bool)
    mask=ndi.binary_fill_holes(components==label)
    return np.asarray(mask,dtype=bool)


def patch_body_occupancy(mask, patch_grid: int = 14):
    """Coverage fractions over each 2D CNN patch, not a tumor mask."""
    import torch
    import torch.nn.functional as F
    if patch_grid < 1:
        raise ValueError("patch_grid must be positive")
    m=torch.from_numpy(np.asarray(mask,dtype=np.float32))[None,None]
    out=F.adaptive_avg_pool2d(m,(patch_grid,patch_grid))
    return out[0,0].numpy()


def body_patch_candidate_mask(img, zs, *, patch_grid=14, input_size=224,
                              threshold=-500, minimum_occupancy=0.35):
    """Boolean (Z,P,P) candidates. Empty out-of-body slices are allowed."""
    occupancy=[]
    for z in zs:
        m=ct_body_mask(img[:,:,int(z)],size=input_size,hu_threshold=threshold)
        occupancy.append(patch_body_occupancy(m,patch_grid)>=minimum_occupancy)
    return np.asarray(occupancy,dtype=bool)
