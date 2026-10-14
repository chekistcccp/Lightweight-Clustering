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
    # Match torch.nn.functional.interpolate(..., mode="nearest") used for
    # evaluation ground-truth masks; linspace endpoints would misalign ROI.
    ys=np.minimum((np.arange(size)*img.shape[0]//size),img.shape[0]-1).astype(np.int64)
    xs=np.minimum((np.arange(size)*img.shape[1]//size),img.shape[1]-1).astype(np.int64)
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
    """Coverage fractions per patch from a summed-area table, NumPy-only."""
    m=np.asarray(mask,dtype=np.float32)
    if m.ndim!=2 or patch_grid<1 or patch_grid>min(m.shape):
        raise ValueError("Invalid body mask / patch_grid")
    h,w=m.shape
    ys=np.linspace(0,h,patch_grid+1).astype(np.int64)
    xs=np.linspace(0,w,patch_grid+1).astype(np.int64)
    integral=np.pad(m,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
    y0,y1=ys[:-1,None],ys[1:,None]
    x0,x1=xs[None,:-1],xs[None,1:]
    sums=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0]
    area=(y1-y0)*(x1-x0)
    return sums/np.maximum(area,1)


def body_patch_candidate_mask(img, zs, *, patch_grid=14, input_size=224,
                              threshold=-500, minimum_occupancy=0.35):
    """Boolean (Z,P,P) candidates. Empty out-of-body slices are allowed."""
    occupancy=[]
    for z in zs:
        m=ct_body_mask(img[:,:,int(z)],size=input_size,hu_threshold=threshold)
        occupancy.append(patch_body_occupancy(m,patch_grid)>=minimum_occupancy)
    return np.asarray(occupancy,dtype=bool)
