"""Exact parallel-ray intersections and seed-connected area for local surface meshes."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

STEP = 2
RADIUS = 96
PITCH_UM = 9.6
AXIS = np.arange(-RADIUS, RADIUS + 1, STEP)
V, U = np.meshgrid(AXIS, AXIS, indexing="ij")
R2 = U*U + V*V


def frame(site):
    normal = np.asarray(site["normal_xyz"], dtype=float)
    normal /= np.linalg.norm(normal)
    u = np.asarray(site["tangent_u_xyz"], dtype=float)
    u -= np.dot(u,normal)*normal
    u /= np.linalg.norm(u)
    v = np.cross(normal,u)
    if np.dot(v,site["tangent_v_xyz"]) < 0:
        v = -v
    return np.stack((u,v,normal))


def faces(xyz, valid=None, quality=None):
    """Keep complete quads, split each along its first-to-last diagonal."""
    xyz = np.asarray(xyz)
    support = np.isfinite(xyz).all(-1) & (xyz >= 0).all(-1)
    if valid is not None:
        support &= valid
    full = support[:-1,:-1] & support[1:,:-1] & support[:-1,1:] & support[1:,1:]
    corners = (np.s_[:-1,:-1],np.s_[:-1,1:],np.s_[1:,1:],np.s_[1:,:-1])
    def select(field):
        a,b,c,d = [field[sl][full] for sl in corners]
        return np.concatenate((np.stack((a,b,c),1),np.stack((a,c,d),1)))
    q = select(np.asarray(quality, dtype=float)) if quality is not None else None
    return select(xyz), q


@dataclass
class Projection:
    low: np.ndarray
    high: np.ndarray
    confidence: np.ndarray
    area_factor: np.ndarray

    @property
    def center(self):
        out = np.full(self.low.shape,np.nan)
        present = np.isfinite(self.low)
        out[present] = (self.low[present]+self.high[present])/2
        return out


def project(triangles, site, quality=None):
    """Rasterize both extremal intersections. Each 2D pixel is a ray in the seed frame."""
    tri = (np.asarray(triangles,dtype=float) - site["seed_xyz"]) @ frame(site).T
    n = len(AXIS)
    low, high = np.full(n*n,np.inf),np.full(n*n,-np.inf)
    qmin, area = np.full(n*n,np.inf),np.full(n*n,-np.inf)
    if len(tri):
        a, b, c = tri[:,0],tri[:,1],tri[:,2]
        ab, ac = b-a,c-a
        normals = np.cross(ab,ac)
        det = normals[:,2]
        finite = np.isfinite(tri).all(axis=(1,2)) & (np.abs(det)>1e-9)
        uv = (tri[:,:,:2]+RADIUS)/STEP
        # Roundoff at shared triangle edges must not create missing grid samples.
        lower = np.maximum(0,np.ceil(uv.min(1)-1e-7)).astype(int)
        upper = np.minimum(n-1,np.floor(uv.max(1)+1e-7)).astype(int)
        extent = upper-lower
        keep = finite & (extent>=0).all(1)
        a,ab,ac,det,normals,lower,extent = (x[keep] for x in (a,ab,ac,det,normals,lower,extent))
        q = np.asarray(quality)[keep] if quality is not None else None
        weight = np.linalg.norm(normals,axis=1)/np.abs(det)
        if len(a):
            max_x,max_y = extent.max(0)
            for dy in range(max_y+1):
                for dx in range(max_x+1):
                    ids = np.flatnonzero((extent[:,0]>=dx)&(extent[:,1]>=dy))
                    if not len(ids):
                        continue
                    ix,iy = lower[ids,0]+dx,lower[ids,1]+dy
                    x,y = ix*STEP-RADIUS-a[ids,0],iy*STEP-RADIUS-a[ids,1]
                    s = (x*ac[ids,1]-y*ac[ids,0])/det[ids]
                    t = (ab[ids,0]*y-ab[ids,1]*x)/det[ids]
                    good = (s>=-1e-7)&(t>=-1e-7)&(s+t<=1+1e-7)
                    if not good.any():
                        continue
                    flat = iy[good]*n+ix[good]
                    ids,s,t = ids[good],s[good],t[good]
                    z = a[ids,2]+s*ab[ids,2]+t*ac[ids,2]
                    score = np.ones(z.shape) if q is None else q[ids,0]*(1-s-t)+q[ids,1]*s+q[ids,2]*t
                    np.minimum.at(low,flat,z)
                    np.maximum.at(high,flat,z)
                    np.minimum.at(qmin,flat,score)
                    np.maximum.at(area,flat,weight[ids])
    return Projection(*(x.reshape(n,n) for x in (low,high,qmin,area)))


def score(reference, prediction, threshold, tolerance=2):
    known = np.isfinite(reference.low)&(reference.high-reference.low<=0.5)&(reference.area_factor<=2)&(R2<=RADIUS**2)
    area = np.where(known,reference.area_factor,0)*((STEP*PITCH_UM)/10000)**2
    present = np.isfinite(prediction.low) & (prediction.confidence+1e-9>=threshold)
    common = known & present
    perpendicular = np.full(known.shape,np.inf)
    perpendicular[common] = np.abs(prediction.center[common]-reference.center[common])/reference.area_factor[common]
    correct = common & (perpendicular<=tolerance+1e-8) & ((prediction.high-prediction.low)<=1+1e-8)
    labs,_ = label(correct)
    centers = np.unique(labs[(R2<=4**2)&correct])
    connected = np.isin(labs,centers) & correct
    area[R2<=12**2]=0
    return dict(reference=float(area.sum()),accepted=float(area[common].sum()),
                incorrect=float(area[common & ~correct].sum()),correct=float(area[correct].sum()),
                connected=float(area[connected].sum()),out_of_reference_rays=int((present&~known&(R2<=RADIUS**2)).sum()))


def load_grid(path,site):
    with np.load(path) as d:
        triangles,q = faces(d["xyz"],d.get("valid"),d.get("quality"))
    return project(triangles,site,q)
