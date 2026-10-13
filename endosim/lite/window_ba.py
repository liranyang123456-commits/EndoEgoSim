"""Window-level differentiable dense bundle adjustment for endoscopic egomotion.

Pairs (i,j) contribute flow reprojection residuals. Poses are represented by frame-0-to-frame-i
camera-to-world transforms; inverse depth is optimised per source frame and tied to a learned
network prior. A Schur complement eliminates depth updates before solving the small pose system.
The implementation differentiates through a fixed number of damped GN iterations; robust IRLS
weights are treated as fixed within each iteration.
"""
from __future__ import annotations
import torch
from .egolite import chain


def _skew(v):
    z=torch.zeros_like(v[...,0])
    return torch.stack([torch.stack([z,-v[...,2],v[...,1]],-1),
                        torch.stack([v[...,2],z,-v[...,0]],-1),
                        torch.stack([-v[...,1],v[...,0],z],-1)],-2)

def _se3_exp(xi):
    rho,phi=xi[...,:3],xi[...,3:]
    th=phi.norm(dim=-1,keepdim=True).clamp(min=1e-9)
    Kx=_skew(phi/th); th3=th[...,None]
    I=torch.eye(3,dtype=xi.dtype,device=xi.device).expand(Kx.shape)
    R=I+torch.sin(th3)*Kx+(1-torch.cos(th3))*(Kx@Kx)
    V=I+((1-torch.cos(th3))/th3)*Kx+((th3-torch.sin(th3))/th3)*(Kx@Kx)
    return R,(V@rho[...,None])[...,0]

def dense_ba(T_cw,dinv,dinv_prior,edges,flows,weights,K,u,v,iters=4,prior_sigma=0.15,
             huber=1.5,damping=1e-3,prior_weight=1.0,dt=None):
    """Differentiable dense BA.

    T_cw: (N,4,4) world-to-camera; dinv/prior: (N,h,w), inverse mm;
    edges: sequence of (source_i,target_j); flows (E,2,h,w) pixels; weights (E,h,w).
    K,u,v are at the stride-grid pixel coordinates. Outputs refined poses and inverse depth.
    """
    dev=T_cw.device
    if dt is None: dt=T_cw.dtype
    T=T_cw.to(dt).clone();
    d=dinv.to(dt).clamp(min=1e-6,max=1.0)
    dp=dinv_prior.to(dt).clamp(min=1e-6,max=1.0)
    N, h, w = d.shape
    M = N - 1; D = N * h * w
    fx,fy,cx,cy=(K[0,0].to(dt),K[1,1].to(dt),K[0,2].to(dt),K[1,2].to(dt))
    ray=torch.stack([(u.to(dt)-cx)/fx,(v.to(dt)-cy)/fy,torch.ones_like(u,dtype=dt)],-1)
    ei=torch.as_tensor([a for a,b in edges],device=dev,dtype=torch.long)
    ej=torch.as_tensor([b for a,b in edges],device=dev,dtype=torch.long)
    q=torch.stack([u.to(dt),v.to(dt)],-1)[None]+flows.to(dt).permute(0,2,3,1)
    for _ in range(iters):
        Ri,ti,Rj,tj=T[ei,:3,:3],T[ei,:3,3],T[ej,:3,:3],T[ej,:3,3]
        Rji=Rj@Ri.transpose(1,2); tji=tj-(Rji@ti[...,None])[...,0]
        di=d[ei]; Xi=ray[None]/di[...,None].clamp(min=1e-6)
        Xj=torch.einsum('eab,ehwb->ehwa',Rji,Xi)+tji[:,None,None]
        z=Xj[...,2].clamp(min=1e-3)
        proj=torch.stack([fx*Xj[...,0]/z+cx,fy*Xj[...,1]/z+cy],-1)
        r=q-proj; rn=r.norm(dim=-1)
        wrob=torch.where(rn>huber,huber/rn.clamp(min=1e-9),torch.ones_like(rn)).detach()
        robust=weights.to(dt)*wrob*(Xj[...,2]>1e-3)
        # Scalar IRLS weights form per-pixel 2x2 identity blocks; broadcasting (E,H,W) over matrix axes
        # can create incorrect broadcasting when the number of flow edges coincides with a spatial size.
        Wt=robust[...,None,None]
        Jp=torch.zeros(*z.shape,2,3,dtype=dt,device=dev)
        Jp[...,0,0]=fx/z; Jp[...,0,2]=-fx*Xj[...,0]/z**2
        Jp[...,1,1]=fy/z; Jp[...,1,2]=-fy*Xj[...,1]/z**2
        I3=torch.eye(3,dtype=dt,device=dev).expand(*z.shape,3,3)
        Jj=Jp@torch.cat([I3,-_skew(Xj)],-1)
        Ji=-Jp@torch.einsum('eab,ehwbc->ehwac',Rji,torch.cat([I3,-_skew(Xi)],-1))
        dXd=torch.einsum('eab,ehwb->ehwa',Rji,-Xi/di[...,None].clamp(min=1e-6))
        Jd=Jp@dXd[...,None]
        Hpp=torch.zeros(N,N,6,6,dtype=dt,device=dev); gp=torch.zeros(N,6,dtype=dt,device=dev)
        JiT,JjT,JdT=Ji.transpose(-1,-2),Jj.transpose(-1,-2),Jd.transpose(-1,-2)
        Hii=(JiT@(Wt*Ji)).sum((1,2)); Hjj=(JjT@(Wt*Jj)).sum((1,2)); Hij=(JiT@(Wt*Jj)).sum((1,2))
        gi=(JiT@(Wt*r[...,None])).sum((1,2))[...,0]; gj=(JjT@(Wt*r[...,None])).sum((1,2))[...,0]
        Hpp.index_put_((ei,ei),Hii,accumulate=True); Hpp.index_put_((ej,ej),Hjj,accumulate=True)
        Hpp.index_put_((ei,ej),Hij,accumulate=True); Hpp.index_put_((ej,ei),Hij.transpose(1,2),accumulate=True)
        gp.index_add_(0,ei,gi); gp.index_add_(0,ej,gj)
        Hdd=torch.zeros(N,h,w,dtype=dt,device=dev); gd=torch.zeros(N,h,w,dtype=dt,device=dev)
        Hdd.index_add_(0,ei,(JdT@(Wt*Jd))[...,0,0]); gd.index_add_(0,ei,(JdT@(Wt*r[...,None]))[...,0,0])
        lam=prior_weight/(prior_sigma*dp).clamp(min=1e-9)**2
        Hdd=Hdd+lam; gd=gd+lam*(dp-d)
        Hpd=torch.zeros(N,N*h*w,6,dtype=dt,device=dev)
        pix=(ei[:,None,None]*h*w+torch.arange(h*w,device=dev).view(1,h,w)).reshape(-1)
        Hid=(JiT@(Wt*Jd))[...,0].reshape(-1,6); Hjd=(JjT@(Wt*Jd))[...,0].reshape(-1,6)
        Hpd.index_put_((ei[:,None].expand(-1,h*w).reshape(-1),pix),Hid,accumulate=True)
        Hpd.index_put_((ej[:,None].expand(-1,h*w).reshape(-1),pix),Hjd,accumulate=True)
        Hm=Hpp[1:,1:].permute(0,2,1,3).reshape(6*M,6*M); gm=gp[1:].reshape(6*M)
        C=Hpd[1:].permute(0,2,1).reshape(6*M,D); hd=Hdd.reshape(D); gdv=gd.reshape(D)
        Q=C/hd[None]; S=Hm-Q@C.T; rhs=gm-Q@gdv
        S=S+torch.diag(damping*torch.diagonal(S)+1e-6)
        dx=torch.linalg.solve(S,rhs)
        dd=(gdv-C.T@dx)/hd
        dR,dtt=_se3_exp(dx.view(M,6))
        rows=[T[0]]
        for i in range(1,N):
            Ri=dR[i-1]@T[i,:3,:3]; ti=dR[i-1]@T[i,:3,3]+dtt[i-1]
            Ti=torch.eye(4,dtype=dt,device=dev); Ti=Ti.clone(); Ti[:3,:3]=Ri; Ti[:3,3]=ti
            rows.append(Ti)
        T=torch.stack(rows); d=(d+dd.view(N,h,w)).clamp(min=1e-5)
    return T,d

def ba_window(R1,t1,Z1,pairs,K,u,v,logw=None,prior_weight=1.0,iters=4,unit=50.0,dt=None):
    """Build consecutive initialization and optimize a clip with multi-gap flow edges."""
    N=R1.shape[0]+1
    T_c2w=chain(R1,t1*unit,torch.ones(N-1,device=R1.device))
    T_cw=torch.linalg.inv(T_c2w)
    dp=1.0/(Z1*unit); dp=torch.cat([dp,dp[-1:]],0)
    edges=[]; flows=[]; weights=[]
    for s in sorted(pairs):
        f,w=pairs[s]
        edges += [(i,i+s) for i in range(N-s)]
        flows.append(f)
        if logw is not None: w=w*torch.exp(logw[min(s-1,len(logw)-1)])
        weights.append(w)
    return dense_ba(T_cw,dp,dp,edges,torch.cat(flows),torch.cat(weights),K,u,v,
                    iters=iters,prior_weight=prior_weight,dt=dt)
