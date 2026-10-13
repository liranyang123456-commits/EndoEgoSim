"""EgoLite: lightweight ego-motion network with pixel-wise motion disentanglement.

For a frame pair (t, t+1) the network predicts, on a 1/8-resolution grid,
  * a pseudo-3D surface: log-depth = beta * centred pseudo-height + residual,
    normalised so that the median depth of frame t is 1;
  * per-pixel motion-layer probabilities (stationary, deforming, instrument);
  * a per-pixel log-variance of the observed flow.
The observed flow is split into a camera term, computed analytically from the
surface and the relative pose, and a scene term (the remainder). The relative
pose is solved by a few damped Gauss-Newton steps on the camera term, weighted
by the stationary probability, a learned share of the deforming probability,
and the inverse variance. Instrument pixels carry no weight.

Relative poses are chained into an absolute trajectory with T_wc(0) = I.
The scale of pair t+1 is converted into the units of pair t by the median depth
that the solved pose assigns to frame-t points in frame t+1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def se3_exp(xi: torch.Tensor):
    """xi = (rho, phi), shape (..., 6). Returns R (...,3,3), t (...,3)."""
    rho, phi = xi[..., :3], xi[..., 3:]
    theta = phi.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = skew(phi / theta)
    th = theta[..., None]
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(K.shape)
    small = (theta[..., 0] < 1e-4)[..., None, None]
    R = eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    V = eye + ((1 - torch.cos(th)) / th) * K + ((th - torch.sin(th)) / th) * (K @ K)
    Pk = skew(phi)
    R = torch.where(small, eye + Pk, R)
    V = torch.where(small, eye + 0.5 * Pk, V)
    t = (V @ rho[..., None])[..., 0]
    return R, t


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector of R (..., 3, 3); accurate away from theta = pi."""
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    s = torch.sqrt((w * w).sum(-1) + 1e-12) * 0.5
    c = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    theta = torch.atan2(s, c)
    factor = torch.where(s > 1e-4, theta / (2 * s.clamp(min=1e-4)), 0.5 + theta ** 2 / 12)
    return w * factor[..., None]


class Encoder(nn.Module):
    """ResNet-18 up to layer3; layer2 and layer3 fused at 1/8 resolution.

    With ctx=True, layer4 (1/32) is added through a zero-initialised projection,
    so a model warm-started from a ctx=False checkpoint computes the same function.
    """

    def __init__(self, feat: int = 128, pretrained: bool = True, ctx: bool = False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.fuse = nn.Sequential(nn.Conv2d(128 + 256, feat, 1), nn.GELU())
        self.ctx = ctx
        if ctx:
            self.layer4 = r.layer4
            self.ctx_proj = nn.Conv2d(512, feat, 1)
            nn.init.zeros_(self.ctx_proj.weight)
            nn.init.zeros_(self.ctx_proj.bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.stem(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        out = self.fuse(torch.cat([c2, F.interpolate(c3, size=c2.shape[-2:], mode="bilinear",
                                                     align_corners=False)], 1))
        if self.ctx:
            c4 = self.ctx_proj(self.layer4(c3))
            out = out + F.interpolate(c4, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return out


def _conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU())


class ResBlock(nn.Module):
    """Dilated residual block whose output conv starts at zero (identity at init)."""

    def __init__(self, ch: int, hidden: int, dilation: int):
        super().__init__()
        self.a = nn.Conv2d(ch, hidden, 3, padding=dilation, dilation=dilation)
        self.b = nn.Conv2d(hidden, ch, 3, padding=1)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(x)))


class EgoLite(nn.Module):
    def __init__(self, feat: int = 128, gn_iters: int = 3, use_ph: bool = True,
                 pretrained: bool = True, stride: int = 8, flow_iters: int = 12,
                 train_flow: bool = False, abs_depth: bool = False, flow_net: str = "small",
                 head_extra: int = 0, head_hidden: int = 256, ctx: bool = False,
                 temporal: str | None = None, anchor_t: bool = False, sem_head: int = 0,
                 iter_refine: int = 0, depth_prior: str | None = None, ba: bool = False):
        super().__init__()
        self.abs_depth = abs_depth
        self.ba = bool(ba)
        if self.ba:
            self.ba_conf = nn.Conv2d(128, 1, 1)
            nn.init.zeros_(self.ba_conf.weight)
            nn.init.zeros_(self.ba_conf.bias)
            self.ba_logw = nn.Parameter(torch.zeros(3))
            self.ba_logp = nn.Parameter(torch.tensor(0.0))
        self.depth_bias = nn.Parameter(torch.tensor(0.0))
        self.stride = stride
        self.gn_iters = gn_iters
        self.use_ph = use_ph
        self.flow_iters = flow_iters
        self.train_flow = train_flow
        self.temporal = temporal
        from torchvision.models.optical_flow import (Raft_Large_Weights, Raft_Small_Weights,
                                                     raft_large, raft_small)
        if flow_net == "large":
            self.flow = raft_large(weights=Raft_Large_Weights.DEFAULT if pretrained else None)
        else:
            self.flow = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None)
        if not train_flow:
            for p in self.flow.parameters():
                p.requires_grad_(False)
        self.enc = Encoder(feat, pretrained, ctx=ctx)
        in_ch = 3 * feat + 2 + (2 if use_ph else 0)
        self.pix = nn.Sequential(_conv(in_ch, 128), _conv(128, 128), _conv(128, 128))
        self.pix_extra = nn.Sequential(*[ResBlock(128, head_hidden, 2 ** (i % 4))
                                         for i in range(head_extra)])
        self.depth_prior = depth_prior
        if depth_prior == "midas":
            # frozen monocular depth prior replaces the shading-based pseudo-height inputs
            from .depth_prior import DepthPrior
            self.dprior = DepthPrior()
        self.iter_refine = int(iter_refine)
        if self.iter_refine:
            # residual -> motion layers -> pose, repeated (shared weights): evidence is the camera-flow
            # residual and the feature mismatch after aligning frame k+1 by camera motion alone
            self.r_red = nn.Conv2d(128, 16, 1)
            self.r_net = nn.Sequential(_conv(128 + 16 + 2 + 1 + 3 + 1, 96), _conv(96, 96))
            self.r_out = nn.Conv2d(96, 4, 1)
            nn.init.zeros_(self.r_out.weight)
            nn.init.zeros_(self.r_out.bias)
        self.sem_head = int(sem_head)
        if self.sem_head:
            # anatomy motion groups (anchor / large / medium / instrument): which tissue can move
            self.sem_out = nn.Conv2d(128, self.sem_head, 1)
        self.anchor_t = int(anchor_t)
        if self.anchor_t == 1:
            # per-pixel temporal anchor evidence: (pair features, scene flow, |scene flow|,
            # first-pass layer logits) over consecutive pairs -> correction of the layer logits
            self.a_red = nn.Conv2d(128, 32, 1)
            self.a_net = nn.Sequential(
                nn.Conv3d(32 + 2 + 1 + 3, 64, (5, 3, 3), padding=(2, 1, 1)), nn.GELU(),
                nn.Conv3d(64, 64, (5, 3, 3), padding=(4, 1, 1), dilation=(2, 1, 1)), nn.GELU())
            self.a_out = nn.Conv3d(64, 3, 1)
        elif self.anchor_t == 2:
            # long temporal field (~63 pairs, several breathing cycles); adds pose-free observed
            # flow and per-pixel window statistics (mean/std of observed flow, std of scene flow)
            self.a_red = nn.Conv2d(128, 32, 1)
            layers = [nn.Conv3d(32 + 2 + 1 + 3 + 2 + 6, 64, 3, padding=1), nn.GELU()]
            for dil in (2, 4, 8, 16):
                layers += [nn.Conv3d(64, 64, 3, padding=(dil, 1, 1), dilation=(dil, 1, 1)), nn.GELU()]
            self.a_net = nn.Sequential(*layers)
            self.a_out = nn.Conv3d(64, 3, 1)
        if self.anchor_t:
            nn.init.zeros_(self.a_out.weight)
            nn.init.zeros_(self.a_out.bias)
        if temporal:
            self.t_in = nn.Sequential(nn.Linear(256 + 12, 128), nn.GELU())
            self.t_gru = nn.GRU(128, 128, batch_first=True, bidirectional=(temporal == "bi"))
            self.t_out = nn.Linear(256 if temporal == "bi" else 128, 12)
            nn.init.zeros_(self.t_out.weight)
            with torch.no_grad():
                self.t_out.bias.zero_()
                self.t_out.bias[6:] = -8.0
        self.layer_out = nn.Conv2d(128, 3, 1)
        self.logvar_out = nn.Conv2d(128, 1, 1)
        self.depth_out = nn.Conv2d(128, 1, 1)
        self.pose_out = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 6))
        for m in (self.logvar_out, self.depth_out, self.pose_out[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.layer_out.bias.copy_(torch.tensor([1.0, 0.0, -2.0]))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.deform_logit = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------ grid
    def _grid(self, h, w, device, dtype):
        s = self.stride
        v, u = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) * s + (s - 1) / 2.0,
            torch.arange(w, device=device, dtype=dtype) * s + (s - 1) / 2.0,
            indexing="ij")
        return u, v

    @staticmethod
    def _warp(x, u, v, flow, H, W):
        gx = (2 * (u + flow[:, 0]) + 1) / W - 1
        gy = (2 * (v + flow[:, 1]) + 1) / H - 1
        grid = torch.stack([gx, gy], -1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # --------------------------------------------------------------- geometry
    @staticmethod
    def project(Z, u, v, K, R, t):
        """Camera flow of frame-t pixels under X' = R X + t. Shapes: Z (N,h,w)."""
        fx, fy, cx, cy = (K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2])
        fx, fy, cx, cy = (a[:, None, None] for a in (fx, fy, cx, cy))
        X = torch.stack([(u - cx) / fx * Z, (v - cy) / fy * Z, Z], -1)
        Xp = torch.einsum("nij,nhwj->nhwi", R, X) + t[:, None, None, :]
        zp = Xp[..., 2].clamp(min=1e-4)
        up = fx * Xp[..., 0] / zp + cx
        vp = fy * Xp[..., 1] / zp + cy
        return torch.stack([up - u, vp - v], 1), Xp

    def solve_pose(self, Z, u, v, K, f_obs, w, R, t, lam=1e-2, huber=1.5,
                   max_rot=0.2, max_trans=0.5, prior=None, iters=None, return_H=False):
        """Damped, Huber-reweighted Gauss-Newton on sum_i w_i |f_cam_i(R,t) - f_obs_i|^2.

        prior = (R_p, t_p, prec) adds sum_k prec_k * r_k^2 with r = (t_p - t, log(R_p R^T)).
        """
        fx, fy = K[:, 0, 0][:, None, None], K[:, 1, 1][:, None, None]
        H_flow = None
        for _ in range(self.gn_iters if iters is None else iters):
            f_cam, Xp = self.project(Z, u, v, K, R, t)
            e = (f_obs - f_cam).permute(0, 2, 3, 1)            # (N,h,w,2)
            en = e.norm(dim=-1).detach()
            w_rob = torch.where(en > huber, huber / en.clamp(min=1e-6), torch.ones_like(en)).detach()
            X, Y, Zp = Xp[..., 0], Xp[..., 1], Xp[..., 2].clamp(min=1e-4)
            dproj = torch.zeros(*Zp.shape, 2, 3, device=Z.device, dtype=Z.dtype)
            dproj[..., 0, 0] = fx / Zp
            dproj[..., 0, 2] = -fx * X / Zp ** 2
            dproj[..., 1, 1] = fy / Zp
            dproj[..., 1, 2] = -fy * Y / Zp ** 2
            dX = torch.cat([torch.eye(3, device=Z.device, dtype=Z.dtype).expand(*Zp.shape, 3, 3),
                            -skew(Xp)], -1)                     # (N,h,w,3,6)
            J = dproj @ dX                                      # (N,h,w,2,6)
            ww = (w * w_rob * (Xp[..., 2] > 1e-3).to(w.dtype))[..., None, None]
            Jt = J.transpose(-1, -2)
            H = (Jt @ (ww * J)).sum((1, 2))                     # (N,6,6)
            b = (Jt @ (ww * e[..., None])).sum((1, 2))[..., 0]  # (N,6)
            H_flow = H
            if prior is not None:
                Rp, tp, prec = prior
                r0 = torch.cat([tp - t, so3_log(Rp @ R.transpose(-1, -2))], -1)
                Jp = torch.zeros(*t.shape[:-1], 6, 6, device=Z.device, dtype=Z.dtype)
                Jp[..., :3, :3] = torch.eye(3, device=Z.device, dtype=Z.dtype)
                Jp[..., :3, 3:] = -skew(t)
                Jp[..., 3:, 3:] = torch.eye(3, device=Z.device, dtype=Z.dtype)
                JpT = Jp.transpose(-1, -2)
                H = H + JpT @ (prec[..., :, None] * Jp)
                b = b + (JpT @ (prec * r0)[..., None])[..., 0]
            diag = torch.diagonal(H, dim1=-2, dim2=-1)
            H = H + torch.diag_embed(lam * diag + 1e-6)
            delta = torch.linalg.solve(H, b[..., None])[..., 0]
            dn_t = delta[..., :3].norm(dim=-1, keepdim=True)
            dn_r = delta[..., 3:].norm(dim=-1, keepdim=True)
            delta = torch.cat([
                delta[..., :3] * torch.clamp(max_trans / dn_t.clamp(min=1e-9), max=1.0),
                delta[..., 3:] * torch.clamp(max_rot / dn_r.clamp(min=1e-9), max=1.0)], -1)
            dR, dt = se3_exp(delta)
            R = dR @ R
            t = (dR @ t[..., None])[..., 0] + dt
        f_cam, Xp = self.project(Z, u, v, K, R, t)
        if return_H:
            return R, t, f_cam, Xp, H_flow
        return R, t, f_cam, Xp

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_flow:
            self.flow.eval()
        return self

    def observed_flow(self, img0, img1):
        ctx = torch.enable_grad() if (self.train_flow and self.training) else torch.no_grad()
        with ctx, torch.autocast(device_type=img0.device.type, enabled=False):
            return self.flow(img0.float() * 2 - 1, img1.float() * 2 - 1,
                             num_flow_updates=self.flow_iters)[-1]

    # ---------------------------------------------------------------- forward
    def forward(self, img0, img1, K, ph0=None, ph1=None, flow=None, clip=None):
        """img*: (N,3,H,W) in [0,1]; K: (N,3,3) at input resolution;
        ph*: (N,1,H,W) pseudo-height in [0,1]; flow: optional (N,2,H,W) t->t+1 in pixels.
        clip=(B,P): the N pairs are B clips of P consecutive pairs; with a temporal
        model this runs the second, prior-regularised solve inside forward."""
        N, _, H, W = img0.shape
        s = self.stride
        if self.depth_prior:
            ph0, ph1 = self.dprior(img0).to(img0.dtype), self.dprior(img1).to(img0.dtype)
        if flow is None:
            flow = self.observed_flow(img0, img1)
        f0 = self.enc(img0)
        f1 = self.enc(img1)
        h, w = f0.shape[-2:]
        f_obs = F.avg_pool2d(flow, s, s)
        u, v = self._grid(h, w, img0.device, f_obs.dtype)
        f1w = self._warp(f1, u, v, f_obs.float(), H, W).to(f0.dtype)
        parts = [f0, f1w, (f0 - f1w).abs(), f_obs.to(f0.dtype) / s]
        if self.use_ph:
            if ph0 is None or ph1 is None:
                raise ValueError("pseudo-height maps are required when use_ph=True")
            p0 = F.avg_pool2d(ph0, s, s)
            p1w = self._warp(F.avg_pool2d(ph1, s, s), u, v, f_obs.float(), H, W)
            parts += [p0.to(f0.dtype), (p0 - p1w).to(f0.dtype)]
        x = self.pix_extra(self.pix(torch.cat(parts, 1)))
        logits = self.layer_out(x).float()
        logvar = self.logvar_out(x).float().clamp(-6, 6)[:, 0]
        r = self.depth_out(x).float()[:, 0]
        if self.use_ph:
            hc = p0.float()[:, 0]
            hc = hc - hc.mean((1, 2), keepdim=True)
            logz = self.beta * hc + r
        else:
            logz = r
        if self.abs_depth:
            logz = logz + self.depth_bias
        else:
            logz = logz - logz.flatten(1).median(1).values[:, None, None]
        Z = logz.exp()
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1).float()
        with torch.autocast(device_type=img0.device.type, enabled=False):
            xi0 = self.pose_out(pooled) * 0.05
            R0, t0 = se3_exp(xi0)

        with torch.autocast(device_type=img0.device.type, enabled=False):
            prob = torch.softmax(logits, 1)
            share = torch.sigmoid(self.deform_logit.float())
            w_pix = (prob[:, 0] + share * prob[:, 1]) * torch.exp(-logvar)
            ablate = getattr(self, "ablate", None)          # inference-time component ablations
            if ablate == "uniform":
                w_pix = torch.ones_like(w_pix)
            elif ablate == "nolayer":
                w_pix = torch.exp(-logvar)
            elif ablate == "novar":
                w_pix = prob[:, 0] + share * prob[:, 1]
            elif ablate == "staticonly":
                w_pix = prob[:, 0] * torch.exp(-logvar)
            R, t, f_cam, Xp, H = self.solve_pose(
                Z, u.float(), v.float(), K.float(), f_obs.float(), w_pix, R0.float(), t0.float(),
                return_H=True, iters=0 if ablate == "nogn" else None)
            zp = Xp[..., 2]
            rho = zp.flatten(1).median(1).values.clamp(min=1e-3)
        out = {
            "R": R, "t": t, "R0": R0, "t0": t0, "rho": rho, "zp": zp,
            "logits": logits, "logvar": logvar, "Z": Z, "logz": logz,
            "f_obs": f_obs.float(), "f_cam": f_cam, "f_scene": f_obs.float() - f_cam,
            "w": w_pix, "w_ba": (w_pix * torch.exp(self.ba_conf(x).float()[:, 0].clamp(-6, 6)) if self.ba else w_pix),
            "u": u, "v": v, "pooled": pooled, "H": H, "K": K.float(),
            "feat": x,
        }
        if self.sem_head:
            out["sem_logits"] = self.sem_out(x).float()
        if self.ba:
            out["w_ba"] = w_pix * torch.exp(self.ba_conf(x).float()[:, 0].clamp(-6, 6))
        if self.iter_refine:
            out = self.iterate_refine(out, f0, f1, img0.shape[2], img0.shape[3])
        if self.anchor_t and clip is not None:
            out = self.anchor_refine(out, *clip)
        if self.temporal and clip is not None:
            out = self.temporal_refine(out, *clip)
        return out

    def iterate_refine(self, out: dict, f0, f1, H: int, W: int) -> dict:
        """Alternate motion-layer correction and pose solving (DytanVO/DROID-style updates)."""
        s = self.stride
        u, v = out["u"], out["v"]
        R, t = out["R"].float(), out["t"].float()
        logits, logvar = out["logits"].float(), out["logvar"].float()
        R1, t1, logits1 = R, t, logits
        f_cam = out["f_cam"].float()
        x = out["feat"]
        for _ in range(self.iter_refine):
            res = (out["f_obs"] - f_cam) / s
            f1c = self._warp(f1, u, v, f_cam, H, W).to(f0.dtype)
            mism = self.r_red((f0 - f1c).abs()).float()
            inp = torch.cat([x.float(), mism, res, res.norm(dim=1, keepdim=True),
                             logits, logvar[:, None]], 1)
            d = self.r_out(self.r_net(inp.to(x.dtype))).float()
            logits = logits + d[:, :3]
            logvar = (logvar + d[:, 3]).clamp(-6, 6)
            with torch.autocast(device_type=f0.device.type, enabled=False):
                prob = torch.softmax(logits, 1)
                share = torch.sigmoid(self.deform_logit.float())
                w_pix = (prob[:, 0] + share * prob[:, 1]) * torch.exp(-logvar)
                R, t, f_cam, Xp = self.solve_pose(out["Z"].float(), u.float(), v.float(), out["K"],
                                                  out["f_obs"].float(), w_pix, R, t, iters=2)
        out = dict(out)
        out.update({"R1": R1, "t1": t1, "logits1": logits1, "R": R, "t": t, "logits": logits,
                    "logvar": logvar, "w": w_pix, "f_cam": f_cam, "zp": Xp[..., 2],
                    "f_scene": out["f_obs"] - f_cam})
        return out

    def anchor_refine(self, out: dict, B: int, P: int) -> dict:
        """Re-weight anchors from per-pixel evidence over consecutive pairs, then re-solve.

        Absolute stationarity is not identifiable from one pair when tissue motion is
        within the flow noise; over several pairs a static pixel keeps a scene flow
        consistent with zero world motion while moving tissue does not.
        """
        s = self.stride
        f_scene = (out["f_obs"] - out["f_cam"]).float()
        h, w = f_scene.shape[-2:]
        parts = [self.a_red(out["feat"]).float(), f_scene / s,
                 f_scene.norm(dim=1, keepdim=True) / s, out["logits"].float()]
        if self.anchor_t == 2:
            f_obs = out["f_obs"].float() / s
            fo, fs = f_obs.view(B, P, 2, h, w), (f_scene / s).view(B, P, 2, h, w)
            stats = torch.cat([fo.mean(1), fo.std(1, unbiased=False), fs.std(1, unbiased=False)], 1)
            parts += [f_obs, stats[:, None].expand(B, P, 6, h, w).reshape(B * P, 6, h, w)]
        x = torch.cat(parts, 1)
        x = x.view(B, P, -1, h, w).transpose(1, 2)                 # (B, C, P, h, w)
        with torch.autocast(device_type=x.device.type, enabled=False):
            dl = self.a_out(self.a_net(x)).transpose(1, 2).reshape(B * P, 3, h, w)
            logits = out["logits"].float() + dl
            prob = torch.softmax(logits, 1)
            share = torch.sigmoid(self.deform_logit.float())
            w_pix = (prob[:, 0] + share * prob[:, 1]) * torch.exp(-out["logvar"].float())
            R, t, f_cam, Xp = self.solve_pose(
                out["Z"].float(), out["u"].float(), out["v"].float(), out["K"].float(),
                out["f_obs"].float(), w_pix, out["R"].float(), out["t"].float())
        out = dict(out)
        out.update({"R1": out["R"], "t1": out["t"], "logits1": out["logits"], "logits": logits,
                    "R": R, "t": t, "f_cam": f_cam, "zp": Xp[..., 2], "w": w_pix,
                    "f_scene": out["f_obs"] - f_cam})
        return out

    def temporal_refine(self, out: dict, B: int, P: int) -> dict:
        """Second solve with a per-pair pose prior from a GRU over the clip.

        The GRU sees pooled pair features, the first-pass pose, and the log
        information of the first-pass flow Hessian; it predicts a pose correction
        and a per-axis prior precision relative to the mean flow information.
        """
        with torch.autocast(device_type=out["R"].device.type, enabled=False):
            R1, t1, H = out["R"], out["t"], out["H"]
            hd = torch.diagonal(H, dim1=-2, dim2=-1).clamp(min=1e-6)
            href = hd.mean(-1, keepdim=True).detach()
            feat = torch.cat([out["pooled"], t1 * 20.0, so3_log(R1) * 20.0,
                              torch.log(hd / href)], -1)
            y, _ = self.t_gru(self.t_in(feat).view(B, P, -1))
            y = self.t_out(y).reshape(B * P, 12)
            d = y[:, :6] * 0.01
            Rd, _ = se3_exp(torch.cat([torch.zeros_like(d[:, :3]), d[:, 3:]], -1))
            Rp, tp = Rd @ R1, t1 + d[:, :3]
            prec = torch.exp(y[:, 6:].clamp(-12, 8)) * href
            R, t, f_cam, Xp = self.solve_pose(
                out["Z"], out["u"].float(), out["v"].float(), out["K"], out["f_obs"], out["w"],
                Rp, tp, prior=(Rp, tp, prec))
        out = dict(out)
        out.update({"R1": R1, "t1": t1, "R": R, "t": t, "f_cam": f_cam, "zp": Xp[..., 2],
                    "f_scene": out["f_obs"] - f_cam, "prec": prec, "Rp": Rp, "tp": tp})
        return out


NEW_MODULE_PREFIXES = ("enc.layer4.", "enc.ctx_proj.", "pix_extra.", "t_in.", "t_gru.", "t_out.",
                       "a_red.", "a_net.", "a_out.", "sem_out.", "r_red.", "r_net.", "r_out.", "dprior.",
                       "depth_bias", "ba_")


def build_from_args(cfg: dict, pretrained: bool = False) -> EgoLite:
    """EgoLite from a training-args dict (checkpoint 'args' or argparse vars)."""
    return EgoLite(use_ph=not cfg.get("no_ph", False), gn_iters=cfg.get("gn_iters", 3),
                   pretrained=pretrained, abs_depth=cfg.get("abs_depth", False),
                   flow_net=cfg.get("flow_net", "small"), head_extra=cfg.get("head_extra", 0),
                   ctx=cfg.get("ctx", False), temporal=cfg.get("temporal") or None,
                   train_flow=cfg.get("train_flow", False), anchor_t=cfg.get("anchor_t", False),
                   sem_head=cfg.get("sem_head", 0), iter_refine=cfg.get("iter_refine", 0),
                   depth_prior=cfg.get("depth_prior") or None, ba=bool(cfg.get("ba_skips")))


def pixel_log_ratio(out: dict, Z_next: torch.Tensor, stride: int = 8):
    """Per-pixel log(depth of frame-k point in frame k+1 / frame k+1's own depth there).

    out: forward output of pairs (k, k+1); Z_next: (N,h,w) normalised depth of
    frame k+1 from pairs (k+1, k+2). Returns (log_ratio, inside) on the 1/8 grid.
    """
    h, w = Z_next.shape[-2:]
    up = out["u"] + out["f_cam"][:, 0]
    vp = out["v"] + out["f_cam"][:, 1]
    W, H = w * stride, h * stride
    gx = (2 * up + 1) / W - 1
    gy = (2 * vp + 1) / H - 1
    grid = torch.stack([gx, gy], -1)
    zs = F.grid_sample(Z_next[:, None], grid, mode="bilinear", padding_mode="border",
                       align_corners=False)[:, 0]
    inside = (gx.abs() < 1) & (gy.abs() < 1) & (out["zp"] > 1e-3)
    ratio = torch.log(out["zp"].clamp(min=1e-3)) - torch.log(zs.clamp(min=1e-3))
    return ratio, inside


def pixel_rho(out: dict, Z_next: torch.Tensor, stride: int = 8) -> torch.Tensor:
    """Scale ratio between consecutive pairs from pixel correspondences.

    Frame-k points are moved into frame k+1 by the solved pose; their depth
    there, divided by frame k+1's own depth at the landing pixel, gives frame
    k+1's unit in frame-k units. The median runs over likely-stationary pixels.
    """
    ratio, inside = pixel_log_ratio(out, Z_next, stride)
    prob = torch.softmax(out["logits"], 1)[:, 0]
    rows = []
    for r, m, p, fb in zip(ratio, inside, prob, out["rho"]):
        sel = m & (p > 0.3)
        if sel.sum() < 32:
            sel = m
        rows.append(r[sel].median().exp() if sel.sum() >= 16 else fb)
    return torch.stack(rows).clamp(min=1e-3)


def chain(R: torch.Tensor, t: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Compose relative poses X_{k+1} = R_k X_k + t_k into camera-to-world poses.

    Pair k is in units of the median depth of frame k; rho_k converts frame k+1
    units into frame k units. Returns (S,4,4) with T_wc(0) = I, in frame-0 units.
    """
    S = R.shape[0] + 1
    out = [torch.eye(4, device=R.device, dtype=R.dtype)]
    c = torch.ones((), device=R.device, dtype=R.dtype)
    for k in range(S - 1):
        T = torch.eye(4, device=R.device, dtype=R.dtype)
        T[:3, :3] = R[k]
        T[:3, 3] = t[k] * c
        out.append(out[-1] @ torch.linalg.inv(T))
        c = c * rho[k]
    return torch.stack(out)
