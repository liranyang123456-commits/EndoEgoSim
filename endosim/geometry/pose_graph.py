"""SE(3) 位姿图优化（g2o 风格 Gauss-Newton / LM，Huber 鲁棒核）。

用途：MD-VGGT-R 的重叠短窗产生冗余相对位姿边（skip 1/2/4，跨窗口多观测）。
现有 compose_edge_graph 是链式投票，只利用局部冗余；本模块把所有边作为软约束
做全局优化，期望在长序列上抑制链式漂移累积。

约定（与 endosim.geometry.se3 一致）：
- 节点 X_i = T_wc(i)，4x4 c2w；第 0 帧固定为 I（gauge）。
- 边 (i, j) 测量 Z_ij = X_i^{-1} X_j（relative 约定）。
- 误差 e = se3_log(Z^{-1} X_i^{-1} X_j)，xi = [w(3), v(3)] 旋转在前。
- 右扰动更新 X_i <- X_i Exp(delta_i)。
- 雅可比（小残差近似，g2o EdgeSE3 常用形式）：
    J_j = I_6
    J_i = -Adj(X_j^{-1} X_i)
  其中 Adj(T) = [[R, 0], [-hat(t) R, R]]（旋转块在前）。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .se3 import compose, hat, inverse, relative, se3_exp, se3_log


def adjoint_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) 伴随，xi=[w,v] 旋转在前：Adj(T) = [[R,0],[hat(t)R, R]]。

    自洽性：T Exp(xi) T^{-1} = Exp(Adj(T) xi)（selftest 数值验证）。
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    adj = np.zeros((6, 6), dtype=np.float64)
    adj[:3, :3] = R
    adj[3:, :3] = hat(t) @ R
    adj[3:, 3:] = R
    return adj


def _edge_residual(X: np.ndarray, i: int, j: int, Z: np.ndarray) -> np.ndarray:
    """e = se3_log(Z^{-1} X_i^{-1} X_j)。"""
    return se3_log(compose(inverse(Z), relative(X[i], X[j])))


def optimize_pose_graph(
    n: int,
    edges: dict,
    init: np.ndarray | None = None,
    weights: dict | None = None,
    huber_delta: float | None = None,
    iters: int = 20,
    damping: float = 1e-6,
    fix_first: bool = True,
    tol: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """位姿图 GN/LM 优化。

    n: 节点数（帧数）。
    edges: {(i,j): Z_ij}，Z_ij = X_i^{-1} X_j（relative 约定）。
    weights: 可选 {(i,j): w} 边权重（乘在信息矩阵上）。
    huber_delta: None 则自适应（首轮残差中位数的 3 倍）。
    init: (n,4,4) 初值；None 时用链式组合兜底。
    返回 (poses, stats)。
    """
    edge_list = sorted((int(i), int(j)) for (i, j) in edges.keys())
    if not edge_list:
        return (np.repeat(np.eye(4)[None], n, axis=0), {"n_edges": 0, "converged": False})
    Zs = {e: np.asarray(edges[e], dtype=np.float64) for e in edge_list}

    if init is None:
        # 链式兜底初值
        X = np.repeat(np.eye(4)[None], n, axis=0)
        for t in range(1, n):
            cands = [
                (s, e) for (s, e) in edge_list if e == t
            ]
            if cands:
                s, e = min(cands, key=lambda se: se[1] - se[0])
                X[t] = compose(X[s], Zs[(s, e)])
    else:
        X = np.array(init, dtype=np.float64, copy=True)

    var_nodes = list(range(1 if fix_first else 0, n))
    if not var_nodes:
        return X, {"n_edges": len(edge_list), "converged": True, "iters": 0}
    vidx = {node: k for k, node in enumerate(var_nodes)}
    nvar = 6 * len(var_nodes)

    def residuals_and_jac(Xcur):
        rows, cols, vals = [], [], []
        res = np.zeros(6 * len(edge_list), dtype=np.float64)
        for k, (i, j) in enumerate(edge_list):
            Z = Zs[(i, j)]
            e = _edge_residual(Xcur, i, j, Z)
            res[6 * k: 6 * k + 6] = e
            Jj = np.eye(6)
            Ji = -adjoint_se3(relative(Xcur[j], Xcur[i]))
            for (node, J) in ((i, Ji), (j, Jj)):
                if node in vidx:
                    base = 6 * vidx[node]
                    for c in range(6):
                        rows.extend(range(6 * k, 6 * k + 6))
                        cols.extend([base + c] * 6)
                        vals.extend(J[:, c])
        Jmat = sp.csr_matrix(
            (vals, (rows, cols)), shape=(6 * len(edge_list), nvar)
        )
        return res, Jmat

    # 自适应 Huber 阈值
    res0, _ = residuals_and_jac(X)
    en = np.linalg.norm(res0.reshape(-1, 6), axis=1)
    if huber_delta is None:
        huber_delta = max(3.0 * float(np.median(en)), 1e-6)

    converged = False
    last_cost = None
    for it in range(iters):
        res, Jmat = residuals_and_jac(X)
        en = np.linalg.norm(res.reshape(-1, 6), axis=1)
        # Huber 权重（对每条边的 6 维残差整体缩放）
        w = np.ones_like(en)
        big = en > huber_delta
        w[big] = huber_delta / np.maximum(en[big], 1e-12)
        if weights:
            for k, e in enumerate(edge_list):
                w[k] *= float(weights.get(e, 1.0))
        W = sp.diags(np.repeat(w, 6))
        JW = Jmat.T @ W
        H = JW @ Jmat + damping * sp.eye(nvar)
        b = -JW @ res
        delta = spla.spsolve(H, b)
        if not np.isfinite(delta).all():
            break
        Xnew = np.array(X, copy=True)
        for node in var_nodes:
            d = delta[6 * vidx[node]: 6 * vidx[node] + 6]
            Xnew[node] = compose(Xnew[node], se3_exp(d))
        cost = float(np.sum(np.minimum(en ** 2, huber_delta * (2 * en - huber_delta))))
        X = Xnew
        step = float(np.linalg.norm(delta))
        if last_cost is not None and abs(last_cost - cost) < tol * max(1.0, last_cost):
            converged = True
            break
        if step < 1e-10:
            converged = True
            break
        last_cost = cost

    res_f, _ = residuals_and_jac(X)
    enf = np.linalg.norm(res_f.reshape(-1, 6), axis=1)
    stats = {
        "n_edges": len(edge_list),
        "iters": it + 1,
        "converged": converged,
        "huber_delta": float(huber_delta),
        "residual_median": float(np.median(enf)),
        "residual_p95": float(np.percentile(enf, 95)),
        "n_hubered": int((enf > huber_delta).sum()),
    }
    return X, stats


def selftest() -> dict:
    """合成图单测：噪声 + 冗余边 + 离群边，图优化应优于链式投票。"""
    rng = np.random.default_rng(7)
    n = 24
    # 真值：前进 + 微旋转
    gt = [np.eye(4)]
    for t in range(1, n):
        step = np.eye(4)
        step[:3, 3] = [0.0, 0.0, 2.0]
        step[:3, :3] = se3_exp(np.array([0.01, 0.005, 0.0, 0, 0, 0]))[:3, :3]
        gt.append(compose(gt[-1], step))
    gt = np.stack(gt)

    def noisy_edge(i, j, sigma_t=0.05, sigma_r=0.005, outlier=False):
        Z = relative(gt[i], gt[j])
        noise = np.concatenate([
            rng.normal(0, sigma_r, 3), rng.normal(0, sigma_t, 3)
        ])
        if outlier:
            noise += np.concatenate([
                rng.normal(0, 0.3, 3), rng.normal(0, 3.0, 3)
            ])
        return compose(Z, se3_exp(noise))

    edges = {}
    for t in range(n - 1):
        edges[(t, t + 1)] = noisy_edge(t, t + 1)
    for skip in (2, 4):
        for t in range(n - skip):
            edges[(t, t + skip)] = noisy_edge(t, t + skip, sigma_t=0.08)
    # 10% 离群边
    for k in rng.choice(len(edges), size=max(1, len(edges) // 10), replace=False):
        i, j = sorted(edges.keys())[k]
        edges[(i, j)] = noisy_edge(i, j, outlier=True)

    # 链式（仅 skip-1 从 0 出发）
    chain = [np.eye(4)]
    for t in range(1, n):
        chain.append(compose(chain[-1], edges[(t - 1, t)]))
    chain = np.stack(chain)

    opt, stats = optimize_pose_graph(n, edges, init=chain)

    def ate(traj):
        # 简单 Sim(3)-free ATE：首帧对齐后的中心 RMSE（同一 gauge）
        c = traj[:, :3, 3] - traj[0, :3, 3][None]
        g = gt[:, :3, 3] - gt[0, :3, 3][None]
        s = float(np.sum(c * g) / max(np.sum(c * c), 1e-12))
        return float(np.sqrt(np.mean(np.sum((s * c - g) ** 2, axis=1))))

    out = {
        "ate_chain": ate(chain),
        "ate_graph": ate(opt),
        "stats": stats,
    }
    assert out["ate_graph"] < out["ate_chain"], (
        f"graph {out['ate_graph']:.4f} not better than chain {out['ate_chain']:.4f}"
    )
    # 伴随自洽性：解析 Adj 与数值 Adj（扰动法）一致
    T = gt[5]
    adj_num = np.zeros((6, 6))
    for c in range(6):
        d = np.zeros(6)
        d[c] = 1e-8
        M = compose(compose(T, se3_exp(d)), inverse(T))
        adj_num[:, c] = se3_log(M) / 1e-8
    assert np.allclose(adj_num, adjoint_se3(T), atol=1e-5), "adjoint mismatch"
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(selftest(), indent=1, default=float))
