"""
Phase 6-7: 对比实验 + 损伤模拟
=================================
对比：真实连接组 vs 度保持随机化 vs ER vs 小世界网络
损伤：按介数中心性/随机移除节点，观察性能下降
"""
import os, pickle, warnings, signal
from contextlib import contextmanager
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
for d in ["comparison", "lesion"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

class TimeoutError_(Exception): pass
@contextmanager
def time_limit(seconds, label="op"):
    def handler(signum, frame):
        raise TimeoutError_(f"⏰ {label} timeout ({seconds}s)")
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        yield
    except TimeoutError_ as e:
        print(f"  {e}")
        raise
    finally:
        signal.alarm(0)

# ── 加载 ──
print("加载连接组数据...")
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"), allow_pickle=True)
neuron_names = data["neuron_names"].tolist()
neuron_meta = pickle.loads(data["neuron_meta"].item())
adj_real = data["adj_matrix"].astype(np.float64)
N = len(neuron_names)
with open(os.path.join(OUTPUT_DIR, "connectome_graph.pkl"), "rb") as f:
    G = pickle.load(f)
print(f"已加载 {N}x{N} 连接矩阵")

def scale_rho(W, target=0.9):
    e = np.linalg.eigvals(W)
    r = np.max(np.abs(e))
    return W * (target / max(r, 1e-12)), r

# ════════════════════════════════════════════════
# Null 模型生成
# ════════════════════════════════════════════════
def make_null_models():
    print("\n── 生成 Null 模型 ──")
    models = {"Real Connectome": adj_real}
    G_undirected = G.to_undirected()

    # 1. 度保持随机化
    print("  度保持随机化网络...")
    with time_limit(120, "random_reference"):
        GR = nx.random_reference(G_undirected, seed=42)
        adj_deg = nx.to_numpy_array(GR, nodelist=sorted(GR.nodes()))
        # 确保大小一致
        names_sub = sorted(GR.nodes())
        idx = [neuron_names.index(n) for n in names_sub if n in neuron_names]
        adj_full = np.zeros_like(adj_real)
        for i, ni in enumerate(idx):
            for j, nj in enumerate(idx):
                adj_full[ni, nj] = adj_deg[i, j]
        models["Degree-preserving Random"] = adj_full

    # 2. ER 随机图
    print("  Erdős-Rényi 随机图...")
    p = 2 * G.number_of_edges() / (N * (N - 1))
    adj_er = np.random.RandomState(42).rand(N, N) < p
    adj_er = adj_er.astype(np.float64)
    adj_er, _ = scale_rho(adj_er)
    np.fill_diagonal(adj_er, 0)
    models["Erdős-Rényi"] = adj_er

    # 3. 小世界网络 (Watts-Strogatz)
    print("  Watts-Strogatz 小世界网络...")
    k = max(3, int(2 * G.number_of_edges() / N))
    WS = nx.watts_strogatz_graph(N, k, 0.3, seed=42)
    adj_ws = nx.to_numpy_array(WS)
    adj_ws, _ = scale_rho(adj_ws)
    models["Watts-Strogatz"] = adj_ws

    print(f"  已生成 {len(models)} 个网络: {list(models.keys())}")
    return models


# ════════════════════════════════════════════════
# 基准：Lorenz MSE
# ════════════════════════════════════════════════
def eval_lorenz_mse(W, n_res=150, predict_step=5):
    def gen_lorenz(n=4000):
        x,y,z = 1.0,1.0,1.0; traj = np.zeros((n,3))
        for i in range(n):
            dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-8/3*z
            x+=dx*0.02; y+=dy*0.02; z+=dz*0.02; traj[i]=[x,y,z]
        return traj

    n_step, n_discard = 4000, 500
    data = gen_lorenz(n_step)
    mn, sd = data.mean(0), data.std(0)
    dn = (data - mn) / sd
    U, Y = dn[:-predict_step], dn[predict_step:]

    n_use = min(n_res, W.shape[0])
    idx = np.random.RandomState(42).choice(W.shape[0], n_use, replace=False)
    W_sub = W[np.ix_(idx, idx)]
    W_sub, _ = scale_rho(W_sub, 0.9)

    from model import ConnectomeReservoir
    model = ConnectomeReservoir(W_sub, 3, 3, input_scaling=2.0)
    model.fit_ridge(U, Y, discard=n_discard, alpha=1e-6)
    Yp = model.predict(U, discard=n_discard)
    Yt = Y[n_discard:]
    m = min(len(Yp), len(Yt))
    return float(np.mean((Yp[:m] - Yt[:m]) ** 2))
# ════════════════════════════════════════════════
def run_comparison():
    print("\n" + "=" * 60)
    print("Phase 6: 对比实验 (Lorenz MSE)")
    print("=" * 60)

    models = make_null_models()
    results = {}

    for name, W in tqdm(models.items(), desc="Evaluating models"):
        try:
            with time_limit(120, name):
                mse = eval_lorenz_mse(W, n_res=150, predict_step=5)
                results[name] = mse
                print(f"\n  {name}: MSE={mse:.2e}")
        except Exception as e:
            print(f"\n  ⚠ {name}: {e}")
            results[name] = None

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    vals = [results[n] if results[n] is not None else 0 for n in names]
    colors_ = ["#2E86AB", "#A23B72", "#F18F01", "#70A288"]
    bars = ax.bar(names, vals, color=colors_[:len(names)], alpha=0.8, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05,
                f"{v:.2e}", ha="center", fontsize=10)
    ax.set_ylabel("Lorenz MSE (lower is better)")
    ax.set_title("Connectome vs Null Models: Lorenz Prediction")
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison", "null_model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  结果已保存至 output/comparison/")
    return results


# ════════════════════════════════════════════════
# Phase 7: 损伤模拟
# ════════════════════════════════════════════════
def run_lesion():
    print("\n" + "=" * 60)
    print("Phase 7: 损伤模拟")
    print("=" * 60)

    from model import ConnectomeReservoir

    def gen_lorenz(n=4000):
        x,y,z = 1.0,1.0,1.0; traj = np.zeros((n,3))
        for i in range(n):
            dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-8/3*z
            x+=dx*0.02; y+=dy*0.02; z+=dz*0.02; traj[i]=[x,y,z]
        return traj

    n_step, n_discard = 4000, 500
    data = gen_lorenz(n_step)
    mn, sd = data.mean(0), data.std(0)
    dn = (data - mn) / sd
    U, Y = dn[:-5], dn[5:]

    # 计算介数中心性排序
    print("计算节点重要性排序...")
    G_undirected = G.to_undirected()
    try:
        with time_limit(120, "betweenness"):
            bc = nx.betweenness_centrality(G_undirected, k=50, seed=42)
    except:
        bc = nx.degree_centrality(G_undirected)
    nodes_sorted = [n for n, _ in sorted(bc.items(), key=lambda x: -x[1])]

    removal_ratios = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    mse_random = []
    mse_hub = []
    seed = np.random.RandomState(42)

    for ratio in tqdm(removal_ratios, desc="Lesion ratio"):
        n_remove = max(1, int(N * ratio))

        # 随机损伤（3 次平均）
        mses_r = []
        for rep in range(3):
            W_lesion = adj_real.copy()
            remove_idx = seed.choice(N, n_remove, replace=False)
            W_lesion[remove_idx, :] = 0
            W_lesion[:, remove_idx] = 0
            W_lesion, _ = scale_rho(W_lesion, 0.9)
            # 降维
            keep = np.setdiff1d(np.arange(N), remove_idx)
            W_sub = W_lesion[np.ix_(keep, keep)]
            if W_sub.shape[0] < 10:
                mses_r.append(10.0)
                continue
            model = ConnectomeReservoir(W_sub, 3, 3, input_scaling=2.0)
            model.fit_ridge(U, Y, discard=n_discard, alpha=1e-6)
            Yp = model.predict(U, discard=n_discard)
            Yt = Y[n_discard:]
            m = min(len(Yp), len(Yt))
            mses_r.append(float(np.mean((Yp[:m] - Yt[:m]) ** 2)))
        mse_random.append(np.mean(mses_r))

        # 按 hub 损伤
        hub_names = nodes_sorted[:n_remove]
        hub_idx = [neuron_names.index(n) for n in hub_names if n in neuron_names]
        W_lesion = adj_real.copy()
        W_lesion[hub_idx, :] = 0
        W_lesion[:, hub_idx] = 0
        W_lesion, _ = scale_rho(W_lesion, 0.9)
        keep = np.setdiff1d(np.arange(N), hub_idx)
        W_sub = W_lesion[np.ix_(keep, keep)]
        if W_sub.shape[0] >= 10:
            model = ConnectomeReservoir(W_sub, 3, 3, input_scaling=2.0)
            model.fit_ridge(U, Y, discard=n_discard, alpha=1e-6)
            Yp = model.predict(U, discard=n_discard)
            Yt = Y[n_discard:]
            m = min(len(Yp), len(Yt))
            mse_hub.append(float(np.mean((Yp[:m] - Yt[:m]) ** 2)))
        else:
            mse_hub.append(10.0)

    # 归一化到基线
    baseline = mse_random[0]
    mse_random_n = [m / baseline for m in mse_random]
    mse_hub_n = [m / baseline for m in mse_hub]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r*100 for r in removal_ratios], mse_random_n, 'o-', lw=2, ms=8,
            color="steelblue", label="Random lesion")
    ax.plot([r*100 for r in removal_ratios], mse_hub_n, 's-', lw=2, ms=8,
            color="coral", label="Hub-targeted lesion")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("Neurons Removed (%)"); ax.set_ylabel("Normalized MSE")
    ax.set_title("Lesion Analysis: Effect of Neuron Removal on Lorenz Prediction")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lesion", "lesion_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  结果已保存至 output/lesion/")
    return removal_ratios, mse_random_n, mse_hub_n


# ════════════════════════════════════════════════
if __name__ == "__main__":
    tasks = [
        (600, "Comparison", run_comparison),
        (600, "Lesion", run_lesion),
    ]
    for t, name, fn in tasks:
        try:
            with time_limit(t, name):
                fn()
        except Exception as e:
            print(f"⚠ {name}: {e}")
    print("\n✅ Phase 6-7 完成！")
