"""
Phase 8: 状态空间可视化
=======================
PCA / t-SNE 降维展示 Reservoir 状态轨迹。
"""
import os, pickle, warnings, signal
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(os.path.join(OUTPUT_DIR, "state_space"), exist_ok=True)

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

print("加载数据...")
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"), allow_pickle=True)
neuron_names = data["neuron_names"].tolist()
adj_matrix = data["adj_matrix"].astype(np.float64)
N = len(neuron_names)
from model import ConnectomeReservoir

def scale_rho(W, target=0.9):
    e = np.linalg.eigvals(W)
    r = np.max(np.abs(e))
    return W * (target / max(r, 1e-12)), r

W_scaled, _ = scale_rho(adj_matrix, 0.9)

# ════════════════════════════════════════════════
# 1. PCA: 不同输入条件下的状态轨迹
# ════════════════════════════════════════════════
print("\n── [1/3] PCA 状态空间可视化 ──")

n_step = 500
inputs = {
    "Sine wave": np.sin(np.linspace(0, 8*np.pi, n_step)).reshape(-1, 1),
    "Random noise": np.random.randn(n_step, 1) * 0.5,
    "Step pulse": np.zeros((n_step, 1)),
}
# Step: first 100 steps = 1, rest = 0
inputs["Step pulse"][:100, 0] = 1.0

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax_idx, (cond_name, U) in enumerate(inputs.items()):
    model = ConnectomeReservoir(W_scaled, 1, 1, input_scaling=1.0)
    X = model.collect(U, discard=0)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    ax = axes[ax_idx]
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=range(len(X_2d)),
                    cmap="viridis", s=10, alpha=0.7)
    ax.scatter(X_2d[0, 0], X_2d[0, 1], c="red", s=80, marker="*", label="Start")
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1], c="blue", s=80, marker="s", label="End")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax.set_title(f"Input: {cond_name}")
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Time step")

plt.suptitle("C. elegans Reservoir State Space (PCA)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "state_space", "pca_trajectories.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  PCA 图已保存")

# ════════════════════════════════════════════════
# 2. t-SNE: 不同输入类别的状态分布
# ════════════════════════════════════════════════
print("\n── [2/3] t-SNE 可视化 ──")

n_step_tsne = 300
categories = {
    "Left": np.column_stack([np.ones(n_step_tsne), np.zeros(n_step_tsne)]),
    "Right": np.column_stack([np.zeros(n_step_tsne), np.ones(n_step_tsne)]),
    "None": np.zeros((n_step_tsne, 2)),
}

all_states = []
all_labels = []
model = ConnectomeReservoir(W_scaled, 2, 1, input_scaling=1.5)

for label, U in categories.items():
    model.reset()
    states = model.collect(U, discard=0)
    all_states.append(states)
    all_labels.extend([label] * len(states))

all_states = np.vstack(all_states)

try:
    with time_limit(300, "t-SNE"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(all_states)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_ = {"Left": "#E74C3C", "Right": "#3498DB", "None": "#95A5A6"}
    for label in ["Left", "Right", "None"]:
        mask = [l == label for l in all_labels]
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=colors_[label], label=label, s=8, alpha=0.6)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE: Reservoir States under Different Input Conditions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "state_space", "tsne_states.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  t-SNE 图已保存")
except Exception as e:
    print(f"  ⚠ t-SNE skipped: {e}")

# ════════════════════════════════════════════════
# 3. 不同谱半径下的状态轨迹对比
# ════════════════════════════════════════════════
print("\n── [3/3] 不同谱半径状态对比 ──")

n_step_rho = 300
U_rho = np.sin(np.linspace(0, 6*np.pi, n_step_rho)).reshape(-1, 1)
rhos = [0.5, 0.9, 1.05]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, rho in enumerate(rhos):
    W_rho, _ = scale_rho(adj_matrix, rho)
    model = ConnectomeReservoir(W_rho, 1, 1, input_scaling=1.0)
    X = model.collect(U_rho, discard=0)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    ax = axes[idx]
    ax.plot(X_2d[:, 0], X_2d[:, 1], lw=1, alpha=0.7)
    ax.scatter(X_2d[0, 0], X_2d[0, 1], c="red", s=80, marker="*", label="Start")
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1], c="blue", s=80, marker="s", label="End")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Spectral Radius ρ={rho}")
    ax.legend(fontsize=9)

plt.suptitle("State Space Trajectories under Different Spectral Radii", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "state_space", "rho_trajectories.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  谱半径对比图已保存")

print("\n✅ Phase 8 完成！图片在 output/state_space/")
