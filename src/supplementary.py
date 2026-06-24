"""
Supplementary Analysis: 补全参考文献中的关键分析
=================================================
参照 Casal 2018 和 Galella 2018，新增：
1. 网络剪枝（Pruning）：提取 reservoir 核心
2. E/I 平衡：应用 30% 抑制性突触
3. 脉冲输入响应（规则/不规则间隔）
4. NARMA 任务（复杂输入记忆测试）
5. 可靠性分析（intra-vs-inter-series 相关性）
6. 距离-信息传播分析
"""
import os, pickle, warnings, signal
from contextlib import contextmanager
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
for d in ["supplement"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

class TimeoutError_(Exception): pass
@contextmanager
def time_limit(seconds, label="op"):
    def handler(signum, frame): raise TimeoutError_(f"⏰ {label}")
    try:
        signal.signal(signal.SIGALRM, handler); signal.alarm(seconds); yield
    except TimeoutError_ as e: print(f"  {e}"); raise
    finally: signal.alarm(0)

print("加载连接组数据...")
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"), allow_pickle=True)
neuron_names = data["neuron_names"].tolist()
neuron_meta = pickle.loads(data["neuron_meta"].item())
adj = data["adj_matrix"].astype(np.float64)
N = len(neuron_names)
from model import ConnectomeReservoir
Gu = nx.from_numpy_array(adj > 0.01).to_undirected()
print(f"已加载 {N}x{N} 连接矩阵")

# ════════════════════════════════════════════════
# 1. Reservoir 剪枝（Pruning）
# ════════════════════════════════════════════════
print("\n── [1/6] Reservoir Pruning ──")

def prune_network(W, neuron_names_):
    """迭代移除零入度/零出度节点，返回剪枝后的矩阵和层级信息"""
    G_dir = nx.from_numpy_array(W > 0.01, create_using=nx.DiGraph)
    mapping = {i: neuron_names_[i] for i in range(len(neuron_names_))}
    G_dir = nx.relabel_nodes(G_dir, mapping)

    input_neurons = set()   # 零入度 → 输入层
    output_neurons = set()  # 零出度 → 输出层
    reservoir_neurons = set(G_dir.nodes())

    changed = True
    while changed:
        changed = False
        # 找零入度（排除已有 input 的节点）
        zero_in = {n for n in reservoir_neurons if G_dir.in_degree(n) == 0}
        # 找零出度
        zero_out = {n for n in reservoir_neurons if G_dir.out_degree(n) == 0}

        if zero_in:
            input_neurons.update(zero_in)
            reservoir_neurons -= zero_in
            G_dir.remove_nodes_from(zero_in)
            changed = True

        if zero_out:
            output_neurons.update(zero_out)
            reservoir_neurons -= zero_out
            G_dir.remove_nodes_from(zero_out)
            changed = True

    print(f"  输入层: {len(input_neurons)} 神经元")
    print(f"  Reservoir: {len(reservoir_neurons)} 神经元")
    print(f"  输出层: {len(output_neurons)} 神经元")
    # 分类输出层
    innervations = [n for n in output_neurons if "NMJ" in n or any(
        m in n for m in ["MUSCLE", "muscle"])]
    print(f"  其中肌肉接头: {len(innervations)}")
    return input_neurons, reservoir_neurons, output_neurons

in_n, res_n, out_n = prune_network(adj, neuron_names)

# 可视化剪枝后的网络结构
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
categories = [("Input Layer", in_n, "#E74C3C"),
              ("Reservoir Core", res_n, "#2E86AB"),
              ("Output Layer", out_n, "#A23B72")]
for ax, (title, nodes, color) in zip(axes, categories):
    types = Counter(neuron_meta.get(n, {}).get("type", "unknown") for n in nodes)
    labels, vals = zip(*sorted(types.items(), key=lambda x: -x[1]))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", colors=plt.cm.Set2(np.linspace(0,1,len(labels))))
    ax.set_title(f"{title} ({len(nodes)} neurons)", fontsize=13)
plt.suptitle("Pruned Network Structure: Neuron Type Distribution", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "pruning_structure.png"),
            dpi=150, bbox_inches="tight"); plt.close()
print("  剪枝结构图已保存")

# ════════════════════════════════════════════════
# 2. E/I 平衡效应
# ════════════════════════════════════════════════
print("\n── [2/6] E/I Balance Effect ──")

def apply_ei_balance(W, inhib_ratio=0.3, seed=42):
    """随机将 inhib_ratio 比例的突触设为抑制性（负权重）"""
    rng = np.random.RandomState(seed)
    W_ei = W.copy()
    mask = (W_ei > 0)
    n_syn = mask.sum()
    n_inhib = int(n_syn * inhib_ratio)
    inhib_idx = np.where(mask.flatten())[0]
    rng.shuffle(inhib_idx)
    chosen = inhib_idx[:n_inhib]
    W_ei.flat[chosen] *= -1
    return W_ei

def scale_rho(W, target=0.9):
    e = np.linalg.eigvals(W); r = np.max(np.abs(e))
    return W * (target / max(r, 1e-12)), r

# 对比无抑制 vs 30% 抑制
ei_ratios = [0.0, 0.3]
ei_results = {}

for ratio in ei_ratios:
    W_ei = apply_ei_balance(adj, ratio)
    W_ei, _ = scale_rho(W_ei, 0.9)

    # 快照无输入下的自发活动
    model = ConnectomeReservoir(W_ei, 1, 1, input_scaling=1.0)
    n_step = 500
    U = np.zeros((n_step, 1))
    X = model.collect(U, discard=0)
    # 随机选 5 个神经元画轨迹
    idx_plot = np.random.RandomState(42).choice(N, 5, replace=False)
    ei_results[ratio] = (X, idx_plot)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (ratio, (X, ix)) in enumerate(ei_results.items()):
    ax = axes[idx]
    for i in ix:
        ax.plot(X[:, i], lw=1, alpha=0.7, label=neuron_names[i] if idx == 0 else "")
    ax.set_xlabel("Time step"); ax.set_ylabel("State")
    title = "No Inhibition" if ratio == 0 else "30% Inhibition"
    ax.set_title(f"Spontaneous Activity: {title}")
    if idx == 0: ax.legend(fontsize=8, ncol=2)
plt.suptitle("Effect of Excitation/Inhibition Balance on Reservoir Dynamics", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "ei_balance.png"),
            dpi=150, bbox_inches="tight"); plt.close()
print("  E/I 平衡图已保存")

# ════════════════════════════════════════════════
# 3. 脉冲输入响应
# ════════════════════════════════════════════════
print("\n── [3/6] Pulse Input Responses ──")

W_scaled, _ = scale_rho(adj, 0.9)
n_step_pulse = 300
n_pulse = 5

# 规则脉冲
U_regular = np.zeros((n_step_pulse, 1))
interval = n_step_pulse // (n_pulse + 1)
for i in range(1, n_pulse + 1):
    U_regular[i * interval, 0] = 2.0

# 不规则脉冲
U_irregular = np.zeros((n_step_pulse, 1))
rng = np.random.RandomState(42)
pulse_times = sorted(rng.choice(range(10, n_step_pulse-10), n_pulse, replace=False))
for t in pulse_times:
    U_irregular[t, 0] = 2.0

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, (U, title) in enumerate([(U_regular, "Regular Pulses"),
                                   (U_irregular, "Irregular Pulses")]):
    model = ConnectomeReservoir(W_scaled, 1, 1, input_scaling=2.0)
    X = model.collect(U, discard=0)
    # 画输入
    ax = axes[0, col]
    ax.stem(range(n_step_pulse), U.flatten(), linefmt="gray", markerfmt="ko", basefmt=" ")
    ax.set_ylabel("Input"); ax.set_title(title)
    # 画几个神经元响应
    ax = axes[1, col]
    for i in np.random.RandomState(42).choice(N, 8, replace=False):
        ax.plot(X[:, i], lw=1, alpha=0.6)
    ax.set_xlabel("Time step"); ax.set_ylabel("Neuron State")
    ax.set_title("Reservoir Response")

plt.suptitle("Reservoir Response to Pulse Inputs", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "pulse_responses.png"),
            dpi=150, bbox_inches="tight"); plt.close()
print("  脉冲响应图已保存")

# ════════════════════════════════════════════════
# 4. NARMA 任务
# ════════════════════════════════════════════════
print("\n── [4/6] NARMA10 Task ──")

def generate_narma10(n_step=3000, seed=42):
    """生成 NARMA10 时间序列"""
    rng = np.random.RandomState(seed)
    u = rng.rand(n_step + 10) * 0.5
    y = np.zeros(n_step + 10)
    for n in range(10, n_step + 10):
        y[n] = 0.3*y[n-1] + 0.05*y[n-1]*np.sum(y[n-10:n]) + 1.5*u[n-1]*u[n-10] + 0.1
    return u[10:], y[10:]

n_narma, n_discard = 3000, 500
u_narma, y_narma = generate_narma10(n_narma)
U_narma = u_narma.reshape(-1, 1)
Y_narma = y_narma.reshape(-1, 1)

model = ConnectomeReservoir(W_scaled, 1, 1, input_scaling=1.0)
model.fit_ridge(U_narma, Y_narma, discard=n_discard, alpha=1e-4)

Yp = model.predict(U_narma, discard=n_discard)
Yt = Y_narma[n_discard:]
m = min(len(Yp), len(Yt))
mse_narma = float(np.mean((Yp[:m] - Yt[:m])**2))
corr_narma = pearsonr(Yt[:m].flatten(), Yp[:m].flatten())[0]

fig, ax = plt.subplots(figsize=(12, 5))
t = np.arange(min(500, m))
ax.plot(t, Yt[:500], 'k-', lw=1.5, alpha=0.7, label="Target")
ax.plot(t, Yp[:500], 'r-', lw=1.5, alpha=0.8, label=f"ESN (MSE={mse_narma:.4f}, r={corr_narma:.3f})")
ax.set_xlabel("Time step"); ax.set_ylabel("NARMA output"); ax.legend()
ax.set_title("NARMA10 Task: C. elegans Connectome Reservoir Performance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "narma10.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  NARMA10: MSE={mse_narma:.4f}, r={corr_narma:.3f}")

# ════════════════════════════════════════════════
# 5. 可靠性分析（Intra-vs-Inter-series correlation）
# ════════════════════════════════════════════════
print("\n── [5/6] Reliability Analysis ──")

n_trials = 20
n_step_rel = 200
U_rel = np.sin(np.linspace(0, 6*np.pi, n_step_rel)).reshape(-1, 1)

all_trials = []
for t in range(n_trials):
    model = ConnectomeReservoir(W_scaled, 1, 1, input_scaling=1.0, leaky_rate=0.3)
    X = model.collect(U_rel, discard=0)
    all_trials.append(X)

all_trials = np.array(all_trials)  # (n_trials, n_step, N)
nrn_sample = np.random.RandomState(42).choice(N, 20, replace=False)

# intra-series: 相同输入不同 trial 的同一神经元相关性
intra_corrs = []
# inter-series: 相同输入不同 trial 的不同神经元相关性
inter_corrs = []

for ni in nrn_sample:
    for t1 in range(n_trials):
        for t2 in range(t1+1, n_trials):
            r_intra, _ = pearsonr(all_trials[t1, :, ni], all_trials[t2, :, ni])
            intra_corrs.append(r_intra)
            # inter: 随机选另一个神经元
            nj = np.random.RandomState(t1*100+t2*10+ni).choice(N, 1)[0]
            r_inter, _ = pearsonr(all_trials[t1, :, ni], all_trials[t2, :, nj])
            inter_corrs.append(r_inter)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(intra_corrs, bins=30, alpha=0.6, color="steelblue", label=f"Intra-series (mean={np.mean(intra_corrs):.3f})")
ax.hist(inter_corrs, bins=30, alpha=0.6, color="coral", label=f"Inter-series (mean={np.mean(inter_corrs):.3f})")
ax.axvline(np.mean(intra_corrs), color="steelblue", ls="--", lw=2)
ax.axvline(np.mean(inter_corrs), color="coral", ls="--", lw=2)
ax.set_xlabel("Pearson Correlation"); ax.set_ylabel("Count")
ax.set_title("Reliability Analysis: Intra-series vs Inter-series Correlations")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "reliability.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Intra-series mean r={np.mean(intra_corrs):.3f}")
print(f"  Inter-series mean r={np.mean(inter_corrs):.3f}")

# ════════════════════════════════════════════════
# 6. 距离-信息传播分析
# ════════════════════════════════════════════════
print("\n── [6/6] Distance-Information Propagation ──")

# 计算所有节点对的最短路径距离
shortest_paths = dict(nx.all_pairs_shortest_path_length(Gu))
distances = []
correlations = []

n_pairs = 2000
rng_pairs = np.random.RandomState(42)
all_nodes = list(Gu.nodes())
all_indices = list(range(N))

# 用前一步收集的状态
model = ConnectomeReservoir(W_scaled, 1, 1, input_scaling=1.0)
U_dist = np.sin(np.linspace(0, 8*np.pi, 500)).reshape(-1, 1)
X_dist = model.collect(U_dist, discard=0)

for _ in range(n_pairs):
    i, j = rng_pairs.choice(all_indices, 2, replace=False)
    try:
        d = shortest_paths[neuron_names[i]][neuron_names[j]]
    except:
        continue
    r, _ = pearsonr(X_dist[:, i], X_dist[:, j])
    distances.append(d)
    correlations.append(r)

fig, ax = plt.subplots(figsize=(10, 6))
# 按距离分组统计
unique_dists = sorted(set(distances))
means, stds = [], []
for d in unique_dists:
    vals = [correlations[k] for k, dd in enumerate(distances) if dd == d]
    means.append(np.mean(vals)); stds.append(np.std(vals))
ax.errorbar(unique_dists, means, yerr=stds, fmt='o-', lw=2, ms=8, color="steelblue",
            capsize=5, capthick=2)
ax.set_xlabel("Graph Distance (shortest path length)")
ax.set_ylabel("Mean State Correlation")
ax.set_title("Information Propagation: State Correlation vs Network Distance")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "supplement", "distance_correlation.png"),
            dpi=150, bbox_inches="tight"); plt.close()
print("  距离-相关性图已保存")

print("\n✅ Supplementary analysis complete!")
print(f"   新增 6 张图到 output/supplement/")
