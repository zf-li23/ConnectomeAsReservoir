"""
Phase 1: 网络拓扑分析
======================
带进度条和超时机制，适合本地运行。
"""
import os, pickle, warnings, signal
from contextlib import contextmanager
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(os.path.join(OUTPUT_DIR, "network"), exist_ok=True)

# ── 超时上下文管理器 ──────────────────────────
class TimeoutError_(Exception):
    pass

@contextmanager
def time_limit(seconds, label="operation"):
    def handler(signum, frame):
        raise TimeoutError_(f"\n  ⏰ {label} 超时 ({seconds}s)")
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        yield
    except TimeoutError_ as e:
        print(str(e))
        raise
    finally:
        signal.alarm(0)

# ── 加载数据 ──────────────────────────────────
print("加载连接组数据...")
with open(os.path.join(OUTPUT_DIR, "connectome_graph.pkl"), "rb") as f:
    G = pickle.load(f)
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"),
               allow_pickle=True)
neuron_meta = pickle.loads(data["neuron_meta"].item())
neuron_names = data["neuron_names"].tolist()
adj_matrix = data["adj_matrix"]
N, E = G.number_of_nodes(), G.number_of_edges()
print(f"图: {N} 节点, {E} 边 (有向)\n")

results = {}
G_undirected = G.to_undirected()
largest_cc = max(nx.connected_components(G_undirected), key=len)
G_lcc = G_undirected.subgraph(largest_cc).copy()

# ════════════════════════════════════════════════
# 1. 基础拓扑指标（轻量）
# ════════════════════════════════════════════════
print("── [1/6] 基础拓扑指标 ──")
in_deg = np.array([d for _, d in G.in_degree()])
out_deg = np.array([d for _, d in G.out_degree()])
total_deg = in_deg + out_deg
results["avg_in_degree"] = float(np.mean(in_deg))
results["avg_out_degree"] = float(np.mean(out_deg))
results["max_in_degree"] = int(np.max(in_deg))
results["max_out_degree"] = int(np.max(out_deg))
results["avg_clustering"] = float(nx.average_clustering(G_undirected))
results["assortativity"] = float(nx.degree_assortativity_coefficient(G_undirected))
results["density"] = float(nx.density(G))
results["degree_CV_in"] = float(np.std(in_deg) / max(np.mean(in_deg), 1e-10))
results["degree_CV_out"] = float(np.std(out_deg) / max(np.mean(out_deg), 1e-10))

with time_limit(30, "avg_shortest_path_length"):
    results["avg_shortest_path_length"] = float(nx.average_shortest_path_length(G_lcc))
    results["diameter"] = int(nx.diameter(G_lcc))

for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ════════════════════════════════════════════════
# 2. 小世界性分析
# ════════════════════════════════════════════════
print("\n── [2/6] 小世界性分析 ──")
avg_spl = results["avg_shortest_path_length"]
avg_cc = results["avg_clustering"]
C_rand, L_rand = None, None
try:
    with time_limit(90, "small-world analysis"):
        print("  生成度保持随机化网络...")
        GR = nx.random_reference(G_undirected, seed=42)
        print("  计算指标...")
        C_rand = nx.average_clustering(GR)
        LCC_rand = max(nx.connected_components(GR), key=len)
        GR_lcc = GR.subgraph(LCC_rand)
        L_rand = nx.average_shortest_path_length(GR_lcc, method='unweighted')
        print(f"  C_real={avg_cc:.4f}, C_rand={C_rand:.4f}")
        print(f"  L_real={avg_spl:.4f}, L_rand={L_rand:.4f}")
except Exception as e:
    print(f"  ⚠ 使用 ER 近似: {e}")
    p = 2 * E / (N * (N - 1))
    C_rand = p
    L_rand = np.log(N) / max(np.log(max(N * p, 1.1)), 0.1)

if C_rand and L_rand:
    sigma = (avg_cc / C_rand) / (avg_spl / L_rand)
    omega = (L_rand / avg_spl) - (avg_cc / C_rand)
    results.update(C_random=float(C_rand), L_random=float(L_rand),
                   small_world_sigma=float(sigma), small_world_omega=float(omega))
    print(f"  小世界系数 σ = {sigma:.4f}  {'✅ >1' if sigma > 1 else '❌'}")

# ════════════════════════════════════════════════
# 3. 中心性分析
# ════════════════════════════════════════════════
print("\n── [3/6] 中心性分析 ──")
bc = {}
try:
    with time_limit(120, "betweenness_centrality"):
        print("  计算介数中心性 (k=50)...")
        bc = nx.betweenness_centrality(G_undirected, k=50, normalized=True, seed=42)
except:
    print("  ⚠ 超时，使用度中心性代替")
    dc_fb = nx.degree_centrality(G)
    bc = dict(sorted(dc_fb.items(), key=lambda x: -x[1])[:10])

if not bc:
    bc = {n: 0.0 for n in G.nodes()}
sorted_bc = sorted(bc.items(), key=lambda x: -x[1])
print("  Top 10 中枢节点:")
for name, val in sorted_bc[:10]:
    t = neuron_meta.get(name, {}).get("type", "?")
    print(f"    {name:6s} ({t:8s}) bc={val:.4f}")

dc = nx.degree_centrality(G)
sorted_dc = sorted(dc.items(), key=lambda x: -x[1])
results["top10_bc"] = [n for n, _ in sorted_bc[:10]]
results["top10_dc"] = [n for n, _ in sorted_dc[:10]]

# ════════════════════════════════════════════════
# 4. 模块度分析
# ════════════════════════════════════════════════
print("\n── [4/6] 模块度分析 ──")
communities, modularity = None, 0
try:
    with time_limit(60, "louvain"):
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_undirected, seed=42)
        modularity = nx.community.modularity(G_undirected, communities)
        results["num_communities"] = len(communities)
        results["modularity"] = float(modularity)
        print(f"  社区数: {len(communities)}, Q={modularity:.4f}")
        for i, c in enumerate(communities):
            sizes = Counter(neuron_meta.get(n, {}).get("type", "?") for n in c)
            print(f"  社区 {i}: {len(c)} 节点, {dict(sizes.most_common(3))}")
except Exception as e:
    print(f"  ⚠ {e}")

# ════════════════════════════════════════════════
# 5. 保存
# ════════════════════════════════════════════════
print("\n── [5/6] 保存 ──")
with open(os.path.join(OUTPUT_DIR, "network", "analysis_results.pkl"), "wb") as f:
    pickle.dump(results, f)
print("  已保存")

# ════════════════════════════════════════════════
# 6. 可视化
# ════════════════════════════════════════════════
print("\n── [6/6] 可视化 ──")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 6a. 度分布
ax = axes[0, 0]
bins = np.logspace(np.log10(1), np.log10(max(total_deg) + 1), 30)
ax.hist(in_deg[in_deg > 0], bins=bins, alpha=0.6, label=f"In (avg={np.mean(in_deg):.1f})")
ax.hist(out_deg[out_deg > 0], bins=bins, alpha=0.6, label=f"Out (avg={np.mean(out_deg):.1f})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Degree")
ax.set_ylabel("Count"); ax.set_title("Degree Distribution"); ax.legend()

# 6b. 介数中心性
ax = axes[0, 1]
ax.hist(list(bc.values()), bins=30, color="steelblue", edgecolor="white")
ax.set_xlabel("Betweenness Centrality"); ax.set_ylabel("Count")
ax.set_title("Betweenness Centrality Distribution")

# 6c. 同配性
ax = axes[0, 2]
dc_vals = np.array([dc[n] for n in G.nodes()])
ax.scatter(total_deg, dc_vals, s=8, alpha=0.5, c="coral")
ax.set_xlabel("Total Degree"); ax.set_ylabel("Degree Centrality")
ax.set_title(f"Assortativity r={results['assortativity']:.3f}")

# 6d. 小世界
ax = axes[1, 0]
C_p = C_rand if C_rand else 0; L_p = L_rand if L_rand else avg_spl
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [avg_cc, avg_spl], w, label="Real", color="steelblue")
ax.bar(x + w/2, [C_p, L_p], w, label="Random", color="coral")
ax.set_xticks(x); ax.set_xticklabels(["Clustering", "Path Length"])
ax.set_ylabel("Value"); ax.set_title(f"Small-World σ={results.get('small_world_sigma', 0):.2f}")
ax.legend()

# 6e. 社区
ax = axes[1, 1]
if communities:
    sizes = sorted([len(c) for c in communities], reverse=True)
    ax.bar(range(len(sizes)), sizes, color="mediumseagreen")
    ax.set_title(f"Community Sizes (Q={modularity:.3f})")
else:
    ax.text(0.5, 0.5, "Skipped", ha="center", va="center", transform=ax.transAxes)
ax.set_xlabel("Community index"); ax.set_ylabel("Size")

# 6f.连接矩阵 + hub
ax = axes[1, 2]
top5 = [n for n, _ in sorted_bc[:5]]
top5_idx = [neuron_names.index(n) for n in top5 if n in neuron_names]
ax.imshow(adj_matrix[:100, :100], cmap="YlOrRd", aspect="auto", interpolation="none")
for name, idx in zip(top5, top5_idx):
    if idx < 100:
        ax.annotate(name, (idx, idx), fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
ax.set_title("Top-5 Hub Neurons"); ax.set_xlabel("Postsynaptic"); ax.set_ylabel("Presynaptic")

plt.suptitle("C. elegans Connectome: Network Topology Analysis", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "network", "network_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ Phase 1 完成！")
print(f"   平均度: {results['avg_in_degree']:.1f}, "
      f"σ={results.get('small_world_sigma', 0):.2f}, "
      f"Q={results.get('modularity', 0):.3f}")
print(f"   中枢节点: {results['top10_bc'][:3]}")
