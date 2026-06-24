"""
Phase 0: 数据加载与预处理
===========================
加载 Varshney 2011 C. elegans 连接组数据集，构建邻接矩阵，
补充神经元类型元数据，保存为处理后的 .npz 文件。
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse

# ── 路径配置 ────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CE_SYNAPSE_DIR = os.path.join(DATA_DIR, "Ce_synapse")
NEURON_CONN_DIR = os.path.join(DATA_DIR, "NeuronalConnectivity")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 加载 Varshney 2011 连接组 ──────────────
print("=" * 60)
print("加载 Varshney 2011 连接组数据...")
print("=" * 60)

conn = pd.read_excel(
    os.path.join(NEURON_CONN_DIR, "NeuronConnect.xls")
)
col_pre, col_post = conn.columns[0], conn.columns[1]
print(f"原始记录数: {len(conn)}")
print(f"连接类型: {conn['Type'].unique()}")

# ── 2. 构建有向加权邻接矩阵 ──────────────────
# 按 (突触前, 突触后) 分组，汇总所有类型的突触数量
adj_df = conn.groupby([col_pre, col_post])["Nbr"].sum().unstack(fill_value=0)
adj_df = adj_df.fillna(0)

# 统一神经元名称列表
all_neurons = sorted(set(adj_df.index) | set(adj_df.columns))
# 确保矩阵是方阵
for n in all_neurons:
    if n not in adj_df.index:
        adj_df.loc[n] = 0
    if n not in adj_df.columns:
        adj_df[n] = 0
adj_df = adj_df.loc[all_neurons, all_neurons]

adj_matrix = adj_df.values.astype(np.float64)
n_neurons = len(all_neurons)

print(f"\n神经元总数: {n_neurons}")
print(f"邻接矩阵形状: {adj_matrix.shape}")
nz = (adj_matrix > 0).sum()
print(f"非零连接数: {nz}")
print(f"稀疏度: {nz / adj_matrix.size * 100:.2f}%")
print(f"连接总数(加权和): {adj_matrix.sum():.0f}")

# ── 3. 加载神经元类型元数据 ──────────────────
# 3a. 从 Varshney NeuronType 获取
nt = pd.read_excel(os.path.join(NEURON_CONN_DIR, "NeuronType.xls"))
neuron_meta = {}
for _, row in nt.iterrows():
    name = str(row["Neuron"]).strip()
    neuron_meta[name] = {
        "soma_position": row["Soma Position"],
        "soma_region": str(row["Soma Region"]).strip(),
    }

# 3b. 从 Ce_synapse name_neurons.txt 获取神经元类型
names_df = pd.read_csv(
    os.path.join(CE_SYNAPSE_DIR, "name_neurons.txt"),
    sep=r"\s+",
    header=None,
    names=["Neuron", "Class", "Type"],
)
type_map = dict(zip(names_df["Neuron"], names_df["Type"]))

# 合并元数据
for i, neuron in enumerate(all_neurons):
    meta = neuron_meta.get(neuron, {})
    ntype = type_map.get(neuron, "unknown")
    meta["type"] = ntype
    neuron_meta[neuron] = meta

# 统计
type_counts = {}
for n in all_neurons:
    t = neuron_meta.get(n, {}).get("type", "unknown")
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"\n神经元类型分布: {dict(sorted(type_counts.items()))}")

# ── 4. 构建 NetworkX 图 ─────────────────────
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
mapping = {i: name for i, name in enumerate(all_neurons)}
G = nx.relabel_nodes(G, mapping)

print(f"\nNetworkX 图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")

# ── 5. 保存处理后的数据 ─────────────────────
output = {
    "adj_matrix": adj_matrix,  # (N, N) float64
    "neuron_names": all_neurons,  # list of str
    "neuron_meta": neuron_meta,  # dict
    "G": G,
}

npz_path = os.path.join(OUTPUT_DIR, "connectome_processed.npz")
# Save non-graph data as npz
np.savez_compressed(
    npz_path,
    adj_matrix=adj_matrix,
    neuron_names=np.array(all_neurons, dtype=object),
    neuron_meta=pickle.dumps(neuron_meta),
)
# Save graph separately via pickle
with open(os.path.join(OUTPUT_DIR, "connectome_graph.pkl"), "wb") as f:
    pickle.dump(G, f)

print(f"\n数据已保存至: {npz_path}")
print(f"图已保存至: {os.path.join(OUTPUT_DIR, 'connectome_graph.pkl')}")

# ── 6. 初步可视化 ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 6a. 连接矩阵热图
ax = axes[0]
im = ax.imshow(adj_matrix[:100, :100], cmap="YlOrRd", aspect="auto", interpolation="none")
ax.set_title("Connectome Adjacency Matrix (first 100 neurons)", fontsize=13)
ax.set_xlabel("Postsynaptic neuron")
ax.set_ylabel("Presynaptic neuron")
plt.colorbar(im, ax=ax, shrink=0.8)

# 6b. 度分布
ax = axes[1]
in_degrees = [d for _, d in G.in_degree()]
out_degrees = [d for _, d in G.out_degree()]
ax.hist(in_degrees, bins=30, alpha=0.6, label="In-degree", color="steelblue")
ax.hist(out_degrees, bins=30, alpha=0.6, label="Out-degree", color="coral")
ax.set_xlabel("Degree")
ax.set_ylabel("Count")
ax.set_title("Degree Distribution", fontsize=13)
ax.legend()
ax.set_yscale("log")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "connectome_overview.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nPhase 0 完成！")
