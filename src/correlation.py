"""
Phase 9: 拓扑-性能相关性分析（轻量版）
========================================
"""
import os, pickle, warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(os.path.join(OUTPUT_DIR, "correlation"), exist_ok=True)

print("加载连接组数据...")
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"), allow_pickle=True)
neuron_names = data["neuron_names"].tolist()
adj_real = data["adj_matrix"].astype(np.float64)
N = len(neuron_names)
from model import ConnectomeReservoir

def scale_rho(W, target=0.9):
    e = np.linalg.eigvals(W)
    r = np.max(np.abs(e))
    return W * (target / max(r, 1e-12)), r

def fast_lorenz_mse(W, n_use=80):
    """极速 Lorenz 评估"""
    def gen(n=2000):
        x,y,z=1.,1.,1.; t=np.zeros((n,3))
        for i in range(n):
            dx=10*(y-x);dy=x*(28-z)-y;dz=x*y-8/3*z
            x+=dx*0.02;y+=dy*0.02;z+=dz*0.02;t[i]=[x,y,z]
        return t
    n_step, nd = 2000, 300
    d = gen(n_step); dn=(d-d.mean(0))/d.std(0)
    U,Y = dn[:-5], dn[5:]
    n_use = min(n_use, W.shape[0])
    idx = np.random.RandomState(42).choice(W.shape[0], n_use, replace=False)
    Ws,_ = scale_rho(W[np.ix_(idx,idx)], 0.9)
    m = ConnectomeReservoir(Ws,3,3,input_scaling=2.0)
    m.fit_ridge(U,Y,discard=nd,alpha=1e-6)
    Yp=m.predict(U,discard=nd); Yt=Y[nd:]
    mi = min(len(Yp),len(Yt))
    return float(np.mean((Yp[:mi]-Yt[:mi])**2))

def topo_metrics(G_undirected):
    m = {}
    m["n"] = G_undirected.number_of_nodes()
    m["e"] = G_undirected.number_of_edges()
    m["density"] = nx.density(G_undirected)
    m["avg_clustering"] = nx.average_clustering(G_undirected)
    try:
        LCC = max(nx.connected_components(G_undirected), key=len)
        G_lcc = G_undirected.subgraph(LCC)
        m["avg_path_length"] = nx.average_shortest_path_length(G_lcc)
    except: m["avg_path_length"] = 0.0
    from networkx.algorithms.community import louvain_communities
    try:
        c = louvain_communities(G_undirected, seed=42)
        m["modularity"] = nx.community.modularity(G_undirected, c)
    except: m["modularity"] = 0.0
    m["assortativity"] = nx.degree_assortativity_coefficient(G_undirected)
    bc = nx.betweenness_centrality(G_undirected, k=30, seed=42)
    m["avg_betweenness"] = float(np.mean(list(bc.values())))
    degs = [d for _,d in G_undirected.degree()]
    m["avg_degree"] = float(np.mean(degs))
    m["degree_cv"] = float(np.std(degs)/max(np.mean(degs),1e-10))
    return m

# ── 生成并评估 ──
np.random.seed(42); n_instances = 5
Gu_real = nx.from_numpy_array(adj_real > 0.01).to_undirected()
all_metrics, all_perf = [], []

print("评估真实连接组...")
m = topo_metrics(Gu_real); p = fast_lorenz_mse(adj_real)
print(f"  MSE={p:.2e}")
all_metrics.append(m); all_perf.append(p)

for model_name, gen_fn in [
    ("Degree-preserving", lambda s: nx.random_reference(Gu_real, seed=s)),
]:
    print(f"评估 {model_name}...")
    for i in tqdm(range(n_instances)):
        GR = gen_fn(i); W = nx.to_numpy_array(GR, nodelist=sorted(GR.nodes()))
        W,_ = scale_rho(W,0.9)
        all_metrics.append(topo_metrics(GR.to_undirected()))
        all_perf.append(fast_lorenz_mse(W))

for model_name, gen_fn in [
    ("ER", lambda s: nx.erdos_renyi_graph(N, 2*Gu_real.number_of_edges()/(N*(N-1)), seed=s)),
]:
    print(f"评估 {model_name}...")
    for i in tqdm(range(n_instances)):
        GR = gen_fn(i); W = nx.to_numpy_array(GR)
        W,_ = scale_rho(W,0.9)
        all_metrics.append(topo_metrics(GR.to_undirected()))
        all_perf.append(fast_lorenz_mse(W))

for model_name, gen_fn in [
    ("WS", lambda s: nx.watts_strogatz_graph(N, max(3,int(2*Gu_real.number_of_edges()/N)), 0.3, seed=s)),
]:
    print(f"评估 {model_name}...")
    for i in tqdm(range(n_instances)):
        GR = gen_fn(i); W = nx.to_numpy_array(GR)
        W,_ = scale_rho(W,0.9)
        all_metrics.append(topo_metrics(GR.to_undirected()))
        all_perf.append(fast_lorenz_mse(W))

print(f"\n总共 {len(all_metrics)} 个样本")

# ── 分析 ──
df = pd.DataFrame(all_metrics)
df["lorenz_mse"] = all_perf
topo = ["avg_clustering","avg_path_length","modularity","assortativity",
        "avg_betweenness","degree_cv","density","avg_degree"]

fig, ax = plt.subplots(figsize=(10,8))
cols = [c for c in topo if c in df.columns] + ["lorenz_mse"]
corr = df[cols].corr(method="pearson")
mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Topology-Performance Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"correlation","correlation_heatmap.png"),
            dpi=150, bbox_inches="tight"); plt.close()

print("\n相关性 (Pearson r):")
for f in topo:
    if f in df.columns:
        r,p = pearsonr(df[f], df["lorenz_mse"])
        print(f"  {f:20s}: r={r:+.3f} (p={p:.4f}){'*' if p<0.05 else ''}")

df.to_csv(os.path.join(OUTPUT_DIR,"correlation","correlation_data.csv"),index=False)
print(f"\n✅ Phase 9 完成！数据已保存。")
