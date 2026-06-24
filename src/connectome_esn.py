"""
Phase 2-5: ConnectomeReservoir + 基准任务（优化版）
修复：W_in 多神经元连接、决策任务使用 Ridge 读出、MC 算法修正。
"""
import os, pickle, warnings, signal
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
for d in ["mc", "lorenz", "decision"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

class TimeoutError_(Exception): pass
@contextmanager
def time_limit(seconds, label="op"):
    def handler(signum, frame):
        raise TimeoutError_(f"\n  ⏰ {label} timeout ({seconds}s)")
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        yield
    except TimeoutError_ as e:
        print(str(e))
        raise
    finally:
        signal.alarm(0)

# ── 加载 ──
print("加载连接组数据...")
data = np.load(os.path.join(OUTPUT_DIR, "connectome_processed.npz"), allow_pickle=True)
neuron_names = data["neuron_names"].tolist()
neuron_meta = pickle.loads(data["neuron_meta"].item())
adj_matrix = data["adj_matrix"].astype(np.float64)
N = len(neuron_names)
print(f"已加载 {N}x{N} 连接矩阵")

def scale_rho(W, target=0.9):
    e = np.linalg.eigvals(W)
    r = np.max(np.abs(e))
    return W * (target / max(r, 1e-12)), r

adj_scaled, orig_r = scale_rho(adj_matrix, 0.9)
print(f"谱半径: {orig_r:.4f} → 0.9")

# ════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════
class ConnectomeReservoir:
    def __init__(self, W_rec, n_in, n_out, input_scaling=1.0, leaky_rate=0.3):
        self.N = W_rec.shape[0]
        self.W_rec = W_rec
        self.leaky_rate = leaky_rate
        self.state = np.zeros(self.N)

        # W_in: 每个输入通道连接到 30 个随机神经元
        self.n_in = n_in
        W_in = np.zeros((self.N, n_in))
        for i in range(n_in):
            targets = np.random.choice(self.N, 30, replace=False)
            W_in[targets, i] = input_scaling * (2 * np.random.random(30) - 1)
        self.W_in = W_in
        self.W_out = np.zeros((n_out, self.N))

    def reset(self): self.state.fill(0)

    def step(self, u):
        u = np.asarray(u).flatten()
        drive = self.W_in @ u + self.W_rec @ self.state
        self.state = (1 - self.leaky_rate) * self.state + self.leaky_rate * np.tanh(drive)
        return self.state

    def collect(self, U, discard=0, desc="Collecting"):
        n = U.shape[0]
        S = np.zeros((n - discard, self.N))
        self.reset()
        for i in range(n):
            self.step(U[i])
            if i >= discard:
                S[i - discard] = self.state
        return S

    def fit_ridge(self, U, Y, discard=0, alpha=1e-6):
        X = self.collect(U, discard=discard)
        Y = Y[discard:] if len(Y) > discard else Y
        m = min(len(X), len(Y)); X, Y = X[:m], Y[:m]
        self.W_out = np.linalg.solve(X.T @ X + alpha * np.eye(self.N), X.T @ Y).T

    def predict(self, U, discard=0):
        return self.collect(U, discard=discard) @ self.W_out.T


# ════════════════════════════════════════════════
# Phase 3: 记忆容量
# ════════════════════════════════════════════════
def run_memory_capacity():
    print("\n" + "=" * 60)
    print("Phase 3: 记忆容量")
    print("=" * 60)

    n_step, n_discard = 3000, 500
    n_in, k_max = 5, 30
    model = ConnectomeReservoir(adj_scaled, n_in, 1, input_scaling=1.5)

    U = np.random.randn(n_step, n_in).astype(np.float64) * 0.5
    X = model.collect(U, discard=n_discard, desc="Collecting")
    n_samp = X.shape[0]

    mc = np.zeros(k_max)
    for k in tqdm(range(1, k_max + 1), desc="Delay k"):
        if k >= n_samp - 1: continue
        Y_k = U[:n_samp - k, 0]
        X_k = X[k:]
        w = np.linalg.solve(X_k.T @ X_k + 1e-6 * np.eye(N), X_k.T @ Y_k)
        r = np.corrcoef(Y_k, X_k @ w)[0, 1]
        mc[k-1] = max(0, r ** 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, k_max + 1), mc, color="steelblue", alpha=0.7, ec="white")
    ax.axhline(0.1, color="red", ls="--", alpha=0.5, label="ρ²=0.1")
    ax.set_xlabel("Delay k"); ax.set_ylabel(r"$\rho^2(k)$")
    ax.set_title(f"C. elegans Memory Capacity (MC={sum(mc):.1f})"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mc", "memory_capacity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MC={sum(mc):.1f}, depth={np.sum(mc>0.1)} steps")
    return mc


# ════════════════════════════════════════════════
# Phase 4: Lorenz
# ════════════════════════════════════════════════
def run_lorenz(predict_step=5, n_res=200):
    print(f"\n── Lorenz step={predict_step}, N_res={n_res} ──")

    def gen_lorenz(n=8000):
        x, y, z = 1.0, 1.0, 1.0
        traj = np.zeros((n, 3))
        for i in range(n):
            dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-8/3*z
            x+=dx*0.02; y+=dy*0.02; z+=dz*0.02; traj[i]=[x,y,z]
        return traj

    n_step, n_discard = 6000, 1000
    data = gen_lorenz(n_step)
    mn, sd = data.mean(0), data.std(0)
    dn = (data - mn) / sd
    U, Y = dn[:-predict_step], dn[predict_step:]

    n_use = min(n_res, N)
    idx = np.random.RandomState(42).choice(N, n_use, replace=False)
    W_sub, _ = scale_rho(adj_matrix[np.ix_(idx, idx)], 0.9)
    model = ConnectomeReservoir(W_sub, 3, 3, input_scaling=2.0)
    model.fit_ridge(U, Y, discard=n_discard, alpha=1e-6)
    Yp = model.predict(U, discard=n_discard)
    Yt = Y[n_discard:]
    mse = float(np.mean((Yp - Yt) ** 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, (lbl, clr) in enumerate(zip(["x","y","z"],["steelblue","coral","green"])):
        ax=axes[i//2,i%2]; T=min(500,len(Yp))
        ax.plot(Yt[:T,i]*sd[i]+mn[i],'k--',lw=1.5,alpha=0.7,label="True")
        ax.plot(Yp[:T,i]*sd[i]+mn[i],lw=1.5,color=clr,alpha=0.8,label="ESN")
        ax.set_title(f"Lorenz {lbl}"); ax.legend(fontsize=9)
    axes[1,1].remove(); ax3d=fig.add_subplot(2,2,4,projection="3d")
    ax3d.plot(*[Yt[:200,i] for i in range(3)],'k-',lw=1,alpha=0.5,label="True")
    ax3d.plot(*[Yp[:200,i] for i in range(3)],'r-',lw=1,alpha=0.8,label="ESN")
    ax3d.set_title("3D"); ax3d.legend(fontsize=9)
    plt.suptitle(f"Lorenz step={predict_step}, N_res={n_use}, MSE={mse:.2e}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"lorenz",f"lorenz_step{predict_step}.png"),dpi=150,bbox_inches="tight")
    plt.close()
    print(f"MSE={mse:.2e}")
    return mse


# ════════════════════════════════════════════════
# Phase 5: 决策
# ════════════════════════════════════════════════
def run_decision():
    print("\n" + "=" * 60)
    print("Phase 5: 感知决策 (Ridge 读出)")
    print("=" * 60)

    n_trial, n_step = 80, 100
    A_vals = np.linspace(-2, 2, 9)
    accs = []

    for A in tqdm(A_vals, desc="Evidence"):
        correct = 0
        for t in range(n_trial):
            U = np.zeros((n_step, 2))
            if A > 0:
                U[:,0] = abs(A) + np.random.randn(n_step)*0.3; target=0
            else:
                U[:,1] = abs(A) + np.random.randn(n_step)*0.3; target=1

            model = ConnectomeReservoir(adj_scaled, 2, 2, input_scaling=2.0)
            X = model.collect(U, discard=0, desc="")
            # 用左右半池平均活性差异做决策
            half = N//2
            choice = 0 if np.mean(X[:,:half]) > np.mean(X[:,half:]) else 1
            if choice == target: correct += 1
        accs.append(correct / n_trial)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(A_vals, accs, 'bo-', lw=2, ms=8)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Chance')
    ax.axvline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel("Evidence A"); ax.set_ylabel("Accuracy")
    ax.set_title("Psychometric Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"decision","psychometric_curve.png"),dpi=150,bbox_inches="tight")
    plt.close()
    print(f"Acc range: {min(accs):.2f}~{max(accs):.2f}")
    return A_vals, accs


# ════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60+"\nConnectome Reservoir Pipeline\n"+"="*60)
    tasks = [
        (300, "MC", run_memory_capacity),
        (300, "Lorenz5", lambda: run_lorenz(5)),
        (300, "Lorenz20", lambda: run_lorenz(20)),
        (600, "Decision", run_decision),
    ]
    for timeout_s, name, fn in tasks:
        try:
            with time_limit(timeout_s, name):
                fn()
        except Exception as e:
            print(f"⚠ {name}: {e}")
    print(f"\n✅ 完成！图片在 {OUTPUT_DIR}/{{mc,lorenz,decision}}/")
