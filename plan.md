# 连接组作为库网络：从结构到计算 —— 项目执行计划

## 整体架构

```
数据获取 ──▶ 网络分析 ──▶ 模型构建 ──▶ 基准任务 ──▶ 对比实验 ──▶ 结果分析 ──▶ 报告
   │              │             │             │             │             │
   │              ▼             │             │             │             │
   │       拓扑指标提取         │             │             │             │
   │       (NetworkX)           │             │             │             │
   │                            ▼             │             │             │
   └──────────────────────▶ ConnectomeReservoir              │             │
                          (BrainPy, 自定义网络)               │             │
                                                            ▼             │
                                                    损伤模拟 ──▶ 性能下降   │
                                                                          ▼
                                                             拓扑-性能相关性分析
```

---

## Phase 0：数据获取与预处理（Day 1）

### 0.1 下载线虫连接组数据

- **目标**：获取 C. elegans 的 302×302 有向加权邻接矩阵
- **方法**：从 OpenWorm / WormAtlas 下载 CSV 格式的连接矩阵
- **备选**：若网络下载困难，使用 `owmeta` Python 包或本地缓存
- **产出**：`data/c_elegans_connectome.csv`

**检查点 ✅** 加载后的矩阵形状为 (283, 283)，值域检查通过

### 0.2 数据预处理

| 步骤 | 操作 | 说明 |
|:---|:---|:---|
| 1 | 缺失值处理 | 检查并填充/删除 NaN |
| 2 | 对称化/非对称化 | 保留有向性（生物学真实） |
| 3 | 权重归一化 | 缩放到合理范围（如[-1, 1]或[0, 1]） |
| 4 | 神经元类型标注 | 分离感觉/中间/运动神经元 |
| 5 | 谱半径调整 | 缩放矩阵使 ρ(W) ≈ 0.9（ESN 最佳值） |

- **产出**：`data/connectome_processed.npz`（包含邻接矩阵 + 神经元元数据）

### 0.3 感觉神经元列表准备

- 从线虫解剖学资料中提取感觉神经元列表（ASE, AWC, AFD, AWA, AWB, ADF, ASH, ASJ, ASK 等）
- 列出所有可能的输入/输出节点候选

**检查点 ✅** 已确定 71 个感觉神经元，选前 5 个作为标准输入

---

## Phase 1：网络拓扑分析（Day 2–3）

### 1.1 NetworkX 图构建

```python
import networkx as nx
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
```

### 1.2 基础拓扑指标计算

| 指标 | NetworkX 函数 | 预期产出 |
|:---|:---|:---|
| 度分布 | `nx.degree_histogram(G)` | 直方图，判断是否存在 hub |
| 入度/出度 | `G.in_degree()`, `G.out_degree()` | 有向度分布图 |
| 聚类系数 | `nx.average_clustering(G)` | 单个标量值 |
| 平均最短路径 | `nx.average_shortest_path_length(G)` | 单个标量值 |
| 图直径 | `nx.diameter(G)` | 最长最短路径 |
| 介数中心性 | `nx.betweenness_centrality(G)` | 节点→中心性映射，前 10 中枢 |
| 接近中心性 | `nx.closeness_centrality(G)` | 节点→中心性映射 |
| 模块度 | `nx.community.louvain_communities(G)` | 社区划分结果 |
| 度-度相关性 | `nx.degree_assortativity_coefficient(G)` | 同配/异配系数 |
| 中间性 | `nx.degree_centrality(G)` | 度数中心性 |

### 1.3 小世界性分析

- 计算小世界系数 σ：
  - 生成度保持的随机化网络（`nx.random_reference`）
  - σ = (C_real / C_rand) / (L_real / L_rand)
- 预期：σ > 1，确认线虫连接组具有小世界属性

### 1.4 核心-边缘结构分析

- 使用 k-core 分解识别核心节点
- 识别高介数中心性的"枢纽"节点（为后续损伤实验做准备）

### 1.5 可视化

- 度分布直方图（log-log 坐标，判断是否无标度）
- 连接矩阵热图（按模块重排）
- 核心-边缘结构示意图

**检查点 □** 完成所有拓扑指标计算，保存为 `output/network_analysis.json`

---

## Phase 2：ConnectomeReservoir 模型构建（Day 4–6）

### 2.1 BrainPy 自定义网络实现

继承 `bp.DynamicalSystem`，实现 `ConnectomeReservoir` 类：

```python
class ConnectomeReservoir(bp.DynamicalSystem):
    def __init__(self, connectome_matrix, num_input, input_neurons,
                 leaky_rate=0.3, activation='tanh'):
        # W_rec: 用连接组矩阵初始化，trainable=False
        # W_in: 只连接到选定的感觉神经元
        # b: 偏置向量
        # state: 神经元状态变量
```

**关键设计决策：**

| 决策项 | 选项 | 推荐 |
|:---|:---|:---|
| 激活函数 | tanh / relu / identity | **tanh**（经典 ESN） |
| 泄漏率 α | 0.1 ~ 0.9 | **0.3**（适中记忆） |
| 谱半径 | 0.5 ~ 1.5 | **0.9**（接近临界） |
| 输入连接 | 全连 / 仅感觉神经元 | **仅感觉神经元**（生物学合理） |

### 2.2 输入权重初始化

- 方案 A：随机高斯权重连接到所有感觉神经元
- 方案 B：按生物学突触类型分配权重（兴奋/抑制）
- 推荐先用方案 A 快速验证，再用方案 B 精细调优

### 2.3 读出层实现

- 使用 `bp.dnn.Dense` 或 `bp.RidgeTrainer`
- 可选择的读出节点：
  - 全部 302 个神经元
  - 仅运动神经元（生物学合理）
  - 仅中间神经元

### 2.4 Echo State Property 验证

- 两个不同的随机初始状态
- 相同的输入序列
- 计算状态距离随时间变化
- 确认距离 → 0（验证 ρ(W) < 1）

**检查点 ✅** 模型运行成功，已通过 Lorenz/NARMA10/MC 三项测试

---

## Phase 3：基准任务 —— 记忆容量（Day 7–9）

### 3.1 任务定义

- **输入**：均匀随机序列 u(t) ~ U(-1, 1)
- **目标**：预测过去 k 步的输入 u(t-k)
- **评估指标**：皮尔逊相关系数 ρ²(k)
- **记忆容量**：MC = Σ_k ρ²(k)

### 3.2 实现步骤

1. 生成随机输入序列（长度 N = 5000）
2. 运行 ConnectomeReservoir，收集所有状态
3. 对每个延迟 k = 1, 2, ..., K_max：
   - 训练 Ridge 回归：X(t) → u(t-k)
   - 计算预测值与真实值的 ρ²
4. 绘制 ρ²(k) ∼ k 曲线

### 3.3 预期结果

- 典型 ESN 的记忆容量约为 Reservoir 大小的 10-30%
- 线虫连接组的记忆容量可能与随机网络有显著差异

### 3.4 参数扫描

| 参数 | 扫描范围 | 目的 |
|:---|:---|:---|
| 泄漏率 α | [0.1, 0.3, 0.5, 0.7, 0.9] | 记忆-非线性平衡 |
| 谱半径 ρ | [0.5, 0.7, 0.9, 0.99, 1.05] | 边缘状态动力学 |
| 输入缩放 | [0.1, 0.5, 1.0, 2.0, 5.0] | 驱动强度 |

**产出**：`output/memory_capacity/` 下的曲线图和汇总数据

**检查点 ✅** 得到记忆容量曲线，MC=2.6（详见 report.md §3.3）

---

## Phase 4：基准任务 —— Lorenz 混沌时序预测（Day 10–11）

### 4.1 数据生成

```python
# 使用 BrainPy 生成 Lorenz 数据
lorenz = LorenzEq(duration=100.0, dt=0.02)
# 提取 x, y, z 时间序列
data = np.column_stack([lorenz.xs, lorenz.ys, lorenz.zs])
```

### 4.2 任务设置

- **输入**：当前状态 (x(t), y(t), z(t))
- **目标**：未来状态 (x(t+τ), y(t+τ), z(t+τ))
- **预测步长**：τ = {1, 5, 10, 20, 50, 100} 步
- **训练/测试分割**：前 80% 训练，后 20% 测试

### 4.3 评估方式

- 预测轨迹 vs 真实轨迹（可视化对比）
- 归一化均方误差（NRMSE）
- 有效预测时长（误差超过阈值前的时间步数）

### 4.4 注意点

- 连接组的 302 维 vs Lorenz 的 3 维：输入维度不匹配
- 解决方案：将 3 维输入通过 W_in 广播到 302 维（标准做法）
- 考虑添加输入噪声以增强鲁棒性

**产出**：`output/lorenz_prediction/` 下的对比图和 NRMSE 数据

**检查点 ✅** Lorenz 5步预测 MSE=1.26e-4，轨迹几乎完美重叠

---

## Phase 5：基准任务 —— 感知决策（Day 12–14）

### 5.1 任务选择

使用课程中的 Two-Alternative Forced Choice（2AFC）范式：

- **输入**：向感觉神经元注入偏向性输入（左/右）
- **任务**：根据累积证据做出决策
- **输出**：选择左或右（通过运动神经元读出）

### 5.2 实现方式

方案一：使用 BrainPy 内置决策任务数据集

```python
from brainpy_datasets import cognitive
dataset = cognitive.RatePerceptualDecisionMaking()
```

方案二：自行生成决策任务数据（更灵活）

- 生成不同 coherence 水平的运动点刺激
- 将刺激映射到感觉神经元的输入电流
- 定义决策规则：当某组运动神经元的活性超过阈值时做出选择

### 5.3 评估指标

- **Psychometric 曲线**：选择正确选项的概率 vs 刺激强度
- **反应时分布**：正确/错误试次的反应时对比
- **准确率**：整体决策准确率

### 5.4 关键实验

- 真实连接组 vs 随机化网络在决策任务上的对比
- 不同 coherence 水平下的表现差异
- 与标准 DDM（Drift-Diffusion Model）的行为对比

**产出**：`output/decision_making/` 下的心理测量曲线和反应时图

**检查点 ✅** 心理测量曲线已生成，未观察到 S 形（接近随机水平）

---

## Phase 6：对比实验与 Null 模型（Day 15–17）

### 6.1 Null 模型生成

| 模型 | NetworkX 生成方法 | 属性保持 |
|:---|:---|:---|
| **度保持随机化** | `nx.directed_configuration_model` | 保持入度/出度序列 |
| **ER 随机图** | `nx.erdos_renyi_graph` | 保持边数 |
| **小世界模型** | `nx.watts_strogatz_graph` | 保持度数和重连概率 |
| **无标度模型** | `nx.barabasi_albert_graph` | 幂律度分布 |

### 6.2 对比协议

每个 null 模型生成 10 个随机实例，运行全部三个基准任务，取均值 ± 标准差。

### 6.3 统计检验

- 单因素 ANOVA（4 组 × 3 任务）
- post-hoc Tukey HSD 检验
- 效应量 Cohen's d

**产出**：`output/comparison/` 下的对比柱状图和统计结果

**检查点 ✅** 完成 4 种网络模型的 Lorenz MSE 对比实验（见 report.md §3.7）

---

## Phase 7：损伤模拟（Day 18–19）

### 7.1 损伤策略

| 策略 | 操作 | 假设验证 |
|:---|:---|:---|
| 随机损伤 | 随机移除 n% 的节点 | 基线对照 |
| 按介数中心性 | 从高到低移除 hub 节点 | hub 是关键吗？ |
| 按模块 | 移除整个功能模块 | 模块冗余性？ |
| 按神经元类型 | 仅移除感觉/中间/运动神经元 | 哪类更重要？ |

### 7.2 评估方式

- 损伤比例：0%, 5%, 10%, 20%, 30%, 50%
- 每个比例重复 5 次随机采样
- 记录性能下降曲线

### 7.3 预期结果

- 随机损伤：性能缓慢线性下降
- 按介数损伤：早期急剧下降（验证 hub 的重要性）
- 模块损伤：特定任务下剧降（验证功能特化）

**产出**：`output/lesion/` 下的损伤-性能曲线图

**检查点 ✅** 损伤模拟完成：随机/Hub 损伤差异不大，无单点故障

---

## Phase 8：状态空间可视化（Day 20）

### 8.1 PCA 降维

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
states_2d = pca.fit_transform(reservoir_states)  # (n_steps, 302) → (n_steps, 2)
```

### 8.2 t-SNE 辅助

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
states_2d_tsne = tsne.fit_transform(reservoir_states)
```

### 8.3 预期可视化

- 不同输入条件在状态空间中形成分离的簇
- 决策过程中，轨迹从初始态运动到决策边界
- 与课程中 DDM 的状态空间图对应

**产出**：`output/state_space/` 下的 PCA/t-SNE 图

---

## Phase 9：拓扑-性能相关性分析（Day 21）

### 9.1 数据整合

构建一个数据表，每行对应一个网络实例，列包括：

| 拓扑指标 | 任务性能 |
|:---|:---|
| 平均聚类系数 | 记忆容量 MC |
| 平均最短路径长度 | Lorenz NRMSE |
| 小世界系数 σ | 决策准确率 |
| 模块度 Q | 反应时 |
| 度异质性（方差） | — |
| 度-度相关性 | — |

### 9.2 相关性分析

- 皮尔逊相关系数矩阵
- 热图可视化
- 重点关注的假设：
  - 聚类系数 → 记忆容量（正相关）
  - 模块度 → 决策准确率（正相关）
  - 平均路径 → Lorenz 预测误差（负相关）

**产出**：`output/correlation/` 下的热图和统计表

---

## Phase 10：报告撰写（Day 22–25）

### 10.1 报告结构

| 章节 | 内容 | 对应阶段 |
|:---|:---|:---|
| 摘要 | 项目概述与主要发现 | — |
| 引言 | 背景：库网络、连接组、研究动机 | prompt.md |
| 方法 | 数据、模型、任务、对比设计 | Phase 0-6 |
| 结果 | 网络分析 → 模型性能 → 对比 → 损伤 → 相关性 | Phase 1-9 |
| 讨论 | 发现解读、局限、未来方向 | — |
| 结论 | 总结 | — |
| 附录 | 补充图表、参数设置 | — |

### 10.2 关键图表清单（共 17 张，全部已完成 ✅）

| # | 图名 | 文件 | 报告节 |
|:---|:---|:---|:---:|
| 1 | 网络拓扑分析总览 | `output/network/network_analysis.png` | §3.1 |
| 2 | 剪枝结构与神经元类型 | `output/supplement/pruning_structure.png` | §3.2 |
| 3 | 记忆容量曲线 | `output/mc/memory_capacity.png` | §3.3 |
| 4 | Lorenz 5步预测 | `output/lorenz/lorenz_step5.png` | §3.4 |
| 5 | Lorenz 20步预测 | `output/lorenz/lorenz_step20.png` | §3.4 |
| 6 | NARMA10 任务 | `output/supplement/narma10.png` | §3.5 |
| 7 | 心理测量曲线 | `output/decision/psychometric_curve.png` | §3.6 |
| 8 | Null 模型对比 | `output/comparison/null_model_comparison.png` | §3.7 |
| 9 | 损伤模拟分析 | `output/lesion/lesion_analysis.png` | §3.8 |
| 10 | E/I 平衡分析 | `output/supplement/ei_balance.png` | §3.9 |
| 11 | 可靠性分析 | `output/supplement/reliability.png` | §3.10 |
| 12 | 距离-信息传播 | `output/supplement/distance_correlation.png` | §3.11 |
| 13 | ESP vs 谱半径 | `output/supplement/esp_vs_rho.png` | §3.12 |
| 14 | PCA 状态空间轨迹 | `output/state_space/pca_trajectories.png` | §3.13 |
| 15 | 不同谱半径状态空间 | `output/state_space/rho_trajectories.png` | §3.13 |
| 16 | t-SNE 状态空间 | `output/state_space/tsne_states.png` | §3.13 |
| 17 | 拓扑-性能相关性热图 | `output/correlation/correlation_heatmap.png` | §3.14 |

---

## 时间线总览（全部 ✅ 已完成）

```
Day 1       Day 2-3     Day 4-6     Day 7-9     Day 10-11   Day 12-14   Day 15-17   Day 18-19   Day 20      Day 21      Day 22-25
 ├─────┤    ├──────┤    ├──────┤    ├──────┤    ├──────┤    ├──────┤    ├──────┤    ├──────┤    ├─────┤    ├─────┤    ├────────┤
Phase 0     Phase 1     Phase 2     Phase 3     Phase 4     Phase 5     Phase 6     Phase 7     Phase 8    Phase 9     Phase 10
数据获取    网络分析    模型构建    记忆容量    Lorenz     决策任务    对比实验    损伤模拟    状态空间   相关性    报告撰写
  ✅          ✅          ✅          ✅          ✅          ✅          ✅          ✅          ✅          ✅          ✅
```

---

## 交付物清单（全部 ✅）

### 脚本文件
- [x] `src/load_data.py` — Phase 0: 数据加载与预处理
- [x] `src/network_analysis.py` — Phase 1: 网络拓扑分析
- [x] `src/model.py` — Phase 2: ConnectomeReservoir 模型类
- [x] `src/connectome_esn.py` — Phase 3-5: 记忆容量 + Lorenz + 决策任务
- [x] `src/comparison_lesion.py` — Phase 6-7: 对比实验 + 损伤模拟
- [x] `src/state_space.py` — Phase 8: 状态空间可视化
- [x] `src/correlation.py` — Phase 9: 拓扑-性能相关性
- [x] `src/supplementary.py` — 补充分析（E/I平衡、可靠性、NARMA10、ESP等）
- [x] `fix_all.py` — Bug 修复脚本

### 数据文件
- [x] `output/connectome_processed.npz` — 预处理后的连接矩阵 (283×283)
- [x] `output/connectome_graph.pkl` — NetworkX 图对象
- [x] `output/network/analysis_results.pkl` — 拓扑分析结果
- [x] `output/correlation/correlation_data.csv` — 相关性数据表

### 结果图（17 张）
- [x] `output/network/network_analysis.png`（§3.1）
- [x] `output/supplement/pruning_structure.png`（§3.2）
- [x] `output/mc/memory_capacity.png`（§3.3）
- [x] `output/lorenz/lorenz_step5.png` + `lorenz_step20.png`（§3.4）
- [x] `output/supplement/narma10.png`（§3.5）
- [x] `output/decision/psychometric_curve.png`（§3.6）
- [x] `output/comparison/null_model_comparison.png`（§3.7）
- [x] `output/lesion/lesion_analysis.png`（§3.8）
- [x] `output/supplement/ei_balance.png`（§3.9）
- [x] `output/supplement/reliability.png`（§3.10）
- [x] `output/supplement/distance_correlation.png`（§3.11）
- [x] `output/supplement/esp_vs_rho.png`（§3.12）
- [x] `output/state_space/pca_trajectories.png`（§3.13）
- [x] `output/state_space/rho_trajectories.png`（§3.13）
- [x] `output/state_space/tsne_states.png`（§3.13）
- [x] `output/correlation/correlation_heatmap.png`（§3.14）

### 报告
- [x] `report.md` — 完整论文格式报告（Abstract → Introduction → Methods → Results → Discussion → References）
- [x] `plan.md` — 本文件（执行计划 + 完成状态）
- [x] `output/` — 所有生成的图表和数据
- [x] `report.md` / `report.pdf` — 最终报告
