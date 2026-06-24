# 连接组作为库网络：从结构到计算——项目计划书


## 一、项目概述

本项目旨在将生物神经元的真实连接组（Connectome）建模为一个库网络（Reservoir Computing Network），通过计算神经科学的视角，探索“结构如何决定计算能力”这一核心问题。

**核心理念**：库网络的核心思想是——一个固定的、高维的、循环的、非线性的动力系统，可以通过训练一个简单的线性读出层来完成计算任务。生物神经网络恰恰具备这些属性：固定的结构连接、高维的状态空间、丰富的循环连接、非线性的神经元动力学。因此，用真实的连接组数据来构建库网络，是检验“生物结构是否具有计算优势”的自然实验。

**课程主题对接**：本项目将综合运用课程中的多个主题——
- **单神经元模型**（leaky integrate-and-fire 或 rate-based 神经元）
- **突触模型**（连接矩阵中的突触权重）
- **库网络计算**（核心框架）
- **工作记忆/决策模型**（用延迟匹配或抉择任务作为读出目标）

**复杂系统网络分析**：项目将包含对连接组拓扑结构的深入分析（度分布、聚类系数、模块度、小世界属性等），并将其与计算性能相关联——这正是复杂系统研究的核心视角。


## 二、项目目标

### 主要目标
1. **构建**：使用真实的神经元连接组数据，在 BrainPy 框架中构建一个库网络模型。
2. **评估**：在标准的库网络基准任务（记忆容量、时序预测、决策任务）上评估其计算性能。
3. **对比**：将真实连接组的性能与随机化/null 模型进行对比，量化“结构信息”带来的计算优势。
4. **分析**：通过复杂网络分析，揭示哪些拓扑特征（如模块度、中枢节点、小世界性）与计算性能相关。

### 次要目标
- 模拟“损伤实验”：移除特定节点（如高介数中心性的神经元），观察性能下降模式。
- 状态空间可视化：用 PCA/t-SNE 展示不同输入条件下储备池状态的演化轨迹。


## 三、数据集选择：线虫（C. elegans）连接组

### 为什么选择线虫？

| 考量因素 | 线虫连接组的优势 |
| :--- | :--- |
| **规模小** | 仅 302 个神经元，~5000 个突触，完全可以在本地运行 |
| **数据完备** | 连接矩阵、神经元类型、神经调质信息均有详细记录 |
| **易于获取** | OpenWorm 项目提供多种格式（CSV、NeuroML、Python API） |
| **已有标杆** | BAAIWorm 等项目提供了可对比的基准 |
| **计算友好** | 可在个人电脑上完成全部仿真和分析 |

### 数据获取方式

**推荐方案**：使用 OpenWorm 的 Python 包 `owmeta` 或直接下载 CSV 格式的连接矩阵。

```python
# 方式一：通过 OpenWorm Python API
from owmeta.core import connectome
conn = connectome.Connection(data_license="CC0")  # 获取连接数据

# 方式二：直接加载 CSV（推荐，更轻量）
import pandas as pd
import numpy as np

# 从 OpenWorm 数据仓库下载连接矩阵
adj_matrix = pd.read_csv('c_elegans_connectome.csv', index_col=0)
# 302 x 302 的有向加权邻接矩阵
```

**数据下载地址**：
- OpenWorm 数据门户：`https://docs.openworm.org/en/latest/Download/`
- WormAtlas 神经连接图：`https://www.wormatlas.org/neuronalwiring.html`


## 四、技术方案

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据层                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  C. elegans 连接组 (302x302 有向加权邻接矩阵)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
├─────────────────────────────────────────────────────────────────┤
│                       网络分析层                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  拓扑分析：度分布、聚类系数、模块度、小世界性、中心性   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
├─────────────────────────────────────────────────────────────────┤
│                       模型层 (BrainPy)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  储备池：以连接组为 W_rec 的 leaky-integrator 神经元池  │   │
│  │  输入层：选择感觉神经元作为输入节点                      │   │
│  │  读出层：线性回归 (Ridge) 训练输出权重                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
├─────────────────────────────────────────────────────────────────┤
│                       任务层                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  记忆容量任务 | 时序预测 (Lorenz) | 决策任务           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
├─────────────────────────────────────────────────────────────────┤
│                       结果层                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  性能对比图 | 损伤模拟 | 状态空间可视化               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 技术栈

| 组件 | 工具 | 说明 |
| :--- | :--- | :--- |
| **计算框架** | BrainPy (v3.0+) | 课程专用，支持 `brainpy.dyn.Reservoir` 和自定义网络 |
| **环境** | conda 环境 `zf-li23` | 已配置好 |
| **数值计算** | NumPy, SciPy | 矩阵运算、稀疏矩阵 |
| **网络分析** | NetworkX | 复杂网络拓扑分析 |
| **机器学习** | scikit-learn | Ridge 回归训练读出层 |
| **可视化** | Matplotlib, Seaborn | 结果图表 |
| **数据处理** | Pandas | 加载连接矩阵 |

### 4.3 BrainPy 实现方案

BrainPy 提供了两种实现路径：

#### 方案 A：使用内置 `brainpy.dyn.Reservoir` 类

```python
import brainpy as bp
import brainpy.math as bm

# 使用内置 Reservoir 类
reservoir = bp.dyn.Reservoir(
    input_shape=(num_input,),      # 输入维度
    num_out=302,                   # 302 个神经元
    leaky_rate=0.3,               # 泄漏率
    activation='tanh',             # 激活函数
    rec_connectivity=1.0,          # 全连接（但我们会用自定义权重）
    spectral_radius=0.9,           # 谱半径
)
```

**问题**：内置 `Reservoir` 类使用随机连接，不支持直接注入自定义的连接矩阵。

#### 方案 B：自定义网络（推荐）

使用 BrainPy 的 `bp.DynamicalSystem` 基类，手动构建 leaky-integrator 储备池：

```python
class ConnectomeReservoir(bp.DynamicalSystem):
    """基于连接组的储备池网络"""
    def __init__(self, connectome_matrix, num_input, leaky_rate=0.3):
        super().__init__()
        
        # 连接组权重矩阵 (302 x 302)
        self.W_rec = bm.TrainVar(connectome_matrix, trainable=False)  # 固定，不训练
        
        # 输入权重矩阵 (随机初始化)
        self.W_in = bm.TrainVar(bm.random.normal(0, 0.1, (num_input, 302)))
        
        # 偏置
        self.b = bm.TrainVar(bm.zeros(302))
        
        # 参数
        self.leaky_rate = leaky_rate
        self.num_neurons = 302
        
    def update(self, x):
        # leaky-integrate 动力学
        # r(t+1) = (1-alpha)*r(t) + alpha*tanh(W_in*x + W_rec*r(t) + b)
        r_new = (1 - self.leaky_rate) * self.r + \
                self.leaky_rate * bm.tanh(self.W_in @ x + self.W_rec @ self.r + self.b)
        self.r = r_new
        return self.r
```

**为什么选择方案 B**：
- 完全控制连接矩阵（直接使用真实连接组）
- 与 BrainPy 的训练接口兼容（可使用 `bp.RidgeTrainer`）
- 更灵活，便于后续的损伤模拟和网络分析

### 4.4 输入/输出节点的选择

基于线虫的生物学知识：
- **输入节点**：选择感觉神经元（如 ASE、AWC、AFD 等），模拟外部刺激输入
- **读出节点**：选择运动神经元或中间神经元，输出决策或行为指令

在项目中，可以尝试多种选择方案，并比较其性能差异。

### 4.5 评估任务

#### 任务 1：记忆容量（Memory Capacity）

测试网络能“记住”多长时间前的输入信息。这是库网络最基础的评估指标。

```
输入：随机时间序列 u(t)
任务：预测过去 k 步的输入 u(t-k)
记忆容量 = Σ_k ρ²(u(t-k), 预测值)
```

#### 任务 2：时序预测（Lorenz 混沌系统）

预测 Lorenz 吸引子的未来状态，这是库网络的经典基准任务。

```python
# BrainPy 内置 Lorenz 数据
import brainpy_datasets as bd
data = bd.chaos.LorenzEq(100, dt=0.01)  # 生成 Lorenz 时间序列
```

#### 任务 3：感知决策任务

使用 BrainPy 内置的认知任务数据集：

```python
from brainpy_datasets import cognitive
dataset = cognitive.RatePerceptualDecisionMaking()  # 速率感知决策任务
```

### 4.6 对比实验设计

| 对比组 | 说明 | 目的 |
| :--- | :--- | :--- |
| **真实连接组** | 使用真实的 C. elegans 连接矩阵 | 基准 |
| **随机化 null 模型** | 保持度序列不变，随机重连 | 检验拓扑结构的作用 |
| **Erdős–Rényi 随机网络** | 相同节点数和边数的随机网络 | 检验密度/边数的作用 |
| **小世界网络** | Watts-Strogatz 模型，相同度分布 | 检验小世界性的作用 |


## 五、复杂系统网络分析

这是项目的亮点之一。我们将对连接组进行系统的拓扑分析，并与计算性能建立关联。

### 5.1 基础拓扑指标

| 指标 | 计算方法 | 生物学意义 |
| :--- | :--- | :--- |
| **度分布** | `networkx.degree_histogram()` | 网络是否存在 hub 节点 |
| **聚类系数** | `networkx.average_clustering()` | 局部信息传递效率 |
| **平均最短路径** | `networkx.average_shortest_path_length()` | 全局信息传递效率 |
| **介数中心性** | `networkx.betweenness_centrality()` | 哪些节点是“交通枢纽” |
| **模块度** | `networkx.community.modularity()` | 网络是否存在功能模块 |

### 5.2 小世界性分析

检验线虫连接组是否具有小世界属性（高聚类系数 + 短平均路径）。

```python
import networkx as nx

# 计算小世界系数
def small_worldness(G):
    # 与相同度分布的随机图对比
    rand_G = nx.random_reference(G)
    C_real = nx.average_clustering(G)
    C_rand = nx.average_clustering(rand_G)
    L_real = nx.average_shortest_path_length(G)
    L_rand = nx.average_shortest_path_length(rand_G)
    sigma = (C_real / C_rand) / (L_real / L_rand)
    return sigma  # > 1 表示具有小世界性
```

### 5.3 核心-边缘结构

识别网络中的核心节点（高连接密度的子集）和边缘节点，分析核心节点在计算任务中的作用。

### 5.4 拓扑-性能相关性

将拓扑指标（如模块度、介数中心性分布）与任务性能（记忆容量、决策准确率）进行相关性分析，回答：**哪些结构特征赋予网络计算能力？**


## 六、预期仿真结果

### 6.1 性能对比图
- **横轴**：任务类型（记忆容量、时序预测、决策）
- **纵轴**：性能指标（准确率/误差）
- **四组对比**：真实连接组、随机化 null、ER 随机、小世界网络
- **预期**：真实连接组在至少一项任务上显著优于 null 模型

### 6.2 记忆容量曲线
- **横轴**：时间延迟（k 步）
- **纵轴**：相关性 ρ²
- **预期**：真实连接组比随机网络具有更长的记忆衰退时间常数

### 6.3 损伤模拟图
- 按介数中心性从高到低依次移除节点
- **横轴**：移除节点比例
- **纵轴**：任务性能下降百分比
- **预期**：移除高中心性节点会导致性能急剧下降

### 6.4 状态空间可视化
- 使用 PCA 将 302 维状态投影到 2D
- 不同颜色代表不同输入类别
- **预期**：不同输入对应的状态轨迹在空间中形成可分离的簇

### 6.5 拓扑-性能热图
- 行：各拓扑指标
- 列：各任务性能
- 颜色：相关系数
- **预期**：模块度、小世界系数等与记忆容量呈正相关


## 七、项目时间规划

| 阶段 | 内容 | 预估时间 |
| :--- | :--- | :--- |
| **第 1 周** | 数据获取与预处理；网络拓扑分析 | 2-3 天 |
| **第 2 周** | BrainPy 模型构建与调试 | 3-4 天 |
| **第 3 周** | 基准任务运行（记忆容量、时序预测） | 2-3 天 |
| **第 4 周** | 决策任务 + 对比实验 + 损伤模拟 | 3-4 天 |
| **第 5 周** | 结果分析与可视化 | 2-3 天 |
| **第 6 周** | 论文撰写 | 3-4 天 |


## 八、参考文献

[1] Suárez, L. E., et al. (2024). Connectome-based reservoir computing with the conn2res toolbox. *Nature Communications*, 15(1). 

[2] Gauthier, D. J., et al. (2021). Next-generation reservoir computing. *Nature Communications*, 12(1), 1-8. 

[3] Yang, G. R., & Wang, X. J. (2020). Artificial neural networks for neuroscientists: A primer. *Neuron*, 107(6), 1048-1070. 

[4] Wang, C., et al. (2024). A Differentiable Approach to Multi-scale Brain Modeling. *arXiv:2406.19708*. 

[5] OpenWorm Project. C. elegans connectome data. `https://docs.openworm.org/` 

[6] Lukoševičius, M. (2012). A practical guide to applying echo state networks. *Neural Networks: Tricks of the Trade*, 659-686. 

