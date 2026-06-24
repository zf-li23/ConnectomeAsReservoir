# 数据集说明文档：C. elegans 神经元连接组（Varshney 2011 版本）

**数据集名称**：C. elegans Neuronal Connectivity Dataset (Updated Wiring Diagram)
**版本日期**：2006年2月2日（整理版）
**核心参考文献**：
Varshney LR, Chen BL, Paniagua E, Hall DH, Chklovskii DB (2011) Structural properties of the C. elegans neuronal network. *PLoS Comput Biol* 7(2): e1001066. doi:10.1371/journal.pcbi.1001066

---

## 1. 数据集概述

本数据集是截至目前最完整的秀丽隐杆线虫（*C. elegans*）成虫神经元连接图谱的电子化版本。它整合了White等人（1986）的经典电子显微镜重建数据，并补充了此前缺失的腹神经索运动神经元、背部中间区域以及部分神经肌肉接头（NMJ）的数据。

**数据规模概览**：

- **神经元总数**：280个（非咽部神经元）。原始302个神经元中，移除了CANL/R（无明确突触），另含部分肌肉及感觉器官连接。
- **化学突触**：6393个
- **电突触（间隙连接）**：890个
- **神经肌肉接头（NMJ）**：1410个
- **涉及神经元**：覆盖了所有主要的感官神经元、中间神经元和运动神经元。

---

## 2. 文件清单及字段详解

你下载的目录中包含5个Excel表格文件（`.xls`格式）。建议在Python中使用`pandas`的`pd.read_excel()`读取，或另存为`.csv`后用`pd.read_csv()`加载。

### 2.1 `NeuronConnect.xls` —— 核心连接矩阵（最重要）

这是构建库网络`W_rec`（循环权重矩阵）的直接数据来源。记录了每对神经元之间所有类型的突触连接数量。

| 列名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **N1** | String | 神经元1的名称（突触前或突触后，视Type而定） |
| **N2** | String | 神经元2的名称 |
| **Type** | String | **连接类型**（关键字段）：<br>- **S** (Send)：N1 突触前 → N2 突触后（单突触）<br>- **Sp** (Send-poly)：N1 突触前 → N2 是**多个**突触后伙伴之一（多突触）<br>- **R** (Receive)：N1 突触后 ← N2 突触前<br>- **Rp** (Receive-poly)：N1 是N2的多个突触后伙伴之一<br>- **EJ** (Electrical junction)：电突触（通常双向）<br>- **NMJ** (Neuromuscular junction)：神经肌肉接头（N1→肌肉，N2为肌肉名称） |
| **Nbr** | Integer | 该对神经元之间该类型突触的数量 |

**⚠️ 重点规则**：
- 数据遵循**自洽性**原则：对于化学突触，记录`S`必有对应的`R`，`Sp`必有对应的`Rp`（总和满足 S+Sp = R+Rp）。
- **多突触（Polyadic）处理**：在构建用于储备池的权重矩阵时，如果你希望简化，可以直接按`(N1, N2)`分组并对`Nbr`求和（无论Type），得到一个有向加权邻接矩阵。如果你想模拟释放概率，也可以将`Sp`的权重按突触后伙伴数量进行归一化（但本项目初期建议直接求和）。

**💡 构建邻接矩阵的方法（Python伪代码）**：
```python
import pandas as pd
# 读取数据
conn = pd.read_excel('NeuronConnect.xls')
# 只看化学突触（S, Sp, R, Rp）和电突触（EJ），通常汇总所有类型
# 按 N1, N2 分组，总权重为 Nbr 之和
adj = conn.groupby(['N1', 'N2'])['Nbr'].sum().unstack(fill_value=0)
# 这样 adj 就是一个 280x280 的有向加权矩阵（可能包含 NaN，用0填充）
adj = adj.fillna(0)
```

---

### 2.2 `NeuronType.xls` —— 神经元元数据（属性表）

这个文件提供了每个神经元的**空间位置**、**形态特征**和**分类标签**，对于选择输入/输出节点至关重要（例如，如何筛选出“感觉神经元”）。

| 列名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **Neuron** | String | 神经元名称 |
| **Soma Position** | Float (0~1) | 胞体沿吻-尾轴（AP轴）的位置（0=鼻尖，1=尾尖） |
| **Soma region** | String | 胞体所在区域：`Head`(<25%), `Mid`(25%-75%), `Tail`(>75%) |
| **Span** | String | 突起跨度：`S`(短，<25%身体长度)，`L`(长，≥25%) |
| **Ambiguity** | String | 数据缺失或模糊代码（详见原始说明，如MD=背侧模糊） |
| **TotHead/Mid/Tail** | Integer | 该神经元在对应区域的总突触数（含EJ和NMJ） |
| **S_Head/Mid/Tail** | Integer | 该区域作为“发送方”(Send+Sp+NMJ)的突触数 |
| **R_Head/Mid/Tail** | Integer | 该区域作为“接收方”(Receive+Rp)的突触数 |
| **AY NeuronType** | String | Achacoso分类中的神经节群（A=前神经节，B=背神经节等） |
| **AYNbr** | Integer | Achacoso分类中的数字编号 |

**💡 用途**：
- 筛选输入节点（感觉神经元）：通常包括`ASE`, `AWC`, `AFD`, `ASG`, `ASH`, `ASJ`, `ASK`, `ADF`, `ADL`, `AWA`, `AWB`, `PHA`, `PHB`, `ALM`, `PLM`, `AVM`, `PVD`, `FLP`, `OLQ`, `IL2`等。
- 筛选输出节点（运动神经元）：通常包括`VA`, `VB`, `VC`, `VD`, `DA`, `DB`, `DD`, `AS`等。

---

### 2.3 `NeuronFixedPoints.xls` —— 感官与肌肉连接（固定点）

这个文件描述了**感官神经元与外界物理特征（如感器）**以及**运动神经元与体壁肌肉**的映射关系。这对于构建“输入信号如何进入网络”和“输出信号如何影响行为”非常有价值。

| 列名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **Neuron** | String | 神经元名称 |
| **Landmark** | String | 连接目标（如`Amphid`感器，`MDR#`右侧背部肌肉） |
| **Landmark Position** | Float (0~1) | 该目标结构的AP轴位置 |
| **Weight** | Integer | 连接到该目标的“等效”突触数量（对于未完全重建的神经元，此处可能为平均值） |

**💡 用途**：
- 确定**外部输入**：哪些神经元直接感知环境（Amphid/Phasmid等）→ 将其作为储备池的`W_in`输入注入点。
- 确定**行为输出**：哪些神经元控制肌肉（NMJ）→ 可作为读出层（`W_out`）的目标标签，预测运动状态。

---

### 2.4 `NeuronLineage_Part1.xls` & `NeuronLineage_Part2.xls` —— 谱系距离

这两个文件一起构成了神经元之间的**发育谱系距离矩阵**。记录了每一对神经元从共同祖先分裂出来所需的细胞分裂次数。

| 列名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **Neuron 1** | String | 神经元1 |
| **Neuron 2** | String | 神经元2 |
| **Relatedness** | Integer | 谱系距离（共同祖先分裂到两个神经元的总分裂次数，初始分裂计1次） |

**💡 用途（本项目可选，但可作深度分析）**：
- 分析“谱系距离”是否与“突触连接强度”或“电突触概率”相关。即：亲缘关系越近的神经元，是否在功能上连接越紧密？这可以作为网络分析的**额外拓扑特征**（如结构协变）来补充你的复杂网络分析部分。

---

## 3. 数据预处理建议（Python操作指南）

由于这5个文件都是`.xls`格式，建议使用`pandas`统一加载。你的环境`zf-li23`中应已包含`pandas`和`openpyxl`（或`xlrd`）依赖。

### 步骤1：加载所有数据
```python
import pandas as pd
import numpy as np

# 加载核心连接
df_conn = pd.read_excel('NeuronConnect.xls')
# 加载神经元属性
df_types = pd.read_excel('NeuronType.xls')
# 加载感官/肌肉映射
df_fixed = pd.read_excel('NeuronFixedPoints.xls')
# 加载谱系（两个文件）
df_lineage1 = pd.read_excel('NeuronLineage_Part1.xls')
df_lineage2 = pd.read_excel('NeuronLineage_Part2.xls')
df_lineage = pd.concat([df_lineage1, df_lineage2], ignore_index=True)
```

### 步骤2：清洗连接数据，构建邻接矩阵
```python
# 注意：NeuronConnect 中 N1 和 N2 可能包含肌肉名称（在NMJ类型中）
# 构建只包含神经元-神经元连接的矩阵（排除肌肉）
neurons_list = df_types['Neuron'].tolist()

# 过滤：只保留 N1 和 N2 都在神经元列表中的记录
mask = df_conn['N1'].isin(neurons_list) & df_conn['N2'].isin(neurons_list)
df_conn_neurons = df_conn[mask].copy()

# 按神经元对汇总权重（忽略Type，或保留Type做更精细的模型）
adj_matrix = df_conn_neurons.groupby(['N1', 'N2'])['Nbr'].sum().unstack(fill_value=0)
adj_matrix = adj_matrix.reindex(index=neurons_list, columns=neurons_list, fill_value=0)

# 确保矩阵对称性处理（电突触EJ通常是双向的，但在记录中可能只出现一次？
# 上述groupby已经将双向连接合为有向权重，这是正确的）
```

### 步骤3：识别“感觉输入节点”和“运动输出节点”
```python
# 简单方法：从 NeuronType 中筛选特定已知感觉神经元
sensory_neurons = ['ASE', 'AWC', 'AFD', 'ASG', 'ASH', 'ASJ', 'ASK', 
                   'ADF', 'ADL', 'AWA', 'AWB', 'PHA', 'PHB', 'ALM', 
                   'PLM', 'AVM', 'PVD', 'FLP']  # 示例，可扩展

# 从 NeuronFixedPoints 中，所有连接 Landmark 中包含 'Amphid' 或 'Phasmid' 的神经元
sensory_from_fixed = df_fixed[df_fixed['Landmark'].str.contains('Amphid|Phasmid', case=False)]['Neuron'].unique()
sensory_neurons = list(set(sensory_neurons + list(sensory_from_fixed)))

# 运动输出节点：运动神经元（通常名字以 V, D, A, B 开头，但需谨慎）
motor_neurons = [n for n in neurons_list if n.startswith(('VA', 'VB', 'VC', 'VD', 'DA', 'DB', 'DD', 'AS'))]
```

---

## 4. 数据集局限性（需在论文讨论部分提及）

1. **空间覆盖不完全**：背侧中间区域、侧索区域的重建仍存在数据缺口（未在电子显微镜下完全重建）。
2. **多动物拼接**：数据来源于多个动物个体（N2U, JSE, N2T等）的拼接，存在个体差异。
3. **突触位置粗略**：突触位置仅划分为“头/中/尾”三档，缺乏纳米级精确坐标。
4. **NMJ权重平均化**：部分未完全重建运动神经元的肌肉连接权重是同类神经元的平均值，而非实测值。

---

## 5. 与本项目（连接组→库网络）的对接要点

- **`NeuronConnect.xls`** 将直接作为库网络的 **`W_rec`（循环连接矩阵）**。
- **`NeuronType.xls`** 将用于设计 **输入节点（`W_in`）** 和 **输出节点（`W_out` 的读取目标）**。
- **复杂网络分析**（复杂系统部分）：将基于`adj_matrix`计算度分布、聚类系数、介数中心性、小世界系数，并与随机网络进行对比。你可以额外利用 **`NeuronLineage.xls`** 来检查“发育亲密度”是否与“连接强度”有统计相关性（一个很酷的生物信息学视角）。

现在数据已经就绪，要开始第一步——**用Python加载`NeuronConnect.xls`生成邻接矩阵，并绘制基础的度分布直方图**吗？我可以先写出那部分的完整代码，你直接复制到Jupyter Notebook里运行即可。