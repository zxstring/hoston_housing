# 波士顿房价预测 — 模型原理详解

> 面向读者：有编程基础，了解 Python 和基本数据结构，初次接触机器学习。
> 阅读后你将理解五个模型的内部机制、为什么集成方法优于单模型，以及如何选择合适的算法。

---

## 目录

1. [前置概念：什么是"模型"](#0-前置概念什么是模型)
2. [Ridge 回归](#1-ridge-回归)
3. [决策树](#2-决策树)
4. [随机森林](#3-随机森林)
5. [XGBoost](#4-xgboost)
6. [LightGBM](#5-lightgbm)
7. [综合对比](#6-综合对比)

---

## 0. 前置概念：什么是"模型"

### 从编程类比

假设你要写一个函数来预测房价：

```python
def predict_price(features):
    # 你的代码
    return price
```

传统编程：你手动写出规则，比如：

```python
def predict_price(features):
    return features["RM"] * 3.5 + features["LSTAT"] * (-0.6) + 10
```

机器学习：**你只定义函数的形式（模型结构），让计算机从数据中自动找出最佳参数。**

```python
def predict_price(features, weights):
    # 模型只定义结构，参数 weights 是未知的
    return features["RM"] * weights["w1"] + features["LSTAT"] * weights["w2"] + weights["bias"]
```

### 训练 = 在一堆可能的参数中找到最优解

```
数据集（506 条已标注样本）
    ↓
"喂给"模型结构
    ↓
模型尝试不同参数组合
    ↓
计算每条样本的"预测值 - 真实值"的误差
    ↓
调整参数使误差减小
    ↓
重复直到误差收敛
    ↓
得到最终模型参数
```

### 几个贯穿全文的核心概念

**训练集 vs 测试集**：把 506 条数据分为两部分：
- 训练集（~80%）：用来调整模型参数，就像考前刷题
- 测试集（~20%）：用来评估最终效果，就像真正的考试——这 20% 的数据模型从未见过

**过拟合 vs 欠拟合**：

```
欠拟合                        刚好                        过拟合
预测线是直的                   预测线捕捉了趋势             预测线穿过每一个点
明显没学到规律                 泛化能力最强                 记住了噪音，测试集很惨
```

用编程类比理解过拟合：你把 404 条训练样本的训练集误差「硬编码」成 0，但测试集会暴露你根本没有学到真正的规律。

**交叉验证（Cross-Validation）**：一种更稳健的评估方式。把训练集切成 5 份，轮流用 4 份训练、1 份验证，取平均值：

```
Fold 1: [验证][训练][训练][训练][训练] → score₁
Fold 2: [训练][验证][训练][训练][训练] → score₂
Fold 3: [训练][训练][验证][训练][训练] → score₃
Fold 4: [训练][训练][训练][验证][训练] → score₄
Fold 5: [训练][训练][训练][训练][验证] → score₅

最终 CV Score = mean(score₁, ..., score₅)
```

**评价指标**：
- **MSE（均方误差）**：$\frac{1}{n}\sum(y_i - \hat{y}_i)^2$。值越小越好，对大误差特别敏感（平方放大）
- **MAE（平均绝对误差）**：$\frac{1}{n}\sum|y_i - \hat{y}_i|$。单位与原始数据相同（千美元），更直观
- **R²（决定系数）**：表示"模型解释了多少比例的 y 的变异"。R²=0 表示模型跟猜平均值一样烂；R²=1 表示完美预测；R²=0.9 表示模型解释了 90% 的变异

---

## 1. Ridge 回归

### 1.1 从线性回归讲起

最简单的房价预测公式：

```
MEDV = β₀ + β₁×CRIM + β₂×ZN + ... + β₁₃×LSTAT
```

即：
```python
def predict(X, beta):
    return beta[0] + sum(beta[j] * X[j] for j in range(1, 14))
```

训练线性回归的目标：找到一组 $\beta$ 使预测值与真实值的差距最小。

数学表达——最小化均方误差：

$$\text{Loss} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

这有闭式解（不需要迭代，直接套公式）：

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

用 numpy 一行就能算：
```python
beta = np.linalg.inv(X.T @ X) @ X.T @ y
```

### 1.2 问题：多重共线性

波士顿数据中，RAD 和 TAX 的相关系数是 **0.91**——它们几乎在说同一件事。

在数学上，这意味着 $X^TX$ 矩阵**几乎不可逆**（行列式接近 0）。结果：

- 系数的值会非常巨大且敏感——训练数据稍微变化，系数可能从 +1000 跳到 -800
- 虽然训练集误差看起来还行，但测试集上会崩
- 系数本身不可解释——你不能说"这个特征对房价贡献 X"

用一个程序化的比喻：

```python
# 两个几乎一样的特征
RAD  = [1, 2, 3, 1, 2, 3, ...]
TAX  = [200, 310, 420, 190, 305, 415, ...]  # 与 RAD 高度相关

# 线性回归试图同时拟合这两个系数
# 解不唯一：下面两组参数给出几乎一样的预测
# 方案1: β_RAD=5,    β_TAX=0
# 方案2: β_RAD=-500, β_TAX=2.5
# 计算机会给出一个数值极端不稳定的解
```

### 1.3 Ridge 的解决方案：加惩罚项

Ridge 在损失函数上加了一个 **L2 正则化项**：

$$\text{Loss}_{ridge} = \underbrace{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}_{\text{普通 MSE}} + \underbrace{\alpha \sum_{j=1}^{p} \beta_j^2}_{\text{L2 惩罚}}$$

关键直觉：**模型允许有一定的误差（第一项），但不允许参数太大（第二项）。**

参数解释：
- $\alpha$（alpha）：惩罚力度。由我们通过交叉验证来选
  - $\alpha \to 0$：惩罚几乎无效，退化为普通线性回归
  - $\alpha \to \infty$：惩罚极强，所有 $\beta$ 趋近于 0，模型只会猜均值
- $\beta_j^2$：系数的平方——大系数的惩罚呈二次增长，所以模型会"愿意"让多个系数均匀变小，而非让一个系数独大

闭式解变为：

$$\hat{\beta}_{ridge} = (X^TX + \alpha I)^{-1}X^Ty$$

注意 $+\alpha I$——在主对角线上加了 $\alpha$，使得矩阵一定可逆。这是 Ridge 在数值上的巧妙之处。

```python
# 数学本质
# 原本: (X^T X)^(-1)   ← 可能接近奇异
# Ridge: (X^T X + αI)^(-1)  ← αI 保证了对角线上有最小值，一定可逆
```

### 1.4 为什么有效

```
             无正则化:  β 可以任意大
                       └→ 过拟合
             有正则化:  β 被压缩
                       └→ 模型更简单、泛化更好
```

在波士顿数据上：RAD 和 TAX 高度相关时，Ridge 不会给它们分配 "1000" 和 "-980" 这样相互抵消的极端系数，而是让它们都保持合理的大小。

### 1.5 在本项目中的超参数

Ridge 只有一个关键超参数：

| 参数 | 含义 | 搜索范围 |
|------|------|----------|
| `alpha` | L2 正则化强度 | 0.01 ~ 100 |

最佳值通过 `RandomizedSearchCV` 找到：**alpha ≈ 0.56**。

### 1.6 局限

- 只能学习**线性关系**。RM 对房价的影响可能是非线性的（多一间房的价值在低端房和高端房不同），Ridge 无法捕获这个模式
- 对离群值敏感——MSE 的平方惩罚会让模型被极端值"牵着走"
- 这就是为什么我们还需要后面的非线性模型

---

## 2. 决策树

### 2.1 核心直觉

决策树不做数学运算，而是**反复问"是否"问题**：

```
问: RM ≤ 6.8 ?
    ├── 是 → 问: LSTAT ≤ 14.4 ?
    │        ├── 是 → 预测: 22.5k
    │        └── 否 → 预测: 28.1k
    └── 否 → 问: RM ≤ 7.4 ?
             ├── 是 → 预测: 34.2k
             └── 否 → 预测: 42.8k
```

用编程来理解，决策树就是一堆嵌套的 if-else：

```python
def predict(features):
    if features["RM"] <= 6.8:
        if features["LSTAT"] <= 14.4:
            return 22.5
        else:
            return 28.1
    else:
        if features["RM"] <= 7.4:
            return 34.2
        else:
            return 42.8
```

### 2.2 树是怎么"学"出来的

**贪心策略**：每次分裂都选择"此刻最优"的特征和阈值。

对一个候选分裂（特征=RM，阈值=6.8）：

```
分裂前: 父节点有 100 条数据
  MEDV 值: [18, 24, 35, 15, 42, 22, ...]
  方差: 85.3（数据很散乱）

分裂后:
  左子节点 (RM≤6.8): 60 条数据    右子节点 (RM>6.8): 40 条数据
    MEDV: [18,24,15,22,...]         MEDV: [35,42,38,...]
    方差: 25.1                      方差: 30.4

增益 = 85.3 - (60/100×25.1 + 40/100×30.4) = 85.3 - 27.2 = 58.1
```

遍历所有特征和候选阈值，选**增益最大**的 split。

具体算法（回归树）：

1. 对每个特征 $j$：
   - 对该特征的所有值排序
   - 对每两个相邻值的中间点作为候选阈值 $s$
   - 计算：分裂后左右子节点的 MSE 加权和
2. 选择使以下值最小的 $(j, s)$：

$$\min_{j,s}\left[ \min_{c_1}\sum_{x_i\in R_1}(y_i-c_1)^2 + \min_{c_2}\sum_{x_i\in R_2}(y_i-c_2)^2 \right]$$

其中 $R_1$ 和 $R_2$ 是被分裂出的两个区域，$c_1$ 和 $c_2$ 分别是两个区域的预测值（取区域内 y 的均值）。

3. 递归重复，直到满足停止条件。

### 2.3 停止条件（防止过拟合）

如果不加限制，树会一直分裂直到每个叶子只有 1 条数据——完美拟合训练集，但测试集毫无泛化。

```python
class DecisionTree:
    def should_stop(self, node):
        if len(node.samples) < min_samples_split:  # 样本太少，不值得分
            return True
        if node.depth >= max_depth:                 # 已经到了最大深度
            return True
        if len(node.samples) < 2 * min_samples_leaf: # 分了之后叶子太小
            return True
        if mse_gain < tol:                          # 再分也没啥收益了
            return True
        return False
```

本项目搜索的超参数：

| 参数 | 含义 | 如果太小 | 如果太大 |
|------|------|----------|----------|
| `max_depth` | 最大深度 | 过拟合 | 欠拟合 |
| `min_samples_split` | 节点少于 N 条数据就不分了 | 过拟合 | 欠拟合 |
| `min_samples_leaf` | 叶子少于 N 条数据就不分了 | 过拟合 | 欠拟合 |

### 2.4 优点 vs 缺点

**优点**：
- 不需要特征标准化（只用比较大小，不需要计算距离）
- 天然处理非线性关系
- 可解释——你能完整画出一棵树的决策路径

**缺点**：
- **高方差**：数据稍微变化，树的结构可能完全不同——第一次分裂选 RM，换个 random_state 可能变成 LSTAT
- **容易过拟合**：如果不加剪枝约束，能背下整个训练集
- **不如集成方法**：单棵树的表达能力有限

### 2.5 本项目表现

R²=0.79，CV 标准差 ±6.2（很大，证实了高方差问题）。这直接引出了下面的随机森林。

---

## 3. 随机森林

### 3.1 核心思想：三个臭皮匠顶个诸葛亮

如果一棵树容易过拟合（方差大），那就种一片森林，让每棵树的错误互相抵消。

```python
# 单棵树
prediction = tree.predict(x)        # 高方差

# 随机森林
predictions = [tree_i.predict(x) for tree_i in forest]
prediction = mean(predictions)       # 低方差
```

统计学原理：如果每棵树的预测是独立的随机变量，方差为 $\sigma^2$，那么 N 棵树的平均值方差为 $\sigma^2 / N$。

### 3.2 两重随机性

**如果每棵树都用同样的数据、同样的特征，那它们都会犯同样的错误**——平均值方差不会降低。

所以需要**故意制造差异**。

#### 第一重：Bootstrap 抽样

对 404 条训练数据进行**有放回抽样**，每棵树得到一个不同的训练集：

```python
def bootstrap_sample(data, n_samples):
    """有放回抽 n_samples 条"""
    indices = np.random.randint(0, len(data), size=n_samples)
    return data[indices]

# 每棵树得到不同的训练集
tree_1_data = bootstrap_sample(train_data, 404)
tree_2_data = bootstrap_sample(train_data, 404)
# ...
```

有放回意味着：一些样本可能被抽中多次，一些可能一次都没被抽中。大约 **63%** 的数据会被每棵树看到，剩下 37%（OOB, Out-Of-Bag）可以作为天然的验证集。

#### 第二重：随机特征子集

每次分裂时，只随机考虑一部分特征：

```python
def find_best_split(node, features, max_features=0.5):
    # 不是看所有特征
    candidate_features = random.sample(features, k=int(len(features) * max_features))
    # 只在候选特征中找最优分裂
    return best_split_among(candidate_features)
```

这迫使每棵树从不同的角度观察数据。比如树 1 主要看 RM 和 LSTAT，树 2 主要看 DIS 和 NOX，树 3 看 CRIM 和 AGE……最终的多样性使得平均预测更稳定。

### 3.3 为什么这有效

```
单棵树:                    随机森林:
                          ┌─ 树1（数据子集A, 特征子集D）
                          ├─ 树2（数据子集B, 特征子集E）
一棵树承担所有压力         ├─ 树3（数据子集C, 特征子集F）
过拟合风险高               ├─ ...
                          └─ 树N
                            ↓
                          取平均值 → 过拟合的风险被 N 份分摊
```

关键公式——**偏差-方差分解**：

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- 随机森林通过 Bootstrap + 随机特征，**降低方差**（不同树的错误互相抵消）
- 但同时不增加偏差（每棵树依然是表达能力强的决策树）
- 这就是它几乎总是优于单棵决策树的原因

### 3.4 本项目超参数

| 参数 | 含义 | 搜索范围 |
|------|------|----------|
| `n_estimators` | 树的数量 | 50 ~ 300 |
| `max_depth` | 每棵树最大深度 | 3 ~ 20 |
| `min_samples_split` | 分裂所需最少样本 | 2 ~ 15 |
| `min_samples_leaf` | 叶子最少样本数 | 1 ~ 8 |
| `max_features` | 每次分裂看多少比例的特征 | 0.3 ~ 0.7 |

### 3.5 本项目表现

R²=0.86，比单棵决策树提升了 7 个百分点。没有数学创新，纯粹是"多棵树的平均"带来的统计收益。

---

## 4. XGBoost

### 4.1 从随机森林到 Boosting：思维方式的转变

随机森林：**N 个专家独立给出意见，取平均。**

```python
# Bagging 思维
opinions = []
for tree in forest:           # 并行训练，互不依赖
    opinions.append(tree.predict(x))
result = mean(opinions)       # 直接平均
```

XGBoost：**每个新专家专门修正前面所有人的错误。**

```python
# Boosting 思维
prediction = initial_guess    # 最开始猜平均值
for iteration in range(N):    # 串行训练，每棵新树依赖前面的结果
    residual = y_true - prediction               # 当前的误差
    new_tree = train_tree_to_predict(residual)   # 训练一棵树，目标是修正误差
    prediction += learning_rate * new_tree(x)    # 逐步修正
```

用编程中的迭代优化来理解：

```python
# 类比：调试一个复杂函数
def my_function(x):
    return some_complex_logic(x)  # 有 bug，输出不对

# 随机森林的思路：重写 200 个版本，每个有点不同，取平均
# Boosting 的思路：一步步修正
v1 = my_function(x)                    # 第一版
v2 = v1 + fix_for_case_A(x)            # 修正 A 类错误
v3 = v2 + fix_for_case_B(x)            # 修正 B 类错误
v4 = v3 + fix_for_edge_cases(x)        # 修正边界情况
# 最终版本 = v1 + v2修正 + v3修正 + v4修正
```

### 4.2 数学推导

设我们有一个已有模型 $\hat{y}^{(t-1)}$（前 t-1 轮的累积预测）。

第 t 轮要加一棵新树 $f_t$：

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

损失函数（MSE + 正则化）：

$$L^{(t)} = \sum_{i=1}^{n} (y_i - \hat{y}_i^{(t-1)} - f_t(x_i))^2 + \Omega(f_t)$$

其中 $\Omega(f_t)$ 是正则化项（惩罚树太复杂）。

**XGBoost 的独特之处**：用**泰勒展开的二阶近似**来指导分裂。

一阶导数（梯度）$g_i$ 告诉我们"误差在哪个方向"：

$$g_i = \frac{\partial (y_i - \hat{y}_i)^2}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)$$

二阶导数（Hessian）$h_i$ 告诉我们"误差变化有多快"：

$$h_i = \frac{\partial^2 (y_i - \hat{y}_i)^2}{\partial \hat{y}_i^2} = 2$$

经过推导，对于某个树结构（哪些叶子、每个叶子什么值），最优叶子权重为：

$$w_j^* = -\frac{\sum_{i \in \text{leaf}_j} g_i}{\sum_{i \in \text{leaf}_j} h_i + \lambda}$$

而分裂的增益公式为：

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R + \lambda}\right] - \gamma$$

其中 $G_L$、$G_R$ 分别是分裂后左右子节点内样本的梯度之和，$H_L$、$H_R$ 是 Hessian 之和。

**为什么要用二阶导数？** 一阶只告诉你"往哪走"，二阶告诉你"坡有多陡"——知道坡度就能更好判断在哪里停下来，避免走过头。这就是为什么 XGBoost 通常比传统的 Gradient Boosting 收敛更快。

### 4.3 正则化机制

XGBoost 有**多层防护**防止过拟合，这是它在小数据集上依然优秀的核心原因：

```python
# 1. learning rate —— 每棵树只迈一小步
prediction += 0.1 * new_tree(x)  # 而不是直接加 new_tree(x)

# 2. L1/L2 正则化 —— 惩罚叶子权重大小
#    叶子值不能太大，避免个别叶子主导预测

# 3. subsample —— 每次迭代随机用部分数据
#    类似随机森林的 Bootstrap

# 4. colsample_bytree —— 每次迭代随机用部分特征
#    也类似随机森林

# 5. max_depth —— 限制树的深度
#    太深的树 = 过拟合
```

### 4.4 本项目超参数

| 参数 | 含义 | 作用 |
|------|------|------|
| `n_estimators` | 树的数量（迭代次数） | 太少欠拟合，太多过拟合 |
| `max_depth` | 每棵树最大深度 | 控制单棵树的复杂度 |
| `learning_rate` | 每棵树的贡献比例 | **最关键**——小值 + 多棵树 = 稳定 |
| `subsample` | 每次迭代用多少比例的数据 | 防过拟合，增加随机性 |
| `colsample_bytree` | 每次迭代用多少比例的特征 | 同上 |
| `reg_lambda` | L2 正则化系数 | 惩罚大叶子权重 |
| `reg_alpha` | L1 正则化系数 | 产生稀疏权重 |

### 4.5 本项目表现

**R²=0.922**，是所有模型中的最佳。Boosting 在小规模结构化数据上的威力得到充分验证。

---

## 5. LightGBM

### 5.1 和 XGBoost 的本质相同

LightGBM 也是 Gradient Boosting。串行训练树，每棵新树修正前序误差。核心区别在于**如何加速**和**如何生长树**。

### 5.2 独有优化一：GOSS（Gradient-based One-Side Sampling）

传统 Boosting 每轮用全部数据。GOSS 的想法：

```
所有样本按梯度的绝对值排序
    │
    ├── 梯度大的样本 (前 a%)：全部保留
    │   这些是"难样本"——模型目前对它们预测很差，对新树的贡献大
    │
    └── 梯度小的样本 (剩余部分)：随机抽样 b%
        这些是"易样本"——模型已经很擅长了，用少量即可
        抽样时乘一个放大系数 (1-a)/b 来补偿信息损失
```

Python 模拟：

```python
def goss_sampling(data, gradients, a=0.2, b=0.1):
    n = len(data)
    # 按梯度绝对值排序
    sorted_idx = np.argsort(np.abs(gradients))[::-1]
    top_a = int(n * a)     # 保留梯度最大的前 20%
    rest_b = int(n * b)    # 从剩余 80% 中随机抽 10%

    keep = sorted_idx[:top_a]  # 高梯度样本
    sample = np.random.choice(sorted_idx[top_a:], size=rest_b, replace=False)
    return np.concatenate([keep, sample])
    # 采样后数据量 = 20% + 10% = 30% 的原始数据
    # 但几乎保留了全部信息量
```

### 5.3 独有优化二：Leaf-wise 生长

XGBoost 是 **level-wise**：每一层的所有叶子同时分裂：

```
          [根]              ← 层级 0
         /    \
       [A]    [B]           ← 层级 1（A 和 B 必须同时分裂）
      /  \    /  \
    [C]  [D] [E]  [F]       ← 层级 2（C、D、E、F 必须同时分裂）
```

LightGBM 是 **leaf-wise**：每次只选增益最大的那个叶子分裂：

```
迭代 1:   [根]              ← 产生 A, B
         /    \
       [A]    [B]

迭代 2: 选增益最大的叶子 → A
         [根]
        /    \
      [A]    [B]            ← 只有 A 分裂了
     /   \
   [C]   [D]

迭代 3: 选增益最大的叶子 → 可能是 B
         [根]
        /    \
      [A]    [B]            ← B 分裂了
     /   \   /  \
   [C]   [D][E] [F]
```

**影响**：同样的叶子数量，leaf-wise 的树更"深"更"不规则"，表达能力强于 level-wise，但也更容易过拟合。所以 LightGBM 必须严格控制 `num_leaves` 和 `max_depth`。

### 5.4 与 XGBoost 的功能对比

| 维度 | XGBoost | LightGBM |
|------|---------|----------|
| 树生长 | Level-wise（每层全分） | Leaf-wise（挑最好的叶子分） |
| 数据采样 | subsample 均匀采样 | GOSS（保留难样本） |
| 特征处理 | 预排序 + 分位点近似 | 直方图（离散化后找分割点） |
| 小数据 | 通常更好 | 可能过拟合 |
| 大数据 | 较慢 | 通常更快 |
| 参数数量 | 少 | 多（num_leaves 等额外参数） |

### 5.5 本项目表现

R²=0.88，略低于 XGBoost。原因：506 条数据对 leaf-wise 策略来说太少，不规则深树容易过拟合。在大数据集上（如百万行），LightGBM 的优势才会体现。

---

## 6. 综合对比

### 6.1 五个模型的演变逻辑

```
普通线性回归（数值不稳定，高相关特征无法处理）
    │
    ├─→ Ridge：加 L2 惩罚 → 稳定系数 → R²=0.79
    │     局限：只能学线性关系
    │
    └─→ 决策树：if/else 规则 → 非线性 → R²=0.79
          局限：单棵树方差大
             │
             ├─→ 随机森林：并行 200 棵树 + Bootstrap + 随机特征
             │        平均化消噪 → R²=0.86
             │
             └─→ Boosting：串行修正残差
                       │
                       ├─→ XGBoost：二阶优化 + 多层正则化 → R²=0.92
                       └─→ LightGBM：GOSS 采样 + Leaf-wise → R²=0.88
```

### 6.2 数值对比

| | Ridge | 决策树 | 随机森林 | XGBoost | LightGBM |
|---|---|---|---|---|---|
| **R²** | 0.79 | 0.79 | 0.86 | **0.92** | 0.88 |
| **MSE** | 15.60 | 15.53 | 11.52 | 7.17 | 8.78 |
| **MAE** | 2.46 | 2.77 | 2.08 | 1.89 | 2.11 |
| **训练时间** | <1s | <1s | ~30s | ~40s | ~70s |
| **CV 方差** | ±5.6 | ±6.2 | ±3.7 | ±2.1 | ±3.6 |
| **过拟合风险** | 低 | 高 | 中 | 低 | 中 |

### 6.3 选择建议

```
你的数据量 < 1000 条？
  └─ 是 → XGBoost（正则化强，小数据不易过拟合）

你的数据量 > 100000 条？
  └─ 是 → LightGBM（GOSS + Leaf-wise 速度快）

你需要的不是最高精度，而是可解释的模型？
  └─ 是 → Ridge 或浅决策树

特征之间有强相关性？
  └─ 是 → Ridge 或 XGBoost（能处理共线性）

你想发论文/打比赛追求最高分？
  └─ Stacking：3-4 个不同类型的模型做基模型，简单模型（如 Ridge）做顶层
```

### 6.4 你可能遇到的问题和解答

**Q：为什么不用神经网络？**
A：506 条数据训练神经网络会严重过拟合。深度模型需要大量数据（至少几万条）才能发挥优势。这也是为什么 Gradient Boosting 是表格数据的王者——它在这种规模下最适配。

**Q：超参数调优到底在调什么？**
A：本质上是"自动试错"。你定义参数范围，RandomizedSearchCV 随机采样 20-100 组参数组合，每组做 5 折交叉验证，最终选 CV 得分最高的那组。它不保证找到全局最优，但在实践中已经够好。

**Q：R² 0.92 还能再高吗？**
A：波士顿数据集的信息量天花板大约在 R²=0.93-0.95。506 条数据、13 个原始特征能承载的信息就这么多。想再高需要外部数据（如更多年的房价数据、更细粒度的地理信息）或者人工特征工程的突破。