# 线性回归：Loss → 闭式解

## 1. 线性回归本质是用算法获取参数吗？

**是的，但不一定是"迭代算法"。** 线性回归的目标是找到一组参数（系数）$\theta$，让预测值 $\hat{y} = X\theta$ 尽可能接近真实值 $y$。

获取参数有两条路：

| 方式 | 代表 | 特点 |
|------|------|------|
| **闭式解（解析解）** | 正规方程 Normal Equation | 一步到位，数学推导直接出结果 |
| **迭代优化** | 梯度下降 Gradient Descent | 一步步逼近，适合大数据/高维场景 |

所以更准确地说：**线性回归的本质是在假设空间（所有线性函数）中，按最小化均方误差的准则，找到最优参数。** 这个准则（MSE Loss）天然存在一个闭式解。

---

## 2. Loss 怎么推导到闭式解？

### 第一步：写出 Loss 的矩阵形式

$$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

设 $\hat{y} = X\theta$，其中 $X$ 是 $n \times d$ 的特征矩阵，$\theta$ 是 $d \times 1$ 的参数向量。那么：

$$J(\theta) = \frac{1}{n} (y - X\theta)^T (y - X\theta)$$

一句话概括：**Loss 是残差向量 $(y - X\theta)$ 的长度的平方（除以 n）。**

### 第二步：对 $\theta$ 求导，令导数为零

展开：

$$J(\theta) = \frac{1}{n} \left( y^T y - y^T X\theta - \theta^T X^T y + \theta^T X^T X \theta \right)$$

中间两项是标量的转置，相等（$y^T X\theta = (y^T X\theta)^T = \theta^T X^T y$），所以：

$$J(\theta) = \frac{1}{n} \left( y^T y - 2\theta^T X^T y + \theta^T X^T X \theta \right)$$

对 $\theta$ 求导：

$$\frac{\partial}{\partial \theta} (\theta^T A) = A, \quad \frac{\partial}{\partial \theta} (\theta^T A \theta) = 2A\theta \quad (\text{当 } A \text{ 对称})$$

代入得：

$$\frac{\partial J}{\partial \theta} = \frac{1}{n} \left( -2X^T y + 2X^T X \theta \right)$$

令导数 $= 0$：

$$-2X^T y + 2X^T X \theta = 0$$

$$X^T X \theta = X^T y$$

### 第三步：解出 $\theta$

$$\boxed{\theta = (X^T X)^{-1} X^T y}$$

这就是**正规方程（Normal Equation）**，也就是闭式解。

---

## 3. 直观理解

$(X^T X)^{-1} X^T$ 其实就是矩阵版"除法"——**伪逆（Moore-Penrose pseudoinverse）**，记作 $X^+$。

所以闭式解可以写成：

$$\theta = X^+ y$$

意思是：**参数 = 特征的伪逆 × 目标值**。你在"解"一个方程组 $X\theta = y$，但由于方程数（样本数 $n$）远大于未知数（特征数 $d$），没有精确解，所以用伪逆求最小二乘意义上的最优近似解。

---

## 4. 和本项目的关系

本项目用的是 **Ridge 回归**（加 L2 正则化），它的闭式解是：

$$\theta = (X^T X + \lambda I)^{-1} X^T y$$

多的那个 $\lambda I$ 保证了矩阵可逆（即使特征之间有共线性），同时对参数做了收缩。

而 XGBoost / LightGBM 这类树模型根本不是线性模型，不存在闭式解，只能用迭代优化。
