---
key: 3
title: 变分自编码器
math: true
tags: [AI, math, VAE]
---

###### 说明：本文是对变分自编码器的学习总结

<!-- more -->

论文链接：[🔗](https://arxiv.org/abs/1312.6114)

## 1. 当我们谈论 VAE 时，我们在谈论什么？

变分自编码器（Variational Autoencoder，VAE）是一种生成模型，它通过学习数据的潜在表示来生成新的数据样本。VAE 的主要目标是找到一个潜在变量的分布，使得原始数据可以被表示为这个分布的概率分布。

通常的 VAE 模型包括一个编码器（Encoder）和一个解码器（Decoder）。而在无监督方法中，我们仅需要编码器，而不需要解码器。

接下来让我们 dive into the details of VAE。论文的 introduction 部分我们略过，直接看 method 部分。


## 2. Method

![](/assets/images/vae.png)

本节中的策略可用于为具有连续潜在变量(隐变量)的各种有向图形模型推导下限估计器（随机目标函数）。我们在这里将自己限制在常见情况下：

- 我们有一个 i.i.d. 数据集，每个数据点都有潜在变量。
- 我们希望对（全局）参数执行最大似然（ML）或最大后验（MAP）推理，以及对潜在变量进行变分推理。

这里的潜在变量是什么就很值得玩味了。

### 2.1 问题场景(Problem Scenario)

让我们考虑一些数据集$\mathbf{X} = \{\mathbf{x}^{(i)}\}_{i=1}^N$，它由一些连续或离散变量$x$的$N$个i.i.d.样本组成。

我们假设数据是由某个随机过程生成的，涉及一个未观察到的连续随机变量$z$。该过程包括以下两个步骤：

1.从某个先验分布$p_{\boldsymbol{\theta}^*}(\mathbf{z})$生成一个值$\mathbf{z}^{(i)}$  
2.值$\mathbf{x}^{(i)}$是从某个条件分布$p_{\boldsymbol{\theta}^*}(\mathbf{x}|\mathbf{z})$生成的

我们假设先验$p_{\boldsymbol{\theta}^* }(\mathbf{z})$和似然$p_{\boldsymbol{\theta}^*}(\mathbf{x}|\mathbf{z})$来自分布$p_{\boldsymbol{\theta}}(\mathbf{z})$和$p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$的参数族，并且它们的概率密度函数（PDF）几乎在$\boldsymbol{\theta}$和$\mathbf{z}$的任何地方都是可微分的。

#### 隐藏的挑战

然而，不幸的是，这个过程的很多内容都隐藏在我们看不见的地方：

- 我们不知道真实参数$\boldsymbol{\theta}^*$。
- 我们也不知道潜在变量$\mathbf{z}^{(i)}$的值。

非常重要的是，我们没有对边际或后验概率做出常见的简化假设。相反，我们在这里对一种通用算法感兴趣，该算法甚至可以在以下情况下有效工作：

#### 难解性

- **边际似然难以处理**：  
  边际似然$p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})d\mathbf{z}$的积分难以处理，因此我们无法评估或区分边际似然。

- **真实后验密度难以处理**：  
  真实后验密度$$p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})}$$难以处理，因此不能使用期望最大算法。

- **积分复杂性**：  
  任何合理的均场变分贝叶斯推断算法所需的积分也难以处理。这些难解性很常见，出现在中等复杂似然函数$p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$的情况下，例如具有非线性隐藏层的神经网络。

#### 数据集规模

- **数据集过大**：  
  数据太多，批量优化成本太高。我们希望使用小批量甚至单个数据点进行参数更新。

- **基于采样的解决方案过慢**：  
  基于采样的解决方案（例如蒙特卡罗期望最大算法）通常太慢，因为它通常涉及每个数据点昂贵的采样循环。

#### 相关问题与解决方案

我们对上述场景中的三个相关问题感兴趣，并提出了解决方案：

1. **参数$\boldsymbol{\theta}$的有效近似最大似然估计或最大后验估计估计**：  
   参数本身可能很有趣，例如，如果我们正在分析一些自然过程。它们还允许我们模拟隐藏的随机过程并生成类似于真实数据的人工数据。

2. **潜在变量$\mathbf{z}$的有效近似后验推断**：  
   对于参数$\boldsymbol{\theta}$的选择，给定观测值$\mathbf{x}$，潜在变量$\mathbf{z}$的有效近似后验推断。这对于编码或数据表示任务非常有用。

3. **变量$\mathbf{x}$的有效近似边际推理**：  
   这使我们能够执行需要通过先验$\mathbf{x}$的各种推理任务。计算机视觉中的常见应用包括图像去噪、修复和超分辨率。
   
为了解决上述问题，让我们引入一个识别模型$q_{\phi}(z|x)$：这是对难以处理的真实后验分布$p_{\theta}(z|x)$的一种近似。需要注意的是，与均值场变分推断中的近似后验不同，该模型不要求具有因子分解形式，其参数$\phi$也不是通过闭式期望计算得到的。相反，我们将提出一种方法，**使识别模型参数$\phi$能够与生成模型参数$\theta$被联合学习**。

从编码理论的视角来看，未观测变量$z$可以解释为潜在表示或编码。因此，在本文中，我们将识别模型$q_{\phi}(z|x)$称为概率**编码器**——当给定数据点$x$时，它会生成一个关于编码$z$可能取值的分布（例如高斯分布），该编码能够生成数据点$x$。类似地，我们将$p_{\theta}(x|z)$称为概率**解码器**——当给定编码$z$时，它会生成一个关于对应数据点$x$可能取值的分布。

### 2.2 变分界(The variational bound)

边际似然由各个数据点的边际似然之和构成$\log p_{\theta}(x^{(1)}, \cdots, x^{(N)}) = \sum_{i=1}^{N} \log p_{\theta}(x^{(i)})$，其中每个项均可重写为：
$$\begin{align} & 
    \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})=D_{KL}(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})||p_{\boldsymbol{\theta}}
    (\mathbf{z}|\mathbf{x}^{(i)}))+\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})\tag{1}
\end{align}$$

第一项的右侧是近似后验分布与真实后验分布之间的KL散度。由于该KL散度非负，第二项的右侧$\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})$被称为数据点 $i$ 边缘似然的（变分）下界，其表达式可表示为：
$$
    \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\geq\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})=\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}\left[-\log q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})+\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})\right]\tag{2}
$$

也可以写成如下形式：
$$
    \mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})=-D_{KL}(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})||p_{\boldsymbol{\theta}}(\mathbf{z}))+\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}|\mathbf{z})\right]\tag{3}
$$

我们需要对下界 $\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)})$ 关于变分参数 $\phi$ 和生成参数 $\theta$ 进行微分和优化。然而，下界关于 $\phi$ 的梯度计算存在一定问题。针对此类问题的常规（朴素）蒙特卡洛梯度估计器为：  

$$
    \nabla_{\boldsymbol{\phi}}\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}\left[f(\mathbf{z})\right]=\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}\left[f(\mathbf{z})\nabla_{q_{\boldsymbol{\phi}}(\mathbf{z})}\log q_{\boldsymbol{\phi}}(\mathbf{z})\right]\simeq\frac{1}{L}\sum_{l=1}^{L}f(\mathbf{z})\nabla_{q_{\boldsymbol{\phi}}(\mathbf{z}^{(l)})}\log q_{\boldsymbol{\phi}}(\mathbf{z}^{(l)})
$$

其中 $\mathbf{z}^{(l)} \sim q_\phi(\mathbf{z} | \mathbf{x}^{(i)})$。  

该梯度估计器的方差极高，因此在实际应用中难以有效使用。  

### 2.3 SGVB估计器（随机梯度变分贝叶斯估计器，Stochastic Gradient Variational Bayes 和 AEVB算法（自动编码变分贝叶斯，Auto-Encoding Variational Bayes 

本节我们将介绍一种针对变分下界及其参数导数的实用估计方法。我们采用条件近似后验分布$q_\phi(\mathbf{z} | \mathbf{x})$的形式，但需要注意的是，该技术同样适用于非条件形式$q_\phi(\mathbf{z})$（即不依赖于x的情况）。

在满足第2.4节所述的特定温和条件下，对于选定的近似后验分布$q_\phi(\mathbf{z} | \mathbf{x})$，我们可以通过一个可微变换$g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x})$对随机变量$\widetilde{\mathbf{z}}\sim q_{\phi}(\mathbf{z}|\mathbf{x})$进行重新参数化，其中$\epsilon$是一个辅助噪声变量：
$$
    \widetilde{\mathbf{z}}=g_\phi(\boldsymbol{\epsilon},\mathbf{x})\quad\mathrm{with~}\quad\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon})\tag{4}
$$

关于如何选择合适的分布 $p(\boldsymbol{\epsilon})$ 和函数 $g_\phi(\boldsymbol{\epsilon},\mathbf{x})$ 的一般性策略，请参阅第 2.4 节的内容。基于此，我们可以按以下方式构建关于 $q_{\phi}(\mathbf{z}|\mathbf{x})$ 的期望函数 $f(z)$ 的蒙特卡洛估计：
$$
    \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})}\left[f(\mathbf{z})\right]=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[f(g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x}^{(i)}))\right]\simeq\frac{1}{L}\sum_{l=1}^{L}f(g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}^{(l)},\mathbf{x}^{(i)}))\quad\mathrm{where}\quad\boldsymbol{\epsilon}^{(l)}\sim p(\boldsymbol{\epsilon})\tag{5}
$$

我们运用这一技术处理变分下界（公式(2)），由此得到通用的随机梯度变分贝叶斯（SGVB）估计量 $\widetilde{\mathcal{L}}^A(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})\simeq\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})$，其表达式为：
$$\begin{aligned}
 & \widetilde{\mathcal{L}}^A(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})=\frac{1}{L}\sum_{l=1}^L\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)},\mathbf{z}^{(i,l)})-\log q_{\boldsymbol{\phi}}(\mathbf{z}^{(i,l)}|\mathbf{x}^{(i)}) \\
 & \mathrm{where}\quad\mathbf{z}^{(i,l)}=g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}^{(i,l)},\mathbf{x}^{(i)})\quad\mathrm{and}\quad\boldsymbol{\epsilon}^{(l)}\sim p(\boldsymbol{\epsilon})
\end{aligned}\tag{6}
$$


**Algorithm 1: Minibatch Version of Auto-Encoding VB (AEVB) Algorithm**  
*(Settings: \(M = 100\) and \(L = 1\) in experiments.)*

1. **Initialize Parameters**  
   \(\theta, \phi \leftarrow\) Initialize parameters.
2. **Repeat until convergence**  
   1. Sample a minibatch \(X^M\) of \(M\) datapoints (drawn from the full dataset).  
   2. Sample noise \(\epsilon\) from the distribution \(p(\epsilon)\).  
   3. Compute the gradients:  
      \[
      g \leftarrow \nabla_{\theta, \phi} \hat{L}^M (\theta, \phi; X^M, \epsilon)
      \]
   4. Update parameters \(\theta, \phi\) using the gradients \(g\) (e.g., SGD or Adagrad [DHS10]).
3. **Return**  
   \(\theta, \phi\) upon convergence.

**test**

$$
\begin{algorithm}[!h]
    \caption{algorithm of SUM}
    \label{alg:AOA}
    \renewcommand{\algorithmicrequire}{\textbf{Input:}}
    \renewcommand{\algorithmicensure}{\textbf{Output:}}
    \begin{algorithmic}[1]
        \REQUIRE $A$, $B$, $C$  %%input
        \ENSURE EEEEE    %%output
        
        \STATE  AAAAA
        \WHILE{$A=B$}
            \STATE BBBBB
        \ENDWHILE
        
        \FOR{each $i \in [1,10]$}
            \IF {$C = 0$}
                \STATE CCCCC
            \ELSE
                \STATE DDDDD
            \ENDIF
        \ENDFOR
        
        \RETURN EEEEE
    \end{algorithmic}
\end{algorithm}
$$