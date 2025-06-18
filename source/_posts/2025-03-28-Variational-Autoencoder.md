---
key: 3
title: 变分自编码器
tags: [AI, math, unsupervised learning, VAE]
math: true
---

###### 说明：作者并非深度学习方面的专家，如有错误欢迎讨论

<!-- more -->

论文链接：[🔗](https://papers.cool/arxiv/1312.6114)

本人的关于论文的粗略翻译[🔗](/2025/04/15/arxiv-1312-6114v11)

由于我个人是做自监督相关的，故本文仅介绍自监督(self-supervised learning,SSL)方面VAE提供了什么。
~~其实在自监督中我们仅仅用到了VAE，没有用到decoder~~

## 1. 背景

在自监督学习中，能否利用无标注数据学习数据的隐式表示是很重要的，隐式数据表示的好坏直接决定了后续下游任务表现的上限。

## 2. VAE提供了什么

变分自编码器（VAE）在自监督学习中的应用主要通过利用无标注数据学习数据的隐式表示，结合生成模型与自监督任务的优势。VAE的编码器将输入数据压缩为低维隐变量分布（通常假设为高斯分布），通过解码器重构数据。隐空间（latent space）的分布特性使其适合作为下游任务的通用特征。

VAE是如何将输入压缩到隐空间的呢？或者说它是怎样建模了一个足够好的隐空间？

对于通常的AutoEncoder，它构造了映射：$f:X \to Z$ ，表示输入 $X$ 经过网络 $f$ 压缩为 $Z$ 。但是对于一张普通的1080p RGB 图片来讲，所有可能性高达$1920\times1080\times3\times2^8$。传统网络擅长构建一对一映射，在这里就会“水土不服”。那么，有什么办法解决这个问题吗？

VAE选择了对隐空间进行建模，假设后验 $p(z|x)$ 服从高斯分布。

但是这带来了问题：

- **边际似然难以处理**：  
  边际似然$p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})d\mathbf{z}$的积分难以处理，因此我们无法评估或区分边际似然。

- **真实后验密度难以处理**：  
  真实后验密度$$p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})}$$难以处理，因此不能使用期望最大算法。

VAE 通过编码器神经网络实现数据压缩：
1. **输入映射**  
   输入数据 \( \mathbf{x} \)（如图像）通过编码器网络 \( q_\phi(\mathbf{z}|\mathbf{x}) \) 映射到**隐变量的概率分布参数**（通常是高斯分布）：
   \[
   \mu_\phi(\mathbf{x}), \sigma_\phi(\mathbf{x}) = \text{Encoder}_\phi(\mathbf{x})
   \]
   - \(\mu\)：隐变量分布的均值向量  
   - \(\sigma\)：标准差向量（决定不确定性）

2. **概率采样（重参数化技巧）**  
   为避免随机采样不可导，使用重参数化技巧生成隐变量 \(\mathbf{z}\)：  
   \[
   \mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
   \]
   - \(\odot\)：逐元素乘法  
   - \(\epsilon\)：从标准正态分布采样的噪声  
   > **意义**：将随机性分离为可导运算，使梯度可回传。

## 3. VAE 的数学推导
VAE 将输入数据压缩到隐空间并建模高质量隐空间的过程，本质是通过**概率编码器 + 隐变量分布约束 + 生成式重构**的联合优化实现的。以下是其核心机制的逐步解析：

#### **隐空间的结构化：KL散度的正则化作用**

VAE 的核心创新是通过 **KL 散度约束**使隐空间具备良好结构：

1. **目标函数：ELBO 分解**  
   VAE 优化证据下界（ELBO）：
   \[
   \mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]}_{\text{重构损失}} - \beta \cdot \underbrace{D_{\text{KL}} \left( q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}) \right)}_{\text{正则化项}}
   \]
   - \( p(\mathbf{z}) \)：先验分布（标准正态分布 \(\mathcal{N}(0, \mathbf{I})\)）

2. **KL 散度的作用**  
   \[
   D_{\text{KL}} \left( \mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, 1) \right) = \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
   \]
   - **压缩信息**：迫使隐变量分布靠近标准正态分布，避免编码器记住输入细节（防止过拟合）。  
   - **解耦诱导**：KL 项惩罚各维度间的相关性，鼓励隐变量维度独立（解耦表示）。  
   - **空间连续性**：隐空间被约束为平滑的流形，支持隐变量插值（如 \(\mathbf{z}_1 \to \mathbf{z}_2\) 生成过渡样本）。

#### **解码器：从隐空间重建数据**
解码器 \( p_\theta(\mathbf{x}|\mathbf{z}) \) 将隐变量 \(\mathbf{z}\) 映射回数据空间：
1. **生成过程**  
   \[
   \hat{\mathbf{x}} = \text{Decoder}_\theta(\mathbf{z})
   \]
   - 目标：最小化重建损失（如 MSE 或交叉熵）。

2. **重建损失的双重角色**  
   - 迫使 \(\mathbf{z}\) 保留足够信息以精确重建输入。  
   - 与 KL 散度博弈：重构损失希望 \(\mathbf{z}\) 包含更多信息，而 KL 散度希望 \(\mathbf{z}\) 更接近简单先验分布。


### 四、**隐空间“足够好”的关键设计**
#### 1. **连续性（Continuity）**
   - **机制**：KL 散度强制隐变量分布覆盖整个标准正态空间，而非孤立点簇。  
   - **效果**：隐空间中相邻点解码后语义相似（如人脸隐空间中微笑程度连续变化）。

#### 2. **解耦性（Disentanglement）**
   - **机制**：KL 散度隐式鼓励隐变量维度独立。显式方法如 \(\beta\)-VAE 进一步强化：  
     \[
     \mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p(\mathbf{x}|\mathbf{z})] - \beta D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))
     \]
   - **效果**：单个隐变量维度控制单一语义特征（如发型、光照）。

#### 3. **鲁棒性（Robustness）**
   - **机制**：概率采样使模型对输入噪声不敏感（方差 \(\sigma\) 建模不确定性）。  
   - **效果**：对遮挡或噪声数据仍能生成合理隐变量。

VAE 通过**概率编码器建模隐变量分布**，并利用 **KL 散度约束**将隐空间结构化，使其具备连续性、解耦性和鲁棒性。解码器的重建需求则确保隐变量保留足够信息。这种概率框架下的生成-推断平衡，使 VAE 的隐空间不仅是高效的数据压缩表示，更是可解释、可操控的语义空间，为自监督学习和生成任务奠定基础。

## 5. 一些参考资料

[机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)

[机器学习-白板推导系列-变分自编码器](https://www.bilibili.com/video/BV1aE411o7qd/?p=170)

[如何避免VAE后验坍塌?](https://zhuanlan.zhihu.com/p/389295612)

[科学空间-苏剑林-变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)

[科学空间-苏剑林-变分自编码器（二）：从贝叶斯观点出发](https://spaces.ac.cn/archives/5343)

[科学空间-苏剑林-变分自编码器（三）：这样做为什么能成？](https://spaces.ac.cn/archives/5383)

[科学空间-苏剑林-变分自编码器（四）：一步到位的聚类方案](https://spaces.ac.cn/archives/5887)

[科学空间-苏剑林-变分自编码器（五）：VAE + BN = 更好的VAE](https://spaces.ac.cn/archives/7381)