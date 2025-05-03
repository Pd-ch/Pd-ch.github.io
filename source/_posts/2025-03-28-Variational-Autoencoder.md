---
key: 3
title: 变分自编码器
tags: [AI, math, unsupervised learning, VAE]
math: true
---

###### 说明：本文是对变分自编码器的学习总结

<!-- more -->

论文链接：[🔗](https://arxiv.org/abs/1312.6114)

本人的关于论文的粗略翻译[🔗](/2025/04/15/arxiv-1312-6114v11)

## 1. 当我们谈论 VAE 时，我们在谈论什么？

变分自编码器（Variational Autoencoder，VAE）是一种生成模型，它通过学习数据的潜在表示来生成新的数据样本。VAE 的主要目标是找到一个潜在变量的分布，使得原始数据可以被表示为这个分布的概率分布。

通常的 VAE 模型包括一个编码器（Encoder）和一个解码器（Decoder）。而在无监督方法中，我们仅需要编码器，而不需要解码器。

接下来让我们 dive into the details of VAE。


## 2. AE

### 一些概念

![](/assets/images/vae.png)

AE（Autoencoder）由两个主要组件组成：编码器$\mathbf{z}=f_{\phi}(\mathbf{x})$和解码器$\mathbf{x}=f_{\theta}(\mathbf{z})$。编码器将输入数据$\mathbf{x}$映射到一个潜在表示$\mathbf{z}$，而解码器将潜在表示$\mathbf{z}$映射回原始数据空间$\mathbf{x}$。

但是这种情况下因为我们没有对$\mathbf{z}$进行约束，所以潜在表示$\mathbf{z}$可能会非常自由，这可能导致潜在表示$\mathbf{z}$的分布非常复杂，无法很好的表示原始数据的分布。我们的数据是有限的，而潜在表示$\mathbf{z}$是无限的，decoder可能只对有限的$\mathbf{z}$有很好的响应，而对我们采样出来的$\mathbf{z}$生成效果不理想。


### 一些假设

原文的2.1 问题场景章节中，我们假设数据是由某个随机过程生成的，这个随机过程涉及一个未观测的连续变量 $\mathbf{z}$ ，这个变量的分布是未知的。该过程包括两个步骤：首先，从某个先验分布 $p_{\theta^*}$ 生成一个值 $\mathbf{z}^{(i)}$ ;然后，值 $\mathbf{x}^{(i)}$ 是从某个条件分布 $p_{\theta^*}(\mathbf{x}|\mathbf{z}^{(i)})$ 生成的。我们**假设**先验 $p_{\theta^*}(\mathbf{z})$ 和似然 $p_{\theta^*}(\mathbf{x}|\mathbf{z})$ 来自 $p_\theta(\mathbf{z})$ 和 $p_\theta(\mathbf{x}|\mathbf{z})$ 的参数分布族，并且他们的概率密度函数几乎可以在 $\theta$ 和 $\mathbf{z}$ 的任何地方上微分。此时我们仍然不知道真实变量 $\theta^*$ 和隐变量 $\mathbf{z}$ 的具体分布。

## 3. 一些参考资料

[机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)

[机器学习-白板推导系列-变分自编码器](https://www.bilibili.com/video/BV1aE411o7qd/?p=170)

[如何避免VAE后验坍塌?](https://zhuanlan.zhihu.com/p/389295612)

[科学空间-苏剑林-变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)

[科学空间-苏剑林-变分自编码器（二）：从贝叶斯观点出发](https://spaces.ac.cn/archives/5343)

[科学空间-苏剑林-变分自编码器（三）：这样做为什么能成？](https://spaces.ac.cn/archives/5383)

[科学空间-苏剑林-变分自编码器（四）：一步到位的聚类方案](https://spaces.ac.cn/archives/5887)

[科学空间-苏剑林-变分自编码器（五）：VAE + BN = 更好的VAE](https://spaces.ac.cn/archives/7381)