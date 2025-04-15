---
key: 3
title: 变分自编码器
tags: [AI, math, unsupervised learning, VAE]
math: true
---

###### 说明：本文是对变分自编码器的学习总结

<!-- more -->

论文链接：[🔗](https://arxiv.org/abs/1312.6114)
本人的关于论文的粗略翻译[🔗]()

## 1. 当我们谈论 VAE 时，我们在谈论什么？

变分自编码器（Variational Autoencoder，VAE）是一种生成模型，它通过学习数据的潜在表示来生成新的数据样本。VAE 的主要目标是找到一个潜在变量的分布，使得原始数据可以被表示为这个分布的概率分布。

![](/assets/images/vae.png)

通常的 VAE 模型包括一个编码器（Encoder）和一个解码器（Decoder）。而在无监督方法中，我们仅需要编码器，而不需要解码器。

接下来让我们 dive into the details of VAE。


## 2. 我的一点看法

### 要讲VAE,首先看看VAE以前的时代。
最初AE（Autoencoder）由两个主要组件组成：编码器$\mathbf{z}=f_{\phi}(\mathbf{x})$和解码器$\mathbf{x}=g_{\theta}(\mathbf{z})$。编码器的目标是将输入数据$\mathbf{x}$映射到一个低维的潜在表示$\mathbf{z}$，而解码器的目标是将潜在表示$\mathbf{z}$映射回原始数据空间$\mathbf{x}$。AE通常使用的Loss函数是$\ell=\|X-\tilde{X}\|^2$。这个Loss函数能够很直观的告诉我们重构出的数据和原始数据之间的差异。

## 3. 一些参考资料

[机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)

[机器学习-白板推导系列-变分自编码器](https://www.bilibili.com/video/BV1aE411o7qd/?p=170)

[如何避免VAE后验坍塌?](https://zhuanlan.zhihu.com/p/389295612)

[科学空间-苏剑林-变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)

[科学空间-苏剑林-变分自编码器（二）：从贝叶斯观点出发](https://spaces.ac.cn/archives/5343)

[科学空间-苏剑林-变分自编码器（三）：这样做为什么能成？](https://spaces.ac.cn/archives/5383)

[科学空间-苏剑林-变分自编码器（四）：一步到位的聚类方案](https://spaces.ac.cn/archives/5887)

[科学空间-苏剑林-变分自编码器（五）：VAE + BN = 更好的VAE](https://spaces.ac.cn/archives/7381)