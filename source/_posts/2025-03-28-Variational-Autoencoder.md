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

VAE选择了对隐空间进行建模，假设后验 $p(z|x)$ 服从高斯分布，


## 3. 一些参考资料

[机器学习方法—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)

[机器学习-白板推导系列-变分自编码器](https://www.bilibili.com/video/BV1aE411o7qd/?p=170)

[如何避免VAE后验坍塌?](https://zhuanlan.zhihu.com/p/389295612)

[科学空间-苏剑林-变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)

[科学空间-苏剑林-变分自编码器（二）：从贝叶斯观点出发](https://spaces.ac.cn/archives/5343)

[科学空间-苏剑林-变分自编码器（三）：这样做为什么能成？](https://spaces.ac.cn/archives/5383)

[科学空间-苏剑林-变分自编码器（四）：一步到位的聚类方案](https://spaces.ac.cn/archives/5887)

[科学空间-苏剑林-变分自编码器（五）：VAE + BN = 更好的VAE](https://spaces.ac.cn/archives/7381)