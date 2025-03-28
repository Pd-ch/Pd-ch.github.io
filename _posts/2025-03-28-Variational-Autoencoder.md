---
layout: post
title: "变分自编码器"
date:   2025-3-28
tags: [AI, math， VAE]
comments: true
author: Pd.ch
---

###### 说明：本文是对变分自编码器的学习总结

<!-- more -->

## 当我们谈论VAE时，我们在谈论什么？
变分自编码器（Variational Autoencoder，VAE）是一种生成模型，它通过学习数据的潜在表示来生成新的数据样本。VAE的主要目标是找到一个潜在变量的分布，使得原始数据可以被表示为这个分布的概率分布。

### **一、核心思想**
1. **自编码器（Autoencoder）的扩展**  
   - 传统自编码器由编码器（Encoder）和解码器（Decoder）组成：  
     - 编码器将输入数据压缩为低维隐变量（Latent Variable）。  
     - 解码器从隐变量重构输入数据。  
   - **VAE 的改进**：将隐变量建模为概率分布（如高斯分布），而非固定值，从而支持生成新数据。

2. **变分推断（Variational Inference）**  
   - 核心目标：近似真实的后验分布 \( p(z|x) \)（隐变量 \( z \) 的条件分布）。  
   - 通过优化变分下界（Evidence Lower Bound, ELBO），用可学习的分布 \( q(z|x) \) 逼近真实后验。
