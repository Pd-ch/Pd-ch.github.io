---
layout: post
title: "简单介绍一下我本人的环境搭建"
date:   2024-9-29
tags: [env setup]
comments: true
author: Pd.ch
---

###### 说明：仅介绍本人主要使用的配置，留档以便后来可快速配置。

<!-- more -->

### 目录

- [Step 0. 系统启动盘制作](#step-0-系统启动盘制作)
- [Step 1. 终端美化与代理设置](#step-1-终端美化与代理设置)
- [Step 2. 安装NVIDIA驱动](#step-2-安装NVIDIA驱动)
- [Step 3. 安装python环境管理工具](#step-3-安装python环境管理工具)

## Step 0. 系统启动盘制作

简单来说，Ventoy是一个制作可启动U盘的开源工具。有了Ventoy你就无需反复地格式化U盘，你只需要把 ISO/WIM/IMG/VHD(x)/EFI 等类型的文件直接拷贝到U盘里面就可以启动了，无需其他操作。你可以一次性拷贝很多个不同类型的镜像文件，Ventoy 会在启动时显示一个菜单来供你进行选择。
下载地址:<https://www.ventoy.net/cn/download.html>
安装到U盘以后，只需要将iso镜像复制到U盘中即可。个人建议前往镜像站下载操作系统的iso镜像。

安装过程因人而异，故此不再赘述。个人比较喜欢Debian，在安装进行到分区这一步时，建议删除除引导分区以外的其它分区，我们不需要swap，以及这样做方便我们创建Btrfs主分区，将它的挂载点设置在“/”下。完成安装。(建议使用DVD镜像，避免安装时过多的等待)

接着进入系统，切换为root用户，为自己创建的用户添加sudo权限，并进行换源。此处不再赘述。值得一提的是，换源不再建议使用tuna(清华)源(用的人太多了，容易断流)。

至此，我们的系统就初步配置好了。

## Step 1. 终端美化与代理设置

终端个人推荐使用zsh，主题使用powerlevel10k，插件仅需zsh-autosuggestions与zsh-syntax-highlighting即可。
配置可参照:<https://www.haoyep.com/posts/zsh-config-oh-my-zsh/>

代理的设置建议如下
新建 **~/scripts/proxy.sh**,并在该脚本文件中复制以下代码,其中hostip和port按需更改:
~~~
#!/bin/sh
hostip=127.0.0.1
port=7890

PROXY_HTTP="http://${hostip}:${port}"

set_proxy(){
  export http_proxy="${PROXY_HTTP}"
  export HTTP_PROXY="${PROXY_HTTP}"

  export https_proxy="${PROXY_HTTP}"
  export HTTPS_proxy="${PROXY_HTTP}"

  export ALL_PROXY="${PROXY_SOCKS5}"
  export all_proxy=${PROXY_SOCKS5}

  git config --global http.https://github.com.proxy ${PROXY_HTTP}
  git config --global https.https://github.com.proxy ${PROXY_HTTP}

  echo "Proxy has been opened."
}

unset_proxy(){
  unset http_proxy
  unset HTTP_PROXY
  unset https_proxy
  unset HTTPS_PROXY
  unset ALL_PROXY
  unset all_proxy
  git config --global --unset http.https://github.com.proxy
  git config --global --unset https.https://github.com.proxy

  echo "Proxy has been closed."
}

test_setting(){
  echo "Host IP:" ${hostip}
  echo "Try to connect to Google..."
  resp=$(curl -I -s --connect-timeout 5 -m 5 -w "%{http_code}" -o /dev/null www.google.com)
  if [ ${resp} = 200 ]; then
    echo "Proxy setup succeeded!"
  else
    echo "Proxy setup failed!"
  fi
}

if [ "$1" = "set" ]
then
  set_proxy

elif [ "$1" = "unset" ]
then
  unset_proxy

elif [ "$1" = "test" ]
then
  test_setting
else
  echo "Unsupported arguments."
fi
~~~
在你的.zshrc或者.bashrc中添加**alias proxy="source ~/scripts/proxy.sh"**

使用时只需要在终端输入proxy set;proxy unset;proxy test.

至此,终端美化与代理设置就初步完成了

## Step 2. 安装NVIDIA驱动

你完全可以**sudo apt install nvidia-driver**来安装开源驱动
但是我更推荐安装闭源驱动，如果有内核更新，记得要**sudo apt install linux-headers-$(uname -r)**.

安装 apt GPG keyring 包，目的是获取 GPG 密钥
~~~
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
~~~

可以到此位置<https://developer.download.nvidia.com/compute/cuda/repos/>浏览具体发行版

apt 安装驱动
~~~
sudo apt update
sudo apt -y install nvidia-driver cuda-drivers
~~~

至此,你已经几乎完成了环境的搭建。

## Step 3. 安装python环境管理工具

这个看个人品味,我推荐使用miniconda,venv或者mamba.镜像站使用tuna或者bfsu.
在此笔者假设读者熟悉上面三种的任意一种
创建完虚拟的环境后,进入虚拟环境对pip进行换源.直接运行
~~~
pip install torch
~~~
如果在安装完torch后安装了大量nvidia*的包,那么就可以放心了,你安装的是pytorch with GPU

##### 你已经完成了深度学习环境搭建,立刻开始愉快的学习吧！