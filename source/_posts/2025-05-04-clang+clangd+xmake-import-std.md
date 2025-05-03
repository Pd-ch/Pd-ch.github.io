---
key: 3
title: "Clang+Clangd+Xmake使用import std"
tags: [C++, clang, clangd, xmake, std module]
math: true
---

###### 说明：本文介绍如何使用Clang+Clangd+Xmake启用C++-23新特性"mport std"。

<!-- more -->

## 1. 环境配置

我个人使用的环境是：
- Debian unstable
- visual studio code
- Clang 20。1.5
- Clangd 20.1.5
- Xmake 2.9.9

### 1.1 安装llvm全家桶

根据[apt.llvm.org](https://apt.llvm.org/)的说明，添加llvm源：

``` bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20 all
```

**注意**：需要有对应的libc++-20-dev包。如果有任何疑问可以使用apt policy libc++-20-dev查看。
**警告**：如果clang，clangd，与libc++-dev版本对不上，clangd就不能正常工作。

### 1.2 安装xmake

根据[xmake.io](https://xmake.io/#/zh-cn/guide/installation)的说明进行安装：

``` bash
wget https://xmake.io/shget.text -O - | bash
```

### 1.3 VSCode + Clangd + XMake配置

可以参考[[万字长文]Visual Studio Code 配置 C/C++ 开发环境的最佳实践(VSCode + Clangd + XMake)](https://zhuanlan.zhihu.com/p/398790625)。

注意
```
{
  "clangd.arguments": ["--compile-commands-dir=${workspaceFolder}/.vscode"]
}
```
需要设置，否则clangd无法找到编译命令。

## 2. 启用import std

### 2.1 配置xmake

在xmake.lua中添加：
``` lua
add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "./.vscode" })

set_plat("linux")
set_toolchains("clang")
set_runtimes("c++_static")
set_config("sdk", "/usr/lib/llvm-20/")

target("stdmodules")
    set_kind("binary")
    add_files("*.cpp")
    set_languages("c++23")
    set_policy("build.c++.modules", true)
target_end()
```

编写一个简单的程序：
``` cpp
import std;

auto main() -> int {
    std::println("Hello, World!");
    return 0;
}
```

此时你的项目结构应该是这样的：
```
├── xmake.lua
└── main.cpp
```

编译：
``` bash
xmake
```

产物位于**build/build/linux/x86_64/release/**目录下。

## enjoy it!