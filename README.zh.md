<p align="center">
  <img src="doc/source/_static/img/logo.svg" width="150"><br />
</p>

# fairseq2: FAIR 序列建模工具包 2

[![Nightly](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml/badge.svg)](https://github.com/facebookresearch/fairseq2/actions/workflows/nightly.yaml)
[![PyPI version](https://img.shields.io/pypi/v/fairseq2)](https://pypi.org/project/fairseq2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](README.md)** | **[简体中文](README.zh.md)**

**文档：[稳定版](https://facebookresearch.github.io/fairseq2/stable)，[每夜版 (Nightly)](https://facebookresearch.github.io/fairseq2/nightly)** | **安装：[Linux](#在-linux-上安装)，[macOS](#在-macos-上安装)，[Windows](#在-windows-上安装)，[从源码安装](INSTALL_FROM_SOURCE.md)** | **贡献：[贡献指南](CONTRIBUTING.md)**

fairseq2 是一个序列建模工具包，旨在帮助研究人员为内容生成任务训练自定义模型。

### 近期基于 fairseq2 的 FAIR 研究成果
* [Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages](https://arxiv.org/abs/2511.09690)
* [Bridging Offline and Online Reinforcement Learning for LLMs](https://arxiv.org/pdf/2506.21495)
* [R.I.P.: Better Models by Survival of the Fittest Prompts](https://arxiv.org/abs/2501.18578)
* [Learning to reason for factuality](https://arxiv.org/abs/2508.05618)
* [Diverse Preference Optimization](https://arxiv.org/abs/2501.18101)
* [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/abs/2412.08821)
* [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187)

### fairseq2 与初代 fairseq 有何不同？
fairseq2 是一个从零开始构建的新项目，可以视为初代 [fairseq](https://github.com/facebookresearch/fairseq) 的全面重构（reboot），旨在提供简洁且模块化的 API。值得注意的是，它在设计理念上与前代截然不同——从单体式（monolithic）框架转变为可扩展、低侵入性的架构，让研究人员能够独立掌控自己的项目代码库。

> 由于 fairseq2 是一个完全独立的新项目，而非初代 fairseq 的增量更新，因此我们特意避免将其命名为 fairseq version 2，以体现其独特的独立身份。

## 核心特性
* 官方提供用于语言模型[指令微调 (Instruction Finetuning)](https://facebookresearch.github.io/fairseq2/stable/tutorials/end_to_end_fine_tuning.html)与[偏好优化 (Preference Optimization)](https://facebookresearch.github.io/fairseq2/stable/tutorials/preference_optimization.html)的标准 Recipe 配方
* 基于 DDP、FSDP 与张量并行（Tensor Parallelism）的分布式多 GPU、多节点[训练架构](https://facebookresearch.github.io/fairseq2/stable/basics/trainer.html)。支持 70B+ 参数规模模型。
* 原生支持 vLLM，并内置采样（Sampling）与束搜索（Beam Search）序列生成器
* 借助 setuptools [扩展机制](https://facebookresearch.github.io/fairseq2/stable/basics/runtime_extensions.html)实现高扩展性。无需 Fork 或拉取独立分支即可轻松注册新模型、优化器、学习率调度器与训练器单元。
* 现代 PyTorch 生态工具链集成。原生采用组合式能力（如 `torch.compile`）、PyTorch FSDP 及其他相关前沿特性
* 基于流式的高吞吐[数据管线 API](https://facebookresearch.github.io/fairseq2/stable/basics/data_pipeline.html)（由 C++ 编写），支持音频以及（即将推出的）视频解码
* 编程式[资产卡片 (Asset Cards)](https://facebookresearch.github.io/fairseq2/stable/basics/assets.html)，为模型、数据集与分词器提供受版本控制的规范化访问
* 基于内置 *structured* API 构建灵活且具备确定性的配置系统

## 快速入门
访问我们的[官方文档网站](https://facebookresearch.github.io/fairseq2/stable/)深入了解 fairseq2。

## 在 Linux 上安装

### 系统依赖
fairseq2 依赖 [libsndfile](https://github.com/libsndfile/libsndfile)，大多数 Linux 发行版均可通过系统包管理器安装。在基于 Ubuntu 的系统上，请运行：

```sh
sudo apt install libsndfile1
```

类似地，在 Fedora 上请运行：

```sh
sudo dnf install libsndfile
```

对于其他 Linux 发行版，请参阅其官方文档以了解如何安装软件包。

### pip 安装
要在 Linux x86-64 上安装 fairseq2，请运行：

```sh
pip install fairseq2
```

该命令将安装与 PyPI 上发布的 PyTorch 兼容的 fairseq2 版本。

目前，我们暂不为基于 ARM 架构的系统（如树莓派或 NVIDIA Jetson）提供预构建轮子。请参阅[从源码安装](INSTALL_FROM_SOURCE.md)了解如何在这些系统上构建与安装 fairseq2。

### 特定版本与变体 (Variants)
除了 PyPI 之外，FAIR 的官方软件包仓库还针对不同的 PyTorch 和 CUDA 版本提供了预构建轮子包。下表展示了当前支持的版本组合矩阵：

<table>
  <thead>
    <th>fairseq2</th>
    <th>PyTorch</th>
    <th>Python</th>
    <th>Variant*</th>
    <th>Arch</th>
  </thead>
  <tbody>
    <tr>
      <td rowspan=5><code>HEAD</code></td>
      <td><code>2.13.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu130</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.11.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.9.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td rowspan=3><code>0.8</code></td>
      <td><code>2.9.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td rowspan=3><code>0.7</code></td>
      <td><code>2.9.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td rowspan=3><code>0.6</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu126</code>, <code>cu128</code></td>
      <td><code>x86_64</code></td>
    </tr>
    <tr>
      <td><code>2.6.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>cpu</code>, <code>cu124</code></td>
      <td><code>x86_64</code></td>
    </tr>
  </tbody>
</table>

*\* cuXYZ 指代 CUDA XY.Z（例如 cu118 代表 CUDA 11.8）*

要安装特定的组合，请先按照 [pytorch.org](https://pytorch.org/get-started/locally) 上的指引安装目标 PyTorch 版本，然后运行以下命令（以 PyTorch `2.9.1` 和变体 `cu126` 为例）：

```sh
pip install fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.1/cu126
```

> [!WARNING]
> fairseq2 依赖 PyTorch 的 C++ API，而各 PyTorch 发行版之间不保证 API/ABI 兼容性。这意味着**你必须安装与本地 PyTorch 版本完全匹配的 fairseq2 变体**。否则可能会出现进程立即崩溃或偶发段错误（segmentation fault）等异常。出于同样的原因，若升级了 PyTorch 版本，也必须同步升级 fairseq2 安装。

### 每夜构建版 (Nightlies)
针对 Linux，我们在 FAIR 软件包仓库中也提供了每夜构建版本（Nightly）。支持的变体与上方*特定版本与变体*表格完全一致。安装所需版本的 PyTorch 后，可运行以下命令安装相应的 Nightly 包（以 PyTorch `2.9.1` 和变体 `cu128` 为例）：

```sh
pip install fairseq2\
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.9.1/cu128
```

## 在 macOS 上安装

### 系统依赖
fairseq2 依赖 [libsndfile](https://github.com/libsndfile/libsndfile)，可通过 Homebrew 安装：

```sh
brew install libsndfile
```

### pip 安装
要在基于 ARM64（即 Apple Silicon）架构的 Mac 上安装 fairseq2，请运行：

```sh
pip install fairseq2
```

该命令将安装与 PyPI 上发布的 PyTorch 兼容的 fairseq2 版本。

目前，我们暂不为基于 Intel 架构的 Mac 提供预构建轮子。请参阅[从源码安装](INSTALL_FROM_SOURCE.md)了解如何在 Intel Mac 上进行构建与安装。

### 特定版本与变体 (Variants)
除了 PyPI 之外，FAIR 软件包仓库中也针对不同 PyTorch 版本提供了预构建轮子包。下表展示了当前支持的组合：

<table>
  <thead>
    <th>fairseq2</th>
    <th>PyTorch</th>
    <th>Python</th>
    <th>Arch</th>
  </thead>
  <tbody>
    <tr>
      <td><code>HEAD</code></td>
      <td><code>2.9.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td><code>0.8</code></td>
      <td><code>2.9.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td><code>0.7</code></td>
      <td><code>2.9.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td rowspan=2><code>0.6</code></td>
      <td><code>2.8.0</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
    <tr>
      <td><code>2.7.1</code></td>
      <td><code>&gt;=3.10</code>, <code>&lt;=3.12</code></td>
      <td><code>arm64</code></td>
    </tr>
  </tbody>
</table>

要安装特定的组合，请先按照 [pytorch.org](https://pytorch.org/get-started/locally) 上的指引安装目标 PyTorch 版本，然后运行以下命令（以 PyTorch `2.9.1` 为例）：

```sh
pip install fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.1/cpu
```

> [!WARNING]
> fairseq2 依赖 PyTorch 的 C++ API，而各 PyTorch 发行版之间不保证 API/ABI 兼容性。这意味着**你必须安装与本地 PyTorch 版本完全匹配的 fairseq2 变体**。否则可能会出现进程立即崩溃或偶发段错误（segmentation fault）等异常。出于同样的原因，若升级了 PyTorch 版本，也必须同步升级 fairseq2 安装。

### 每夜构建版 (Nightlies)
针对 macOS，我们在 FAIR 软件包仓库中同样提供了每夜构建版本。支持的变体与上方 macOS 矩阵完全一致。安装所需版本的 PyTorch 后，可运行以下命令安装对应的 Nightly 包（以 PyTorch `2.9.1` 为例）：

```sh
pip install fairseq2\
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.9.1/cpu
```

## 在 Windows 上安装
fairseq2 原生不支持 Windows，且在可预见的未来暂无支持计划。不过，你可以通过[适用于 Linux 的 Windows 子系统](https://learn.microsoft.com/zh-cn/windows/wsl/about)（即 WSL）以及 WSL 2 中引入的完整 CUDA 支持来使用 fairseq2。基于 WSL 的安装步骤请参阅[在 Linux 上安装](#在-linux-上安装)小节。

## 从源码安装
请参阅[从源码安装指南](INSTALL_FROM_SOURCE.md)。

## 贡献代码
我们热烈欢迎对 fairseq2 的任何贡献！请参阅[贡献指南](CONTRIBUTING.md)了解如何格式化代码、运行测试并提交你的 Pull Request。

## 引用 fairseq2
如果你在研究中使用了 fairseq2 并希望引用它，请使用以下 BibTeX 条目：

```bibtex
@software{balioglu2023fairseq2,
  author = {Can Balioglu and Alexander Erben and Martin Gleize and Artyom Kozhevnikov and Ilia Kulikov and Julien Yao},
  title = {fairseq2},
  url = {http://github.com/facebookresearch/fairseq2},
  year = {2023},
}
```

## 开源协议
本项目采用 MIT 许可协议，详情请参阅 [LICENSE](LICENSE) 文件。

---

> **说明**：本文档为基于英文原文的中文翻译版本。如翻译内容与英文原文存在不一致之处，请以 [英文原版 (README.md)](README.md) 为准。
