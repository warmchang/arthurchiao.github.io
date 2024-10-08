---
layout    : post
title     : "GPU Performance (Data Sheets) Quick Reference (2023)"
date      : 2023-10-25
lastupdate: 2024-05-19
categories: gpu
---

This post provides a concise reference for the performance of popular GPU
models from NVIDIA and Huawei/HiSilicon, primarily intended for personal use.

----

* TOC
{:toc}

----

# 1 Introduction

## 1.1 Naming convention of NVIDIA GPUs

The first letter in GPU model names denote their GPU architectures, with:

1. **<mark><code>T</code></mark>** for Turing;
1. **<mark><code>A</code></mark>** for Ampere;
1. **<mark><code>V</code></mark>** for Volta;
1. **<mark><code>H</code></mark>** for Hopper; 2022
1. **<mark><code>L</code></mark>** for Ada Lovelace;

# 2 Comparison of `L2/L4/T4/A10/V100`

|                    | L2               | L4               | T4               | A10                | A30        | V100 PCIe/SMX2 |
|:-------------------|:-----------------|:-----------------|:-----------------|:-------------------|:-----------|:--------|
| Designed for       | Data center      | Data center      |Data center       | (Desktop) **<mark>Graphics-intensive</mark>** workloads | Desktop | Data center |
| Year               | 2023             | 2023             | 2018             | 2020               |            | 2017         |
| Manufacturing      |                  |                  | 12nm             | 12nm               |            |              |
| Architecture       | Ada Lovelace     | Ada Lovelace     | Turing           | Ampere             | Ampere     | Volta        |
| Max Power          |                  | 72W              | 70 watts         | 150 watts          | 165 watts  | 250/300watts |
| GPU Mem            | 24GB GDDR6       | 24GB             | 16GB GDDR6       | 24GB GDDR6         | 24GB HBM2  | 16/32GB <mark>HBM2</mark> |
| GPU Mem BW         | 300 GB/s         | 300 GB/s         | 400 GB/s         | 600 GB/s           | **<mark><code>933GB/s</code></mark>**  | **<mark><code>900 GB/s</code></mark>** |
| Interconnect       | PCIe Gen4 64GB/s | PCIe Gen4 64GB/s | PCIe Gen3 32GB/s | PCIe Gen4 66 GB/s  | PCIe Gen4 64GB/s, NVLINK 200GB/s | PCIe Gen3 32GB/s, NVLINK **<mark><code>300GB/s</code></mark>** |
| FP32      `TFLOPS` | 24.1             | 30.3            | 8.1              | 31.2               | 10.3       | 14/15.7      |
| TF32      `TFLOPS` | 48.3             | `120*`          |                  |                    |            |              |
| BF16      `TFLOPS` | 95.6             | `242*`          |                  | 125                | 165        | NOT support |
| FP16      `TFLOPS` |                  | `242*`          |                  | 125                | 165        |  |
| INT8      `TFLOPS` | 193/193          | `485*`          |                  | 250                | 330        |  |
| INT4      `TFLOPS` |                  | NO              |                  |                    | 661        |  |

Notes:

* `*`: **<mark>with sparsity</mark>**.

Datasheets:

1. [L4](https://nvdam.widen.net/s/rvq98gbwsw/l4-datasheet-2595652)
1. [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)
1. [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)
1. [A30](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
1. [V100-PCIe/V100-SXM2/V100S-PCIe](https://www.nvidia.com/en-us/data-center/v100/)

# 3 Comparison of `A100/A800/H100/H800/910B/H200`

|                | A800 (PCIe/SXM)  | A100 (PCIe/SXM)  | <mark>Huawei Ascend 910B</mark>| H800  (PCIe/SXM) | H100 (PCIe/SXM) | H200 (PCIe/SXM) |
|:---------------|:-----------------|:-----------------|:-----------------|:-----------------|:--------------|:--------|
| Year           | 2022             | 2020             | 2023             | 2022             | 2022          | 2024 |
| Manufacturing  | 7nm              | 7nm              | 7+nm             | 4nm              | 4nm           | 4nm  |
| Architecture   | Ampere           | Ampere           | HUAWEI Da Vinci  | Hopper           | Hopper        | Hopper |
| Max Power      | 300/400 W        | 300/400 W        | 400 W            |                  | 350/700 W     | 700W |
| GPU Mem        | 80G HBM2e        | 80G HBM2e        | 64G HBM2e        | 80G HBM3         | 80G HBM3      | 141GB HBM3e |
| GPU Mem BW     |                  | 1935/2039 GB/s   |                  |                  | 2/3.35 TB/s   | 4.8 TB/s |
| GPU Interconnect (**<mark>one-to-one max bw</mark>**)    | NVLINK 400GB/s   | PCIe Gen4 64GB/s, NVLINK 600GB/s | HCCS **<mark><code>56GB/s</code></mark>** | NVLINK 400GB/s   | PCIe Gen5 128GB/s, NVLINK **<mark><code>900GB/s</code></mark>** | PCIe Gen5 128GB/s, NVLINK 900 GB/s |
| GPU Interconnect (**<mark>one-to-many total bw</mark>**) | NVLINK 400GB/s   | PCIe Gen4 64GB/s, NVLINK 600GB/s | HCCS **<mark><code>392GB/s</code></mark>** | NVLINK 400GB/s   | PCIe Gen5 128GB/s, NVLINK **<mark><code>900GB/s</code></mark>** | PCIe Gen5 128GB/s, NVLINK 900 GB/s|
| FP32  `TFLOPS` |              | `19.5`         |                  |                  | `51 | 67*`    | `67*`   |
| TF32  `TFLOPS` |              | `156 | 312*`   |                  |                  | `756 | 989*`  | `989*`  |
| BF16  `TFLOPS` |              | `156 | 312*`   |                  |                  | `1513 | 1979*`| `1979*` |
| FP16  `TFLOPS` |              | `312 | 624*`   | `320`            |                  | `1513 | 1979*`| `1979*` |
| FP8   `TFLOPS` | NOT support  | NOT support    |                  |                  | `3026 | 3958*`| `3958*` |
| INT8  `TFLOPS` |              | `624 | 1248*`  | `640`            |                  | `3026 | 3958*`| `3958*` |

Notes:

* `*`: **<mark>with sparsity</mark>**.

H100 vs. A100 in one word: **<mark> 3x performance, 2x price</mark>**.

Datasheets:

1. [A100](https://www.nvidia.com/en-us/data-center/a100/)
1. [H100](https://www.nvidia.com/en-us/data-center/h100/)
1. [~~Huawei Ascend-910B~~](https://www.hisilicon.com/en/products/Ascend/Ascend-910) (404)
1. `910` paper: [Ascend: a Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing](https://ieeexplore.ieee.org/abstract/document/9407221), HPCA, 2021

## 3.1 Note on inter-GPU bandwidth: `HCCS vs. NVLINK`

For 8-card A800 and 910B modules: 910B HCCS has a total bandwidth of `392GB/s`,
which appears to be comparable to A800 NVLink (`400GB/s`). However, there are
some differences. To clarify them,

* NVIDIA NVLink: **<mark>full-mesh topology</mark>** as below, so (bi-directional)
  **<mark><code>GPU-to-GPU max bandwidth</code></mark>** is **<mark><code>400GB/s</code></mark>**
  (note that below is `8*A100` module, 600GB/s, `8*A800` shares a similar full-mesh topology);

    <p align="center"><img src="/assets/img/gpu-notes/8x-a100-node-hw-topo.png" width="100%" height="100%"></p>

* Huawei HCCS: **<mark>peer-to-peer topology</mark>** (no stuffs like NVSwitch chip), so (bi-directional)
  **<mark><code>GPU-to-GPU max bandwidth</code></mark>** is **<mark><code>56GB/s</code></mark>**;

    <p align="center"><img src="/assets/img/gpu-notes/ascend-910b-x8-topo.png" width="50%" height="50%"></p>

# 4 Comparison of `H20`/`L20`/`Ascend 910B`

|      | <mark>Huawei Ascend 910B</mark>| L20  (PCIe)      | H20              | H100 (PCIe/SXM) |
|:-------------------|:-----------------|:-----------------|:-----------------|:--------|
| Year               | 2023             | 2023             | 2023             | 2022 |
| Manufacturing      | 7+nm             | 4nm              | 4nm              | 4nm |
| Architecture       | HUAWEI Da Vinci  | Ada Lovelace     | Hopper           | Hopper |
| Max Power          | 400 watt         | 350W             | 500W             | 350/700 watt |
| GPU Mem            | 64G HBM2e        | 48G GDDR6        | 96G HBM3         | 80G HBM3 |
| GPU Mem BW         |                  | 864GB/s          | <mark>4.0TB/s</mark> | 2/3.35 TB/s|
| L2 Cache           |                  | **<mark><code>96MB</code></mark>** | 60MB             | 50MB        |
| GPU Interconnect (**<mark>one-to-one max bandwidth</mark>**)| HCCS 56GB/s  | PCIe Gen4 64GB/s | PCIe Gen5 128GB/s, **<mark><code>NVLINK 900GB/s</code></mark>** | PCIe Gen5 128GB/s, NVLINK 900GB/s |
| GPU Interconnect (**<mark>one-to-many total bw</mark>**)    | HCCS 392GB/s | PCIe Gen4 64GB/s | PCIe Gen5 128GB/s, **<mark><code>NVLINK 900GB/s</code></mark>** | PCIe Gen5 128GB/s, NVLINK 900GB/s |
| FP32 `TFLOPS`      |                  | 59.8             | 44               | 51/67        |
| TF32 `TFLOPS`      |                  | 59.8             | 74               | 756/989 |
| BF16 `TFLOPS`      |                  | **<mark><code>119 | 119</code></mark>**      | **<mark><code>148 | 148</code></mark>**  | `1513 | 1979*` |
| FP16 `TFLOPS`      | 320              |                  |                  | `1513 | 1979*` |
| FP8  `TFLOPS`      |                  |                  | `296 | 296`      | `3026 | 3958*` |
| INT8 `TFLOPS`      | 640              | `239 | 239`      | `296 | 296`      | `3026 | 3958*` |

Notes:

* `*`: **<mark>with sparsity</mark>**;
* L20 max power 350W: collected with dcgm-exporter.

# 5 Notes on US "Chip Export Controls" targeting China

## 5.1 Export Controls `2022.10`

According to
[Implementation of Additional Export Controls: Certain Advanced Computing and Semiconductor Manufacturing Items; Supercomputer and Semiconductor End Use; Entity List Modification](https://www.federalregister.gov/documents/2022/10/13/2022-21658/implementation-of-additional-export-controls-certain-advanced-computing-and-semiconductor),
for chips that can be shipped to the Chinese market, the following conditions must be met:

1. aggregate bidirectional transfer rate must **<mark><code>< 600 Gbyte/s</code></mark>**; **<mark>AND</mark>**,
2. aggregated processing performance must **<mark><code>< 4800 bit TOPS</code></mark>** (TFLOPS), which is
   equivalent to:

    * **<mark><code>< 300 TFLOPS FP16</code></mark>**
    * **<mark><code>< 150 TFLOPS FP32</code></mark>**

A100 and H100 are subjected to these restrictions, that's why there are tailored versions: A800 and H800.

## 5.2 Export Controls `2023.10`

According to
[Implementation of Additional Export Controls: Certain Advanced Computing Items; Supercomputer and Semiconductor End Use; Updates and Corrections](https://www.federalregister.gov/documents/2023/10/25/2023-23055/implementation-of-additional-export-controls-certain-advanced-computing-items-supercomputer-and),
**<mark>in addition to</mark>** the above `2022.10 Export Controls`, chips that
meet one of the following conditions are also **<mark>prohibited</mark>** from being sold in the Chinese market:

1. total processing performance in **<mark><code>2400~4800 bit TOPS</code></mark>**
  AND performance density in **<mark><code>1.6~5.92</code></mark>**;

    2400 bit TOPS is equivalent to:

    * **<mark><code>150 TFLOPS FP16</code></mark>**
    * **<mark><code>75 TFLOPS FP32</code></mark>**

2. total processing performance **<mark><code>>= 1600 bit TOPS</code></mark>**
  AND performance density in **<mark><code>3.2~5.92</code></mark>**;

These restrictions cover most high-performance GPUs, including the old model A800.
However, it should be noted that there is also room for low-computing-power
but high-transfer-rate models, such as the rumored "148TFLOPS + 96GB HBM + 900GB/s
NVLink" H20 GPU.

----

<a href="https://notbyai.fyi"><img src="/assets/img/Written-By-Human-Not-By-AI-Badge-white.svg" alt="Written by Human, Not by AI"></a>
<a href="https://notbyai.fyi"><img src="/assets/img/Written-By-Human-Not-By-AI-Badge-black.svg" alt="Written by Human, Not by AI"></a>
