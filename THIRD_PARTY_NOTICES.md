# 第三方代码与许可说明 / Third-Party Notices

本仓（UnSegGNetSTD）整体以 **MIT License** 发布，见 `LICENSE`
（`Copyright (c) 2024 Sai Gopal Reddy Kovvuri`，该版权行沿用上游 UnSegGNet 项目；
本仓相对上游的 STD 扩展以同一许可发布）。

本仓包含或派生自下列第三方项目，它们各自保留原有许可。凡本文件未逐一核实的条目，
一律注明"见上游仓库"，不作推断。

## 1. 代码来源

| 组件 | 本仓文件 | 上游 | 许可 | 备注 |
| --- | --- | --- | --- | --- |
| DINO | `vision_transformer.py` | https://github.com/facebookresearch/dino | **Apache-2.0** | 已核实。内容为 DINO 的 ViT 实现（其自身注明 largely copy-paste from `rwightman/pytorch-image-models`，timm 为 Apache-2.0），本仓做了少量修改 |
| DINO 预训练权重 | 运行时下载 `dino_deitsmall8_pretrain_full_checkpoint.pth` | 同上（`dl.fbaipublicfiles.com`） | 见上游仓库 | 权重不随本仓分发，首次运行时自动下载 |
| dino-vit-features | `extractor.py`、`features_extract.py` | https://github.com/ShirAmir/dino-vit-features | **MIT** | 已核实。`extractor.py` 文件头明确写有 "Code was taken from .../dino-vit-features/blob/main/extractor.py" |
| DeepCut | `gnn_pool.py`、`util.py`、`train_fn.py` 等的基础实现 | https://github.com/SAMPL-Weizmann/DeepCut | **MIT**（`© 2023 The Weizmann Institute of Science`） | 已核实。`README_ORIGIN.md` 声明 DeepCut "laid the foundation for our code" |
| bilateral solver | `bilateral_solver.py` | https://github.com/poolio/bilateral_solver | **MIT**（`© 2016 Ben Poole`） | 已核实。`bilateral_solver.py` 文件头注明该来源；本仓版本经由 DeepCut 传入 |
| UnSegGNet（原项目） | 本仓的起点 | 原仓库已归档，后继维护版本：https://github.com/ksgr5566/UnSeGArmaNet | **MIT**（`© 2024 Sai Gopal Reddy Kovvuri`） | 已核实。出处与致谢见 `README_ORIGIN.md` |
| MedSAM | 通过 pip 安装（`script.sh` / `requirements.txt`），本仓不含其源码 | https://github.com/bowang-lab/MedSAM | **Apache-2.0** | 已核实；本仓 pin 到 commit `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488` |
| MedSAM 权重 | `--process MEDSAM_INFERENCE` 时由 `transformers` 下载 | https://huggingface.co/wanglab/medsam-vit-base | **Apache-2.0** | 已核实（模型卡 `license: apache-2.0`）；权重不随本仓分发 |
| PyTorch Geometric | `torch_geometric`（含 `torch_scatter`/`torch_sparse` 扩展） | https://github.com/pyg-team/pytorch_geometric | **MIT** | 已核实。以 pip wheel 形式安装，源码不入库 |
| PyTorch / torchvision | pip 依赖 | https://github.com/pytorch/pytorch 、https://github.com/pytorch/vision | BSD-3-Clause（见上游仓库） | 以 pip wheel 形式安装，源码不入库；许可细节以官方仓库为准 |

## 2. 数据集

本仓**不分发**任何数据集文件（`datasets/` 整体在 `.gitignore` 中，仅保留
`datasets/DATASETS.md` 与 `datasets/dataset.py`）。使用前请自行获取并遵守各自条款：

| 数据集 | 获取方式 |
| --- | --- |
| CUB-200-2011 + segmentations | Caltech Data（记录 `65de6-vp158` / `w9d68-gec53`），见 `datasets/DATASETS.md`；官方页面 https://www.vision.caltech.edu/datasets/cub_200_2011/ |
| DUTS | http://saliencydetection.net/duts/ |
| ECSSD | 通过 `deeplake` 的 `hub://activeloop/ecssd` 在线读取 |
| CVC | 需本地自备（`datasets/CVC/metadata.csv` + `png_image_path`/`png_mask_path`） |

各数据集的许可与引用要求请以上游页面为准，本文件不作推断。

## 3. 未完全确认的条目

- DINO **预训练权重**的具体使用条款：上游仓库 `facebookresearch/dino` 的代码为
  Apache-2.0，权重文件本身的使用条款以上游 README/模型卡为准，本文件不代为断言。
- Bilateral solver 在本仓中是经 DeepCut 传递的副本，DeepCut 侧对它的许可声明未单独核实，
  故同时给出 DeepCut 与 `poolio/bilateral_solver` 两个上游链接。
- 上游 UnSegGNet 的原始仓库地址已不可用（`README_ORIGIN.md` 指向的
  `ksgr5566/UnSeGArmaNet` 为后继版本），若需引用最初版本请以论文为准。

如发现遗漏或与上游许可不一致之处，欢迎提 issue 更正。
