# 复现指南 / Reproduction Guide

本文件给出复现 UnSegGNet + STD 的最小信息集：环境、硬件假设、完整命令、随机性、
耗时现状与已知失败模式。论文中的参考数值不在此重复，**请以论文为准**。

> 版本事实来源：`requirements.txt`（完整 pin 列表）与 `script.sh`（安装顺序与 wheel 索引）。
> 本文件只摘录复现必需的部分；其余依赖以 `requirements.txt` 为准。

## 1. 环境 / Environment

### Python

`requirements.txt` 与 `script.sh` **都没有固定 Python 版本**。`README_ORIGIN.md`
记录上游项目使用 **Python 3.11**，本仓沿用该环境。其他版本未经验证。

### 深度学习栈（来自 `requirements.txt`）

| 组件 | 版本 |
| --- | --- |
| torch | `2.4.0+cu124` |
| torchvision | `0.19.0+cu124` |
| torch_scatter | `2.1.2+pt24cu124` |
| torch_sparse | `0.6.18+pt24cu124` |
| torch_cluster | `1.6.3+pt24cu124` |
| pyg-lib | `0.4.0+pt24cu124` |
| torch-geometric | `2.7.0` |
| numpy | `2.4.4` |
| transformers | `5.5.3`（`SamModel` / `SamProcessor`，用于 MedSAM baseline） |
| optuna | `4.8.0` |
| plotly / kaleido | `6.7.0` / `1.2.0`（`fig.write_image` 导出静态图需要 kaleido） |
| opencv-python | `4.13.0.92` |
| scikit-learn | `1.8.0`（`KMEANS_DINO` 与 IoU 相关工具） |
| deeplake | `3.9.52`（在线加载 ECSSD） |

CUDA wheel 相关包（`nvidia-*-cu12`、`triton==3.0.0`）由 `torch==2.4.0+cu124`
带入，版本见 `requirements.txt`。

MedSAM Python 包在 `requirements.txt` 中已 pin 到 commit：

```
-e git+https://github.com/bowang-lab/MedSAM.git@d71e8a1a99ad751840a22a7fa3ecfb4166fb1488#egg=medsam
```

该 commit 是 `bowang-lab/MedSAM` 的当前 HEAD（核实时间见仓库提交记录）。
`segment.py` 通过 `transformers` 加载 Hugging Face 上的
`wanglab/medsam-vit-base` 权重；同时 `features_extract.py` 在模块顶层
`import segment_anything`，该模块由 MedSAM 包安装提供。

### 安装顺序（`script.sh`）

```bash
./script.sh
```

`script.sh` 依次执行：

1. `pip install torch==2.4.0+cu124 torchvision==0.19.0 -f <pytorch wheel 镜像>`；
2. 读取 `torch.__version__` 拼出 PyG 的 wheel 索引，安装 `torch-scatter`、`torch-sparse`；
3. `pip install torch_geometric class_resolver`；
4. `pip install -r requirements.txt`（覆盖上面多数包的精确版本）；
5. `pip install` 上面 pin 住的 MedSAM commit；
6. MedSAM 权重是可选的，脚本会用 `transformers` 从 Hugging Face Hub 预取
   （`script.sh` 的 `fetch_medsam_weights_hf`），失败也不影响默认 DINO 流程。

> 注意：第 2 步的 `torch-scatter`/`torch-sparse` 在 `script.sh` 里不带版本号，
> 但 `requirements.txt` 会装回 `+pt24cu124` 的固定版本。若 wheel 索引不可用，可直接
> 按 `requirements.txt` 的版本安装。

### 外部权重

| 权重 | 获取方式 |
| --- | --- |
| DINO `dino_deitsmall8_pretrain_full_checkpoint.pth`（约 336 MB） | `segment.py` 首次运行时自动从 `https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth` 下载到仓库根目录；离线时可手动放置同名文件 |
| MedSAM（`wanglab/medsam-vit-base`） | `transformers` 自动从 Hugging Face Hub 下载；离线时设置 `HF_HOME` 指向预置缓存，或用 `HF_ENDPOINT` 镜像 |
| `model.pt`（约 108 KB，**已入库**） | 随机初始化检查点，见 `README.md`；可用 `--gen_init` 重建 |
| `medsam_vit_b.pth`（约 358 MB） | **不需要**。本仓不加载它；旧版的 `gdown` 兜底已移除 |

## 2. 硬件假设 / Hardware

- **单卡**。`--device` 默认 `cuda`，代码里没有任何 `DataParallel`/`DistributedDataParallel`
  或多进程训练逻辑，一张 GPU 即可；CPU 也可运行但会非常慢。
- 代码是**逐图训练**：每张图新建一个 GNN，训练 `--epochs`（默认 20）个 epoch，图上没有 batch 维度。
- 按代码推断的规模（224×224 输入，`--resolution` 默认值）：
  - `ViTExtractor` 默认 `stride=4` 改写 DINO ViT-S/8 的 patch 卷积步长，得到
    `55 × 55 = 3025` 个 patch 节点（`util.graph_to_mask` 里 `stride = 4` 的 reshape 与之一致）；
  - 节点特征维度 `384`（`segment.py`: `self.feats_dim = 384`）；
  - 邻接矩阵 `3025 × 3025`（约 9.15 M 个元素；DMON 路径为 `float32`，约 36.6 MB，
    NCUT 路径保持 `float64`，约 73.2 MB）；
  - GNN 宽度 `GNNpool(384, 64, 32, 2, ...)`。
  - **峰值显存：仓库未记录，作者未测量**。按上述张量规模推断属于单卡小显存即可运行的量级，
    但这里不给具体 GB 数字。

## 3. 数据准备 / Data

按 `datasets/DATASETS.md` 准备：

- **CUB**：`datasets/CUB_200_2011/`（`images/`、`images.txt`、`train_test_split.txt`）
  与 `datasets/segmentations/`；
- **DUTS**：`datasets/DUTS-TE/DUTS-TE-Image`、`datasets/DUTS-TE/DUTS-TE-Mask`；
- **ECSSD**：无需手工下载，`datasets/dataset.py` 通过 `deeplake.load("hub://activeloop/ecssd")`
  在线读取（需要联网 + `deeplake`）。

数据集目录整体被 `.gitignore` 排除，不随仓库发布。

## 4. 完整命令 / Commands

所有命令都在仓库根目录执行。

### 4.1 Baseline（GCN softmax，`--activation selu`）

```bash
python main.py --dataset ECSSD --activation selu --threshold 0.3 --bs False
python main.py --dataset DUTS  --activation selu --threshold 0.3 --bs False
python main.py --dataset CUB   --activation selu --threshold 0.3 --bs False
```

### 4.2 STD-GCN（`--activation STD`）

```bash
python main.py --dataset ECSSD --activation STD --threshold 0.3 \
  --alpha 2.0 --lam 1.0 --eps 3.0 --nb_iter 3 --bs False
```

`--alpha/--lam/--eps/--nb_iter` 的默认值只是 CLI 默认值；论文中的取值来自 4.3 的搜索，
并由搜索产生的 `best_std_params.json` 记录（该文件不随仓库发布）。

### 4.3 Optuna 超参数搜索（50 trials）

```bash
python StdParamSearch.py --dataset ECSSD --activation STD --gen_init --n_trials 50
```

搜索范围（`StdParamSearch.py`）：`alpha`、`lam`、`eps` 在 `[0.1, 10]` 上对数采样，
`nb_iter` 在 `[1, 10]` 上整数采样；目标为最大化 mIoU。结果写入
`search_results/optuna.db`、`search_results/<dataset>/best_std_params.json` 与
`search_results/<dataset>/std_visualizations/*.png`。

### 4.4 掩码导出与可视化

```bash
# 需要 search_results/<dataset>/best_std_params.json（自行搜索得到）
python SegmentViz.py --dataset ECSSD --activation STD
python SegmentViz.py --dataset ECSSD --activation STD --save_as_baseline   # 存 baseline 掩码

# 没有搜索结果时，显式给参数并跳过读取
python SegmentViz.py --dataset ECSSD --activation STD --no_load_hparams \
  --alpha <alpha> --lam <lambda> --eps <epsilon> --nb_iter <T>
```

导出目录为 `output/<dataset>/`，每个样本写 `XXXX_img.png` / `XXXX_gt.png` /
`XXXX_pred.png`（或 `XXXX_baseline.png`）。

### 4.5 复用评估函数

```bash
python train_fn.py --dataset DUTS --activation selu --threshold 0.3 --bs False
```

### 4.6 MedSAM baseline

```bash
python main.py --dataset CUB --process MEDSAM_INFERENCE
```

注意：见第 7 节的失败模式 6 —— 当前 CLI 入口不会传 `prompt_box`，因此
`MEDSAM_INFERENCE` 需要自己调用 `Segmentation.segment(img, mask, prompt_box=[x0, y0, x1, y1])`。

## 5. 随机性 / Randomness

- 入口脚本在启动时调用 `cli_utils.set_seed(args.seed)`，`--seed` 默认 **42**。
  `set_seed` 同时设置 `random`、`numpy`、`torch`（CPU）与 `torch.cuda` 的种子，并设置
  `torch.backends.cudnn.deterministic = True`、`torch.backends.cudnn.benchmark = False`。
- 模型初始化：`--gen_init` 在训练前保存一份 `state_dict`（默认 `model.pt`）；此后**每张图**
  都从这份检查点恢复 GCN/MLP 参数，因此 baseline 与 STD 共享同一初始化，差异只来自
  softmax 与 STD 正则化。` --alpha/--lam/--eps` 是 STD 独有标量，检查点里没有时用 CLI 值。
- `--process KMEANS_DINO` 的 `KMeans(random_state=42)` 固定为 42，不受 `--seed` 影响。
- Optuna 搜索本身**未固定 sampler 种子**（`StdParamSearch.py` 未给 `TPESampler(seed=...)`），
  因此每次搜索得到的 trial 序列不同；每个 trial 内部的图像评估仍用 `--seed 42`。
  即：单次评估可复现，50-trial 搜索过程不可位级复现。
- 即便种子固定，CUDA/cuDNN 与非确定性算子的差异仍可能带来极小的数值漂移。

## 6. 运行耗时 / Runtime

**未记录。** 仓库与 README 中都没有任何实际耗时（wall-clock）或每图秒数的记录，
本文件也不臆造数字。

可确定的运行结构：`main.py` / `train_fn.py` 会遍历数据集的**全部测试图**，每张图独立训练
`--epochs`（默认 20）个 epoch；`StdParamSearch.py` 的 50 个 trial 每个都要完整遍历一次数据集，
因此搜索总成本约为"单次全量评估 × 50"。

## 7. 常见失败模式 / Known failure modes

1. **数据集路径不对**。`datasets/dataset.py` 以 `datasets/` 目录为根：
   CUB 缺文件时抛 `FileNotFoundError: CUB dataset is incomplete; missing <path>`；
   DUTS 数量或文件名不对齐时抛 `ValueError`（`DUTS image/mask count mismatch` /
   `DUTS image and mask filenames are not aligned`）；CVC 缺 `metadata.csv` 或配对不全时抛
   `FileNotFoundError`。
2. **ECSSD 依赖 deeplake 在线拉取**。无网络、`deeplake` 未安装或 hub 访问受限时，
   `Dataset("ECSSD")` 会在构造阶段失败。
3. **初始化检查点缺失**。非 `--gen_init` 且 `--checkpoint` 指向的文件不存在时抛
   `FileNotFoundError: Initialization checkpoint not found ...`。`main.py` 总会重新生成，
   `SegmentViz.py` / `train_fn.py` 需要显式 `--gen_init`。
4. **检查点与本仓结构不兼容**。加载时若缺少 `alpha/lam/entropy_epsilon` 之外的共享参数，
   会抛 `RuntimeError: Checkpoint is missing shared model parameters: [...]`。
5. **DINO 权重下载失败**。`segment.py` 只在文件不存在时下载；公司网络/离线环境请手动放置
   `dino_deitsmall8_pretrain_full_checkpoint.pth` 到仓库根目录。
6. **MedSAM 相关**：
   - 权重下载失败 → `transformers` 抛 Hub 连接/认证错误；离线请预置 `HF_HOME`；
   - `features_extract.py` 顶层 `from segment_anything import sam_model_registry`，若 MedSAM 包
     未安装，**所有** process 都会在 import 阶段报 `ModuleNotFoundError: segment_anything`；
   - `--process MEDSAM_INFERENCE` 的 `segment()` 要求显式 `prompt_box`，缺省会抛
     `ValueError: MEDSAM_INFERENCE requires an explicit prompt_box`；`main.py`/`SegmentViz.py`
     不传该参数，异常会被逐图捕获，最终以 `RuntimeError: No samples were evaluated successfully.`
     退出；
   - MedSAM 权重获取失败不影响默认流程（`features_extract.py` 只 import
     `sam_model_registry`，从不构建 SAM 模型）。
7. **`SegmentViz.py` 找不到超参数**：默认读取
   `search_results/<dataset>/best_std_params.json`，该文件不随仓库发布，缺失时抛
   `FileNotFoundError: Configuration file not found`。用 `--no_load_hparams` + 显式参数绕过。
8. **Optuna 绘图失败**：`fig.write_image` 需要 `kaleido`；缺失时会在搜索**结束后**报错，
   但 `optuna.db` 与 `best_std_params.json` 已经写好。
9. **无样本成功**：所有入口在 `total_samples == 0` 时抛
   `RuntimeError: No samples were evaluated successfully.`，而不是输出一个无效的 mIoU。

## 8. 参考数值

参考的定量结果（mIoU 等）见论文。本仓库不重复这些数字。
