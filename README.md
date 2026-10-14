# Lightweight-Clustering：无健康对照 CT 异常定位

> **状态：研究原型 / 尚未在 MSD Colon 上验证。** 本项目的 `torc_approx` 是依论文思路独立编写的 **TORC-inspired 近似算法**，不是 2025 TPAMI 官方 MATLAB P-code 的严格翻译，更不代表通过 MATLAB 数值等价验证。不能以“官方 TORC Python 复现”对外宣称。

## 目标
在**只有结直肠癌患者 CT、没有独立健康患者 CT**的条件下，使用冻结轻量视觉骨干 + 无训练聚类/近邻方法生成病灶异常热图。对照三种方法：`patch_knn`、`torc_approx`、`torc_consensus`。**不使用肿瘤标注构造特征库、选层、筛选切片、选择参数。**

### 上手（Windows WSL / 单张 GPU）
1. 用户自行创建并激活 Conda 环境。参考：`conda create -n lwcluster python=3.11 -y && conda activate lwcluster`。安装依赖：`pip install -r requirements.txt`；根据自身 CUDA 环境单独安装兼容 PyTorch，**run.sh 不修改环境**。
2. 手动下载 **Medical Segmentation Decathlon Task10_Colon 原始压缩包**，直接放入 `data/archives/`（也支持 `data/` 根目录，tar / tar.gz / zip；不解压）。项目会自动、安全地解压，并定位 `imagesTr/labelsTr`。不要上传患者原始数据。
3. 权重由代码通过 ModelScope 自动下载到 `model/`：已核实的模型仓库 `timm/tf_mobilenetv3_small_minimal_100.in1k`。网络可用时第一次运行自动获取；如果已放好原始 ModelScope 权重目录则跳过下载。
4. `bash run.sh` 完成检查、解压、患者级划分、冻结特征缓存、共同参考集建库、三组实验、CSV/JSON 指标汇总；再次运行复用缓存。可通过 `bash run.sh --max-cases 8` 快速诊断（**限制病例仅用于调试，不用于论文结果**）。
5. `python -m pytest -q` 验证算法级测试；`python scripts/matlab_parity.py --help` 说明 MATLAB 一致性验证流程。

**所有运行参数：** `bash run.sh --help`。建议正式实验传入 `--slice-stride 1`，默认 `--slice-stride 4` 仅用于快速初验；指标明确报告评估覆盖的切片数和患癌阳性像素数。

### 输出结构
- `outputs/split.json`：固定的患者级划分（seed 42），防止跨患者数据泄漏。
- `outputs/cache/*.npz`：提取后的归一化 Patch 特征、切片索引；缓存与原始影像掩码分离。
- `outputs/predictions/{method}/{patient}.npz`：每个患者的低分辨率得分图 + z 切片索引。
- `outputs/metrics/per_case.csv`、`summary.json`：每方法逐例和总体 Pixel AUROC、AP、区域 Dice@top5%（辅助指标）以及时间等。
- `outputs/torc_diagnostics.json`：簇数量、簇大小、支持患者数、归档算法参数。

### 三个公平对照
三组方法在**同一患者划分、同一预训练骨干、同一候选 Patch 集合**运行：
- `patch_knn`：KNN 距离均值。
- `torc_approx`：TORC-inspired 聚类生成的簇支持/离群校正 + KNN 距离。
- `torc_consensus`：在上一组基础上增加**跨患者簇覆盖率**校正。

注意：这是 **abnormal-only localization**，不是在缺少健康阴性病例时测得的病例级正常/异常诊断准确率。所有患者患癌，不能报告病例级 binary AUROC。匿名医学影像需遵守数据许可及伦理要求。

### TORC 迁移状态 / 版权
论文：J. Yang, C.-T. Lin, *Autonomous clustering by fast find of mass and distance peaks*, IEEE TPAMI (2025), DOI: [10.1109/TPAMI.2025.3535743](https://doi.org/10.1109/TPAMI.2025.3535743)。
- 官方：https://github.com/JieYangBruce/TorqueClustering （核心 `.p` 不可读取，Windows MEX 不能在 WSL 原生直接调用）。
- 非官方参考：https://github.com/Cognet-74/TorqueClusteringPy ；不能假定与官方一致。
- 本仓库**没有复制、解密或打包官方 P-code/MEX**；`torc_approx.py` 根据公开思想独立实现，以不同名称防止混淆。
- 官方代码采用 CC BY-NC-SA 4.0 且明确限制商业使用；使用官方实现或衍生代码时需满足其许可、保留署名并取得商业许可。本文提供的研究代码亦仅供非商业学术验证。
- `scripts/matlab_parity.py` 读取在本地 MATLAB 由**用户自行运行官方源码**得到的 .mat 输出，比对一致性（允许标签置换）；验证前不声称等价。

### 论文基本约束
- 所有超参数预先固定，不可用 test mask 调参；没有验证集时只报告**探索性**结果。
- 输出热图不等于经过组织病理验证的病灶定位；罕见正常结构可能被判为异常。
- 内存：官方基于完整 N×N 距离阵；本项目通过 **按患者抽样的上限 `--torc-max-points`** 控制聚类规模。计算、显存、内存和耗时均应如实报告。
- 需要至少多次患者级随机划分 / 外部数据测试与稳定性实验才能支持较高层次投稿，**不能仅凭运行成功保证发表**。
