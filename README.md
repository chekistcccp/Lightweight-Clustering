# Lightweight-Clustering：无健康对照结肠癌 CT 异常定位（V3）

**运行状态：已通过代码语法与合成数据单元测试，尚未在用户真实 MSD CT 上运行 V3。**  
**算法声明：TORC-inspired 独立近似实现，并非官方 TPAMI MATLAB P-code 的等价迁移。**

## 研究目标

在**只拥有结直肠癌患者 CT、不依赖额外健康患者 CT**的条件下，复用冻结 MobileNetV3-Small（来自 ModelScope）的 CT Patch 特征，评估无训练异常定位：

1. 2×2 析因消融：**参考库规模 512 / 8192** × **随机 / 余弦最远点多样性采样**；
2. TORC-inspired 聚类稀有度是否有独立增益；
3. **查询 Patch 局部跨患者近邻支持度**是否改善排序（不再使用先前全部簇均覆盖101名患者的全局支持定义）；
4. 三个预先固定的**参考库采样种子**的稳定性（患者划分始终不变，因此不等于外部验证或独立测试）。

训练参考只从训练患者的 CT 提取，严禁从肿瘤分割掩码选择特征或调参；只有评估阶段读取 `labelsTr`。

## 一、用户手动准备

- Windows WSL、单卡 CUDA GPU（已有 RTX 4090 可用），自建并激活 Conda 环境。脚本**不创建环境、不自动安装或升级 PyTorch**。
- 自行安装与 CUDA 环境匹配的 `torch` 和 `torchvision`，其余依赖 `pip install -r requirements.txt`。
- 将 MSD Task10 Colon 的**原始压缩包**放进 `data/archives/`，不必手工解压；项目自动安全解压和定位 `imagesTr/labelsTr`。来源见 [数据说明](docs/DATA_GUIDE.md)。
- 代码首次使用 ModelScope 自动下载 `timm/tf_mobilenetv3_small_minimal_100.in1k` 权重至 `model/`；也可自行准备原始模型仓库并传 `--offline`。
- 先前的 `outputs/full/cache/` 特征可以继续沿用（冻结骨干、特征层和输入尺寸不变时）；**训练 CT 的 HU Body Mask 在每次调用中只计算一次**，三个采样种子共享。

## 二、执行命令

先完成一个采样种子的快速真实数据检查：

```bash
git pull origin main
bash run.sh --max-cases 8 --slice-stride 4 --sampling-seeds 42
```

确认报告已生成，正式运行（默认 **seed 42/123/2026**、完整 126 个已标注 CT 中的固定患者级 80/20 划分、全部切片）：

```bash
bash run.sh
```

如需先完成仅一个采样种子的全数据实验：

```bash
bash run.sh --sampling-seeds 42
```

查看完整参数：`bash run.sh --help`。所有人工建立的 Conda 环境/数据文件不会被删除。

## 三、八组固定对照

| 名称 | 参考库 | 设计/要验证的因素 |
| --- | --- | --- |
| `patch_knn` | 原未过滤512 | 旧版原始基线、全图评分 |
| `body_patch_knn` | CT体部、随机512 | 体部过滤；本版512从同采样种子的8192池前缀提取，与V2旧512抽样可能不同 |
| `body_diverse_512` | CT体部、多样性512 | 固定512，孤立验证“多样性” |
| `body_random_8192` | CT体部、随机8192 | 固定随机策略，孤立验证“容量” |
| `diverse_patch_knn` | CT体部、多样性8192 | 2×2 析因的最后一格，也是新基线 |
| `torc_approx` | 同上8192＋512 TORC代表点 | 无监督簇稀有度增益 |
| `local_consensus_knn` | 同上8192＋16局部近邻 | 无TORC、仅局部患者来源集中度修正 |
| `torc_consensus` | 同上8192＋TORC＋局部支持度 | 两项修正是否互补 |

**公平对比规则：** 512 库从同一采样种子对应的 8192 库中按患者轮转选出的**前512个 Patch**，消除更换候选池的额外干扰；E3/E4 同大型库。所有版本使用相同冻结特征和 kNN K=5，局部支持用16个近邻，预设 TORC/支持度修正系数均为0.25；不以测试肿瘤掩码调参。

体部掩码由 CT HU > -500、最大连通域等常规形态学操作获得。V3 在**全图、体部 ROI**分别计算指标，并记录被体部掩码排除的肿瘤像素数；不能仅因删除空气区域导致的全图性能提升就称为肿瘤定位改进。

TORC 聚类仅在最大512个训练代表 Patch 上运行，再将簇标签无监督传播给8192参考 Patch，**不是对8192点进行完整 TORC 聚类**。算法详情见 [研究协议](docs/PROTOCOL.md)。

## 四、最终结果：仅一个本地文件

```text
outputs/full/experiment_results.json
```

- JSON schema 3.0.0，包含全部8组×每个采样种子的患者级 Pixel AUROC/AP、全图/体部 Top0.1%、Top0.5%、Top1%、Top5% 的 Dice/IoU/Precision/Recall/FPR、混淆数、21个**数值**PR采样点；
- 各采样种子的患者宏平均、同患者配对差值，以及跨采样种子的均值和离散程度；
- 每个种子的聚类数量/大小、参考库样本与患者覆盖、局部支持度分布、时间、配置及软件版本；
- **绝不包含 CT 图像、分割掩码、异常热力图、特征矩阵或像素预测数组。** 没有健康对照病例，故不得报告患者级健康/癌症AUROC；尚未实现AUPRO，不能虚构该指标。
- `--max-cases 8` 调试结果在 `outputs/debug_8/experiment_results.json`，不当作正式科研指标；
- 重新运行前会将已有旧结果移动至本机 `outputs/history/`，避免丢失。
- 默认不保存预测 NPZ；用户主动添加 `--save-predictions` 才会把预测网格**留在本机 Git 忽略目录**。

**请只把本机 `experiment_results.json` 手动上传回当前 ChatGPT 对话。严禁任何真实数值或病例结果同步到 GitHub。** `data/`、`model/`、`outputs/` 全部已被 Git 忽略；脚本自身不执行 git add/commit/push，CI 只跑无真实病患数据的合成测试。

可额外启用本地 Git 提交前保护：

```bash
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

## 五、研究局限

- 前两轮研究已经查看过当前固定测试集，第三轮方法属于**探索性改进**；跨采样种子波动不能替代新患者的独立外部测试。
- 结直肠肿瘤极稀疏，较高 Pixel AUROC 不代表病灶定位已达临床可用水平；必须同时检查体部 Pixel AP、Top0.1–1% Dice、患者级稳定性及体部遗漏病灶。
- 跨患者支持集中度只是无监督异常启发式，正常但少见的结构可能被误检为病灶。
- TORC 原文：Yang & Lin, *Autonomous clustering by fast find of mass and distance peaks*, IEEE TPAMI 2025, DOI [10.1109/TPAMI.2025.3535743](https://doi.org/10.1109/TPAMI.2025.3535743)；[官方 MATLAB](https://github.com/JieYangBruce/TorqueClustering)核心 P-code 不可直接逐行翻译，项目目前只是独立近似，不声明 MATLAB 等价。
- 更详尽操作与 Codex 规则：[AGENTS.md](AGENTS.md) · [docs/HANDOFF.md](docs/HANDOFF.md) · [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 六、代码测试

```bash
python -m pytest -q
```

这只验证合成数据与算法约束。真实 WSL/RTX4090、ModelScope 权重和完整 MSD 流水线必须由用户在本地运行检查。
