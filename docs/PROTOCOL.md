# Lightweight-Clustering V2 研究协议（2026-10-14）

## 核心研究问题
仅有结直肠癌患者 CT，缺乏健康对照患者；使用无训练冻结特征与聚类/近邻完成病灶异常区域定位。E0（旧512 Patch KNN）是不可更改的可比基线。所有病例均为阳性患者，**不能计算患者级正常/异常检测 AUROC**。任何异常簇不等同于肿瘤分割真值；TORC 是 **独立近似实现，不是官方 MATLAB/P-code 等价移植**。

## 数据、泄漏边界与可复现性
- MSD Task10_Colon 的 imagesTr/labelsTr；固定种子42，80/20患者级划分。旧 split.json 固定时不重新随机抽样；绝不在训练特征构建/参考选择/超参数调试时使用 labelsTr。
- 2.5D 轴位 (z-1,z,z+1)；HU软组织窗 [-150,250]，冻结 MobileNetV3-Small 两层 timm 中间特征 (1,2)，resize至224×224、patch grid=14×14；slice-stride=1 为正式默认。
- Body Mask：只使用CT HU>-500、2D形态学闭运算、最大连通域和填洞。每个参考 Patch 必须至少35%在体部内；不能使用 GT 修正掩码。被遗漏的 GT 肿瘤像素数仅用于**事后诊断**。
- 统一KNN K=5；GPU 可用时以 chunked torch.matmul 精确计算 cosine KNN，避免8192×8192全量TORC距离阵；TORC 只对512代表点运行。
- CNN特征缓存沿用旧实验(模型、输入、层配置和slice stride不变)。参考库由训练患者独立重建，用户自行准备 Conda/压缩数据，权重走ModelScope下载。

## 五组逐项独立消融
| 实验 | 流程 | 参考规模 | 科学问题 |
| --- | --- | --- | --- |
| E0 patch_knn | 旧全图参考，随机患者均衡采样 | 512 | 基础对照 |
| E1 body_patch_knn | CT-only body筛选，随机均衡，维持同预算 | 512 | 体外背景是否污染特征 |
| E2 diverse_patch_knn | E1基础上采用余弦最远点优先 + 患者均衡大库 | 8192 | 正常结构覆盖和特征多样性是否提高 |
| E3 torc_approx | E2大库 + 512代表点TORC-inspired聚类 + 簇稀有度评分 | 8192+512 | 聚类是否提供超越大库KNN的增益 |
| E4 torc_consensus | E3同库 + 簇跨患者支持惩罚 | 8192+512 | Consensus是否有额外判别力 |

注意：E0 与 E1/E2 的参考库**有意不同**（这是消融变量），E2/E3/E4 **必须完全共享大库**。全图评分时对E1–E4的体外区域置低值，体内计算时按原值评价。不可只报告全图AUROC上升而忽略体内AUROC/AP，或根据测试表现选择最有利的ROI。

### 评分定义
- E0 / E1 / E2: 近邻余弦距离的均值除以**各自仅从训练参考计算的尺度**；
- E3 = E2 + 0.25×(1 − 近邻簇大小占比/最大簇大小占比)；
- E4 = E3 + 0.25×(1 − 近邻簇覆盖的独立训练患者占比)。
- TORC只在512代表Patch上训练无监督簇结构，之后由训练参考特征余弦近邻传播簇标签至8192 Patch。簇大小与跨患者覆盖度均**在大库上重新统计**。
- 系数是研究先验设定，未经验证最优；不要用test mask选超参。

## 指标与统计
- **两套ROI**：full-image 和 CT-body-only。各自计算患者像素 Pixel AUROC、AP；有正负像素才计算 AUROC/AP，无法计算保留 null。
- **四种面积预算**：Top 0.1%、0.5%、1%、5%（占当前ROI有效像素数），输出 Dice、IoU、Precision、Recall、FPR、TP/FP/FN/TN。top-k计算使用确定性排序，不以GT优化阈值。
- 每病例提供21组数值 precision-recall 取样点，保留与AP的一致性。数值PR点不是图像，不包含像素级prediction数组。
- 所有方法报告患者宏平均/离散程度及相对E0配对差值，另列体部掩码覆盖比例与排除的肿瘤GT像素数。未来正式投稿应扩展Bootstrap CI、患者多折和跨数据集验证；**当前不实现AUPRO**。
- 峰值显存使用torch.cuda.max_memory_allocated口径，只代表torch张量分配，不代表整进程显存。记录每阶段时间与实际运行配置。

## 实验数据隐私与交接
- 唯一需要用户回传的指标报告：`outputs/full/experiment_results.json`。历史本机备份仅供用户自用，调试与正式报告隔离。
- 严禁真实实验数据、患者级指标、任何报告JSON、日志、图像、缓存通过 GitHub commit/PR/Issue/Release/Actions Artifact/CI日志同步。
- 脚本不提交、不推送任何实验结果；不上传热图和NPZ。默认不保存图像；即使加 --save-predictions，也只允许本地。
- 汇总必须等所有方法及病例成功后才原子写入，不得将未完成结果标为完成。

## MATLAB 兼容性协议
1. 在 MATLAB 中从官方 https://github.com/JieYangBruce/TorqueClustering 获取允许使用的官方 P-code/MEX，在 MATLAB 内对小规模 `DM` 运行 `[Idx,Idx_with_noise]=TorqueClustering(DM,0,1,0)`；保存 `DM` 与 `Idx` 到 .mat。
2. 运行 `python scripts/matlab_parity.py --golden golden.mat` 对比相同距离矩阵下的 ARI、簇数。
3. 该脚本测试的是差异，不会使独立近似自动变成正式官方等价复现。只有覆盖多个 benchmark 且满足输出一致性后，才可进入严格迁移验证阶段。
4. 遵守作者 CC BY-NC-SA 4.0 和非商用许可，不复制加密 P-code/MEX，也不逆向工程。

## 性能/资源限界
N×N 内存为 O(N²)；矩阵 512×512 float32 大约1 MiB，本实现另外分配 1024×1024，并逐步凝聚，时间显著高于标准 KNN。严禁未经说明对数万 Patch 直接运行这个版本。


## V3 固定研究协议 — 扩展析因与局部跨患者评分（2026-10-15）
本节覆盖上面的V2默认运行说明，旧V2仍用于历史方法追踪。

### 首要问题：旧测试结果能否解释容量与多样性？
使用 2×2 factorial：
- R512: `body_patch_knn`；D512: `body_diverse_512`；
- R8192: `body_random_8192`；D8192: `diverse_patch_knn`。
每个采样seed先各建一份 R8192、D8192；从相应完整库截取按患者均衡轮转的前512个元素作为其小规模库，控制规模效应不伴随候选池改变。四组在同一训练患者、同一 body ROI、同一特征和评价协议运行。旧 `patch_knn` 继续作为旧实验独立基线。

对每位患者、每个seed的单项性能指标 Y：
- 容量主效应 = [ (R8192−R512) + (D8192−D512) ] / 2；
- 多样性主效应 = [ (D512−R512) + (D8192−R8192) ] / 2；
- 交互项 = (D8192−R8192) − (D512−R512)。
逐患者先配对后求均值，按seed重复统计。**改变参考库尺度用于计算距离归一化（仅训练数据校准），不可从测试label估计归一化系数。**

### 次要问题：TORC与局部跨患者支持
E2 `diverse_patch_knn` 固定为8192库基线；
- `torc_approx`：+ 0.25×TORC-inspired簇大小稀有度；
- `local_consensus_knn`：+ 0.25×局部16-NN患者来源集中度；
- `torc_consensus`：同时添加两个0.25固定权重；
- 局部患者集中度：基于查询Patch前16个特征近邻的 cosine distance，采用仅由训练参考距离尺度决定的指数权重，再用**加权Simpson相同患者概率**归一化到[0,1]，高集中度代表少数患者提供近邻支持，禁止访问肿瘤mask；
- 局部支持度不同于旧版“整个TORC簇覆盖多少患者”；并单独输出局部评分的分布及是否存在变化。所有版本同一个完整参考库，同一个KNN前5距离基线。
- `torc_approx` **不是官方MATLAB P-code等价移植**，不可据此否定原始TORC论文。

### 重复性、指标、隐私
- 患者划分使用用户历史的随机seed=42（固定101/25），对参考库抽样使用种子42、123、2026；这些重复**不能充当独立患者验证**，也不能仅凭多seed微小波动宣称统计显著。
- 每方法、每患者、每seed记录full/body Pixel AUROC、AP、Top0.1%、0.5%、1%、5%的Dice/IoU/Precision/Recall/FPR/TP/FP/FN/TN、21个数值PR点，额外汇总每seed宏平均、跨seed离散程度、9组预设配对方法比较、析因主效应与交互项。
- 输出仅本地 `outputs/full/experiment_results.json`，V3 schema=3.0.0；若上一轮同名结果存在，先在本机 `outputs/history/` 归档。实验未完成时不能写完成报告。
- 在旧25例测试上已经进行过研究设计反馈，本轮为探索性；提交论文须新外部测试或患者级独立验证，不能把25人×3seed写作75人。
- 不同步真实指标、CT、可视化及任何患者相关文件到GitHub。CI仅运行合成测试，不读取真实数据。
