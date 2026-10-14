# AGENTS.md — Lightweight-Clustering 科研与代码代理约束

## 核心科学要求（不可改变）
- 任务：所有训练病例为结直肠癌 CT，无独立健康患者 CT；生成像素级肿瘤异常定位分数，不得声称健康/患病二分类 AUROC。
- 主数据：MSD Task10_Colon（用户把原始压缩包放在 `data/archives/` 或 `data/`）；自动解包/索引/预处理。保留压缩包，禁止任何患者影像上传仓库。
- 患者级固定随机拆分 seed 42；不存在相同患者在训练建库及测试评价两侧。不能用 `labelsTr` 构建参考库、切片选择、超参数调优或特征过滤，**仅在评价时访问掩码**。
- 冻结的轻量 MobileNetV3；ModelScope 官方模型卡已核实 `timm/tf_mobilenetv3_small_minimal_100.in1k`。模型存 `model/`，数据 `data/`，中间缓存 `outputs/cache/`，指标 `outputs/metrics/`，均不提交 Git。
- 五组固定消融（原基线、体部过滤、扩大且多样化 KNN、TORC-inspired、TORC+Consensus）：`patch_knn`、`torc_approx`、`torc_consensus`；各方法必须共享患者拆分、冻结特征和评价口径；E2/E3/E4 共用大型参考库，E0/E1 使用各自明确规定的 512 点采样作为消融变量，不得逐方法按测试标签调参。
- 官方 MATLAB 的 `TorqueClustering.p`、其他 .p 和 `TORC.mexw64` 不是可直接逐行迁移的开源 .m；本项目实现属于**近似独立实现，未经官方 parity 验证**。写论文不得用“严格等价”或“官方 TORC”描述。
- 本算法可能让常见肿瘤形成簇、让罕见健康解剖成为离群点；研究报告必须包括失败病例、簇患者覆盖率和异常富集证据，不能只提供好看的热力图。

## 执行习惯
- 用户**手动创建/激活 Conda** 环境、手动获取数据压缩包；脚本不 `conda create`、不 `pip install`、不升级 torch、也不运行 Slurm。
- `bash run.sh` 是唯一正式全流程入口：自检→安全解压→索引→冻结推理缓存→体部筛选/患者均衡采样→小 TORC 代表集→五方法→全图及体部 ROI 评价→结果归档；缓存与断点复跑必须有效；依赖不完整/无数据/无法下载权重时**明确失败而非输出虚构数字**。
- 默认面向单卡 GPU（WSL）、适度显存、混合精度可选；不要引入 Qwen、扩散模型、额外骨干、伪标签网络、其他数据集或训练步骤，除非用户明确变更。
- 固定患者 seed、预处理窗、slice stride、模型 revision、torch/timm/numpy 版本及脚本 git SHA；每次结果带配置快照。比对 Pixel AUROC、Pixel AP、AUPRO（若实现）、Dice@top5%（辅助，不作为监督阈值）、运行时间与峰值显存。
- 代码应容错单切片/极小患者数/无正例像素，空预测不填零；异常只在真实数据运行后定量确认。
- 正式结果不能使用 `--max-cases` 之类调试开关；不能利用测试掩码挑最优超参或报告乐观偏差。

## 开发与交接
- 改代码前先读 README、本文件以及 `docs/HANDOFF.md`；结束后更新 HANDOFF：修改内容、测试命令/结果、当前阻塞、下一步。
- 每项变更应更新 `tests/`，有确定性小型合成数据测试；数据未提供时只允许运行 smoke/unit test，不得假称在 MSD 上通过。
- 不要同步 `data/`、`model/`、`outputs/`、`*.nii.gz`、凭据、代理配置、token、原始医学影像、conda 目录。
- 本仓库面向学术研究，研究使用 TORC 时遵守原作者署名与 CC BY-NC-SA 4.0 非商业条款。

## 实验结果唯一交接文件（必须执行）
- 任何完整实验或调试实验只生成**一份需要人工回传的指标报告**：正式 `outputs/full/experiment_results.json`；调试 `outputs/debug_N/experiment_results.json`。JSON 含全部已实现数值指标、逐病例指标、聚类诊断、参数、患者拆分、时间、显存、版本与相对基线的配对差值。
- JSON 必须**完全不包含** CT、NIfTI、像素掩码、可重建患者影像的特征数组、热力图/预测图像。默认不写预测 NPZ，`--save-predictions` 才允许本地保存。
- **禁止以任何形式将用户实验结果同步至 GitHub**：包括 commit/push、PR/Issue 文本、Gist、GitHub Release、Actions artifact、workflow logs、文档表格或仓库的其他目录。不得把任何真实指标转写到 README、HANDOFF、AGENTS。只向用户指出本机 JSON 路径，让用户在当前 ChatGPT 网页**手动附加这个 JSON 文件**供分析。
- 强制遵守 `.gitignore`；默认启用本地保护建议：`chmod +x .githooks/pre-commit && git config core.hooksPath .githooks`。代码会拒绝把结果写到 Git 可跟踪位置（结果仍不具备加密保障）。
- 结果仅在全部病例和全部方法计算成功后原子写入；失败时不产生新的 completed JSON。不要把 debug subset 当正式统计结果，不能伪造缺失指标。
- 每次修改报告结构要增加或修改 `tests/test_reporting.py` 与结果拦截测试，确保不含图像数组、没有遗漏患者/方法、不能外泄到 GitHub。

## V2 特征选择与评价（禁止使用肿瘤真值决策）
- E0 `patch_knn`：原512个未过滤参考 Patch；E1 `body_patch_knn`：同512预算、CT体部过滤；E2 `diverse_patch_knn`：患者均衡的8192多样化 Patch；E3 `torc_approx` 和 E4 `torc_consensus`：与 E2 共用完整 KNN 库，只聚类512个训练代表 Patch，再将簇标签传播。
- E0–E4 固定相同患者拆分、冻结 MobileNetV3、窗位和 kNN K=5；不得根据测试掩码选择池化层、面积阈值或系数。可使用 body ROI，但它完全由 CT HU 确定，不能使用 GT。
- 全图和体部内的 Pixel AUROC/AP 分开报告；面积预算 Top0.1%、0.5%、1%、5% 固定，并输出 Dice/IoU/Precision/Recall/FPR 和 TP/FP/FN/TN。21点数值 PR 曲线不生成图。须记录被体部掩码排除的 GT 肿瘤像素数。
- 本地 `outputs/full/experiment_results.json` 是唯一需要回传的汇总报告，仍**严禁任何真实指标或结果通过仓库/Issue/PR/Actions/Release 同步**。

## V3：2026-10-15 进一步实验（取代V2作为默认入口）
- **默认一键运行**：`run.sh` 指向 `lightweight_clustering.v3_experiment`。V2 保留作历史对照代码，不是目前的默认实验。
- 实验总共8组：`patch_knn`、`body_patch_knn`（随机512）、`body_diverse_512`（多样性512）、`body_random_8192`（随机8192）、`diverse_patch_knn`（多样性8192）、`torc_approx`、`local_consensus_knn`、`torc_consensus`。
- 2×2析因需相同患者、相同冻结特征、相同CT体部ROI，512点必须从同方法8192点的大型候选库按患者轮转取前512点；对患者逐例计算 bank_size、diversity 的主效应及交互项；不要把同时改变库容量和采样方式的E1→E2变化直接归因于任何一个单独因素。
- 替换无效的全局Consensus：新评分为**查询Patch周围16个参考邻居**的距离加权同患者集中度，与全局TORC簇是否覆盖101人无关。独立测试局部评分、TORC评分、两者结合（共享8192库）；记录局部支持度分布，如完全无方差须说明模块没有信息增益，不能写为有效创新。
- `--sampling-seeds 42 123 2026` 为默认；**患者划分始终固定在 `--seed 42`**。不同seed仅反映参考库采样方差，不等于新患者或跨中心独立验证，更不能把25患者×3抽样等同75独立患者。
- 先 `bash run.sh --max-cases 8 --slice-stride 4 --sampling-seeds 42`，后 `bash run.sh`。输入数据和ModelScope权重由原规则提供。保持预训练网络、K=5、体部HU门限以及Top-k预算不随测试标签变动。
- 一份完整V3报告为 `outputs/full/experiment_results.json`（schema 3.0.0），包含8方法×全部采样seed的患者级指标、每seed宏平均、析因主效应/交互项、跨seed描述统计、局部患者支持诊断和完整数值PR采样。报告中**绝不允许**CT、mask、热力图、特征阵或预测像素阵。
- **严禁任何真实实验结果数据被同步到 GitHub**，包括报告JSON、患者指标、GitHub Actions logs/artifacts、Issue/PR文字、文档图表、Release。只提交公开代码/测试中的合成数字。用户只把本机JSON手动传回当前对话。
- 真实数据尚未在此版本运行，GitHub CI通过只证明无真实数据的Python/合成单测通过，不能假称V3已经取得性能提升。
