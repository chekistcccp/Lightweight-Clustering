# AGENTS.md — Lightweight-Clustering 科研与代码代理约束

## 核心科学要求（不可改变）
- 任务：所有训练病例为结直肠癌 CT，无独立健康患者 CT；生成像素级肿瘤异常定位分数，不得声称健康/患病二分类 AUROC。
- 主数据：MSD Task10_Colon（用户把原始压缩包放在 `data/archives/` 或 `data/`）；自动解包/索引/预处理。保留压缩包，禁止任何患者影像上传仓库。
- 患者级固定随机拆分 seed 42；不存在相同患者在训练建库及测试评价两侧。不能用 `labelsTr` 构建参考库、切片选择、超参数调优或特征过滤，**仅在评价时访问掩码**。
- 冻结的轻量 MobileNetV3；ModelScope 官方模型卡已核实 `timm/tf_mobilenetv3_small_minimal_100.in1k`。模型存 `model/`，数据 `data/`，中间缓存 `outputs/cache/`，指标 `outputs/metrics/`，均不提交 Git。
- 三组唯一主对照：`patch_knn`、`torc_approx`、`torc_consensus`；必须共享相同的特征、划分、参考库和评分后处理，不得各自偷偷调参。
- 官方 MATLAB 的 `TorqueClustering.p`、其他 .p 和 `TORC.mexw64` 不是可直接逐行迁移的开源 .m；本项目实现属于**近似独立实现，未经官方 parity 验证**。写论文不得用“严格等价”或“官方 TORC”描述。
- 本算法可能让常见肿瘤形成簇、让罕见健康解剖成为离群点；研究报告必须包括失败病例、簇患者覆盖率和异常富集证据，不能只提供好看的热力图。

## 执行习惯
- 用户**手动创建/激活 Conda** 环境、手动获取数据压缩包；脚本不 `conda create`、不 `pip install`、不升级 torch、也不运行 Slurm。
- `bash run.sh` 是唯一正式全流程入口：自检→安全解压→索引→冻结推理缓存→TORC 限规模聚类→三方法→评估→结果归档；缓存与断点复跑必须有效；依赖不完整/无数据/无法下载权重时**明确失败而非输出虚构数字**。
- 默认面向单卡 GPU（WSL）、适度显存、混合精度可选；不要引入 Qwen、扩散模型、额外骨干、伪标签网络、其他数据集或训练步骤，除非用户明确变更。
- 固定患者 seed、预处理窗、slice stride、模型 revision、torch/timm/numpy 版本及脚本 git SHA；每次结果带配置快照。比对 Pixel AUROC、Pixel AP、AUPRO（若实现）、Dice@top5%（辅助，不作为监督阈值）、运行时间与峰值显存。
- 代码应容错单切片/极小患者数/无正例像素，空预测不填零；异常只在真实数据运行后定量确认。
- 正式结果不能使用 `--max-cases` 之类调试开关；不能利用测试掩码挑最优超参或报告乐观偏差。

## 开发与交接
- 改代码前先读 README、本文件以及 `docs/HANDOFF.md`；结束后更新 HANDOFF：修改内容、测试命令/结果、当前阻塞、下一步。
- 每项变更应更新 `tests/`，有确定性小型合成数据测试；数据未提供时只允许运行 smoke/unit test，不得假称在 MSD 上通过。
- 不要同步 `data/`、`model/`、`outputs/`、`*.nii.gz`、凭据、代理配置、token、原始医学影像、conda 目录。
- 本仓库面向学术研究，研究使用 TORC 时遵守原作者署名与 CC BY-NC-SA 4.0 非商业条款。
