# Codex / 实验交接

## 2026-10-14 初始设计
- 研究问题：没有独立正常患者影像的 MSD Colon CT 异常定位。
- 状态：初版代码框架和 TORC-inspired 近似聚类，**尚未验证 MATLAB 一致性，尚未在真实 MSD/4090 上执行**。
- 数据：MSD Task10_Colon 原始压缩包放 `data/archives/`，自动安全解压；仅取公开有掩码的 imagesTr/labelsTr。
- 模型：ModelScope `timm/tf_mobilenetv3_small_minimal_100.in1k`；自建 Conda，外部安装 torch。
- 实验：患者级 split → 2.5D CT 归一化 → 冻结双层 CNN 特征 → 共享 coreset → KNN/TORC-inspired/TORC+跨患者支持 → 患者级保存预测和指标。
- 目前阻塞：官方核心文件为 MATLAB P-code；需在用户 MATLAB 机器单独生成 `.mat` golden outputs 完成 parity；用户尚未提供公开数据与实际 GPU 环境以跑真实端到端。
- 验收第一步：`python -m pytest -q`；第二步：`bash run.sh --max-cases 8` 调试；第三步：`bash run.sh --slice-stride 1`，核查结果数值与数据泄漏。
- 后续工作：若结果不足，先验证特征可区分肿瘤/健康组织、核实 CT 方向及实际推理时延；不要未经证明就堆叠网络模块。


## 2026-10-14 实装更新与自检
- 已提交 README.md、AGENTS.md、docs/PROTOCOL.md、docs/DATA_GUIDE.md、docs/TORC_PORTING.md、run.sh、依赖/环境自检、Python 聚类实现、完整 CT 流水线、Matlab golden comparison、synthetic tests、GitHub Actions。
- 已修复 Tensor/CUDA 指标导出引用、直接 DM 接口、支持原始嵌套归档、安全解压路径遍历、忽略 MSD 包中 macOS `._` 资源分叉文件、缓存权重修改时间指纹。
- GitHub Actions 在 commit `3368a811b184dbca1bf44590ad32eab99253b887` 显示 **success**，执行了 Python compileall 和合成单元测试。此检查未运行实际医学 CT 输入、模型权重下载、CUDA 推理和 MATLAB parity。
- **重要**：尽管存在第三方旧版文本 MATLAB .m 镜像，当前正式官方仓库主算法仍发布为 P-code/MEX；其授权与当前官方版本存在差异风险。本仓库没有复制该第三方 MATLAB 文件。当前算法是 *TORC-inspired* 而非官方严格移植。
- 下一步用户实验命令：激活自己的 Conda 环境、下载并放置 `Task10_Colon.tar` 到 `data/archives/`、自行安装相应 CUDA PyTorch，随后 `bash run.sh --max-cases 8` 做真实数据烟雾测试；无异常后 `bash run.sh --slice-stride 1` 完整运行。
- 仍需由用户在 MATLAB 环境执行官方算法并提供 `golden.mat`，方能比较真实输出；即使单例 ARI=1 也不能声称完全移植。

## 2026-10-14 用户结果回传需求（重要）
- 改成唯一可回传的指标文件 `outputs/full/experiment_results.json`（调试模式 `outputs/debug_N/experiment_results.json`），包含所有当前计算的数值指标、逐病例记录、成对差值、簇诊断、版本、环境、分组、耗时和显存。
- 默认不生成预测数组；`--save-predictions` 才可在本地保存 NPZ，不进入 JSON。
- 报告生成在**所有方法成功结束后**；不生成独立指标 CSV/summary/run_manifest 文件。之前运行留下的老文件不会自动删除。
- 新增 `reporting.py`、`tests/test_reporting.py`、`scripts/check_staged.py`、`.githooks/pre-commit`；Git 忽略策略扩展。Git hooks 启用前需用户在 WSL 执行 `chmod +x .githooks/pre-commit && git config core.hooksPath .githooks`。
- **不得将任何真实实验结果提交 GitHub、写进 Issue/PR/日志/文档，也不能上传至 Actions Artifacts；只告知用户本机结果 JSON 路径，等用户手动回传本 ChatGPT 网页。**
- 尚未在真实 MSD 数据集、ModelScope GPU 权重与 Windows WSL 环境进行完整运行；GitHub CI 仅验证代码语法及合成单元测试。


## 2026-10-14 V2 多组消融代码更新（真实运行待用户执行）

- 默认 `run.sh` 已切换为 `python -m lightweight_clustering.v2_experiment`；旧 `experiment.py` 留作 legacy 基线函数和回归测试，禁止将旧脚本的3组实验误称当前5组。
- V2 提供五组：`patch_knn`、`body_patch_knn`、`diverse_patch_knn`、`torc_approx`、`torc_consensus`。E0/1 同512参考预算；E2/3/4共用更大患者均衡、多样性采样的8192特征库；TORC只在训练库512代表点聚类后无监督传播标签。
- 新增 `lightweight_clustering/body.py`：CT HU 阈值、二值闭运算、最大连通域、填洞、patch占比过滤；GT 标签只在最终评价读取。
- 新增 `lightweight_clustering/sampling.py`：余弦最远点优先、患者均衡分配、代表集和簇标签传播。大型KNN用CUDA/CPU chunked torch matmul而非构造巨大TORC全距阵。
- 新增 `lightweight_clustering/metrics_v2.py`：全图+体部ROI的Pixel AUROC/AP及Top0.1%、0.5%、1%、5%各预算Dice/IoU/Precision/Recall/FPR、TP/FP/FN/TN、数值PR曲线。区域指标在每位患者的选定ROI内计算，别跨ROI混比。
- `reporting.py` 报告 schema 增至2.0.0，仍只输出本机一份 `outputs/full/experiment_results.json`（包含全部数值诊断，不含图像），失败不生成新的完成报告。重复运行前旧结果自动移到本机 `outputs/history/` 以防丢失。
- GitHub Actions 的合成单元测试与语法检查可以证明基本逻辑，但**没有在真实MSD Colon、ModelScope权重和用户4090上跑过端到端V2**；请用户先运行 `bash run.sh --max-cases 8 --slice-stride 4`，检查完成后运行 `bash run.sh`。
- **绝不可同步真实实验结果到GitHub**，包括 commit/PR/Issue/GitHub Actions Artifacts、logs等；只允许用户在本ChatGPT窗口上传本地JSON。
- 后续优先检查：原缓存可复用，HU体部ROI没有漏掉大比例病灶，完整结果中E2/E3/E4确实共用同一库，Patient-level指标及分区对照是否有实际增益。不要仅因全图AUROC提升即声称算法改进。
