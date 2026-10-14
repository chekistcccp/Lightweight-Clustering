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
