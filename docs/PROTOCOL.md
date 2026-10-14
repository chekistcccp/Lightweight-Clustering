# 研究实验协议（2026-10-14）

## 原假设
仅癌症患者 CT 中的局部特征包含高频的相对正常结构。采用 TORC-inspired 簇规模及跨患者簇一致性校正，可能改进基于 KNN 的异常热图。此假设未经验证；聚类簇/离群点不等同于病灶。

## 数据与伦理
- MSD Task10_Colon 公布 imagesTr/labelsTr；全部病灶患者，不含独立健康对照。
- 固定 patient-level split (seed42, 80/20)；train 侧仅影像输入，label 仅在 eval 侧读取；特别注意 CT 切片 Z 轴取最后一维，并保证图像/掩码空间一致。
- 所有切片按规则 stride 采样，与是否有病灶无关；正式要求 stride=1，快测 stride=4。记录切片率，不得把子集指标说成完整体积指标。
- 不使用测试掩码挑选窗口、特征层、过滤阈值或超参数。若重新调参，需要独立 validation 人群并最终锁定 test。

## 模型与方法（仅单主干）
ModelScope `timm/tf_mobilenetv3_small_minimal_100.in1k`；固定 out_indices=(1,2)，2.5D 的 z-1,z,z+1 使用相同 HU soft-tissue window [-150,250]，224² 输入；多层特征池化到 14² 并归一化。
参考患者各抽至多64 Patch，再 round-robin 选择总上限512 Patch；只从 train 患者抽样。
训练：**无**；TORC 聚类：仅 coreset，使用近似质量/距离凝聚及能量跃迁估计簇数，明确不是官方等价实现。
公式 / 同一像素基线：
- `patch_knn` = mean top-k cosine distances / median train reference kNN scale；
- `torc_approx` = baseline + 0.25*(1 - nearest-cluster-size / max-cluster-size)；
- `torc_consensus` = 上述 + 0.25*(1 - nearest-cluster-distinct-patient fraction)。
分数不做 GT 依赖调优。三个公式均为我们的**工程研究假设**，不可写成官方 TORC 定义。

## 主要指标
- 病例 macro Pixel AUROC、Pixel AP（只有正负像素兼具的患者可计算），Dice@top5% 辅助展示；按患者分别输出切片数、阳性像素、无定义指标个数。
- 因所有病例阳性，不得报告患者级诊断 AUROC。典型失败病例、组织误检、未覆盖病灶需要真实统计。
- 推理计时和 CUDA peak allocated（请区分总时长含 ModelScope 下载、缓存与推理）。
- 对比 E1/E2/E3 共享参考库；不声称“仅 TORC 聚类效果”优于其他方法。
- 审稿前补充患者重采样 bootstrap CI、AUPRO（若用于主指标）、独立外部 CT 数据、分组与扰动鲁棒性。初期不会自动虚构这些结果。

## MATLAB 兼容性协议
1. 在 MATLAB 中从官方 https://github.com/JieYangBruce/TorqueClustering 获取允许使用的官方 P-code/MEX，在 MATLAB 内对小规模 `DM` 运行 `[Idx,Idx_with_noise]=TorqueClustering(DM,0,1,0)`；保存 `DM` 与 `Idx` 到 .mat。
2. 运行 `python scripts/matlab_parity.py --golden golden.mat` 对比相同距离矩阵下的 ARI、簇数。
3. 该脚本测试的是差异，不会使独立近似自动变成正式官方等价复现。只有覆盖多个 benchmark 且满足输出一致性后，才可进入严格迁移验证阶段。
4. 遵守作者 CC BY-NC-SA 4.0 和非商用许可，不复制加密 P-code/MEX，也不逆向工程。

## 性能/资源限界
N×N 内存为 O(N²)；矩阵 512×512 float32 大约1 MiB，本实现另外分配 1024×1024，并逐步凝聚，时间显著高于标准 KNN。严禁未经说明对数万 Patch 直接运行这个版本。

## 统一指标报告与隐私交付（2026-10-14 更新）
正式结果唯一需回传的文件：`outputs/full/experiment_results.json`。按病例记录各方法的 Pixel AUROC/AP、Dice/IoU/Precision/Recall/FPR@top5%、TP/FP/FN/TN；各方法按患者 macro 描述统计，配对 delta、簇结构、时间、环境、版本及数据拆分汇入同一 JSON。任何无定义指标用 JSON null，不得使用 0 或推测数值代替。
输入图像、掩码、预测图、特征张量不能出现在这一 JSON；默认不产生预测图像，显式 `--save-predictions` 只允许本机存储。
所有实验指标和日志只能保存在用户本机，**不能同步/提交 GitHub，不能被 GitHub Actions 上传**；仅由用户手动附加 JSON 到当前 ChatGPT 会话分析。
