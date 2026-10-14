# TORC MATLAB → Python 移植审核记录

## 检索结论（2026-10-14）
**当前官方仓库**：https://github.com/JieYangBruce/TorqueClustering
- 主入口 `TorqueClustering.p`、重要的 `Nab_dec.p`、`Updateljmat.p`、`Final_label.p` 是 MATLAB P-code，文本源码不可直接查看；提供 `TORC.mexw64`，只能作为匹配平台 MATLAB 的二进制调用入口，不是 Linux Python 库。
- 官方公开接口：`[Idx,Idx_with_noise]=TorqueClustering(ALL_DM,K,isnoise,isfig)`；参数 K=0 自动簇数，isnoise=1 噪声检测。
- 官方说明第三方 Python 翻译可能与 MATLAB 不同。
- 官方仓库明确标注原作者 Jie Yang 的代码归属，并使用 CC BY-NC-SA 4.0 条款约束官方代码。

**其他参考，不是官方保证**：
- 贡献者 Python 项目：https://github.com/Cognet-74/TorqueClusteringPy
- 非官方历史源代码仓库：https://github.com/HNYMZRL/TorqueClustering ，可看到较早版本的 `.m` 文本。但其仓库许可证元数据与当前官方仓库不一致；不能仅凭该镜像推定具有授权复制/再许可当前官方软件的权利。**此项目没有搬运其源码。**

## 当前实现的精确边界
`lightweight_clustering/torc_approx.py` 独立实现了：
1. 使用共享距离矩阵；
2. 每个现存簇寻找质量不少于自身的最近邻；
3. 按簇质量乘积和距离平方构造能量；
4. 选择最小能量的候选对，凝聚更新，并以能量跃迁估计簇数；
5. 在测试集 Patch 上结合簇大小和跨患者簇覆盖率作异常评分。

**与官方 TORC 的已知差异**：
- 当前使用逐对凝聚 + single-link 更新，官方算法包含逐层图连接、连通分量以及独立的 link power/剪枝细节；
- 当前自动确定簇数使用自定义的能量跃迁，**没有实现官方 Nab_dec 的计算**；
- 没有复现官方特殊噪声标记规则，输出不是官方 `Idx_with_noise`；
- 官方初始最近邻/相同距离 tie-breaking 规则、MATLAB 与 Python 的排序差异尚未做跨平台验证；
- 不支持超过1024候选点的精确 NxN 凝聚，实际默认512点。

因此：
- 项目可用于**独立 TORC-inspired 聚类研究与消融**；
- 不能标注为“完成了官方算法的严格移植”，不能将任何结果称为与官方 MATLAB 等价；
- 发表时需清楚引用原始 TORC 方法与本项目新评分设计。

## 官方结果对比
在你安装 MATLAB 的电脑，合法获取官方库后执行：

```matlab
rng(42);
X = rand(40, 8);
DM = pdist2(X, X, 'cosine');
[Idx, Idx_with_noise] = TorqueClustering(DM, 0, 1, 0);
save('golden.mat','DM','Idx','Idx_with_noise');
```

再在本项目 Conda 环境执行：

```bash
python scripts/matlab_parity.py --golden golden.mat
```

输出 `ARI_auto_vs_MATLAB`、`ARI_fixed_K_vs_MATLAB`、簇数。对比采用**同一 DM**，但当前属于不同的近似算法；一个数据集 ARI=1 并不自动证明整体一致性。

若未来要求**完全等价**，需要合法获得可阅读的官方 MATLAB 原实现或作者授权的源码、逐模块移植、把 isnoise/K/tie-break 与原 MATLAB 输出做多数据集回归测试，再将其引入 CT 实验；当前不声称完成上述工作。
