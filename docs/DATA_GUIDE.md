# 数据与权重准备（仅需原始压缩包）

## 1. MSD Task10 Colon（结肠癌 CT）
权威入口：
- [MSD 官方 AWS 下载页面](https://medicaldecathlon.com/dataaws/)：选择 `Task10_Colon`；
- [AWS Open Data Registry: MSD](https://registry.opendata.aws/msd/)：公开、可匿名读取的 `s3://msd-for-monai/`；说明 AWS CLI 无需账户即可列出；
- [Grand Challenge 入口](https://decathlon-10.grand-challenge.org/)；
- 授权：数据为 CC BY-SA 4.0，学术使用需引用 Medical Segmentation Decathlon 数据论文。

用户只需在浏览器/官方提供的下载渠道获取 `Task10_Colon.tar` 或官方同等原始压缩包，**不解压、不改名**，放：

```text
Lightweight-Clustering/
  data/
    archives/
      Task10_Colon.tar
```

然后 `bash run.sh` 会自动安全解压、识别嵌套路径并匹配 `imagesTr/*.nii.gz` 与 `labelsTr/*.nii.gz`。自带的官方 `imagesTs/` 没有公开的官方测试掩码，不能用于直接计算分割指标。本项目对 126 个已公开标注的 train cases 做固定患者级随机拆分。

注意某些镜像包可能存在 macOS 创建的 `._colon_001.nii.gz` 资源分叉伪文件；代码会自动忽略，避免把它们算成真正患者 CT。

当目录里已存在解包后的 `Task10_Colon/imagesTr` 和 `labelsTr` 时，同样可以直接识别，不要求重新压缩。

## 2. 预训练轻量骨干
已检索到的 ModelScope 模型仓库：
- [timm/tf_mobilenetv3_small_minimal_100.in1k](https://modelscope.cn/models/timm/tf_mobilenetv3_small_minimal_100.in1k)
- 2M 级参数模型，ImageNet 预训练，仅用冻结特征。
- `bash run.sh` 首次运行通过 `modelscope.snapshot_download` 下载到 `model/timm__tf_mobilenetv3_small_minimal_100.in1k/`。
- 手动离线准备时，将该 ModelScope 仓库原始文件完整复制到上述目录，并传入 `--offline`；权重必须匹配本项目 timm 骨干，否则立即报错，不允许随机初始化。
- `model/` 已被 gitignore 排除。

## 3. 硬件与依赖
- Windows WSL (Ubuntu) + 单张 CUDA GPU（建议用户自己的 RTX 4090 48GB）。
- 用户自己 `conda create/activate`，安装与 CUDA 驱动匹配的 `torch` / `torchvision`，其余使用 `pip install -r requirements.txt`；脚本自己只读检查环境，不自动安装。
- 初次 `bash run.sh --max-cases 8` 走通流程；正式实验 `bash run.sh --slice-stride 1`。
- 全部源码、标注使用协议和学术限制见 `AGENTS.md`、`docs/PROTOCOL.md`。

## 4. 数据审核 / 防止结果混淆
- 一位患者仅对应一个 3D CT volume，`imagesTr` 和 `labelsTr` 同名。
- ROI mask 仅用于最终统计，不进入参考特征库和聚类流程。
- `outputs/full/split.json` 固定病例划分；调试 `outputs/debug_N` 不污染正式结果。
- 每次评估报告被抽样的切片总数以及包含阳性像素的案例数。没有真实指标时不能编造。
