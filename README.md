# SR — Refine, Don't Commit

这是 GRN_S5 当前保留的 **S5 + C + R，κ=2** 版本源码快照。
模型核心源码与对应 C/R 训练记录中的 SHA256 一致；本次发布没有改动模型结构、损失或采样算法。

本仓库仅包含代码、必要配置和许可证。**不包含模型权重、训练/测试图像、教师缓存、生成结果、论文图稿或运行环境二进制。**
它不是开箱即用的完整离线实验包；仅克隆本仓库不能直接生成图像。

## 模型

- LR 128×128 → HR 512×512；LR 结构编码器和冻结 DINOv2 条件。
- GRN2b：28 层、2304 维、18 heads；4 个 HBQ 位平面、64×32×32 latent。
- 50 轮完整 bit 重预测，C 为已有校准器的轨迹适配，κ=2。
- R：6 个残差块，588,672 参数，仅在最后解码前修正连续 latent。
- 未启用额外 Qwen/DINO 条件适配器；不包含后续未接受的 Gram/Patch/R16K 训练方案。

当前采用版本并不代表所有指标最优。模型与权重身份见 [`configs/best_model.json`](configs/best_model.json)。
权重 SHA256 只用于标识，不意味着权重已发布。

## 源码位置

| 路径 | 内容 |
|---|---|
| `code/grn/` | GRN、HBQ、LR 条件、校准器和末端 R，以及训练与推理工具 |
| `code/tools/` | C/R 适配、教师缓存、推理 pipeline 及其本地依赖 |
| `code/t2iv_train.py` | S5 基座训练入口 |
| `scripts/` | 项目级推理、训练、C/R 适配与八指标评测入口 |
| `protocol/` | 原退化与评测实现及其依赖；保留原许可证 |
| `configs/stage_c.json` | C 阶段实际参数；16,384 更新，2,048 来源 |
| `configs/stage_r.json` | R 阶段实际参数；4,096 更新，κ固定2 |
| `manifests/` | 运行路径映射、采样协议和原S5训练参数；不含训练图身份清单 |
| `environment/` | 原环境启动脚本和依赖文本；不含Python、GPU库或已安装包 |

## 使用前需自行恢复的资产

原入口保持项目的资产完整性检查，未为上传源码而跳过检查。需要从持有的完整实验包恢复相应资产：

1. `weights/`：S5、HBQ、DINOv2、固定空文本特征，以及所需初始化权重。
2. `outputs/trajectory_expanded_20261108/train_cr/adapter.pt`：本版本 C+R 权重；仅在本地放置，Git忽略该目录。
3. `protocol/data/`：指定训练或评测数据；评分另需 `protocol/offline_cache/` 中的指标权重。
4. 训练的数据资格/来源清单、教师缓存；推理完整性入口所需的原包 `verification/copy/` 与文件校验记录。
5. 匹配的 Python/GPU 环境。当前实验使用 MACA PyTorch；`environment/python` 是原包启动器，不能在缺少原环境目录时直接工作。

`code/requirements.txt` 为上游依赖声明；`environment/requirements-maca.txt` 与 lock 文件记录本实验环境。
不要把 MACA 厂商 PyTorch wheel 当成 NVIDIA 通用安装包。这里没有新增未经验证的跨平台安装流程。

恢复完整资产和环境之后，在仓库根目录执行原入口：

```bash
export GRN_S5_DETERMINISTIC=1
./environment/python scripts/infer.py --preflight
./environment/python scripts/infer.py --smoke \
  --adapter outputs/trajectory_expanded_20261108/train_cr/adapter.pt \
  --kappa 2 --output outputs/cr_smoke_new
```

上述命令不是“仅clone源码即可运行”的承诺。正式入口、配置和冻结模型源码均保留原样，方便与已有实验包核对。

## 实验性：质量对比修订引导（默认关闭）

用于提升无参考指标（MANIQA / CLIPIQA / MUSIQ）的可选推理控制。**不传任何 `--qg-*` / `--plane-temperatures` 参数时，采样语句与生成签名字段均与原版一致**（源码哈希会因文件改动而变化）。尚未在权重上验证，不附带任何质量结论。

每轮对全部 bit 的 log-odds 做有界外推，远离一个“低质量”预测：

```
l_g = l_pos + clamp(w * (l_pos - l_neg), -bound, +bound) * plane_weight
```

| 负分支 | 来源 | 额外开销 | 需要训练 |
|---|---|---|---|
| `degraded_lr` | 同一模型 + 再退化的 LR（高斯模糊 / 下采样再上采样），占用原本被丢弃的第二条打包分支 | 无额外主干 token | 否 |
| `calibrator` | 负校准器 C-（与 C 同结构，337K 参数），读同一份 hidden | 无额外主干前向 | 是，仅 C- |

```bash
# 1) 免训练：再退化 LR 作为负分支，只引导低两个位平面
./environment/python scripts/infer.py --benchmarks RealSR --indices 0,1,2,3 \
  --adapter outputs/trajectory_expanded_20261108/train_cr/adapter.pt --kappa 2 \
  --qg-scale 2 --qg-bound 2 --qg-start 0.3 --qg-planes 0,0,1,1 \
  --qg-negative degraded_lr --qg-lr-sigma 1.2 --output outputs/qg_lr_w2

# 2) 免训练：分位平面采样（0 = argmax；可与引导同时使用）
./environment/python scripts/infer.py --benchmarks RealSR --indices 0,1,2,3 \
  --adapter outputs/trajectory_expanded_20261108/train_cr/adapter.pt --kappa 2 \
  --plane-temperatures 0,0,0.7,1 --output outputs/plane_t

# 3) 训练负校准器 C-（复用已有教师缓存，不跑主干），再用它做引导
./environment/python scripts/train_negative.py --cache <teacher_cache>/cache.json \
  --adapter outputs/trajectory_expanded_20261108/train_cr/adapter.pt \
  --target blur --blur-sigma 2.0 --output outputs/negative_blur2 --execute
./environment/python scripts/infer.py --benchmarks RealSR --indices 0,1,2,3 \
  --adapter outputs/trajectory_expanded_20261108/train_cr/adapter.pt --kappa 2 \
  --qg-scale 2 --qg-negative calibrator \
  --qg-negative-adapter outputs/negative_blur2/negative.pt --output outputs/qg_cneg_w2
```

数学自检（CPU，无需权重）：`./environment/python code/tools/test_sr_quality_guidance.py`。

注意：引导会移动感知—失真工作点，PSNR / LPIPS 预期下降；启用后必须作为单独工作点报告并披露全部 `--qg-*` 设置。`cfg_list` 仍固定为 1，零前缀无条件分支从未训练，本功能不使用它。

## 许可证

GRN原MIT许可证保留于根目录和 `code/LICENSE`；协议/相关上游代码的原许可证保留于 `protocol/LICENSE`。
训练数据和权重不在本仓库中，其授权不因源码发布而改变。
