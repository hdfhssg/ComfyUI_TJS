# ComfyUI_TJS - Truncated Jump Sampling

中文 | [English](README_EN.md)

ComfyUI_TJS 是一个用于 ComfyUI 的自定义节点插件，用来实验
**TJS（Truncated Jump Sampling，截断跳步采样）**。

该插件用于验证一种基于 endpoint / denoised 预测的训练无关采样加速思路。

## 论文

```bibtex
@article{peng2026x,
  title={x-Prediction Is All You Need: Training-Free Accelerated Generation via Endpoint Decodability},
  author={Peng, Xin and Gao, Ang},
  journal={arXiv preprint arXiv:2607.06114},
  year={2026}
}
```

## 功能简介

TJS 的核心思想是：不一定要把扩散/流匹配采样轨迹完整跑完。模型在中间步已经能给出一个
`x0` / `denoised` 端点预测，因此可以先运行一部分采样步骤，再直接输出端点估计，从而减少 NFE。

简化流程如下：

```text
k* = ceil(gamma * K)
run sampler from sigma[0] to sigma[k*]
return x0_hat(xt, sigma[k*]) with one endpoint model call
```

其中：

- `K` 是原始完整采样步数。
- `gamma` 是早退比例，例如 `0.6` 表示只跑大约 60% 的轨迹。
- `k*` 是实际早退步。
- `x0_hat` 是模型在中间 latent 上预测的干净端点。

在 ComfyUI 中，endpoint estimate 对应模型包装器给出的原生 `denoised` latent，因此理论上可以兼容 diffusion、flow matching、Flux/SD3 类模型包装。

当 `gamma = 1.0` 时，节点会运行完整采样 schedule，并跳过额外 endpoint call，等价于普通 KSampler 的边界情况。

## 已测试模型

| 模型 | 类型 | 状态 |
|---|---|---|
| SDXL | 扩散模型 | 已测试 |
| SD3.5M | 流匹配模型 | 已测试 |
| Z-Image-Turbo | 流匹配模型 | 已测试 |
| Anima | 扩散模型 | 已测试 |
| Krea2 | 扩散模型 | 已测试 |
| Krea2-Turbo | 扩散模型 | 已测试 |

### 即将测试

- Qwen-Image-Edit-2511
- 扩散类的视频生成模型

## 更新日志

### 2026-07-13 修复 TJS 速度低于理论值的问题

**问题**：当 `gamma = 1.0`（退出时间设为 1）时，TJS 采样器比原版 KSampler 慢，导致 TJS 加速比例低于理论值。

**根因**：旧实现使用两次独立的采样调用来完成 TJS —— 第一次截断采样到 `sigma[k*]`，第二次单独的 endpoint decode 调用 `sample_custom(sigmas=[sigma*, 0])`。每次调用都经过完整的 ComfyUI 采样管线（创建 CFGGuider、加载模型、准备条件、清理），导致理论上的 NFE = k* + 1 中那个 "+1" 远不止一次前向传播的开销。

**修复**：改为单次采样调用。在截断 sigmas 末尾追加 0：`[sigma_0, ..., sigma_{k*}, 0]`，让采样器在同一次调用中多做一步（sigma* → 0）。通过 k-diffusion 回调在最后一步同时捕获 `denoised`（x0 预测）和 `x`（sigma* 处的状态），消除了第二次采样的全部开销。

修复后 `gamma = 1.0` 时与原版 KSampler 速度完全一致，其他 gamma 值的加速比也更接近理论值。

## 效果示意

Endpoint decodability 示意：上排是直接解码中间 noisy latent，早期噪声较重；下排是使用 endpoint / denoised 预测得到的干净端点估计。

<img src="assets/endpoint_decodability.png" alt="Endpoint decodability" width="760">

Z-Image-Turbo 在不同 NFE 下的生成质量变化：

<img src="assets/z_image_turbo_low_nfe_gallery.png" alt="Z-Image-Turbo low NFE gallery" width="760">

SDXL 30 步采样设置下的 NFE 渐进对比：

<img src="assets/sdxl_30step_nfe_progression.png" alt="SDXL 30-step NFE progression" width="760">

FID 与 NFE 的趋势示意：

<img src="assets/fid_vs_nfe_benchmark.png" alt="FID versus NFE benchmark" width="760">

## 安装

把本文件夹复制到 ComfyUI 的自定义节点目录：

```text
ComfyUI/custom_nodes/ComfyUI_TJS/
```

然后重启 ComfyUI。

## 节点说明

### TJS Sampler (Truncated Jump Sampling)

这是主要的一站式节点，可以作为普通文生图采样器的替代节点使用。

输入：

| Input | 中文说明 |
|---|---|
| `model` | ComfyUI 加载的模型 |
| `total_steps` | 原始完整采样步数 `K`，例如 `30` |
| `early_exit_gamma` | 早退比例，例如 `0.6` |
| `cfg` | CFG scale |
| `sampler_name` | 截断采样阶段使用的 ComfyUI sampler |
| `scheduler` | ComfyUI sigma scheduler |
| `positive` / `negative` | 正向/反向条件 |
| `latent_image` | 空 latent 或输入 latent |
| `seed` | 随机种子 |
| `model_type` | 信息标记：`auto`、`diffusion` 或 `flow` |
| `denoise` | 可选 denoise 强度 |

输出：

| Output | 中文说明 |
|---|---|
| `latent_x0` | endpoint decoded latent，接到 VAE Decode |
| `latent_xt` | 早退时刻的中间 noisy latent |
| `k_star` | 实际早退步 |
| `nfe_used` | 早退时为 `k_star + 1`；`gamma = 1.0` 时为 `K` |
| `nfe_saving_pct` | 相对完整步数节省的 NFE 百分比 |
| `sigma_at_exit` | endpoint decode 使用的 sigma |

### TJS Advanced Sampler (KSampler Advanced + Endpoint)

这是 KSampler Advanced 的 TJS 增强版。它在 KSampler Advanced 的基础上内置了
TJS endpoint decode，无需两节点串联即可实现截断跳步采样。

与 `TJSSampler` 相比，它额外支持：

- `add_noise` = `enable` / `disable`：控制是否添加初始噪声（用于 img2img）。
- `start_at_step`：从指定步开始采样（用于多阶段工作流）。
- `noise_seed`：与 KSampler Advanced 一致的种子参数。

输入：

| Input | 中文说明 |
|---|---|
| `model` | ComfyUI 加载的模型 |
| `add_noise` | `enable` 添加噪声（文生图），`disable` 不添加（img2img） |
| `noise_seed` | 随机种子 |
| `steps` | 完整采样步数 `K` |
| `early_exit_gamma` | 早退比例 `gamma` |
| `cfg` | CFG scale |
| `sampler_name` | ComfyUI sampler |
| `scheduler` | ComfyUI scheduler |
| `positive` / `negative` | 正向/反向条件 |
| `latent_image` | 空 latent 或输入 latent |
| `start_at_step` | 起始步（默认 `0`） |
| `model_type` | 信息标记 |

输出与 `TJSSampler` 相同：`latent_x0`、`latent_xt`、`k_star`、`nfe_used`、`nfe_saving_pct`、`sigma_at_exit`。

### TJS Custom Advanced (SamplerCustomAdvanced + Endpoint)

这是 ComfyUI 自定义采样器（高级）的 TJS 版本，镜像原生 `SamplerCustomAdvanced` 接口，
额外增加 `early_exit_gamma` 参数。支持所有 ComfyUI guider 类型（CFGGuider、BasicGuider、DualCFGGuider）。

输入与 `SamplerCustomAdvanced` 一致（noise、guider、sampler、sigmas、latent_image），加上 `early_exit_gamma`。

输出：`latent_x0`、`latent_xt`、`k_star`、`nfe_used`、`nfe_saving_pct`、`sigma_at_exit`。

## 使用示例

### 直接使用 TJS Sampler

例如设置：

```text
total_steps = 30
early_exit_gamma = 0.6
```

节点会运行到：

```text
k* = ceil(0.6 * 30) = 18
```

然后执行一次 endpoint call。总成本约为：

```text
NFE = 18 + 1 = 19
```

相对 30 步完整采样，约节省 `36.7%` NFE。

### 使用 TJS Advanced Sampler

`TJSAdvancedSampler` 是一体化节点，无需串联 KSampler Advanced：

```text
steps = 30
early_exit_gamma = 0.6
add_noise = enable
start_at_step = 0
```

节点会自动计算 `k* = 18`，运行截断采样，然后执行 endpoint decode。

对于 img2img 工作流：

```text
add_noise = disable
start_at_step = 0
```

此时不添加噪声，直接从输入 latent 开始采样。
