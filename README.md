# ComfyUI_TJS - Truncated Jump Sampling

中文 | [English](README_EN.md)

ComfyUI_TJS 是一个用于 ComfyUI 的自定义节点插件，用来实验
**TJS（Truncated Jump Sampling，截断跳步采样）**。

该插件用于验证一种基于 endpoint / denoised 预测的训练无关采样加速思路。

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

## 安装

把本文件夹复制到 ComfyUI 的自定义节点目录：

```text
ComfyUI/custom_nodes/ComfyUI_TJS/
```

然后重启 ComfyUI。

## 节点说明

### TJS Sampler (Truncated Jump Sampling)

这是主要可用节点，可以作为普通文生图采样器的替代节点使用。

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

### TJS Decode (Endpoint / Advanced KSampler)

状态：实验性 / TODO。

这个节点目前还没有完全写对，暂时不要把它当成可靠正式功能。它的目标用法是配合
KSampler Advanced：先让 KSampler Advanced 跑到某个中间 `end_at_step`，保留 leftover noisy latent，
再用 TJS Decode 把这个中间状态转换成 endpoint `x0` 预测。

后续需要继续修复和验证的问题：

- 如何严格对齐 KSampler Advanced 的中间 latent 与对应 sigma。
- 如何保证不同 sampler / scheduler 下 endpoint decode 的输入缩放完全正确。
- 如何在工作流里可靠传递中间步信息，而不是让用户手动猜连续时间 `t`。

当前节点设计为：用户不需要知道连续时间 `t` 或 sigma，只需要填入和 KSampler Advanced 相同的
`steps`、`sampler_name`、`scheduler`、`end_at_step`，节点内部尝试重建：

```text
sigma* = sigmas[end_at_step]
```

注意：在 KSampler Advanced 中，需要设置：

```text
return_with_leftover_noise = enable
```

如果关闭 leftover noise，高级采样器会强制采到 sigma=0，输出已经是完整去噪 latent，此时就没有可供 TJS decode 的中间 noisy endpoint state。

### TJS Decode (Manual Sigma)

调试节点。只有当你已经明确知道某个中间 latent 对应的精确 sigma 时才建议使用。

普通 Advanced KSampler 工作流不要优先用这个节点。

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

### 计划中的 Advanced KSampler 用法

这部分仍是 TODO，当前 TJS Decode 节点还需要修复。

目标工作流如下：

1. 使用 KSampler Advanced。
2. 设置 `steps = 30`，`end_at_step = 18`。
3. 设置 `return_with_leftover_noise = enable`。
4. 把 KSampler Advanced 输出的中间 latent 接到 `TJS Decode (Endpoint / Advanced KSampler)`。
5. TJS Decode 使用相同的 `steps`、`end_at_step`、`sampler_name`、`scheduler`。
6. 把 `latent_x0` 接到 VAE Decode。

## 当前限制

- `TJS Sampler` 是当前主要实现。
- `TJS Decode (Endpoint / Advanced KSampler)` 仍是实验节点，需要后续修复。
- 不同模型、sampler、scheduler 对 latent 缩放和 denoised 语义可能不同，实际效果需要逐模型验证。
- 当前插件更适合作为算法验证和 ComfyUI 原型实验，不建议直接用于严肃生产流程。
