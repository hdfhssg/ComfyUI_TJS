# ComfyUI_TJS - Truncated Jump Sampling

[中文](README.md) | English

ComfyUI_TJS is a custom node plugin for ComfyUI, built to experiment with
**TJS (Truncated Jump Sampling)**.

This plugin explores a training-free sampling acceleration idea based on
endpoint / denoised prediction.

## What It Does

TJS is based on a simple idea: diffusion and flow-matching samplers may not
need to traverse the full trajectory. At an intermediate step, the model
already produces an `x0` / `denoised` endpoint estimate. TJS runs only part of
the sampling trajectory, then uses that endpoint prediction as the final latent.

Simplified algorithm:

```text
k* = ceil(gamma * K)
run sampler from sigma[0] to sigma[k*]
return x0_hat(xt, sigma[k*]) with one endpoint model call
```

Where:

- `K` is the full sampling step budget.
- `gamma` is the early-exit ratio, for example `0.6`.
- `k*` is the actual early-exit step.
- `x0_hat` is the clean endpoint estimate predicted from the intermediate latent.

In ComfyUI, this endpoint estimate corresponds to the model wrapper's native
`denoised` latent. In principle, this makes the approach usable with diffusion,
flow-matching, Flux/SD3-style model wrappers, and related models.

When `gamma = 1.0`, the node runs the full sampling schedule and skips the
extra endpoint call, matching the ordinary KSampler boundary case.

## Installation

Copy this folder into ComfyUI's custom node directory:

```text
ComfyUI/custom_nodes/ComfyUI_TJS/
```

Then restart ComfyUI.

## Nodes

### TJS Sampler (Truncated Jump Sampling)

This is the main usable node. It can be used as an experimental replacement for
a normal text-to-image sampler.

Inputs:

| Input | Description |
|---|---|
| `model` | Loaded ComfyUI model |
| `total_steps` | Full sampling step budget `K`, for example `30` |
| `early_exit_gamma` | Early-exit ratio, for example `0.6` |
| `cfg` | CFG scale |
| `sampler_name` | ComfyUI sampler used for the truncated trajectory |
| `scheduler` | ComfyUI sigma scheduler |
| `positive` / `negative` | Positive / negative conditioning |
| `latent_image` | Empty latent or input latent |
| `seed` | Random seed |
| `model_type` | Informational selector: `auto`, `diffusion`, or `flow` |
| `denoise` | Optional denoise strength |

Outputs:

| Output | Description |
|---|---|
| `latent_x0` | Endpoint decoded latent; connect this to VAE Decode |
| `latent_xt` | Intermediate noisy latent at the early-exit step |
| `k_star` | Actual early-exit step |
| `nfe_used` | `k_star + 1` for early exit, or `K` when `gamma = 1.0` |
| `nfe_saving_pct` | NFE saving percentage relative to full sampling |
| `sigma_at_exit` | Sigma used by the endpoint decode |

### TJS Decode (Endpoint / Advanced KSampler)

Status: experimental / TODO.

This node is not fully correct yet. Do not treat it as a reliable production
feature. The intended future workflow is to pair it with KSampler Advanced:
run KSampler Advanced to an intermediate `end_at_step`, keep the leftover noisy
latent, then use TJS Decode to convert that intermediate state into an endpoint
`x0` prediction.

Open issues to fix:

- Strictly align the KSampler Advanced intermediate latent with the correct sigma.
- Ensure the input scaling for endpoint decode is correct across samplers and schedulers.
- Pass intermediate step information reliably through the workflow instead of asking users to guess continuous time `t`.

The current node attempts to infer:

```text
sigma* = sigmas[end_at_step]
```

from the same `steps`, `sampler_name`, `scheduler`, and `end_at_step` used by
KSampler Advanced.

Important: in KSampler Advanced, use:

```text
return_with_leftover_noise = enable
```

If leftover noise is disabled, KSampler Advanced forces the final sigma to zero,
so the output is already fully denoised and there is no intermediate noisy state
left for TJS Decode.

### TJS Decode (Manual Sigma)

Debug-only node. Use it only when you already know the exact sigma for an
intermediate latent.

For normal Advanced KSampler workflows, prefer the planned
`TJS Decode (Endpoint / Advanced KSampler)` workflow after it is fixed.

## Example

### Direct TJS Sampler Usage

Example settings:

```text
total_steps = 30
early_exit_gamma = 0.6
```

The node runs to:

```text
k* = ceil(0.6 * 30) = 18
```

Then it performs one endpoint call:

```text
NFE = 18 + 1 = 19
```

Compared with full 30-step sampling, this saves about `36.7%` NFE.

### Planned Advanced KSampler Workflow

This part is still TODO because the current TJS Decode node needs further
repair and validation.

Target workflow:

1. Use KSampler Advanced.
2. Set `steps = 30` and `end_at_step = 18`.
3. Set `return_with_leftover_noise = enable`.
4. Feed the intermediate latent into `TJS Decode (Endpoint / Advanced KSampler)`.
5. Use the same `steps`, `end_at_step`, `sampler_name`, and `scheduler` in TJS Decode.
6. Connect `latent_x0` to VAE Decode.

## Current Limitations

- `TJS Sampler` is the main implemented path.
- `TJS Decode (Endpoint / Advanced KSampler)` is still experimental and needs future repair.
- Latent scaling and `denoised` semantics may differ across models, samplers, and schedulers.
- This plugin is intended for research prototyping and ComfyUI experiments, not production workflows.
