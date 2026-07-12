"""
ComfyUI nodes for Truncated Jump Sampling (TJS).

TJS runs the ordinary sampler for k* early steps, then asks the model for the
endpoint prediction at the early-exit state. In ComfyUI/k-diffusion terms that
endpoint prediction is the model's `denoised` latent at sigma*.

Nodes provided:
  - TJSSampler              — one-shot TJS sampler (gamma-based early exit)
  - TJSAdvancedSampler      — KSampler-Advanced variant with TJS endpoint decode
  - TJSCustomAdvanced       — SamplerCustomAdvanced variant (guider/sampler/sigmas inputs)
  - TJSDecode               — endpoint decode for a KSampler-Advanced leftover latent
  - TJSDecodeManualSigma    — endpoint decode with an explicit sigma (debug)
"""

import math

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers

try:
    import comfy.utils
except Exception:  # pragma: no cover - older ComfyUI builds may differ.
    pass

import latent_preview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_enabled():
    utils = getattr(comfy, "utils", None)
    return bool(getattr(utils, "PROGRESS_BAR_ENABLED", True))


def _sampler_object(name):
    if hasattr(comfy.samplers, "sampler_object"):
        return comfy.samplers.sampler_object(name)
    return comfy.samplers.ksampler(name)


def _calculate_sigmas(model, steps, sampler_name, scheduler, denoise):
    """Compute the full sigma schedule, matching KSampler's internal logic.

    This handles ``DISCARD_PENULTIMATE_SIGMA_SAMPLERS`` (dpm_2, uni_pc, …)
    identically to ``comfy.samplers.KSampler``, so the returned sigma array
    is directly comparable to what KSampler / KSamplerAdvanced would use.
    """
    device = comfy.model_management.get_torch_device()
    if hasattr(comfy.samplers, "KSampler"):
        ks = comfy.samplers.KSampler(
            model, steps, device,
            sampler=sampler_name, scheduler=scheduler, denoise=denoise,
        )
        return ks.sigmas.to(device)

    # Fallback for very old ComfyUI builds without comfy.samplers.KSampler.
    from comfy_extras.nodes_custom_sampler import BasicScheduler

    return BasicScheduler().get_sigmas(model, scheduler, steps, denoise)[0].to(device)


def _fix_latent(model, latent):
    fixed = latent.copy()
    fixed["samples"] = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    return fixed


def _prepare_noise(latent, seed):
    samples = latent["samples"]
    batch_inds = latent.get("batch_index")
    if hasattr(comfy.sample, "prepare_noise"):
        return comfy.sample.prepare_noise(samples, seed, batch_inds)

    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(
        samples.size(),
        dtype=samples.dtype,
        layout=samples.layout,
        generator=generator,
        device="cpu",
    )


def _zeros_like_noise(latent_samples):
    return torch.zeros(
        latent_samples.size(),
        dtype=latent_samples.dtype,
        layout=latent_samples.layout,
        device="cpu",
    )


def _sample_custom(
    model,
    noise,
    cfg,
    sampler,
    sigmas,
    positive,
    negative,
    latent_image,
    noise_mask=None,
    callback=None,
    seed=0,
):
    """Call ``comfy.sample.sample_custom`` (or its older equivalent)."""
    disable_pbar = not _progress_enabled()

    if hasattr(comfy.sample, "sample_custom"):
        try:
            return comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent_image,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        except TypeError as exc:
            if "noise_mask" not in str(exc):
                raise
            return comfy.sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent_image,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )

    device = comfy.model_management.get_torch_device()
    return comfy.samplers.sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        device,
        sampler,
        sigmas,
        latent_image=latent_image,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def _sample_standard(
    model,
    latent,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    denoise,
):
    """Mirror ComfyUI's normal KSampler path for the full-schedule case."""
    latent_samples = latent["samples"]
    noise = _prepare_noise(latent, seed)
    noise_mask = latent.get("noise_mask")
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not _progress_enabled()

    try:
        return comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    except TypeError as exc:
        if "noise_mask" not in str(exc):
            raise
        return comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )


def _sample_advanced(
    model,
    latent,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    """Run KSampler-Advanced-style sampling with start / stop step control.

    Mirrors ``common_ksampler`` + ``comfy.sample.sample`` but exposes the
    ``start_step`` / ``last_step`` / ``force_full_denoise`` / ``disable_noise``
    parameters that KSamplerAdvanced uses.
    """
    latent_samples = latent["samples"]

    if disable_noise:
        noise = _zeros_like_noise(latent_samples)
    else:
        noise = _prepare_noise(latent, seed)

    noise_mask = latent.get("noise_mask")

    # Progress-bar step count: the actual number of steps that will run.
    cb_total = steps
    if last_step is not None and last_step < steps:
        cb_total = max(1, last_step - (start_step or 0))
    callback = latent_preview.prepare_callback(model, cb_total)
    disable_pbar = not _progress_enabled()

    try:
        return comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=1.0,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    except TypeError as exc:
        if "noise_mask" not in str(exc):
            raise
        return comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=1.0,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )


def _callback_x0_to_latent(model, x0_output, fallback_samples):
    """Extract the denoised x0 captured by the callback, or fall back."""
    if "x0" not in x0_output:
        return fallback_samples
    return model.model.process_latent_out(x0_output["x0"].detach().cpu())


def _make_tjs_callback(model, total_steps, capture_step, x0_output, xt_output):
    """Create a callback that wraps latent_preview's callback and captures xt.

    The k-diffusion callback provides ``denoised`` (x0 prediction) and ``x``
    (current state before step) at each step.  ``latent_preview.prepare_callback``
    already stores the last step's ``denoised`` in *x0_output*.  This wrapper
    additionally saves ``x`` at *capture_step* (the endpoint-decode step) into
    *xt_output*, so both latent_x0 and latent_xt are captured in a single pass.
    """
    base_callback = latent_preview.prepare_callback(model, total_steps, x0_output)

    def callback(step, x0, x, total):
        base_callback(step, x0, x, total)
        if step == capture_step:
            xt_output["xt"] = x.detach().clone()

    return callback


def _endpoint_decode(model, latent_xt, sigma, cfg, positive, negative, seed=0):
    """One model evaluation at *sigma*, returning the denoised endpoint (x0).

    Internally this runs a single euler step ``[sigma → 0]`` with zero noise.
    The k-diffusion callback captures the model's ``denoised`` output (the x0
    prediction at *sigma*), which is the TJS endpoint estimate.

    This mirrors the pattern used by ComfyUI's built-in ``SamplerCustom`` node
    (``x0_output`` dict + ``prepare_callback``).
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return latent_xt

    device = comfy.model_management.get_torch_device()
    sigmas = torch.tensor([sigma, 0.0], dtype=torch.float32, device=device)
    sampler = _sampler_object("euler")
    noise = _zeros_like_noise(latent_xt)
    x0_output = {}
    callback = latent_preview.prepare_callback(model, 1, x0_output)

    samples = _sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_xt,
        noise_mask=None,
        callback=callback,
        seed=seed,
    )
    return _callback_x0_to_latent(model, x0_output, samples)


# ---------------------------------------------------------------------------
# TJSSampler  (existing one-shot node, kept compatible)
# ---------------------------------------------------------------------------

class TJSSampler:
    """One-shot TJS sampler with gamma-based early exit.

    Runs a single sampling call with sigmas ``[sigma_0, ..., sigma_{k*}, 0]``
    where ``k* = ceil(gamma * steps)``.  The appended 0 makes the sampler's
    last step the endpoint decode, and the callback captures both x0 (the
    model's denoised prediction at sigma*) and xt (the state at sigma*) in
    a single pass — no separate endpoint-decode call needed.

    When ``early_exit_gamma >= 1.0`` the full schedule is used (identical to
    the standard KSampler).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 30, "min": 2, "max": 10000}),
                "early_exit_gamma": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff},
                ),
                "model_type": (
                    ["auto", "diffusion", "flow"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "latent_x0",
        "latent_xt",
        "k_star",
        "nfe_used",
        "nfe_saving_pct",
        "sigma_at_exit",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/TJS"

    def sample(
        self,
        model,
        total_steps,
        early_exit_gamma,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        seed,
        model_type="auto",
        denoise=1.0,
    ):
        latent = _fix_latent(model, latent_image)
        gamma = float(early_exit_gamma)

        # Full-schedule fast path (gamma ≈ 1.0): identical to KSampler.
        if gamma >= 0.999999:
            samples = _sample_standard(
                model, latent, seed, total_steps, cfg,
                sampler_name, scheduler, positive, negative, denoise,
            )
            out_x0 = latent.copy()
            out_x0["samples"] = samples
            out_xt = latent.copy()
            out_xt["samples"] = samples
            print(
                "[ComfyUI_TJS] "
                f"K={total_steps} gamma={early_exit_gamma:.3f} "
                "full-schedule fast path "
                f"NFE={total_steps} saving=0.0% mode={model_type}"
            )
            return (out_x0, out_xt, total_steps, total_steps, 0.0, 0.0)

        latent_samples = latent["samples"]
        noise_mask = latent.get("noise_mask")

        sigmas = _calculate_sigmas(model, total_steps, sampler_name, scheduler, denoise)
        actual_steps = max(1, int(sigmas.shape[-1]) - 1)

        k_star = max(1, math.ceil(gamma * total_steps))
        k_star = min(k_star, actual_steps)

        # If k* reaches the end of the schedule, use the full-schedule path
        # (appending 0 to sigmas that already end with 0 wastes an NFE).
        if k_star >= actual_steps:
            samples = _sample_standard(
                model, latent, seed, total_steps, cfg,
                sampler_name, scheduler, positive, negative, denoise,
            )
            out_x0 = latent.copy()
            out_x0["samples"] = samples
            out_xt = latent.copy()
            out_xt["samples"] = samples
            print(
                "[ComfyUI_TJS] "
                f"K={total_steps} gamma={early_exit_gamma:.3f} "
                f"k*={k_star}>=actual_steps={actual_steps} "
                "full-schedule path "
                f"NFE={total_steps} saving=0.0% mode={model_type}"
            )
            return (out_x0, out_xt, total_steps, total_steps, 0.0, 0.0)

        sigma_star = float(sigmas[k_star].detach().cpu())

        # Single-call TJS: append 0 to truncated sigmas so the sampler's
        # last step (sigma* → 0) is the endpoint decode.  The callback
        # captures both x0 (denoised) and xt (state at sigma*) in one pass,
        # eliminating the overhead of a separate endpoint-decode call.
        tjs_sigmas = torch.cat([
            sigmas[: k_star + 1],
            torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device),
        ])

        noise = _prepare_noise(latent, seed)
        sampler = _sampler_object(sampler_name)
        x0_output = {}
        xt_output = {}
        callback = _make_tjs_callback(
            model, k_star + 1, k_star, x0_output, xt_output,
        )

        samples = _sample_custom(
            model,
            noise,
            cfg,
            sampler,
            tjs_sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            seed=seed,
        )

        latent_x0 = _callback_x0_to_latent(model, x0_output, samples)
        if "xt" in xt_output:
            latent_xt = model.model.process_latent_out(
                xt_output["xt"].detach().cpu()
            )
        else:
            latent_xt = samples

        nfe_used = k_star + 1
        nfe_saving = max(0.0, (1.0 - nfe_used / float(total_steps)) * 100.0)

        out_x0 = latent.copy()
        out_x0["samples"] = latent_x0

        out_xt = latent.copy()
        out_xt["samples"] = latent_xt

        print(
            "[ComfyUI_TJS] "
            f"K={total_steps} gamma={early_exit_gamma:.3f} "
            f"k*={k_star} sigma*={sigma_star:.6g} "
            f"NFE={nfe_used} saving={nfe_saving:.1f}% mode={model_type}"
        )
        return (out_x0, out_xt, k_star, nfe_used, round(nfe_saving, 1), sigma_star)


# ---------------------------------------------------------------------------
# TJSAdvancedSampler  (new — KSampler Advanced + TJS endpoint decode)
# ---------------------------------------------------------------------------

class TJSAdvancedSampler:
    """KSampler-Advanced variant with built-in TJS endpoint decode.

    Runs a single sampling call with sigmas ``[sigma_start, ..., sigma_{k*}, 0]``
    where the appended 0 makes the last step the endpoint decode.  The callback
    captures both x0 (denoised at sigma*) and xt (state at sigma*) in one pass,
    avoiding the overhead of a separate endpoint-decode sampling call.

    Supports ``add_noise = disable`` for img2img and ``start_at_step > 0`` for
    multi-stage workflows — just like KSamplerAdvanced, but with TJS early-exit
    built in.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"advanced": True}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "early_exit_gamma": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "advanced": True},
                ),
                "model_type": (
                    ["auto", "diffusion", "flow"],
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "latent_x0",
        "latent_xt",
        "k_star",
        "nfe_used",
        "nfe_saving_pct",
        "sigma_at_exit",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/TJS"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        early_exit_gamma,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        model_type="auto",
    ):
        latent = _fix_latent(model, latent_image)
        gamma = float(early_exit_gamma)
        disable_noise = add_noise == "disable"

        # Full sigma schedule (identical to what KSamplerAdvanced computes).
        sigmas = _calculate_sigmas(model, steps, sampler_name, scheduler, 1.0)
        actual_steps = max(1, int(sigmas.shape[-1]) - 1)

        # k* = ceil(gamma * steps), clamped to the valid range.
        k_star = max(1, math.ceil(gamma * steps))
        k_star = min(k_star, actual_steps)

        # If start_at_step is beyond k*, adjust so at least one step runs.
        if start_at_step >= k_star:
            print(
                "[ComfyUI_TJS] WARNING: start_at_step >= k*, "
                f"adjusting start_at_step to 0 (was {start_at_step}, k*={k_star})"
            )
            start_at_step = 0

        # ---- Full-schedule fast path (gamma ≈ 1.0) ----------------------
        if gamma >= 0.999999 or k_star >= actual_steps:
            samples = _sample_advanced(
                model,
                latent,
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=None,
                force_full_denoise=True,
            )
            out_x0 = latent.copy()
            out_x0["samples"] = samples
            out_xt = latent.copy()
            out_xt["samples"] = samples
            print(
                "[ComfyUI_TJS] Advanced "
                f"K={steps} gamma={gamma:.3f} "
                f"start={start_at_step} full-schedule "
                f"NFE={actual_steps} saving=0.0% mode={model_type}"
            )
            return (out_x0, out_xt, actual_steps, actual_steps, 0.0, 0.0)

        # ---- Truncated TJS path (single-call) ---------------------------
        sigma_star = float(sigmas[k_star].detach().cpu())

        # Append 0 to truncated sigmas: [sigma_start, ..., sigma*, 0].
        # The sampler's last step (sigma* → 0) is the endpoint decode,
        # captured via callback — no separate sampling call needed.
        tjs_sigmas = torch.cat([
            sigmas[start_at_step : k_star + 1],
            torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device),
        ])

        latent_samples = latent["samples"]
        noise_mask = latent.get("noise_mask")
        if disable_noise:
            noise = _zeros_like_noise(latent_samples)
        else:
            noise = _prepare_noise(latent, noise_seed)

        sampler = _sampler_object(sampler_name)

        tjs_steps = k_star + 1 - start_at_step  # truncated + endpoint
        x0_output = {}
        xt_output = {}
        callback = _make_tjs_callback(
            model, tjs_steps, tjs_steps - 1, x0_output, xt_output,
        )

        samples = _sample_custom(
            model,
            noise,
            cfg,
            sampler,
            tjs_sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            seed=noise_seed,
        )

        latent_x0 = _callback_x0_to_latent(model, x0_output, samples)
        if "xt" in xt_output:
            latent_xt = model.model.process_latent_out(
                xt_output["xt"].detach().cpu()
            )
        else:
            latent_xt = samples

        steps_run = k_star - start_at_step
        nfe_used = tjs_steps  # = steps_run + 1
        nfe_saving = max(0.0, (1.0 - nfe_used / float(actual_steps)) * 100.0)

        out_x0 = latent.copy()
        out_x0["samples"] = latent_x0

        out_xt = latent.copy()
        out_xt["samples"] = latent_xt

        print(
            "[ComfyUI_TJS] Advanced "
            f"K={steps} gamma={gamma:.3f} "
            f"k*={k_star} sigma*={sigma_star:.6g} "
            f"start={start_at_step} steps_run={steps_run} "
            f"NFE={nfe_used} saving={nfe_saving:.1f}% mode={model_type}"
        )
        return (out_x0, out_xt, k_star, nfe_used, round(nfe_saving, 1), sigma_star)


# ---------------------------------------------------------------------------
# TJSDecode  (endpoint decode for KSampler-Advanced leftover latent)
# ---------------------------------------------------------------------------

class TJSDecode:
    """Endpoint decode node for use with KSampler Advanced.

    Workflow:
      1. KSamplerAdvanced  →  end_at_step = N, return_with_leftover_noise = enable
      2. TJSDecode         →  same steps / sampler_name / scheduler / end_at_step

    The node reconstructs ``sigma = sigmas[end_at_step]`` from the shared
    schedule parameters, then performs a single endpoint model call to obtain
    the denoised x0 prediction.

    Important: KSamplerAdvanced must have ``return_with_leftover_noise = enable``.
    If it is disabled, the output latent is already fully denoised and this
    node has nothing to decode (sigma will be 0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "end_at_step": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Must match the end_at_step used in KSampler Advanced.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "model_type": (
                    ["auto", "diffusion", "flow"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent_x0", "sigma_at_decode", "decode_step")
    FUNCTION = "decode"
    CATEGORY = "sampling/TJS"

    def decode(
        self,
        model,
        latent,
        steps,
        end_at_step,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        model_type="auto",
        seed=0,
        denoise=1.0,
    ):
        fixed = _fix_latent(model, latent)

        # Reconstruct the sigma schedule exactly as KSamplerAdvanced would.
        sigmas = _calculate_sigmas(model, steps, sampler_name, scheduler, denoise)
        actual_steps = max(1, int(sigmas.shape[-1]) - 1)

        # Clamp end_at_step to the valid range (mirrors KSampler's last_step).
        decode_step = max(0, min(int(end_at_step), actual_steps))
        sigma = float(sigmas[decode_step].detach().cpu())

        if sigma <= 0.0:
            # The leftover latent is already fully denoised — nothing to decode.
            print(
                "[ComfyUI_TJS] WARNING: sigma at decode step is 0. "
                "This means the input latent is already fully denoised. "
                "Did you forget to set return_with_leftover_noise = enable "
                "in KSampler Advanced?"
            )
            out = fixed.copy()
            out["samples"] = fixed["samples"]
            return (out, sigma, decode_step)

        latent_x0 = _endpoint_decode(
            model,
            fixed["samples"],
            sigma,
            cfg,
            positive,
            negative,
            seed=seed,
        )
        out = fixed.copy()
        out["samples"] = latent_x0
        print(
            "[ComfyUI_TJS] advanced decode "
            f"steps={steps} end_at_step={end_at_step} "
            f"decode_step={decode_step} sigma={sigma:.6g} "
            f"actual_steps={actual_steps} mode={model_type}"
        )
        return (out, sigma, decode_step)


# ---------------------------------------------------------------------------
# TJSDecodeManualSigma  (debug node)
# ---------------------------------------------------------------------------

class TJSDecodeManualSigma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "sigma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "model_type": (
                    ["auto", "diffusion", "flow"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_x0",)
    FUNCTION = "decode"
    CATEGORY = "sampling/TJS"

    def decode(
        self,
        model,
        latent,
        sigma,
        cfg,
        positive,
        negative,
        model_type="auto",
        seed=0,
    ):
        fixed = _fix_latent(model, latent)
        latent_x0 = _endpoint_decode(
            model,
            fixed["samples"],
            sigma,
            cfg,
            positive,
            negative,
            seed=seed,
        )
        out = fixed.copy()
        out["samples"] = latent_x0
        print(
            f"[ComfyUI_TJS] manual decode sigma={float(sigma):.6g} "
            f"mode={model_type}"
        )
        return (out,)


# ---------------------------------------------------------------------------
# TJSCustomAdvanced  (new — SamplerCustomAdvanced + TJS endpoint decode)
# ---------------------------------------------------------------------------

class TJSCustomAdvanced:
    """TJS for the "custom advanced sampler" paradigm.

    Mirrors ComfyUI's built-in ``SamplerCustomAdvanced`` node (the one under
    "自定义采样器(高级)" / "sampler > custom"), but adds TJS early-exit:

      k* = ceil(gamma * (len(sigmas) - 1))

    The node runs the guider with sigmas ``[sigma_0, ..., sigma_{k*}, 0]`` —
    the appended 0 makes the sampler's last step the endpoint decode, and the
    callback captures both x0 (denoised) and xt (state at sigma*) in a single
    pass.  This avoids the overhead of a second ``guider.sample()`` call.

    Inputs are identical to SamplerCustomAdvanced (noise, guider, sampler,
    sigmas, latent_image) plus ``early_exit_gamma``.

    Workflow::

        BasicScheduler → sigmas
        CFGGuider      → guider
        KSamplerSelect → sampler
        RandomNoise    → noise

        noise + guider + sampler + sigmas + latent_image
            → TJSCustomAdvanced
                → latent_x0 (endpoint decoded, feed to VAE Decode)
                → latent_xt (truncated sampling output)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "Noise source (e.g. Random Noise)."}),
                "guider": ("GUIDER", {"tooltip": "Guider object (e.g. CFGGuider, BasicGuider)."}),
                "sampler": ("SAMPLER", {"tooltip": "Sampler object (e.g. from KSamplerSelect)."}),
                "sigmas": ("SIGMAS", {"tooltip": "Sigma schedule (e.g. from BasicScheduler)."}),
                "latent_image": ("LATENT",),
                "early_exit_gamma": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "TJS early-exit ratio. k* = ceil(gamma * steps).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "latent_x0",
        "latent_xt",
        "k_star",
        "nfe_used",
        "nfe_saving_pct",
        "sigma_at_exit",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/TJS"

    def sample(self, noise, guider, sampler, sigmas, latent_image, early_exit_gamma):
        gamma = float(early_exit_gamma)

        # Fix latent channels using the guider's model_patcher.
        model_patcher = guider.model_patcher
        latent = latent_image.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model_patcher, latent["samples"],
        )
        latent["samples"] = latent_samples
        noise_mask = latent.get("noise_mask")

        # Handle empty sigmas.
        sigmas_list = sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas)
        if len(sigmas_list) == 0:
            out = latent.copy()
            print("[ComfyUI_TJS] Custom Advanced: empty sigmas, returning input")
            return (out, out, 0, 0, 0.0, 0.0)

        total_steps = len(sigmas_list) - 1
        if total_steps <= 0:
            out = latent.copy()
            print("[ComfyUI_TJS] Custom Advanced: no steps to run")
            return (out, out, 0, 0, 0.0, 0.0)

        disable_pbar = not _progress_enabled()

        # ---- Full-schedule fast path (gamma ≈ 1.0) ----------------------
        if gamma >= 0.999999 or total_steps <= 1:
            noise_tensor = noise.generate_noise(latent)
            x0_output = {}
            callback = latent_preview.prepare_callback(
                model_patcher, total_steps, x0_output,
            )
            samples = guider.sample(
                noise_tensor, latent_samples, sampler, sigmas,
                denoise_mask=noise_mask, callback=callback,
                disable_pbar=disable_pbar, seed=noise.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())

            out_xt = latent.copy()
            out_xt["samples"] = samples

            if "x0" in x0_output:
                x0_out = model_patcher.model.process_latent_out(
                    x0_output["x0"].detach().cpu(),
                )
                out_x0 = latent.copy()
                out_x0["samples"] = x0_out
            else:
                out_x0 = out_xt

            out_x0.pop("downscale_ratio_spacial", None)
            out_x0.pop("downscale_ratio_temporal", None)
            out_xt.pop("downscale_ratio_spacial", None)
            out_xt.pop("downscale_ratio_temporal", None)

            print(
                f"[ComfyUI_TJS] Custom Advanced: full-schedule "
                f"K={total_steps} NFE={total_steps}"
            )
            return (out_x0, out_xt, total_steps, total_steps, 0.0, 0.0)

        # ---- Compute k* and sigma* --------------------------------------
        k_star = max(1, math.ceil(gamma * total_steps))
        k_star = min(k_star, total_steps)

        # If k* reaches the end of the schedule, the appended 0 would be
        # redundant (sigmas already ends with 0).  Use the full-schedule
        # path instead to avoid wasting an NFE on a σ=0 → σ=0 step.
        if k_star >= total_steps:
            noise_tensor = noise.generate_noise(latent)
            x0_output = {}
            callback = latent_preview.prepare_callback(
                model_patcher, total_steps, x0_output,
            )
            samples = guider.sample(
                noise_tensor, latent_samples, sampler, sigmas,
                denoise_mask=noise_mask, callback=callback,
                disable_pbar=disable_pbar, seed=noise.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())

            out_xt = latent.copy()
            out_xt["samples"] = samples

            if "x0" in x0_output:
                x0_out = model_patcher.model.process_latent_out(
                    x0_output["x0"].detach().cpu(),
                )
                out_x0 = latent.copy()
                out_x0["samples"] = x0_out
            else:
                out_x0 = out_xt

            out_x0.pop("downscale_ratio_spacial", None)
            out_x0.pop("downscale_ratio_temporal", None)
            out_xt.pop("downscale_ratio_spacial", None)
            out_xt.pop("downscale_ratio_temporal", None)

            print(
                f"[ComfyUI_TJS] Custom Advanced: full-schedule "
                f"K={total_steps} NFE={total_steps}"
            )
            return (out_x0, out_xt, total_steps, total_steps, 0.0, 0.0)

        sigma_star = float(sigmas_list[k_star])

        # ---- Single-call TJS (truncated + endpoint decode) -------------
        # Append 0 to truncated sigmas so the sampler's last step
        # (sigma* → 0) is the endpoint decode, captured via callback.
        # This eliminates the overhead of a second guider.sample() call
        # (model loading, condition preparation, cleanup) for what is
        # just a single forward pass.
        tjs_sigmas = torch.cat([
            sigmas[:k_star + 1],
            torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device),
        ])

        noise_tensor = noise.generate_noise(latent)
        x0_output = {}
        xt_output = {}
        callback = _make_tjs_callback(
            model_patcher, k_star + 1, k_star, x0_output, xt_output,
        )

        samples = guider.sample(
            noise_tensor, latent_samples, sampler, tjs_sigmas,
            denoise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        latent_x0_tensor = _callback_x0_to_latent(
            model_patcher, x0_output, samples,
        )
        if "xt" in xt_output:
            latent_xt_tensor = model_patcher.model.process_latent_out(
                xt_output["xt"].detach().cpu(),
            )
        else:
            latent_xt_tensor = samples

        nfe_used = k_star + 1
        nfe_saving = max(0.0, (1.0 - nfe_used / float(total_steps)) * 100.0)

        # Build output dicts.
        out_x0 = latent.copy()
        out_x0["samples"] = latent_x0_tensor

        out_xt = latent.copy()
        out_xt["samples"] = latent_xt_tensor

        # Clean up downscale ratio attributes if present.
        out_x0.pop("downscale_ratio_spacial", None)
        out_x0.pop("downscale_ratio_temporal", None)
        out_xt.pop("downscale_ratio_spacial", None)
        out_xt.pop("downscale_ratio_temporal", None)

        print(
            f"[ComfyUI_TJS] Custom Advanced: "
            f"K={total_steps} gamma={gamma:.3f} "
            f"k*={k_star} sigma*={sigma_star:.6g} "
            f"NFE={nfe_used} saving={nfe_saving:.1f}%"
        )
        return (out_x0, out_xt, k_star, nfe_used, round(nfe_saving, 1), sigma_star)


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "TJSSampler": TJSSampler,
    "TJSAdvancedSampler": TJSAdvancedSampler,
    "TJSCustomAdvanced": TJSCustomAdvanced,
    "TJSDecode": TJSDecode,
    "TJSDecodeManualSigma": TJSDecodeManualSigma,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TJSSampler": "TJS Sampler (Truncated Jump Sampling)",
    "TJSAdvancedSampler": "TJS Advanced Sampler (KSampler Advanced + Endpoint)",
    "TJSCustomAdvanced": "TJS Custom Advanced (SamplerCustomAdvanced + Endpoint)",
    "TJSDecode": "TJS Decode (Endpoint / Advanced KSampler)",
    "TJSDecodeManualSigma": "TJS Decode (Manual Sigma)",
}
