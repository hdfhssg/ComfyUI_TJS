"""
ComfyUI nodes for Truncated Jump Sampling (TJS).

TJS runs the ordinary sampler for k* early steps, then asks the model for the
endpoint prediction at the early-exit state. In ComfyUI/k-diffusion terms that
endpoint prediction is the model's `denoised` latent at sigma*.
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


def _progress_enabled():
    utils = getattr(comfy, "utils", None)
    return bool(getattr(utils, "PROGRESS_BAR_ENABLED", True))


def _sampler_object(name):
    if hasattr(comfy.samplers, "sampler_object"):
        return comfy.samplers.sampler_object(name)
    return comfy.samplers.ksampler(name)


def _calculate_sigmas(model, steps, sampler_name, scheduler, denoise):
    device = comfy.model_management.get_torch_device()
    if hasattr(comfy.samplers, "KSampler"):
        sampler = comfy.samplers.KSampler(
            model,
            steps,
            device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )
        return sampler.sigmas.to(device)

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


def _callback_x0_to_latent(model, x0_output, fallback_samples):
    if "x0" not in x0_output:
        return fallback_samples
    return model.model.process_latent_out(x0_output["x0"].detach().cpu())


def _endpoint_decode(model, latent_xt, sigma, cfg, positive, negative, seed=0):
    """One model evaluation at sigma, returning ComfyUI's denoised endpoint."""
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


class TJSSampler:
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
        if gamma >= 0.999999:
            samples = _sample_standard(
                model,
                latent,
                seed,
                total_steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                denoise,
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
        sigma_star = float(sigmas[k_star].detach().cpu())

        trunc_sigmas = sigmas[: k_star + 1].clone()
        noise = _prepare_noise(latent, seed)
        sampler = _sampler_object(sampler_name)
        callback = latent_preview.prepare_callback(model, k_star)

        latent_xt = _sample_custom(
            model,
            noise,
            cfg,
            sampler,
            trunc_sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            seed=seed,
        )

        if sigma_star <= 0.0 or k_star >= actual_steps:
            latent_x0 = latent_xt
            nfe_used = k_star
        else:
            latent_x0 = _endpoint_decode(
                model,
                latent_xt,
                sigma_star,
                cfg,
                positive,
                negative,
                seed=seed,
            )
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


class TJSDecode:
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
                        "tooltip": "Use the same end_at_step as KSampler Advanced.",
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
        sigmas = _calculate_sigmas(model, steps, sampler_name, scheduler, denoise)
        actual_steps = max(1, int(sigmas.shape[-1]) - 1)
        decode_step = max(0, min(int(end_at_step), actual_steps))
        sigma = float(sigmas[decode_step].detach().cpu())

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
            f"decode_step={decode_step} sigma={sigma:.6g} mode={model_type}"
        )
        return (out, sigma, decode_step)


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


NODE_CLASS_MAPPINGS = {
    "TJSSampler": TJSSampler,
    "TJSDecode": TJSDecode,
    "TJSDecodeManualSigma": TJSDecodeManualSigma,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TJSSampler": "TJS Sampler (Truncated Jump Sampling)",
    "TJSDecode": "TJS Decode (Endpoint / Advanced KSampler)",
    "TJSDecodeManualSigma": "TJS Decode (Manual Sigma)",
}
