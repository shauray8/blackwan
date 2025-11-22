import sys
if sys.version_info < (3, 12):
    import typing
    if not hasattr(typing, 'override'):
        typing.override = lambda f: f

import torch
torch.set_grad_enabled(False)
from safetensors.torch import load_file
from omegaconf import OmegaConf
from pathlib import Path
import time
from tqdm import tqdm
import traceback

from v2v import encode_video_latent, get_denoising_schedule
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.vae_block3 import VAEEncoderWrapper, VAEDecoderWrapper
from pipeline import CausalInferencePipeline
from wan.modules.vae import WanVAE
from utils.misc import AtomicCounter
import gc
from quack.linear import Linear as QuackLinear
from quack.linear import linear_func

class UntunedQuackLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            bias_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @torch.compiler.disable
    def forward(self, x):
        return linear_func(
            x, self.weight, self.bias,
            fuse_grad_accum=False,
            tuned=False
        )
        
class Models:
    def __init__(self, text_encoder, transformer, pipeline, vae_encoder, vae_decoder):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.pipeline = pipeline
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

def replace_transformer_mlp_linears(model, name_prefix=""):
    replaced = 0
    for name, child in model.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, torch.nn.Linear):
            in_f, out_f = child.in_features, child.out_features
            is_mlp = (
                (in_f == 5120 and out_f == 13824) or
                (in_f == 13824 and out_f == 5120)
            )
            if is_mlp:
                print(f"Replacing MLP: {full_name} | {in_f}→{out_f}")
                new_module = UntunedQuackLinear(
                    in_features=in_f,
                    out_features=out_f,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                with torch.no_grad():
                    new_module.weight.copy_(child.weight)
                    if child.bias is not None:
                        new_module.bias.copy_(child.bias)
                setattr(model, name, new_module)
                replaced += 1
        else:
            replaced += replace_transformer_mlp_linears(child, full_name)
    return replaced

def load_merge_config(config_path: str | Path) -> OmegaConf:
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    return OmegaConf.merge(default_config, config)

def load_text_encoder():
    """Load original text encoder (no replacement)"""
    text_encoder = WanTextEncoder()
    text_encoder.eval().to(dtype=torch.bfloat16).requires_grad_(False)
    return text_encoder.to(torch.cuda.current_device())

def load_transformer(config):
    checkpoint_path = config.checkpoint_path
    state_dict = load_file(checkpoint_path, device="cuda")
    model_name = "Wan2.1-T2V-1.3B" if state_dict["model.blocks.0.self_attn.k.weight"].shape[0] == 1536 else "Wan2.1-T2V-14B"
    timestep_shift = getattr(config, "timestep_shift", 5.0)
    transformer = WanDiffusionWrapper(model_name=model_name, timestep_shift=timestep_shift, is_causal=True)
    transformer.load_state_dict(state_dict)
    transformer = transformer.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    transformer.to(torch.cuda.current_device())

    # Fuse projections (as in original)
    for block in transformer.model.blocks:
        block.self_attn.fuse_projections()

    # Disable FP8 (using Quack instead)
    if getattr(config, "enable_fp8", False):
        print("Skipping FP8 (using QuackLinear in bfloat16)")
        config.enable_fp8 = False

    # Replace ONLY transformer linears
    total = replace_transformer_mlp_linears(transformer, "transformer")
    print(f"Replaced {total} Linear layers in transformer with QuackLinear (bfloat16)")

    return transformer

def load_vae():
    vae_path = "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    vae = WanVAE(vae_pth=vae_path, dtype=torch.float16)
    vae_encoder = VAEEncoderWrapper(vae).eval().to(dtype=torch.float16).requires_grad_(False).to("cuda")
    vae_decoder = VAEDecoderWrapper()
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    decoder_state_dict = {k: v for k, v in vae_state_dict.items() if 'decoder.' in k or 'conv2' in k}
    vae_decoder.load_state_dict(decoder_state_dict, strict=False)
    vae_decoder.eval().to(dtype=torch.float16).requires_grad_(False).to("cuda")
    return vae_encoder, vae_decoder

def load_pipeline(config, device, transformer, text_encoder, vae_decoder):
    return CausalInferencePipeline(config, device=device, generator=transformer, text_encoder=text_encoder, vae=vae_decoder)

def compile_models(models: Models):
    models.vae_decoder = torch.compile(models.vae_decoder, fullgraph=True,) 
    #models.transformer = torch.compile(models.transformer)

def load_all(config: OmegaConf):
    transformer = load_transformer(config)
    text_encoder = load_text_encoder()
    vae_encoder, vae_decoder = load_vae()
    pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
    print("Models loaded")
    models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
    #gc.collect()
    #torch.cuda.empty_cache()
    #print("compiling models")
    #compile_models(models)
    #gc.collect()
    #torch.cuda.empty_cache()
    return models


class GenerateParams:
    def __init__(self, **kwargs):
        self.prompt = kwargs.get("prompt", "")
        self.width = kwargs.get("width", 832)
        self.height = kwargs.get("height", 480)
        self.seed = kwargs.get("seed", None)
        self.num_blocks = kwargs.get("num_blocks", 9)
        self.num_denoising_steps = kwargs.get("num_denoising_steps", 5)
        self.kv_cache_num_frames = kwargs.get("kv_cache_num_frames", 3)
        self.timestep_shift = kwargs.get("timestep_shift", 5.0)
        self.strength = 1.0
        self.input_video = None
        self.start_frame = None
        self.webcam_mode = False
        self.keep_first_frame = False
        self.context_noise = 0.0
        self.block_on_frame = False
        self.request_id = None
        self.interp_blocks = -1

class GenerationSession:
    SESSION_COUNTER = AtomicCounter()
    @torch.inference_mode()
    def __init__(self, params, config, frame_callback=None, models=None):
        self.session_id = self.SESSION_COUNTER.increment()
        self.params = params
        self.width = params.width // 8 * 8
        self.height = params.height // 8 * 8
        self.latent_width = self.width // 8
        self.latent_height = self.height // 8
        self.config = config
        self.frame_callback = frame_callback or (lambda *args: None)
        self.gpu = torch.cuda.current_device()
        self.num_frame_per_block = 3
        self.seed = params.seed if params.seed is not None else torch.seed()
        self.rnd = torch.Generator(self.gpu).manual_seed(self.seed)
        self.num_blocks = params.num_blocks
        self.current_start_frame = 0
        self.block_idx = 0
        self.models = models
        num_latent_frames = self.num_blocks * self.num_frame_per_block
        latent_shape = [1, num_latent_frames, 16, self.latent_height, self.latent_width]
        self.all_latents = torch.zeros(latent_shape, device=self.gpu, dtype=torch.bfloat16).contiguous()
        self.noise = torch.randn(latent_shape, device=self.gpu, dtype=torch.bfloat16, generator=self.rnd).contiguous()
        self.init_models(models, params)
        self.decode_vae_cache = [None] * 55
        self.denoising_step_list = get_denoising_schedule(
            self.zero_padded_timesteps, self.params.strength, steps=self.params.num_denoising_steps
        )

    def init_models(self, models, params):
        attn_size = self.params.kv_cache_num_frames + models.pipeline.num_frame_per_block
        for block in models.pipeline.generator.model.blocks:
            block.self_attn.local_attn_size = -1
        models.pipeline.local_attn_size = attn_size
        models.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
        models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
        models.pipeline.generator.model.block_mask = None
        models.pipeline.scheduler = FlowMatchScheduler(
            shift=params.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        models.pipeline.scheduler.set_timesteps(1000, training=True)
        st = models.pipeline.scheduler.timesteps
        self.zero_padded_timesteps = torch.cat((st.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.gpu)

    def get_clean_context_frames(self, models):
        current_kv_cache_num_frames = self.params.kv_cache_num_frames
        clean_context_frames = self.all_latents[:, :self.current_start_frame]
        if current_kv_cache_num_frames == 1:
            clean_context_frames = clean_context_frames[:, :1]
        else:
            clean_context_frames = torch.cat((
                clean_context_frames[:, :1],
                clean_context_frames[:, 1:][:, -current_kv_cache_num_frames + 1:]
            ), dim=1)
        return clean_context_frames

    def recompute_kv_cache(self, models):
        if self.block_idx == 0:
            models.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
            return
        clean_context_frames = self.get_clean_context_frames(models)
        models.pipeline._initialize_kv_cache(
            batch_size=clean_context_frames.shape[0],
            dtype=clean_context_frames.dtype,
            device=clean_context_frames.device
        )
        block_mask = models.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=clean_context_frames.device,
            num_frames=clean_context_frames.shape[1],
            frame_seqlen=models.pipeline.frame_seq_length,
            num_frame_per_block=models.pipeline.num_frame_per_block,
            local_attn_size=-1,
        )
        context_timestep = torch.zeros([clean_context_frames.shape[0], clean_context_frames.shape[1]], dtype=torch.int64, device=self.gpu)
        models.pipeline.generator.model.block_mask = block_mask
        models.transformer(
            noisy_image_or_video=clean_context_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=models.pipeline.kv_cache1,
            crossattn_cache=models.pipeline.crossattn_cache,
            current_start=min(self.current_start_frame, self.params.kv_cache_num_frames) * models.pipeline.frame_seq_length,
        )
        models.pipeline.generator.model.block_mask = None

    @torch.inference_mode()
    def generate_block(self, models):
        print("here")
        if self.block_idx >= self.num_blocks:
            return None
        if self.block_idx == 0:
            self.conditional_dict = models.text_encoder(text_prompts=[self.params.prompt])
            for k, v in self.conditional_dict.items():
                self.conditional_dict[k] = v.to(dtype=torch.bfloat16).contiguous()
        self.recompute_kv_cache(models)
        noisy_input = self.noise[:, self.current_start_frame:self.current_start_frame + models.pipeline.num_frame_per_block]
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.full([1, models.pipeline.num_frame_per_block], current_timestep, device=self.gpu, dtype=torch.int64)
            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = models.transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=models.pipeline.kv_cache1,
                    crossattn_cache=models.pipeline.crossattn_cache,
                    current_start=min(self.current_start_frame, self.params.kv_cache_num_frames) * models.pipeline.frame_seq_length
                )
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.models.pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn(*denoised_pred.flatten(0, 1).shape, generator=self.rnd, device="cuda", dtype=torch.bfloat16),
                        next_timestep * torch.ones([1 * self.models.pipeline.num_frame_per_block], device="cuda", dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = models.transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=models.pipeline.kv_cache1,
                    crossattn_cache=models.pipeline.crossattn_cache,
                    current_start=min(self.current_start_frame, self.params.kv_cache_num_frames) * models.pipeline.frame_seq_length
                )
        self.all_latents[:, self.current_start_frame:self.current_start_frame + models.pipeline.num_frame_per_block] = denoised_pred
        pixels, self.decode_vae_cache = models.vae_decoder(denoised_pred.half(), *self.decode_vae_cache)
        if self.block_idx == 0:
            pixels = pixels[:, 3:, :, :, :]
        event = torch.cuda.Event()
        event.record()
        self.frame_callback(pixels, [], event)
        self.current_start_frame += models.pipeline.num_frame_per_block
        self.block_idx += 1


def save_video_direct(pixels: torch.Tensor, output_path: Path, fps: int = 16):
    try:
        import subprocess, tempfile, numpy as np
        pixels = pixels[0].cpu().clamp(0, 1)
        frames_np = (pixels.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{pixels.shape[3]}x{pixels.shape[2]}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast", str(output_path)
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdin.write(frames_np.tobytes())
        proc.stdin.close()
        proc.wait()
        return str(output_path) if proc.returncode == 0 else None
    except Exception as e:
        print(f"Failed to save video: {e}")
        return None

def sample_videos(
    prompts_list,
    config_path: str = "configs/self_forcing_server_14b.yaml",
    output_dir: str = "outputs/samples",
    num_blocks: int = 9,
    width: int = 832,
    height: int = 480,
    seed: int = 42,
    num_denoising_steps: int = 5,
    kv_cache_num_frames: int = 3,
    timestep_shift: float = 5.0,
    save_videos: bool = True,
    fps: int = 16,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = load_merge_config(config_path)
    models = load_all(config)
    results = {}
    for prompt_idx, prompt in enumerate(tqdm(prompts_list, desc="Generating")):
        all_frames = []
        def frame_callback(pixels, frame_ids, event):
            event.synchronize()
            all_frames.append(pixels.cpu().add_(1).mul_(0.5).clamp_(0, 1))
        params = GenerateParams(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed + prompt_idx,
            num_blocks=num_blocks,
            num_denoising_steps=num_denoising_steps,
            kv_cache_num_frames=kv_cache_num_frames,
            timestep_shift=timestep_shift,
        )
        session = GenerationSession(params=params, config=config, frame_callback=frame_callback, models=models)
        t0 = time.time()
        for _ in range(num_blocks):
            session.generate_block(models)
        print(f"Prompt {prompt_idx} done in {time.time() - t0:.2f}s")
        combined = torch.cat(all_frames, dim=1) if all_frames else torch.empty(0)
        result = {"prompt": prompt, "num_frames": combined.shape[1], "video_path": None}
        if save_videos and combined.numel() > 0:
            vid_path = output_path / f"prompt_{prompt_idx:03d}.mp4"
            result["video_path"] = save_video_direct(combined, vid_path, fps=fps)
        results[prompt_idx] = result
    return results

if __name__ == "__main__":
    prompts = ["a red panda skateboarding at sunset"]
    sample_videos(
        prompts_list=prompts,
        num_blocks=9,
        seed=42,
        output_dir="outputs/quack_test"
    )
