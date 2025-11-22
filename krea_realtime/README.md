<div align="center">

# Krea Realtime 14B

**Real-time video generation with 14B parameter diffusion model**

</div>

---

## Setup

### 1. Create Virtual Environment

```bash
uv sync
```

### 2. Install Attention Backend

**For NVIDIA B200 GPUs** (recommended):
```bash
uv pip install flash_attn --no-build-isolation
```

### 3. Install FFmpeg

```bash
sudo apt update && sudo apt install ffmpeg
```

### 4. Download Model Checkpoints

```bash
# Base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B

huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-14B

# Krea Realtime model
huggingface-cli download krea/krea-realtime-video \
  krea-realtime-video-14b.safetensors \
  --local-dir checkpoints
```

---

## Usage

Generate videos offline without the WebSocket layer.

#### Basic Example

Create a script to generate videos with all the optimizations mentioned here - https://shauray8.github.io/about_shauray/blogs/krea_realtime_optimization.html:

```bash
uv run python isolated_quack.py
```

## Repository Structure

```
├── configs/                   # Configuration files
├── demo_utils/                # VAE and utility functions
├── model/                     # Model implementations
├── pipeline/                  # Inference pipelines
├── utils/                     # Helper utilities
├── wan/                       # Wan model components
└── templates/                 # Web UI templates
```

---

## Credits

This work is based on:
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) - Original distillation technique
- [Wan 2.1 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) - Base text-to-video model
- [LightX2V](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) - Timestep distilled checkpoint

For technical details on the Self-Forcing scaling and optimization process, see krea.ai's [blog post](https://www.krea.ai/blog/krea-realtime-14b).

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE.md](LICENSE.md) file for details.

---

## Citation

```bibtex
@software{krea_realtime_14b,
  title={Krea Realtime 14B: Real-time Video Generation},
  author={Krea AI},
  year={2025},
  url={https://github.com/krea-ai/realtime-video}
}
```
