# blackwan
Look, I’m trying to make WAN2.2 inference scream on NVIDIA’s B200s (sm_100), a lot of it will be transferable to sm_120 i.e. 5090 or 6000 PRO but not all of it. This repo is raw, messy, and full of experiments to max out MFU, memory usage, and cut latency to whatever I can, you can find some graphs, profiles and such below and on my X (<a>https://x.com/Shauray7</a>). If it’s done, it’s in the repo or posted. If it’s planned, it’s below !
<br><br>
My plan is to approch everything, atleast try to optimize everything (a lot of it fails but you can find some of that as well throughout the repo) 
<br><br>
<h2>What's Done</h2>
  <h3>Brute Force Baselines for WAN2.2</h3>
  <p>
    Starting with the absolute garbage inference pipeline for WAN2.2, on initial profile I can see attention dominates at 59.7% total GPU time, averaging 27.7ms per invocation, with sequence complexity scaling quadratically. Memory subsystem shows 129,295 MB transferred across 10,239 device-to-device operations.
  </p>
  <p>
    CUDA API layer reports ~1.94M kernel launches with 32% overhead in cudaLaunchKernel calls. This probably indicates severe fragmentation from high/low noise transformer swapping during sampling steps. Tensor core util is at 12.5% and 4,800 flash kernel launches with variable block dimensions, Register pressure likely limiting occupancy on B200's 132 SMs ??
  </p>
  <img src="assets/b200_baseline.png" alt="Baseline Speed Graph">

<h3>Basic Batching, chunking seq len</h3>
<p>
  more like reduction in computational complexity by chunking sequence length 59.7% to 46.8% for attn, Avg execution time reduced from 27.7ms to 16.2ms per call, tensor core util fell to 10.1%, but I'm not too worried about that, I have plans on to write cute kernels for those, there are a lot mem bound ops for now on the pipeline 
</p>
<p>
  There are 2 additional D-H and H-D due to batching it switches the high/low transformer, but I have enough VRAM to store an elephant on the HBM3e. So that won't be an issue
</p>
  <img src="assets/b200_baseline.png" alt="Baseline Speed Graph">

<h3>TaylorSeer adition</h3>
<p>
  Lost my sanity getting taylorseer running on WAN2.2. so many errors, rewrites after rewrites for 2 straight days but finally got it working, well sorta, not sure if it’s broken or just bad. I've tried it before on image models, wasn't that bad though. Might look into this later this cannot be right ! but anyways I won't include caching for speedups so this wont matter a lot 
</p>
  <img src="assets/b200_baseline.png" alt="Baseline Speed Graph">

<h3>DataLoader Speedup</h3>
<p>
  Made the data loading layer faster. Batching optimized, threading tuned, overhead dropped. attaching profiles soon did not get time to profile 
</p>

<h2>Ongoing Optimizations (this includes what I've tried but has too many errors)</h2>
<h3>Softmax Kernel</h3>
<p>
    Working on optimized attention. Started with softmax, integrating with GEMM next. Will merge to form one tight kernel for attention.
</p>
<h3>Model Pruning (MOE/SNR/BS)</h3>
<p>
  Stripping out parts of the model that don't move the needle. Pruning Mixture-of-Experts, low SNR blocks, batch tweaks. Will post quality/perf deltas. I wanted to try this since the model came out, I dont like how the SNR/2 decides when to switch the transformer 
</p>

<h3>Lightweight FA4-style Attention</h3>
<p>
  Not full FA4 yet. Just a simplified version for quick gains.
</p>

<h3>Ulysses + Ring (2-4x B200)</h3>
<p>
  Exploring parallel execution with Ulysses and Ring on 2–4 GPUs. Initial runs on 2x B200 show decent scaling.
</p>

<h3>CHORDS (Experimental)</h3>
<p>
  Playing with CHORDS. Might help with better parallelism, since it runs the whole thing distributed not just attn
</p>

<h2>Haven't tried yet, but comes with substantial speedups</h2>
  <ul>
    <li><strong>Tridao’s Full FA4</strong> — Since I wont need any prefil or decode stuff or causal masking for WAN, I will have to do a lot of changes but the bare bones remains the same I guess</li>
    <li><strong>Mega FA4 Kernel</strong> — Full attention kernel, stripped down, merged into one hot path.</li>
    <li><strong>8x B200 Scaling</strong> — Final test: full system graph, all speedups posted, quality retained.</li>
  </ul>

<p>
  I’ve been posting updates, graphs, and demo videos over on Twitter. Follow for ongoing results:
  <br>
  <a href="https://twitter.com/Shauray7" target="_blank">@Shauray7</a>
</p>

<h2>Contributions</h2>
  <p>
    If you’ve got something to improve speed, reduce mem, or clean up kernels make a PR would love to chat. Just keep it grounded in profiling, not theory.
  </p>
