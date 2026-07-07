## Plan: Assignment 2 GPU Study Map

Create a course-safe study path for CS336 Assignment 2 Systems. The goal is to build GPU and distributed-training background before touching implementation details. This plan is for learning and debugging strategy only. It should not be used as generated assignment code, pseudocode, or a shortcut around the assignment.

## Steps

1. Start with GPU background from zero: learn what a GPU is, why it differs from a CPU, what CUDA means, and the basic words `device`, `host`, `kernel`, `thread`, `block`, `warp`, `SM`, `global memory`, `shared memory`, `register`, `Tensor Core`, and `VRAM`.
2. Build GPU performance intuition: understand compute-bound vs memory-bound vs latency/overhead-bound work; why memory movement is often the bottleneck; why operator fusion matters; why arithmetic intensity explains whether a workload can use the GPU well.
3. Study beginner-friendly GPU performance resources in this order: Wikipedia GPU/CUDA pages for vocabulary only, NVIDIA GPU Performance Background, Modal GPU Glossary, Horace He's performance blog, Tim Dettmers' GPU/deep-learning hardware explanation, then NVIDIA Matrix Multiplication Background.
4. Move to official CS336 context: read/watch the assignment-adjacent lectures on resource accounting, GPUs/TPUs, kernels/Triton, and parallelism.
5. Learn CUDA/Triton fundamentals: CUDA programming model at a high level, then Triton vector add, fused softmax, matmul, fused attention, and Triton debugging docs.
6. Study FlashAttention conceptually: IO-aware tiling, online softmax/log-sum-exp stability, recomputation tradeoffs, and FlashAttention-2 work partitioning. Avoid third-party implementation repos as assignment answers.
7. Study PyTorch distributed concepts: process groups, rank/world size, collectives, all-reduce, reduce-scatter, all-gather, barriers, and one-process-per-device mental model.
8. Map DDP to the assignment: parameter broadcast, per-parameter autograd hooks, async gradient synchronization, and final wait before optimizer step.
9. Map FSDP and sharded optimizer: parameter sharding, all-gather before compute, reduce-scatter after backward, fp32 master weights, mixed precision compute dtype, and optimizer state sharding.
10. Use the repo tests only as conceptual landmarks: identify what behavior is being checked, then write and debug your own implementation independently.

## Beginner Resources

For a fuller beginner-friendly summary with diagrams and reading flow, see [beginner_resources.md](beginner_resources.md).

- Wikipedia GPU: https://en.wikipedia.org/wiki/Graphics_processing_unit
  - Use for lightweight vocabulary: GPU vs CPU, parallel structure, VRAM, GPGPU.
- Wikipedia CUDA: https://en.wikipedia.org/wiki/CUDA
  - Use for quick orientation for CUDA as NVIDIA's GPU programming platform. Skim the ontology/table and programming flow.
- NVIDIA GPU Performance Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
  - Best first serious doc. Covers SMs, memory hierarchy, execution model, arithmetic intensity, memory/math/latency limits.
- Modal GPU Glossary: https://modal.com/gpu-glossary/readme
  - Term-by-term reference when words like warp scheduler, compute capability, or CUDA programming model feel slippery.
- Horace He, Making Deep Learning Go Brrrr From First Principles: https://horace.io/brrr_intro.html
  - Intuition for compute vs bandwidth vs overhead, kernel fusion, and PyTorch performance.
- Tim Dettmers, Which GPU(s) to Get for Deep Learning: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
  - Practical explanation of Tensor Cores, memory bandwidth, cache hierarchy, and why transformers care about bandwidth.
- NVIDIA Matrix Multiplication Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
  - Read after the above. Connects matrix multiplication, tiling, arithmetic intensity, Tensor Core alignment, and tile/wave quantization.
- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  - Use Part 1 only at first. Do not try to read the full guide linearly.
- Siboehm CUDA Matmul Worklog: https://siboehm.com/articles/22/CUDA-MMM
  - Advanced optional visual read after basics. Useful for seeing coalescing, shared memory, tiling, occupancy, and arithmetic intensity in action.

## Relevant Assignment Files

- `README.md` - states Assignment 2 covers profiling/benchmarking, Triton FlashAttention2, and distributed memory-efficient training.
- `LOCAL_SETUP.md` - local note naming FlashAttention, DDP, FSDP, and sharded optimizer as Assignment 2 implementation areas.
- `tests/adapters.py` - high-level API expectations for FlashAttention, DDP, FSDP, and sharded optimizer.
- `tests/test_attention.py` - conceptual target for forward/backward FlashAttention correctness and saved log-sum-exp.
- `tests/test_ddp.py` - conceptual target for rank-0 broadcast and gradient synchronization equivalence.
- `tests/test_fsdp.py` - conceptual target for sharding, mixed precision, gradient sync, and full parameter gathering.
- `tests/test_sharded_optimizer.py` - conceptual target for optimizer state sharding equivalence.

## Checkpoints

Before Triton/FlashAttention, you should be able to explain:

- CPU vs GPU
- Host vs device
- Kernel launch
- Grid/block/thread/warp
- SM
- Global memory vs shared memory vs registers
- Tensor Cores
- Memory bandwidth
- Arithmetic intensity

After studying the assignment-specific topics, you should be able to explain without code:

- Why FlashAttention saves memory
- What `all_reduce` does in DDP
- Why FSDP decomposes all-reduce into all-gather plus reduce-scatter
- Why mixed precision keeps master weights in fp32
- Why optimizer states dominate memory

## Debugging Habits

- Use toy tensor shapes before full-size experiments.
- Add shape and dtype assertions while debugging.
- Print rank-by-rank invariants when debugging distributed code.
- Use profiler/Nsight concepts to reason about bottlenecks.
- Prefer official docs and course materials before third-party implementations.
- Do not look for or copy third-party assignment solution repositories.
