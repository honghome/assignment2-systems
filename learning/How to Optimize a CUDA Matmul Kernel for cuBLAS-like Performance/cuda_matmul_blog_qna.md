# CUDA Matmul Blog Q&A Notes

Source: Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"

Use this file as a running log while reading the blog. Each time you ask a question, we can add a short answer here, grounded in the blog and general CUDA/GPU performance concepts.

## Reading Map

The blog builds one SGEMM kernel step by step and uses each optimization to explain a GPU performance idea.

| Blog Stage | Main Idea | Why It Matters |
| --- | --- | --- |
| Naive kernel | One CUDA thread computes one output element of C. | Establishes grid/block/thread indexing and the baseline memory pattern. |
| Global memory coalescing | Arrange neighboring threads to access neighboring addresses. | Coalesced loads use memory bandwidth much more efficiently. |
| Shared memory caching | Load reusable tiles of A and B into shared memory. | Reduces repeated global memory traffic. |
| 1D block tiling | One thread computes multiple output elements in one dimension. | Increases reuse and reduces shared-memory pressure per result. |
| 2D block tiling | One thread computes a small 2D tile of outputs. | Raises arithmetic intensity by doing more FLOPs per byte loaded. |
| Vectorized accesses | Use wider loads/stores and layout changes. | Reduces instruction count and helps the hardware move aligned chunks efficiently. |
| Autotuning | Search tile sizes and kernel parameters. | Best parameters depend on GPU architecture and matrix shapes. |
| Warp tiling | Add an explicit warp-level tiling structure. | Matches the hardware scheduling level more closely and improves locality. |

## Key Mental Model

A fast matmul kernel is mostly about arranging reuse across memory levels:

```text
Global memory -> shared memory -> registers -> FMA instructions
```

The optimization pattern is:

```text
Load a tile once, reuse it many times, keep neighboring threads accessing neighboring memory, and choose tile sizes that match the GPU.
```

## Core Terms

| Term | Short Meaning |
| --- | --- |
| SGEMM | Single-precision GEMM, usually C = alpha * A @ B + beta * C. |
| Grid | All blocks launched for one CUDA kernel call. |
| Block | A group of CUDA threads that can cooperate through shared memory. |
| Thread | The scalar CUDA program instance; each thread has its own registers. |
| Warp | A hardware-scheduled group of 32 threads. |
| Global memory / GMEM | Large off-chip GPU memory; high bandwidth but high latency. |
| Shared memory / SMEM | Fast on-chip memory shared by threads in one block. |
| Register | Fast per-thread storage used for temporary values and accumulators. |
| Coalescing | Combining neighboring threads' memory accesses into fewer wider transactions. |
| Arithmetic intensity | FLOPs per byte moved; higher usually means better compute utilization. |
| Occupancy | Fraction of possible active warps resident on an SM. Useful, but not the only goal. |
| Tiling | Splitting matrices into chunks so data can be reused in faster memory. |
| Autotuning | Benchmarking many parameter choices to find the best one for a device/workload. |

## Questions And Answers

### Q1. What do M and N mean, and why use 32 for gridDim/blockDim?

**Question:**

What do `M` and `N` mean here? Why do we initialize `gridDim` and `blockDim` with 32 as the base, and how can this handle matrices with dimension 4092?

**Answer:**

For matrix multiplication, the blog is using the usual shape convention:

```text
A has shape M x K
B has shape K x N
C has shape M x N
```

So `M` is the number of rows in `A` and `C`, `N` is the number of columns in `B` and `C`, and `K` is the shared inner dimension used in each dot product.

In the naive kernel, each CUDA thread computes one element of `C`. The thread computes its output position from its block position plus its position inside the block:

```text
output row = block row offset + thread row offset
output col = block col offset + thread col offset
```

The value `32` is not the matrix size. It is the block tile size. A block with shape `32 x 32` has `1024` threads, so one block is responsible for a `32 x 32` patch of the output matrix `C`. This is the maximum common CUDA block size in terms of thread count, since many NVIDIA GPUs allow up to 1024 threads per block.

For a `4092 x 4092` output matrix, the grid contains enough `32 x 32` blocks to cover all rows and columns:

```text
ceil(4092 / 32) = 128 blocks along each dimension
```

That means the launch covers a logical `4096 x 4096` area, because `128 * 32 = 4096`. The actual matrix is only `4092 x 4092`, so there are a few extra threads along the right and bottom edges. The boundary check `x < M && y < N` makes those extra threads do nothing.

Mental picture:

```text
Big C matrix: 4092 x 4092
Block tile:      32 x 32
Grid needed:    128 x 128 blocks
Covered area:  4096 x 4096 thread positions
Extra edge threads: ignored by the bounds check
```

So CUDA is not limited to `32 x 32` matrices. It launches many `32 x 32` blocks, and together those blocks cover the full matrix.

**Related blog section:**

Kernel 1: Naive Implementation

### Q2. How does one thread compute a 4092-element dot product?

**Question:**

If each CUDA block handles a `32 x 32` patch of `C`, and each thread computes one element of that patch, how does each dot product still use all `4092` elements from `A` and `B`?

**Answer:**

The important split is:

```text
x and y choose which output cell C[x, y] this thread owns.
i walks through the full dot product dimension K.
```

For square matrices in the blog example:

```text
M = 4092
N = 4092
K = 4092

A shape: 4092 x 4092
B shape: 4092 x 4092
C shape: 4092 x 4092
```

Each thread first gets one output coordinate:

```text
Thread responsibility:

    C[x, y]
```

That single output value is one dot product:

```text
C[x, y] = row x of A dot column y of B
```

Diagram:

```text
                 B, column y
                      y
                      |
                      v
                    B[0,y]
                    B[1,y]
                    B[2,y]
                      ...
                  B[4091,y]

A, row x:  A[x,0]   A[x,1]   A[x,2]   ...   A[x,4091]

Dot product for C[x,y]:

    C[x,y] = A[x,0]    * B[0,y]
           + A[x,1]    * B[1,y]
           + A[x,2]    * B[2,y]
           + ...
           + A[x,4091] * B[4091,y]
```

So the block size controls how many output cells are assigned at once, not how long each dot product is. A `32 x 32` block creates `1024` threads, and those threads compute `1024` different output cells of `C`. Inside every one of those threads, the inner loop still runs from `0` to `K - 1`, which is `0` to `4091` here.

For this naive kernel, no two valid threads should write the same `C[x,y]`. Each valid thread owns a separate output coordinate, and for that coordinate it loops over the full `K` range:

```text
one thread -> one C[x,y]
that thread -> i = 0, 1, 2, ..., 4091
```

The only threads that do not compute a `C[x,y]` are the extra edge threads created because `4092` is not divisible by `32`. Those fail the `x < M && y < N` check and skip the write.

Another way to picture it:

```text
One CUDA block covers this patch of C:

			   32 columns
		  +----------------+
 32 rows  | thread thread  |
		  | thread thread  |  each thread owns one C[x,y]
		  |      ...       |
		  +----------------+

But for each one of those C[x,y] cells:

	the thread looks across all 4092 entries of A's row
	and down all 4092 entries of B's column.
```

Concrete mini-example with smaller numbers:

```text
Suppose K = 4.

C[2,3] = A[2,0] * B[0,3]
       + A[2,1] * B[1,3]
       + A[2,2] * B[2,3]
       + A[2,3] * B[3,3]
```

For the blog's `4092` case, it is the same pattern, just with `4092` multiply-add terms instead of `4`.

**Related blog section:**

Kernel 1: Naive Implementation

### Q3. Does the GPU really run this logic separately for each thread?

**Question:**

The thread logic seems complicated. It sounds like there is a separate function execution for each thread. Does this really happen on the GPU?

**Answer:**

Yes, but with one important nuance: the CUDA programming model lets you write the kernel as if you are writing the logic for one thread. When the kernel is launched, the GPU creates many thread instances of that same kernel.

Conceptually:

```text
kernel launch
  -> thread 0 runs the kernel logic with its own threadIdx/blockIdx
  -> thread 1 runs the kernel logic with its own threadIdx/blockIdx
  -> thread 2 runs the kernel logic with its own threadIdx/blockIdx
  -> ...
```

So in the naive matmul kernel, every valid thread runs the same instructions:

```text
1. Compute my C[x,y] coordinate.
2. Loop over i = 0 ... K-1.
3. Accumulate A[x,i] * B[i,y].
4. Write the result to C[x,y].
```

But on real NVIDIA hardware, the GPU does not usually schedule each thread fully independently one by one. Threads are grouped into warps of 32 threads. A warp usually executes the same instruction for 32 threads at once, with each thread having its own values of `x`, `y`, `i`, and `tmp`.

Mental picture:

```text
CUDA code view:

  one thread runs one scalar program

Hardware view:

  one warp = 32 threads
  those 32 threads move through the same kernel instructions together
  each thread has different data/registers
```

For the inner dot-product loop, if `K = 4092`, then each thread has its own loop over `i = 0 ... 4091`. Threads in the same warp are usually at the same loop step at the same time, but they are computing different output cells of `C`.

Example:

```text
Thread A computes C[0,0]
Thread B computes C[0,1]
Thread C computes C[0,2]

At loop step i = 17:

Thread A uses A[0,17] * B[17,0]
Thread B uses A[0,17] * B[17,1]
Thread C uses A[0,17] * B[17,2]
```

That is why the code can be written once, but applied to millions of output elements: the GPU creates many thread instances, and the hardware runs them in large groups.

**Related blog section:**

Kernel 1: Naive Implementation

### Q4. Does each thread own a compute unit, registers, and shared memory?

**Question:**

Does each thread own its own compute unit and registers, but share some memory with other threads?

**Answer:**

Almost. The key distinction is:

```text
Each thread owns its own registers and local temporary values.
Threads do not each own a dedicated compute unit.
Threads in the same block can share shared memory.
All threads can access global memory.
```

A CUDA thread has its own private state, such as:

```text
x
y
i
tmp
```

These usually live in registers. So if 1024 threads are in a block, they each have their own `tmp`; they are not all writing to the same `tmp` variable.

But compute units are hardware resources inside an SM. Threads are scheduled onto those resources. A thread uses a CUDA core or other execution unit when one of its instructions is issued, but it does not permanently own that unit.

Mental model:

```text
Thread's private state:

  registers: x, y, i, tmp

Shared by block:

  shared memory / SMEM

Shared by all threads on the GPU:

  global memory / GMEM, including A, B, C

Hardware execution resources:

  CUDA cores, load/store units, warp schedulers
  used by many threads over time
```

For the naive matmul kernel in this section, the code does not use shared memory yet. It reads `A` and `B` from global memory, accumulates in a private register variable like `tmp`, and writes one result to global memory `C[x,y]`.

Later kernels in the blog introduce shared memory. At that point, threads in the same block cooperate by loading tiles of `A` and `B` into shared memory, then reusing those tiles.

Simple picture:

```text
One block on one SM:

  thread 0: private registers
  thread 1: private registers
  thread 2: private registers
    ...
  thread 1023: private registers

  all threads in this block can access the block's shared memory

GPU-wide:

  all blocks can access global memory A, B, C
```

So the best short version is: each thread has private registers, threads in a block can share shared memory, and compute units are shared hardware that execute many threads' instructions.

**Related blog section:**

Kernel 1: Naive Implementation; Kernel 3: Shared Memory Cache-Blocking

### Q5. At what level are compute units shared?

**Question:**

If compute units are not owned by individual threads, at what level are they shared?

**Answer:**

Compute units are shared at the SM level.

An NVIDIA GPU has many SMs, or streaming multiprocessors. Each SM contains hardware execution resources such as:

```text
warp schedulers
CUDA cores
load/store units
special function units
Tensor Cores, on modern GPUs
```

Blocks are assigned to SMs. Once a block is resident on an SM, its warps are scheduled onto that SM's compute units.

Picture:

```text
Whole GPU
|
+-- SM 0
|   |-- compute units shared by warps resident on SM 0
|   |-- blocks currently assigned to SM 0
|
+-- SM 1
|   |-- compute units shared by warps resident on SM 1
|   |-- blocks currently assigned to SM 1
|
+-- SM 2
  |-- compute units shared by warps resident on SM 2
  |-- blocks currently assigned to SM 2
```

So the sharing hierarchy is roughly:

```text
registers: private to one thread
shared memory: shared by threads in one block
compute units: shared by warps/threads resident on one SM
global memory: shared by all SMs on the GPU
```

If a thread block is placed on `SM 0`, the threads in that block do not use CUDA cores in `SM 1`. They run on the compute units inside `SM 0`. Another block placed on `SM 1` uses `SM 1`'s own compute units.

Within an SM, scheduling happens mainly at the warp level. A warp scheduler chooses ready warps and issues their next instruction to the SM's execution units. This is why a thread does not own a CUDA core: many warps take turns using the SM's hardware.

Mental model:

```text
Threads own data state.
Warps are the scheduling unit.
SMs own the compute hardware.
The GPU owns many SMs.
```

**Related blog section:**

Kernel 2: Global Memory Coalescing; Kernel 3: Shared Memory Cache-Blocking

### Q6. If blocks on the same SM share compute units, why do we need blocks?

**Question:**

If different blocks on the same SM share compute units, why does CUDA need blocks at all?

**Answer:**

Blocks are not mainly about owning compute units. Blocks are a way to group threads for work assignment, cooperation, and scheduling.

The main reasons we need blocks are:

```text
1. Divide a huge problem into manageable chunks.
2. Give a group of threads a private shared-memory workspace.
3. Allow synchronization among threads in that group.
4. Give the GPU scheduler independent chunks of work to place on SMs.
```

For the naive matmul kernel, a block handles one `32 x 32` patch of `C`:

```text
Huge C matrix
|
+-- block (0,0) computes one 32 x 32 patch
+-- block (0,1) computes one 32 x 32 patch
+-- block (1,0) computes one 32 x 32 patch
+-- block (1,1) computes one 32 x 32 patch
+-- ...
```

This makes the problem easy to map onto the GPU. The GPU can assign different blocks to different SMs, and each SM can work independently.

The deeper reason blocks matter appears in optimized kernels. Threads in the same block can cooperate:

```text
Threads in one block can:
  share shared memory
  use __syncthreads()
  cooperatively load a tile of A and B
  reuse that tile to compute many C values

Threads in different blocks cannot:
  synchronize with each other inside one kernel
  directly share block-level shared memory
```

So a block is like a small team of threads. The team has its own workspace, can coordinate internally, and computes one chunk of the overall output.

Compute units are shared at the SM level, but blocks define which threads can cooperate.

Short version:

```text
Thread: does scalar work and owns private registers.
Warp: hardware scheduling group of 32 threads.
Block: cooperation group with shared memory and synchronization.
Grid: all blocks for one kernel launch.
SM: hardware place where blocks run and share compute units.
```

**Related blog section:**

Kernel 1: Naive Implementation; Kernel 3: Shared Memory Cache-Blocking

### Q7. What do GEMM and SGEMM mean?

**Question:**

What does `GEMM` mean, and what does `SGEMM` mean?

**Answer:**

`GEMM` means general matrix-matrix multiplication. It is the standard BLAS name for this operation:

```text
C = alpha * A @ B + beta * C
```

Here:

```text
A @ B      means matrix multiplication
alpha      is a scalar multiplier for the matrix product
beta       is a scalar multiplier for the old value of C
C          is both an input and the output
```

So GEMM is slightly more general than just:

```text
C = A @ B
```

because it can also scale the product and add a scaled previous `C`.

`SGEMM` means single-precision GEMM. The `S` stands for single precision, which usually means `float32`.

Common BLAS naming:

```text
SGEMM: float32 GEMM
DGEMM: float64 GEMM
HGEMM: float16 GEMM, in some libraries/docs
```

In the blog, the author is optimizing CUDA `SGEMM`, so the matrices contain 32-bit floating point values and the operation is:

```text
C = alpha * A @ B + beta * C
```

For deep learning, GEMM/SGEMM is important because many neural network operations reduce to large matrix multiplications.

**Related blog section:**

Kernel 1: Naive Implementation

### Q8. Why do GEMM formulas include alpha and beta?

**Question:**

Why do we need `alpha` and `beta` in `C = alpha * A @ B + beta * C`? What do they mean?

**Answer:**

`alpha` and `beta` are scalar coefficients. They let the GEMM operation do more than plain matrix multiplication.

The full GEMM formula is:

```text
C_new = alpha * (A @ B) + beta * C_old
```

Meaning:

```text
alpha controls how much of the new matrix product to use.
beta controls how much of the old C matrix to keep.
```

Common cases:

```text
alpha = 1, beta = 0
  C = A @ B
  plain matrix multiplication

alpha = 1, beta = 1
  C = A @ B + C
  add the new product to an existing C

alpha = 0.5, beta = 0
  C = 0.5 * (A @ B)
  scale the matrix product

alpha = 1, beta = 0.1
  C = A @ B + 0.1 * C
  mix the new product with a scaled old C
```

Why include them? Because many numerical programs need this combined operation, and doing it inside one library call avoids extra passes over memory. Without `alpha` and `beta`, you might need separate operations:

```text
temp = A @ B
temp = alpha * temp
C = temp + beta * C
```

That would read and write large matrices multiple times. GEMM fuses the scaling and addition into the matrix multiply output step.

In the blog's naive kernel, each thread computes one dot product into `tmp`, then writes:

```text
C[x,y] = alpha * tmp + beta * C[x,y]
```

So `tmp` is the new dot product for `A @ B`, and `C[x,y]` on the right side is the old value of `C` before this GEMM call.

For learning the core matmul idea, you can mentally set:

```text
alpha = 1
beta = 0
```

Then the operation becomes simply:

```text
C = A @ B
```

**Related blog section:**

Kernel 1: Naive Implementation

### Q9. What is the real usage of alpha and beta?

**Question:**

What is the real usage of `alpha` and `beta` in GEMM?

**Answer:**

The real usage is to combine matrix multiplication, scaling, and accumulation in one highly optimized operation.

The GEMM formula is:

```text
C_new = alpha * (A @ B) + beta * C_old
```

This is useful because `C` can be huge. If you did each step separately, you would read and write large matrices multiple times. GEMM lets the kernel compute the dot product and directly write the final scaled/combined result.

Real cases:

```text
1. Plain matrix multiplication

  alpha = 1, beta = 0
  C = A @ B
```

This is the simplest use case.

```text
2. Accumulate into an existing output

  alpha = 1, beta = 1
  C = A @ B + C
```

This is useful when `C` already contains partial results and the new matrix product should be added into it.

```text
3. Scale the matrix product

  alpha = 0.5, beta = 0
  C = 0.5 * (A @ B)
```

This is useful when the math formula needs a scaled product.

```text
4. Mix a new product with an existing matrix

  alpha = 1, beta = 0.9
  C = A @ B + 0.9 * C
```

This kind of pattern appears in numerical algorithms that repeatedly update a matrix.

```text
5. Add bias or previous output-like data

  alpha = 1, beta = 1
  C starts with some existing values
  C = A @ B + C
```

In neural-network libraries, this idea can be used when the output buffer is prefilled with something that should be added to the matmul result, such as a broadcasted bias or accumulated contribution. Exact implementations vary by library, and many modern frameworks may fuse bias/add/activation in other specialized kernels too.

The performance reason is the big one:

```text
Separate operations:
   read/write temporary A @ B
   read/write scaled result
   read/write C again

GEMM with alpha/beta:
   compute dot product
   combine with old C
   write final C once
```

So `alpha` and `beta` are not needed to understand basic matrix multiplication, but they make GEMM a more useful and more memory-efficient building block for real numerical programs.

**Related blog section:**

Kernel 1: Naive Implementation

### Q10. Why is the minimum memory read `3 * 4092^2 * 4B`?

**Question:**

Why does the blog say the total data to read is `3 * 4092^2 * 4B = 201 MB`, and the total data to store is `4092^2 * 4B = 67 MB`?

**Answer:**

This calculation is a theoretical lower bound for GEMM:

```text
C = alpha * A @ B + beta * C
```

For square matrices of size `4092 x 4092`, each matrix has:

```text
4092 * 4092 = 16,744,464 elements
```

The blog is using `float32`, and each `float32` takes `4` bytes:

```text
16,744,464 elements * 4 bytes = 66,977,856 bytes
```

That is about `67 MB` per matrix.

For GEMM, the kernel must at least read:

```text
A:     4092^2 float32 values
B:     4092^2 float32 values
C_old: 4092^2 float32 values, if beta is not zero
```

So the minimum read volume is:

```text
3 * 4092^2 * 4 bytes
= 3 * 66,977,856 bytes
= 200,933,568 bytes
≈ 201 MB
```

Then it must write the final output matrix `C_new`:

```text
C_new: 4092^2 float32 values
```

So the minimum store volume is:

```text
4092^2 * 4 bytes
= 66,977,856 bytes
≈ 67 MB
```

The total minimum global-memory traffic is therefore:

```text
reads  ≈ 201 MB
writes ≈  67 MB
total  ≈ 268 MB
```

Why read old `C`? Because the full GEMM formula includes `beta * C_old`:

```text
C_new[x,y] = alpha * dot(A row x, B column y) + beta * C_old[x,y]
```

If `beta = 0`, then mathematically old `C` is not needed, so a specialized implementation could skip reading old `C`. But the blog is discussing the general GEMM lower bound, where `beta` may be nonzero.

Important: this `268 MB` is an ideal minimum. The naive kernel actually reads much more from global memory because many different threads repeatedly load the same values from `A` and `B`. Later optimizations try to get closer to the lower bound by reusing data through cache/shared memory/registers.

**Related blog section:**

Kernel 1: Lower Bounding the Fastest Possible Runtime

### Q11. Why is total work `2 * 4092^3 + 4092^2` FLOPs?

**Question:**

I do not understand this calculation: for each of the `4092^2` entries of `C`, we perform a dot product of two vectors of size `4092`, involving a multiply and an add at each step. FMA counts as two FLOPs. Why does this give `2 * 4092^3 + 4092^2 = 137 GFLOPs`?

**Answer:**

Start with one output cell:

```text
C[x,y] = A[x,0] * B[0,y]
  + A[x,1] * B[1,y]
  + A[x,2] * B[2,y]
  + ...
  + A[x,4091] * B[4091,y]
```

That is a dot product of length `4092`.

For each `i`, the kernel does roughly:

```text
tmp += A[x,i] * B[i,y]
```

This has two floating-point operations:

```text
A[x,i] * B[i,y]   -> 1 multiply
tmp + product     -> 1 add
```

So each `i` costs about `2` FLOPs. Since `i` runs `4092` times:

```text
one C[x,y] costs about 2 * 4092 FLOPs
```

There are `4092 * 4092 = 4092^2` output cells in `C`, so the matrix multiplication part costs:

```text
4092^2 output cells * (2 * 4092 FLOPs per cell)
= 2 * 4092^3 FLOPs
```

Then GEMM also adds the old `C` term:

```text
C_new[x,y] = alpha * tmp + beta * C_old[x,y]
```

The blog simplifies this as one extra add per output cell, so it adds:

```text
4092^2 extra FLOPs
```

Therefore:

```text
total FLOPs = 2 * 4092^3 + 4092^2
```

Numerically:

```text
4092^2 = 16,744,464
4092^3 = 68,517,541,888

2 * 4092^3 + 4092^2
= 2 * 68,517,541,888 + 16,744,464
= 137,051,083,776 + 16,744,464
= 137,067,828,240 FLOPs
≈ 137 billion FLOPs
≈ 137 GFLOPs
```

Small terminology note: `FLOP` means one floating-point operation. `FLOP/s` means floating-point operations per second. In this sentence, the blog is counting total work, so `137 GFLOPs` means about `137 billion floating-point operations`, not speed yet.

About FMA:

```text
tmp += a * b
```

may run as one hardware instruction called fused multiply-add, or FMA. Even though it is one instruction, it performs a multiply and an add, so performance math counts it as `2` FLOPs.

**Related blog section:**

Kernel 1: Lower Bounding the Fastest Possible Runtime

### Q12. Why does `alpha * tmp + beta * C_old[x,y]` count as only one extra FLOP?

**Question:**

Why does the blog count only `+ 4092^2` extra FLOPs for `C_new[x,y] = alpha * tmp + beta * C_old[x,y]`? It has both multiplication and addition.

**Answer:**

Good catch. Strictly speaking, the final GEMM expression can involve more than one FLOP per output element:

```text
C_new[x,y] = alpha * tmp + beta * C_old[x,y]
```

If counted literally, that can be:

```text
alpha * tmp        -> 1 multiply
beta * C_old      -> 1 multiply
add them together -> 1 add
```

So a strict count could add up to `3 * M * N` extra FLOPs.

The blog uses a common simplified performance-count convention. It mainly counts the matrix multiply work:

```text
2 * M * N * K
```

and then adds one extra `M * N` term for the final addition with `C`, because GEMM is described as matrix multiplication followed by an addition of a matrix.

For this size, the difference is tiny compared with the dot-product work:

```text
Main matmul work:       2 * 4092^3  ≈ 137.05 billion FLOPs
One extra per C entry:      4092^2  ≈  16.7 million FLOPs
Three extra per C entry: 3 * 4092^2 ≈  50.2 million FLOPs
```

`50 million` is much smaller than `137 billion`, so the final scaling/addition is a lower-order term. It barely changes the headline number.

Also, in many benchmark settings, `alpha` and `beta` are often simple values such as:

```text
alpha = 1
beta = 0
```

or:

```text
alpha = 1
beta = 1
```

In those cases, some multiplications may be optimized away or treated as not central to the GEMM performance model.

So the most precise explanation is:

```text
The exact final-output operation may cost 1 to 3 extra FLOPs per C element,
depending on alpha/beta and counting convention.

The blog counts it as +M*N because that is a simple lower-order GEMM estimate,
and the dominant term is still 2*M*N*K.
```

For learning, the most important part is the dominant term:

```text
2 * M * N * K
```

because each output element does a length-`K` dot product, and each dot-product step is one multiply plus one add.

**Related blog section:**

Kernel 1: Lower Bounding the Fastest Possible Runtime

### Q13. What does `300 GFLOPs` mean, and how is it calculated?

**Question:**

The blog says the naive kernel achieves about `300 GFLOPs` on an A6000 GPU. What does `GFLOPs` mean here, and how is `300` calculated?

**Answer:**

Here the blog is talking about performance speed, so it really means:

```text
300 GFLOP/s
```

That means:

```text
300 billion floating-point operations per second
```

Terminology:

```text
FLOP    = one floating-point operation
GFLOP   = one billion floating-point operations
FLOP/s  = floating-point operations per second
GFLOP/s = billion floating-point operations per second
TFLOP/s = trillion floating-point operations per second
```

The performance number is computed from:

```text
performance = total FLOPs / runtime_seconds
```

Earlier, the blog estimated the total work for one `4092 x 4092` SGEMM as about:

```text
137 GFLOPs of work
```

That means one full matrix multiplication needs about `137 billion` floating-point operations.

The `0.5` seconds comes from the blog's naive-kernel discussion. Right after showing the naive kernel, the author says the kernel takes about `0.5s` to process three `4092^2` FP32 matrices on an A6000 GPU. That is the measured runtime being used for this rough performance estimate.

If the naive kernel takes about `0.5` seconds, then:

```text
performance = 137 GFLOPs / 0.5 seconds
            = 274 GFLOP/s
```

That is close to the blog's rough `~300 GFLOP/s` number. The exact number depends on the measured runtime and the precise FLOP-count convention.

So the idea is:

```text
total work:  about 137 billion FLOPs
runtime:     about 0.46 to 0.5 seconds
speed:       about 300 billion FLOPs per second
```

Why is this considered bad? Because the A6000 is advertised near `30 TFLOP/s` for FP32. Since:

```text
30 TFLOP/s = 30,000 GFLOP/s
```

the naive kernel's `300 GFLOP/s` is only about:

```text
300 / 30000 = 0.01 = 1%
```

of the GPU's advertised peak. The rest of the blog explains why: the naive kernel wastes a lot of memory bandwidth and does not reuse data well.

**Related blog section:**

Kernel 1: Memory Access Pattern of the Naive Kernel

### Q14. Why does CUDA have 3D blocks and 3D thread indices?

**Question:**

Why do we need three dimensions for blocks and threads? In the blog's warp illustration, why are `x`, `y`, `z` shown with values like `4`, `16`, and `64`? Is this a design choice for the kernel, or just for illustration?

**Answer:**

CUDA gives both the grid and each block up to three dimensions because many problems are naturally 1D, 2D, or 3D.

Examples:

```text
1D problem: vector operations, token arrays
2D problem: images, matrices
3D problem: volumes, simulations, 3D grids
```

For matrix multiplication, we usually only need a 2D layout because the output `C` is a 2D matrix:

```text
blockIdx.x / threadIdx.x -> one matrix dimension
blockIdx.y / threadIdx.y -> the other matrix dimension
blockIdx.z / threadIdx.z -> usually unused, often 1
```

The blog's diagram is not saying this matmul kernel needs `threadIdx.z`. It is explaining the general CUDA rule for converting a 3D thread coordinate into a single linear `threadId`:

```text
threadId = threadIdx.x
         + blockDim.x * threadIdx.y
         + blockDim.x * blockDim.y * threadIdx.z
```

This linear `threadId` matters because neighboring `threadId`s are grouped into warps.

The `4`, `16`, and `64` in the diagram are strides from the example dimensions:

```text
blockDim.x = 4
blockDim.y = 4
blockDim.z = 4
```

Then:

```text
moving by 1 in x changes threadId by 1
moving by 1 in y changes threadId by blockDim.x = 4
moving by 1 in z changes threadId by blockDim.x * blockDim.y = 16
total threads in the example block = 4 * 4 * 4 = 64
```

So the diagram labels are not a special matmul design. The author chose small numbers so the flattening pattern can be drawn clearly.

The key lesson is:

```text
threadIdx.x changes fastest.
Then threadIdx.y.
Then threadIdx.z.
```

Because warps are formed from consecutive linear `threadId`s, the `x` dimension is usually the dimension where neighboring threads sit next to each other in warp order. That becomes important for memory coalescing: if neighboring threads in a warp access neighboring memory addresses, the GPU can combine their memory loads into fewer transactions.

For the naive matmul earlier, the launch used:

```text
blockDim = (32, 32, 1)
```

That means:

```text
32 threads in x
32 threads in y
1 thread in z
1024 total threads per block
```

So in this blog section, 3D indexing is being introduced to explain CUDA's general thread-ordering rule, but the matmul example itself mostly uses 2D blocks.

**Related blog section:**

Kernel 2: Global Memory Coalescing

### Q15. How should I understand the `threadId` flattening formula?

**Question:**

How should I understand this formula?

```text
threadId = threadIdx.x
         + blockDim.x * threadIdx.y
         + blockDim.x * blockDim.y * threadIdx.z
```

**Answer:**

This formula converts a 3D thread coordinate into a single 1D thread number inside the block.

It is the same idea as flattening a 2D or 3D array into linear memory.

First think in 2D. Suppose a block has:

```text
blockDim.x = 4
blockDim.y = 3
```

The threads form a `3 x 4` rectangle:

```text
y=0:  (0,0)  (1,0)  (2,0)  (3,0)
y=1:  (0,1)  (1,1)  (2,1)  (3,1)
y=2:  (0,2)  (1,2)  (2,2)  (3,2)
```

CUDA numbers them by letting `x` move fastest:

```text
y=0:    0      1      2      3
y=1:    4      5      6      7
y=2:    8      9     10     11
```

For 2D, the formula is:

```text
threadId = threadIdx.x + blockDim.x * threadIdx.y
```

Why multiply `threadIdx.y` by `blockDim.x`? Because each complete row in `y` contains `blockDim.x` threads. To move down one row, you skip one full row of `x` values.

Example:

```text
threadIdx = (2, 1)
blockDim.x = 4

threadId = 2 + 4 * 1
         = 6
```

Now add `z`. Suppose:

```text
blockDim.x = 4
blockDim.y = 3
blockDim.z = 2
```

One full `z` slice contains:

```text
blockDim.x * blockDim.y = 4 * 3 = 12 threads
```

So if `threadIdx.z = 1`, we need to skip the whole first 2D slice, which has `12` threads. That is why the `z` term is:

```text
blockDim.x * blockDim.y * threadIdx.z
```

Example:

```text
threadIdx = (2, 1, 1)
blockDim  = (4, 3, 2)

threadId = threadIdx.x
         + blockDim.x * threadIdx.y
         + blockDim.x * blockDim.y * threadIdx.z

         = 2
         + 4 * 1
         + 4 * 3 * 1

         = 2 + 4 + 12
         = 18
```

Mental rule:

```text
x offset: how far inside the current row
y offset: how many full x-rows to skip
z offset: how many full x-y slices to skip
```

This is why `x` is the fastest-changing dimension:

```text
(0,0,0) -> threadId 0
(1,0,0) -> threadId 1
(2,0,0) -> threadId 2
(3,0,0) -> threadId 3
(0,1,0) -> threadId 4
```

This matters for warps because consecutive `threadId`s are grouped into the same warp. So, neighboring `threadIdx.x` values usually become neighboring lanes in a warp.

**Related blog section:**

Kernel 2: Global Memory Coalescing

### Q16. Why does global memory coalescing help?

**Question:**

Why does coalescing help? Why is loading by consecutive `threadId`s more efficient? Is it because data is arranged in memory in the order of `threadId`?

**Answer:**

Coalescing helps because GPU global memory is most efficient when a warp's threads access consecutive or nearby memory addresses at the same time.

It is not that data is arranged in memory by `threadId`. Data is arranged by the array layout. For a normal row-major matrix, memory is laid out like this:

```text
Matrix A in row-major memory:

A[0,0], A[0,1], A[0,2], ..., A[0,K-1],
A[1,0], A[1,1], A[1,2], ..., A[1,K-1],
A[2,0], A[2,1], A[2,2], ..., A[2,K-1],
...
```

So consecutive columns in the same row are consecutive in memory:

```text
A[x,0], A[x,1], A[x,2], A[x,3]  are next to each other
```

But consecutive rows at the same column are far apart:

```text
A[0,i], A[1,i], A[2,i], A[3,i]
```

These are separated by `K` elements in memory.

The GPU executes memory instructions for a warp, usually 32 threads. If those 32 threads request neighboring addresses, the hardware can combine the requests into a small number of large memory transactions.

Good/coalesced pattern:

```text
thread 0 loads address 1000
thread 1 loads address 1004
thread 2 loads address 1008
thread 3 loads address 1012
...
```

These are consecutive `float32` values, because each float is 4 bytes. The GPU can fetch them together.

Bad/non-coalesced pattern:

```text
thread 0 loads address 1000
thread 1 loads address 26368
thread 2 loads address 51736
thread 3 loads address 77104
...
```

Now the addresses are spread out. The GPU has to issue more memory transactions, so it wastes bandwidth and time.

The important relationship is:

```text
consecutive threadIds in a warp
    should ideally access
consecutive memory addresses
```

The first line is about thread scheduling. The second line is about array memory layout. Coalescing happens when those two orders line up.

For the naive matmul mapping, nearby threads in a warp vary mostly in `threadIdx.x`, and the code used:

```text
x = blockIdx.x * blockDim.x + threadIdx.x
y = blockIdx.y * blockDim.y + threadIdx.y
```

So consecutive threads get different `x` values but the same or similar `y` values. For loading `A[x,i]`, that means nearby threads access:

```text
thread 0: A[x+0, i]
thread 1: A[x+1, i]
thread 2: A[x+2, i]
thread 3: A[x+3, i]
```

Those are different rows at the same column. In row-major memory, those are not consecutive. They are far apart by a stride of `K`.

The coalescing kernel changes how threads are assigned to `C[x,y]` so consecutive threads vary in `y` instead:

```text
x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
```

Now nearby threads access the same row but neighboring columns of `B`:

```text
thread 0: B[i, y+0]
thread 1: B[i, y+1]
thread 2: B[i, y+2]
thread 3: B[i, y+3]
```

In row-major memory, those are consecutive addresses, so the warp's loads can be coalesced.

Tiny mental picture:

```text
Memory likes this:

  thread 0 -> value 0
  thread 1 -> value 1
  thread 2 -> value 2
  thread 3 -> value 3

Memory dislikes this:

  thread 0 -> value 0
  thread 1 -> value 4092
  thread 2 -> value 8184
  thread 3 -> value 12276
```

So the answer is: coalescing helps because GPU memory fetches chunks, not isolated scalar values efficiently. Consecutive `threadId`s are not automatically efficient; they are efficient only when those consecutive threads also request consecutive memory addresses.

**Related blog section:**

Kernel 2: Global Memory Coalescing

### Q17. What if the matrix dimension is huge, like 100000?

**Question:**

If the matrix is very big, for example dimension `100000`, how can each thread handle such a big dimension?

**Answer:**

In the naive kernel, each valid thread computes one output element `C[x,y]`. If `K = 100000`, then that one thread would run a loop of length `100000`:

```text
for i = 0 to 99999:
  tmp += A[x,i] * B[i,y]
```

So from a programming point of view, yes, a thread can handle a long dot product. It just loops longer.

But there are two big practical issues.

First, the work becomes enormous. For square matrix multiply with size `100000`:

```text
total FLOPs ≈ 2 * 100000^3
      = 2 * 10^15 FLOPs
```

That is `2 quadrillion` floating-point operations.

Second, the memory may not fit. One `100000 x 100000` float32 matrix has:

```text
100000^2 = 10,000,000,000 elements
10,000,000,000 * 4 bytes = 40,000,000,000 bytes ≈ 40 GB
```

For `A`, `B`, and `C`, that is about:

```text
3 * 40 GB = 120 GB
```

before counting temporary buffers, alignment, optimizer state, or other memory needs. Many GPUs cannot hold that all at once.

So the short answer is:

```text
A thread can loop over K = 100000,
but the naive method is inefficient and the full matrices may not fit in memory.
```

Real high-performance matmul does not want each thread to repeatedly read all `100000` values from global memory independently. Instead, it uses tiling:

```text
1. Load a smaller tile of A and B.
2. Let many threads reuse that tile.
3. Accumulate partial results.
4. Move to the next tile along K.
```

Conceptually, for huge `K`, the dot product is split into chunks:

```text
C[x,y] = sum over all K

     = sum over K tile 0
     + sum over K tile 1
     + sum over K tile 2
     + ...
```

For very large matrices, systems may also split the computation across multiple GPUs or stream tiles from CPU memory/storage, because a single GPU may not have enough memory.

Important distinction:

```text
Can one thread express the computation? yes.
Is that how fast matmul is implemented? no.
```

The later kernels in the blog are about making this practical: instead of every thread loading huge rows/columns independently, blocks cooperate to reuse data through shared memory and registers.

**Related blog section:**

Kernel 3 and later: tiling/shared-memory optimization motivation

## Open Confusions

Use this section for things that still feel fuzzy after an answer.

- TBD

## Takeaways

- The first big win is making global memory accesses coalesced.
- Shared memory helps when many threads reuse the same input tile.
- Computing more outputs per thread can reduce memory traffic per output, but increases register use.
- High performance is a balance among global memory bandwidth, shared memory pressure, register pressure, occupancy, and instruction throughput.
- cuBLAS is fast partly because it contains many specialized kernels and dispatches based on shape, dtype, and hardware.
