# Complete Guide to GPU Architecture and Triton Programming

## Table of Contents
1. [Introduction](#introduction)
2. [GPU Hardware Architecture](#gpu-hardware-architecture)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Execution Model](#execution-model)
5. [GPU Scheduling and Occupancy](#gpu-scheduling-and-occupancy)
6. [What is Triton?](#what-is-triton)
7. [Practical Application: Your Code](#practical-application-your-code)
8. [Key Design Patterns](#key-design-patterns)

---

## Introduction

GPUs are fundamentally different from CPUs. While CPUs optimize for latency (low execution time), GPUs optimize for throughput (maximum parallel execution). Understanding GPU architecture is critical for writing efficient kernel code.

**Key Principle**: GPU = Thousands of threads organized hierarchically, all trying to hide memory latency through parallelism.

---

## GPU Hardware Architecture

### The GPU Device

A modern GPU (e.g., RTX 4090) consists of:

- **~128 Streaming Multiprocessors (SMs)** - the primary compute units
- **~200-300 GB/s memory bandwidth** - the highway for data
- **GPU HBM (High Bandwidth Memory)** - main memory (24 GB)
- **Device-wide L2 Cache** - unified cache (2-12 MB)
- **PCIe Connection** - link to CPU

### Streaming Multiprocessor (SM)

Each SM is a semi-independent compute core with:

- **64 CUDA Cores (FP32)** - floating-point compute units
- **~256 KB Register File** - private per-thread storage
- **~96-192 KB SRAM/Shared Memory** - shared by running blocks
- **L1 Cache (~32 KB)** - automatic caching of loads/stores
- **Warp Scheduler** - decides which warp executes each cycle

**Critical Property**: Only ONE kernel can run on an SM at a time. All its resources belong entirely to that kernel's blocks.

### CUDA Cores and Execution

CUDA cores are NOT equivalent to threads (1-to-1 mapping). Instead:

```
64 CUDA cores + 32-thread warp = Resources shared by all threads
All 32 threads execute the SAME instruction in parallel (SIMD - Single Instruction, Multiple Data)
Multiple warps time-multiplex on the same cores
```

---

## Memory Hierarchy

### Complete Memory Map

```
GPU (1 device)
│
├─ L2 Cache (~2-12 MB, device-wide, shared by ALL threads)
├─ HBM (GPU Main Memory, ~24 GB, device-wide, shared by ALL threads)
│
└─ SMs (128 per GPU)
    │
    └─ SM 0 (can run 1-8 blocks simultaneously)
        │
        ├─ Block A (e.g., 256 threads)
        │   ├─ Registers: ~256 KB (divided among threads)
        │   │   └─ Each thread has ~1 KB available from pool
        │   │   └─ Thread 0: private, ~10-20 registers (~40-80 bytes)
        │   │   └─ Thread 1: private, ~10-20 registers (cannot access Thread 0's)
        │   │   └─ ... 256 threads total
        │   │
        │   └─ SRAM (Shared Memory): ~96 KB (shared by ALL 256 threads in Block A)
        │       └─ When Block A completes, this SRAM is freed
        │       └─ Completely isolated from Block B's SRAM
        │
        ├─ Block B (also running simultaneously, if it fits!)
        │   ├─ Registers: ~256 KB (separate register pool from Block A)
        │   └─ SRAM: ~96 KB (separate, cannot see Block A's SRAM)
        │
        └─ L1 Cache: ~32 KB (shared by ALL threads on SM 0)
            └─ Automatically caches loads from both Block A and Block B
```

### Memory Access by Component

| Component | Access Type | Scope | Latency |
|-----------|-------------|-------|---------|
| **Register** | Read-Write (Private) | Per-thread only | ~1 cycle |
| **SRAM/Shared** | Read-Write (Shared) | Per-block (block threads only) | ~10-20 cycles |
| **L1 Cache** | Read-Only (Automatic) | Per-SM (all threads on SM) | ~20-50 cycles |
| **L2 Cache** | Read-Only (Automatic) | Device-wide (all threads) | ~50-150 cycles |
| **HBM/DRAM** | Read-Write (Device) | Device-wide (all threads) | ~100-400 cycles |
| **CPU RAM** | Read-Write (PCIe) | System-wide | 1000+ cycles |

### Access Rules

1. **Registers**: A thread CANNOT access another thread's registers, even in the same warp
2. **SRAM**: Threads in a block can access the block's SRAM. Other blocks cannot access it
3. **L1 Cache**: Automatic, transparent to programmer. All threads on SM benefit from cache hits
4. **L2 Cache**: Transparent, device-wide. Any cache miss goes here before HBM
5. **HBM**: Accessible by all threads on the GPU. Slowest, bottleneck for most kernels

---

## Execution Model

### Thread Organization Hierarchy

Threads are organized in a strict hierarchy:

```
Grid (all blocks launched in one kernel call)
 ├─ Thread Block 0 (atomic scheduling unit, runs on one SM)
 │   ├─ Warp 0 (Threads 0-31)          ← Execution unit
 │   ├─ Warp 1 (Threads 32-63)         ← Execution unit
 │   ├─ Warp 2 (Threads 64-95)         ← Execution unit
 │   └─ ... more warps
 │
 ├─ Thread Block 1
 │   ├─ Warp 8 (Threads 0-31 of Block 1)
 │   └─ ...
 │
 └─ ... more blocks
```

### Understanding Warps

**A warp is a group of 32 consecutive threads that execute the same instruction in lockstep.**

**Critical Properties**:

1. **Automatic Grouping**: Warps are created automatically by GPU hardware. You don't explicitly create them.
   - Threads 0-31 form Warp 0
   - Threads 32-63 form Warp 1
   - Threads 64-95 form Warp 2
   - etc.

2. **SIMD Execution**: All 32 threads execute the exact same instruction on different data simultaneously
   ```
   Warp 0 (32 threads all executing: output[idx] = input[idx] + 1)
   Thread 0:  output[0] = input[0] + 1
   Thread 1:  output[1] = input[1] + 1
   Thread 2:  output[2] = input[2] + 1
   ...
   Thread 31: output[31] = input[31] + 1
   └─ All complete in ~1 cycle (parallel execution)
   ```

3. **Warp Divergence**: If threads take different execution paths, the GPU serializes
   ```cuda
   // BAD - warp divergence:
   if (threadIdx.x % 2 == 0) {
       expensive_operation();
   } else {
       different_operation();
   }
   // Result: Threads 0,2,4,... execute first (~100 cycles)
   //         Then threads 1,3,5,... execute (~100 cycles)
   //         Total: ~200 cycles instead of ~100
   ```

4. **No Direct Memory Ownership**: Warps don't own memory. Each thread has registers, but the warp as a unit doesn't have a memory level.

5. **Physical Location**: All threads in a block run on ONE SM. All warps in that block execute on that same SM.

### Understanding Thread Blocks

**A thread block is a group of threads (typically 128-1024) that:**

- Execute on the SAME SM
- Can communicate via SRAM
- Can synchronize with barriers (`__syncthreads()`)
- Belong to one kernel + one grid position

**Block Scheduling**:
- GPU scheduler assigns blocks to available SMs
- Multiple blocks can run on the same SM simultaneously if resources permit
- A block is atomic - it cannot be split across multiple SMs

**Example**: 512 thread block splits into 16 warps (512/32)
```
Block (512 threads)
├─ Warp 0 (Threads 0-31)
├─ Warp 1 (Threads 32-63)
├─ Warp 2 (Threads 64-95)
...
└─ Warp 15 (Threads 480-511)
```

### Understanding the Grid

**The grid is the complete problem space - all blocks launched in one kernel call.**

```python
# CUDA syntax:
kernel<<<num_blocks, block_size>>>(args);
       ↑                ↑
       └─ Grid dimensions    └─ Threads per block
```

**Grid can be 1D, 2D, or 3D**:
- 1D: `<<<256, 256>>>` = 256 blocks
- 2D: `<<<(16, 16), (16, 16)>>>` = 256 blocks in 16×16 arrangement
- 3D: `<<<(8, 4, 2), 256>>>` = 64 blocks in 3D arrangement

**Grid Properties**:
- Fixed at kernel launch (cannot dynamically add blocks)
- Blocks in the grid are independent - no direct communication
- GPU scheduler assigns blocks to SMs automatically

---

## GPU Scheduling and Occupancy

### The Warp Scheduler

Each SM has a **warp scheduler** that decides which warps execute each cycle.

**Execution Timeline**:

```
SM with 8 warps (256 threads):

Cycle 0:
├─ Warp 0 and Warp 1 execute next instruction
├─ Uses 64 cores (32 + 32 threads)
└─ All other warps wait to be scheduled

Cycle 1:
├─ Warp 0: issues LOAD from HBM → goes WAITING status
├─ Warp 1: issues LOAD from HBM → goes WAITING status
├─ Scheduler picks Warp 2 and Warp 3 (next ready)
└─ Warp 2 & 3 execute (Warp 0 & 1 waiting for memory ~300 cycles)

Cycle 2:
├─ Warps 2 & 3 continue
├─ Warps 0 & 1 still WAITING for memory
└─ Scheduler keeps Warps 2 & 3 active

... (many cycles pass) ...

Cycle ~300:
├─ Warps 0 & 1's memory arrives! Status changes to READY
├─ Scheduler can now run any of: 0, 1, 2, 3, ... (all ready)
├─ Picks Warps 0 & 1
└─ Execution resumes

This is called "latency hiding" - while one warp waits, others compute
```

### Occupancy

**Occupancy** = Percentage of hardware resources utilized

```
Max warps per SM: 64 (varies by GPU generation)
If you have 8 warps running: 8/64 = 12.5% occupancy

Why does occupancy matter?
- Low occupancy: Few warps ready to execute
- Long stalls: Warp waiting for memory, nothing else to do
- Idle cores: GPU compute power wasted

High occupancy: Many warps available
- Scheduler always finds ready work
- Memory latency is hidden
- Excellent performance
```

### Resource Constraints

Multiple factors limit how many blocks can run on one SM:

```
Register Constraints:
├─ Total registers per SM: ~256 KB = 65,536 registers
├─ Each thread in block needs registers (determined by compiler)
├─ If each thread needs 20 registers
├─ And block has 256 threads
├─ Then block needs: 256 × 20 = 5,120 registers
├─ Max blocks: 65,536 / 5,120 ≈ 12 blocks possible

SRAM Constraints:
├─ Total SRAM per SM: ~192 KB
├─ Each block allocates SRAM (you specify via __shared__)
├─ If each block requests 96 KB SRAM
├─ Then max blocks: 192 KB / 96 KB = 2 blocks possible

Actual Limit = MIN(register limit, SRAM limit) = 2 blocks
```

### The Occupancy Calculation

```python
def calculate_occupancy():
    total_registers = 65_536
    total_sram = 192_000  # bytes
    
    # Block properties
    block_threads = 256
    registers_per_thread = 20  # compiler calculates
    shared_memory_size = 96_000  # bytes (user specifies)
    
    # How many blocks can fit?
    registers_per_block = block_threads * registers_per_thread
    max_blocks_by_regs = total_registers // registers_per_block
    max_blocks_by_sram = total_sram // shared_memory_size
    
    max_blocks = min(max_blocks_by_regs, max_blocks_by_sram)
    
    # Actual warps running
    warps_per_block = block_threads // 32
    actual_warps = max_blocks * warps_per_block
    max_possible_warps = 64  # Varies by architecture
    
    occupancy = actual_warps / max_possible_warps
    return occupancy

# Example:
# registers_per_block = 256 * 20 = 5,120
# max_blocks_by_regs = 65,536 / 5,120 ≈ 12
# max_blocks_by_sram = 192,000 / 96,000 = 2
# max_blocks = min(12, 2) = 2
# warps_per_block = 256 / 32 = 8
# actual_warps = 2 * 8 = 16
# occupancy = 16 / 64 = 25%
```

### Scheduling in Practice

```
You launch 1024 blocks (each 256 threads)
GPU has 128 SMs

GPU calculates:
├─ Max blocks per SM: 2 (from occupancy calculation)
├─ Can launch simultaneously: 128 × 2 = 256 blocks
└─ Remaining: 768 blocks queued

Timeline:
T0:   Blocks 0-255 running (all 128 SMs have 2 blocks each)
T1:   First batch finishes
      Blocks 256-511 start on freed SMs
T2:   Blocks 256-511 finish
      Blocks 512-767 start
T3:   Blocks 512-767 finish
      Blocks 768-1023 start
T4:   All blocks complete
```

---

## What is Triton?

### The Problem Triton Solves

Writing optimized CUDA kernels requires:
- Manual thread index calculation
- Manual block/grid dimension tuning
- Memory access optimization (coalescing, cache efficiency)
- Register/SRAM allocation management
- Occupancy calculation and tuning
- Many lines of verbose code

Result: Takes days to write and optimize a single kernel.

### What Triton Does

Triton is a **Python-based compiler framework** that:

1. **Abstracts away low-level details**: You think in "programs" (similar to blocks) processing chunks of data
2. **Automatically generates optimized CUDA**: Handles thread organization, warp synchronization, memory access patterns
3. **Supports `tl.constexpr` for specialization**: Different BLOCK_SIZE values generate different specialized kernels
4. **Handles vectorization automatically**: Loading/storing operations are compiled to efficient SIMD code
5. **Provides high-level operations**: `tl.load()`, `tl.store()`, `tl.reduce()`, etc. automatically generate correct code

**Key Abstraction: The Program**

Instead of thinking about threads, Triton lets you think about "programs":
- Each program processes a chunk of data (defined by BLOCK_SIZE)
- Multiple programs run in parallel on the GPU
- Triton automatically creates the underlying thread blocks

### Triton vs CUDA

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Lines of Code** | 50-200 | 20-50 |
| **Development Time** | Days | Hours |
| **Optimization Level** | High manual effort | Mostly automatic |
| **Performance Overhead** | None | ~0-5% |
| **Learning Curve** | Steep | Moderate |
| **When to Use** | Maximum performance needed | Good performance, quick dev |

### Why Use Triton?

1. **Speed**: 5-10x faster development than CUDA
2. **Correctness**: Less boilerplate means fewer bugs
3. **Flexibility**: Easy to experiment with different block sizes and strategies
4. **Integration**: Seamlessly integrates with PyTorch and other ML frameworks
5. **Specialization**: `tl.constexpr` generates optimal kernels for each block size

### Triton Usage in Industry

- **PyTorch**: Native Triton integration for custom CUDA kernels
- **flashattention**: Optimized attention kernels written in Triton
- **vLLM**: Fast LLM inference using Triton kernels
- **OpenAI/Research**: Original developer, used extensively for custom ops

---

## Practical Application: Your Code

### Vector Addition (`vector_add.py`)

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)  # Which program am I? 0, 1, 2, ...
    
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # All indices this program processes
    mask = offsets < n_elements  # Handle boundary case
    
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()  # Total elements to process
    grid = lambda meta : (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # If n_elements = 256,000 and BLOCK_SIZE = 1024:
    # grid = 250 programs (ceil(256000/1024))
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

**Execution Flow**:

1. Launch 250 programs (blocks) with BLOCK_SIZE=1024
2. GPU scheduler assigns first 128 programs to 128 SMs (one block per SM)
3. Each program processes 1024 elements:
   - Program 0: elements [0:1024]
   - Program 1: elements [1024:2048]
   - Program 249: elements [254976:256000]

4. Triton compiles this to CUDA that:
   - Creates threads for parallel loading/storing
   - Organizes them into warps automatically
   - Each warp loads/computes in parallel
   - Memory is coalesced efficiently

5. When a program completes, SM becomes available
   - Program 128 starts on SM 0 (now free)
   - Program 129 starts on SM 1
   - Continue until all 250 programs complete

**Performance Characteristics**:
- **Memory bandwidth bound**: Mostly reading/writing, minimal computation
- **GBps metric**: GB per second (how fast can we move data?)
- **Optimal occupancy**: High occupancy important here to hide load latency

### Fused Softmax (`softmax.py`)

```python
def softmax(x):  # x shape: (4096, 2048)
    # ... setup code ...
    
    # Calculate occupancy for this specific kernel
    properties = triton.runtime.driver.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiProcessorCount"]  # e.g., 128
    NUM_REGS = properties["max_num_regs"]
    TOTAL_SRAM_PER_SM = properties["max_shared_mem"]  # e.g., 192 KB
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # Power of 2 for efficiency
    
    # Auto-tune based on occupancy
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    
    # Calculate how many blocks can actually fit
    kernel = _softmax_kernel.warmup(
        x, y, x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    kernel.init_handles()
    
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    
    # Calculate occupancy limits
    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    
    grid = (num_programs, 1, 1)
    
    kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols)
    return y

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    PID = tl.program_id(0)  # Which program block am I?
    row_step = tl.num_programs(0)  # Total number of programs
    
    # Persistent kernel pattern: each block processes multiple independent rows
    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        # Load one row of data
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        
        # Compute softmax in parallel
        row_minus_max = row - tl.max(row, axis=0)  # Horizontal max reduction
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)  # Horizontal sum reduction
        softmax_output = numerator / denominator
        
        # Store result
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

**Execution Flow**:

1. Calculate occupancy and determine how many blocks can fit per SM
   - num_programs = min(128 SMs × programs_per_sm, 4096 rows)
   - If programs_per_sm = 2: num_programs = min(256, 4096) = 256

2. Launch 256 programs (blocks)

3. Each program processes multiple rows (persistent kernel pattern):
   - Program 0: rows [0, 256, 512, ...]
   - Program 1: rows [1, 257, 513, ...]
   - Program 255: rows [255, 511, 767, ...]

4. Within each program:
   - Load one row (2048 elements)
   - Compute max over row
   - Compute exp
   - Compute sum
   - Divide
   - Store result

5. Multi-stage pipelining (`num_stages=4`):
   - Stage 1: Load iteration N+1 data
   - Stage 2: Compute iteration N
   - Stage 3: Store iteration N-1
   - Stage 4: Overlap for latency hiding

**Why Fused Softmax is Faster**:

```
Naive implementation (unfused):
├─ Load row from HBM   → 300 cycles latency
├─ Compute max         → 10 cycles
├─ Store intermediate  → 300 cycles
├─ Load exp data       → 300 cycles
├─ Compute exp         → 10 cycles
├─ Store exp result    → 300 cycles
├─ Load sum data       → 300 cycles
├─ Compute sum         → 10 cycles
├─ Store sum           → 300 cycles
├─ Load denominator    → 300 cycles
├─ Divide              → 10 cycles
└─ Store final result  → 300 cycles
Total: 8 passes through HBM (~3000 cycles bandwidth limited)

Fused implementation:
├─ Load row from HBM   → 300 cycles latency
├─ Compute max, exp, sum, divide (all in registers) → 30 cycles
└─ Store result back   → 300 cycles
Total: 1 pass through HBM (~600 cycles total)

Speedup: ~5x faster! (HBM bandwidth is the real bottleneck)
```

**Performance Characteristics**:
- **Bandwidth bound**: Every element touched = 1 load + 1 store
- **Low arithmetic intensity**: ~2 memory ops per ~50 FLOPs
- **Occupancy critical**: Need high occupancy to hide load latency
- **Multi-staging essential**: Overlap compute with next load

---

## Key Design Patterns

### Pattern 1: Blocking/Chunking

**Problem**: Process 1M elements efficiently

**Solution**: Divide into blocks
```
Grid: 1024 blocks, BLOCK_SIZE=1024
│
├─ Block 0: processes elements [0:1024]
├─ Block 1: processes elements [1024:2048]
└─ ...
└─ Block 1023: processes elements [1047552:1048576]

Advantage: Each block fits on one SM with good occupancy
```

### Pattern 2: Persistent Kernels

**Problem**: Load balancing when number of work items ≠ number of blocks

**Solution**: Blocks process multiple items in a loop
```
for item_idx in range(PID, total_items, num_programs):
    process(item_idx)

Benefits:
- Better load balancing (no idle blocks)
- Amortizes kernel launch overheads
- Cooperates with warp scheduler naturally
```

### Pattern 3: Multi-Stage Pipelining

**Problem**: Hide memory latency while computing

**Solution**: Overlap stages
```
Cycle T:   Load iteration N+1 (goes to HBM queue)
Cycle T+1: Compute iteration N (while N+1 loads)
Cycle T+2: Store iteration N-1 (while N+1 loads and N computes)

Effect: HBM latency (~300 cycles) is mostly hidden
```

### Pattern 4: Fused Operations

**Problem**: Memory moves between HBM and GPU multiple times, very slow

**Solution**: Combine operations in one kernel
```
Instead of:
output = softmax(x)     # Load x, store intermediate
output = e^output       # Load intermediate, store temporary
output = output / sum   # Load temporary, load sum, store result
(3 HBM passes)

Do:
for each_row:
    load x
    max_x = max(x)  (in registers)
    exp_x = exp(x - max_x)  (in registers)
    sum_exp = sum(exp_x)  (in registers)
    output = exp_x / sum_exp  (in registers)
    store output
(1 HBM pass)
```

### Pattern 5: Horizontal Reductions

**Problem**: Need max/sum across entire warp

**Solution**: Use warp-level reduction
```python
# Warp-level max:
value = thread_local_value
value = max(value, tl.shuffle_xor(value, 16))  # Compare with thread 16 away
value = max(value, tl.shuffle_xor(value, 8))   # Compare with thread 8 away
value = max(value, tl.shuffle_xor(value, 4))   # Compare with thread 4 away
value = max(value, tl.shuffle_xor(value, 2))   # Compare with thread 2 away
value = max(value, tl.shuffle_xor(value, 1))   # Compare with adjacent thread
# value now contains max of all 32 threads in warp
```

---

## Quick Reference

### GPU Terminology

| Term | Definition |
|------|-----------|
| **SM** | Streaming Multiprocessor, the compute core of GPU |
| **CUDA Core** | Individual processing unit, 64 per SM |
| **Thread** | Individual instance of kernel code |
| **Warp** | 32 threads executing in lockstep |
| **Block/Thread Block** | 128-1024 threads running on one SM |
| **Grid** | All blocks launched in one kernel call |
| **Occupancy** | % of GPU resources actively used |
| **PID** | Program ID, identifier for which block you are |

### Memory Quick Facts

| Memory | Size Per SM | Access Time | Scope |
|--------|------------|-------------|-------|
| Register | 256 KB | 1 cycle | Per-thread |
| SRAM | 96-192 KB | 10-20 cycles | Per-block |
| L1 Cache | 32 KB | 20-50 cycles | Per-SM |
| L2 Cache | 2-12 MB | 50-150 cycles | Device-wide |
| HBM | 24 GB | 100-400 cycles | Device-wide |

### Triton Quick Facts

| Concept | Meaning |
|---------|---------|
| `@triton.jit` | Decorator for GPU kernel |
| `tl.program_id(0)` | Which block am I? |
| `tl.constexpr` | Compile-time constant, generates specialized kernels |
| `tl.load()` | Load data from memory |
| `tl.store()` | Store data to memory |
| `tl.arange()` | Create offset array |
| `BLOCK_SIZE` | Elements processed per program iteration |
| `num_stages` | Pipelining stages for latency hiding |

---

## Summary

**GPU evolution from your understanding**:

1. GPU has many **SMs** (128 per GPU), each with **64 cores**
2. **Blocks** run on SMs (multiple per SM if they fit)
3. **Blocks** contain **threads** (organized into **warps**)
4. **Threads** execute in **warps** (32-thread groups in lockstep)
5. **Warps** are scheduled to hide memory latency
6. **Memory hierarchy** ranges from fast registers to slow HBM
7. **Occupancy** determines how well memory latency is hidden
8. **Triton** abstracts complexity and auto-generates optimized kernels
9. **Fused kernels** minimize HBM passes (biggest bottleneck)

The goal of GPU programming: **Maximize parallelism while minimizing memory traffic through smart algorithms and occupancy management.**

