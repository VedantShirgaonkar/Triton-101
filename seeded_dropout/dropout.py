# trying parallel pseudo-random no state generation in triton for dropout on SRAM, instead of calling a separate kernel for random no generation and storing it on HBM

import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _seeded_dropout_kernel(
    input_ptr, output_ptr,
    p, seed,
    n_elements,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask = mask)
    random_tensor = tl.rand(seed, offsets) # rand->uniform distribution between 0 and 1, randn-> normal distribution with mean 0 and variance 1, randint-> random integers between low and high
    x_keep = random_tensor > p
    output = tl.where(x_keep, x / (1-p), 0.0) # if we keep the value, we divide by (1-p) to maintain the expected value, if we drop the value, we set it to 0
    tl.store(output_ptr + offsets, output, mask = mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout_kernel[grid](
        x, output,
        p, seed,
        n_elements, 
        BLOCK_SIZE=1024
    )
    return output


x = torch.randn(size = (8,), device=DEVICE)
output1 = seeded_dropout(x, p=0.5, seed=42)
output2 = seeded_dropout(x, p=0.5, seed=43)
print("Input:", x)
print("Output with seed 42:", output1)
print("Output with seed 43:", output2)