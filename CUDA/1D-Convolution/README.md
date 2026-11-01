# 1D Convolution

## Problem

Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved
  output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied
  where it fully overlaps with the input.

  
  

  
  input

  
  
  
  
  
  
  
  1
  2
  3
  4
  5

  
  

  
  kernel

  
  
  
  
  
  1
  0
  -1

  
  1&#xd7;1
  2&#xd7;0
  3&#xd7;(-1)

  
  = 1 + 0 + (-3) = -2

  
  - 
  
    
      

    
  

  
  output

  
  
  -2

  
  &#x2026;

  The input consists of two arrays:

  `input`: A 1D array of 32-bit floating-point numbers.

  - `kernel`: A 1D array of 32-bit floating-point numbers representing the convolution kernel.

The output should be written to the `output` array, which will have a size of `input_size - kernel_size + 1`.

  The convolution operation is defined mathematically as:

\[
output[i] = \sum_{j=0}^{kernel\_size-1} input[i + j] \cdot kernel[j]
\]

  where \(i\) ranges from 0 to \(input\_size - kernel\_size\).

## Implementation Requirements

  - Use only native features (external libraries are not permitted)

  - The
    `solve` function signature must remain unchanged
  

  - The final result must be stored in the array
    `output`
  

## Example 1:

```

Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]

```

## Example 2:

```

Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]

```

## Constraints

  - 1 &le; `input_size` &le; 1,500,000

  - 1 &le; `kernel_size` &le; 2047

  - `kernel_size` &le; `input_size`

  - Performance is measured with `input_size` = 1,500,000, `kernel_size` = 2,047

## Details

| Field | Value |
|-------|-------|
| Problem ID | 1d-convolution |
| Language | CUDA |
| Problem URL | [LeetGPU](https://leetgpu.com/challenges/1d-convolution) |
| Submitted At | 2026-11-01 09:44:38 UTC |

---
*Synced automatically by [LeetGPU Sync](https://github.com/your-username/leetgpu-sync)*
