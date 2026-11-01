# Softmax

## Problem

Given the query matrix `Q` of size `M&times;d`, key matrix `K` of size
  `N&times;d`, value matrix `V` of size `N&times;d`, and upstream gradient
  `dO` of size `M&times;d`, compute the gradients of the softmax attention operation
  with respect to `Q`, `K`, and `V`. The forward pass computes
  $$O = \text{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d}}\Bigr)V,$$
  where softmax is applied row-wise. The backward pass produces `dQ` (M&times;d),
  `dK` (N&times;d), and `dV` (N&times;d) via:
  $$P = \text{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d}}\Bigr), \quad dV = P^T\,dO,$$
  $$dP = dO\,V^T, \quad dS_{ij} = P_{ij}\!\left(dP_{ij} - \sum_k P_{ik}\,dP_{ik}\right), \quad \widetilde{dS} = \frac{dS}{\sqrt{d}},$$
  $$dQ = \widetilde{dS}\,K, \quad dK = \widetilde{dS}^T Q.$$

## Implementation Requirements

  - Use only GPU native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final results must be stored in `dQ`, `dK`, and `dV`

## Example

Input:

`Q` (2&times;4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]
`K` (3&times;4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}
\]
`V` (3&times;4):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0
\end{bmatrix}
\]
`dO` (2&times;4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]

Output:

`dQ` (2&times;4):
\[
\begin{bmatrix}
-0.74 & 0.10 & 0.65 & 0.00 \\
-0.55 & 0.00 & 0.55 & 0.00
\end{bmatrix}
\]
`dK` (3&times;4):
\[
\begin{bmatrix}
-0.74 & -0.55 & 0.00 & 0.00 \\
0.10 & 0.00 & 0.00 & 0.00 \\
0.65 & 0.55 & 0.00 & 0.00
\end{bmatrix}
\]
`dV` (3&times;4):
\[
\begin{bmatrix}
0.45 & 0.27 & 0.00 & 0.00 \\
0.27 & 0.45 & 0.00 & 0.00 \\
0.27 & 0.27 & 0.00 & 0.00
\end{bmatrix}
\]

## Constraints

  - Matrix `Q` and `dO` are of size `M&times;d`; matrices `K`, `V` are of size `N&times;d`

  - 1 &le; `M`, `N` &le; 100,000

  - 1 &le; `d` &le; 128

  - All input values are `float32`

  - Performance is measured with `M` = 8,192, `N` = 4,096, `d` = 128

## Details

| Field | Value |
|-------|-------|
| Problem ID | softmax |
| Language | CUDA |
| Problem URL | [LeetGPU](https://leetgpu.com/challenges/softmax) |
| Submitted At | 2026-11-01 09:44:03 UTC |

---
*Synced automatically by [LeetGPU Sync](https://github.com/your-username/leetgpu-sync)*
