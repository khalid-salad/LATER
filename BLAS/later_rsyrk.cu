#include "LATER.h"

#include <cuda_fp16.h>

#define BLOCKSIZE 2048

float pan = 0.0;
float ge = 0.0;

void later_rsyrk(cublasHandle_t handle, int n, int k, float alpha, float *A,
                 int lda, float beta, float *C, int ldc, __half *hwork) {
  if (n <= BLOCKSIZE) {
    __half *ah = hwork;

    dim3 grid1((n + 1) / 32, (k + 1) / 32);
    dim3 block1(32, 32);
    s2h<<<grid1, block1>>>(n, k, A, lda, ah, n);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, ah,
                 CUDA_R_16F, n, ah, CUDA_R_16F, n, &beta, C, CUDA_R_32F, ldc,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return;
  }
  later_rsyrk(handle, n / 2, k, alpha, A, lda, beta, C, ldc, hwork);
  __half *Ah = hwork;
  __half *Bh = hwork + n / 2 * k;

  dim3 grid((n / 2 + 1) / 32, (k + 1) / 32);
  dim3 block(32, 32);
  s2h<<<grid, block>>>(n / 2, k, A + n / 2, lda, Ah, n / 2);
  s2h<<<grid, block>>>(n / 2, k, A, lda, Bh, n / 2);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n / 2, n / 2, k, &alpha, Ah,
               CUDA_R_16F, n / 2, Bh, CUDA_R_16F, n / 2, &beta, C + n / 2,
               CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  later_rsyrk(handle, n / 2, k, alpha, A + n / 2, lda, beta,
              C + n / 2 + ldc / 2 * n, ldc, hwork);
}