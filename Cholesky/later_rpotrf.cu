#include "LATER.h"
#include <assert.h>
#include <cuda_fp16.h>

#define BLOCKSIZE 2048
#define LWORK 65536

int chol_info;
int lwork;

int *dev_info;

float chol_panel = 0.0;
float chol_gemm = 0.0;

/*
This function performs recursive Cholesky factorization
*/

void u_potrf(cudaCtxt ctxt, int n, float *A, int lda, float *work,
             __half *hwork) {
  float ms;
  if (n <= BLOCKSIZE) {
    cusolverDnSpotrf(ctxt.cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda,
                     work, LWORK, dev_info);

    return;
  } else {
    auto n1 = n / 2;
    auto n2 = n - n1;
    u_potrf(ctxt, n1, A, lda, work, hwork);
    auto A12 = A + n * n1;
    auto A22 = A + n1 + n * n1;
    later_rtrsm(ctxt.cublas_handle, 'l', 'l', 't', n1, n2, A, lda, A12, lda,
                hwork);
    later_rsyrk(ctxt.cublas_handle, n2, n1, -1.0, A12, lda, 1.0, A22, lda,
                hwork);
    ms = stopTimer();
    chol_gemm += ms;
    auto tflops = 2.0 * n / 2 * n / 2 * n / 2 / ms / 1e9;
    u_potrf(ctxt, n2, A22, lda, work, hwork);
  }
}

void l_potrf(cudaCtxt ctxt, int n, float *A, int lda, float *work,
             __half *hwork) {
  float ms;
  if (n <= BLOCKSIZE) {
    startTimer();
    cusolverDnSpotrf(ctxt.cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda,
                     work, LWORK, dev_info);

    chol_panel += stopTimer();

    return;
  } else {
    auto n1 = n / 2;
    auto n2 = n - n1;
    l_potrf(ctxt, n1, A, lda, work, hwork);
    startTimer();
    auto A21 = A + n1;
    auto A22 = A21 + n1 + n * n1;
    later_rtrsm(ctxt.cublas_handle, 'l', 'r', 't', n2, n1, A, lda, A21, lda,
                hwork);
    later_rsyrk(ctxt.cublas_handle, n2, n1, -1.0, A21, lda, 1.0, A22, lda,
                hwork);
    ms = stopTimer();
    chol_gemm += ms;
    auto tflops = 2.0 * n / 2 * n / 2 * n / 2 / ms / 1e9;
    l_potrf(ctxt, n2, A22, lda, work, hwork);
  }
}

void later_rpotrf(char uplo, int n, float *A, int lda, float *work,
                  __half *hwork) {
  cudaCtxt ctxt;
  cublasCreate(&ctxt.cublas_handle);
  cusolverDnCreate(&ctxt.cusolver_handle);
  // printMatrixDeviceBlock("A.csv", n,n, A,n);
  cudaMalloc(&dev_info, sizeof(int));

  if (uplo == 'l') {
    l_potrf(ctxt, n, A, lda, work, hwork);
  } else if (uplo == 'u') {
    u_potrf(ctxt, n, A, lda, work, hwork);
  }

  cublasDestroy(ctxt.cublas_handle);
  cusolverDnDestroy(ctxt.cusolver_handle);
  return;
}