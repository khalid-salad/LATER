#include "LATER.h"
#include <assert.h>
#include <cuda_fp16.h>

#define BLOCKSIZE 2048
#define LWORK 65536

int chol_info;
int lwork;

int* dev_info;

void u_potrf(cudaCtxt ctxt, int n, float* A, int lda, float* work, __half* hwork) {
    if (n <= BLOCKSIZE) {
        cusolverDnSpotrf(ctxt.cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, lda, work, LWORK, dev_info);
        return;
    } else {
        auto n1 = n / 2;
        auto n2 = n - n1;
        u_potrf(ctxt, n1, A, lda, work, hwork);
        auto A12 = A + n * n1;
        auto A22 = A + n1 + n * n1;
        later_rtrsm(ctxt.cublas_handle, 'u', 'l', 't', n1, n2, A, lda, A12, lda, hwork);
        later_rsyrk(ctxt.cublas_handle, n2, n1, -1.0, A12, lda, 1.0, A22, lda, hwork);
        u_potrf(ctxt, n2, A22, lda, work, hwork);
    }
}

void l_potrf(cudaCtxt ctxt, int n, float* A, int lda, float* work, __half* hwork) {
    if (n <= BLOCKSIZE) {
        cusolverDnSpotrf(ctxt.cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, LWORK, dev_info);
        return;
    } else {
        auto n1 = n / 2;
        auto n2 = n - n1;
        l_potrf(ctxt, n1, A, lda, work, hwork);
        auto A21 = A + n1;
        auto A22 = A21 + n1 + n * n1;
        later_rtrsm(ctxt.cublas_handle, 'l', 'r', 't', n2, n1, A, lda, A21, lda, hwork);
        later_rsyrk(ctxt.cublas_handle, n2, n1, -1.0, A21, lda, 1.0, A22, lda, hwork);
        l_potrf(ctxt, n2, A22, lda, work, hwork);
    }
}

void later_rpotrf(char uplo, int n, float* A, int lda, float* work, __half* hwork) {
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle);
    cusolverDnCreate(&ctxt.cusolver_handle);
    cudaMalloc(&dev_info, sizeof(*dev_info));

    if (uplo == 'l') {
        l_potrf(ctxt, n, A, lda, work, hwork);
    } else if (uplo == 'u') {
        u_potrf(ctxt, n, A, lda, work, hwork);
    }

    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
    return;
}