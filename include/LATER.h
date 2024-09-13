#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "OC_gemm.h"

struct cudaCtxt {
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
};

/**
 * QR FACTORIZATION
 */

/**
 * recursive Gram-Schmidt QR factorization
 */
void later_rgsqrf(cudaCtxt ctxt, int m, int n, float* A, int lda, float* R, int ldr, float* work, int lwork,
                  __half* hwork, int lhwork);

/**
 * recursive Householder QR factorization
 */
void later_rhouqr(cudaCtxt ctxt, int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work,
                  int lwork, __half* hwork, int lhwork, float* U);

/**
 * form explicit Q from rhouqr result
 */
void later_ormqr(int m, int n, float* W, int ldw, float* Y, int ldy, float* work);

/**
 * block Householder QR factorization
 */
void later_bhouqr(int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work, int lwork,
                  __half* hwork, int lhwork, float* U);

/**
 * form explicit Q from bhouqr result
 */
void later_ormqr2(int m, int n, float* W, int ldw, float* Y, int ldy, float* work);

void later_oc_qr_rec(cudaCtxt ctxt, int m, int n, float* A, int lda, float* R, int ldr,
                     std::shared_ptr<Mem_pool> _pool);

void later_oc_qr_blk(cudaCtxt ctxt, int m, int n, float* A, int lda, float* R, int ldr,
                     std::shared_ptr<Mem_pool> _pool);

/*
BLAS-3 matrix operations
*/

/**
 * recursive triangular solve
 */
void later_rtrsm(cublasHandle_t handle, char uplo, char leri, char trans, int m, int n, float* A, int lda, float* B,
                 int ldb, __half* work);

/**
 * recursive symmetric rank k update
 * C := alpha * A * A**T + beta * C or C := alpha * A**T * A + beta * C
 */
void later_rsyrk(cublasHandle_t handle, int n, int k, float alpha, float* A, int lda, float beta, float* C, int ldc,
                 __half* work);

/**
 * recursive triangular matrix multiply
 */
void later_rtrmm(int m, int n, float* A, int lda, float* B, int ldb, float* C, int ldc, float* tempC, __half* hwork);

/*
These functions are related to EVD
*/

void later_qdwh_polar(cudaCtxt ctxt, int n, float* A, int lda, float* H, int ldh, float* tmpA, float* work,
                      __half* hwork);

void later_sy2sb_rec(cudaCtxt ctxt, int n, int ns, float* A, float* oriA, int lda, float* work, int lwork,
                     __half* hwork, int lhwork);

void ssytrd_sy2sb(cudaCtxt ctxt, int n, int nb, float* A, float* A_cpy, int lda, float* work, int lwork, __half* hwork,
                  int lhwork);

/*
These functions are related to Cholesky factorization
*/

/**
 * cholesky factorization of device matrix A
 * A must be positive definite
 */
void later_rpotrf(char uplo, int n, float* A, int lda, float* work, __half* hwork);

/*
Below functions are the integration of often-used functions
*/

/*


This is not supported
*/

/**
 * Call startTimer() at first and then call stopTimer() to get the time consumption
 *
 * Cannot be called nesting, e.g.
 * startTimer();
 *  ...
 *  startTimer();
 *  ...
 *  stopTimer();
 *  ...
 * stopTimer();
 */
void startTimer();

/**
 * returns time consumption
 * can only be called after startTimer() has been called
 * Cannot be called nesting, e.g.
 * startTimer();
 *  ...
 *  startTimer();
 *  ...
 *  stopTimer();
 *  ...
 * stopTimer();
 */
float stopTimer();

/**
 * convert single matrix to half matrix
 */
__global__ void s2h(int m, int n, float* as, int ldas, __half* ah, int ldah);

/**
 * convert half matrix to single matrix
 */
__global__ void h2s(int m, int n, __half* ah, int ldah, float* as, int ldas);

/**
 * copy half-precision matrix from as to ah
 */
__global__ void h2h(int m, int n, __half* as, int ldas, __half* ah, int ldah);

/**
 * copy single precision matrix from as to aas
 */
__global__ void s2s(int m, int n, float* as, int ldas, float* aas, int ldaas);

/**
 * transpose a part/whole single matrix on GPU
 */
__global__ void transpose(int m, int n, float* dA, int lda, float* tmpA);

/**
 * copies the lower triangle to upper triangle for square matrix on GPU
 */
__global__ void copy_lower_to_upper(int ldax, int lday, float* a);

/**
 * Generate a matrix whose values are from uniform(0, 1) on GPU
 */
void generateUniformMatrix(float* dA, int m, int n);

/**
 * Generate a matrix whose values are from normal(0, 1) on GPU
 */
void generateNormalMatrix(float* dA, int m, int n);

/**
 * return host matrix hA with random values
 * such that hA is a positive definite matrix
 */
float* uniformPositiveDefiniteMatrix(size_t n);

/**
 * return the f-norm of a device matrix dA
 */
float snorm(int m, int n, float* dA);

/**
 * set device matrix a to be an identity matrix
 */
__global__ void setEye(int m, int n, float* a, int lda);

/**
 * Initialize device matrix a with value val
 */
__global__ void setInitialValue(int m, int n, float* a, int lda, float val);

/*
template<typename T>
void printMatrixDeviceBlock(char *path,int m,int n, T* A, int lda)
{
    std::ofstream file;
    file.open(path);
    float *Ah = new T[lda*n];
    cudaMemcpy(Ah, A, sizeof(T)*lda*n, cudaMemcpyDeviceToHost);
    file << std::setprecision(7);
    for(int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            file << Ah[i+j*lda];
            if (j!=n-1) file << ' ';
        }
        file << std::endl;
    }
    delete[] Ah;
}*/
template <typename T> void printMatrixDeviceBlock(const char* const filename, int m, int n, T* dA, int lda) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        printf("fault!\n");
        return;
    }
    // printf("Perform printmatrixdevice\n");
    float* ha;
    ha = (float*)malloc(sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cudaMemcpy(&ha[0], &dA[i + j * lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1)
                fprintf(f, "\n");
            else
                fprintf(f, ",");
        }
    }
    fclose(f);
    // cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    // printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// utilities
__global__ void deviceCopy(int m, int n, float* da, int lda, float* db, int ldb);
void sSubstract(cublasHandle_t handle, int m, int n, float* dA, int lda, float* dB, int ldb);
__global__ void clearTri(char uplo, int m, int n, float* a, int lda);

__global__ void sSubstractAndSquare(int m, int n, float* dA, int lda, float* dB, int ldb);

#ifdef DBGPRINT
#define dbgprintf(fmt, ...)                                                                                            \
    do {                                                                                                               \
        printf(fmt, __VA_ARGS__);                                                                                      \
    } while (0)
#else
#define dbgprintf(fmt, ...) void(0)
#endif

#ifdef DEBUG_CUDA_KERNEL_LAUNCH
#define CHECK_KERNEL()                                                                                                 \
    do {                                                                                                               \
        gpuErrchk(cudaDeviceSynchronize());                                                                            \
        gpuErrchk(cudaPeekAtLastError());                                                                              \
    } while (0)
#else
#define CHECK_KERNEL(x)                                                                                                \
    do {                                                                                                               \
    } while (0)
#endif

void print_env();

size_t free_mem();
