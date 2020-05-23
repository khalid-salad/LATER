#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>



/*
These three functions are related with QR factorization

rgsqrf: recursive Gram-Schmidt QR factorization

rhouqr: recursive Householder QR factorization

bhouqr: block Householder QR factorization
*/
void later_rgsqrf(int m, int n, float* A, int lda, float* R, int ldr);

void later_rhouqr(int m, int n, float* A, int lda, float* R, int ldr);

void later_bhouqr(int m, int n, float* A, int lda, float* R, int ldr);


/*
Below functions are the integration of often-used functions
*/

/*
Call startTimer() at first and then call stopTimer() to get the time consumption

Cannot be called nesting
Like:
startTimer();
    ...
    startTimer();
    ...
    stopTimer();
    ...
stopTimer();
This is not supported
*/
void startTimer();

void stopTimer();