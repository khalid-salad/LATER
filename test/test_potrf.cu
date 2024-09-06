#include "LATER.h"

int n;
bool checkFlag = false;

int parseArguments(int argc, char *argv[]) {
  n = atoi(argv[1]);
  printf("n = %d\n", n);
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "-check") == 0) {
      checkFlag = true;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: ./test_potrf n [options]\n");
    printf("Options:\n\t-check: enable checking the orthogonality and backward "
           "error\n");
    return EXIT_FAILURE;
  }
  if (parseArguments(argc, argv) != 0) {
    return EXIT_FAILURE;
  }
  print_env();
  float *A;
  cudaMalloc(&A, sizeof(*A) * n * n);
  slowGenerateUniformPositiveDefiniteMatrix(A, n);

  float *twork;
  cudaMalloc(&twork, sizeof(float) * n * n);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float sone = 1.0;
  float snegone = -1.0;
  float szero = 0.0;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &sone, A, n, A, n,
              &szero, twork, n);

  cudaMemcpy(A, twork, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);

  float normA = snorm(n, n, A);

  float *work;
  cudaMalloc(&work, sizeof(*work) * 128 * 128);

  __half *hwork;
  cudaMalloc(&hwork, sizeof(*hwork) * n / 2 * n);

  // printMatrixDeviceBlock("AA.csv", n, n, A, n);

  printf("n = %d\n", n);
  startTimer();
  later_rpotrf('l', n, A, n, work, hwork);
  auto ms = stopTimer();
  // printMatrixDeviceBlock("LL.csv", n, n, A, n);

  if (checkFlag) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &snegone, A, n, A, n,
                &sone, twork, n);
    printf("Backward error ||LL^T-A||/||A|| = %.6e\n",
           snorm(n, n, twork) / normA);
  }

  cudaFree(A);
  cudaFree(twork);
  cudaFree(work);
  cudaFree(hwork);
}