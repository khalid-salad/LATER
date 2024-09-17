#include "LATER.h"

int n;
bool checkFlag = false;

int parseArguments(int argc, char* argv[]) {
    n = atoi(argv[1]);
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: ./test_potrf n [options]\n");
        printf("Options:\n\t-check: enable checking the orthogonality and backward "
               "error\n");
        return EXIT_FAILURE;
    }
    if (parseArguments(argc, argv) != 0) {
        return EXIT_FAILURE;
    }
    auto num_bytes = sizeof(float) * n * n;
    float* hA = uniformPositiveDefiniteMatrix(n);
    float* dA;
    cudaMalloc(&dA, num_bytes);
    cudaMemcpy(dA, hA, num_bytes, cudaMemcpyHostToDevice);

    float* work;
    __half* hwork;
    cudaMalloc(&work, num_bytes);
    cudaMalloc(&hwork, num_bytes);

    startTimer();
    later_rpotrf('l', n, dA, n, work, hwork);
    auto ms = stopTimer();

    if (checkFlag) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &snegone, dA, n, dA, n, &sone, twork, n);
        printf("Backward error ||LL^T-A||/||A|| = %.6e\n", snorm(n, n, twork) / normA);
    }
    std::cout << ms << std::endl;
    cudaFree(dA);
    cudaFree(work);
    cudaFree(hwork);
}