#include "LATER.h"

int n;
bool checkFlag = false;

#define BLOCKSIZE 2048
#define LWORK 65536

int parseArguments(int argc, char* argv[]) {
    n = atoi(argv[1]);
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

/**
 * returns the wall clock time for cublas cholesky decomp and a pointer to the
 * device matrix dA_cubls, the cholesky decomp of dA. Matrix dA is not modified
 */
std::pair<float, float*> cublas_wall_clock_time(char uplo, int n, float* dA, float* work) {
    // create device array dA_cubls and copy dA to it
    auto num_bytes = sizeof(float) * n * n;
    float* dA_cubls;
    cudaMalloc(&dA_cubls, num_bytes);
    cudaMemcpy(dA_cubls, dA, num_bytes, cudaMemcpyDeviceToDevice);

    // cublas cholesky decomp on dA_cubls
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    int* dev_info;
    cudaMalloc(&dev_info, sizeof(*dev_info));
    auto fill_mode = uplo == 'l' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    startTimer();
    cusolverDnSpotrf(handle, fill_mode, n, dA_cubls, n, work, LWORK, dev_info);
    auto ms = stopTimer();
    return {ms, dA_cubls};
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: ./test_potrf n [options]\n");
        printf("Options:\n\t--check: enable checking the orthogonality and backward "
               "error\n");
        return EXIT_FAILURE;
    }
    if (parseArguments(argc, argv) != 0) {
        return EXIT_FAILURE;
    }
    auto num_bytes = sizeof(float) * n * n;
    float* dA = uniformPositiveDefiniteMatrix(n);
    float* work;
    __half* hwork;
    cudaMalloc(&work, num_bytes);
    cudaMalloc(&hwork, num_bytes);

    auto [ms_cubls, dA_cubls] = cublas_wall_clock_time('l', n, dA, work);
    auto dA_later = dA;

    startTimer();
    later_rpotrf('l', n, dA_later, n, work, hwork);
    auto ms_later = stopTimer();

    if (checkFlag) {
        float snegone = -1, sone = 1;

        printf("Backward error ||LL^T-A||/||A|| = %.6e\n", snorm(n, n, twork) / normA);
    }

    std::cout << ms_later << std::endl;
    cudaFree(dA_later);
    cudaFree(dA_cubls);
    cudaFree(work);
    cudaFree(hwork);
}