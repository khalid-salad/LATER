#include "OC_gemm.h"

#define cudaChk(stat)                                                          \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
    exit(1);
  }
}
const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}
#define cublasChk(stat)                                                        \
  { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %s %s %d\n", cublasGetErrorString(stat),
            file, line);
    exit(1);
  }
}

size_t free_mem() {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free;
}

int is_mul_overflow(int a, int b) {
  if (a >= 0 && b >= 0) {
    return INT_MAX / a < b;
  } else if (a < 0 && b < 0) {
    return INT_MAX / a > b;
  } else if (a * b == INT_MIN) {
    return 0;
  } else {
    return a < 0 ? is_mul_overflow(-a, b) : is_mul_overflow(a, -b);
  }
}

void OC_gemm::tile_size() {
  auto free_mem_stream = free_mem() / num_stream;
  int i = 1;
  do {
    tm = M / i;
    tn = N / i;
    tk = K / i;
    i *= 2;
    if (is_mul_overflow(tk, sizeof(float)) ||
        is_mul_overflow(tn, sizeof(float)) ||
        is_mul_overflow(tm, tn * sizeof(float)) ||
        is_mul_overflow(tm, tk * sizeof(float)) ||
        is_mul_overflow(tn, tk * sizeof(float)))
      continue;
    dev_mem_per_stream =
        size_t(tm * tn * sizeof(float)) + size_t(tm * tk * sizeof(half)) +
        size_t(tn * tk * sizeof(half)) + size_t(tm * tk * sizeof(float)) +
        size_t(tn * tk * sizeof(float));
    if (dev_mem_per_stream < free_mem_stream)
      break;
  } while (true);
}

OC_gemm::OC_gemm(int _M, int _N, int _K, int _num_stream)
    : M(_M), N(_N), K(_K), num_stream(_num_stream) {
  streams = new cudaStream_t[num_stream];
  handles = new cublasHandle_t[num_stream];
  A_tiles = new half *[num_stream];
  B_tiles = new half *[num_stream];
  C_tiles = new float *[num_stream];
  fA_tiles = new float *[num_stream];
  fB_tiles = new float *[num_stream];
  for (int i = 0; i < num_stream; i++) {
    cudaChk(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    cublasChk(cublasCreate(&handles[i]));
    cublasChk(cublasSetStream(handles[i], streams[i]));
    cublasChk(cublasSetMathMode(handles[i], CUBLAS_TENSOR_OP_MATH));
  }
  tile_size();
  for (int i = 0; i < num_stream; i++) {
    cudaChk(cudaMalloc((void **)&A_tiles[i], tm * tk * sizeof(half)));
    cudaChk(cudaMalloc((void **)&B_tiles[i], tn * tk * sizeof(half)));
    cudaChk(cudaMalloc((void **)&C_tiles[i], tn * tm * sizeof(float)));
    cudaChk(cudaMalloc((void **)&fA_tiles[i], tm * tk * sizeof(float)));
    cudaChk(cudaMalloc((void **)&fB_tiles[i], tn * tk * sizeof(float)));
  }
}

OC_gemm::~OC_gemm() {
  for (int i = 0; i < num_stream; i++) {
    cudaChk(cudaStreamDestroy(streams[i]));
    cublasChk(cublasDestroy(handles[i]));
  }
  delete[] streams;
  delete[] handles;
  for (int i = 0; i < num_stream; i++) {
    cudaChk(cudaFree(A_tiles[i]));
    cudaChk(cudaFree(B_tiles[i]));
    cudaChk(cudaFree(C_tiles[i]));
    cudaChk(cudaFree(fA_tiles[i]));
    cudaChk(cudaFree(fB_tiles[i]));
  }
  delete[] A_tiles;
  delete[] B_tiles;
  delete[] fA_tiles;
  delete[] fB_tiles;
  delete[] C_tiles;
}

void OC_gemm::gemm(cublasOperation_t transa, cublasOperation_t transb,
                   const float &alpha, const half *A, int lda, const half *B,
                   int ldb, const float &beta, float *C, int ldc) {
  for (size_t i = 0; i < (M / tm); i++) {
    for (size_t j = 0; j < (N / tn); j++) {
      const auto stream_id = (i * N / tn + j) % num_stream;
      auto stream = streams[stream_id];
      auto pC = &C[j * tn * M + i * tm];
      for (int a = 0; a < (K / tk); a++) {
        int tlda, tldb;
        const half *pA, *pB;
        if (transa == CUBLAS_OP_N) {
          pA = &A[a * tk * M + i * tm];
          cublasChk(cublasSetMatrixAsync(tm, tk, sizeof(half), pA, M,
                                         A_tiles[stream_id], tm, stream));
          tlda = tm;
        } else {
          pA = &A[i * tk * M + a * tm];
          cublasChk(cublasSetMatrixAsync(tk, tm, sizeof(half), pA, K,
                                         A_tiles[stream_id], tk, stream));
          tlda = tk;
        }
        if (transb == CUBLAS_OP_N) {
          pB = &B[j * tn * K + a * tk];
          cublasChk(cublasSetMatrixAsync(tk, tn, sizeof(half), pB, K,
                                         B_tiles[stream_id], tk, stream));
          tldb = tk;
        } else {
          pB = &B[a * tn * K + j * tk];
          cublasChk(cublasSetMatrixAsync(tn, tk, sizeof(half), pB, N,
                                         B_tiles[stream_id], tn, stream));
          tldb = tn;
        }
        cublasChk(cublasGemmEx(handles[stream_id], transa, transb, tm, tn, tk,
                               &alpha, A_tiles[stream_id], CUDA_R_16F, tlda,
                               B_tiles[stream_id], CUDA_R_16F, tldb, &beta,
                               C_tiles[stream_id], CUDA_R_32F, tm, CUDA_R_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      cublasChk(cublasGetMatrixAsync(tm, tn, sizeof(float), C_tiles[stream_id],
                                     tm, pC, M, stream));
      cudaChk(cudaMemsetAsync(C_tiles[stream_id], 0, tm * tn * sizeof(float),
                              stream));
    }
  }
  cudaChk(cudaDeviceSynchronize());
}

__global__ void AB2half(half *__restrict__ hA, half *__restrict__ hB,
                        const float *__restrict__ fA,
                        const float *__restrict__ fB, const int size_A,
                        const int size_B) {
  const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (auto i = tid; i < size_A; i += gridDim.x * blockDim.x)
    hA[i] = __float2half(fA[i]);
  for (auto i = tid; i < size_B; i += gridDim.x * blockDim.x)
    hB[i] = __float2half(fB[i]);
}

void OC_gemm::gemm(cublasOperation_t transa, cublasOperation_t transb,
                   const float &alpha, const float *A, int lda, const float *B,
                   int ldb, const float &beta, float *C, int ldc) {
  for (size_t i = 0; i < (M / tm); i++) {
    for (size_t j = 0; j < (N / tn); j++) {
      const auto stream_id = (i * N / tn + j) % num_stream;
      auto stream = streams[stream_id];
      auto pC = &C[j * tn * M + i * tm];
      for (int a = 0; a < (K / tk); a++) {
        int tlda, tldb;
        const float *pA, *pB;
        if (transa == CUBLAS_OP_N) {
          pA = &A[a * tk * M + i * tm];
          cublasChk(cublasSetMatrixAsync(tm, tk, sizeof(float), pA, M,
                                         fA_tiles[stream_id], tm, stream));
          tlda = tm;
        } else {
          pA = &A[i * tk * M + a * tm];
          cublasChk(cublasSetMatrixAsync(tk, tm, sizeof(float), pA, K,
                                         fA_tiles[stream_id], tk, stream));
          tlda = tk;
        }
        if (transb == CUBLAS_OP_N) {
          pB = &B[j * tn * K + a * tk];
          cublasChk(cublasSetMatrixAsync(tk, tn, sizeof(float), pB, K,
                                         fB_tiles[stream_id], tk, stream));
          tldb = tk;
        } else {
          pB = &B[a * tn * K + j * tk];
          cublasChk(cublasSetMatrixAsync(tn, tk, sizeof(float), pB, N,
                                         fB_tiles[stream_id], tn, stream));
          tldb = tn;
        }
        int size_A = tm * tk;
        int size_B = tn * tk;
        AB2half<<<(std::max(size_A, size_B) / 1024) + 1, 1024, 0, stream>>>(
            A_tiles[stream_id], B_tiles[stream_id], fA_tiles[stream_id],
            fB_tiles[stream_id], size_A, size_B);
        cublasChk(cublasGemmEx(handles[stream_id], transa, transb, tm, tn, tk,
                               &alpha, A_tiles[stream_id], CUDA_R_16F, tlda,
                               B_tiles[stream_id], CUDA_R_16F, tldb, &beta,
                               C_tiles[stream_id], CUDA_R_32F, tm, CUDA_R_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      cublasChk(cublasGetMatrixAsync(tm, tn, sizeof(float), C_tiles[stream_id],
                                     tm, pC, M, stream));
      cudaChk(cudaMemsetAsync(C_tiles[stream_id], 0, tm * tn * sizeof(float),
                              stream));
    }
  }
  cudaChk(cudaDeviceSynchronize());
}

#ifdef TEST_OC

template <typename T> void prt(T *arr, int size) {
  for (int i = 0; i < size; i++)
    std::cout << arr[i] << ", ";
  puts("");
}
int main(int ac, char **av) {
  cudaChk(cudaFree(0));
  if (ac < 4)
    puts("Usage: ./a.out m n k [used GPU mem%]");
  int m = atoi(av[1]);
  int n = atoi(av[2]);
  int k = atoi(av[3]);
  if (ac > 4) {
    size_t mem = free_mem() * (atof(av[4]) / 100);
    volatile char *p;
    cudaChk(cudaMalloc((void **)&p, mem));
  }
  const size_t elements = size_t(m) * size_t(k) + size_t(n) * size_t(k);
  std::vector<half> h_data(elements);
  std::vector<float> f_data(elements);
  std::uniform_real_distribution<float> distribution(0.0f, 2.0f);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  std::generate_n(f_data.begin(), elements, generator);
#pragma omp parallel for
  for (size_t i = 0; i < f_data.size(); i++)
    h_data[i] = __float2half(f_data[i]);
  auto hA = h_data.data();
  auto hB = &h_data.data()[size_t(m) * size_t(k)];
  auto fA = f_data.data();
  auto fB = &f_data.data()[size_t(m) * size_t(k)];
  std::vector<float> C_data(size_t(m) * size_t(n), 0.0f);
  auto C = C_data.data();

  float alpha = 1.0f;
  float beta = 1.0f;

  // prt(fA, size_t(m) * size_t(k));
  // prt(fB, size_t(n) * size_t(k));

  OC_gemm OC(m, n, k);
  puts("Created");
  OC.gemm(CUBLAS_OP_N, CUBLAS_OP_N, alpha, hA, m, hB, k, beta, C, m);
  // prt(C, size_t(m) * size_t(n));
  std::cout << C[0] << std::endl;
  OC.gemm(CUBLAS_OP_N, CUBLAS_OP_N, alpha, fA, m, fB, k, beta, C, m);
  // prt(C, size_t(m) * size_t(n));
  std::cout << C[0] << std::endl;
}
#endif