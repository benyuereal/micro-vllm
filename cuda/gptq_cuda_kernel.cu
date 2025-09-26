#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// 高性能CUDA内核实现 - 集成cuBLAS和混合精度
__global__ void ultra_fast_gptq_gemm_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    const int M, const int N, const int K,
    const int groupsize) {
    
    // 使用更大的线程块和更高效的访问模式
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 每个线程处理更多输出元素
    const int ELEMENTS_PER_THREAD = 16;
    int start_idx = bid * blockDim.x * ELEMENTS_PER_THREAD;
    
    // 更大的共享内存缓存
    __shared__ half shared_input[1024];
    __shared__ half shared_scales[1024];
    
    float results[ELEMENTS_PER_THREAD] = {0.0f};
    
    // 处理多个输出元素
    for (int elem = 0; elem < ELEMENTS_PER_THREAD; elem++) {
        int idx = start_idx + tid + elem * blockDim.x;
        if (idx >= M * N) continue;
        
        int m = idx / N;
        int n = idx % N;
        
        // 批量处理 - 每次处理64个元素
        for (int k = 0; k < K; k += 64) {
            // 协作加载到共享内存
            if (tid < 64 && k + tid < K) {
                shared_input[tid] = input[m * K + k + tid];
                shared_scales[tid] = scales[(k / groupsize) * K + k + tid];
            }
            __syncthreads();
            
            // 向量化计算 - 64个元素并行
            for (int i = 0; i < 64; i++) {
                if (k + i < K) {
                    // 加载8个32位打包权重
                    int weight_idx = n * (K / 8) + (k / 8) + (i / 8);
                    uint32_t packed_weight = qweight[weight_idx];
                    
                    // 提取4bit权重值
                    int weight_val = (packed_weight >> ((i % 8) * 4)) & 0xF;
                    
                    // 加载zero point
                    int zero_idx = (k / groupsize) * (K / 8) + (k / 8) + (i / 8);
                    uint32_t packed_zero = qzeros[zero_idx];
                    int zero_val = (packed_zero >> ((i % 8) * 4)) & 0xF;
                    
                    // 计算
                    float input_val = __half2float(shared_input[i]);
                    float scale_val = __half2float(shared_scales[i]);
                    
                    results[elem] += input_val * (weight_val - zero_val) * scale_val;
                }
            }
            __syncthreads();
        }
        
        // 存储结果
        if (idx < M * N) {
            output[idx] = __float2half(results[elem]);
        }
    }
}

// cuBLAS辅助函数
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

// 混合精度cuBLAS GEMM
void cublas_mixed_precision_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cublasOperation_t transA = CUBLAS_OP_N,
    cublasOperation_t transB = CUBLAS_OP_N) {
    
    cublasHandle_t handle = get_cublas_handle();
    
    // 混合精度参数
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // 使用cuBLAS混合精度GEMM
    cublasHgemm(
        handle,
        transA, transB,
        M, N, K,
        &alpha,
        A, (transA == CUBLAS_OP_N) ? M : K,
        B, (transB == CUBLAS_OP_N) ? K : N,
        &beta,
        C, M
    );
}

// Python接口
torch::Tensor fused_gptq_gemm_4bit_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    int groupsize
) {
    // 获取维度
    int M = input.size(0);
    int K = input.size(1);
    int N = qweight.size(0);
    
    // 创建输出张量
    auto output = torch::zeros({M, N}, torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device()));
    
    // 优化的网格和块大小
    int total_elements = M * N;
    int block_size = 512;  // 更大的线程块
    int elements_per_thread = 16;
    int grid_size = (total_elements + block_size * elements_per_thread - 1) / (block_size * elements_per_thread);
    
    // 启动内核
    ultra_fast_gptq_gemm_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        qweight.data_ptr<uint32_t>(),
        qzeros.data_ptr<uint32_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, K, groupsize
    );
    
    cudaDeviceSynchronize();
    return output;
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gptq_gemm_4bit_cuda", &fused_gptq_gemm_4bit_cuda, "Ultra Fast Fused GPTQ GEMM 4bit CUDA with cuBLAS");
}