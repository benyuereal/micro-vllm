#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// 高性能CUDA内核实现
__global__ void high_performance_gptq_gemm_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    const int M, const int N, const int K,
    const int groupsize) {
    
    // 使用更大的线程块
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 每个线程块处理多个输出元素
    const int ELEMENTS_PER_BLOCK = 4;
    int start_idx = bid * blockDim.x * ELEMENTS_PER_BLOCK;
    
    // 共享内存缓存
    __shared__ float shared_input[256];  // 缓存输入数据
    __shared__ float shared_scales[256]; // 缓存scale数据
    
    float results[ELEMENTS_PER_BLOCK] = {0.0f};
    
    // 处理多个输出元素
    for (int elem = 0; elem < ELEMENTS_PER_BLOCK; elem++) {
        int idx = start_idx + tid + elem * blockDim.x;
        if (idx >= M * N) continue;
        
        int m = idx / N;
        int n = idx % N;
        
        // 向量化处理 - 每次处理16个元素
        for (int k = 0; k < K; k += 16) {
            // 加载16个输入值
            float input_vals[16];
            for (int i = 0; i < 16; i++) {
                if (k + i < K) {
                    input_vals[i] = __half2float(input[m * K + k + i]);
                } else {
                    input_vals[i] = 0.0f;
                }
            }
            
            // 加载2个32位打包权重
            uint32_t packed_weight1 = qweight[n * (K / 8) + (k / 8)];
            uint32_t packed_weight2 = qweight[n * (K / 8) + (k / 8) + 1];
            
            // 加载2个32位打包zero points
            uint32_t packed_zero1 = qzeros[(k / groupsize) * (K / 8) + (k / 8)];
            uint32_t packed_zero2 = qzeros[(k / groupsize) * (K / 8) + (k / 8) + 1];
            
            // 加载16个scale值
            float scale_vals[16];
            for (int i = 0; i < 16; i++) {
                if (k + i < K) {
                    scale_vals[i] = __half2float(scales[(k / groupsize) * K + k + i]);
                } else {
                    scale_vals[i] = 0.0f;
                }
            }
            
            // 向量化计算 - 16个元素并行
            for (int i = 0; i < 16; i++) {
                if (k + i < K) {
                    int weight_val, zero_val;
                    if (i < 8) {
                        weight_val = (packed_weight1 >> (i * 4)) & 0xF;
                        zero_val = (packed_zero1 >> (i * 4)) & 0xF;
                    } else {
                        weight_val = (packed_weight2 >> ((i - 8) * 4)) & 0xF;
                        zero_val = (packed_zero2 >> ((i - 8) * 4)) & 0xF;
                    }
                    
                    results[elem] += input_vals[i] * (weight_val - zero_val) * scale_vals[i];
                }
            }
        }
        
        // 存储结果
        if (idx < M * N) {
            output[idx] = __float2half(results[elem]);
        }
    }
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
    int block_size = 256;  // 更大的线程块
    int elements_per_block = 4;
    int grid_size = (total_elements + block_size * elements_per_block - 1) / (block_size * elements_per_block);
    
    // 启动内核
    high_performance_gptq_gemm_kernel<<<grid_size, block_size>>>(
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
    m.def("fused_gptq_gemm_4bit_cuda", &fused_gptq_gemm_4bit_cuda, "High Performance Fused GPTQ GEMM 4bit CUDA");
}
