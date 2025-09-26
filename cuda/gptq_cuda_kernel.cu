#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 向量化的CUDA内核实现 - 支持BFloat16
__global__ void vectorized_gptq_gemm_kernel(
    const half* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    const int M, const int N, const int K,
    const int groupsize) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (idx >= total_elements) return;
    
    int m = idx / N;
    int n = idx % N;
    
    float result = 0.0f;
    
    // 向量化处理 - 每次处理8个元素
    for (int k = 0; k < K; k += 8) {
        // 加载8个输入值到寄存器
        float2 input_vals = __half22float2(*((half2*)(input + m * K + k)));
        float2 input_vals2 = __half22float2(*((half2*)(input + m * K + k + 2)));
        float2 input_vals3 = __half22float2(*((half2*)(input + m * K + k + 4)));
        float2 input_vals4 = __half22float2(*((half2*)(input + m * K + k + 6)));
        
        // 加载8个量化权重
        uint32_t packed_weight = qweight[n * (K / 8) + (k / 8)];
        
        // 提取8个4bit权重值
        int weight_vals[8];
        for (int i = 0; i < 8; i++) {
            weight_vals[i] = (packed_weight >> (i * 4)) & 0xF;
        }
        
        // 加载8个zero points
        uint32_t packed_zero = qzeros[(k / groupsize) * (K / 8) + (k / 8)];
        int zero_vals[8];
        for (int i = 0; i < 8; i++) {
            zero_vals[i] = (packed_zero >> (i * 4)) & 0xF;
        }
        
        // 加载8个scale值
        float2 scale_vals = __half22float2(*((half2*)(scales + (k / groupsize) * K + k)));
        float2 scale_vals2 = __half22float2(*((half2*)(scales + (k / groupsize) * K + k + 2)));
        float2 scale_vals3 = __half22float2(*((half2*)(scales + (k / groupsize) * K + k + 4)));
        float2 scale_vals4 = __half22float2(*((half2*)(scales + (k / groupsize) * K + k + 6)));
        
        // 向量化计算
        result += input_vals.x * (weight_vals[0] - zero_vals[0]) * scale_vals.x;
        result += input_vals.y * (weight_vals[1] - zero_vals[1]) * scale_vals.y;
        result += input_vals2.x * (weight_vals[2] - zero_vals[2]) * scale_vals2.x;
        result += input_vals2.y * (weight_vals[3] - zero_vals[3]) * scale_vals2.y;
        result += input_vals3.x * (weight_vals[4] - zero_vals[4]) * scale_vals3.x;
        result += input_vals3.y * (weight_vals[5] - zero_vals[5]) * scale_vals3.y;
        result += input_vals4.x * (weight_vals[6] - zero_vals[6]) * scale_vals4.x;
        result += input_vals4.y * (weight_vals[7] - zero_vals[7]) * scale_vals4.y;
    }
    
    // 存储结果
    output[idx] = __float2half(result);
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
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // 启动内核
    vectorized_gptq_gemm_kernel<<<grid_size, block_size>>>(
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
    m.def("fused_gptq_gemm_4bit_cuda", &fused_gptq_gemm_4bit_cuda, "Vectorized Fused GPTQ GEMM 4bit CUDA");
}