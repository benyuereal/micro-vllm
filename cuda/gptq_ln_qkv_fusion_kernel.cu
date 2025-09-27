#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <math.h>

// vLLM风格的分块大小和常量
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// 高性能LayerNorm实现：向量化计算
__forceinline__ __device__ void compute_layernorm_stats(
    const half* input, int seq_idx, int hidden_dim, 
    float& mean, float& var, float eps) {
    
    // 向量化计算均值和方差
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // 使用half2向量化，提高计算效率
    const half2* input2 = (const half2*)(input + seq_idx * hidden_dim);
    int half_dim = hidden_dim / 2;
    
    for (int i = 0; i < half_dim; i++) {
        half2 val2 = input2[i];
        float val1 = __half2float(__low2half(val2));
        float val2_f = __half2float(__high2half(val2));
        
        sum += val1 + val2_f;
        sum_sq += val1 * val1 + val2_f * val2_f;
    }
    
    // 处理奇数维度
    if (hidden_dim % 2 == 1) {
        float val = __half2float(input[seq_idx * hidden_dim + hidden_dim - 1]);
        sum += val;
        sum_sq += val * val;
    }
    
    mean = sum / hidden_dim;
    var = (sum_sq / hidden_dim) - (mean * mean);
    var = fmaxf(var, eps);
}

// vLLM风格的向量化点积
__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr,
                                           const float g_result,
                                           const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

// 完整的GPTQ 4bit解量化（参考vLLM实现）
__forceinline__ __device__ void dequant_4bit_8_gptq(
    uint32_t qw, uint32_t qz, half2 (&dq)[4], const float scale) {
  // 提取4bit值并反量化
  for (int i = 0; i < 4; i++) {
    int w = (qw >> (i * 8)) & 0xFF;
    int z = (qz >> (i * 8)) & 0xFF;
    
    // GPTQ解量化：weight = (qweight - qzeros) * scale
    half2 w01 = __halves2half2(
        __int2half_rn((w & 0xF) - (z & 0xF)), 
        __int2half_rn((w >> 4) - (z >> 4))
    );
    
    // 应用缩放
    dq[i] = __hmul2(w01, __float2half2_rn(scale));
  }
}

// 优化的融合LayerNorm + GPTQ QKV投影内核
template <int m_count>
__global__ void fused_ln_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size * seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight,     // GPTQ量化权重 [hidden_dim//8, hidden_dim*3]
    const uint32_t* __restrict__ qzeros,      // GPTQ量化零点 [num_groups, groupsize//8]
    const half* __restrict__ scales,          // GPTQ量化缩放 [num_groups, hidden_dim*3]
    const half* __restrict__ ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* __restrict__ ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* __restrict__ qkv_output,            // QKV输出 [batch_size * seq_len, hidden_dim * 3]
    const int batch_size, const int seq_len, const int hidden_dim,
    const int groupsize, const float eps) {
  
  auto t = threadIdx.x;
  
  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, hidden_dim);
  
  int n = offset_n + t * 4;
  
  // 共享内存用于LayerNorm计算
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  __shared__ float shared_mean[m_count];
  __shared__ float shared_var[m_count];
  
  // 1. 加载输入数据到共享内存
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      // 输入是3D张量 [batch_size, seq_len, hidden_dim]，需要正确计算索引
      int seq_idx = offset_m + m;
      const half* input_ptr = input + seq_idx * hidden_dim;
      block_a[m][t] = input_ptr[offset_k + t];
    }
  }
  
  __syncthreads();
  
  // 2. 计算LayerNorm的均值和方差（优化：协作计算）
  if (t == 0) {
    // 只有线程0计算统计信息，避免重复计算
    for (int m = 0; m < m_count; ++m) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        compute_layernorm_stats(
            input, seq_idx, hidden_dim, 
            shared_mean[m], shared_var[m], eps
        );
      }
    }
  }
  
  __syncthreads();
  
  // 3. 应用LayerNorm归一化（优化：向量化 + 减少分支）
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      if (offset_m + m < batch_size * seq_len) {
        // 预计算sqrt，避免重复计算
        float sqrt_var = sqrtf(shared_var[m]);
        
        // 向量化归一化计算
        float val = __half2float(block_a[m][t]);
        float normalized = (val - shared_mean[m]) / sqrt_var;
        
        // 应用LayerNorm权重和偏置
        normalized = normalized * __half2float(ln_weight[offset_k + t]) + 
                     __half2float(ln_bias[offset_k + t]);
        
        block_a[m][t] = __float2half(normalized);
      }
    }
  }
  
  __syncthreads();
  
  // 4. 零初始化输出
  if (n >= hidden_dim * 3) return;  // QKV总维度
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++) {
      int output_idx = (offset_m + m) * hidden_dim * 3 + n;
      if (output_idx < batch_size * seq_len * hidden_dim * 3) {
        *((uint64_t*)(qkv_output + output_idx)) = 0;
      }
    }
  }
  
  __syncthreads();
  
  // 5. GPTQ解量化和矩阵乘法（参考vLLM实现）
  int num_groups = hidden_dim / groupsize; // 计算总组数
  int group = offset_k / groupsize; // 当前组
  int nextgroup_k_boundary = (group + 1) * groupsize; // 下一组的K边界
  
  // a, b offset
  int qk_offset_in_qweight = offset_k / (32 / 4); // 32 elements per uint32_t, 4 bits per element
  
  const uint32_t* b_ptr = qweight + qk_offset_in_qweight * (hidden_dim * 3) + n;
  const uint32_t* z_ptr_base = qzeros + group * (groupsize / 8); // Base pointer for qzeros of current group
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;
  
  // Initial group scales
  float scales_qkv[4];
  for (int i = 0; i < 4; i++) {
    scales_qkv[i] = __half2float(scales[group * (hidden_dim * 3) + n + i]);
  }
  
  // Column result
  float block_c[m_count][4] = {};
  
  // Dequantize and multiply（优化：减少分支）
  int k = offset_k;
  while (k < end_k) {
    // 检查是否需要切换到下一组（优化：减少分支预测失败）
    if (k >= nextgroup_k_boundary) {
      group++;
      if (group >= num_groups) break;  // 边界检查
      nextgroup_k_boundary = (group + 1) * groupsize;
      z_ptr_base = qzeros + group * (groupsize / 8);
      // 预加载下一组的scales
      for (int i = 0; i < 4; i++) {
        scales_qkv[i] = __half2float(scales[group * (hidden_dim * 3) + n + i]);
      }
    }
    
    const uint32_t* current_z_ptr = z_ptr_base + (k % groupsize) / 8;
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (const int4*)b_ptr;
      const int4* z_ptr4 = (const int4*)current_z_ptr;
      int4 load_int4 = *b_ptr4;
      int4 load_zeros = *z_ptr4;
      
      half2 dq[4][4];
      // 使用完整的GPTQ解量化（包含qzeros处理）
      dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_qkv[0]);
      dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_qkv[1]);
      dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_qkv[2]);
      dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_qkv[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c[m][0], 1.0f);
        block_c[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c[m][1], 1.0f);
        block_c[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c[m][2], 1.0f);
        block_c[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c[m][3], 1.0f);
      }
      
      b_ptr += (hidden_dim * 3); // Move to next column in qweight
      current_z_ptr += (groupsize / 8); // Move to next qzeros block
      a_ptr += 8; // Move to next 8 elements in input
    }
    
    k += 32; // Process 32 elements in K dimension
  }
  
  // 6. 输出QKV到正确的格式
  if (n < hidden_dim * 3) {  // 确保不超出QKV总维度
    for (int m = 0; m < m_count; m++) {
      int output_base = (offset_m + m) * hidden_dim * 3 + n;
      
      if (output_base < batch_size * seq_len * hidden_dim * 3) {
        // 直接写入（每个线程处理不同的输出位置，不需要atomicAdd）
        half* qkv_out = qkv_output + output_base;
        qkv_out[0] = __float2half_rn(block_c[m][0]);
        qkv_out[1] = __float2half_rn(block_c[m][1]);
        qkv_out[2] = __float2half_rn(block_c[m][2]);
        qkv_out[3] = __float2half_rn(block_c[m][3]);
      }
    }
  }
}

// PyTorch包装函数
torch::Tensor fused_ln_qkv_gptq_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int groupsize,
    float eps
) {
    // 检查输入张量
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor [batch_size, seq_len, hidden_dim]");
    
    // 验证输入形状
    TORCH_CHECK(input.size(0) == batch_size, "input batch_size mismatch");
    TORCH_CHECK(input.size(1) == seq_len, "input seq_len mismatch");
    TORCH_CHECK(input.size(2) == hidden_dim, "input hidden_dim mismatch");
    
    // 验证GPTQ参数
    TORCH_CHECK(hidden_dim % groupsize == 0, "hidden_dim must be divisible by groupsize");
    TORCH_CHECK(groupsize > 0, "groupsize must be positive");
    TORCH_CHECK(eps > 0.0f, "eps must be positive");
    
    // 确保张量是连续的
    input = input.contiguous();
    qweight = qweight.contiguous();
    qzeros = qzeros.contiguous();
    scales = scales.contiguous();
    ln_weight = ln_weight.contiguous();
    ln_bias = ln_bias.contiguous();
    
    // 将3D输入重塑为2D [batch_size*seq_len, hidden_dim]
    auto input_2d = input.view({batch_size * seq_len, hidden_dim});
    
    // 获取张量数据指针（使用vLLM相同的方法）
    const half* input_ptr = reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>());
    const uint32_t* qweight_ptr = qweight.data_ptr<uint32_t>();
    const uint32_t* qzeros_ptr = qzeros.data_ptr<uint32_t>();
    const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    const half* ln_weight_ptr = reinterpret_cast<const half*>(ln_weight.data_ptr<at::Half>());
    const half* ln_bias_ptr = reinterpret_cast<const half*>(ln_bias.data_ptr<at::Half>());
    
    // 创建输出张量 [batch_size * seq_len, hidden_dim * 3]
    auto qkv_output = torch::zeros({batch_size * seq_len, hidden_dim * 3}, 
                                  torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    
    half* qkv_output_ptr = reinterpret_cast<half*>(qkv_output.data_ptr<at::Half>());
    
    // 调用原始C函数
    {
        // 优化的网格和块大小（基于vLLM）
        dim3 blockDim, gridDim;
        blockDim.x = BLOCK_KN_SIZE;  // 128 threads per block
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(hidden_dim * 3, BLOCK_KN_SIZE * 4);  // QKV总维度
        gridDim.y = DIVIDE(batch_size * seq_len, BLOCK_M_SIZE_MAX);  // 序列维度
        gridDim.z = DIVIDE(hidden_dim, BLOCK_KN_SIZE);  // 隐藏维度
        
        // 添加网格大小验证
        TORCH_CHECK(gridDim.x > 0 && gridDim.y > 0 && gridDim.z > 0, 
                   "Invalid grid dimensions");
        TORCH_CHECK(gridDim.x <= 65535 && gridDim.y <= 65535 && gridDim.z <= 65535,
                   "Grid dimensions exceed CUDA limits");
        
        // 启动融合内核（使用CUDA流，与vLLM一致）
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        fused_ln_qkv_gptq_kernel<1><<<gridDim, blockDim, 0, stream>>>(
            input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, ln_weight_ptr, ln_bias_ptr,
            qkv_output_ptr, batch_size, seq_len, hidden_dim, groupsize, eps
        );
        
        cudaDeviceSynchronize();
    }
    
    // 分割QKV并重塑为正确的格式
    auto q = qkv_output.slice(1, 0, hidden_dim).view({batch_size, seq_len, hidden_dim});
    auto k = qkv_output.slice(1, hidden_dim, hidden_dim*2).view({batch_size, seq_len, hidden_dim});
    auto v = qkv_output.slice(1, hidden_dim*2).view({batch_size, seq_len, hidden_dim});
    
    // 返回QKV张量的元组
    return torch::stack({q, k, v}, 0);
}

// PyTorch模块导出函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_qkv_gptq_cuda", &fused_ln_qkv_gptq_cuda, "Fused LayerNorm + QKV GPTQ CUDA Kernel");
}