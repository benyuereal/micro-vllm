#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// 常量定义
#define BLOCK_KN_SIZE 32
#define BLOCK_M_SIZE_MAX 16
#define DIVIDE(x, y) ((x + y - 1) / y)

// RMSNorm统计计算（只计算方差，不计算均值）
__forceinline__ __device__ void compute_rmsnorm_stats(
    const half* input, int seq_idx, int hidden_dim, 
    float& rms, float eps) {
    
    float sum_sq = 0.0f;
    
    // 使用half2向量化，提高计算效率
    const half2* input2 = (const half2*)(input + seq_idx * hidden_dim);
    int half_dim = hidden_dim / 2;
    
    for (int i = 0; i < half_dim; i++) {
        half2 val2 = input2[i];
        float val1 = __half2float(__low2half(val2));
        float val2_f = __half2float(__high2half(val2));
        
        sum_sq += val1 * val1 + val2_f * val2_f;
    }
    
    // 处理奇数维度
    if (hidden_dim % 2 == 1) {
        float val = __half2float(input[seq_idx * hidden_dim + hidden_dim - 1]);
        sum_sq += val * val;
    }
    
    // RMSNorm: rms = sqrt(mean(x^2) + eps)
    float mean_sq = sum_sq / hidden_dim;
    rms = sqrtf(mean_sq + eps);
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

// 优化的融合内核：RMSNorm + GPTQ QKV投影（完整vLLM优化版本）
template <int m_count>
__global__ void fused_rmsnorm_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size * seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight_q,   // Q权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qweight_k,   // K权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qweight_v,   // V权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qzeros_q,    // Q零点 [num_groups, groupsize//8]
    const uint32_t* __restrict__ qzeros_k,    // K零点 [num_groups, groupsize//8]
    const uint32_t* __restrict__ qzeros_v,    // V零点 [num_groups, groupsize//8]
    const half* __restrict__ scales_q,        // Q缩放 [num_groups, hidden_dim]
    const half* __restrict__ scales_k,        // K缩放 [num_groups, hidden_dim]
    const half* __restrict__ scales_v,        // V缩放 [num_groups, hidden_dim]
    const half* __restrict__ rms_weight,      // RMSNorm权重 [hidden_dim]
    half* __restrict__ q_output,              // Q输出 [batch_size * seq_len, hidden_dim]
    half* __restrict__ k_output,              // K输出 [batch_size * seq_len, hidden_dim]
    half* __restrict__ v_output,              // V输出 [batch_size * seq_len, hidden_dim]
    const int batch_size, const int seq_len, const int hidden_dim,
    const int groupsize, const float eps) {
  
  auto t = threadIdx.x;
  
  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, hidden_dim);
  int n = offset_n + t * 4;
  
  // 共享内存用于RMSNorm计算
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  __shared__ float shared_rms[m_count];
  
  // 1. 计算RMSNorm的RMS（全局计算）
  if (t == 0) {
    for (int m = 0; m < m_count; ++m) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        compute_rmsnorm_stats(
            input, seq_idx, hidden_dim, 
            shared_rms[m], eps
        );
      }
    }
  }
  
  __syncthreads();
  
  // 2. 加载输入数据到共享内存并应用RMSNorm
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        // 预计算1/rms，避免重复计算
        float inv_rms = 1.0f / shared_rms[m];
        
        // 直接应用RMSNorm归一化
        float val = __half2float(input[seq_idx * hidden_dim + offset_k + t]);
        float normalized = val * inv_rms;
        
        // 应用RMSNorm权重
        normalized = normalized * __half2float(rms_weight[offset_k + t]);
        
        block_a[m][t] = __float2half(normalized);
      }
    }
  }
  
  __syncthreads();
  
  // 3. GPTQ QKV投影
  // Q投影
  if (n < hidden_dim) {
    float block_c_q[m_count][4] = {0.0f};
    
    int group = 0;
    int nextgroup_k_boundary = groupsize;
    const uint32_t* z_ptr_base_q = qzeros_q;
    float scales_q_val[4];
    
    for (int k = offset_k; k < end_k; k += 32) {
      const half* a_ptr = &block_a[0][k - offset_k];
      const int a_stride = BLOCK_KN_SIZE;
      const uint32_t* b_ptr_q = &qweight_q[k * hidden_dim + n];
      
      if (k >= nextgroup_k_boundary) {
        group++;
        if (group >= hidden_dim / groupsize) break;
        nextgroup_k_boundary = (group + 1) * groupsize;
        z_ptr_base_q = qzeros_q + group * (groupsize / 8);
        for (int i = 0; i < 4; i++) {
          scales_q_val[i] = __half2float(scales_q[group * hidden_dim + n + i]);
        }
      }
      
      const uint32_t* current_z_ptr = z_ptr_base_q + (k % groupsize) / 8;
      
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int4* b_ptr4 = (const int4*)b_ptr_q;
        const int4* z_ptr4 = (const int4*)current_z_ptr;
        int4 load_int4 = *b_ptr4;
        int4 load_zeros = *z_ptr4;
        
        half2 dq[4][4];
        dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_q_val[0]);
        dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_q_val[1]);
        dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_q_val[2]);
        dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_q_val[3]);
        
#pragma unroll
        for (int m = 0; m < m_count; m++) {
          block_c_q[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_q[m][0], 1.0f);
          block_c_q[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_q[m][1], 1.0f);
          block_c_q[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_q[m][2], 1.0f);
          block_c_q[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_q[m][3], 1.0f);
        }
        
        b_ptr_q += hidden_dim;
        current_z_ptr += (groupsize / 8);
        a_ptr += 8;
      }
      
      k += 32;
    }
    
    // 输出Q结果
    if (n < hidden_dim) {
      for (int m = 0; m < m_count; m++) {
        int seq_idx = offset_m + m;
        if (seq_idx < batch_size * seq_len) {
          q_output[seq_idx * hidden_dim + n] = __float2half(block_c_q[m][0]);
        }
      }
    }
  }
  
  // K投影（类似Q投影）
  if (n < hidden_dim) {
    float block_c_k[m_count][4] = {0.0f};
    
    int group = 0;
    int nextgroup_k_boundary = groupsize;
    const uint32_t* z_ptr_base_k = qzeros_k;
    float scales_k_val[4];
    
    for (int k = offset_k; k < end_k; k += 32) {
      const half* a_ptr = &block_a[0][k - offset_k];
      const int a_stride = BLOCK_KN_SIZE;
      const uint32_t* b_ptr_k = &qweight_k[k * hidden_dim + n];
      
      if (k >= nextgroup_k_boundary) {
        group++;
        if (group >= hidden_dim / groupsize) break;
        nextgroup_k_boundary = (group + 1) * groupsize;
        z_ptr_base_k = qzeros_k + group * (groupsize / 8);
        for (int i = 0; i < 4; i++) {
          scales_k_val[i] = __half2float(scales_k[group * hidden_dim + n + i]);
        }
      }
      
      const uint32_t* current_z_ptr = z_ptr_base_k + (k % groupsize) / 8;
      
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int4* b_ptr4 = (const int4*)b_ptr_k;
        const int4* z_ptr4 = (const int4*)current_z_ptr;
        int4 load_int4 = *b_ptr4;
        int4 load_zeros = *z_ptr4;
        
        half2 dq[4][4];
        dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_k_val[0]);
        dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_k_val[1]);
        dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_k_val[2]);
        dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_k_val[3]);
        
#pragma unroll
        for (int m = 0; m < m_count; m++) {
          block_c_k[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_k[m][0], 1.0f);
          block_c_k[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_k[m][1], 1.0f);
          block_c_k[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_k[m][2], 1.0f);
          block_c_k[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_k[m][3], 1.0f);
        }
        
        b_ptr_k += hidden_dim;
        current_z_ptr += (groupsize / 8);
        a_ptr += 8;
      }
      
      k += 32;
    }
    
    // 输出K结果
    if (n < hidden_dim) {
      for (int m = 0; m < m_count; m++) {
        int seq_idx = offset_m + m;
        if (seq_idx < batch_size * seq_len) {
          k_output[seq_idx * hidden_dim + n] = __float2half(block_c_k[m][0]);
        }
      }
    }
  }
  
  // V投影（类似Q投影）
  if (n < hidden_dim) {
    float block_c_v[m_count][4] = {0.0f};
    
    int group = 0;
    int nextgroup_k_boundary = groupsize;
    const uint32_t* z_ptr_base_v = qzeros_v;
    float scales_v_val[4];
    
    for (int k = offset_k; k < end_k; k += 32) {
      const half* a_ptr = &block_a[0][k - offset_k];
      const int a_stride = BLOCK_KN_SIZE;
      const uint32_t* b_ptr_v = &qweight_v[k * hidden_dim + n];
      
      if (k >= nextgroup_k_boundary) {
        group++;
        if (group >= hidden_dim / groupsize) break;
        nextgroup_k_boundary = (group + 1) * groupsize;
        z_ptr_base_v = qzeros_v + group * (groupsize / 8);
        for (int i = 0; i < 4; i++) {
          scales_v_val[i] = __half2float(scales_v[group * hidden_dim + n + i]);
        }
      }
      
      const uint32_t* current_z_ptr = z_ptr_base_v + (k % groupsize) / 8;
      
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int4* b_ptr4 = (const int4*)b_ptr_v;
        const int4* z_ptr4 = (const int4*)current_z_ptr;
        int4 load_int4 = *b_ptr4;
        int4 load_zeros = *z_ptr4;
        
        half2 dq[4][4];
        dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_v_val[0]);
        dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_v_val[1]);
        dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_v_val[2]);
        dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_v_val[3]);
        
#pragma unroll
        for (int m = 0; m < m_count; m++) {
          block_c_v[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_v[m][0], 1.0f);
          block_c_v[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_v[m][1], 1.0f);
          block_c_v[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_v[m][2], 1.0f);
          block_c_v[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_v[m][3], 1.0f);
        }
        
        b_ptr_v += hidden_dim;
        current_z_ptr += (groupsize / 8);
        a_ptr += 8;
      }
      
      k += 32;
    }
    
    // 输出V结果
    if (n < hidden_dim) {
      for (int m = 0; m < m_count; m++) {
        int seq_idx = offset_m + m;
        if (seq_idx < batch_size * seq_len) {
          v_output[seq_idx * hidden_dim + n] = __float2half(block_c_v[m][0]);
        }
      }
    }
  }
}

// PyTorch包装函数
torch::Tensor fused_ln_qkv_gptq_cuda(
    torch::Tensor input,
    torch::Tensor qweight_q, torch::Tensor qweight_k, torch::Tensor qweight_v,
    torch::Tensor qzeros_q, torch::Tensor qzeros_k, torch::Tensor qzeros_v,
    torch::Tensor scales_q, torch::Tensor scales_k, torch::Tensor scales_v,
    torch::Tensor rms_weight,
    int batch_size, int seq_len, int hidden_dim, int groupsize, float eps) {
    
    // 输入验证
    TORCH_CHECK(input.dim() == 3, "输入必须是3D张量 [batch_size, seq_len, hidden_dim]");
    TORCH_CHECK(input.size(0) == batch_size, "批次大小不匹配");
    TORCH_CHECK(input.size(1) == seq_len, "序列长度不匹配");
    TORCH_CHECK(input.size(2) == hidden_dim, "隐藏维度不匹配");
    
    // 确保输入是连续的
    input = input.contiguous();
    
    // 创建输出张量
    auto q_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    auto k_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    auto v_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    
    // 获取数据指针
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    half* q_output_ptr = reinterpret_cast<half*>(q_output.data_ptr<at::Half>());
    half* k_output_ptr = reinterpret_cast<half*>(k_output.data_ptr<at::Half>());
    half* v_output_ptr = reinterpret_cast<half*>(v_output.data_ptr<at::Half>());
    
    // 设置CUDA流
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // 启动内核
    dim3 blockDim(BLOCK_KN_SIZE, 1, 1);
    dim3 gridDim(DIVIDE(hidden_dim, BLOCK_KN_SIZE * 4), 
                 DIVIDE(batch_size * seq_len, BLOCK_M_SIZE_MAX), 
                 DIVIDE(hidden_dim, BLOCK_KN_SIZE));
    
    fused_rmsnorm_qkv_gptq_kernel<BLOCK_M_SIZE_MAX><<<gridDim, blockDim, 0, stream>>>(
        input_ptr, 
        qweight_q.data_ptr<uint32_t>(),
        qweight_k.data_ptr<uint32_t>(),
        qweight_v.data_ptr<uint32_t>(),
        qzeros_q.data_ptr<uint32_t>(),
        qzeros_k.data_ptr<uint32_t>(),
        qzeros_v.data_ptr<uint32_t>(),
        reinterpret_cast<const half*>(scales_q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(rms_weight.data_ptr<at::Half>()),
        q_output_ptr, k_output_ptr, v_output_ptr,
        batch_size, seq_len, hidden_dim, groupsize, eps
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA内核执行失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 返回QKV元组
    return torch::stack({q_output, k_output, v_output});
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_qkv_gptq_cuda", &fused_ln_qkv_gptq_cuda, "Fused LayerNorm + GPTQ QKV CUDA");
}
