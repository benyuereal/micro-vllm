#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor forward(
    torch::Tensor hidden,           
    torch::Tensor attn_out,         
    torch::Tensor attn_proj_weight, 
    torch::Tensor norm_weight,      
    torch::Tensor gate_up_weight,   
    torch::Tensor down_weight,      
    float eps
) {
    hidden = hidden.contiguous();
    attn_out = attn_out.contiguous();
    
    int batch_size = hidden.size(0);
    int hidden_size = hidden.size(2);
    
    // 1. Attention Projection + Residual
    auto attn_out_flat = attn_out.view({batch_size, -1});
    auto attn_proj = torch::matmul(attn_out_flat, attn_proj_weight).unsqueeze(1);
    auto attn_residual = hidden + attn_proj;
    
    // 2. RMSNorm
    auto normalized = torch::rms_norm(attn_residual, {hidden_size}, norm_weight, eps);
    
    // 3. MLP Core
    auto x = normalized.view({batch_size, hidden_size});
    auto gate_up = torch::matmul(x, gate_up_weight);
    auto chunks = gate_up.chunk(2, -1);
    
    // 🔥 关键修复：与 Python 保持一致 (up=前半, gate=后半)
    auto up = chunks[0];
    auto gate = chunks[1];
    auto silu_gate = torch::silu(gate) * up;
    auto output = torch::matmul(silu_gate, down_weight);
    
    // 4. Residual
    return attn_residual + output.view({batch_size, 1, hidden_size});
}

PYBIND11_MODULE(cpp_mlp, m) {
    m.def("forward", &forward, "Fused MLP forward (C++)");
}